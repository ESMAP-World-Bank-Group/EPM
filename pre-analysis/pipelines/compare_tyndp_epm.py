# -*- coding: utf-8 -*-
"""Compare the EPM reference run with TYNDP 2026, TYNDP 2024 and the national plans.

Reads the merged EPM run, the EPM inputs (read only), the tidy and control tables
of pre-analysis/output_tyndp/, the 2026 KPI dashboard, the 2024 market workbooks
and tools/results_report/reference/national_plans.json. Writes to the tidy folder:

    compare_epm_tyndp_plans.csv     one row per domain, key, year, source and item
    compare_manifest.csv            rows and sha256 of the table and of each source

Method
------
* Domains: capacity (GW), demand and net_imports (TWh), border (MW, directed).
* EPM has no 2035 on the five year grid: capacity, net imports and limits use 2036
  and model_year says so. Demand comes from the forecast input, which holds 2035.
* TYNDP capacity is the published market run. Rows absent from it but present in
  PEMMDB are kept with nature input_only: GE and AZ, TR offshore wind, MD solar.
* TYNDP demand keeps its components, EPM has one figure. Pumping and battery load
  are left out, so the TYNDP total is smaller than generation plus net imports.
* A border is summed over the nodes of a country pair, GR03 aside. The 2024 grid is
  a 2030 grid: its 2040 row is the maximum flow of climate year 2009, not a limit.
* EPM limits take the highest quarter. Moldova and Ukraine are annual volumes over
  8,760 hours, not transfer capacities.
* The sign of net_imports follows the KPI dashboard, a net exporter is negative.
* NT 2024 and NT+ 2026 stay separate sources. Gaps are EPM minus the source.
* Nothing here writes to epm/input.

Usage (from EPM/):  python pre-analysis/pipelines/compare_tyndp_epm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_tyndp_tidy import CONFIG, OUT_DIR, REPO, Run, workbook, write
from check_tyndp_controls import (CAPACITY_ITEMS, KPI_COUNTRIES, MARKET_2024, kpi_long, read_market_2024,
                                  sha256_file, tidy_capacity)

EPM_RUN = REPO / "epm" / "output_view" / "robg_s5_v3_20261024"
EPM_INPUT = REPO / "epm" / "input" / "data_blacksea"
PLANS = REPO / "tools" / "results_report" / "reference" / "national_plans.json"
SCENARIO = "S5_Base_v4"
DEMAND_FILE = EPM_INPUT / "load" / "pDemandForecast_v2.csv"
ZONE_FILE = EPM_INPUT / "zcmap_robg.csv"
EXT_LIMITS = EPM_INPUT / "trade" / "pExtTransferLimit_robg.csv"
INT_LIMITS = EPM_INPUT / "trade" / "pTransferLimit_robg_hvdc.csv"

EPM, NT_PLUS, NT, PLAN = "EPM S5_Base_v4", "TYNDP 2026 NT+", "TYNDP 2024 NT", "National plan"
YEARS = [2030, 2035, 2040]
MODEL_YEAR = {2030: 2030, 2035: 2036, 2040: 2040}
COUNTRIES = {"Romania": "RO", "Bulgaria": "BG", "Turkiye": "TR", "Georgia": "GE", "Armenia": "AM",
             "Azerbaijan": "AZ", "Moldova": "MD", "Ukraine": "UA"}
TRADE_COUNTRIES = ("RO", "BG", "TR", "MD", "UA")

GROUPS = ["Solar", "Wind", "Hydro and pumped", "Nuclear", "Gas", "Coal", "Battery", "Other"]
EPM_GROUP = {"PV": "Solar", "PV+Storage": "Solar", "Solar": "Solar", "CSP": "Solar", "Solar Thermal": "Solar",
             "Onshore Wind": "Wind", "Offshore Wind": "Wind", "ROR": "Hydro and pumped",
             "Reservoir": "Hydro and pumped", "PSH": "Hydro and pumped", "Nuclear": "Nuclear", "Gas": "Gas",
             "Coal": "Coal", "Battery": "Battery"}
TYNDP_GROUP = {"Solar": "Solar", "Onshore wind": "Wind", "Offshore wind": "Wind", "Run of river": "Hydro and pumped",
               "Pondage": "Hydro and pumped", "Reservoir": "Hydro and pumped",
               "Pumped storage turbine": "Hydro and pumped", "Nuclear": "Nuclear", "Natural gas": "Gas",
               "Coal": "Coal", "Battery": "Battery", "Other RES": "Other", "Oil": "Other", "Other non-RES": "Other",
               "Hydrogen GT": "Other", "Adequacy units": "Other"}
PLAN_GROUP = {"Solar": "Solar", "Wind": "Wind", "Hydro": "Hydro and pumped", "Nuclear": "Nuclear", "Gas": "Gas",
              "Coal": "Coal", "Battery": "Battery", "Biomass": "Other", "Geothermal": "Other"}
DEMAND_PARTS = [("Native plus fixed", "Electricity Demand (Native + Fixed)"), ("EV", "Net EV Load"),
                ("Heat", "Heat Electrical Usage"), ("Electrolyser", "Electrolyzer Load")]

ZONE_CODE = {"Romania": "RO", "Bulgaria": "BG", "Trakia": "TR", "Hungary": "HU", "Serbia": "RS", "Moldova": "MD",
             "Ukraine": "UA", "NorthMacedonia": "MK", "Greece": "GR"}
BORDER_PAIRS = [("RO", "HU"), ("RO", "RS"), ("RO", "UA"), ("RO", "MD"), ("RO", "BG"), ("BG", "RS"), ("BG", "MK"),
                ("BG", "GR"), ("BG", "TR"), ("TR", "GR")]
BORDER_YEARS = {"Transelectrica plan 2024-2033 Tab 5.4.2": {  # export NTC, tables as images, read by hand
    ("RO", "HU"): {2030: (2030, 1700.0), 2035: (2033, 2300.0)},
    ("RO", "BG"): {2030: (2030, 2550.0), 2035: (2033, 2550.0)},
    ("RO", "RS"): {2030: (2030, 1300.0), 2035: (2033, 2000.0)},
    ("RO", "UA"): {2030: (2030, 500.0), 2035: (2033, 500.0)}}}
PLAN_DEMAND = {("RO", 2030): (79.0, "NECP 2024 WAM")}
PLAN_TRADE = {("BG", 2030): (4.5, "ESO plan")}


def row(domain, key, year, model_year, source, nature, item, value, unit, note=""):
    return dict(domain=domain, key=key, year=year, model_year=model_year, source=source, nature=nature, item=item,
                value=None if value is None else float(value), unit=unit, note=note)


def epm_capacity(summary: pd.DataFrame) -> list[dict]:
    rows = []
    capacity = summary[summary.attribute == "Capacity: MW"]
    for country, code in COUNTRIES.items():
        for year in YEARS:
            part = capacity[(capacity.country == country) & (capacity.year == MODEL_YEAR[year])]
            groups = part.resolution.map(EPM_GROUP).fillna("Other")
            for group, value in part[SCENARIO].groupby(groups).sum().items():
                rows.append(row("capacity", code, year, MODEL_YEAR[year], EPM, "model_result", group, value / 1e3, "GW"))
    return [r for r in rows if r["key"] not in ("MD", "UA")]


def tyndp_capacity(cap: pd.DataFrame, controls: pd.DataFrame) -> list[dict]:
    rows = []
    for edition, source, scenario_years in (("2026", NT_PLUS, YEARS), ("2024", NT, [2030, 2040])):
        for code in COUNTRIES.values():
            for year in scenario_years:
                frame = controls[(controls.edition == edition) & (controls.country == code) & (controls.year == year)]
                for item, group in TYNDP_GROUP.items():
                    published = frame.loc[frame.item == item, "published"].sum()
                    pick = next(p[-1] for p in CAPACITY_ITEMS if p[0] == item)
                    tidy = 0.0 if pick is None else np.nan_to_num(tidy_capacity(cap, edition, code, year, "capacity", "GW", pick))
                    if published > 0:
                        rows.append(row("capacity", code, year, None, source, "market_run", group, published, "GW"))
                    elif tidy > 0:
                        rows.append(row("capacity", code, year, None, source, "input_only", group, tidy, "GW",
                                        f"{item}: PEMMDB input, not in a market run"))
    frame = pd.DataFrame(rows)
    keys = ["domain", "key", "year", "source", "nature", "item", "unit"]
    summed = frame.groupby(keys, as_index=False, dropna=False).agg(
        model_year=("model_year", "first"), value=("value", "sum"), note=("note", lambda n: "; ".join(sorted(set(n) - {""}))))
    return summed.to_dict("records")


def plan_capacity() -> list[dict]:
    plans = json.loads(PLANS.read_text(encoding="utf-8"))
    rows = []
    for country, techs in plans["capacity_gw"].items():
        for tech, values in techs.items():
            for year, value in zip(plans["years"], values):
                if year in YEARS and value is not None:
                    rows.append(row("capacity", COUNTRIES[country], year, None, PLAN, "plan", PLAN_GROUP[tech], value, "GW",
                                    plans["source"][country]))
    frame = pd.DataFrame(rows)
    keys = ["domain", "key", "year", "source", "nature", "item", "unit", "note"]
    return frame.groupby(keys, as_index=False, dropna=False).agg(model_year=("model_year", "first"), value=("value", "sum")).to_dict("records")


def epm_demand() -> list[dict]:
    zones = pd.read_csv(ZONE_FILE, encoding="utf-8-sig")
    forecast = pd.read_csv(DEMAND_FILE, encoding="utf-8-sig")
    forecast = forecast[forecast["type"] == "Energy"].merge(zones, on="z")
    rows = []
    for country, code in COUNTRIES.items():
        part = forecast[forecast.c == country]
        if not part.empty:
            for year in YEARS:
                rows.append(row("demand", code, year, year, EPM, "model_input", "Total", part[str(year)].sum() / 1e3, "TWh"))
    return rows


def tyndp_demand(kpi: pd.DataFrame, dem: pd.DataFrame) -> list[dict]:
    rows = []
    balance = kpi[(kpi.section == "Energy Balance [TWh]") & (kpi.weather == "Weighted WS")]
    for code in TRADE_COUNTRIES:
        for year in YEARS:
            part = balance[(balance.country == code) & (balance.year == year)].set_index("label").value
            parts = {item: part[label] for item, label in DEMAND_PARTS if label in part.index and pd.notna(part[label])}
            rows += [row("demand", code, year, None, NT_PLUS, "market_run", item, value, "TWh") for item, value in parts.items()]
            rows.append(row("demand", code, year, None, NT_PLUS, "market_run", "Total", sum(parts.values()), "TWh",
                            "Sum of the components, pumping and battery load excluded"))
    totals = dem[(dem.edition == "2024") & (dem.metric == "annual_energy") & (dem.weather == "2009")]
    for code in ("RO", "BG", "TR", "MD", "UA", "GE"):
        part = totals[totals.node.str[:2] == code].groupby("year").value.sum()
        for year in (2030, 2040):
            if year in part.index:
                note = "Equals the 2030 series in the source" if code in ("MD", "UA") and year == 2040 else ""
                rows.append(row("demand", code, year, None, NT, "input_only", "Total", part[year], "TWh", note))
    return rows


def net_imports(balance: pd.DataFrame, kpi: pd.DataFrame, yearly: dict[int, dict]) -> list[dict]:
    rows = []
    net = balance[balance.uni == "Net interchange: GWh"]
    for country, code in COUNTRIES.items():
        part = net[net.c == country]
        if not part.empty:
            for year in YEARS:
                value = part.loc[part.y == MODEL_YEAR[year], "value"].sum()
                rows.append(row("net_imports", code, year, MODEL_YEAR[year], EPM, "model_result", "Net imports", value / 1e3, "TWh"))
    kpis = kpi[(kpi.label == "Net Imports") & (kpi.weather == "Weighted WS")]
    for code in TRADE_COUNTRIES:
        for year in YEARS:
            part = kpis[(kpis.country == code) & (kpis.year == year)]
            if not part.empty:
                rows.append(row("net_imports", code, year, None, NT_PLUS, "market_run", "Net imports", part.value.iloc[0], "TWh"))
        for year, book in yearly.items():
            value = 0.0
            for border, series in book["flows"].items():
                start, end = border.split("-")
                if start[:2] == code and end[:2] != code:
                    value -= series.sum() / 1e6
                elif end[:2] == code and start[:2] != code:
                    value += series.sum() / 1e6
            rows.append(row("net_imports", code, year, None, NT, "market_run", "Net imports", value, "TWh", "Climate year 2009"))
    return rows + [row("net_imports", code, year, None, PLAN, "plan", "Net imports", value, "TWh", source)
                   for (code, year), (value, source) in PLAN_TRADE.items()]


def epm_limits() -> dict[tuple, dict]:
    """Highest quarter per directed country pair and model year."""
    ext = pd.read_csv(EXT_LIMITS, encoding="utf-8-sig")
    ext.columns = ["z", "zext", "q", "direction"] + list(ext.columns[4:])
    internal = pd.read_csv(INT_LIMITS, encoding="utf-8-sig")
    limits: dict[tuple, dict] = {}
    for year in MODEL_YEAR.values():
        for r in ext[ext.z.isin(ZONE_CODE) & ext.zext.isin(ZONE_CODE)].itertuples(index=False):
            a, b = ZONE_CODE[r.z], ZONE_CODE[r.zext]
            pair = (a, b) if r.direction == "Export" else (b, a)
            slot = limits.setdefault(pair, {})
            slot[year] = max(slot.get(year, 0.0), float(getattr(r, f"_{list(ext.columns).index(str(year))}")))
        part = internal[internal.z.isin(ZONE_CODE) & internal.z2.isin(ZONE_CODE)]
        for (a, b), value in part.groupby(["z", "z2"])[str(year)].max().items():
            limits.setdefault((ZONE_CODE[a], ZONE_CODE[b]), {})[year] = float(value)
    return limits


def borders(ntc: pd.DataFrame, flows: pd.DataFrame) -> list[dict]:
    rows, limits = [], epm_limits()
    directed = [p for pair in BORDER_PAIRS for p in (pair, pair[::-1])]
    for a, b in directed:
        key = f"{a}>{b}"
        for year in YEARS:
            if (a, b) in limits:
                note = "Annual volume over 8,760 h" if "MD" in (a, b) or "UA" in (a, b) else ""
                rows.append(row("border", key, year, MODEL_YEAR[year], EPM, "model_limit", "Limit",
                                limits[(a, b)][MODEL_YEAR[year]], "MW", note))
        part = ntc[(ntc.node_from.str[:2] == a) & (ntc.node_to.str[:2] == b) & (ntc.node_from != "GR03") & (ntc.node_to != "GR03")]
        for edition, source, metric in (("2026", NT_PLUS, "ntc_reference"), ("2024", NT, "ntc_reference")):
            sums = part[(part.edition == edition) & (part.metric == metric)].groupby("year").value.sum()
            for year in YEARS:
                if year in sums.index:
                    rows.append(row("border", key, year, None, source, "planned_ntc", "Limit", sums[year], "MW"))
        now = part[(part.edition == "2026") & (part.metric == "ntc_2025")].value.sum()
        if now > 0:
            rows.append(row("border", key, 2025, None, NT_PLUS, "ntc_2025", "Limit", now, "MW"))
        seen = flows[(flows.edition == 2024) & (flows.year == 2040) & (flows.node_from.str[:2] == a) & (flows.node_to.str[:2] == b)]
        seen = seen[(seen.node_from != "GR03") & (seen.node_to != "GR03")]
        if not seen.empty:
            rows.append(row("border", key, 2040, None, NT, "max_flow", "Limit", seen.max_flow.sum(), "MW", "Maximum flow, climate year 2009"))
        for source, table in BORDER_YEARS.items():
            for year, (document_year, value) in table.get((a, b), {}).items():
                rows.append(row("border", key, year, document_year, PLAN, "planned_ntc", "Limit", value, "MW",
                                f"{source}, value for {document_year}"))
    return rows


def add_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    totals = frame.groupby(["domain", "key", "year", "source", "item"], as_index=False).value.sum()
    base = totals[totals.source == EPM].rename(columns={"value": "epm"})[["domain", "key", "year", "item", "epm"]]
    frame = frame.merge(base, on=["domain", "key", "year", "item"], how="left")
    missing = frame.domain.eq("capacity") & frame.epm.isna() & frame.key.isin(frame.loc[frame.source == EPM, "key"])
    frame["epm_minus_source"] = frame.epm.where(~missing, 0.0) - frame.value
    return frame.drop(columns="epm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="Folder with manifest.csv, default from the config")
    parser.add_argument("--run", help="Merged run folder under epm/output_view, default robg_s5_v3_20261024")
    parser.add_argument("--scenario", help="Scenario column of the merged run, default S5_Base_v4")
    parser.add_argument("--name", default="compare_epm_tyndp_plans", help="Output table name")
    args = parser.parse_args()

    global EPM_RUN, SCENARIO, EPM, MODEL_YEAR
    if args.run:
        EPM_RUN = REPO / "epm" / "output_view" / args.run
    if args.scenario:
        SCENARIO = args.scenario
        EPM = f"EPM {SCENARIO}"
    years = set(pd.read_csv(EPM_RUN / "summary.csv", usecols=["year"]).year.dropna().astype(int))
    MODEL_YEAR = {y: (y if y in years else y + 1) for y in YEARS}

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = args.root or (REPO / cfg["tyndp_root"]).resolve()
    run = Run(root, cfg)

    read = lambda name, **kw: pd.read_csv(OUT_DIR / f"{name}.csv", low_memory=False, **kw)
    cap = read("capacity", dtype={"edition": str})
    dem = read("demand_annual", dtype={"edition": str}, keep_default_na=False, na_values=[""])
    dem["weather"] = dem["weather"].astype(str)
    controls = read("controls_capacity", dtype={"edition": str})
    ntc, flows = read("ntc", dtype={"edition": str}), read("controls_ntc")

    summary = pd.read_csv(EPM_RUN / "summary.csv", usecols=["country", "zone", "attribute", "resolution", "year", SCENARIO])
    balance = pd.read_csv(EPM_RUN / SCENARIO / "output_csv" / "pEnergyBalance.csv")

    print("KPI dashboard", flush=True)
    kpi_pack = run.package("2026/outputs", "NT+_KPI_Dashboard.xlsx")
    book = workbook(kpi_pack["path"].read_bytes())
    kpi = pd.concat([kpi_long(book[c], c) for c in KPI_COUNTRIES], ignore_index=True)
    book.close()

    print("2024 market workbooks", flush=True)
    sources = [dict(table=f"source {kpi_pack['source_package']}", rows=0, sha256=kpi_pack["source_sha256"])]
    yearly = {}
    for year, name in MARKET_2024.items():
        path = root / "2024" / name
        sources.append(dict(table=f"source 2024/{name}", rows=0, sha256=sha256_file(path)))
        yearly[year] = read_market_2024(path, year, run.core)
    for path in (EPM_RUN / "summary.csv", EPM_RUN / SCENARIO / "output_csv" / "pEnergyBalance.csv", DEMAND_FILE,
                 EXT_LIMITS, INT_LIMITS, PLANS):
        sources.append(dict(table=f"source {path.relative_to(REPO).as_posix()}", rows=0, sha256=sha256_file(path)))

    rows = (epm_capacity(summary) + tyndp_capacity(cap, controls) + plan_capacity() + epm_demand()
            + tyndp_demand(kpi, dem) + net_imports(balance, kpi, yearly) + borders(ntc, flows))
    rows += [row("demand", code, year, None, PLAN, "plan", "Total", value, "TWh", source)
             for (code, year), (value, source) in PLAN_DEMAND.items()]
    frame = add_gaps(pd.DataFrame(rows))
    columns = ["domain", "key", "year", "model_year", "source", "nature", "item", "value", "unit", "epm_minus_source", "note"]
    frame = frame[columns].sort_values(columns[:7], key=lambda s: s.astype(str)).reset_index(drop=True)

    manifest: list[dict] = []
    write(frame.round(4), args.name, manifest)
    manifest_name = "compare_manifest" if args.name == "compare_epm_tyndp_plans" else f"{args.name}_manifest"
    pd.DataFrame(manifest + sources).to_csv(OUT_DIR / f"{manifest_name}.csv", index=False, lineterminator="\n")
    print(frame.groupby(["domain", "source", "nature"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
