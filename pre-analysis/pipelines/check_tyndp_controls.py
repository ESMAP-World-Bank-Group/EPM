# -*- coding: utf-8 -*-
"""Control the tidy TYNDP tables against the published market results.

Reads the tidy tables of pre-analysis/output_tyndp/, the 2026 KPI dashboard, the
2026 hourly dashboards and the 2024 market output workbooks, and writes to the
same folder:

    controls_capacity.csv    tidy capacity against the published capacity
    controls_demand.csv      tidy demand against the published demand and peaks
    controls_ntc.csv         tidy NTC against the hourly flows, per directed border
    controls_closure.csv     hourly imports and exports against the KPI dashboard
    controls_anomalies.csv   measured evidence for the anomalies listed in A2
    controls_summary.csv     counts per control and status
    controls_manifest.csv    rows and sha256 of each table and of each source

Method
------
* The input side comes from tidy tables only, the published side from the results
  of the market runs. A row that differs is reported, nothing is corrected.
* 2026 uses the weighted KPI column for capacity and the three weather scenarios
  of each year for demand and flows. 2024 uses the climate year 2009 workbooks,
  available for 2030 and 2040.
* A capacity or demand row is ok when the gap is below 0.5 percent or below
  ABS_TOL. Flows are in MW, positive from the first to the second node of the
  column name. An hour is at the limit when it is within 1 MW of the NTC.
* KPI country sheets add up the nodes of a country, so GR is GR00 + GR03.
* KPI wind and solar leave out the capacity dedicated to electrolysers: those
  rows compare the tidy total minus the dedicated split (ok_excl_dedicated).
* KPI Biofuel moves thermal units with biofuel out of their fuel line: a gap that
  closes over Other RES, gas, coal and oil is ok_biofuel_reclass.
* KPI Fixed Load holds the behind the meter stream and the fixed imports, not the
  EV input files. Those files are compared with the charging station load
  (ev_not_same_stream when they differ).
* The 2024 Line-data package holds the 2030 reference grid only, so 2040 borders
  are no_limit_in_tidy and max_flow is the only published figure.
* The 2024 workbooks are not in manifest.csv: their sha256 goes to the manifest of
  the controls.
* Nothing here writes to epm/input.

The output is deterministic: a rerun on the same packages gives the same hashes.

Usage (from EPM/):  python pre-analysis/pipelines/check_tyndp_controls.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_tyndp_tidy import CONFIG, OUT_DIR, REPO, Run, clean, num, workbook, write

KPI_COUNTRIES = ["RO", "BG", "TR", "MD", "UA", "HU", "RS", "MK", "GR", "SK", "PL"]
FLOW_COUNTRIES = ["RO", "BG", "TR", "MD", "UA"]  # every border of a core node appears in one of these files
DEMAND_ONLY_COUNTRIES = ["GR"]
HOURLY = "NT+_TimeSeriesDashboard/{y}/Weather scenario {w}/{c}_TimeSeriesDashboard_TY{y}_WS{w}_SoS.xlsx"
WEATHER = {2030: ["003", "021", "029"], 2035: ["032", "037", "059"], 2040: ["065", "071", "077"],
           2050: ["091", "092", "106"]}
MARKET_2024 = {2030: "MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx",
               2040: "MMStandardOutputFile_NT2040_Plexos_CY2009_2.5_v40.xlsx"}
REL_TOL = 0.005
ABS_TOL = {"GW": 0.005, "GWh": 0.005, "TWh": 0.01}
LIMIT_TOL = 1.0  # MW
FOSSIL_GAS_FACTOR = 57.0  # kg/net GJ, common data
PS_TECHS = ("hydro pump storage open loop", "hydro pure pump storage closed loop")
DEDICATED_ITEMS = ("Onshore wind", "Offshore wind", "Solar")
BIOFUEL_ITEMS = ("Other RES", "Natural gas", "Coal", "Oil")
STEP_SHARE = 0.2  # a step between two hours above this share of the peak is not a load shape

# item, KPI labels, labels of the 2024 yearly outputs, tidy metric, unit, tidy selection (family or name prefixes)
CAPACITY_ITEMS = [
    ("Onshore wind", ["Wind Onshore"], ["Wind Onshore"], "capacity", "GW", ("onshore wind",)),
    ("Offshore wind", ["Wind Offshore Radial", "Wind Offshore Hub"], ["Wind Offshore"], "capacity", "GW",
     ("offshore wind",)),
    ("Solar", ["Solar PV", "Solar CSP"], ["Solar (Photovoltaic)", "Solar (Thermal)"], "capacity", "GW",
     ("photovoltaic", "rooftop", "solar thermal")),
    ("Run of river", ["Run of River"], ["Run-of-River"], "capacity", "GW", ("hydro run of river",)),
    ("Pondage", ["Pondage"], ["Pondage"], "capacity", "GW", ("hydro pondage",)),
    ("Reservoir", ["Reservoir"], ["Reservoir"], "capacity", "GW", ("hydro reservoir",)),
    ("Pumped storage turbine", ["PS Turbine"],
     ["Pump Storage - Open Loop (turbine)", "Pump Storage - Closed Loop (turbine)"], "capacity", "GW", PS_TECHS),
    ("Pumped storage pump", ["PS Pump"],
     ["Pump Storage - Open Loop (pump)", "Pump Storage - Closed Loop (pump)"], "pumping_capacity", "GW", PS_TECHS),
    ("Other RES", ["Biofuel", "Other RES"], ["Others renewable"], "capacity", "GW", "other_res"),
    ("Nuclear", ["Nuclear"], ["Nuclear"], "capacity", "GW", ("nuclear",)),
    ("Natural gas", ["Natural Gas"], ["gas "], "capacity", "GW", ("gas ",)),
    ("Oil", ["Crude Oil"], ["light oil", "heavy oil"], "capacity", "GW", ("heavy oil", "light oil")),
    ("Coal", ["Coal"], ["lignite ", "hard coal "], "capacity", "GW", ("hard coal", "lignite")),
    ("Other non-RES", ["Other Non RES"], ["Others non-renewable"], "capacity", "GW", "other_non_res"),
    ("Hydrogen GT", ["Hydrogen GT"], ["Hydrogen CCGT"], "capacity", "GW", ("hydrogen ccgt",)),
    ("Battery", ["Batteries"], ["Battery Storage discharge (gen.)"], "capacity", "GW", ("battery",)),
    ("Battery energy", ["Batteries [GWh]"], [], "storage_energy", "GWh", ("battery",)),
    ("Electrolyser", ["Electrolyzers E-Market (P2G) [GW Elec]", "Electrolyzers DRES (P2G) [GW Elec]"],
     ["Electrolyser (load)"], "capacity", "GW", ("electrolyser",)),
    ("Adequacy units", ["Adequacy Units"], [], "capacity", "GW", None),
]
DEMAND_ITEMS = [  # item, section of the KPI sheet, KPI label, tidy streams, fixed imports netted
    ("Native market", "Consumption [TWh]", "Native Demand (E-Market)", ["market"], False),
    ("Native prosumer", "Consumption [TWh]", "Native Demand (Prosumer)", ["prosumer"], False),
    ("Fixed market", "Consumption [TWh]", "Fixed Load (Emarket)", [], True),
    ("Fixed prosumer", "Consumption [TWh]", "Fixed Load (Prosumer)", ["prosumer_btm_fixed"], False),
    ("Native plus fixed", "Energy Balance [TWh]", "Electricity Demand (Native + Fixed)",
     ["market", "prosumer", "prosumer_btm_fixed"], True),
]
EV_LABELS = ["Charging Station Load (E-Market)", "Charging Station Load (Prosumer)"]


def status(control, tidy, unit: str) -> str:
    published, value = (0.0 if pd.isna(v) else float(v) for v in (control, tidy))
    tol = ABS_TOL[unit]
    if abs(published) <= tol and abs(value) <= tol:
        return "absent"
    if abs(published) <= tol:
        return "only_tidy"
    if abs(value) <= tol:
        return "only_published"
    gap = abs(published - value)
    return "ok" if gap <= tol or gap <= REL_TOL * abs(published) else "differs"


def kpi_long(ws, country: str) -> pd.DataFrame:
    """One row per section, label, year and weather column of a KPI country sheet."""
    rows = [tuple(r) + (None,) * 30 for r in ws.iter_rows(values_only=True)]
    cols, year = {}, None
    for j in range(3, 30):
        found = re.search(r"20\d\d", clean(rows[2][j]))
        year = int(found.group()) if found else year
        if clean(rows[3][j]):
            cols[j] = (year, clean(rows[3][j]))
    out, section = [], "Installed Capacity [GW]"
    for r in rows[4:]:
        a, b = clean(r[0]), clean(r[1])
        if a and not b:
            section = a
        elif b:
            for j, (year, weather) in cols.items():
                value = num(r[j])
                if value is not None:
                    out.append((country, section, b, year, weather, value))
    return pd.DataFrame(out, columns=["country", "section", "label", "year", "weather", "value"])


def tidy_capacity(cap: pd.DataFrame, edition: str, country: str, year: int, metric: str, unit: str, pick) -> float:
    rows = cap[(cap.edition == edition) & (cap.year == year) & (cap.node.str[:2] == country)
               & (cap.metric == metric) & (cap.level == "total")]
    if isinstance(pick, str):
        rows = rows[rows.family == pick]
    else:
        rows = rows[rows.technology.str.lower().str.startswith(pick)]
    if rows.empty:
        return np.nan
    factor = {"MW": 1e-3, "MWh": 1e-3, "GWh": 1.0, "GW": 1.0}
    return float((rows.value * rows.unit.map(factor)).sum())


def picked(technology: str, family: str) -> bool:
    for *_, pick in CAPACITY_ITEMS:
        if pick == family or (isinstance(pick, tuple) and technology.lower().startswith(pick)):
            return True
    return False


def control_capacity(cap: pd.DataFrame, kpi: pd.DataFrame, yearly: dict[int, dict]) -> pd.DataFrame:
    rows = []
    dedicated = cap[(cap.level == "split") & (cap.metric == "capacity") & cap.technology.str.contains("dedicated")]
    dedicated = dedicated.assign(technology=dedicated.technology.str.lower())
    cap = cap[cap.level == "total"]
    for edition, years in (("2026", WEATHER), ("2024", yearly)):
        for year in years:
            for country in KPI_COUNTRIES:
                for item, labels, labels_2024, metric, unit, pick in CAPACITY_ITEMS:
                    low = high = np.nan
                    if edition == "2026":
                        sel = kpi[(kpi.country == country) & (kpi.year == year) & (kpi.label.isin(labels))
                                  & (kpi.section == "Installed Capacity [GW]")]
                        by_weather = sel.groupby("weather").value.sum()
                        control = by_weather.get("Weighted WS", np.nan)
                        low, high = by_weather.min() if len(by_weather) else np.nan, by_weather.max() if len(by_weather) else np.nan
                    else:
                        table = yearly[year]["capacity"].get(country)
                        if table is None or not labels_2024:
                            continue
                        control = sum(v for k, v in table.items() if label_match(k, labels_2024)) / 1e3
                    tidy = np.nan if pick is None else tidy_capacity(cap, edition, country, year, metric, unit, pick)
                    apart = 0.0
                    if edition == "2026" and item in DEDICATED_ITEMS:
                        sel = dedicated[(dedicated.year == year) & (dedicated.node.str[:2] == country)
                                        & dedicated.technology.str.startswith(pick)]
                        apart = float(sel.value.sum()) / 1e3
                    net = tidy - apart
                    state = status(control, net, unit)
                    if state == "ok" and apart > ABS_TOL[unit]:
                        state = "ok_excl_dedicated"
                    rows.append(dict(edition=edition, year=year, country=country, item=item, unit=unit,
                                     published=control, tidy=tidy, tidy_dedicated=apart,
                                     gap=net - control if pick else np.nan, weather_low=low, weather_high=high,
                                     status=state))
    left = cap[cap.metric == "capacity"]
    left = left[left.node.str[:2].isin(KPI_COUNTRIES)]
    left = left[~left.family.isin(("reserves", "dsr"))]
    for tech, family in sorted({(t, f) for t, f in zip(left.technology, left.family)}):
        if not picked(tech, family):
            rows.append(dict(edition="all", year=0, country="all", item=f"unmapped {tech}", unit="GW", status="unmapped"))
    frame = pd.DataFrame(rows)
    group = frame[(frame.edition == "2026") & frame.item.isin(BIOFUEL_ITEMS)]
    for (year, country), sel in group.groupby(["year", "country"]):
        wrong = sel[~sel.status.isin(("ok", "absent"))]
        if len(wrong) and status(sel.published.sum(), sel.tidy.sum(), "GW") == "ok":
            frame.loc[wrong.index, "status"] = "ok_biofuel_reclass"
    return frame[frame.status != "absent"].reset_index(drop=True)


def label_match(label: str, wanted: list[str]) -> bool:
    """Exact label, or prefix when the wanted entry is in lower case."""
    return any(label == w or (w.islower() and label.lower().startswith(w)) for w in wanted)


def control_demand(dem: pd.DataFrame, kpi: pd.DataFrame, yearly: dict[int, dict], hourly: pd.DataFrame,
                   exchanges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    energy = dem[dem.metric == "annual_energy"]
    fixed = exchanges[(exchanges.edition == "2026") & (exchanges.technology == "Total hourly series")
                      & (exchanges.metric == "positive_sum")]
    for year, cases in WEATHER.items():
        for country in KPI_COUNTRIES:
            imported = fixed[(fixed.year == year) & (fixed.node.str[:2] == country)].value.sum()
            for case in cases:
                weather = f"WS{case}"
                sel = energy[(energy.edition == "2026") & (energy.year == year) & (energy.weather == weather)
                             & (energy.node.str[:2] == country)]
                base = dict(edition="2026", year=year, zone=country, weather=weather, unit="TWh")
                for item, section, label, streams, netted in DEMAND_ITEMS:
                    found = kpi[(kpi.country == country) & (kpi.year == year) & (kpi.weather == weather)
                                & (kpi.section == section) & (kpi.label == label)]
                    control = found.value.sum() if len(found) else np.nan
                    tidy = sel[sel.technology.isin(streams)].value.sum() - (imported if netted else 0.0)
                    rows.append(dict(**base, item=item, published=control, tidy=tidy, gap=tidy - control,
                                     status=status(control, tidy, "TWh")))
                found = kpi[(kpi.country == country) & (kpi.year == year) & (kpi.weather == weather)
                            & (kpi.section == "Consumption [TWh]") & kpi.label.isin(EV_LABELS)]
                control = found.value.sum()
                tidy = sel[sel.technology.isin(("ev_market", "ev_prosumer"))].value.sum()
                state = status(control, tidy, "TWh")
                rows.append(dict(**base, item="EV charging", published=control, tidy=tidy, gap=tidy - control,
                                 status="ev_not_same_stream" if state not in ("ok", "absent") else state))
    for year, book in yearly.items():
        for country, value in book["demand"].items():
            sel = energy[(energy.edition == "2024") & (energy.year == year) & (energy.weather == "2009")
                         & (energy.node.str[:2] == country)]
            tidy = sel.value.sum() if len(sel) else np.nan
            control = value / 1e3
            rows.append(dict(edition="2024", year=year, zone=country, item="Native total", weather="2009", unit="TWh",
                             published=control, tidy=tidy, gap=tidy - control, status=status(control, tidy, "TWh")))
    peaks = dem[(dem.metric == "peak") & (dem.edition == "2026") & (dem.technology == "market")]
    for r in hourly.itertuples():
        base = dict(edition="2026", year=r.year, zone=r.zone, weather=r.weather)
        sel = energy[(energy.edition == "2026") & (energy.year == r.year) & (energy.weather == r.weather)
                     & (energy.node == r.zone) & (energy.technology == "market")]
        tidy = sel.value.sum() if len(sel) else np.nan
        rows.append(dict(**base, item="Hourly native market", unit="TWh", published=r.twh, tidy=tidy,
                         gap=tidy - r.twh, status=status(r.twh, tidy, "TWh")))
        sel = peaks[(peaks.year == r.year) & (peaks.weather == r.weather) & (peaks.node == r.zone)]
        tidy = sel.value.sum() if len(sel) else np.nan
        rows.append(dict(**base, item="Hourly native market peak", unit="GW", published=r.gw, tidy=tidy,
                         gap=tidy - r.gw, status=status(r.gw, tidy, "GW")))
    frame = pd.DataFrame(rows)
    return frame[frame.status != "absent"].reset_index(drop=True)


def read_hourly(z: zipfile.ZipFile, country: str, year: int, case: str, with_flows: bool) -> tuple[list, dict]:
    """Native demand of each zone and the hourly E-Market flows of one country file."""
    book = workbook(z.read(HOURLY.format(c=country, y=year, w=case)))
    head_rows = list(book["E-Market"].iter_rows(max_row=8, values_only=True))
    zones = []
    for j, category in enumerate(head_rows[6]):
        if clean(category).startswith("Native Demand"):
            zones.append(dict(year=year, weather=f"WS{case}", zone=clean(head_rows[5][j]),
                              twh=float(head_rows[3][j]) / 1e3, gw=float(head_rows[1][j]) / 1e3))
    flows = {}
    if with_flows:
        rows = book["Exchanges"].iter_rows(values_only=True)
        top = [next(rows) for _ in range(10)]
        cols = {}
        for j, category in enumerate(top[6]):
            if clean(category).startswith("E-Market Exchanges"):
                cols.setdefault(clean(top[7][j]), j)
        data = np.array([[num(r[j]) or 0.0 for j in cols.values()] for r in rows], dtype=float)
        flows = {name: data[:, i] for i, name in enumerate(cols)}
    book.close()
    return zones, flows


def read_market_2024(path: Path, year: int, nodes: list[str]) -> dict:
    """Capacity, demand by country and core border flows of one 2024 market workbook."""
    book = workbook(path.read_bytes())
    rows = list(book["Yearly Outputs"].iter_rows(values_only=True))
    header = rows[5]
    columns = {j: clean(h) for j, h in enumerate(header) if clean(h)[:2] in KPI_COUNTRIES and j >= 2}
    capacity, demand, section = {}, {}, ""
    for r in rows[6:]:
        section = clean(r[0]) or section
        for j, node in columns.items():
            value = num(r[j])
            if value is None:
                continue
            country = node[:2]
            if section == "Installed Capacities [MW]" and clean(r[1]):
                table = capacity.setdefault(country, {})
                table[clean(r[1])] = table.get(clean(r[1]), 0.0) + value
            elif section.startswith("Native Demand"):
                demand[country] = demand.get(country, 0.0) + value
    sheet = book["Crossborder exchanges"].iter_rows(values_only=True)
    top = [next(sheet) for _ in range(11)]
    names = {j: clean(v) for j, v in enumerate(top[10]) if "->" in clean(v)}
    names = {j: n for j, n in names.items() if any(end in nodes for end in n.split("->"))}
    data = np.array([[num(r[j]) or 0.0 for j in names] for r in sheet if r[0] is not None], dtype=float)
    book.close()
    return dict(capacity=capacity, demand=demand, flows={n.replace("->", "-"): data[:, i] for i, n in enumerate(names.values())})


def flow_rows(flows: dict, edition: str, year: int, weather: str) -> list[dict]:
    rows = []
    for border, series in flows.items():
        a, b = border.split("-")[:2]
        for frm, to, directed in ((a, b, series), (b, a, -series)):
            rows.append(dict(edition=edition, year=year, weather=weather, border=border, node_from=frm, node_to=to,
                             max_flow=float(directed.max()), series=directed))
    return rows


def control_ntc(rows: list[dict], ntc: pd.DataFrame) -> pd.DataFrame:
    limits = ntc[ntc.metric.isin(("ntc_reference", "ntc_2025"))]
    reference = limits[limits.metric == "ntc_reference"].set_index(["edition", "year", "node_from", "node_to"]).value
    current = limits[limits.metric == "ntc_2025"].drop_duplicates(["node_from", "node_to"]).set_index(["node_from", "node_to"]).value
    grouped: dict[tuple, list] = {}
    for r in rows:
        grouped.setdefault((r["edition"], r["year"], r["border"], r["node_from"], r["node_to"]), []).append(r)
    out = []
    for key, group in sorted(grouped.items()):
        edition, year, border, frm, to = key
        limit = reference.get((edition, year, frm, to), np.nan)
        limit_2025 = current.get((frm, to), np.nan) if edition == "2026" else np.nan
        top = max(g["max_flow"] for g in group)
        shares = [float((g["series"] >= limit - LIMIT_TOL).mean()) if not np.isnan(limit) else np.nan for g in group]
        energy = [float(g["series"].clip(min=0).sum()) / 1e6 for g in group]
        if top == 0 and np.isnan(limit):
            verdict = "no_flow_no_limit"
        elif np.isnan(limit):
            verdict = "no_limit_in_tidy"
        elif top > limit + LIMIT_TOL:
            verdict = "above_reference"
        elif top >= limit - LIMIT_TOL:
            verdict = "reaches_reference"
        else:
            verdict = "below_reference"
        out.append(dict(edition=edition, year=year, border=border, node_from=frm, node_to=to, weathers=len(group),
                        max_flow=top, ntc_reference=limit, ntc_2025=limit_2025, hours_at_limit_share=np.mean(shares),
                        flow_twh=np.mean(energy), status=verdict))
    return pd.DataFrame(out)


def control_closure(rows: list[dict], kpi: pd.DataFrame) -> pd.DataFrame:
    """Sum of the hourly border flows of a country against its KPI imports and exports."""
    imports: dict[tuple, float] = {}
    for r in rows:
        if r["edition"] != "2026":
            continue
        energy = float(r["series"].clip(min=0).sum()) / 1e6
        for country, kind in ((r["node_to"][:2], "Imports"), (r["node_from"][:2], "Exports")):
            if r["node_from"][:2] != r["node_to"][:2]:
                key = (r["year"], r["weather"], country, kind)
                imports[key] = imports.get(key, 0.0) + energy
    out = []
    for (year, weather, country, kind), hourly in sorted(imports.items()):
        if country not in FLOW_COUNTRIES:
            continue
        found = kpi[(kpi.country == country) & (kpi.year == year) & (kpi.weather == weather) & (kpi.label == kind)
                    & (kpi.section == "Exchange [TWh]")]
        control = found.value.iloc[0] if len(found) else np.nan
        out.append(dict(year=year, weather=weather, country=country, item=kind, unit="TWh", published=control,
                        hourly=hourly, gap=hourly - control, status=status(control, hourly, "TWh")))
    return pd.DataFrame(out)


def input_series(book, node: str) -> dict[str, np.ndarray]:
    """Hourly columns of one node sheet of a demand workbook, keyed by weather label."""
    header, data = None, []
    for r in book[node].iter_rows(values_only=True):
        if header is None:
            if clean(r[0]) == "Date":
                header = [clean(v) for v in r]
                cols = [j for j in range(header.index("Hour") + 1, len(header)) if header[j]]
            continue
        data.append([num(r[j]) or 0.0 for j in cols])
    values = np.array(data, dtype=float)
    return {header[j]: values[:, i] for i, j in enumerate(cols)}


def evidence_bg_peak(run: Run, dem: pd.DataFrame, emit) -> None:
    """Shape of the Bulgarian market demand, 2040 against 2050, and the same tests on every other node."""
    pack = run.package("2026/inputs", "Demand.zip")
    with zipfile.ZipFile(pack["path"]) as z:
        for year in (2040, 2050):
            member = next(m for m in z.namelist() if re.search(rf"Demand/{year}/ELECTRICITY_MARKET {year}\.xlsx$", m))
            book = workbook(z.read(member))
            for node in [n for n in run.nodes if n in book.sheetnames]:
                for weather, series in input_series(book, node).items():
                    if weather not in tuple(f"WS{c}" for c in WEATHER[year]):
                        continue
                    stamps = pd.date_range(f"{year}-01-01", periods=len(series), freq="h")
                    step = np.diff(series)
                    limit = STEP_SHARE * series.max()
                    night = series[stamps.hour < 6].mean()
                    midday = series[(stamps.hour >= 9) & (stamps.hour < 18)].mean()
                    rising = np.where(step > limit)[0]
                    emit("bg_peak_2050", node, year, weather, "steps_above_20pct_of_peak",
                         int((np.abs(step) > limit).sum()), "count")
                    emit("bg_peak_2050", node, year, weather, "night_over_midday", night / midday, "ratio")
                    if node != "BG00":
                        continue
                    top = int(series.argmax())
                    winter = series[np.isin(stamps.month, (12, 1, 2))].mean()
                    summer = series[np.isin(stamps.month, (6, 7, 8))].mean()
                    for item, value, unit in (
                            ("peak", series.max() / 1e3, "GW"),
                            ("mean", series.mean() / 1e3, "GW"),
                            ("load_factor", series.mean() / series.max(), "share"),
                            ("hour_of_peak", top + 1, "hour of year"),
                            ("month_of_peak", int(stamps[top].month), "month"),
                            ("hours_above_7_5_gw", int((series > 7500).sum()), "hours"),
                            ("hours_above_10_gw", int((series > 10000).sum()), "hours"),
                            ("largest_hourly_step", np.abs(step).max() / 1e3, "GW"),
                            ("median_hours_between_rising_steps", float(np.median(np.diff(rising))) if len(rising) > 1 else 0, "hours"),
                            ("winter_over_summer_mean", winter / summer, "ratio")):
                        emit("bg_peak_2050", "BG00", year, weather, item, value, unit)
            book.close()
    peaks = dem[(dem.edition == "2026") & (dem.metric == "peak") & (dem.technology == "market")]
    means = peaks.groupby(["node", "year"]).value.mean().unstack()
    for node in means.index:
        if {2040, 2050} <= set(means.columns) and means.loc[node, 2040] > 0:
            emit("bg_peak_2050", node, 2050, "mean", "peak_ratio_2050_over_2040", means.loc[node, 2050] / means.loc[node, 2040], "ratio")


def evidence_gr03(dem: pd.DataFrame, cap: pd.DataFrame, emit) -> None:
    sel = dem[(dem.edition == "2026") & (dem.metric == "annual_energy") & (dem.technology == "market")]
    for (node, year), value in sel[sel.node.isin(("GR00", "GR03"))].groupby(["node", "year"]).value.mean().items():
        emit("gr03_2050", node, year, "mean", "market_demand", value, "TWh")
    sel = cap[(cap.edition == "2026") & (cap.metric == "capacity") & (cap.level == "total") & (cap.node == "GR03")]
    for year, value in sel.groupby("year").value.sum().items():
        emit("gr03_2050", "GR03", year, "", "capacity_total", value / 1e3, "GW")


def evidence_flat_demand(run: Run, yearly: dict[int, dict], emit) -> None:
    """Are the 2024 UA and MD series the same in 2030 and 2040, weather by weather?"""
    pack = run.package("2024/inputs", "Demand-Profiles.zip")
    with zipfile.ZipFile(pack["path"]) as z:
        members = {y: next(m for m in z.namelist() if re.search(rf"/NT/Electricity demand profiles/{y}_National Trends\.xlsx$", m)
                           and "__MACOSX" not in m) for y in (2030, 2040)}
        books = {y: workbook(z.read(m)) for y, m in members.items()}
        for node in ("UA00", "MD00", "RO00"):
            first, second = (input_series(books[y], node) for y in (2030, 2040))
            for year, book in yearly.items():
                emit("flat_demand_2024", node, year, "2009", "market_run_native_demand", book["demand"][node[:2]] / 1e3, "TWh")
            weathers = sorted(set(first) & set(second))
            same = [w for w in weathers if len(first[w]) == len(second[w]) and np.allclose(first[w], second[w], atol=0.5)]
            ratios = [second[w].sum() / first[w].sum() for w in weathers if first[w].sum() > 0]
            emit("flat_demand_2024", node, 0, "all", "weather_years_compared", len(weathers), "count")
            emit("flat_demand_2024", node, 0, "all", "weather_years_identical", len(same), "count")
            emit("flat_demand_2024", node, 0, "all", "energy_ratio_2040_over_2030", float(np.mean(ratios)), "ratio")
        for book in books.values():
            book.close()


def evidence_not_in_market(cap: pd.DataFrame, kpi: pd.DataFrame, emit) -> None:
    """Input capacity with no published capacity, generation or hourly column in the 2026 market run."""
    total = cap[(cap.edition == "2026") & (cap.level == "total") & (cap.metric == "capacity")]
    for node, tech, labels in (("TR00", "Offshore wind", ("Wind Offshore Radial", "Wind Offshore Hub")),
                               ("MD00", "Photovoltaic", ("Solar PV",))):
        found = kpi[(kpi.country == node[:2]) & (kpi.weather == "Weighted WS") & kpi.label.isin(labels)]
        for year in WEATHER:
            rows = found[found.year == year]
            value = total[(total.year == year) & (total.node == node) & (total.technology == tech)].value.sum()
            emit("not_in_market_run", node, year, "2026", f"input_capacity {tech}", value, "MW")
            for section, unit in (("Installed Capacity [GW]", "GW"), ("Generation [TWh]", "TWh")):
                emit("not_in_market_run", node, year, "2026", f"kpi {section}",
                     rows[rows.section == section].value.sum(), unit)


def evidence_tr_other(cap: pd.DataFrame, yearly: dict[int, dict], kpi: pd.DataFrame, emit) -> None:
    sel = cap[(cap.node == "TR00") & (cap.family == "other_res") & (cap.metric == "capacity")]
    for (edition, year), value in sel.groupby(["edition", "year"]).value.sum().items():
        emit("tr_other_res", "TR00", year, edition, "sum_of_technologies", value, "MW")
    for year, book in yearly.items():
        emit("tr_other_res", "TR00", year, "2024", "market_run_capacity", book["capacity"]["TR"].get("Others renewable", np.nan), "MW")
    found = kpi[(kpi.country == "TR") & (kpi.weather == "Weighted WS") & (kpi.section == "Installed Capacity [GW]")]
    for year in WEATHER:
        rows = found[found.year == year]
        emit("tr_other_res", "TR00", year, "2026", "kpi_capacity",
             rows[rows.label.isin(("Other RES", "Biofuel"))].value.sum() * 1e3, "MW")
        emit("tr_other_res", "TR00", year, "2026", "kpi_battery", rows[rows.label == "Batteries"].value.sum() * 1e3, "MW")


def evidence_gas_factor(factors: pd.DataFrame, prices: pd.DataFrame, emit) -> None:
    """Fossil, biomethane and synthetic shares that reproduce both the factor and the blended price."""
    for edition in ("2024", "2026"):
        rows = factors[(factors.edition == edition) & (factors.technology == "Gas CCGT new")]
        for year, factor in zip(rows.year, rows.value):
            emit("gas_co2_factor", "ALL", year, edition, "factor", factor, "kg/net GJ")
            fossil = factor / FOSSIL_GAS_FACTOR
            emit("gas_co2_factor", "ALL", year, edition, "fossil_share_from_factor", fossil, "share")
            price = prices[(prices.edition == edition) & (prices.year == year)].set_index("technology").value
            if "Gas" not in price:
                continue
            gas, natural, bio, syn = (float(price[k]) for k in ("Gas", "Natural Gas", "Biomethane", "Synthetic Methane"))
            synthetic = (gas - fossil * natural - (1 - fossil) * bio) / (syn - bio)
            emit("gas_co2_factor", "ALL", year, edition, "synthetic_share_implied", synthetic, "share")
            emit("gas_co2_factor", "ALL", year, edition, "biomethane_share_implied", 1 - fossil - synthetic, "share")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="Folder with manifest.csv, default from the config")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = args.root or (REPO / cfg["tyndp_root"]).resolve()
    run = Run(root, cfg)

    read = lambda name, **kw: pd.read_csv(OUT_DIR / f"{name}.csv", dtype={"edition": str}, low_memory=False, **kw)
    cap, dem, ntc = read("capacity"), read("demand_annual", keep_default_na=False, na_values=[""]), read("ntc")
    factors, prices, exchanges = read("emission_factors"), read("fuel_co2_prices"), read("exchanges")
    dem["weather"] = dem["weather"].astype(str)

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

    print("Hourly dashboards", flush=True)
    ts_pack = run.package("2026/outputs", "NT+_TimeSeriesDashboard.zip")
    sources.append(dict(table=f"source {ts_pack['source_package']}", rows=0, sha256=ts_pack["source_sha256"]))
    zones, rows, mismatches = [], [], 0
    with zipfile.ZipFile(ts_pack["path"]) as z:
        for year, cases in WEATHER.items():
            for case in cases:
                print(f"  {year} WS{case}", flush=True)
                seen: dict[str, np.ndarray] = {}
                for country in FLOW_COUNTRIES + DEMAND_ONLY_COUNTRIES:
                    found, flows = read_hourly(z, country, year, case, country in FLOW_COUNTRIES)
                    zones += found
                    for border, series in flows.items():
                        if border in seen:
                            mismatches += int(not np.allclose(seen[border], series, atol=0.01))
                        else:
                            seen[border] = series
                rows += flow_rows({b: s for b, s in seen.items() if b.count("-") == 1}, "2026", year, f"WS{case}")
    for year, book_2024 in yearly.items():
        rows += flow_rows(book_2024["flows"], "2024", year, "2009")
    hourly = pd.DataFrame(zones).drop_duplicates(["year", "weather", "zone"]).sort_values(["zone", "year", "weather"])

    print("Evidence for the anomalies", flush=True)
    evidence: list[dict] = []

    def emit(anomaly, node, year, weather, item, value, unit):
        evidence.append(dict(anomaly=anomaly, node=node, year=year, weather=weather, item=item,
                             value=float(value), unit=unit))

    evidence_bg_peak(run, dem, emit)
    evidence_gr03(dem, cap, emit)
    evidence_flat_demand(run, yearly, emit)
    evidence_not_in_market(cap, kpi, emit)
    evidence_tr_other(cap, yearly, kpi, emit)
    evidence_gas_factor(factors, prices, emit)
    emit("parser", "ALL", 0, "", "border_columns_differing_between_country_files", mismatches, "count")

    tables = {
        "controls_capacity": control_capacity(cap, kpi, yearly),
        "controls_demand": control_demand(dem, kpi, yearly, hourly, exchanges),
        "controls_ntc": control_ntc(rows, ntc),
        "controls_closure": control_closure(rows, kpi),
        "controls_anomalies": pd.DataFrame(evidence),
    }
    summary = []
    for name, frame in tables.items():
        if "status" in frame:
            keys = ["edition", "status"] if "edition" in frame else ["status"]
            for key, n in frame.groupby(keys).size().items():
                key = key if isinstance(key, tuple) else ("2026", key)
                summary.append(dict(control=name, edition=key[0], status=key[1], rows=int(n)))
    tables["controls_summary"] = pd.DataFrame(summary)

    manifest: list[dict] = []
    for name, frame in tables.items():
        write(frame.round(6), name, manifest)
    pd.DataFrame(manifest + sources).to_csv(OUT_DIR / "controls_manifest.csv", index=False, lineterminator="\n")
    print(tables["controls_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
