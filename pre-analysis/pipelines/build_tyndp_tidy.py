# -*- coding: utf-8 -*-
"""Tidy long tables from the raw TYNDP packages, one table per data family.

Reads the zip packages listed in Data/EU/TYNDP/manifest.csv without extracting
them and writes to pre-analysis/output_tyndp/:

    capacity.csv         installed capacity and storage by node and technology
    thermal.csv          units, outages, ramps, must run by node, common data as node ALL
    exchanges.csv        non modelled partners and annual sums of the hourly exchange series
    demand_annual.csv    annual energy and peak by node, demand stream and weather year
    ntc.csv              2025 NTC, reference grid and project increments by directed border
    candidates.csv       TYNDP 2024 grid investment candidates with capex
    generator_costs.csv  TYNDP 2024 generator investment costs
    fuel_co2_prices.csv  fuel and CO2 prices
    emission_factors.csv CO2 factors by fuel and plant type
    coverage.csv         rows per node, family and edition, with the files read
    issues.csv           everything a parser could not place, nothing is fixed silently
    checks.csv           spot checks from config/tyndp_scope.yaml
    tidy_manifest.csv    rows and sha256 of each table

Method
------
* Scope, editions and spot checks come from config/tyndp_scope.yaml. National
  Trends 2024 and National Trends+ 2026 stay apart.
* Every row keeps its package, its member inside the package and the sha256 of
  the package from the manifest. A package whose sha256 differs from the
  manifest stops the run: new data needs a new download entry first.
* Year and scenario come from the file path. The 2026 PEMMDB header cells are
  empty, the market node cell is filled in both editions.
* Hourly series only give annual aggregates. Zero capacities are dropped.
* GW become MW. Storage energy keeps its source unit, GWh or MWh.
* level = total or split. Sum only the total rows: splits (e-market, dedicated,
  shared RES, radial, hub, rooftop in 2026) are part of a total row.
* An Other Non-RES price band repeated for other weather years counts once.
* Nothing here writes to epm/input. Mapping to EPM parameters is a later step.

The output is deterministic: a rerun on the same packages gives the same hashes.

Usage (from EPM/):  python pre-analysis/pipelines/build_tyndp_tidy.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd
import yaml

_PRE_ANALYSIS = Path(__file__).resolve().parent.parent
REPO = _PRE_ANALYSIS.parent
CONFIG = _PRE_ANALYSIS / "config" / "tyndp_scope.yaml"
OUT_DIR = _PRE_ANALYSIS / "output_tyndp"

SOURCE_COLS = ["source_package", "source_member", "source_sha256"]
NODE_COLS = ["edition", "scenario", "year", "node", "family", "technology", "level", "metric", "value", "unit"]
PEMMDB = {"2024": ("2024/inputs", "PEMMDB2.zip"), "2026": ("2026/inputs", "PEMMDB_2.X.zip")}
MEMBER_RE = re.compile(r"PEMMDB_(\w+?)_NationalTrends_(\d{4})\.xlsx$")
THERMAL_COLS = [  # column, metric, unit
    (3, "units", "count"),
    (4, "biofuel_units", "count"),
    (5, "biofuel_usage", "percent"),
    (32, "forced_outage_rate", "percent"),
    (33, "planned_outage_days", "days"),
    (34, "planned_outage_winter_min", "percent"),
    (35, "min_stable_power", "percent of max power"),
    (36, "ramp_up", "MW/h"),
    (37, "ramp_down", "MW/h"),
    (38, "fixed_generation_reduction", "percent of max power"),
]
COMMON_SECOND_TABLE = [  # column, metric, unit
    (4, "forced_outage_rate", "share"),
    (5, "mean_time_to_repair", "days"),
    (6, "planned_outage_days", "days"),
    (7, "planned_outage_winter_min", "share"),
    (8, "min_stable_power", "share of max power"),
    (9, "ramp_up", "share of max power per minute"),
    (10, "ramp_down", "share of max power per minute"),
    (11, "fixed_generation_reduction", "share of max power"),
]
PRICE_COMPONENTS = ("natural gas", "biomethane", "synthetic methane", "crude oil")


def num(value) -> float | None:
    """Number from a cell, None for empty or text cells."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def clean(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean(value).lower()).strip("_")


class Run:
    """Holds the packages, the scope and the rows of one build."""

    def __init__(self, root: Path, cfg: dict):
        self.root = root
        self.cfg = cfg
        self.core = list(cfg["core_nodes"])
        self.nodes = self.core + list(cfg["neighbour_nodes"])
        self.tables: dict[str, list[dict]] = {}
        self.issues: list[dict] = []
        self.files: list[dict] = []
        manifest = pd.read_csv(root / "manifest.csv", dtype=str)
        self.sha = {(r.folder, r.file): r.sha256 for r in manifest.itertuples()}

    def package(self, folder: str, name: str) -> dict:
        """Check a package against the manifest and return its source fields."""
        path = self.root / folder / name
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        expected = self.sha[(folder, name)]
        if digest.hexdigest() != expected:
            raise SystemExit(f"{folder}/{name}: sha256 differs from manifest.csv")
        return {"path": path, "source_package": f"{folder}/{name}", "source_sha256": expected}

    def add(self, table: str, pack: dict, member: str, **row) -> None:
        row.update(source_package=pack["source_package"], source_member=member, source_sha256=pack["source_sha256"])
        self.tables.setdefault(table, []).append(row)

    def issue(self, edition: str, member: str, text: str) -> None:
        self.issues.append({"edition": edition, "member": member, "issue": text})


def workbook(data: bytes) -> openpyxl.Workbook:
    return openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)


def head(ws, n: int) -> list[tuple]:
    """First n rows, padded so that positional access is safe."""
    return [tuple(r) + (None,) * 60 for r in ws.iter_rows(values_only=True, max_row=n)]


def parse_thermal(run, ws, emit, emit_thermal):
    rows = head(ws, 70)
    start = next(i for i, r in enumerate(rows) if clean(r[0]).startswith("Fuel & Plant type"))
    fuel = ""
    for i in range(start + 4, len(rows)):
        r = rows[i]
        fuel = clean(r[0]) or fuel
        cap = num(r[2])
        if cap is None or not fuel:
            continue
        tech = f"{fuel} {clean(r[1])}".strip()
        if cap == 0:
            continue
        emit("thermal", tech, "total", "capacity", cap, "MW")
        for col, metric, unit in THERMAL_COLS:
            value = num(r[col])
            if value is not None:
                emit_thermal(tech, metric, value, unit)
        below = rows[i + 1] if i + 1 < len(rows) else ()
        months = [num(v) for v in below[7:19]] if clean(below[6:7][0] if len(below) > 6 else "").startswith("%") else []
        months = [m for m in months if m is not None]
        if months:
            emit_thermal(tech, "must_run_share_mean", sum(months) / len(months), "percent of capacity")


def parse_res(run, ws, sheet, edition, emit, flag):
    for r in head(ws, 30)[7:]:
        label, value = clean(r[0]), num(r[1])
        if not value:
            continue
        unit = re.search(r"\((GW|GWh|MW|MWh)\)", label)
        if unit is None:
            flag(f"{sheet}: no unit in '{label}'")
            continue
        name = re.sub(r"^(Installed capacities|Storage capacities)\s*", "", label)
        name = re.sub(r"\s*\((GW|GWh|MW|MWh)\).*$", "", name).strip(" :")
        metric = "storage_energy" if label.startswith("Storage") else "capacity"
        total = re.sub(r"\s*Total$", "", name)
        if metric == "storage_energy":
            level = "total"
        elif name.endswith("Total") or total in ("Onshore wind", "Offshore wind", "Photovoltaic", "Thermal Solar",
                                                 "Solar Thermal with Storage"):
            level = "total"
        elif total == "Rooftop":
            level = "split" if edition == "2026" else "total"
        else:
            level = "split"
        factor = 1000.0 if unit.group(1) == "GW" else 1.0
        emit(sheet.lower(), total, level, metric, value * factor, "MW" if unit.group(1) in ("GW", "MW") else unit.group(1))


def parse_hydro(run, ws, emit, flag):
    for r in head(ws, 30)[8:]:
        label, value = clean(r[0]), num(r[1])
        if not value:
            continue
        m = re.match(r"(.*?)\s*-\s*(.*?)\s*\((GWh|MW)\)", label)
        if m is None:
            flag(f"Hydro: no pattern for '{label}'")
            continue
        name, what, unit = m.group(1), m.group(2).lower(), m.group(3)
        tech = "Hydro " + re.sub(r"\s*\(.*\)", "", name) + (" closed loop" if "closed" in name else " open loop" if "open" in name else "")
        if "reservoir" in what or unit == "GWh":
            metric = "storage_energy"
        elif "turbin" in what:
            metric = "capacity"
        elif "pump" in what:
            metric = "pumping_capacity"
        else:
            flag(f"Hydro: no metric for '{label}'")
            continue
        emit("hydro", tech, "total", metric, abs(value), unit)


def parse_battery(run, ws, emit):
    rows = head(ws, 30)
    start = next(i for i, r in enumerate(rows) if clean(r[0]) == "Type")
    for r in rows[start + 1:]:
        name = clean(r[0])
        if not name or not num(r[2]):
            continue
        tech = re.sub(r"\s*Total$", "", name)
        level = "total" if tech == "Battery" else "split"
        emit("battery", tech, level, "capacity", num(r[2]), "MW")
        emit("battery", tech, level, "pumping_capacity", abs(num(r[3]) or 0), "MW")
        emit("battery", tech, level, "storage_energy", num(r[4]) or 0, "MWh")
        emit("battery", tech, level, "efficiency", num(r[6]) or 0, "share")


def parse_electrolyser(run, ws, emit):
    rows = head(ws, 30)
    start = next(i for i, r in enumerate(rows) if clean(r[0]) == "Type")
    for r in rows[start + 1:]:
        name = clean(r[0])
        if not name or not num(r[2]):
            continue
        emit("electrolyser", name, "total", "capacity", num(r[2]), "MW")
        emit("electrolyser", name, "total", "efficiency", num(r[4]) or 0, "share")
        emit("electrolyser", name, "total", "storage_energy", num(r[5]) or 0, "GWh")


def parse_reserves(run, ws, emit):
    section = ""
    for r in head(ws, 30)[8:]:
        label = clean(r[0])
        if not label:
            continue
        if r[1] is None:
            section = "FCR" if "FCR" in label else "FRR" if "FRR" in label else "Simplified"
            continue
        value = num(r[2])
        if value:
            emit("reserves", f"{section}: {label}", "total" if label.startswith("Total") else "split", "requirement", value, "MW")


def parse_other_res(run, ws, emit, flag):
    rows = head(ws, 12)
    names = next((r for r in rows if "Small Biomass" in [clean(v) for v in r]), None)
    if names is None:
        return
    i = rows.index(names)
    parts = 0.0
    for col in range(4, 9):
        value = num(rows[i + 1][col])
        if value:
            parts += value
            emit("other_res", clean(names[col]), "total", "capacity", value, "MW")
    total = num(names[1]) or 0.0
    if abs(total - parts) > 0.5:
        flag(f"Other RES: sheet total {total} differs from the technologies, {parts}")


def parse_other_non_res(run, ws, emit, flag):
    rows = head(ws, 17)
    labels = {clean(r[1]): r for r in rows if clean(r[1])}
    caps = labels.get("Installed capacity (MW)")
    if caps is None:
        return
    bands: dict[str, float] = {}
    for col in range(2, 40):
        value = num(caps[col])
        if not value:
            continue
        band = clean(rows[7][col]) or f"Price Band {col - 1}"
        # A band repeated for other weather years is the same plant, not extra capacity.
        if band in bands:
            if abs(bands[band] - value) > 1e-6:
                flag(f"Other Non-RES: {band} repeats with {value} after {bands[band]}")
            continue
        bands[band] = value
        kind = clean(labels.get("PEMMDB type(s)", (None,) * 60)[col]).replace("OtherNon-RES/", "")
        tech = f"Other non-RES {kind} {band}".replace("  ", " ")
        emit("other_non_res", tech, "total", "capacity", value, "MW")
        offer = num(labels.get("Avg. Market Offer Price (€/MWh)", (None,) * 60)[col])
        if offer is not None:
            emit("other_non_res", tech, "total", "offer_price", offer, "EUR/MWh")


def parse_dsr(run, ws, emit):
    rows = head(ws, 14)
    labels = {clean(r[1]): r for r in rows if clean(r[1])}
    caps = labels.get("Capacity")
    if caps is None:
        return
    for col in range(2, 13):
        value = num(caps[col])
        if not value:
            continue
        tech = f"DSR band {col - 1}"
        emit("dsr", tech, "total", "capacity", value, "MW")
        price = num(labels.get("Price", (None,) * 60)[col])
        if price is not None:
            emit("dsr", tech, "total", "offer_price", price, "EUR/MWh")


def parse_exchanges(run, ws, add):
    """Partner capacities, then annual sums of the hourly total exchange column."""
    partner = ""
    pos = neg = 0.0
    seen = False
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        r = tuple(r) + (None,) * 4
        label = clean(r[0]).lower()
        if i < 27:
            if label.startswith("name of the non"):
                partner = clean(r[1])
            elif partner and label.startswith("average import"):
                add(partner, "import_capacity", num(r[1]), "MW")
            elif partner and label.startswith("average export"):
                add(partner, "export_capacity", num(r[1]), "MW")
            continue
        value = num(r[2])
        if i > 27 and value is not None:
            seen = True
            pos += max(value, 0.0)
            neg += min(value, 0.0)
    if seen and (pos or neg):
        add("Total hourly series", "positive_sum", pos / 1e6, "TWh")
        add("Total hourly series", "negative_sum", neg / 1e6, "TWh")


def build_pemmdb(run: Run) -> None:
    for edition, (folder, name) in PEMMDB.items():
        pack = run.package(folder, name)
        scenario = run.cfg["editions"][edition]["scenario"]
        years = run.cfg["editions"][edition]["years"]
        with zipfile.ZipFile(pack["path"]) as z:
            members = [n for n in z.namelist() if "__MACOSX" not in n and MEMBER_RE.search(n)]
            for member in sorted(members):
                node, year = MEMBER_RE.search(member).groups()
                year = int(year)
                if node not in run.nodes or year not in years:
                    continue
                run.files.append({"edition": edition, "family": "pemmdb", "node": node, "year": year})
                wb = workbook(z.read(member))
                base = dict(edition=edition, scenario=scenario, year=year, node=node)
                cell = next(iter(head(wb["Thermal"], 4)[3][1:2]), None)
                if clean(cell) != node:
                    run.issue(edition, member, f"market node cell reads '{clean(cell)}'")

                def emit(family, tech, level, metric, value, unit):
                    run.add("capacity", pack, member, **base, family=family, technology=tech, level=level,
                            metric=metric, value=value, unit=unit)

                def emit_thermal(tech, metric, value, unit):
                    run.add("thermal", pack, member, **base, family="thermal", technology=tech, level="total",
                            metric=metric, value=value, unit=unit)

                def add_exchange(partner, metric, value, unit):
                    if value:
                        run.add("exchanges", pack, member, **base, family="exchanges", technology=partner,
                                level="total", metric=metric, value=value, unit=unit)

                def flag(text):
                    run.issue(edition, member, text)

                parse_thermal(run, wb["Thermal"], emit, emit_thermal)
                parse_res(run, wb["Wind"], "Wind", edition, emit, flag)
                parse_res(run, wb["Solar"], "Solar", edition, emit, flag)
                parse_hydro(run, wb["Hydro"], emit, flag)
                parse_battery(run, wb["Battery"], emit)
                parse_electrolyser(run, wb["Electrolyser"], emit)
                parse_reserves(run, wb["Reserves"], emit)
                parse_other_res(run, wb["Other RES"], emit, flag)
                parse_other_non_res(run, wb["Other Non-RES"], emit, flag)
                parse_dsr(run, wb["DSR"], emit)
                parse_exchanges(run, wb["Exchanges"], add_exchange)
                wb.close()
    run.issue("2024", "Solar sheets", "Assumption: Rooftop is added to Photovoltaic. The file does not say so, "
              "RO 2030 gives 7.27 + 1.03 = 8.30 GW. In 2026 Rooftop is a part of Photovoltaic.")


def demand_sheet(run, ws, base, stream, pack, member) -> None:
    """Annual energy and peak per weather column of one node sheet."""
    header, totals, peaks, labels, previous = None, {}, {}, {}, None
    for r in ws.iter_rows(values_only=True):
        if header is None:
            if clean(r[0]) == "Date":
                header = [clean(v) for v in r]
                first = header.index("Hour") + 1
                cols = [j for j in range(first, len(header)) if header[j]]
                if previous:
                    labels = {j: clean(previous[j]) for j in cols if j < len(previous)}
            previous = r
            continue
        for j in cols:
            value = num(r[j]) if j < len(r) else None
            if value is not None:
                totals[j] = totals.get(j, 0.0) + value
                peaks[j] = max(peaks.get(j, 0.0), value)
    if header is None:
        run.issue(base["edition"], member, f"{ws.title}: no Date header")
        return
    for j, total in totals.items():
        if total == 0:
            continue
        common = dict(**base, family="demand", technology=stream, level="total", weather=header[j],
                      weather_label=labels.get(j, ""))
        run.add("demand_annual", pack, member, **common, metric="annual_energy", value=total / 1e6, unit="TWh")
        # The behind the meter stream is negative, self consumption has no peak to report.
        if total > 0:
            run.add("demand_annual", pack, member, **common, metric="peak", value=peaks[j] / 1e3, unit="GW")


def build_demand(run: Run) -> None:
    streams = {
        "ELECTRICITY_MARKET": "market",
        "ELECTRICITY_PROSUMER": "prosumer",
        "ELECTRICITY_PROSUMER_BEHIND_THE_METER_FIXED_LOAD": "prosumer_btm_fixed",
        "EV_FIXED_LOAD_PROFILES_ELECTRICITY_MARKET": "ev_market",
        "EV_FIXED_LOAD_PROFILES_ELECTRICITY_PROSUMER": "ev_prosumer",
    }
    pack = run.package("2024/inputs", "Demand-Profiles.zip")
    jobs = []
    with zipfile.ZipFile(pack["path"]) as z:
        for member in z.namelist():
            m = re.search(r"/NT/Electricity demand profiles/(\d{4})_National Trends\.xlsx$", member)
            if m and "__MACOSX" not in member:
                jobs.append((member, int(m.group(1)), "total"))
        run_demand_jobs(run, z, pack, "2024", jobs)
    pack = run.package("2026/inputs", "Demand.zip")
    jobs = []
    with zipfile.ZipFile(pack["path"]) as z:
        for member in z.namelist():
            m = re.search(r"Demand/(\d{4})/(\w+) \d{4}\.xlsx$", member)
            if m and "__MACOSX" not in member and m.group(2) in streams:
                jobs.append((member, int(m.group(1)), streams[m.group(2)]))
        run_demand_jobs(run, z, pack, "2026", jobs)


def run_demand_jobs(run, z, pack, edition, jobs) -> None:
    scenario = run.cfg["editions"][edition]["scenario"]
    for member, year, stream in sorted(jobs):
        print(f"  demand {edition} {year} {stream}", flush=True)
        wb = workbook(z.read(member))
        for node in run.nodes:
            if node in wb.sheetnames:
                run.files.append({"edition": edition, "family": "demand", "node": node, "year": year})
                base = dict(edition=edition, scenario=scenario, year=year, node=node)
                demand_sheet(run, wb[node], base, stream, pack, member)
        wb.close()


def directed(run, pack, member, base, border, a, b, metric, forward, backward, project="") -> None:
    for frm, to, value in ((a, b, forward), (b, a, backward)):
        if value is not None:
            run.add("ntc", pack, member, **base, border=border, node_from=frm, node_to=to, project=project,
                    metric=metric, value=value, unit="MW")


def build_ntc(run: Run) -> None:
    def in_scope(border):
        ends = str(border).split("-")
        return len(ends) == 2 and any(e in run.core for e in ends)

    pack = run.package("2024/inputs", "Line-data.zip")
    member = "Line data/ReferenceGrid_Electricity.xlsx"
    base = dict(edition="2024", scenario="NT", year=2030)
    with zipfile.ZipFile(pack["path"]) as z:
        wb = workbook(z.read(member))
    for r in wb["2030"].iter_rows(values_only=True, min_row=2):
        if in_scope(r[0]):
            a, b = r[0].split("-")
            directed(run, pack, member, base, r[0], a, b, "ntc_reference", num(r[1]), num(r[2]))
    wb.close()

    pack = run.package("2026/inputs", "Line-data.zip")
    member = "Line-data/ReferenceGrid_Electricity.xlsx"
    with zipfile.ZipFile(pack["path"]) as z:
        wb = workbook(z.read(member))
    seen_2025: dict[str, tuple] = {}
    for sheet in wb.sheetnames:
        year = int(sheet.split("_")[1])
        base = dict(edition="2026", scenario="NT+", year=year)
        it = wb[sheet].iter_rows(values_only=True)
        hdr = [clean(h) for h in next(it)]
        projects = sorted({h[:-2] for h in hdr if re.fullmatch(r"\w+_d", h)})
        where = {h: j for j, h in enumerate(hdr)}
        for r in it:
            if not in_scope(r[0]):
                continue
            a, b = r[0].split("-")
            now = (num(r[1]), num(r[2]))
            if r[0] in seen_2025 and seen_2025[r[0]] != now:
                run.issue("2026", member, f"{r[0]}: 2025 NTC differs in {sheet}, {seen_2025[r[0]]} vs {now}")
            if r[0] not in seen_2025:
                seen_2025[r[0]] = now
                directed(run, pack, member, dict(base, year=2025), r[0], a, b, "ntc_2025", *now)
            ref = (num(r[3]), num(r[4]))
            directed(run, pack, member, base, r[0], a, b, "ntc_reference", *ref)
            added = [0.0, 0.0]
            for p in projects:
                d = num(r[where[f"{p}_d"]]) or 0.0
                i = num(r[where[f"{p}_in"]]) or 0.0
                if d or i:
                    added[0] += d
                    added[1] += i
                    directed(run, pack, member, base, r[0], a, b, "project_increment", d, i, project=p)
            for k in (0, 1):
                if None not in (now[k], ref[k]) and abs(now[k] + added[k] - ref[k]) > 0.5:
                    run.issue("2026", member, f"{r[0]} {sheet} direction {k + 1}: 2025 {now[k]} + projects {added[k]} != reference {ref[k]}")
    wb.close()


def build_investment(run: Run) -> None:
    pack = run.package("2024/inputs", "Investment-Datasets.zip")
    with zipfile.ZipFile(pack["path"]) as z:
        member = "Investment Datasets/GRID.xlsx"
        wb = workbook(z.read(member))
        for r in wb["Electricity"].iter_rows(values_only=True, min_row=2):
            if r[1] in run.core or r[2] in run.core:
                run.add("candidates", pack, member, edition="2024", scenario=clean(r[4]), year=r[3], border=clean(r[0]),
                        node_from=r[1], node_to=r[2], capacity_forward_mw=num(r[5]), capacity_backward_mw=num(r[6]),
                        capex_meur=num(r[7]))
        wb.close()
        member = "Investment Datasets/GENERATOR.xlsx"
        wb = workbook(z.read(member))
        for r in wb["TEMPLATE"].iter_rows(values_only=True, min_row=2):
            if r[0] == "All" or r[0] in run.nodes:
                run.add("generator_costs", pack, member, edition="2024", scenario=clean(r[2]), year=r[1], node=r[0],
                        technology=clean(r[3]), capex_eur_per_mw=num(r[4]), opex_eur_per_mw_year=num(r[5]),
                        efficiency=num(r[6]))
        wb.close()


def build_prices(run: Run) -> None:
    def price(pack, member, edition, scenario, year, name, value, unit):
        role = "component" if name.lower().strip() in PRICE_COMPONENTS else "used"
        if value is not None:
            run.add("fuel_co2_prices", pack, member, edition=edition, scenario=scenario, year=year,
                    technology=name, role=role, metric="price", value=value, unit=unit)

    pack = run.package("2024/inputs", "Prices.zip")
    member = "Prices/2023 06 22 TYNDP 2024 Commodity prices Final.xlsx"
    with zipfile.ZipFile(pack["path"]) as z:
        wb = workbook(z.read(member))
    rows = head(wb["Matrix 2024"], 23)
    years = {j: int(rows[2][j]) for j in (3, 4, 5)}
    for r in rows[3:22]:
        name = clean(r[1])
        if not name or " DE" in name[-3:] or " GA" in name[-3:]:
            continue
        scenario = "NT" if "NT+" in name else "All"
        name = "Gas" if name.startswith("Gas (blend") else name
        unit = clean(r[2]).replace("€", "EUR2021").replace("Gj", "GJ")
        for j, year in years.items():
            price(pack, member, "2024", scenario, year, name, num(r[j]), unit)
    wb.close()

    pack = run.package("2026/inputs", "TYNDP2026Scenarios_Fuel_CO2_Prices.xlsx")
    member = "TYNDP2026Scenarios_Fuel_CO2_Prices.xlsx"
    wb = openpyxl.load_workbook(pack["path"], read_only=True, data_only=True)
    rows = head(wb["NT"], 20)
    years = {j: int(rows[3][j]) for j in (3, 4, 5, 6)}
    unit = ""
    for r in rows[4:20]:
        unit = clean(r[0]) or unit
        for j, year in years.items():
            price(pack, member, "2026", "NT+", year, clean(r[1]), num(r[j]), unit.replace("€", "EUR"))
    wb.close()


def build_emissions(run: Run) -> None:
    def table(pack, member, wb, edition, scenario):
        rows = head(wb["CO2 emission factor"], 32)
        cols = {j: int(m.group(1)) for j, v in enumerate(rows[2]) for m in [re.fullmatch(r"NT(\d{4})", clean(v))] if m}
        for r in rows[4:]:
            if num(r[1]) is None:
                continue
            tech = f"{clean(r[2])} {clean(r[3]).replace('-', '')}".strip()
            for j, year in cols.items():
                value = num(r[j])
                if value is not None:
                    run.add("emission_factors", pack, member, edition=edition, scenario=scenario, year=year,
                            technology=tech, metric="co2_factor", value=value, unit="kg/net GJ")

    pack = run.package("2024/inputs", "CO2-emission-factors-in-TYNDP2024-v2.xlsx.zip")
    member = "CO2 emission factors in TYNDP2024 v2.xlsx"
    with zipfile.ZipFile(pack["path"]) as z:
        wb = workbook(z.read(member))
    table(pack, member, wb, "2024", "NT")
    wb.close()
    pack = run.package("2026/inputs", "CO2_emission_factors_in_TYNDP2026.xlsx")
    wb = openpyxl.load_workbook(pack["path"], read_only=True, data_only=True)
    table(pack, "CO2_emission_factors_in_TYNDP2026.xlsx", wb, "2026", "NT+")
    wb.close()


def build_common(run: Run) -> None:
    """Common thermal data, 2026 only: the 2024 packages carry no such file."""
    pack = run.package("2026/inputs", "PEMMDB_CommonData.xlsx")
    member = "PEMMDB_CommonData.xlsx"
    wb = openpyxl.load_workbook(pack["path"], read_only=True, data_only=True)
    rows = head(wb.worksheets[0], 68)

    def emit(r, metric, value, unit):
        tech = f"{clean(r[2])} {clean(r[3]).replace('-', '')}".strip()
        run.add("thermal", pack, member, edition="2026", scenario="NT+", year="", node="ALL", family="thermal_common",
                technology=tech, level="total", metric=metric, value=value, unit=unit)

    names = {j: slug(rows[4][j]) for j in range(5, 17) if clean(rows[4][j])}
    units = {j: clean(rows[7][j]).replace("€", "EUR").replace("%", "share") for j in names}
    names.update({17: "transition_hot_to_warm", 18: "transition_hot_to_cold"})
    units.update({17: "hours", 18: "hours"})
    for r in rows[8:35]:
        if num(r[1]) is None:
            continue
        for j, metric in names.items():
            value = num(r[j])
            if value is not None:
                emit(r, metric, value, units[j])
    for r in rows[41:68]:
        if num(r[1]) is None:
            continue
        for j, metric, unit in COMMON_SECOND_TABLE:
            value = num(r[j])
            if value is not None:
                emit(r, metric, value, unit)
    wb.close()


def run_checks(run: Run, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for check in run.cfg["spot_checks"]:
        spec = dict(check)
        table, expected = spec.pop("table"), spec.pop("expected")
        df = frames[table]
        for key, value in spec.items():
            df = df[df[key].astype(str) == str(value)]
        if "level" in df:
            df = df[df["level"] == "total"]
        found = df["value"].sum() if len(df) else float("nan")
        passed = abs(found - expected) <= max(1e-3 * abs(expected), 1e-6)
        out.append({**check, "found": found, "rows": len(df), "passed": passed})
    return pd.DataFrame(out)


def coverage(run: Run, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = [(e, n, y) for e, c in run.cfg["editions"].items() for y in c["years"] for n in run.nodes]
    files = pd.DataFrame(run.files)
    rows = []
    for edition, node, year in keys:
        row = {"edition": edition, "node": node, "year": year}
        for family in ("pemmdb", "demand"):
            row[f"{family}_files"] = len(files[(files.edition == edition) & (files.node == node)
                                               & (files.year == year) & (files.family == family)])
        for table in ("capacity", "thermal", "demand_annual"):
            df = frames[table]
            row[f"{table}_rows"] = len(df[(df.edition == edition) & (df.node == node) & (df.year.astype(str) == str(year))])
        rows.append(row)
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, name: str, manifest: list[dict]) -> None:
    path = OUT_DIR / f"{name}.csv"
    text = df.to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode("utf-8"))
    manifest.append({"table": name, "rows": len(df), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="Folder with manifest.csv, default from the config")
    parser.add_argument("--skip-demand", action="store_true", help="Skip the slow hourly demand workbooks")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = args.root or (REPO / cfg["tyndp_root"]).resolve()
    run = Run(root, cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PEMMDB", flush=True)
    build_pemmdb(run)
    print("Lines", flush=True)
    build_ntc(run)
    print("Investment, prices, factors, common data", flush=True)
    build_investment(run)
    build_prices(run)
    build_emissions(run)
    build_common(run)
    if not args.skip_demand:
        print("Demand", flush=True)
        build_demand(run)

    order = {
        "capacity": NODE_COLS, "thermal": NODE_COLS, "exchanges": NODE_COLS,
        "demand_annual": NODE_COLS[:7] + ["weather", "weather_label"] + NODE_COLS[7:],
        "ntc": ["edition", "scenario", "year", "border", "node_from", "node_to", "project", "metric", "value", "unit"],
        "candidates": ["edition", "scenario", "year", "border", "node_from", "node_to", "capacity_forward_mw",
                       "capacity_backward_mw", "capex_meur"],
        "generator_costs": ["edition", "scenario", "year", "node", "technology", "capex_eur_per_mw",
                            "opex_eur_per_mw_year", "efficiency"],
        "fuel_co2_prices": ["edition", "scenario", "year", "technology", "role", "metric", "value", "unit"],
        "emission_factors": ["edition", "scenario", "year", "technology", "metric", "value", "unit"],
    }
    frames, manifest = {}, []
    for name, cols in order.items():
        df = pd.DataFrame(run.tables.get(name, []), columns=cols + SOURCE_COLS)
        df = df.sort_values(cols[:-2], kind="stable").reset_index(drop=True)
        df = df[~(df.value == 0)] if "value" in df and name in ("capacity", "exchanges") else df
        dup = df.duplicated(subset=[c for c in cols if c not in ("value", "weather_label")]).sum()
        if dup:
            run.issue("all", name, f"{dup} duplicated keys")
        frames[name] = df
        write(df, name, manifest)
    checks = run_checks(run, frames)
    write(coverage(run, frames), "coverage", manifest)
    write(checks, "checks", manifest)
    write(pd.DataFrame(run.issues, columns=["edition", "member", "issue"]), "issues", manifest)
    pd.DataFrame(manifest).to_csv(OUT_DIR / "tidy_manifest.csv", index=False, lineterminator="\n")

    for row in manifest:
        print(f"  {row['table']}: {row['rows']} rows")
    failed = checks[~checks.passed.astype(bool)]
    print(f"Checks: {len(checks) - len(failed)} of {len(checks)} passed, issues: {len(run.issues)}")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
