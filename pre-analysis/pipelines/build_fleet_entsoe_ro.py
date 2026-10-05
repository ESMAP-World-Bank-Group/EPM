# -*- coding: utf-8 -*-
"""Romanian gas and storage fleet recalibrated on ENTSO-E and published CODs (RoBg family, step 5a bis).

Romania is internal only in the RoBg family (zcmap_robg.csv), so no earlier
scenario reads any row touched here. Idempotent: a second run changes nothing.

Step 5a turned every committed row with StYr before 2025 into an existing plant,
which put 5 gas projects in service in 2025 and lifted the Romanian gas fleet to
about 4.5 GW against 2.2 GW in ENTSO-E (A68, start of 2025). This step sets each
gas project to its published status:

* Cancelled or never built, removed with their availability rows: Borzesti 189 MW
  (GEM: cancelled), Isalnita OCGT 600 MW (duplicate of the Isalnita CCGT units 1
  and 2, which carry the real 850 MW project), Alum Tulcea 50 MW (the 5 x 50 MW
  station never left planning; the smelter is idle).
* Committed at the real date: Iernut 430 MW CCGT in 2027 (Romgaz completion
  deadline 31 December 2026), Petromidia cogeneration 80 MW in 2026 (commissioning
  completed December 2025), Isalnita units 1 and 2 and Turceni in 2030 (CE Oltenia
  restructuring plan, commissioning December 2029).
* Added: Mintia CCGT (Mass Global), 1,100 MW in 2027 (in commercial operation by
  end 2026) and 600 MW in 2028 (full commissioning by end 2027). Capex from the
  announced EUR 1 billion investment.

Batteries: the model had no existing Romanian storage. Aggregate at the ENTSO-E
2025 value (137 MW) and a committed 2026 row for the 2025 additions (493 minus
137). Energy at 2.0 h, the Transelectrica ratio of 1 August 2026 (989 MW,
1,975 MWh). Rows go to the four generator files and both pStorageDataInput files.

PV is left unchanged: the model's existing 2,238 MW lies between the ENTSO-E
values for the start of 2025 (1,588) and 2026 (2,788).

Writes the comparison to pre-analysis/output_prices/robg/ro_fleet_entsoe.csv.

Usage (from EPM/):  python pre-analysis/pipelines/build_fleet_entsoe_ro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import (AVAIL_FILE, CACHE, GEN_FILES, STORAGE_FILES, _EPM,  # noqa: E402
                                   _PRE_ANALYSIS, _client, _read, _row, _write)

OUT_CSV = _PRE_ANALYSIS / "output_prices" / "robg" / "ro_fleet_entsoe.csv"
ZONE = "Romania"
NO_RETIREMENT_YEAR = 2060
BATTERY_HOURS = round(1975 / 989, 1)
EUR_USD = 1.0824
MINTIA_CAPEX = round(1000 * EUR_USD / 1700, 2)  # USD m per MW
ANCHOR = "TPP Midia_Navodari_Ext"
TEMPLATE = "Iernut-Ludus - Expansion"
REMOVE = ["Borzești", "Ișalnița", "Alum Tulcea"]
RETIME = {"Iernut-Ludus - Expansion": 2027, "TPP Midia_Navodari_Ext": 2026,
          "Ișalnița - unit 1": 2030, "Ișalnița - unit 2": 2030, "TPP Turceni_gas": 2030}
MINTIA = [("Mintia CCGT phase 1", 1100, 2027), ("Mintia CCGT phase 2", 600, 2028)]
RETIRE_MINTIA = 2087


def installed(year: int) -> pd.Series:
    path = CACHE / f"instcap_RO_{year}.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        _client().query_installed_generation_capacity(
            "RO", start=pd.Timestamp(f"{year}0101", tz="Europe/Brussels"),
            end=pd.Timestamp(f"{year}1231", tz="Europe/Brussels")).to_csv(path)
    return pd.read_csv(path, index_col=0).iloc[0]


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def new_rows(head: list[str], template: list[str], c25: pd.Series, c26: pd.Series) -> list[dict]:
    rows = []
    for name, cap, styr in MINTIA:
        r = dict(zip(head, template))
        r.update(g=name, Status=2, StYr=styr, RetrYr=RETIRE_MINTIA, Capacity=cap,
                 BuildLimitperYear=cap, Capex=MINTIA_CAPEX)
        rows.append(r)
    bat25, bat26 = float(c25["Energy storage"]), float(c26["Energy storage"])
    rows.append(dict(g="Romania_AGG_Battery_ENTSOE", z=ZONE, tech="Storage", f="Battery", Status=1,
                     StYr=2024, RetrYr=NO_RETIREMENT_YEAR, Capacity=bat25))
    rows.append(dict(g="Romania_Battery_2026", z=ZONE, tech="Storage", f="Battery", Status=2,
                     StYr=2026, RetrYr=NO_RETIREMENT_YEAR, Capacity=bat26 - bat25))
    return rows


def fix_generators(path: Path, c25: pd.Series, c26: pd.Series) -> int:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    changes = 0
    kept = []
    for ln in lines:
        p = ln.split(",")
        if p[0] in REMOVE and len(p) > 1 and p[1] == ZONE:
            changes += 1
            continue
        if p[0] in RETIME and (p[ix["Status"]] != "2" or p[ix["StYr"]] != str(RETIME[p[0]])):
            p[ix["Status"]], p[ix["StYr"]] = "2", str(RETIME[p[0]])
            ln = ",".join(p)
            changes += 1
        kept.append(ln)
    names = {ln.split(",")[0]: n for n, ln in enumerate(kept)}
    template = kept[names[TEMPLATE]].split(",")
    add = [_row(head, **{k: _fmt(v) for k, v in r.items()})
           for r in new_rows(head, template, c25, c26) if r["g"] not in names]
    if add:
        at = names[ANCHOR] + 1
        kept[at:at] = add
        changes += len(add)
    _write(path, bom, eol, kept)
    return changes


def fix_storage(path: Path, c25: pd.Series, c26: pd.Series) -> int:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    names = {ln.split(",")[0] for ln in lines}
    trailing = lines[-1] == ""
    body = lines[:-1] if trailing else lines
    add = []
    for r in new_rows(head, [""] * len(head), c25, c26):
        if r["tech"] != "Storage" or r["g"] in names:
            continue
        r = dict(r, CapacityMWh=round(float(r["Capacity"]) * BATTERY_HOURS, 1))
        add.append(_row(head, **{k: _fmt(v) for k, v in r.items() if k in head}))
    _write(path, bom, eol, body + add + ([""] if trailing else []))
    return len(add)


def fix_availability() -> int:
    bom, eol, lines = _read(AVAIL_FILE)
    kept = [ln for ln in lines if ln.split(",")[0] not in REMOVE]
    changes = len(lines) - len(kept)
    names = {ln.split(",")[0]: n for n, ln in enumerate(kept)}
    tail = kept[names[TEMPLATE]].split(",")[1:]
    add = [",".join([name] + tail) for name, _, _ in MINTIA if name not in names]
    if add:
        at = names[ANCHOR] + 1
        kept[at:at] = add
        changes += len(add)
    _write(AVAIL_FILE, bom, eol, kept)
    return changes


def table(c25: pd.Series, c26: pd.Series) -> pd.DataFrame:
    g = pd.read_csv(GEN_FILES[0], encoding="utf-8-sig")
    g = g[g["z"] == ZONE]
    for c in ["Status", "StYr", "RetrYr", "Capacity"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    rows = []
    for etype, techs, fuel in [("Fossil Gas", ["CCGT", "OCGT"], "Gas"), ("Solar", ["PV"], "Solar"),
                               ("Energy storage", ["Storage"], "Battery")]:
        sel = g[g["tech"].isin(techs) & (g["f"] == fuel)]
        row = {"type": etype, "entsoe_2025": c25.get(etype, 0.0), "entsoe_2026": c26.get(etype, 0.0)}
        for y in (2025, 2026):
            live = sel[(sel["Status"].isin([1, 2])) & (sel["StYr"].fillna(0) <= y)
                       & (sel["RetrYr"].fillna(9999) > y)]
            row[f"model_{y}"] = round(live["Capacity"].sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    c25, c26 = installed(2025), installed(2026)
    for path in GEN_FILES:
        print(f"  {path.relative_to(_EPM)}: {fix_generators(path, c25, c26)} change(s)")
    for path in STORAGE_FILES:
        print(f"  {path.relative_to(_EPM)}: {fix_storage(path, c25, c26)} row(s) added")
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {fix_availability()} change(s)")
    t = table(c25, c26)
    OUT_CSV.write_bytes(t.to_csv(index=False).encode("utf-8"))
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
