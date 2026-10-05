# -*- coding: utf-8 -*-
"""Bulgarian fleet recalibrated on ENTSO-E installed capacity (RoBg family, step 5b).

Bulgaria is internal only in the RoBg family (zcmap_robg.csv), so no earlier
scenario reads any row touched here. Idempotent: a second run changes nothing.

Anchor: ENTSO-E Transparency Platform, installed capacity per production type
(A68), values for 2025 and 2026 (capacity at the start of each year). The
model's existing fleet (Status 1) is compared with the 2025 value per type and
the gap is closed with one aggregate row per type, Status 1, StYr 2024,
RetrYr 2060 (no-unplanned-retirement rule). What was added during 2025 (the
2026 value minus the 2025 value, net of rows already committed for 2025) enters
as a committed row, Status 2, StYr 2026.

* PV and onshore wind: aggregates as above.
* Run-of-river: the model had none. One aggregate at the ENTSO-E capacity; its
  quarterly availability is the ENTSO-E 2019-2025 mean capacity factor of
  "Hydro Run-of-river and poundage".
* Reservoir hydro: aggregate for the gap, with the uniform availability profile
  of the existing Bulgarian reservoir plants.
* Batteries: Bulgaria_Battery raised to the 2025 value, the 2025 additions as a
  committed 2026 row. Energy = power x 2.6 h, the ratio of the ENTSO-E May 2026
  figures (3.32 GW, 8.6 GWh) reported by MR East (August 2026). The rows were
  absent from pStorageDataInput, so they had no MWh and stored nothing.
* Chaira pumped storage: offline since 2022 (ENTSO-E pumped generation 0.01
  TWh in 2022-2024). Unit 2 back in late 2024, unit 3 repaired in 2026, unit 1
  expected by 2028 (Bulgarian Minister of Energy, Balkan Green Energy News,
  16 April 2026), unit 4 in procurement, 2030 assumed. One row per unit, 216 MW
  each, units 3, 1 and 4 committed with zero capex (repair, not a new build).
  Storage 8.5 h at full turbine output (NEK, Yadenitsa project page).

Writes the comparison to pre-analysis/output_prices/robg/bg_fleet_entsoe.csv.
ENTSO-E data are cached in Data/cache_entso_e/fleet/ (outside git); a missing
file is fetched with entsoe-py, the token read from config/api_tokens.ini.

Usage (from EPM/):  python pre-analysis/pipelines/build_fleet_entsoe_bg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_PRE_ANALYSIS = _HERE.parents[1]
_EPM = _PRE_ANALYSIS.parent
sys.path.insert(0, str(_HERE.parent))

DATA = _EPM / "epm" / "input" / "data_blacksea"
GEN_FILES = [
    DATA / "supply" / "pGenDataInput.csv",
    DATA / "supply" / "pGenDataInput_hub.csv",
    DATA / "supply" / "pGenDataInput_armnuke.csv",
    DATA / "cesi" / "unwired" / "pGenDataInput_cesi.csv",
]
STORAGE_FILES = [DATA / "supply" / "pStorageDataInput.csv",
                 DATA / "cesi" / "unwired" / "pStorageDataInput_cesi.csv"]
AVAIL_FILE = DATA / "supply" / "pAvailabilityCustom.csv"
CACHE = _PRE_ANALYSIS.parents[1] / "Data" / "cache_entso_e" / "fleet"
OUT_CSV = _PRE_ANALYSIS / "output_prices" / "robg" / "bg_fleet_entsoe.csv"

ZONE = "Bulgaria"
NO_RETIREMENT_YEAR = 2060
BATTERY_HOURS = round(8.6 / 3.32, 1)
CHAIRA_HOURS = 8.5
CHAIRA_UNITS = [("Bulgaria_Chaira_Storage", 1, None), ("Bulgaria_Chaira_U3", 2, 2026),
                ("Bulgaria_Chaira_U1", 2, 2028), ("Bulgaria_Chaira_U4", 2, 2030)]
CHAIRA_UNIT_MW = 216
RESERVOIR_TEMPLATE = "Bulgaria_Belmeken_Hydro"
ROR_TYPE = "Hydro Run-of-river and poundage"
# (ENTSO-E type, tech, fuel, aggregate name, committed 2026 name)
MATCH = [
    ("Solar", "PV", "Solar", "Bulgaria_AGG_PV_ENTSOE", "Bulgaria_AGG_PV_2026"),
    ("Wind Onshore", "OnshoreWind", "Wind", "Bulgaria_AGG_Wind_ENTSOE", None),
    (ROR_TYPE, "ROR", "Water", "Bulgaria_AGG_ROR_ENTSOE", None),
    ("Hydro Water Reservoir", "ReservoirHydro", "Water", "Bulgaria_AGG_Reservoir_ENTSOE", None),
]


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_fleet_entsoe_bg] FAIL: {msg}")


def _client():
    from entsoe import EntsoePandasClient
    from entsoe_pipeline import load_api_token
    return EntsoePandasClient(api_key=load_api_token("entsoe"))


def installed(year: int) -> pd.Series:
    path = CACHE / f"instcap_BG_{year}.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        cap = _client().query_installed_generation_capacity(
            "BG", start=pd.Timestamp(f"{year}0101", tz="Europe/Brussels"),
            end=pd.Timestamp(f"{year}1231", tz="Europe/Brussels"))
        cap.to_csv(path)
    return pd.read_csv(path, index_col=0).iloc[0]


def ror_availability(cap_mw: float) -> list[float]:
    path = CACHE / "gen_BG_quarterly_2019_2025.csv"
    if not path.exists():
        client, rows = _client(), []
        for yr in range(2019, 2026):
            gen = client.query_generation("BG", start=pd.Timestamp(f"{yr}0101", tz="Europe/Sofia"),
                                          end=pd.Timestamp(f"{yr + 1}0101", tz="Europe/Sofia"))
            if isinstance(gen.columns, pd.MultiIndex):
                gen = gen.xs("Actual Aggregated", axis=1, level=1)
            s = gen[ROR_TYPE].resample("h").mean()
            for q, grp in s.groupby(s.index.quarter):
                rows.append({"year": yr, "type": ROR_TYPE, "q": q, "twh": grp.sum() / 1e6, "hours": grp.count()})
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    q = pd.read_csv(path)
    q = q[q["type"] == ROR_TYPE]
    cf = q["twh"] * 1e6 / (cap_mw * q["hours"])
    return [round(v, 4) for v in cf.groupby(q["q"]).mean().sort_index()]


def _read(path: Path) -> tuple[bool, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    return raw.startswith(b"\xef\xbb\xbf"), eol, text.split(eol)


def _write(path: Path, bom: bool, eol: str, lines: list[str]) -> None:
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + eol.join(lines).encode("utf-8"))


def _row(head: list[str], **kv) -> str:
    return ",".join(str(kv.get(c, "")) for c in head)


def plan(c25: pd.Series, c26: pd.Series) -> tuple[list[dict], dict, pd.DataFrame]:
    """New generator rows, capacity overrides and the comparison table, from the base file."""
    g = pd.read_csv(GEN_FILES[0], encoding="utf-8-sig")
    g = g[g["z"] == ZONE]
    for c in ["Status", "StYr", "RetrYr", "Capacity"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    ours = {n for _, _, _, n, m in MATCH} | {m for *_, m in MATCH if m} | {u for u, _, _ in CHAIRA_UNITS} | {"Bulgaria_Battery_2026"}
    base = g[~g["g"].isin(ours)]
    live = base[(base["Status"] == 1) & (base["RetrYr"].fillna(9999) > 2025)]
    committed25 = base[(base["Status"] == 2) & (base["StYr"] == 2025)]
    new, table = [], []
    for etype, tech, fuel, agg, add26 in MATCH:
        model = live[(live["tech"] == tech) & (live["f"] == fuel)]["Capacity"].sum()
        gap = round(c25[etype] - model, 1)
        com = committed25[(committed25["tech"] == tech)]["Capacity"].sum()
        step26 = round(c26[etype] - c25[etype] - com, 1) if add26 else 0.0
        table.append({"type": etype, "entsoe_2025": c25[etype], "entsoe_2026": c26[etype],
                      "model_existing": model, "aggregate_added": gap, "committed_2025": com,
                      "committed_2026_added": step26})
        if gap < 0:
            _fail(f"{etype}: model above ENTSO-E ({model} > {c25[etype]})")
        new.append(dict(g=agg, z=ZONE, tech=tech, f=fuel, Status=1, StYr=2024,
                        RetrYr=NO_RETIREMENT_YEAR, Capacity=gap))
        if add26 and step26 > 0:
            new.append(dict(g=add26, z=ZONE, tech=tech, f=fuel, Status=2, StYr=2026,
                            RetrYr=NO_RETIREMENT_YEAR, Capacity=step26))
    bat25, bat26 = c25["Energy storage"], c26["Energy storage"]
    table.append({"type": "Energy storage", "entsoe_2025": bat25, "entsoe_2026": bat26,
                  "model_existing": live[(live["tech"] == "Storage") & (live["f"] == "Battery")]["Capacity"].sum(),
                  "aggregate_added": 0, "committed_2025": 0, "committed_2026_added": bat26 - bat25})
    table.append({"type": "Hydro Pumped Storage", "entsoe_2025": c25["Hydro Pumped Storage"],
                  "entsoe_2026": c26["Hydro Pumped Storage"], "model_existing": 864,
                  "aggregate_added": 0, "committed_2025": 0, "committed_2026_added": 0})
    new.append(dict(g="Bulgaria_Battery_2026", z=ZONE, tech="Storage", f="Battery", Status=2, StYr=2026,
                    RetrYr=NO_RETIREMENT_YEAR, Capacity=bat26 - bat25))
    for name, status, styr in CHAIRA_UNITS[1:]:
        new.append(dict(g=name, z=ZONE, tech="Storage", f="Water", Status=status, StYr=styr,
                        RetrYr=NO_RETIREMENT_YEAR, Capacity=CHAIRA_UNIT_MW, Capex=0))
    override = {"Bulgaria_Battery": bat25, "Bulgaria_Chaira_Storage": CHAIRA_UNIT_MW}
    return new, override, pd.DataFrame(table)


def fix_generators(path: Path, new: list[dict], override: dict) -> int:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    names = {ln.split(",")[0]: n for n, ln in enumerate(lines)}
    changes = 0
    for g, cap in override.items():
        p = lines[names[g]].split(",")
        if p[ix["Capacity"]] != f"{cap:g}":
            p[ix["Capacity"]] = f"{cap:g}"
            lines[names[g]] = ",".join(p)
            changes += 1
    add = [_row(head, **{k: (f"{v:g}" if isinstance(v, float) else v) for k, v in r.items()})
           for r in new if r["g"] not in names]
    if add:
        at = names["Bulgaria_Battery"] + 1
        lines[at:at] = add
        changes += len(add)
    _write(path, bom, eol, lines)
    return changes


def fix_storage(path: Path, new: list[dict], override: dict, gen_head: list[str]) -> int:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    names = {ln.split(",")[0] for ln in lines}
    trailing = lines[-1] == ""
    body = lines[:-1] if trailing else lines
    base = pd.read_csv(GEN_FILES[0], encoding="utf-8-sig").set_index("g")
    units = [dict(base.loc[g].dropna().to_dict(), g=g, Capacity=cap) for g, cap in override.items()]
    units += [r for r in new if r["tech"] == "Storage"]
    add = []
    for u in units:
        if u["g"] in names:
            continue
        hours = CHAIRA_HOURS if u["f"] == "Water" else BATTERY_HOURS
        u = dict(u, CapacityMWh=round(float(u["Capacity"]) * hours, 1))
        u = {k: (f"{v:g}" if isinstance(v, float) else v) for k, v in u.items() if k in head}
        add.append(_row(head, **u))
    _write(path, bom, eol, body + add + ([""] if trailing else []))
    return len(add)


def fix_availability(ror: list[float]) -> int:
    bom, eol, lines = _read(AVAIL_FILE)
    names = {ln.split(",")[0]: ln for ln in lines}
    trailing = lines[-1] == ""
    body = lines[:-1] if trailing else lines
    add = []
    if "Bulgaria_AGG_ROR_ENTSOE" not in names:
        add.append(",".join(["Bulgaria_AGG_ROR_ENTSOE"] + [f"{v:g}" for v in ror]))
    if "Bulgaria_AGG_Reservoir_ENTSOE" not in names:
        add.append(",".join(["Bulgaria_AGG_Reservoir_ENTSOE"] + names[RESERVOIR_TEMPLATE].split(",")[1:]))
    _write(AVAIL_FILE, bom, eol, body + add + ([""] if trailing else []))
    return len(add)


def main() -> int:
    c25, c26 = installed(2025), installed(2026)
    new, override, table = plan(c25, c26)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_bytes(table.to_csv(index=False).encode("utf-8"))
    print(table.to_string(index=False))
    head = _read(GEN_FILES[0])[2][0].split(",")
    for path in GEN_FILES:
        print(f"  {path.relative_to(_EPM)}: {fix_generators(path, new, override)} change(s)")
    for path in STORAGE_FILES:
        print(f"  {path.relative_to(_EPM)}: {fix_storage(path, new, override, head)} row(s) added")
    ror = ror_availability(c25[ROR_TYPE])
    print(f"  ROR availability Q1-Q4 {ror}")
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {fix_availability(ror)} row(s) added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
