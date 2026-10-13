# -*- coding: utf-8 -*-
"""Romanian and Bulgarian fleet calibrated for the 2025 base year (RoBg family, step 13b to 13e).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so no
earlier scenario reads any row touched here. Rows go to the four generator files
and to pAvailabilityCustom.csv. Idempotent: a second run changes nothing.

13b Romanian coal, unit by unit
-------------------------------
ENTSO-E installed capacity per production unit (A71) at the start of 2025 and 2026.
For each plant the model row carries the 2026 value and a "_2025" tranche carries
the difference, retired at the end of 2025 (RetrYr is the last year in service,
main.gms:832). A plant absent in 2026 keeps its 2025 value and retires in 2025.
Plants of the model absent from the ENTSO-E list in 2024 (Iasi II, Pitesti Sud,
Arad) retire in 2024. Later retirement years are left as they are (exit by 2033).

13c Bulgarian coal, retirement path
-----------------------------------
The 2025 capacity stays: the aggregates (3,966 MW lignite, 669 MW imported coal)
match the ENTSO-E unit list (4,193 MW lignite, 542 MW hard coal). What was wrong is
the path: every unit ran to 2039. Each aggregate is split into three tranches at the
shares of the ESO grid development plan 2025-2034, Table 3.3 (2,902 MW in 2025,
1,627 MW in 2030, 450 MW in 2034): 44 % retires at the end of 2029, 41 % at the end
of 2033 and the last 16 % at the end of 2038, the coal exit year of the Bulgarian
recovery plan. ESO counts net available capacity, hence the use of shares and not
of its megawatts.

13d Romanian solar
------------------
* Rooftop: 3,900 MW of prosumer PV registered by mid-2025 (ANRE), absent from
  ENTSO-E and from the model. One existing row.
* Utility scale: ENTSO-E has 2,788 MW at the start of 2026 against 2,238 MW in
  the model; the difference is a committed 2026 row.

13e Wind
--------
* The six legacy WindMed and WindHigh2 "corrected" rows of the 12_46 workbook
  (5,140 MW committed in 2026 to 2028, none of it built: ENTSO-E wind is 3,027 MW
  at the start of 2025 and 3,068 MW at the start of 2026) become candidates at the
  generic onshore capex.
* Existing wind rows of both countries get an availability that reproduces the
  observed 2023-2024 capacity factor (ENTSO-E generation over mean installed
  capacity) against the zone profile. Candidates keep the atlas profile: new
  turbines are taller and sit on the best sites.
* No wind candidate comes online before 2027 (Wind Park, Generic_Wind_Bulgaria and
  the 2026 legacy rows started in 2025 or 2026): installed wind is flat into 2026 at
  ENTSO-E, and the test run built 970 MW in the 2025 base year.

Writes the comparison to pre-analysis/output_prices/robg/fleet_calibration.csv.

Usage (from EPM/):  python pre-analysis/pipelines/build_fleet_calibration_robg.py
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import AVAIL_FILE, CACHE, DATA, GEN_FILES, _EPM, _PRE_ANALYSIS, _client  # noqa: E402

OUT_CSV = _PRE_ANALYSIS / "output_prices" / "robg" / "fleet_calibration.csv"
BAK = f".bak_robg13_{date.today():%Y%m%d}"
NO_RETIREMENT_YEAR = 2060

# ENTSO-E unit name -> model rows, first row takes the 2026 value
RO_COAL = {
    "TPP Paroseni": ["TPP Paroseni"],
    "TPP Govora": ["TPP Govora"],
    "TPP Oradea": ["TPP Oradea"],
    "TPP Rovinari": ["TPP Rovinari"],
    "TPP Turceni": ["TPP Turceni unit 4 and 5"],
    "TPP Craiova": ["TPP Craiova unit 1", "TPP Craiova unit 2"],
    "TPP Isalnita": ["TPP Isalnita"],
}
RO_COAL_GONE = ["Iasi II", "Pitești Sud", "Arad"]

BG_COAL = ["Bulgaria_Agg_Lignite", "Bulgaria_Agg_Coal"]
ESO_COAL = {2025: 2902, 2030: 1627, 2034: 450}
BG_TRANCHE_RETIRE = [("_2029", 2029), ("_2033", 2033), ("", 2038)]

RO_ROOFTOP = ("Romania_Rooftop_PV", 3900)
RO_PV_2026 = "Romania_PV_2026"
PV_TEMPLATE = "R-Solar"

LEGACY_WIND = ["WindMed1_corrected_2026", "WindMed2_corrected_2026", "WindMed1_corrected_2027",
               "WindHigh2_corrected_2027", "WindMed1_corrected_2028", "WindHigh2_corrected_2028"]
GENERIC_WIND = "Generic Onshore WindMed1"
FIRST_WIND_BUILD = 2027
CF_YEARS = (2023, 2024)


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_fleet_calibration_robg] FAIL: {msg}")


def _read(path: Path) -> tuple[bytes, str, list[str]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text else "\n"
    return (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""), eol, text.split(eol)


def _write(path: Path, bom: bytes, eol: str, lines: list[str]) -> bool:
    new = bom + eol.join(lines).encode("utf-8")
    if path.read_bytes() == new:
        return False
    bak = path.with_name(path.name + BAK)
    if not bak.exists():
        shutil.copyfile(path, bak)
    path.write_bytes(new)
    return True


def _before(path: Path) -> Path:
    """The file as it stood before the first run when the backup is still there (for the comparison table).
    The plan gives the same result on the edited file."""
    baks = sorted(path.parent.glob(path.name + ".bak_robg13_*"))
    return baks[0] if baks else path


def _fmt(v) -> str:
    return f"{round(v, 1):g}" if isinstance(v, float) else str(v)


# ── ENTSO-E inputs ──────────────────────────────────────────────────────────
def units(cc: str, year: int) -> pd.Series:
    df = pd.read_csv(CACHE / f"instcap_units_{cc}_{year}.csv")
    return df.groupby("Name")["Installed Capacity [MW]"].sum()


def installed(cc: str, year: int) -> pd.Series:
    path = CACHE / f"instcap_{cc}_{year}.csv"
    if not path.exists():
        _client().query_installed_generation_capacity(
            cc, start=pd.Timestamp(f"{year}0101", tz="Europe/Brussels"),
            end=pd.Timestamp(f"{year}1231", tz="Europe/Brussels")).to_csv(path)
    return pd.read_csv(path, index_col=0).iloc[0]


def observed_cf(cc: str, etype: str) -> float:
    gen = pd.read_csv(CACHE / f"gen_{cc}_annual_2023_2024.csv")
    cfs = []
    for y in CF_YEARS:
        twh = gen[(gen["year"] == y) & (gen["type"] == etype)]["twh"].sum()
        hours = gen[(gen["year"] == y) & (gen["type"] == etype)]["hours"].max()
        mw = (installed(cc, y)[etype] + installed(cc, y + 1)[etype]) / 2
        cfs.append(twh * 1e6 / (mw * hours))
    return sum(cfs) / len(cfs)


def profile_cf(zone: str, tech: str) -> float:
    v = pd.read_csv(DATA / "supply" / "pVREProfile.csv")
    h = pd.read_csv(DATA / "pHours.csv")
    hm = h.melt(id_vars=["q", "d"], var_name="t", value_name="w")
    s = v[(v["zone"] == zone) & (v["tech"] == tech)].melt(
        id_vars=["zone", "tech", "season", "daytype"], var_name="t")
    m = s.merge(hm, left_on=["season", "daytype", "t"], right_on=["q", "d", "t"])
    if m.empty:
        _fail(f"no {tech} profile for {zone}")
    return float((m["value"] * m["w"]).sum() / m["w"].sum())


# ── Plan ────────────────────────────────────────────────────────────────────
def plan() -> tuple[dict, list[dict], dict, pd.DataFrame]:
    """Field overrides per row, new rows (copied from a parent), availability per row."""
    g = pd.read_csv(_before(GEN_FILES[0]), encoding="utf-8-sig").set_index("g")
    over, new, avail, table = {}, [], {}, []

    u24, u25, u26 = units("RO", 2024), units("RO", 2025), units("RO", 2026)
    for unit, rows in RO_COAL.items():
        c25, c26 = float(u25.get(unit, 0)), float(u26.get(unit, 0))
        head = rows[0]
        if c26 > 0:
            over[head] = {"Capacity": c26}
            rest = c25 - c26
            if len(rows) > 1:
                over[rows[1]] = {"Capacity": rest, "RetrYr": 2025}
            elif rest > 0:
                new.append({"parent": head, "g": f"{head}_2025", "Capacity": rest, "RetrYr": 2025})
        else:
            over[head] = {"Capacity": c25, "RetrYr": 2025}
        table.append({"item": f"RO coal {unit}", "entsoe_2025": c25, "entsoe_2026": c26,
                      "model_before": float(g.loc[rows, "Capacity"].sum())})
    for row in RO_COAL_GONE:
        if row.replace("ș", "s") in " ".join(u24.index):
            _fail(f"{row} is in the ENTSO-E 2024 unit list")
        over[row] = {"RetrYr": 2024}
        table.append({"item": f"RO coal {row}", "entsoe_2025": 0.0, "entsoe_2026": 0.0,
                      "model_before": float(g.loc[row, "Capacity"])})

    shares = [1 - ESO_COAL[2030] / ESO_COAL[2025], (ESO_COAL[2030] - ESO_COAL[2034]) / ESO_COAL[2025],
              ESO_COAL[2034] / ESO_COAL[2025]]
    for row in BG_COAL:
        # the aggregate plus any tranche of an earlier run, so the split never compounds
        cap = float(g.loc[[r for r in [row] + [row + x for x, _ in BG_TRANCHE_RETIRE if x] if r in g.index], "Capacity"].sum())
        parts = [round(cap * s, 1) for s in shares]
        parts[-1] = round(cap - sum(parts[:-1]), 1)
        for (suffix, retr), mw in zip(BG_TRANCHE_RETIRE, parts):
            if suffix:
                new.append({"parent": row, "g": row + suffix, "Capacity": mw, "RetrYr": retr})
            else:
                over[row] = {"Capacity": mw, "RetrYr": retr}
        table.append({"item": f"BG coal {row}", "entsoe_2025": cap, "entsoe_2026": cap,
                      "model_before": cap, "model_2030": parts[1] + parts[2], "model_2034": parts[2]})

    ro = g[g["z"] == "Romania"]
    pv = ro[(ro["tech"] == "PV") & (ro["Status"] == 1) & ~ro.index.isin([RO_ROOFTOP[0], RO_PV_2026])]
    pv_live = float(pv["Capacity"].sum())
    step26 = round(float(installed("RO", 2026)["Solar"]) - pv_live, 1)
    new.append({"parent": PV_TEMPLATE, "g": RO_ROOFTOP[0], "Capacity": float(RO_ROOFTOP[1]),
                "Status": 1, "StYr": 2024, "RetrYr": NO_RETIREMENT_YEAR})
    new.append({"parent": PV_TEMPLATE, "g": RO_PV_2026, "Capacity": step26,
                "Status": 2, "StYr": 2026, "RetrYr": NO_RETIREMENT_YEAR})
    table.append({"item": "RO utility PV", "entsoe_2025": float(installed("RO", 2025)["Solar"]),
                  "entsoe_2026": float(installed("RO", 2026)["Solar"]), "model_before": pv_live})

    capex = float(g.loc[GENERIC_WIND, "Capex"])
    for row in LEGACY_WIND:
        over[row] = {"Status": 3, "Capex": capex}
    table.append({"item": "RO legacy wind to candidates", "model_before": float(g.loc[LEGACY_WIND, "Capacity"].sum())})

    # ENTSO-E wind is flat into 2026 in both countries: no wind candidate comes online before 2027
    early = g[g["z"].isin(["Romania", "Bulgaria"]) & (g["tech"] == "OnshoreWind")
              & ((g["Status"] == 3) | g.index.isin(LEGACY_WIND)) & (g["StYr"] < FIRST_WIND_BUILD)]
    for row in early.index:
        over.setdefault(row, {})["StYr"] = FIRST_WIND_BUILD
    table.append({"item": f"Wind candidates moved to {FIRST_WIND_BUILD}", "model_before": float(early["Capacity"].clip(
        upper=early["BuildLimitperYear"].fillna(early["Capacity"])).sum())})

    for zone, cc in (("Romania", "RO"), ("Bulgaria", "BG")):
        obs, prof = observed_cf(cc, "Wind Onshore"), profile_cf(zone, "OnshoreWind")
        k = round(obs / prof, 4)
        if not 0.3 < k <= 1.0:
            _fail(f"{zone} wind availability {k} out of range")
        rows = g[(g["z"] == zone) & (g["tech"] == "OnshoreWind") & (g["Status"] == 1)].index
        for row in rows:
            avail[row] = k
        table.append({"item": f"{zone} existing wind CF", "entsoe_2025": round(obs, 4),
                      "model_before": round(prof, 4), "model_2030": k})
        pv_obs, pv_prof = observed_cf(cc, "Solar"), profile_cf(zone, "PV")
        table.append({"item": f"{zone} solar CF (check only)", "entsoe_2025": round(pv_obs, 4),
                      "model_before": round(pv_prof, 4)})
    return over, new, avail, pd.DataFrame(table)


# ── Writers ─────────────────────────────────────────────────────────────────
def fix_generators(path: Path, over: dict, new: list[dict]) -> int:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    names = {ln.split(",")[0]: n for n, ln in enumerate(lines)}
    for row in over:
        if row not in names:
            _fail(f"{path.name}: no row {row}")
    out, changes = [], 0
    by_parent: dict[str, list[dict]] = {}
    for r in new:
        by_parent.setdefault(r["parent"], []).append(r)
    for ln in lines:
        p = ln.split(",")
        name = p[0]
        if name in over:
            q = list(p)
            for k, v in over[name].items():
                q[ix[k]] = _fmt(v)
            if q != p:
                changes += 1
            ln = ",".join(q)
        out.append(ln)
        for r in by_parent.get(name, []):
            if r["g"] in names:
                continue
            q = ln.split(",")
            for k, v in r.items():
                if k != "parent":
                    q[ix[k]] = _fmt(v)
            if "BuildLimitperYear" in ix and q[ix["BuildLimitperYear"]]:
                q[ix["BuildLimitperYear"]] = _fmt(r["Capacity"])
            out.append(",".join(q))
            changes += 1
    # rows already present from an earlier run: keep their fields in step
    for r in new:
        if r["g"] in names:
            n = next(i for i, ln in enumerate(out) if ln.split(",")[0] == r["g"])
            q = out[n].split(",")
            for k, v in r.items():
                if k != "parent":
                    q[ix[k]] = _fmt(v)
            if out[n] != ",".join(q):
                out[n] = ",".join(q)
                changes += 1
    _write(path, bom, eol, out)
    return changes


def fix_availability(new: list[dict], avail: dict) -> int:
    bom, eol, lines = _read(AVAIL_FILE)
    names = {ln.split(",")[0]: n for n, ln in enumerate(lines)}
    width = len(lines[0].split(",")) - 1
    changes = 0
    for row, k in avail.items():
        val = ",".join([row] + [f"{k:g}"] * width)
        if row in names:
            if lines[names[row]] != val:
                lines[names[row]] = val
                changes += 1
        else:
            lines.insert(len(lines) - (1 if lines[-1] == "" else 0), val)
            names = {ln.split(",")[0]: n for n, ln in enumerate(lines)}
            changes += 1
    for r in new:
        if r["g"] in names or r["parent"] not in names:
            continue
        at = names[r["parent"]] + 1
        lines.insert(at, ",".join([r["g"]] + lines[names[r["parent"]]].split(",")[1:]))
        names = {ln.split(",")[0]: n for n, ln in enumerate(lines)}
        changes += 1
    _write(AVAIL_FILE, bom, eol, lines)
    return changes


def capacity_check() -> pd.DataFrame:
    g = pd.read_csv(GEN_FILES[0], encoding="utf-8-sig")
    rows = []
    for zone in ("Romania", "Bulgaria"):
        z = g[g["z"] == zone]
        for y in (2025, 2026, 2030, 2034, 2035, 2040):
            live = z[((z["Status"] == 1) & (z["StYr"].fillna(0) < y) | (z["Status"] == 2) & (z["StYr"] <= y))
                     & (z["RetrYr"] >= y)]
            for (tech, f), mw in live.groupby(["tech", "f"])["Capacity"].sum().items():
                if tech in ("ST", "PV", "OnshoreWind"):
                    rows.append({"zone": zone, "year": y, "tech": tech, "fuel": f, "mw": round(mw)})
    return pd.DataFrame(rows).pivot_table(index=["zone", "tech", "fuel"], columns="year", values="mw")


def main() -> int:
    over, new, avail, table = plan()
    for path in GEN_FILES:
        print(f"  {path.relative_to(_EPM)}: {fix_generators(path, over, new)} change(s)")
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {fix_availability(new, avail)} change(s)")
    OUT_CSV.write_bytes(table.to_csv(index=False).encode("utf-8"))
    print(table.to_string(index=False))
    print(capacity_check().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
