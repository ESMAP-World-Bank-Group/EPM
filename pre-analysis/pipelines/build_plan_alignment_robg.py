# -*- coding: utf-8 -*-
"""Romanian and Bulgarian supply aligned on the national plans (RoBg family, steps 15 to 20).

Romania and Bulgaria are internal only in the RoBg family (zcmap_robg.csv), so no
earlier scenario reads any generator or availability row touched here. The fuel
cap goes to a new file wired to the RoBg columns of scenarios.csv only. Run after
build_buildlimits_robg.py and build_fleet_calibration_robg.py, which it partly
overrides. Idempotent: a second run changes nothing.

Why: robg4 (2026-10-12) built 5.1 GW of legacy Romanian wind in 2027-2028 and
1.5 GW of generic Romanian hydro in 2031, no utility PV after 2030, and ran
Bulgarian coal at 14.6 TWh in 2025 against 9.8 TWh observed.

15 Romanian and Bulgarian onshore wind
--------------------------------------
* The six legacy "_corrected" rows of the 12_46 workbook and "Wind Park" (5,769 MW
  in all, no project behind them) get Status 0 and BuildLimitperYear 0. Status 0
  removes a row (input_treatment sets its Capacity to 0 and main.gms treats it
  as existing, so vCap is fixed at 0); BuildLimitperYear 0 alone on a Status 3
  row trips the EPM_STRICT_BUILD_LIMITS guard. The generic classes,
  whose limit follows the build-limit v2 rule, are the only open-ended candidates.
* Generic Romanian onshore candidates and Wind_INECP_3: capex 1.75 $m/MW and FOM
  45,000 $/MW/yr, the onshore values of the Cernavoda LTO economic analysis.
* Every onshore candidate and committed row of both zones (Status 2 or 3) gets
  the availability of the zone's existing fleet, which reproduces the observed
  2023-2024 capacity factor against the zone profile (Romania 0.6853, so about
  26 percent; Bulgaria 0.6851, so about 23 percent). This reverses the step 13e
  choice of keeping the atlas profile for candidates (39 percent in Romania,
  33 percent in Bulgaria), which no plan or observation supports.

16 Romanian generic hydro
-------------------------
Generic ST Hydro and Generic ROR Hydro get Status 0 and BuildLimitperYear 0. The NECP holds
hydro flat; the named rehabilitation rows (_ext) stay.

18 Bulgarian coal
-----------------
A ceiling on annual fuel use (pMaxFuellimit, million MMBtu; fApplyFuelConstraint
is on) caps each coal fuel at a 27 percent capacity factor on the capacity still
online that year. 27 percent is the observed 2024 lignite factor (ENTSO-E
9.33 TWh on the 3,966 MW of the model). Hard coal ran lower in 2024 (0.45 TWh),
but the Bulgarian default MinGenShareAllHours (0.25) sets a floor that a lower
cap would make infeasible, so both fuels share the 27 percent ceiling: about
10.6 TWh in 2025 against 9.8 TWh observed, from 14.6 TWh in robg4.
Availability is left as it is: a lower availability would remove firm capacity
in every hour, not only energy.

20 Nuclear dates
----------------
Cernavoda 3 and 4 in 2033 and 2034 (were 2035 and 2036; SNN EPC contract, TDP).
Kozloduy 7 and 8 in 2035 and 2040 (were 2033 and 2036; Bulgarian NECP).
Lifetimes are kept.

Not done here: step 17 (Romanian gas CHP floor) and step 19 (Bulgarian demand,
whose forecast already carries the profile load factor, 0.58 against 0.571).

Usage (from EPM/):  python pre-analysis/pipelines/build_plan_alignment_robg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from build_fleet_entsoe_bg import AVAIL_FILE, DATA, GEN_FILES, _EPM, _read, _write  # noqa: E402

SCENARIOS = DATA / "scenarios.csv"
FUEL_BASE = DATA / "constraint" / "pMaxFuellimit.csv"
FUEL_ROBG = DATA / "constraint" / "pMaxFuellimit_robg.csv"
FUEL_REL = "constraint/pMaxFuellimit_robg.csv"

LEGACY_WIND = ["WindMed1_corrected_2026", "WindMed2_corrected_2026", "WindMed1_corrected_2027",
               "WindHigh2_corrected_2027", "WindMed1_corrected_2028", "WindHigh2_corrected_2028",
               "Wind Park"]
GENERIC_HYDRO = ["Generic ST Hydro", "Generic ROR Hydro"]
WIND_COST = {"Capex": "1.75", "FOMperMW": "45000"}
WIND_COST_ROWS = ["Wind_INECP_3"]
FLEET_REF = {"Romania": "Fântânele", "Bulgaria": "Bulgaria_AGG_Wind_ENTSOE"}
NUCLEAR = {"Cernavoda-3": 2033, "Cernavoda-4": 2034, "Bulgaria_Kozloduy_7": 2035, "Bulgaria_Kozloduy_8": 2040}

COAL_CF = 0.27
COAL_ROWS = {"DomesticCoal": ["Bulgaria_Agg_Lignite", "Bulgaria_Agg_Lignite_2029", "Bulgaria_Agg_Lignite_2033"],
             "ImportedCoal": ["Bulgaria_Agg_Coal", "Bulgaria_Agg_Coal_2029", "Bulgaria_Agg_Coal_2033"]}
DEFAULT_HR = {"ImportedCoal": 8.5}


def _fail(msg: str) -> None:
    raise SystemExit(f"[build_plan_alignment_robg] FAIL: {msg}")


def _fmt(v: float) -> str:
    return f"{round(v, 2):g}"


def fix_generators(path: Path) -> list[str]:
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    log = []

    def put(p, col, new, n):
        if p[ix[col]] != new:
            log.append(f"{p[0]}: {col} {p[ix[col]] or 'empty'} -> {new}")
            p[ix[col]] = new
            lines[n] = ",".join(p)

    for n, ln in enumerate(lines[1:], start=1):
        p = ln.split(",")
        if len(p) < len(head) or p[ix["z"]] not in ("Romania", "Bulgaria"):
            continue
        g = p[0]
        if g in LEGACY_WIND or g in GENERIC_HYDRO:
            put(p, "BuildLimitperYear", "0", n)
            put(p, "Status", "0", n)
        is_generic_ro_wind = (p[ix["z"]] == "Romania" and p[ix["tech"]] == "OnshoreWind"
                              and p[ix["Status"]] == "3" and g.startswith("Generic Onshore"))
        if is_generic_ro_wind or g in WIND_COST_ROWS:
            for col, v in WIND_COST.items():
                put(p, col, v, n)
        if g in NUCLEAR:
            st, rt = int(p[ix["StYr"]]), int(p[ix["RetrYr"]])
            new = NUCLEAR[g]
            put(p, "StYr", str(new), n)
            put(p, "RetrYr", str(rt - st + new), n)
    _write(path, bom, eol, lines)
    return log


def wind_rows(path: Path) -> dict[str, str]:
    """Onshore wind rows of Status 2 or 3, by name, with their zone."""
    bom, eol, lines = _read(path)
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    out = {}
    for ln in lines[1:]:
        p = ln.split(",")
        if (len(p) >= len(head) and p[ix["z"]] in FLEET_REF and p[ix["tech"]] == "OnshoreWind"
                and p[ix["Status"]] in ("2", "3")):
            out[p[0]] = p[ix["z"]]
    return out


def fix_availability(rows: dict[str, str]) -> list[str]:
    bom, eol, lines = _read(AVAIL_FILE)
    trailing = lines[-1] == ""
    body = lines[:-1] if trailing else lines
    where = {ln.split(",")[0]: n for n, ln in enumerate(body) if ln}
    ref = {}
    for z, g in FLEET_REF.items():
        if g not in where:
            _fail(f"fleet reference {g} missing from {AVAIL_FILE.name}")
        ref[z] = body[where[g]].split(",")[1:]
    log = []
    for g, z in sorted(rows.items()):
        new = ",".join([g] + ref[z])
        if g in where:
            if body[where[g]] != new:
                log.append(f"{g}: {body[where[g]].split(',')[1]} -> {ref[z][0]}")
                body[where[g]] = new
        else:
            body.append(new)
            log.append(f"{g}: added at {ref[z][0]}")
    _write(AVAIL_FILE, bom, eol, body + ([""] if trailing else []))
    return log


def coal_cap() -> dict[str, dict[int, float]]:
    """Million MMBtu per year: COAL_CF x online capacity x 8760 x heat rate."""
    bom, eol, lines = _read(GEN_FILES[0])
    head = lines[0].split(",")
    ix = {c: i for i, c in enumerate(head)}
    rows = {ln.split(",")[0]: ln.split(",") for ln in lines[1:] if ln}
    years = [int(c) for c in _read(FUEL_BASE)[2][0].split(",")[2:] if c.strip()]
    out = {}
    for f, names in COAL_ROWS.items():
        cap = {}
        for y in years:
            mmbtu = 0.0
            for g in names:
                if g not in rows:
                    _fail(f"{g} missing from {GEN_FILES[0].name}")
                p = rows[g]
                if int(p[ix["RetrYr"]]) >= y:
                    hr = float(p[ix["HeatRate"]]) if p[ix["HeatRate"]] else DEFAULT_HR[f]
                    mmbtu += float(p[ix["Capacity"]]) * 8760 * COAL_CF * hr
            if mmbtu > 0:
                cap[y] = mmbtu / 1e6
        out[f] = cap
    return out


def write_fuel_file(cap: dict[str, dict[int, float]]) -> None:
    bom, eol, lines = _read(FUEL_BASE)
    head = lines[0].split(",")
    years = [c for c in head[2:] if c.strip()]
    body = [ln for ln in lines[1:] if ln]
    for f, cy in cap.items():
        body.append(",".join(["Bulgaria", f] + [_fmt(cy[int(y)]) if int(y) in cy else "" for y in years]))
    _write(FUEL_ROBG, bom, eol, [lines[0]] + body + [""])


def wire_scenarios() -> list[str]:
    bom, eol, lines = _read(SCENARIOS)
    head = lines[0].split(",")
    rows = {ln.split(",")[0]: n for n, ln in enumerate(lines) if ln}
    zc = lines[rows["zcmap"]].split(",")
    p = lines[rows["pMaxFuellimit"]].split(",")
    p += [""] * (len(head) - len(p))
    log = []
    for i, s in enumerate(head):
        if i and zc[i].startswith("zcmap_robg") and p[i] != FUEL_REL:
            log.append(f"{s}: pMaxFuellimit {p[i] or 'base'} -> {FUEL_REL}")
            p[i] = FUEL_REL
    lines[rows["pMaxFuellimit"]] = ",".join(p)
    _write(SCENARIOS, bom, eol, lines)
    return log


def main() -> int:
    for path in GEN_FILES:
        log = fix_generators(path)
        print(f"  {path.relative_to(_EPM)}: {len(log)} change(s)")
        if path == GEN_FILES[0]:
            for line in log:
                print(f"    {line}")
    log = fix_availability(wind_rows(GEN_FILES[0]))
    print(f"  {AVAIL_FILE.relative_to(_EPM)}: {len(log)} change(s)")
    for line in log:
        print(f"    {line}")
    cap = coal_cap()
    write_fuel_file(cap)
    print(f"  wrote {FUEL_ROBG.relative_to(_EPM)}")
    for f, cy in cap.items():
        print(f"    {f}: " + ", ".join(f"{y} {v:.1f}" for y, v in sorted(cy.items()) if y in (2024, 2025, 2029, 2030, 2033, 2034, 2038)))
    for line in wire_scenarios():
        print(f"  scenarios.csv: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
