"""Set up the GEC export hubs of the GECO scenarios in the base files of a deployment.

Usage:
    python pre-analysis/catalog/build_hub_zone.py --deployment data_blacksea
    python pre-analysis/catalog/build_hub_zone.py --deployment data_blacksea --dry-run

Decision of 2026-09-12 (register a07_vre_hub_2040 and g07_vre_hub_2040, main: taken): the
CESI hub capacities are the ceiling of the hub candidates in every GECO scenario and the
model decides the pace. The GECOHub variants (LC_GECOHub, CESI_GECOHub) force the CESI
phasing instead. No national fleet is imposed anywhere. No CESI figure lives in this file:
every value is read from cesi/cesi_register.yaml (DVC only, client confidential) at run
time, so this file may be tracked by git.

What it does, per hub zone (GEC_AZ for AzerbaijanMain, GEC_GE for Georgia):
  1. Default files. Rows of the national zone are copied under the hub zone in
     pGenDataInputDefault, pCapexTrajectoriesDefault, pAvailabilityDefault and pVREProfile
     when the hub has none. The hub carries our national costs, availability and profile,
     not a CESI figure.
  2. Hub candidates. One Status 3 row <hub>_<tech>_Hub per technology in pGenDataInput:
     Capacity = CESI hub total by 2040, BuildLimitperYear = Capacity, StYr = first link
     year with a non zero CESI step (offshore starts with the second link), RetrYr = StYr
     plus the life of the technology (HUB_LIFE, as the GEC_AZ rows of 2026-09-11). Rows
     already present are left untouched.
  3. Tranche file supply/pGenDataInput_hub.csv: the base file with the hub candidates
     replaced, in place, by committed tranches (Status 2) at the CESI phasing. GEC_AZ
     follows the register steps (a07 detail by_year, then the 2040 total). GEC_GE is
     built in equal steps at the link years, because the study gives the 2040 total only.
     A tranche inherits everything from its candidate but name, Status, StYr, Capacity and
     BuildLimitperYear. Read by the GECOHub scenarios (scenarios.csv).

Running the script twice changes nothing: copies and candidates are skipped when present,
and the tranche file is a pure function of the base file and the register.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cesi_full import (  # noqa: E402
    HUB, INPUT_ROOT, VRE_FUEL, ZONES, fmt, link_years, load_config, load_register,
    num, read_csv, with_hub_copies, write_csv,
)

HUB_FILE = "pGenDataInput_hub.csv"
# Life of a hub candidate, years from StYr to RetrYr, as the GEC_AZ rows written on 2026-09-11.
HUB_LIFE = {"PV": 25, "OnshoreWind": 30, "OffshoreWind": 30}
# Zone column of each default file that must carry the hub.
DEFAULT_FILES = {"pGenDataInputDefault": "z", "pCapexTrajectoriesDefault": "zone",
                 "pAvailabilityDefault": "z", "pVREProfile": "zone"}


def hub_steps(reg: dict, prefix: str, years: list[int]) -> dict[int, dict[str, float]]:
    """Cumulative hub capacity in GW per tech at each link year, from the register."""
    e = reg.get(f"{prefix}07_vre_hub_2040")
    if e is None:
        sys.exit(f"register entry {prefix}07_vre_hub_2040 is missing")
    total = {t: float(v) for t, v in e["cesi_value"].items()}
    by_year = (e.get("detail") or {}).get("by_year")
    if by_year:
        steps = {int(y): {t: float(v) for t, v in d.items()} for y, d in by_year.items()}
        steps[max(years)] = total
    else:
        steps = {y: {t: v * (i + 1) / len(years) for t, v in total.items()} for i, y in enumerate(years)}
    return steps


def candidate(hub: str, tech: str, steps: dict[int, dict[str, float]], header: list[str]) -> dict:
    """Status 3 hub candidate: ceiling = 2040 total, from the first year with a non zero step."""
    last = max(steps)
    start = min(y for y in steps if steps[y].get(tech, 0.0) > 0.0)
    mw = steps[last][tech] * 1000.0
    r = {k: "" for k in header}
    r.update({"g": f"{hub}_{tech}_Hub", "z": hub, "tech": tech, "f": VRE_FUEL[tech], "Status": "3",
              "StYr": str(start), "RetrYr": str(start + HUB_LIFE[tech]),
              "Capacity": fmt(mw, 1), "BuildLimitperYear": fmt(mw, 1)})
    return r


def tranches(cand: dict, steps: dict[int, dict[str, float]]) -> list[dict]:
    """Committed tranches of one candidate at the phasing steps, cumulative capacity kept."""
    out, prev = [], 0.0
    for y in sorted(steps):
        gap = steps[y].get(cand["tech"], 0.0) * 1000.0 - prev
        if gap > 0.5:
            out.append(dict(cand, g=f"{cand['g']}_{y}", Status="2", StYr=str(y),
                            Capacity=fmt(gap, 1), BuildLimitperYear=fmt(gap, 1)))
            prev += float(fmt(gap, 1))      # rounding lands on the last tranche
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment", default="data_blacksea")
    ap.add_argument("--dry-run", action="store_true", help="log everything, write nothing")
    args = ap.parse_args()
    dep = INPUT_ROOT / args.deployment
    reg = load_register(dep)
    cfg = load_config(dep)
    years = link_years(reg)
    print(f"build_hub_zone: {dep.name}, hubs {', '.join(f'{h} for {z}' for z, h in HUB.items())}, link years {years}")

    # 1. default files: national rows copied under the hub when absent
    print("\n== default files")
    for param, col in DEFAULT_FILES.items():
        header, rows, style = read_csv(cfg[param])
        before = {r[col] for r in rows}
        rows = with_hub_copies(rows, col)
        added = [h for h in HUB.values() if h not in before and any(r[col] == h for r in rows)]
        if added:
            print(f"  {param}: {', '.join(added)} copied from the national zone")
            write_csv(cfg[param], header, rows, style, args.dry_run)
        else:
            print(f"  {param}: every hub present, unchanged")

    # 2. hub candidates in the base fleet file
    print("\n== hub candidates (Status 3, ceiling = study total by the last link year)")
    header, rows, style = read_csv(cfg["pGenDataInput"])
    names = {r["g"] for r in rows}
    steps_of = {HUB[z]: hub_steps(reg, p, years) for z, p in ZONES.items()}
    new = []
    for hub, steps in steps_of.items():
        for tech in steps[max(steps)]:
            c = candidate(hub, tech, steps, header)
            if c["g"] in names:
                print(f"  {c['g']}: present, unchanged")
            else:
                new.append(c)
                print(f"  {c['g']}: added, {c['Capacity']} MW from {c['StYr']} to {c['RetrYr']}")
    if new:
        rows += new
        write_csv(cfg["pGenDataInput"], header, rows, style, args.dry_run)

    # 3. tranche file for the GECOHub scenarios
    print(f"\n== {HUB_FILE} (hub candidates as committed tranches at the study phasing)")
    out_rows = []
    for r in rows:
        if r["z"] in steps_of and r["Status"] == "3" and r["g"].endswith("_Hub"):
            ts = tranches(r, steps_of[r["z"]])
            out_rows += ts
            print(f"  {r['g']}: " + ", ".join(f"{t['StYr']} +{t['Capacity']} MW" for t in ts))
        else:
            out_rows.append(r)
    total = sum(num(r["Capacity"], 0.0) for r in out_rows if r["z"] in steps_of)
    base = sum(num(r["Capacity"], 0.0) for r in rows if r["z"] in steps_of)
    if abs(total - base) > 0.5:
        sys.exit(f"tranches sum to {total:.1f} MW, candidates to {base:.1f} MW")
    write_csv(cfg["pGenDataInput"].parent / HUB_FILE, header, out_rows, style, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
