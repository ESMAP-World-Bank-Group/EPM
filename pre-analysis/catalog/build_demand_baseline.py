"""Write the Baseline demand of AzerbaijanMain from the CESI Task 3 forecast held in the register.

Usage:
    python pre-analysis/catalog/build_demand_baseline.py --deployment data_blacksea
    python pre-analysis/catalog/build_demand_baseline.py --deployment data_blacksea --dry-run

Decision of 2026-09-12 (register a01_dem_energy, a02_dem_peak, main: taken): the Baseline
family (LC_BSSC, LC_GECO, LC_GECOHub) takes the Task 3 Reference forecast plus its electric
vehicle demand for Azerbaijan, gross of losses, without the hydrogen electrolysers that only
the fully aligned family carries (cesi/pDemandForecast_cesi.csv, build_cesi_full.py).
Georgia keeps our own file (g01, g02, main: kept), so this script touches two rows only:
AzerbaijanMain Energy and AzerbaijanMain Peak of the pDemandForecast file named in config.csv.

Perimeter. The Task 3 Azerbaijan figures are mainland only, the Azerenerji perimeter: the
Task 3 base year matches the ministry balance minus the Nakhchivan generation within a
fraction of a percent, the whole country does not, and Nakhchivan is never mentioned in
Task 3 (register a01 conversion, source azerenerji_minenergy_2023_2025). The anchors are
therefore applied to AzerbaijanMain directly and the Nakhchivan rows are not modified.

Path built, year by year, the same way as the fully aligned file:
  - the base year and the first model year keep our own energy; their peak is rebased on the
    metered Azerenerji record (PEAK_RECORD_MW, first model year), the base year pro rata of
    the energy, because our former load factor estimate sat well above the record;
  - linear from our first model year to the first Task 3 anchor;
  - linear between the anchors (2030, 2035, 2040);
  - beyond the last anchor, our own growth (the ratio of our former values).
Energy anchors = a01 detail t3_reference_gwh plus t3_ev_gwh. Peak anchors = a02 cesi_value
(Task 3 peak, EV included). No CESI figure lives in this file: everything CESI is read from
cesi/cesi_register.yaml (DVC only, client confidential) at run time.

Running the script twice gives the same file: the years it keeps and the growth it reuses
are unchanged by its own output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cesi_full import INPUT_ROOT, fmt, interp, load_config, load_register, num, read_csv, write_csv  # noqa: E402

ZONE = "AzerbaijanMain"
ENERGY_ID = "a01_dem_energy"
PEAK_ID = "a02_dem_peak"
# Azerenerji system record, 4,871 MW at 15:00 on 6 August 2025 (public, Report.az; source card
# azerenerji_minenergy_2023_2025). Mainland grid only, which is the AzerbaijanMain zone.
PEAK_RECORD_MW = {2025: 4871.0}


def anchors_energy(reg: dict) -> dict[int, float]:
    """Mainland energy anchors in GWh: Task 3 Reference plus EV, from the register detail."""
    detail = (reg.get(ENERGY_ID) or {}).get("detail") or {}
    ref, ev = detail.get("t3_reference_gwh"), detail.get("t3_ev_gwh")
    if not ref or not ev:
        sys.exit(f"register {ENERGY_ID}: detail needs t3_reference_gwh and t3_ev_gwh")
    if set(ref) != set(ev):
        sys.exit(f"register {ENERGY_ID}: reference and EV anchors must share the same years")
    return {int(y): float(ref[y]) + float(ev[y]) for y in ref}


def anchors_peak(reg: dict) -> dict[int, float]:
    """Mainland peak anchors in MW, Task 3 with EV."""
    if PEAK_ID not in reg:
        sys.exit(f"register entry {PEAK_ID} is missing")
    return {int(y): float(v) for y, v in reg[PEAK_ID]["cesi_value"].items()}


def align(base: dict[int, float], anchors: dict[int, float], years: list[int]) -> dict[int, float]:
    """Our first two years, linear to the first anchor, the anchors, then our growth beyond the last."""
    first, last = min(anchors), max(anchors)
    start = years[1] if len(years) > 1 else years[0]
    out = {}
    for y in years:
        if y <= start:
            out[y] = base[y]
        elif y < first:
            out[y] = base[start] + (anchors[first] - base[start]) * (y - start) / (first - start)
        elif y <= last:
            out[y] = interp(anchors, y)
        else:
            out[y] = anchors[last] * base[y] / base[last] if base[last] else anchors[last]
    return out


def rebase_peak(peak: dict[int, float], energy: dict[int, float], years: list[int]) -> dict[int, float]:
    """First model year peak = metered record; the base year follows the energy ratio; later years untouched."""
    start = years[1] if len(years) > 1 else years[0]
    if start not in PEAK_RECORD_MW:
        sys.exit(f"PEAK_RECORD_MW has no value for the first model year {start}")
    out = dict(peak)
    out[start] = PEAK_RECORD_MW[start]
    for y in years:
        if y < start:
            out[y] = out[start] * energy[y] / energy[start] if energy[start] else out[start]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployment", default="data_blacksea")
    ap.add_argument("--dry-run", action="store_true", help="log everything, write nothing")
    args = ap.parse_args()
    dep = INPUT_ROOT / args.deployment
    reg = load_register(dep)
    path = load_config(dep)["pDemandForecast"]
    header, rows, style = read_csv(path)
    years = [int(c) for c in header[2:]]
    by = {(r["z"], r["type"]): r for r in rows}

    def series(row: dict) -> dict[int, float]:
        return {y: num(row[str(y)], 0.0) for y in years}

    print(f"build_demand_baseline: {dep.name}, {path.relative_to(INPUT_ROOT).as_posix()}, zone {ZONE}")
    e_row, p_row = by[(ZONE, "Energy")], by[(ZONE, "Peak")]
    e_base = series(e_row)
    p_base = rebase_peak(series(p_row), e_base, years)
    log = []
    for kind, row, base, anchors, unit in (("Energy", e_row, e_base, anchors_energy(reg), "GWh"),
                                           ("Peak", p_row, p_base, anchors_peak(reg), "MW")):
        new = align(base, anchors, years)
        for y in years:
            row[str(y)] = fmt(new[y], 2)
        for y in (years[0], years[1], *sorted(anchors), years[-1]):
            log.append(f"    {ZONE} {kind} {y}: {row[str(y)]} {unit}")
    print("  values as written (mainland perimeter, Nakhchivan rows untouched):")
    print("\n".join(log))
    write_csv(path, header, rows, style, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
