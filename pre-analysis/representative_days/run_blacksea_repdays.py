"""Driver: Black Sea representative-days run.

- Extracts the chosen representative YEAR (2023) as single-value CSVs for Load, PV, Wind.
- Runs the representative-days pipeline: 4 calendar seasons x 4 representative days
  (extreme/special days added in a second pass — see --with-extremes).

Inputs (from blacksea_run1, UTC, 15 zones incl. AzerbaijanMain + Nakhchivan):
  Load : output_workflow/blacksea_run1/reprdays_input/Load.csv   (zone,year,month,day,hour,value)
  VRE  : output_workflow/blacksea_run1/ninja/vre_rninja_blacksea_{solar,wind}.csv  (year columns)
"""
from pathlib import Path
import argparse
import pandas as pd

from representativedays_pipeline import run_representative_days_pipeline

BASE = Path(__file__).resolve().parent
RUN1 = BASE.parent / "output_workflow" / "blacksea_run1"
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output" / "blacksea"

# 4 calendar quarters (aligned with pAvailabilityCustom Q1-Q4)
SEASONS_MAP = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}

# Include all zones (BG/RO too) so pHours + per-zone profiles are generated once and never
# need regeneration. Tractability is handled by feature_selection, not by dropping zones.
EXCLUDE_ZONES = []


def extract_year(year: int) -> dict:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    # Load: filter chosen year, drop year col
    load = pd.read_csv(RUN1 / "reprdays_input" / "Load.csv")
    load_y = load[load["year"] == year][["zone", "month", "day", "hour", "value"]].copy()
    # Fill ENTSO-E hourly gaps (NaN) by per-zone time interpolation (both directions)
    n_nan = int(load_y["value"].isna().sum())
    load_y = load_y.sort_values(["zone", "month", "day", "hour"])
    load_y["value"] = load_y.groupby("zone")["value"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    if n_nan:
        print(f"[blacksea-repdays] Load {year}: interpolated {n_nan} NaN hours (ENTSO-E gaps)")
    p = INPUT_DIR / f"Load_{year}.csv"
    load_y.to_csv(p, index=False)
    out["Load"] = str(p)
    # VRE: keep the chosen year column, rename to 'value'
    for tech, fname in [("PV", "vre_rninja_blacksea_solar.csv"), ("Wind", "vre_rninja_blacksea_wind.csv")]:
        v = pd.read_csv(RUN1 / "ninja" / fname)
        cols = ["zone", "month", "day", "hour", str(year)]
        vy = v[cols].rename(columns={str(year): "value"})
        p = INPUT_DIR / f"{tech}_{year}.csv"
        vy.to_csv(p, index=False)
        out[tech] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--days", type=int, default=4, help="representative days per season")
    ap.add_argument("--special-threshold", type=float, default=0.1)
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--feature-selection", type=int, default=20,
                    help="keep only N representative series before the Poncelet optim (tractability)")
    args = ap.parse_args()

    input_files = extract_year(args.year)
    print(f"[blacksea-repdays] Year {args.year} inputs: {input_files}")
    print(f"[blacksea-repdays] Excluding zones: {EXCLUDE_ZONES} | feature_selection={args.feature_selection}")
    gams_model = BASE / "gams" / "OptimizationModelZone.gms"

    result = run_representative_days_pipeline(
        seasons_map=SEASONS_MAP,
        input_files=input_files,
        output_dir=OUTPUT_DIR,
        gams_main_file=str(gams_model),
        year_label=args.year,
        zones_to_exclude=EXCLUDE_ZONES,
        n_representative_days=args.days,
        n_clusters=args.n_clusters,
        n_bins=10,
        feature_selection_count=args.feature_selection,
        special_day_threshold=args.special_threshold,
        verbose=True,
    )
    print("\n[blacksea-repdays] Done. Outputs:")
    for k, v in result["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
