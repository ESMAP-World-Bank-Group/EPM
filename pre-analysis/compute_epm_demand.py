"""
compute_epm_demand.py — Generic EPM pDemandForecast builder from OWID energy data.

Uses Our World in Data (OWID) electricity_demand statistics as the source for
annual energy (GWh) and estimates peak demand (MW) via a configurable load factor.
Data is fetched automatically from GitHub and cached locally.

Best used for countries without an existing hourly demand study (ENTSO-E, CESI, etc.).
For countries covered by ENTSO-E (TUR, ROU, BGR, GEO), the run_blacksea_data.py
pipeline produces better hourly-grounded demand profiles.

Usage:
    python compute_epm_demand.py --country AZE
    python compute_epm_demand.py --country AZE ARM ROU BGR
    python compute_epm_demand.py --all
    python compute_epm_demand.py --country AZE --dry-run
    python compute_epm_demand.py --country AZE --deployment data_blacksea --append
    python compute_epm_demand.py --country AZE --growth 0.025 --load-factor 0.60

Options:
    --country ISO3 [...]  Countries to process
    --all                 Process all countries in _ISO3_TO_OWID mapping
    --deployment NAME     EPM deployment folder (default: data_blacksea)
    --append              Replace existing rows for these countries in pDemandForecast.csv
    --dry-run             Print rows without writing files
    --force-download      Re-download OWID dataset even if cached
    --growth FLOAT        Override CAGR growth rate for all countries (e.g. 0.025)
    --load-factor FLOAT   Peak/average ratio inverse (default 0.58)
    --profile             Also build pDemandProfile from Load.csv (representative days output)
    --load-csv PATH       Path to Load.csv (default: auto-detect from blacksea_run1 output)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_BASE_DIR   = Path(__file__).resolve().parent
_EPM_DIR    = _BASE_DIR.parent / "epm"
_OUTPUT_DIR = _BASE_DIR / "output_demand"
_DEFAULT_LOAD_CSV = (
    _BASE_DIR / "output_workflow" / "blacksea_run1" / "reprdays_input" / "Load.csv"
)

MONTH_TO_SEASON = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",
                   7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}

sys.path.insert(0, str(_BASE_DIR))
from pipelines.owid_energy_pipeline import (
    load_owid_demand,
    get_demand_forecast,
    _ISO3_TO_OWID,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_model_years(deployment: str) -> List[int]:
    """Read model years from pDemandForecast.csv header."""
    path = _EPM_DIR / "input" / deployment / "load" / "pDemandForecast.csv"
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        return sorted(int(c) for c in header if str(c).isdigit())
    # Fallback
    return list(range(2024, 2054))


def _zone_for_iso(iso3: str) -> str:
    """Return EPM zone name for a given ISO-3 code (single-zone countries)."""
    # Same as epm_gendata_config.yaml countries section
    zones = {
        "AZE": "Azerbaijan", "ARM": "Armenia", "GEO": "Georgia",
        "ROU": "Romania",    "BGR": "Bulgaria", "TUR": "Turkiye",
    }
    return zones.get(iso3, iso3.capitalize())


# ── Core ──────────────────────────────────────────────────────────────────────

def build_forecast_rows(
    iso3_list: List[str],
    model_years: List[int],
    load_factor: float,
    growth_rate: Optional[float],
    force_download: bool,
) -> pd.DataFrame:
    """Return DataFrame of pDemandForecast rows for given countries."""
    owid = load_owid_demand(iso3_list, force=force_download)

    year_cols = [str(y) for y in model_years]
    rows = []

    for iso3 in iso3_list:
        zone = _zone_for_iso(iso3)
        try:
            fc = get_demand_forecast(
                owid_df=owid,
                iso3=iso3,
                model_years=model_years,
                load_factor=load_factor,
                growth_rate=growth_rate,
            )
        except ValueError as e:
            print(f"  WARNING: {e} — skipping {iso3}")
            continue

        peak_row   = {"z": zone, "type": "Peak"}
        energy_row = {"z": zone, "type": "Energy"}
        for yr in model_years:
            peak_row[str(yr)]   = fc["Peak"][yr]
            energy_row[str(yr)] = fc["Energy"][yr]

        rows.append(peak_row)
        rows.append(energy_row)

    cols = ["z", "type"] + year_cols
    return pd.DataFrame(rows, columns=cols)


def build_profile_rows(
    iso3_list: List[str],
    load_csv: Path,
) -> pd.DataFrame:
    """Extract pDemandProfile rows from Load.csv for given countries.

    Load.csv format: zone, month, day, hour, value (normalized 0-1).
    Output: zone, season, daytype, t01-t24 (seasonal mean, same for all d1-d6).

    NOTE: d1-d6 all share the same seasonal mean profile — within-season
    variability is lost. Recompute via full representative-days pipeline when
    all region data + VRE profiles are available.
    """
    load = pd.read_csv(load_csv)
    load["season"] = load["month"].map(MONTH_TO_SEASON)

    rows = []
    for iso3 in iso3_list:
        zone = _zone_for_iso(iso3)
        sub  = load[load["zone"] == zone]
        if sub.empty:
            print(f"  WARNING: zone '{zone}' not found in Load.csv — skipping profile")
            continue

        # Seasonal mean hourly profile (4 seasons × 24 hours)
        mean_profile = (
            sub.groupby(["season", "hour"])["value"].mean()
        )

        seasons  = ["Q1", "Q2", "Q3", "Q4"]
        daytypes = ["d1", "d2", "d3", "d4", "d5", "d6"]
        for season in seasons:
            hourly = [mean_profile.get((season, h), 0.0) for h in range(24)]
            for dt in daytypes:
                row = {"zone": zone, "season": season, "daytype": dt}
                for h in range(24):
                    row[f"t{h+1:02d}"] = hourly[h]
                rows.append(row)

        print(f"  [profile] {iso3} ({zone}): Q1_mean={mean_profile.xs('Q1').mean():.3f}  "
              f"Q2_mean={mean_profile.xs('Q2').mean():.3f}  "
              f"Q3_mean={mean_profile.xs('Q3').mean():.3f}  "
              f"Q4_mean={mean_profile.xs('Q4').mean():.3f}")

    t_cols = [f"t{h:02d}" for h in range(1, 25)]
    return pd.DataFrame(rows, columns=["zone", "season", "daytype"] + t_cols)


def print_report(df: pd.DataFrame):
    first_yr = df.columns[2]
    mid_yr   = "2030" if "2030" in df.columns else df.columns[len(df.columns) // 2]
    last_yr  = df.columns[-1]
    print(f"\n{'='*60}")
    print(f"  {'Zone':<20} {'Type':<8} {first_yr:>8} {mid_yr:>8} {last_yr:>8}")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['z']:<20} {row['type']:<8} "
              f"{float(row[first_yr]):>8.0f} {float(row[mid_yr]):>8.0f} {float(row[last_yr]):>8.0f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build pDemandForecast rows from OWID electricity data"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--country", nargs="+", metavar="ISO3")
    grp.add_argument("--all", action="store_true",
                     help="Process all countries in OWID mapping")
    parser.add_argument("--deployment", default="data_blacksea")
    parser.add_argument("--append", action="store_true",
                        help="Replace rows for these countries in pDemandForecast.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--growth", type=float, default=None,
                        help="Override annual growth rate (e.g. 0.02 for 2%%)")
    parser.add_argument("--load-factor", type=float, default=0.58,
                        dest="load_factor",
                        help="Load factor for peak estimation (default 0.58)")
    parser.add_argument("--profile", action="store_true",
                        help="Also build pDemandProfile from Load.csv")
    parser.add_argument("--load-csv", default=None, dest="load_csv",
                        help="Path to Load.csv (default: blacksea_run1 output)")
    args = parser.parse_args()

    if args.all:
        iso3_list = list(_ISO3_TO_OWID.keys())
    else:
        iso3_list = [c.upper() for c in args.country]
        unknown = [c for c in iso3_list if c not in _ISO3_TO_OWID]
        if unknown:
            print(f"WARNING: {unknown} not in OWID mapping — skipping")
            iso3_list = [c for c in iso3_list if c in _ISO3_TO_OWID]
    if not iso3_list:
        print("No valid countries."); sys.exit(1)

    model_years = _read_model_years(args.deployment)
    print(f"[demand] Countries: {iso3_list}")
    print(f"[demand] Model years: {model_years[0]}–{model_years[-1]} "
          f"({len(model_years)} years)")
    print(f"[demand] Load factor: {args.load_factor}, "
          f"Growth: {'OWID CAGR' if args.growth is None else f'{args.growth*100:.1f}%/yr'}")

    df = build_forecast_rows(
        iso3_list=iso3_list,
        model_years=model_years,
        load_factor=args.load_factor,
        growth_rate=args.growth,
        force_download=args.force_download,
    )

    print_report(df)

    # ── pDemandProfile (optional) ─────────────────────────────────────────────
    profile_df = None
    if args.profile:
        load_csv = Path(args.load_csv) if args.load_csv else _DEFAULT_LOAD_CSV
        if not load_csv.exists():
            print(f"  WARNING: Load.csv not found at {load_csv} — skipping profile")
        else:
            print(f"\n[profile] Building pDemandProfile from {load_csv.name}")
            profile_df = build_profile_rows(iso3_list, load_csv)

    if args.dry_run:
        print("\n[demand] DRY RUN — not writing files.")
        if profile_df is not None:
            print(f"[profile] Would write {len(profile_df)} pDemandProfile rows")
        return

    _OUTPUT_DIR.mkdir(exist_ok=True)
    tag = "_".join(iso3_list)
    out_csv = _OUTPUT_DIR / f"pDemandForecast_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[demand] Written: {out_csv}")

    if profile_df is not None:
        out_prof = _OUTPUT_DIR / f"pDemandProfile_{tag}.csv"
        profile_df.to_csv(out_prof, index=False)
        print(f"[profile] Written: {out_prof}")

    if args.append:
        # pDemandForecast
        target = _EPM_DIR / "input" / args.deployment / "load" / "pDemandForecast.csv"
        if not target.exists():
            print(f"  WARNING: {target} not found — skipping forecast append")
        else:
            existing = pd.read_csv(target)
            zones_to_replace = df["z"].unique().tolist()
            existing = existing[~existing["z"].isin(zones_to_replace)]
            updated = pd.concat([existing, df], ignore_index=True)
            updated.to_csv(target, index=False)
            print(f"  Replaced {zones_to_replace} rows in {target.name} "
                  f"({len(updated)} total rows)")

        # pDemandProfile
        if profile_df is not None:
            prof_target = _EPM_DIR / "input" / args.deployment / "load" / "pDemandProfile.csv"
            if not prof_target.exists():
                print(f"  WARNING: {prof_target} not found — skipping profile append")
            else:
                existing_prof = pd.read_csv(prof_target)
                zones_to_replace = profile_df["zone"].unique().tolist()
                existing_prof = existing_prof[~existing_prof["zone"].isin(zones_to_replace)]
                updated_prof = pd.concat([existing_prof, profile_df], ignore_index=True)
                updated_prof.to_csv(prof_target, index=False)
                print(f"  Replaced {zones_to_replace} rows in {prof_target.name} "
                      f"({len(updated_prof)} total rows)")

    print("[demand] Done.")


if __name__ == "__main__":
    main()
