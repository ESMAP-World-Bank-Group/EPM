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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
_EPM_DIR  = _BASE_DIR.parent / "epm"
_OUTPUT_DIR = _BASE_DIR / "output_demand"

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

    if args.dry_run:
        print("\n[demand] DRY RUN — not writing files.")
        return

    _OUTPUT_DIR.mkdir(exist_ok=True)
    tag = "_".join(iso3_list)
    out_csv = _OUTPUT_DIR / f"pDemandForecast_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[demand] Written: {out_csv}")

    if args.append:
        target = _EPM_DIR / "input" / args.deployment / "load" / "pDemandForecast.csv"
        if not target.exists():
            print(f"  WARNING: {target} not found — skipping append")
        else:
            existing = pd.read_csv(target)
            zones_to_replace = df["z"].unique().tolist()
            existing = existing[~existing["z"].isin(zones_to_replace)]
            updated = pd.concat([existing, df], ignore_index=True)
            updated.to_csv(target, index=False)
            print(f"  Replaced {zones_to_replace} rows in {target.name} "
                  f"({len(updated)} total rows)")

    print("[demand] Done.")


if __name__ == "__main__":
    main()
