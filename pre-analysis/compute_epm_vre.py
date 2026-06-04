"""
compute_epm_vre.py — Generic EPM pVREProfile builder from Renewables Ninja output.

Reads solar and wind capacity-factor CSVs produced by run_blacksea_data.py
(via vre_pipeline.py / Renewables Ninja API), computes multi-year seasonal mean
hourly profiles, and appends rows to pVREProfile.csv.

All d1-d6 daytypes within a season share the same seasonal mean profile.
This is a known simplification — recompute via full representative-days pipeline
when all region data is available.

Usage:
    python compute_epm_vre.py --country AZE
    python compute_epm_vre.py --country AZE ARM ROU BGR
    python compute_epm_vre.py --all
    python compute_epm_vre.py --country AZE --dry-run
    python compute_epm_vre.py --country AZE --deployment data_blacksea --append
    python compute_epm_vre.py --country AZE --ninja-dir path/to/ninja/

Options:
    --country ISO3 [...]  Countries to process
    --all                 Process all countries in zone mapping
    --deployment NAME     EPM deployment folder (default: data_blacksea)
    --append              Append/replace rows in pVREProfile.csv
    --dry-run             Print summary without writing files
    --ninja-dir PATH      Path to Ninja CSV directory (default: blacksea_run1/ninja/)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

_BASE_DIR   = Path(__file__).resolve().parent
_EPM_DIR    = _BASE_DIR.parent / "epm"
_OUTPUT_DIR = _BASE_DIR / "output_vre"
_DEFAULT_NINJA_DIR = (
    _BASE_DIR / "output_workflow" / "blacksea_run1" / "ninja"
)

# ISO-3 → EPM zone name (single-zone countries)
_ISO3_TO_ZONE: Dict[str, str] = {
    "AZE": "Azerbaijan",
    "ARM": "Armenia",
    "GEO": "Georgia",
    "ROU": "Romania",
    "BGR": "Bulgaria",
    "TUR": "Turkiye",
}

# EPM tech name for each Ninja file type
_NINJA_TECH: Dict[str, str] = {
    "solar": "PV",
    "wind":  "OnshoreWind",
}

MONTH_TO_SEASON = {
    1:"Q1", 2:"Q1", 3:"Q1",
    4:"Q2", 5:"Q2", 6:"Q2",
    7:"Q3", 8:"Q3", 9:"Q3",
    10:"Q4", 11:"Q4", 12:"Q4",
}

SEASONS  = ["Q1", "Q2", "Q3", "Q4"]
DAYTYPES = ["d1", "d2", "d3", "d4", "d5", "d6"]


# ── Core ───────────────────────────────────────────────────────────────────────

def _read_ninja_csv(ninja_dir: Path, tech_key: str) -> pd.DataFrame:
    """Read a Ninja output CSV (solar or wind) from the given directory."""
    pattern = f"vre_rninja_*_{tech_key}.csv"
    files = list(ninja_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {tech_key} Ninja CSV found in {ninja_dir}. "
            f"Expected pattern: {pattern}"
        )
    path = sorted(files)[-1]
    df = pd.read_csv(path, index_col=0)
    df.index.name = "zone"
    return df


def _compute_seasonal_profile(
    ninja_df: pd.DataFrame,
    zone: str,
) -> Dict[str, List[float]]:
    """Compute multi-year seasonal mean hourly profile for one zone.

    Returns {season: [v_h0, v_h1, ..., v_h23]} normalized so annual peak = 1.0.
    """
    if zone not in ninja_df.index:
        raise ValueError(f"Zone '{zone}' not found in Ninja CSV.")

    sub = ninja_df.loc[zone].copy()
    # sub columns: month, day, hour, 2018, 2019, ... (year columns)
    year_cols = [c for c in sub.columns if str(c).isdigit()]

    # Reshape to long: one row per (month, day, hour, year)
    records = []
    for yr in year_cols:
        for _, row in sub.iterrows():
            records.append({
                "month": int(row["month"]),
                "day":   int(row["day"]),
                "hour":  int(row["hour"]),
                "cf":    float(row[yr]),
            })
    long_df = pd.DataFrame(records)
    long_df["season"] = long_df["month"].map(MONTH_TO_SEASON)

    # Seasonal mean per hour (average over all years, days, and daytypes)
    mean_profile = (
        long_df.groupby(["season", "hour"])["cf"].mean()
    )

    # Normalize: divide by the maximum value across all seasons/hours
    peak = mean_profile.max()
    if peak <= 0:
        peak = 1.0
    norm = mean_profile / peak

    return {s: [norm.get((s, h), 0.0) for h in range(24)] for s in SEASONS}


def build_vre_rows(
    iso3_list: List[str],
    ninja_dir: Path,
    techs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build pVREProfile rows for given countries and tech types.

    techs: list of Ninja keys to process (default: ['solar', 'wind']).
    """
    if techs is None:
        techs = list(_NINJA_TECH.keys())

    # Pre-load Ninja CSVs
    ninja_data: Dict[str, pd.DataFrame] = {}
    for tech_key in techs:
        try:
            ninja_data[tech_key] = _read_ninja_csv(ninja_dir, tech_key)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    t_cols = [f"t{h:02d}" for h in range(1, 25)]
    rows = []

    for iso3 in iso3_list:
        zone = _ISO3_TO_ZONE.get(iso3, iso3.capitalize())

        for tech_key, epm_tech in _NINJA_TECH.items():
            if tech_key not in ninja_data:
                continue

            try:
                profiles = _compute_seasonal_profile(ninja_data[tech_key], zone)
            except ValueError as e:
                print(f"  WARNING: {e} — skipping {iso3}/{epm_tech}")
                continue

            means = {s: np.mean(profiles[s]) for s in SEASONS}
            print(f"  [vre] {iso3} {epm_tech}: "
                  f"Q1={means['Q1']:.3f}  Q2={means['Q2']:.3f}  "
                  f"Q3={means['Q3']:.3f}  Q4={means['Q4']:.3f}")

            for season in SEASONS:
                hourly = profiles[season]
                for dt in DAYTYPES:
                    row = {"zone": zone, "tech": epm_tech, "season": season, "daytype": dt}
                    for h in range(24):
                        row[t_cols[h]] = hourly[h]
                    rows.append(row)

    return pd.DataFrame(rows, columns=["zone", "tech", "season", "daytype"] + t_cols)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build pVREProfile rows from Renewables Ninja output"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--country", nargs="+", metavar="ISO3")
    grp.add_argument("--all", action="store_true",
                     help="Process all countries in zone mapping")
    parser.add_argument("--deployment", default="data_blacksea")
    parser.add_argument("--append", action="store_true",
                        help="Replace rows in pVREProfile.csv for these zones")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ninja-dir", default=None, dest="ninja_dir",
                        help="Path to Ninja CSV directory")
    parser.add_argument("--techs", nargs="+", default=None,
                        choices=list(_NINJA_TECH.keys()),
                        help="Subset of techs to process (default: all)")
    args = parser.parse_args()

    if args.all:
        iso3_list = list(_ISO3_TO_ZONE.keys())
    else:
        iso3_list = [c.upper() for c in args.country]
        unknown = [c for c in iso3_list if c not in _ISO3_TO_ZONE]
        if unknown:
            print(f"WARNING: {unknown} not in zone mapping — skipping")
            iso3_list = [c for c in iso3_list if c in _ISO3_TO_ZONE]
    if not iso3_list:
        print("No valid countries."); sys.exit(1)

    ninja_dir = Path(args.ninja_dir) if args.ninja_dir else _DEFAULT_NINJA_DIR
    if not ninja_dir.exists():
        print(f"ERROR: Ninja directory not found: {ninja_dir}")
        sys.exit(1)

    print(f"[vre] Countries: {iso3_list}")
    print(f"[vre] Ninja dir: {ninja_dir}")

    df = build_vre_rows(iso3_list, ninja_dir, techs=args.techs)

    print(f"\n[vre] Generated {len(df)} rows "
          f"({df['tech'].nunique()} techs x {len(iso3_list)} countries x 24 season/daytype combos)")

    if args.dry_run:
        print("[vre] DRY RUN — not writing files.")
        return

    _OUTPUT_DIR.mkdir(exist_ok=True)
    tag = "_".join(iso3_list)
    out_csv = _OUTPUT_DIR / f"pVREProfile_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[vre] Written: {out_csv}")

    if args.append:
        target = _EPM_DIR / "input" / args.deployment / "supply" / "pVREProfile.csv"
        if not target.exists():
            print(f"  WARNING: {target} not found — skipping append")
        else:
            existing = pd.read_csv(target)
            # Remove existing rows for these zones AND techs
            zones = df["zone"].unique().tolist()
            techs_to_replace = df["tech"].unique().tolist()
            mask = existing["zone"].isin(zones) & existing["tech"].isin(techs_to_replace)
            existing = existing[~mask]
            updated = pd.concat([existing, df], ignore_index=True)
            updated.to_csv(target, index=False)
            print(f"  Replaced {zones} / {techs_to_replace} in {target.name} "
                  f"({len(updated)} total rows)")

    print("[vre] Done.")


if __name__ == "__main__":
    main()
