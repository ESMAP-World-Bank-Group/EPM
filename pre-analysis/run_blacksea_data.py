"""Black Sea Run 1 — load + VRE data preparation.

Builds all inputs needed for representative-days pipeline:
  - Turkey 9 zones : ENTSO-E national disaggregated by pDemandForecast.csv zone weights
  - ROU, BGR, GEO  : ENTSO-E national (1 zone each)
  - ARM             : proxy — ENTSO-E Georgia shape scaled to 7 TWh
  - AZE             : proxy — ENTSO-E Turkey shape scaled to 29.3 TWh
  - VRE             : Renewables Ninja for all 14 zone centroids (~5h, rate-limited)

Usage:
    conda activate esmap_env
    cd pre-analysis
    python run_blacksea_data.py                  # full run
    python run_blacksea_data.py --skip-ninja      # skip Ninja (already downloaded)
    python run_blacksea_data.py --ninja-only      # only run Ninja step
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "pipelines"))

from entsoe_pipeline import run_entsoe_pipeline, country_name_to_iso2
from vre_pipeline import run_renewables_ninja_workflow

# ── Paths ─────────────────────────────────────────────────────────────────────

DEMAND_FORECAST_PATH = BASE_DIR.parent / "epm" / "input" / "data_blacksea" / "load" / "pDemandForecast.csv"
TUR_9Z_GEOJSON      = BASE_DIR / "output_workflow" / "zoning_study" / "TUR_9z" / "epm_export" / "spatial" / "zones.geojson"
OUTPUT_DIR          = BASE_DIR / "output_workflow" / "blacksea_run1"
ENTSOE_DIR          = OUTPUT_DIR / "entsoe"
LOAD_DIR            = OUTPUT_DIR / "load"
NINJA_DIR           = OUTPUT_DIR / "ninja"
REPRDAYS_DIR        = OUTPUT_DIR / "reprdays_input"

# ── Constants ─────────────────────────────────────────────────────────────────

ENTSOE_START    = "2018-01-01"
ENTSOE_END      = "2024-12-31"
ENTSOE_TIMEZONE = "Europe/Brussels"
NINJA_START     = 2018
NINJA_END       = 2024   # exclusive -> 2018–2023

ARM_ANNUAL_MWH  = 7_000_000    # ~7 TWh
AZE_ANNUAL_MWH  = 29_300_000   # ~29.3 TWh
TUR_ANNUAL_MWH  = 290_000_000  # ~290 TWh (proxy — ENTSO-E does not cover Turkey)

# ENTSO-E countries — Turkey excluded (not an ENTSO-E member)
ENTSOE_COUNTRIES = ["Romania", "Bulgaria", "Georgia"]

# Zone centroids for the 5 new 1-zone countries (lon, lat)
NEW_ZONE_CENTROIDS = {
    "Romania":    (25.0,  46.0),
    "Bulgaria":   (25.5,  42.7),
    "Georgia":    (43.4,  42.3),
    "Armenia":    (44.5,  40.2),
    "Azerbaijan": (47.4,  40.7),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def turkey_zone_centroids() -> dict[str, tuple[float, float]]:
    """Read TUR_9z GeoJSON and return {zone_name: (lon, lat)} centroids."""
    with open(TUR_9Z_GEOJSON) as f:
        gj = json.load(f)
    centroids = {}
    for feat in gj["features"]:
        name = feat["properties"]["zone_name"]
        coords_raw = feat["geometry"]["coordinates"]
        pts: list[tuple[float, float]] = []

        def collect(c):
            if isinstance(c[0], list):
                for x in c:
                    collect(x)
            else:
                pts.append((c[0], c[1]))

        collect(coords_raw)
        lon = float(np.mean([p[0] for p in pts]))
        lat = float(np.mean([p[1] for p in pts]))
        centroids[name] = (lon, lat)
    return centroids


def load_entsoe_load(country_name: str) -> pd.DataFrame:
    """Return hourly load DataFrame (timestamp index, MW values) for a country from cached ENTSO-E CSVs."""
    iso2 = country_name_to_iso2(country_name)
    pattern = f"entsoe_load_{iso2.lower()}_*.csv"
    # entsoe_pipeline saves files in <ENTSOE_DIR>/load/ subdirectory
    search_dirs = [ENTSOE_DIR / "load", ENTSOE_DIR]
    files = []
    for d in search_dirs:
        files = sorted(d.glob(pattern))
        if files:
            break
    if not files:
        raise FileNotFoundError(
            f"No ENTSO-E load file for {country_name} ({iso2}) in {ENTSOE_DIR}. "
            "Run without --skip-ninja first to download ENTSO-E data."
        )
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    # Resample to hourly (ENTSO-E may return 15-min or mixed-resolution data)
    combined = combined.resample("h").mean()
    # Keep only the load column (drop sum/crossborder cols)
    load_col = next(
        (c for c in combined.columns if "actual" in str(c).lower() or "load" in str(c).lower()),
        combined.columns[0],
    )
    return combined[[load_col]].rename(columns={load_col: "load_mw"})


def build_proxy_load(source_df: pd.DataFrame, target_annual_mwh: float) -> pd.DataFrame:
    """Scale a load shape to a target annual energy (MWh).

    Args:
        source_df: hourly load with 'load_mw' column.
        target_annual_mwh: desired annual energy in MWh.
    Returns:
        DataFrame with same index and 'load_mw' column scaled to target.
    """
    current_mwh = source_df["load_mw"].sum()
    scale = target_annual_mwh / current_mwh
    return source_df.assign(load_mw=source_df["load_mw"] * scale)


def disaggregate_turkey_load(national_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split Turkey national load into 9 zones using annual energy shares from pDemandForecast.csv."""
    forecast = pd.read_csv(DEMAND_FORECAST_PATH)
    # Keep first available year for energy weights
    year_cols = [c for c in forecast.columns if str(c).isdigit()]
    first_year = year_cols[0]
    energy = forecast[forecast["type"].str.lower() == "energy"][["z", first_year]].copy()
    energy = energy.rename(columns={first_year: "gwh"})
    energy["weight"] = energy["gwh"] / energy["gwh"].sum()

    zones = {}
    for _, row in energy.iterrows():
        zone_df = national_df.copy()
        zone_df["load_mw"] = zone_df["load_mw"] * float(row["weight"])
        zones[row["z"]] = zone_df
    return zones


def to_reprdays_format(zone_loads: dict[str, pd.DataFrame], output_path: Path) -> Path:
    """Convert {zone: hourly_df} to representative-days input CSV (zone, month, day, hour, value).

    Values are normalized 0–1 within each zone (peak = 1).
    """
    records = []
    for zone, df in zone_loads.items():
        s = df["load_mw"].copy()
        peak = s.max()
        if peak > 0:
            s = s / peak
        tmp = pd.DataFrame({
            "zone":  zone,
            "month": df.index.month,
            "day":   df.index.day,
            "hour":  df.index.hour,
            "value": s.values,
        })
        records.append(tmp)

    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values(["zone", "month", "day", "hour"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, float_format="%.4f")
    print(f"[load] Written {output_path}")
    return output_path


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_entsoe():
    """Download ENTSO-E load for Turkey, Romania, Bulgaria, Georgia (2018–2024)."""
    print("\n=== Step 1: ENTSO-E download ===")
    ENTSOE_DIR.mkdir(parents=True, exist_ok=True)
    rc = run_entsoe_pipeline(
        start=ENTSOE_START,
        end=ENTSOE_END,
        timezone=ENTSOE_TIMEZONE,
        country_inputs=ENTSOE_COUNTRIES,
        output_dir=ENTSOE_DIR,
        dry_run=False,
        verbose=True,
        api_tokens_path=None,
        variables=["load"],
        refresh_existing=False,
    )
    if rc != 0:
        raise RuntimeError(f"ENTSO-E pipeline returned exit code {rc}")
    print("[entsoe] Done.")


def step_load():
    """Build load CSVs for all 14 zones."""
    print("\n=== Step 2: Load profiles ===")
    LOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPRDAYS_DIR.mkdir(parents=True, exist_ok=True)

    zone_loads: dict[str, pd.DataFrame] = {}

    # Turkey 9 zones — proxy from Romania shape (ENTSO-E does not cover Turkey)
    print("[load] Building Turkey proxy (Romania shape × 290 TWh)")
    rou_load = load_entsoe_load("Romania")
    tur_national = build_proxy_load(rou_load, TUR_ANNUAL_MWH)
    zone_loads.update(disaggregate_turkey_load(tur_national))

    # Romania, Bulgaria, Georgia — 1 zone each
    for country, zone in [("Romania", "Romania"), ("Bulgaria", "Bulgaria"), ("Georgia", "Georgia")]:
        print(f"[load] Processing {country} -> {zone}")
        zone_loads[zone] = load_entsoe_load(country)

    # Armenia proxy — Georgia shape × 7 TWh
    print("[load] Building Armenia proxy (Georgia shape × 7 TWh)")
    geo_load = load_entsoe_load("Georgia")
    zone_loads["Armenia"] = build_proxy_load(geo_load, ARM_ANNUAL_MWH)

    # Azerbaijan proxy — Turkey shape × 29.3 TWh
    print("[load] Building Azerbaijan proxy (Turkey shape × 29.3 TWh)")
    zone_loads["Azerbaijan"] = build_proxy_load(tur_national, AZE_ANNUAL_MWH)

    to_reprdays_format(zone_loads, REPRDAYS_DIR / "Load.csv")
    print(f"[load] All zone profiles written to {REPRDAYS_DIR / 'Load.csv'}")


def step_ninja():
    """Launch Renewables Ninja for all 14 zone centroids (solar + wind, 2018–2023)."""
    print("\n=== Step 3: Renewables Ninja ===")
    NINJA_DIR.mkdir(parents=True, exist_ok=True)

    tur_centroids = turkey_zone_centroids()
    all_centroids = {**tur_centroids, **{z: (lon, lat) for z, (lon, lat) in NEW_ZONE_CENTROIDS.items()}}

    # Ninja locations format: {zone: (lat, lon)}  — note lat/lon order
    ninja_locations = {
        "solar": {z: (lat, lon) for z, (lon, lat) in all_centroids.items()},
        "wind":  {z: (lat, lon) for z, (lon, lat) in all_centroids.items()},
    }

    n_zones = len(all_centroids)
    n_years = NINJA_END - NINJA_START
    n_calls  = n_zones * 2 * n_years
    print(f"[ninja] {n_zones} zones × 2 techs × {n_years} years = {n_calls} API calls")
    print(f"[ninja] Rate limit: 36 calls/hour -> estimated {n_calls/36:.1f}h")
    print("[ninja] Starting... (safe to leave running overnight)")

    run_renewables_ninja_workflow(
        locations=ninja_locations,
        start_year=NINJA_START,
        end_year=NINJA_END,
        dataset_label="blacksea",
        input_dir=str(NINJA_DIR),
        output_dir=str(NINJA_DIR),
        generate_plots=False,
    )
    print(f"[ninja] Done. CSVs written to {NINJA_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Black Sea Run 1 data preparation")
    parser.add_argument("--skip-ninja",   action="store_true", help="Skip Renewables Ninja step")
    parser.add_argument("--ninja-only",   action="store_true", help="Run only Renewables Ninja step")
    parser.add_argument("--skip-entsoe",  action="store_true", help="Skip ENTSO-E download (use cached)")
    args = parser.parse_args()

    if args.ninja_only:
        step_ninja()
        return

    if not args.skip_entsoe:
        step_entsoe()

    step_load()

    if not args.skip_ninja:
        step_ninja()
    else:
        print("\n[skipped] Renewables Ninja — run with --ninja-only when ready.")

    print("\n=== All done ===")
    print(f"Load profiles : {REPRDAYS_DIR / 'Load.csv'}")
    print(f"Ninja VRE     : {NINJA_DIR}")
    print("Next step     : feed these CSVs into representative_days pipeline via open_data_config.yaml")


if __name__ == "__main__":
    main()
