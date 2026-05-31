"""Compute Armenia pVREProfile rows from Renewables Ninja data.

Approach:
  - PV + Wind: seasonal mean from Renewables Ninja Armenia (2018–2023).
    All d1–d6 within a season share the same mean profile (simplification —
    within-season variability is lost; re-run representative days pipeline
    for Armenia when all Black Sea countries are ready).
  - ROR: proxy from EastAna Turkiye zone (nearest, similar snowmelt hydrology).

Output: Armenia rows appended to epm/input/data_blacksea/supply/pVREProfile.csv
"""

from pathlib import Path
import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
NINJA_DIR = BASE / "pre-analysis" / "output_workflow" / "blacksea_run1" / "ninja"
VRE_CSV   = BASE / "epm" / "input" / "data_blacksea" / "supply" / "pVREProfile.csv"

# Season → months mapping (calendar quarters)
SEASON_MONTHS = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}
DAYTYPES      = ["d1", "d2", "d3", "d4", "d5", "d6"]
HOUR_COLS     = [f"t{str(h).zfill(2)}" for h in range(1, 25)]
YEAR_COLS     = ["2018", "2019", "2020", "2021", "2022", "2023"]


def seasonal_mean_profile(ninja_path: Path, zone: str) -> dict[str, list[float]]:
    """Return {Q: [24 hourly mean CFs]} averaged over all years and days in that season."""
    df = pd.read_csv(ninja_path)
    df = df[df["zone"] == zone].copy()

    # Average across years
    df["cf"] = df[YEAR_COLS].mean(axis=1)

    profiles = {}
    for q, months in SEASON_MONTHS.items():
        season_df = df[df["month"].isin(months)]
        # Mean hourly profile across all days in season
        hourly_mean = season_df.groupby("hour")["cf"].mean().sort_index().values
        profiles[q] = hourly_mean.tolist()
    return profiles


def build_armenia_rows(tech: str, profiles: dict) -> list[dict]:
    """Build pVREProfile rows for Armenia — all d1-d6 in a season share the same mean profile."""
    rows = []
    for q, hourly in profiles.items():
        for d in DAYTYPES:
            row = {"zone": "Armenia", "tech": tech, "season": q, "daytype": d}
            for i, col in enumerate(HOUR_COLS):
                row[col] = round(hourly[i], 6)
            rows.append(row)
    return rows


def proxy_ror_from_eastana(vre_csv: Path) -> list[dict]:
    """Copy EastAna ROR rows from pVREProfile, replace zone with Armenia."""
    df = pd.read_csv(vre_csv)
    ror = df[(df["zone"] == "EastAna") & (df["tech"] == "ROR")].copy()
    ror["zone"] = "Armenia"
    return ror.to_dict("records")


def main():
    print("Computing Armenia VRE profiles...")

    # PV
    print("  PV: reading ninja solar data...")
    pv_profiles = seasonal_mean_profile(NINJA_DIR / "vre_rninja_blacksea_solar.csv", "Armenia")
    pv_rows = build_armenia_rows("PV", pv_profiles)
    print(f"  PV: {len(pv_rows)} rows (4Q × 6d)")

    # Wind
    print("  Wind: reading ninja wind data...")
    wind_profiles = seasonal_mean_profile(NINJA_DIR / "vre_rninja_blacksea_wind.csv", "Armenia")
    wind_rows = build_armenia_rows("OnshoreWind", wind_profiles)
    print(f"  Wind: {len(wind_rows)} rows (4Q × 6d)")

    # ROR proxy
    print("  ROR: proxying from EastAna...")
    ror_rows = proxy_ror_from_eastana(VRE_CSV)
    print(f"  ROR: {len(ror_rows)} rows (proxy EastAna)")

    # Load existing CSV and append
    existing = pd.read_csv(VRE_CSV)

    # Guard: remove any existing Armenia rows to avoid duplicates
    existing = existing[existing["zone"] != "Armenia"]

    all_new = pd.DataFrame(pv_rows + wind_rows + ror_rows)
    all_new = all_new[existing.columns]  # align column order

    combined = pd.concat([existing, all_new], ignore_index=True)
    combined.to_csv(VRE_CSV, index=False)
    print(f"\nWritten {len(all_new)} Armenia rows to {VRE_CSV}")
    print("NOTE: d1–d6 share the same seasonal mean profile.")
    print("      Mark pVREProfile as needs_review — rerun representative days pipeline")
    print("      with all Black Sea countries when full data is available.")


if __name__ == "__main__":
    main()
