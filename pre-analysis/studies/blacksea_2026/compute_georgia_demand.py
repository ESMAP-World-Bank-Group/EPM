"""
compute_georgia_demand.py
Compute pDemandForecast and pDemandProfile rows for Georgia zone.

Sources:
  - Av. 3% Load growth (hourly profiles) 2021-2040.xlsx
    (175,320 hourly rows, 2021-2040, 3% annual growth applied)

Output:
  - STDOUT: CSV rows to append to pDemandForecast.csv and pDemandProfile.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path(
    r"C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team"
    r"\Av. 3% Load growth (hourly profiles) 2021-2040.xlsx"
)
ZONE = "Georgia"

# ── Load ─────────────────────────────────────────────────────────────────────
print("Reading demand file...")
df = pd.read_excel(DATA_FILE)
df.columns = ["datetime", "load_mw"]
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"]   = df["datetime"].dt.year
df["month"]  = df["datetime"].dt.month
df["hour"]   = df["datetime"].dt.hour  # 0–23

def month_to_season(m):
    if m in (1, 2, 3):  return "Q1"
    if m in (4, 5, 6):  return "Q2"
    if m in (7, 8, 9):  return "Q3"
    return "Q4"

df["season"] = df["month"].apply(month_to_season)

# ── pDemandForecast ───────────────────────────────────────────────────────────
# Annual peak (MW) and energy (GWh) from file. Extrapolate 3%/yr beyond 2040.
peak_yr   = df.groupby("year")["load_mw"].max()
energy_yr = df.groupby("year")["load_mw"].sum() / 1000  # MWh → GWh

forecast_years = list(range(2024, 2054))
peak_vals, energy_vals = {}, {}
for y in forecast_years:
    if y in peak_yr.index:
        peak_vals[y]   = round(peak_yr[y], 2)
        energy_vals[y] = round(energy_yr[y], 2)
    else:
        n = y - 2040
        peak_vals[y]   = round(peak_yr[2040]   * (1.03 ** n), 2)
        energy_vals[y] = round(energy_yr[2040] * (1.03 ** n), 2)

header = "z,type," + ",".join(str(y) for y in forecast_years)
peak_row   = f"{ZONE},Peak,"   + ",".join(str(peak_vals[y])   for y in forecast_years)
energy_row = f"{ZONE},Energy," + ",".join(str(energy_vals[y]) for y in forecast_years)

print("\n" + "="*60)
print("# pDemandForecast rows (append to pDemandForecast.csv):")
print("# header:", header)
print(peak_row)
print(energy_row)

# Validation
print(f"\nValidation — 2025: Peak={peak_vals[2025]:.0f} MW, Energy={energy_vals[2025]:.0f} GWh")
print(f"Validation — 2030: Peak={peak_vals[2030]:.0f} MW, Energy={energy_vals[2030]:.0f} GWh")
print(f"Validation — 2040: Peak={peak_vals[2040]:.0f} MW, Energy={energy_vals[2040]:.0f} GWh")

# ── pDemandProfile ────────────────────────────────────────────────────────────
# Use 2025 data. Seasonal mean hourly profile, normalized by global peak.
df_ref = df[df["year"] == 2025].copy()

mean_profile = (
    df_ref.groupby(["season", "hour"])["load_mw"]
    .mean()
    .reset_index()
    .rename(columns={"load_mw": "mean_mw"})
)

global_peak_mw = mean_profile["mean_mw"].max()
mean_profile["norm"] = mean_profile["mean_mw"] / global_peak_mw

# Pivot: (season, hour) → dict
profile_lookup = {
    (row["season"], int(row["hour"])): row["norm"]
    for _, row in mean_profile.iterrows()
}

seasons   = ["Q1", "Q2", "Q3", "Q4"]
daytypes  = ["d1", "d2", "d3", "d4", "d5", "d6"]

profile_rows = []
for s in seasons:
    hourly = [profile_lookup[(s, h)] for h in range(24)]
    for dt in daytypes:
        vals = ",".join(f"{v:.16f}" for v in hourly)
        profile_rows.append(f"{ZONE},{s},{dt},{vals}")

print("\n" + "="*60)
print("# pDemandProfile rows (append to pDemandProfile.csv):")
print(f"# {len(profile_rows)} rows for {ZONE}")
for r in profile_rows[:3]:
    print(r[:120] + "...")
print(f"  ... and {len(profile_rows)-3} more rows")

# ── Write output files ────────────────────────────────────────────────────────
out_dir = Path(__file__).parent / "output_georgia"
out_dir.mkdir(exist_ok=True)

# Forecast
with open(out_dir / "georgia_pDemandForecast.csv", "w") as f:
    f.write(peak_row + "\n")
    f.write(energy_row + "\n")
print(f"\nWritten: {out_dir / 'georgia_pDemandForecast.csv'}")

# Profile
with open(out_dir / "georgia_pDemandProfile.csv", "w") as f:
    for r in profile_rows:
        f.write(r + "\n")
print(f"Written: {out_dir / 'georgia_pDemandProfile.csv'}")
print(f"\nDone. Global peak used for normalisation: {global_peak_mw:.2f} MW (from 2025 seasonal means)")
