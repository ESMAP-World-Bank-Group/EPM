"""
compute_georgia_vre_profiles.py
Compute pVREProfile rows for Georgia: ROR, OnshoreWind, PV.

Source: Timeseries all data.xlsx — sheet "RE data"
  - 8,760 rows (typical year), columns: T(h), RoR, PV, Wind
  - Values are capacity factors (0-1)

Method: seasonal mean hourly profile (4 seasons × 24 hours).
  All d1-d6 daytypes within a season share the same mean profile.
  Normalized: divided by tech-specific annual peak of the seasonal mean.

Output: 72 rows (3 techs × 4 seasons × 6 daytypes × 24h per row)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path(
    r"C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia"
    r"\EPM_Georgia\2022\1. Data\Timeseries all data.xlsx"
)
OUT_DIR = Path(__file__).parent / "output_georgia"
OUT_DIR.mkdir(exist_ok=True)

ZONE = "Georgia"
TECHS = {"ROR": "RoR", "OnshoreWind": "Wind", "PV": "PV"}  # EPM tech → column name

# ── Load ─────────────────────────────────────────────────────────────────────
print("Reading Timeseries all data.xlsx (sheet: RE data)...")
df = pd.read_excel(DATA_FILE, sheet_name="RE data")
# Expected columns: T(h), RoR, PV, Wind
print(f"  Shape: {df.shape}, columns: {list(df.columns)}")

# Rename columns if needed (flexible)
col_map = {}
for c in df.columns:
    lc = str(c).strip().lower()
    if "ror" in lc or "rur" in lc or "run" in lc: col_map[c] = "RoR"
    elif "pv" in lc or "solar" in lc:              col_map[c] = "PV"
    elif "wind" in lc:                             col_map[c] = "Wind"
    elif "t" in lc and len(str(c)) <= 5:           col_map[c] = "hour_num"
df = df.rename(columns=col_map)

# Assign datetime: T=1 = Jan 1 00:00 of a non-leap year (2021)
n = len(df)  # 8760
base = pd.Timestamp("2021-01-01 00:00")
df["datetime"] = [base + pd.Timedelta(hours=i) for i in range(n)]
df["month"] = df["datetime"].dt.month
df["hour"]  = df["datetime"].dt.hour  # 0-23

def month_to_season(m):
    if m in (1, 2, 3):  return "Q1"
    if m in (4, 5, 6):  return "Q2"
    if m in (7, 8, 9):  return "Q3"
    return "Q4"

df["season"] = df["month"].apply(month_to_season)

# ── Compute profiles ──────────────────────────────────────────────────────────
seasons  = ["Q1", "Q2", "Q3", "Q4"]
daytypes = ["d1", "d2", "d3", "d4", "d5", "d6"]

all_rows = []
for epm_tech, col in TECHS.items():
    if col not in df.columns:
        print(f"WARNING: column '{col}' not found, skipping {epm_tech}")
        continue

    # Seasonal mean by hour
    mean_profile = df.groupby(["season", "hour"])[col].mean()

    # Normalise by the peak of all seasonal means (so max = 1.0)
    peak = mean_profile.max()
    if peak == 0:
        print(f"WARNING: zero peak for {epm_tech}, skipping")
        continue
    norm_profile = mean_profile / peak

    print(f"\n{epm_tech} ({col}):")
    for s in seasons:
        vals = [norm_profile[(s, h)] for h in range(24)]
        print(f"  {s}: mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    # Build rows
    for s in seasons:
        hourly = [norm_profile[(s, h)] for h in range(24)]
        for dt in daytypes:
            row = {"zone": ZONE, "tech": epm_tech, "season": s, "daytype": dt}
            for h in range(24):
                row[f"t{h+1:02d}"] = hourly[h]
            all_rows.append(row)

# ── Write ─────────────────────────────────────────────────────────────────────
t_cols = [f"t{h:02d}" for h in range(1, 25)]
cols = ["zone", "tech", "season", "daytype"] + t_cols

out_df = pd.DataFrame(all_rows, columns=cols)
out_path = OUT_DIR / "georgia_pVREProfile.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWritten: {out_path}  ({len(out_df)} rows)")
print("Breakdown:")
print(out_df.groupby("tech").size().to_string())
