import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────
# 1. INPUT PARAMETERS
# ─────────────────────────────────────────────

YEAR = 2024
E_ANNUAL_MWH = 1500000      # Total annual consumption (MWh)
LOSS_PCT = 0.18                 # System losses (18%)
T_BASE = 20.0                   # Thermal comfort baseline (°C)
T_COOLING = 24.0                # Cooling threshold (°C)
T_HEATING = 15.0                # Heating threshold (°C)
ALPHA_C = 0.012                 # Cooling sensitivity per °C
ALPHA_H = 0.005                 # Heating sensitivity per °C (low — tropical)
BETA_S = 0.15                   # Seasonal temperature-demand sensitivity

# Monthly average temperatures (Jan–Dec), °C  — replace with ERA5/NOAA data
T_MONTHLY = np.array([
    22, 23, 25, 27, 29, 30, 28, 28, 27, 26, 24, 22
])

# Sectoral shares
SECTOR_SHARES = {
    "residential": 0.48,
    "commercial":  0.22,
    "industrial":  0.25,
    "public":      0.05,
}

# ─────────────────────────────────────────────
# 2. HOURLY LOAD SHAPES (normalized, sum = 24)
#    Source: adapted from regional benchmarks
# ─────────────────────────────────────────────

def build_hourly_shapes():
    """
    24-point normalized load profiles per sector.
    Values represent relative demand weight vs. flat average (mean ≈ 1.0).
    """

    # Residential: morning peak (7h) + evening peak (20h)
    res = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7,
        1.0, 1.3, 1.2, 1.0, 0.9, 0.9,
        1.0, 1.0, 0.9, 0.9, 1.0, 1.2,
        1.5, 1.8, 1.7, 1.4, 1.0, 0.7
    ])

    # Commercial: flat during business hours (8–18h)
    com = np.array([
        0.3, 0.3, 0.3, 0.3, 0.3, 0.4,
        0.7, 1.1, 1.4, 1.5, 1.5, 1.5,
        1.4, 1.5, 1.5, 1.5, 1.4, 1.2,
        0.9, 0.6, 0.5, 0.4, 0.4, 0.3
    ])

    # Industrial: relatively flat with slight daytime boost
    ind = np.array([
        0.8, 0.8, 0.8, 0.8, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.2, 1.2, 1.1,
        1.1, 1.2, 1.2, 1.2, 1.1, 1.1,
        1.0, 1.0, 0.9, 0.9, 0.8, 0.8
    ])

    # Public (street lighting + admin): peaks at night
    pub = np.array([
        1.2, 1.2, 1.2, 1.2, 1.0, 0.7,
        0.6, 0.7, 0.9, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.9, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.4, 1.3
    ])

    # Normalize each to mean = 1 (sum = 24)
    shapes = {
        "residential": res / res.mean(),
        "commercial":  com / com.mean(),
        "industrial":  ind / ind.mean(),
        "public":      pub / pub.mean(),
    }
    return shapes

SHAPES = build_hourly_shapes()

# ─────────────────────────────────────────────
# 3. BUILD DATETIME INDEX FOR THE FULL YEAR
# ─────────────────────────────────────────────

timestamps = pd.date_range(
    start=f"{YEAR}-01-01",
    end=f"{YEAR}-12-31 23:00",
    freq="h"
)
df = pd.DataFrame(index=timestamps)
df["month"]   = df.index.month          # 1–12
df["hour"]    = df.index.hour           # 0–23
df["dow"]     = df.index.dayofweek     # 0=Mon, 6=Sun
df["is_weekend"] = df["dow"] >= 5

# ─────────────────────────────────────────────
# 4. SEASONAL FACTOR
# ─────────────────────────────────────────────
# Seasonal Factor — captures temperature-driven and behavioral seasonality:
#where β_s is the sensitivity of demand to temperature deviation (heating or cooling), and σ_T is the standard deviation of monthly temperatures

sigma_T = T_MONTHLY.std()
T_mean  = T_MONTHLY.mean()

def seasonal_factor(month_idx):
    T_m = T_MONTHLY[month_idx - 1]
    return 1 + BETA_S * (T_m - T_mean) / (sigma_T + 1e-6)

df["seasonal"] = df["month"].apply(seasonal_factor)

# ─────────────────────────────────────────────
# 5. WEEKLY FACTOR
# ─────────────────────────────────────────────
#Weekly Factor — weekday vs. weekend depression

def weekly_factor(dow):
    if dow < 5:   return 1.00   # Weekday
    elif dow == 5: return 0.82  # Saturday
    else:          return 0.70  # Sunday

df["weekly"] = df["dow"].apply(weekly_factor)

# ─────────────────────────────────────────────
# 6. TEMPERATURE FACTOR (cooling/heating load)
# ─────────────────────────────────────────────
#Temperature Factor — cooling/heating load on top of the base:
# Interpolate monthly temperature to hourly (simple step)
# where T_cooling (~24°C) and T_heating (~15°C) are comfort thresholds, and α_c, α_h are CDD/HDD sensitivities
# (typically 0.01–0.03 per °C for developing countries with low AC penetration).

df["T_hourly"] = T_MONTHLY[df["month"].values - 1]

df["temp_factor"] = (
    1
    + ALPHA_C * np.maximum(df["T_hourly"] - T_COOLING, 0)
    + ALPHA_H * np.maximum(T_HEATING - df["T_hourly"], 0)
)

# ─────────────────────────────────────────────
# 7. COMPOSITE HOURLY LOAD SHAPE (weighted by sector)
# ─────────────────────────────────────────────

def composite_shape(hour):
    total = 0
    for sector, share in SECTOR_SHARES.items():
        total += share * SHAPES[sector][hour]
    return total

df["shape"] = df["hour"].apply(composite_shape)

# ─────────────────────────────────────────────
# 8. COMPUTE LOAD (MW)
# ─────────────────────────────────────────────

# Average flat load (MW) from annual energy target (net of losses)
P_flat = E_ANNUAL_MWH / 8760  # MW (net demand)

df["load_MW"] = (
    P_flat
    * df["seasonal"]
    * df["weekly"]
    * df["shape"]
    * df["temp_factor"]
    * (1 + LOSS_PCT)
)

# ─────────────────────────────────────────────
# 9. CALIBRATION — rescale to preserve E_annual
# ─────────────────────────────────────────────
# The multiplicative factors may shift the integral;
# we rescale to ensure sum(load_MW) = E_annual (gross)

E_gross_target = E_ANNUAL_MWH * (1 + LOSS_PCT)
E_computed = df["load_MW"].sum()          # MWh (each row = 1h)
calib_factor = E_gross_target / E_computed

df["load_MW"] *= calib_factor

# ─────────────────────────────────────────────
# 10. OUTPUT SUMMARY
# ─────────────────────────────────────────────

print("=" * 45)
print(f"  YEARLY LOAD CURVE SUMMARY — {YEAR}")
print("=" * 45)
print(f"  Peak demand         : {df['load_MW'].max():>10.1f} MW")
print(f"  Min demand          : {df['load_MW'].min():>10.1f} MW")
print(f"  Average demand      : {df['load_MW'].mean():>10.1f} MW")
print(f"  Total gross energy  : {df['load_MW'].sum():>10,.0f} MWh")
print(f"  Net energy (target) : {E_ANNUAL_MWH:>10,.0f} MWh")
print(f"  Load factor         : {df['load_MW'].mean()/df['load_MW'].max():>10.2%}")
print(f"  Calibration factor  : {calib_factor:>10.4f}")

# ─────────────────────────────────────────────
# 11. VISUALIZATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Full year load curve
axes[0].plot(df.index, df["load_MW"], linewidth=0.5, color="steelblue")
axes[0].set_title("Yearly Load Curve (8,760 Hours)")
axes[0].set_ylabel("Load (MW)")
#axes[0].set_xlabel("Date")
axes[0].grid(True, alpha=0.3)

# Monthly average daily profile
monthly_profiles = df.groupby(["month", "hour"])["load_MW"].mean().unstack(level=0)
cmap = plt.cm.get_cmap("RdYlBu_r", 12)
for m in range(1, 13):
    axes[1].plot(range(24), monthly_profiles[m],
                 label=pd.Timestamp(f"2024-{m:02d}-01").strftime("%b"),
                 color=cmap(m - 1))
axes[1].set_title("Average Daily Profile by Month")
axes[1].set_ylabel("Load (MW)")
#axes[1].set_xlabel("Hour of Day")
axes[1].legend(ncol=6, fontsize=8)
axes[1].grid(True, alpha=0.3)

# Monthly energy (bar chart)
monthly_energy = df.groupby("month")["load_MW"].sum() / 1000  # GWh
axes[2].bar(range(1, 13), monthly_energy,
            color="steelblue", edgecolor="white")
axes[2].set_title("Monthly Energy Consumption (Gross)")
axes[2].set_ylabel("Energy (GWh)")
#axes[2].set_xlabel("Month")
axes[2].set_xticks(range(1, 13))
axes[2].set_xticklabels([
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
])
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("yearly_load_curve.png", dpi=150, bbox_inches="tight")
plt.show()