import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

INPUT_FILE    = "sapp_load_2024.xlsx"  # adjust sep below if comma-separated
SEP           = "\t"
K             = 4                     # representative days per season
RANDOM_SEED   = 42
TARGET_DAYS   = 365                   # must sum to this (= 8760 / 24)
HOURS_PER_DAY = 24

COUNTRIES = [
    "Eswatini", "Lesotho", "Malawi", "Mozambique", "Namibia",
    "Zimbabwe", "Zambia", "Tanzania", "South Africa", "Angola", "Botswana"
]

SEASONS = ["Q1", "Q2", "Q3", "Q4"]

# ─────────────────────────────────────────────────────────────────────
# SUB-REGION CONFIGURATION
#
# "share"  - fraction of the parent country peak load for this zone
#            (must sum to 1.0 across sub-regions of a country)
# "shape"  - 24-element array of hourly multipliers applied on top of
#            the national aggregate profile to capture sub-regional
#            load shape differences
# ─────────────────────────────────────────────────────────────────────

SUBREGION_CONFIG = {

    "Mozambique": {
        "Mozambique_South": {
            "share": 0.55,
            # Urban/commercial Maputo: strong morning ramp, afternoon peak,
            # residential evening secondary peak. Interconnected with SA.
            "shape": np.array([
                0.90, 0.87, 0.85, 0.84, 0.84, 0.86,
                1.05, 1.08, 1.10, 1.10, 1.08, 1.06,
                1.02, 1.05, 1.07, 1.08, 1.10, 1.10,
                1.08, 1.05, 1.02, 0.98, 0.94, 0.90
            ])
        },
        "Mozambique_Central": {
            "share": 0.28,
            # Mining base load (Tete/Moatize) flattens profile;
            # Beira commercial adds moderate afternoon peak.
            "shape": np.array([
                0.95, 0.93, 0.92, 0.91, 0.91, 0.92,
                0.98, 1.00, 1.02, 1.03, 1.03, 1.02,
                1.01, 1.02, 1.03, 1.04, 1.04, 1.05,
                1.04, 1.02, 1.00, 0.98, 0.97, 0.96
            ])
        },
        "Mozambique_North": {
            "share": 0.17,
            # Residential/rural dominant: lower daytime load factor,
            # stronger evening peak driven by lighting demand.
            "shape": np.array([
                0.92, 0.90, 0.89, 0.88, 0.88, 0.90,
                0.94, 0.96, 0.97, 0.98, 0.98, 0.97,
                0.96, 0.96, 0.97, 0.97, 0.98, 1.02,
                1.08, 1.10, 1.08, 1.02, 0.97, 0.93
            ])
        },
    },

    "Angola": {
        "Angola_North": {
            "share": 0.63,
            # Luanda: hot climate drives AC load; commercial midday peak,
            # strong late-evening residential peak (t19-t22).
            "shape": np.array([
                0.85, 0.82, 0.80, 0.79, 0.79, 0.81,
                0.90, 0.98, 1.05, 1.08, 1.08, 1.06,
                1.04, 1.05, 1.06, 1.06, 1.07, 1.08,
                1.10, 1.10, 1.08, 1.05, 0.98, 0.90
            ])
        },
        "Angola_Central": {
            "share": 0.13,
            # Huambo/Bie: mixed residential/commercial, moderate climate,
            # close to national average shape.
            "shape": np.array([
                0.94, 0.93, 0.92, 0.91, 0.91, 0.93,
                0.98, 1.01, 1.03, 1.04, 1.04, 1.03,
                1.02, 1.02, 1.03, 1.03, 1.04, 1.05,
                1.05, 1.04, 1.02, 1.00, 0.97, 0.95
            ])
        },
        "Angola_South": {
            "share": 0.12,
            # Lubango/Namibe: residential + fishing industry.
            # Cooler climate (altitude), prominent evening peak.
            "shape": np.array([
                0.93, 0.91, 0.90, 0.89, 0.89, 0.91,
                0.97, 1.00, 1.02, 1.03, 1.03, 1.02,
                1.01, 1.01, 1.02, 1.02, 1.03, 1.05,
                1.08, 1.10, 1.08, 1.03, 0.98, 0.94
            ])
        },
        "Angola_East": {
            "share": 0.12,
            # Catoca diamond mine: continuous 24/7 industrial base load.
            # Profile is near-flat with minimal day/night variation.
            "shape": np.array([
                1.02, 1.02, 1.02, 1.02, 1.02, 1.02,
                1.02, 1.02, 1.02, 1.02, 1.02, 1.02,
                1.01, 1.01, 1.01, 1.01, 1.01, 1.01,
                1.00, 1.00, 1.00, 1.00, 1.01, 1.02
            ])
        },
    },
}

# Countries passed through without sub-regional split
COUNTRIES_SIMPLE = [c for c in COUNTRIES if c not in SUBREGION_CONFIG]

# ─────────────────────────────────────────────────────────────────────
# STEP 1 - Load data
# ─────────────────────────────────────────────────────────────────────

df = pd.read_excel(INPUT_FILE)

# ─────────────────────────────────────────────────────────────────────
# STEP 2 - Season assignment and day key
# ─────────────────────────────────────────────────────────────────────

def month_to_season(m):
    if   m in [1, 2, 3]:  return "Q1"
    elif m in [4, 5, 6]:  return "Q2"
    elif m in [7, 8, 9]:  return "Q3"
    else:                  return "Q4"

df["Season"] = df["Month"].apply(month_to_season)
df["DayKey"] = df["Month"].astype(str) + "_" + df["Month-Day"].astype(str)

assert (df.groupby("DayKey").size() == 24).all(), \
    "ERROR: Some days do not have exactly 24 hourly rows. Check input data."

print("Input data loaded successfully.")
print(f"  Total hours : {len(df)}")
print(f"  Total days  : {df['DayKey'].nunique()}")
for s in SEASONS:
    n = df[df["Season"] == s]["DayKey"].nunique()
    print(f"  {s}         : {n} days")

# ─────────────────────────────────────────────────────────────────────
# STEP 3 - Weighted aggregate clustering signal
#
# Each country is weighted by its share of total annual mean load so
# that larger systems contribute proportionally more to the signal.
# ─────────────────────────────────────────────────────────────────────

country_mean_load = df[COUNTRIES].mean()
country_weights   = country_mean_load / country_mean_load.sum()

print("\n=== Clustering weights (proportional to annual mean load) ===")
for c, w in country_weights.items():
    print(f"  {c:<16}: {w:.4f}  (mean = {country_mean_load[c]:.1f} MW)")

df["Aggregate"] = (df[COUNTRIES] * country_weights.values).sum(axis=1)

# ─────────────────────────────────────────────────────────────────────
# STEP 4 - Annual peak per country (used for profile normalisation)
# ─────────────────────────────────────────────────────────────────────

annual_peak = df[COUNTRIES].max()

print("\n=== Annual peak load per country (MW) ===")
for c, p in annual_peak.items():
    print(f"  {c:<16}: {p:.1f} MW")

# ─────────────────────────────────────────────────────────────────────
# STEP 5 - K-means clustering and profile construction
# ─────────────────────────────────────────────────────────────────────

hour_cols = [f"t{h}" for h in range(1, 25)]

raw_weight_rows = []   # raw cluster sizes (days), rescaled in Step 6
profile_rows    = []   # demand profiles

for season in SEASONS:
    df_s     = df[df["Season"] == season].copy()
    day_keys = df_s["DayKey"].unique()
    n_days   = len(day_keys)

    # Build (n_days x 24) matrix for the weighted aggregate signal
    agg_matrix = np.zeros((n_days, 24))
    for i, dk in enumerate(day_keys):
        day_data = df_s[df_s["DayKey"] == dk].sort_values("Hour-day")
        agg_matrix[i, :] = day_data["Aggregate"].values

    # K-means clustering
    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=20)
    km.fit(agg_matrix)
    labels = km.labels_

    # Sort clusters ascending by mean aggregate load: d1=low, d4=high
    cluster_means = [agg_matrix[labels == k].mean() for k in range(K)]
    order         = np.argsort(cluster_means)

    for rank, cluster_idx in enumerate(order):
        rep_label    = f"d{rank + 1}"
        cluster_mask = labels == cluster_idx
        raw_weight   = int(cluster_mask.sum())

        raw_weight_rows.append({
            "q":      season,
            "d":      rep_label,
            "weight": raw_weight
        })

        cluster_day_keys = day_keys[cluster_mask]

        # Helper: mean hourly profile for a country over the cluster days
        def cluster_mean_profile(country):
            mat = np.zeros((cluster_mask.sum(), 24))
            for i, dk in enumerate(cluster_day_keys):
                day_data = df_s[df_s["DayKey"] == dk].sort_values("Hour-day")
                mat[i, :] = day_data[country].values
            return mat.mean(axis=0)

        # -- Simple (unsplit) countries -----------------------------------
        for country in COUNTRIES_SIMPLE:
            rep_profile = cluster_mean_profile(country)
            normalised  = rep_profile / annual_peak[country]

            p_row = {"Zone": country, "m": season, "d": rep_label}
            for h in range(24):
                p_row[f"t{h + 1}"] = round(float(normalised[h]), 2)
            profile_rows.append(p_row)

        # -- Split countries (Mozambique, Angola) -------------------------
        for parent, subregions in SUBREGION_CONFIG.items():
            national_profile = cluster_mean_profile(parent)

            for sr_name, sr_cfg in subregions.items():
                share = sr_cfg["share"]
                shape = sr_cfg["shape"]

                sr_profile  = national_profile * share * shape
                sr_peak_ref = annual_peak[parent] * share
                normalised  = sr_profile / sr_peak_ref

                p_row = {"Zone": sr_name, "m": season, "d": rep_label}
                for h in range(24):
                    p_row[f"t{h + 1}"] = round(float(normalised[h]), 2)
                profile_rows.append(p_row)

# ─────────────────────────────────────────────────────────────────────
# STEP 6 - Rescale weights to integer days summing to TARGET_DAYS (365)
#
# Method: Largest Remainder (Hamilton) applied globally across all
# 16 weight entries (4 seasons x 4 days).
#
# 1. Compute each weight's exact proportional share of TARGET_DAYS.
# 2. Take the floor of each.
# 3. Distribute the remaining days (TARGET_DAYS - sum of floors) one
#    by one to the entries with the largest fractional remainders.
#
# This guarantees:
#   - All weights are positive integers >= 1
#   - Sum of all weights = TARGET_DAYS = 365
#   - Grand total hours = 365 * 24 = 8760
#   - Intra-season proportions are respected as closely as possible
# ─────────────────────────────────────────────────────────────────────

df_raw    = pd.DataFrame(raw_weight_rows)   # columns: q, d, weight
total_raw = df_raw["weight"].sum()

# Exact fractional target for each entry
df_raw["exact"]     = df_raw["weight"] / total_raw * TARGET_DAYS
df_raw["floor"]     = df_raw["exact"].apply(np.floor).astype(int)
df_raw["remainder"] = df_raw["exact"] - df_raw["floor"]

# How many extra days need to be distributed
remainder_days = TARGET_DAYS - df_raw["floor"].sum()

# Assign extra days to entries with largest remainders
top_idx = df_raw["remainder"].nlargest(int(remainder_days)).index
df_raw.loc[top_idx, "floor"] += 1

# Final integer weights
df_raw["int_weight"] = df_raw["floor"]

assert df_raw["int_weight"].sum() == TARGET_DAYS, \
    f"Weight sum {df_raw['int_weight'].sum()} != {TARGET_DAYS}. Check rounding logic."
assert (df_raw["int_weight"] >= 1).all(), \
    "One or more weights rounded down to zero. Increase K or check data."

# ─────────────────────────────────────────────────────────────────────
# STEP 7 - Build output DataFrames
# ─────────────────────────────────────────────────────────────────────

scaled_rows = []
for _, row in df_raw.iterrows():
    w = int(row["int_weight"])
    w_row = {"q": row["q"], "d": row["d"], "weight": w}
    for h in range(1, 25):
        w_row[f"t{h}"] = w
    scaled_rows.append(w_row)

weight_cols = ["q", "d", "weight"] + hour_cols
df_weights  = pd.DataFrame(scaled_rows, columns=weight_cols)

profile_cols = ["Zone", "m", "d"] + hour_cols
df_profiles  = pd.DataFrame(profile_rows)[profile_cols]

# ─────────────────────────────────────────────────────────────────────
# STEP 8 - Verification
# ─────────────────────────────────────────────────────────────────────

print("\n=== Weight table (integer weights) ===")
print(df_weights[["q", "d", "weight"]].to_string(index=False))

print("\n=== Verification ===")
total_days  = df_weights["weight"].sum()
total_hours = total_days * HOURS_PER_DAY
print(f"  Total days  : {total_days}  (target = {TARGET_DAYS})")
print(f"  Total hours : {total_hours}  (target = {TARGET_DAYS * HOURS_PER_DAY})")
for season in SEASONS:
    s_days  = df_weights[df_weights["q"] == season]["weight"].sum()
    s_hours = s_days * HOURS_PER_DAY
    print(f"  {season}: {s_days} days  ({s_hours} hours)")

print("\n=== Output zones ===")
for z in df_profiles["Zone"].unique():
    print(f"  {z}")

print("\n=== Sample: Mozambique sub-regions, Q1 ===")
mask = (df_profiles["Zone"].str.startswith("Mozambique")) & (df_profiles["m"] == "Q1")
print(df_profiles[mask].to_string(index=False))

print("\n=== Sample: Angola sub-regions, Q1 ===")
mask = (df_profiles["Zone"].str.startswith("Angola")) & (df_profiles["m"] == "Q1")
print(df_profiles[mask].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# STEP 9 - Export
# ─────────────────────────────────────────────────────────────────────

df_weights.to_csv("representative_days_weights.csv", index=False)
df_profiles.to_csv("representative_days_demand_profiles.csv", index=False)

print("\nFiles saved:")
print("  representative_days_weights.csv          — integer weight table")
print("  representative_days_demand_profiles.csv  — normalised profiles (0-1)")
print(f"\nTotal zones : {df_profiles['Zone'].nunique()}")
print(f"Profile rows: {len(df_profiles)}")
