"""
prepare_romania_for_blacksea.py
Cleans up data_romania/ extractions and produces blacksea-compatible CSVs
ready to append into data_blacksea/.

Outputs (in data_romania/supply/):
  pGenDataInput_blacksea.csv   — column-aligned, fuel-normalised
  pFuelPrice_blacksea.csv      — country=Romania, fuels in blacksea set
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT / "epm/input/data_romania"
OUT_DIR  = SRC_DIR / "supply"

# ── column schema expected by data_blacksea pGenDataInput ─────────────────────
BS_COLS = [
    "g", "z", "tech", "f", "Status", "StYr", "RetrYr", "Capacity",
    "BuildLimitperYear", "Life", "HeatRate", "RampUpRate", "RampDnRate",
    "ResLimShare", "Capex", "FOMperMW", "VOM", "ReserveCost", "UnitSize",
]

# ── fuel name normalisation ────────────────────────────────────────────────────
# Romania xlsb uses Coal (lignite), WindLow*/Med*, GasCCS, Hydrogen, etc.
FUEL_MAP = {
    "Coal":     "DomesticCoal",   # domestic lignite
    "CoalCCS":  "DomesticCoal",   # treat CCS coal as DomesticCoal (simplify)
    "WindLow1": "Wind",
    "WindLow2": "Wind",
    "WindLow3": "Wind",
    "WindMed1": "Wind",
    "WindMed2": "Wind",
    # WindMed3 / WindHigh* already labelled "Wind" in xlsb → no entry needed
    "GasCCS":   "Gas",            # simplify: no separate CCS fuel
}

# ── generators to drop (incompatible fuels / placeholder rows) ─────────────────
DROP_NAMES = {
    # Hydrogen tech candidates — no H2 fuel in data_blacksea
    "Generic OCGT H2",
    "Generic CCGT H2",
    # Geothermal — no Geothermal fuel
    "Generic Geothermal",
    # CCS retro on Brazi — speculative 2070, GasCCS
    "TPP CCCC Petrom Brazi_Retrofited",
    # 1-axis tracker PV — not modelled separately
    "Generic Solar 1X",
    # Aggregate coal placeholder without documented capacity
    "R-Coal",
    # CCS gas candidates — will be replaced by simplified Gas candidates
    "Generic OCGT + CCS",
    "Generic CCGT + CCS",
    # Wind resource-class generics — replaced by 2 consolidated entries below
    "Generic Onshore WindLow1",
    "Generic Onshore WindLow2",
    "Generic Onshore WindLow3",
    "Generic Onshore WindMed1",
    "Generic Onshore WindMed2",
    "Generic Onshore WindMed3",
    "Generic Onshore WindHigh1",
    "Generic Onshore WindHigh2_2026",
    "Generic Onshore WindHigh2_2027",
    "Generic Offshore Wind Fixed",
    "Generic Offshore Wind Floating",
}

# ── pGenDataInput ──────────────────────────────────────────────────────────────

def clean_gendata() -> pd.DataFrame:
    src = SRC_DIR / "supply/pGenDataInput.csv"
    df  = pd.read_csv(src, dtype=str)

    # Fix TPP Mintia: blank Capacity → 495 MW (2 remaining units as of 2024)
    df.loc[df["g"] == "TPP Mintia", "Capacity"] = "495.0"

    # Normalise fuel names
    df["f"] = df["f"].map(lambda x: FUEL_MAP.get(x, x))

    # Drop incompatible / placeholder generators
    df = df[~df["g"].isin(DROP_NAMES)].copy()

    # Consolidated onshore wind candidate (replaces 9 resource-class generics)
    # Total Romania onshore potential ≈ 120 GW; build limit 2 GW/yr from WB EPM
    onshore = {
        "g": "Generic Onshore Wind Romania", "z": "Romania",
        "tech": "OnshoreWind", "f": "Wind", "Status": "3",
        "StYr": "2025", "RetrYr": "2055", "Life": "30",
        "Capacity": "120000.0", "BuildLimitperYear": "2000.0",
        "Capex": "1.3", "FOMperMW": "30000.0", "VOM": "",
        "ReserveCost": "5.0", "HeatRate": "",
        "RampUpRate": "1.0", "RampDnRate": "1.0", "ResLimShare": "0.0",
        "UnitSize": "", "Linked plants": "", "CapacityMWh": "",
        "Efficiency": "", "CapexMWh": "",
    }

    # Consolidated offshore wind candidate (Black Sea fixed + floating merged)
    offshore = {
        "g": "Generic Offshore Wind Romania", "z": "Romania",
        "tech": "OnshoreWind", "f": "Wind", "Status": "3",
        "StYr": "2028", "RetrYr": "2058", "Life": "30",
        "Capacity": "94000.0", "BuildLimitperYear": "500.0",
        "Capex": "3.0", "FOMperMW": "60000.0", "VOM": "",
        "ReserveCost": "5.0", "HeatRate": "",
        "RampUpRate": "1.0", "RampDnRate": "1.0", "ResLimShare": "0.0",
        "UnitSize": "", "Linked plants": "", "CapacityMWh": "",
        "Efficiency": "", "CapexMWh": "",
    }

    extras = pd.DataFrame([onshore, offshore])
    df = pd.concat([df, extras], ignore_index=True)

    # Reorder / select columns to match data_blacksea schema
    df_out = df.reindex(columns=BS_COLS)
    return df_out


# ── pFuelPrice ─────────────────────────────────────────────────────────────────

def clean_fuelprice() -> pd.DataFrame:
    """
    Take the xlsb-extracted pFuelPrice, drop unsupported fuels, rename columns
    to match data_blacksea (country, fuel), and add a Biomass row.
    Keeps: Gas, DomesticCoal (from 'Coal'), Uranium.
    Drops: CoalCCS, Hydrogen, H2domestic, GasCCS.
    """
    src  = SRC_DIR / "supply/pFuelPrice.csv"
    raw  = pd.read_csv(src, dtype=str)

    # raw columns: z, f, 2024, 2025, ...
    raw = raw.rename(columns={"z": "country", "f": "fuel"})

    # Drop unsupported / simplified-away fuels
    drop_fuels = {"CoalCCS", "Hydrogen", "H2domestic", "GasCCS"}
    raw = raw[~raw["fuel"].isin(drop_fuels)].copy()

    # Rename Coal → DomesticCoal (lignite)
    raw["fuel"] = raw["fuel"].map(lambda x: FUEL_MAP.get(x, x))

    # Add Biomass row (European biomass pellet ~5 USD/GJ, constant)
    years = [c for c in raw.columns if c.isdigit()]
    biomass_row = {"country": "Romania", "fuel": "Biomass"}
    biomass_row.update({y: "5.0" for y in years})
    raw = pd.concat([raw, pd.DataFrame([biomass_row])], ignore_index=True)

    return raw


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    gd = clean_gendata()
    gd_path = OUT_DIR / "pGenDataInput_blacksea.csv"
    gd.to_csv(gd_path, index=False)
    print(f"[gendata]  wrote {len(gd)} rows → {gd_path.relative_to(ROOT)}")

    # Summary by fuel/tech/status
    summary = (
        gd.assign(Status=gd["Status"].astype(str))
          .groupby(["Status", "tech", "f"])
          .agg(n=("g","count"), cap=("Capacity", lambda x: pd.to_numeric(x, errors="coerce").sum()))
          .reset_index()
    )
    print(summary.to_string(index=False))

    fp = clean_fuelprice()
    fp_path = OUT_DIR / "pFuelPrice_blacksea.csv"
    fp.to_csv(fp_path, index=False)
    print(f"\n[fuelprice] wrote {len(fp)} rows → {fp_path.relative_to(ROOT)}")
    print(fp[["country","fuel"]].to_string(index=False))


if __name__ == "__main__":
    main()
