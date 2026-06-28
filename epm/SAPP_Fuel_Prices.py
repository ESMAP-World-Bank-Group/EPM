"""
SAPP Fuel Price Projection Generator
Produces: SAPP_Fuel_Price_Projections_2025_2060.csv
Rows: 12 countries x 7 fuels x 36 years = 3,024 rows
Columns: Country, Fuel, Year,
         Price_Reference_USD_GJ,
         Price_HighFossil_USD_GJ,
         Price_AccelTransition_USD_GJ

Prices are in real 2025 USD/GJ.
Yearly values are linearly interpolated between 5-year anchor points.
Country-specific adjustments are applied for logistics, domestic coal,
and pipeline vs LNG gas access.

Sources: World Bank Commodity Markets Outlook (2025-2026),
         IEA World Energy Outlook 2024 (STEPS, APS, NZE scenarios),
         IRENA Africa 2030/2050, IEA Biogas report 2020.
"""

import csv
import os

# ── COUNTRIES ────────────────────────────────────────────────────────────────
COUNTRIES = [
    "Angola", "Botswana", "DRC", "Eswatini", "Lesotho", "Malawi",
    "Mozambique", "Namibia", "South Africa", "Tanzania", "Zambia", "Zimbabwe"
]

# ── FUELS ────────────────────────────────────────────────────────────────────
FUELS = ["HFO", "Diesel", "Coal", "Natural Gas", "LNG", "Biomass", "Biogas"]

# ── ANCHOR YEARS ─────────────────────────────────────────────────────────────
ANCHOR_YEARS = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]

# ── BASE PRICES AT ANCHOR YEARS (USD/GJ, real 2025) ─────────────────────────
# Three scenarios: Reference, High Fossil, Accelerated Transition
# These represent SAPP coastal/port-adjacent delivered prices (before country adj.)
# Coal = RBCT export benchmark (non-SA); SA uses separate domestic prices below.
# Natural Gas = pipeline benchmark for gas-connected countries.
# LNG = delivered including regasification ($1.5-2.5/GJ infrastructure cost).


BASE_PRICES = {
    "HFO": {
        "Reference":              [11.50, 12.50, 13.00, 13.50, 13.50, 13.00, 12.50, 12.00],
        "High Fossil":            [13.50, 15.50, 17.00, 18.00, 18.50, 18.00, 17.50, 17.00],
        "Accelerated Transition": [10.50,  9.50,  8.50,  7.00,  5.50,  4.50,  4.00,  3.50],
    },
    "Diesel": {
        "Reference":              [15.00, 16.00, 16.50, 17.00, 17.00, 16.50, 16.00, 15.50],
        "High Fossil":            [17.50, 20.00, 22.00, 23.50, 24.00, 23.00, 22.00, 21.00],
        "Accelerated Transition": [14.00, 12.50, 11.00,  9.00,  7.50,  6.00,  5.50,  5.00],
    },
    "Coal": {
        # RBCT export benchmark; non-SA countries add logistics below
        "Reference":              [ 4.20,  4.00,  3.80,  3.50,  3.00,  2.50,  2.00,  1.80],
        "High Fossil":            [ 5.00,  5.50,  5.80,  6.00,  5.50,  5.00,  4.50,  4.00],
        "Accelerated Transition": [ 3.80,  3.20,  2.50,  2.00,  1.60,  1.30,  1.00,  0.80],
    },
    "Natural Gas": {
        # Pipeline delivered prices for gas-connected countries
        # Countries without pipeline access use LNG as proxy (see logic below)
        "Reference":              [ 6.50,  7.00,  7.50,  7.50,  7.50,  7.00,  6.50,  6.00],
        "High Fossil":            [ 7.50,  9.00, 10.50, 11.00, 11.00, 10.50, 10.00,  9.50],
        "Accelerated Transition": [ 6.00,  5.50,  5.00,  4.50,  4.00,  3.50,  3.00,  2.50],
    },
    "LNG": {
        # Delivered to FSRU / regasification terminal
        "Reference":              [10.50, 11.00, 11.50, 11.50, 11.00, 10.50, 10.00,  9.50],
        "High Fossil":            [13.00, 15.00, 17.00, 17.50, 17.00, 16.00, 15.00, 14.00],
        "Accelerated Transition": [ 9.50,  8.50,  7.50,  6.50,  5.50,  4.50,  4.00,  3.50],
    },
    "Biomass": {
        # Delivered feedstock cost to plant gate (residues, forestry, energy crops)
        "Reference":              [ 3.50,  4.00,  4.50,  5.00,  5.50,  6.00,  6.50,  7.00],
        "High Fossil":            [ 3.80,  4.50,  5.00,  5.50,  6.00,  6.80,  7.50,  8.00],
        "Accelerated Transition": [ 3.20,  3.50,  3.80,  4.00,  4.20,  4.50,  4.80,  5.00],
    },
    "Biogas": {
        # Levelized production cost (feedstock + capture + digestion)
        "Reference":              [ 8.00,  7.00,  6.00,  5.50,  5.00,  4.50,  4.20,  4.00],
        "High Fossil":            [ 8.50,  7.50,  6.50,  6.00,  5.50,  5.00,  4.80,  4.50],
        "Accelerated Transition": [ 7.50,  6.00,  5.00,  4.50,  4.00,  3.50,  3.20,  3.00],
    },
}

# ── SOUTH AFRICA DOMESTIC COAL (much cheaper than RBCT export) ───────────────
# Reflects mine-mouth + Eskom tied contracts; rising due to deeper extraction.
SA_COAL_DOMESTIC = {
    "Reference":              [1.80, 2.00, 2.20, 2.30, 2.20, 2.00, 1.80, 1.50],
    "High Fossil":            [1.80, 2.00, 2.20, 2.30, 2.20, 2.00, 1.80, 1.50],
    "Accelerated Transition": [1.80, 2.00, 2.20, 2.00, 1.80, 1.50, 1.20, 1.00],
}

# ── LOGISTICS ADJUSTMENT (additive $/GJ) applied to HFO and Diesel ───────────
# Reflects inland transport distance from coastal ports.
# Coastal / port-adjacent = 0; landlocked = +3.0 to +3.5
LOGISTICS_ADJ = {
    "Angola":       0.0,   # Luanda port; coastal
    "Botswana":     3.0,   # Fully landlocked; road from Durban or Beira
    "DRC":          2.0,   # Partial river access but remote generation sites
    "Eswatini":     3.0,   # Landlocked; road from Maputo or Durban
    "Lesotho":      3.5,   # Fully landlocked; mountainous terrain
    "Malawi":       3.5,   # Landlocked; Nacala or Beira port access
    "Mozambique":   0.0,   # Beira, Maputo, Nacala ports; coastal
    "Namibia":      0.5,   # Walvis Bay port; minor inland logistics
    "South Africa": 0.0,   # Durban, Cape Town ports; domestic refinery
    "Tanzania":     0.5,   # Dar es Salaam port; minor inland logistics
    "Zambia":       3.0,   # Landlocked; Dar es Salaam or Durban corridor
    "Zimbabwe":     3.0,   # Landlocked; Beira or Durban corridor
}

# ── COAL LOGISTICS ADJUSTMENT (additive $/GJ for non-SA importers) ───────────
COAL_LOGISTICS_ADJ = {
    "Angola":       1.5,
    "Botswana":     0.5,   # Close to SA mines; Morupule domestic also available
    "DRC":          2.0,
    "Eswatini":     0.5,
    "Lesotho":      0.5,
    "Malawi":       2.0,
    "Mozambique":   0.5,   # Moatize mine domestic production
    "Namibia":      1.5,
    "Tanzania":     2.0,
    "Zambia":       1.0,
    "Zimbabwe":     0.5,   # Hwange domestic coal available
}

# ── PIPELINE GAS ACCESS ───────────────────────────────────────────────────────
# Countries with domestic pipeline gas access get Natural Gas base prices.
# Others use LNG prices as proxy for Natural Gas (no pipeline = must import LNG).
PIPELINE_GAS_COUNTRIES = {
    "Angola",       # Associated gas (partial power sector use)
    "Mozambique",   # Rovuma basin; major domestic resource
    "Tanzania",     # Songosongo/Mnazi Bay domestic gas
    "South Africa", # ROMPCO Mozambique pipeline + potential Rovuma imports
    "Zimbabwe",     # Potential pipeline access from Mozambique (limited)
    "Zambia",       # Potential future pipeline; currently minimal
}

# Countries NOT in PIPELINE_GAS_COUNTRIES get LNG prices for Natural Gas column.
# This reflects that without pipeline infrastructure, gas supply = LNG import.

# ── HELPER: LINEAR INTERPOLATION ─────────────────────────────────────────────
def interpolate(anchor_values, year):
    """Linearly interpolate between 5-year anchor points."""
    for i in range(len(ANCHOR_YEARS) - 1):
        y1, y2 = ANCHOR_YEARS[i], ANCHOR_YEARS[i + 1]
        if y1 <= year <= y2:
            t = (year - y1) / (y2 - y1)
            return round(anchor_values[i] + t * (anchor_values[i + 1] - anchor_values[i]), 2)
    return None

# ── SCENARIO COLUMN NAMES ────────────────────────────────────────────────────
SCENARIOS = ["Reference", "High Fossil", "Accelerated Transition"]
COL_NAMES = [
    "Price_Reference_USD_GJ",
    "Price_HighFossil_USD_GJ",
    "Price_AccelTransition_USD_GJ"
]

# ── MAIN: GENERATE ROWS ──────────────────────────────────────────────────────
def main():
    output_file = "SAPP_Fuel_Price_Projections_2025_2060.csv"
    rows = []

    for country in COUNTRIES:
        for fuel in FUELS:
            for year in range(2025, 2061):
                prices = []
                for scenario in SCENARIOS:
                    # --- Determine base price ---
                    if fuel == "Coal" and country == "South Africa":
                        # Use domestic mine-mouth pricing
                        price = interpolate(SA_COAL_DOMESTIC[scenario], year)

                    elif fuel == "Natural Gas" and country not in PIPELINE_GAS_COUNTRIES:
                        # No pipeline access: use LNG as proxy for gas price
                        price = interpolate(BASE_PRICES["LNG"][scenario], year)

                    else:
                        price = interpolate(BASE_PRICES[fuel][scenario], year)

                    # --- Apply country-specific adjustments ---
                    if fuel in ["HFO", "Diesel"]:
                        price = round(price + LOGISTICS_ADJ[country], 2)

                    elif fuel == "Coal" and country != "South Africa":
                        price = round(price + COAL_LOGISTICS_ADJ.get(country, 1.0), 2)

                    prices.append(f"{price:.2f}")

                rows.append([country, fuel, str(year)] + prices)

    # --- Write CSV ---
    header = ["Country", "Fuel", "Year"] + COL_NAMES
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows written to: {os.path.abspath(output_file)}")
    print(f"Columns: {', '.join(header)}")

if __name__ == "__main__":
    main()
