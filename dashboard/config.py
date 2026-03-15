"""
EPM Dashboard — Central configuration
Paths, color maps, fuel order, theme constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.parent          # EPM/
EPM_DIR     = REPO_ROOT / "epm"
OUTPUT_ROOT = EPM_DIR / "output"
INPUT_ROOT  = EPM_DIR / "input"
RESOURCES   = EPM_DIR / "resources" / "postprocess"

# ---------------------------------------------------------------------------
# Dash / Bootstrap theme
# ---------------------------------------------------------------------------
# Options: FLATLY (clean blue), COSMO (modern), SOLAR (dark), CYBORG (dark)
THEME_NAME = "FLATLY"

# Navbar brand
APP_TITLE   = "EPM — Electricity Planning Model"
APP_LOGO    = "/assets/logo.png"   # place logo.png in dashboard/assets/

# ---------------------------------------------------------------------------
# Fuel color map  (sourced from epm/resources/postprocess/colors.csv)
# ---------------------------------------------------------------------------
FUEL_COLORS = {
    "Coal":         "#696969",   # dimgray
    "Gas":          "#B22222",   # firebrick
    "LNG":          "#F08080",   # lightcoral
    "Water":        "#00008B",   # darkblue
    "Solar":        "#FFA500",   # orange
    "Wind":         "#00CED1",   # darkturquoise
    "Import":       "#D8BFD8",   # thistle
    "HFO":          "#8B4513",   # saddlebrown
    "LFO":          "#F4A460",   # sandybrown
    "Uranium":      "#800080",   # purple
    "CSP":          "#FF8C00",   # darkorange
    "Battery":      "#BDB76B",   # darkkhaki
    "Diesel":       "#E9967A",   # darksalmon
    "Biomass":      "#008000",   # green
    "Geothermal":   "#FF0000",   # red
    "StorageEnergy":"#006400",   # darkgreen
    "Other":        "#A9A9A9",   # darkgray fallback
}

# Cost category colors
COST_COLORS = {
    "Annualized capex: $m":        "#00008B",
    "Fixed O&M: $m":               "#008000",
    "Variable O&M: $m":            "#BC8F8F",
    "Fuel cost: $m":               "#808080",
    "Spinning Reserve costs: $m":  "#FFD700",
    "Unmet demand cost: $m":       "#B22222",
    "Carbon costs: $m":            "#006400",
    "Trade costs: $m":             "#DDA0DD",
}

# ---------------------------------------------------------------------------
# Fuel stacking order for bar/area charts (bottom → top)
# ---------------------------------------------------------------------------
FUEL_ORDER = [
    "Coal", "Uranium", "HFO", "LFO", "Diesel",
    "Gas", "LNG",
    "Water",
    "Geothermal", "Biomass",
    "CSP", "Solar", "Wind",
    "Battery", "StorageEnergy",
    "Import", "Other",
]

# ---------------------------------------------------------------------------
# Renewable fuels (for RE % calculation)
# ---------------------------------------------------------------------------
RENEWABLE_FUELS = {"Solar", "Wind", "Water", "Geothermal", "Biomass", "CSP", "StorageEnergy"}

# ---------------------------------------------------------------------------
# Output CSV filenames  (relative to <run>/<scenario>/output_csv/)
# ---------------------------------------------------------------------------
# Each key maps to a list of candidate filenames tried in order (first found wins).
# New model format (2026+) listed first; old format (2025) as fallback.
CSV = {
    "summary":              ["pNetPresentCostSystem.csv", "pSummary.csv"],
    "capacity":             ["pCapacityTechFuel.csv",     "pCapacityByFuel.csv"],
    "capacity_country":     ["pCapacityTechFuel.csv",     "pCapacityByFuelCountry.csv"],
    "new_capacity":         ["pNewCapacityTechFuel.csv",  "pNewCapacityFuel.csv"],
    "new_capacity_country": ["pNewCapacityTechFuel.csv",  "pNewCapacityFuelCountry.csv"],
    "energy":               ["pEnergyTechFuel.csv",       "pEnergyByFuel.csv"],
    "energy_country":       ["pEnergyTechFuel.csv",       "pEnergyByFuelCountry.csv"],
    "energy_mix":           ["pEnergyTechFuel.csv",       "pEnergyMix.csv"],
    "emissions":            ["pEmissionsZone.csv",        "pEmissions.csv"],
    "emissions_intensity":  ["pEmissionsIntensityZone.csv","pEmissionsIntensity.csv"],
    "cost_summary":         ["pCosts.csv",                "pCostSummary.csv"],
    "cost_summary_country": ["pCosts.csv",                "pCostSummaryCountry.csv"],
    "cost_summary_full":    ["pCosts.csv",                "pCostSummaryFull.csv"],
    "lcoe":                 ["pPlantAnnualLCOE.csv"],
    "demand_supply":        ["pDemandEnergyZone.csv",     "pDemandSupply.csv"],
    "dispatch":             ["pDispatchTechFuel.csv",     "pDispatch.csv"],
    "fuel_dispatch":        ["pDispatchTechFuel.csv",     "pFuelDispatch.csv"],
    "interchange":          ["pInterchange.csv"],
    "interchange_country":  ["pInterchange.csv",          "pInterchangeCountry.csv"],
    "price":                ["pHourlyPrice.csv",          "pPrice.csv"],
    "utilization":          ["pUtilizationTechFuel.csv",  "pUtilizationByFuel.csv"],
    "peak_capacity":        ["pDemandPeakZone.csv",       "pPeakCapacity.csv"],
    "peak_capacity_country":["pDemandPeakZone.csv",       "pPeakCapacityCountry.csv"],
    "settings":             ["pSettings.csv"],
    "zcmap":                ["pZoneCountry.csv",          "zcmap.csv"],
}

# ---------------------------------------------------------------------------
# Input CSV filenames  (relative to epm/input/<folder>/)
# ---------------------------------------------------------------------------
INPUT_CSV = {
    # Supply
    "gen_data":         ("supply", "pGenDataInput.csv"),
    "gen_data_default": ("supply", "pGenDataInputDefault.csv"),
    "storage_data":     ("supply", "pStorageDataInput.csv"),
    "fuel_price":       ("supply", "pFuelPrice.csv"),
    "capex":            ("supply", "pCapexTrajectoriesCustom.csv"),
    "capex_default":    ("supply", "pCapexTrajectoriesDefault.csv"),
    "availability":     ("supply", "pAvailabilityCustom.csv"),
    # Load
    "demand_forecast":  ("load",   "pDemandForecast.csv"),
    "demand_profile":   ("load",   "pDemandProfile.csv"),
    "demand_data":      ("load",   "pDemandData.csv"),
    "efficiency":       ("load",   "pEnergyEfficiencyFactor.csv"),
    # Trade
    "transfer_limit":   ("trade",  "pTransferLimit.csv"),
    "new_transmission": ("trade",  "pNewTransmission.csv"),
    "trade_price":      ("trade",  "pTradePrice.csv"),
    "ext_transfer":     ("trade",  "pExtTransferLimit.csv"),
    # Constraints
    "emissions_total":   ("constraint", "pEmissionsTotal.csv"),
    "emissions_country": ("constraint", "pEmissionsCountry.csv"),
    "carbon_price":      ("constraint", "pCarbonPrice.csv"),
    "max_fuel":          ("constraint", "pMaxFuellimit.csv"),
    "max_generation":    ("constraint", "pMaxGenerationByFuel.csv"),
    # Reserve
    "planning_reserve": ("reserve", "pPlanningReserveMargin.csv"),
    "spinning_country": ("reserve", "pSpinningReserveReqCountry.csv"),
    "spinning_system":  ("reserve", "pSpinningReserveReqSystem.csv"),
    # Settings
    "settings_input":   ("",        "pSettings.csv"),
    "years":            ("",        "y.csv"),
    "phours":           ("",        "pHours.csv"),
    "zcmap_input":      ("",        "zcmap.csv"),
    "scenarios":        ("",        "scenarios.csv"),
}

# ---------------------------------------------------------------------------
# Input table schemas  — (column, type, editable, min, max, tooltip)
# ---------------------------------------------------------------------------
COLUMN_SCHEMAS = {
    "pGenDataInput.csv": [
        ("z",               "text",    False, None, None, "Zone / country"),
        ("g",               "text",    False, None, None, "Generator unique ID"),
        ("f",               "dropdown",True,  None, None, "Fuel type"),
        ("capacity_mw",     "numeric", True,  0,    99999,"Installed capacity (MW)"),
        ("capex",           "numeric", True,  0,    99999,"Capital cost ($/kW)"),
        ("fixedOM",         "numeric", True,  0,    99999,"Fixed O&M cost ($/kW/year)"),
        ("varOM",           "numeric", True,  0,    9999, "Variable O&M cost ($/MWh)"),
        ("heat_rate",       "numeric", True,  0,    99,   "Heat rate (mmBTU/MWh). Set 0 for renewables."),
        ("commission_year", "numeric", True,  2000, 2100, "Year plant enters service"),
        ("retirement_year", "numeric", True,  2000, 2100, "Year plant retires. Leave blank if unknown."),
        ("min_gen",         "numeric", True,  0,    1,    "Minimum generation as fraction of capacity (0–1)"),
    ],
    "pDemandForecast.csv": [
        ("z",     "text",    False, None, None,  "Zone / country"),
        ("y",     "numeric", False, 2000, 2100,  "Year"),
        ("value", "numeric", True,  0,    999999,"Peak demand (MW)"),
    ],
    "pFuelPrice.csv": [
        ("z",     "text",    False, None, None,  "Zone / country"),
        ("f",     "text",    False, None, None,  "Fuel type"),
        ("y",     "numeric", False, 2000, 2100,  "Year"),
        ("value", "numeric", True,  0,    9999,  "Fuel price ($/mmBTU)"),
    ],
    "pEmissionsTotal.csv": [
        ("y",     "numeric", False, 2000, 2100,  "Year"),
        ("value", "numeric", True,  0,    999999,"System-wide CO2 cap (Mt)"),
    ],
    "pCarbonPrice.csv": [
        ("y",     "numeric", False, 2000, 2100, "Year"),
        ("value", "numeric", True,  0,    9999, "Carbon price ($/tCO2)"),
    ],
    "pPlanningReserveMargin.csv": [
        ("z",     "text",    False, None, None, "Zone / country"),
        ("y",     "numeric", False, 2000, 2100, "Year"),
        ("value", "numeric", True,  0,    1,    "Planning reserve margin as a fraction (e.g. 0.15 = 15%)"),
    ],
    "pTransferLimit.csv": [
        ("z",     "text",    False, None, None,  "Zone from"),
        ("z2",    "text",    False, None, None,  "Zone to"),
        ("y",     "numeric", False, 2000, 2100,  "Year"),
        ("value", "numeric", True,  0,    99999, "Transfer limit (MW)"),
    ],
}
