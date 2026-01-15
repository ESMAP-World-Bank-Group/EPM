"""
Data Builder Service

Builds EPM input folder structure from web form inputs.
Uses data_test as a template and modifies specific files based on user input.
"""

import csv
import shutil
import uuid
from pathlib import Path
from typing import Any

from routes.uploads import copy_session_files_to_scenario

# Path to EPM root
EPM_ROOT = Path(__file__).parent.parent.parent.parent / "epm"
TEMPLATE_FOLDER = EPM_ROOT / "input" / "data_test"
# Store runs in EPM's input folder so epm.py can find them
RUNS_FOLDER = EPM_ROOT / "input"


def ensure_runs_folder():
    """Create runs folder if it doesn't exist."""
    RUNS_FOLDER.mkdir(parents=True, exist_ok=True)


def create_input_folder(scenario_id: str) -> Path:
    """
    Create a new input folder by copying the template.

    Args:
        scenario_id: Unique identifier for this scenario

    Returns:
        Path to the created input folder
    """
    ensure_runs_folder()
    target_folder = RUNS_FOLDER / f"data_{scenario_id}"

    if target_folder.exists():
        shutil.rmtree(target_folder)

    shutil.copytree(TEMPLATE_FOLDER, target_folder)
    return target_folder


def write_years_csv(folder: Path, start_year: int, end_year: int, step: int = 5):
    """
    Write the y.csv file with planning horizon years.

    Args:
        folder: Input folder path
        start_year: First year of planning horizon
        end_year: Last year of planning horizon
        step: Year interval (default 5)
    """
    years_file = folder / "y.csv"
    years = list(range(start_year, end_year + 1, step))

    # Ensure end_year is included
    if years[-1] != end_year:
        years.append(end_year)

    with open(years_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y"])
        for year in years:
            writer.writerow([year])


def write_zones_csv(folder: Path, zones: list[dict]):
    """
    Write the zcmap.csv file with zone-country mapping.

    Args:
        folder: Input folder path
        zones: List of zone dicts with 'code' and 'country' keys
    """
    zcmap_file = folder / "zcmap.csv"

    with open(zcmap_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["z", "c"])
        for zone in zones:
            writer.writerow([zone["code"], zone.get("country", zone["code"])])


def write_settings_csv(folder: Path, settings: dict[str, Any]):
    """
    Update pSettings.csv with user-provided settings.
    Only updates values that are explicitly provided.

    Args:
        folder: Input folder path
        settings: Dictionary of setting abbreviations to values
    """
    settings_file = folder / "pSettings.csv"

    # Read existing settings
    rows = []
    with open(settings_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    # Update values
    for i, row in enumerate(rows):
        if len(row) >= 3 and row[1] in settings:
            rows[i][2] = str(settings[row[1]])

    # Write back
    with open(settings_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_generators_csv(folder: Path, generators: list[dict]):
    """
    Write or update pGenDataInput.csv with generator data.

    Args:
        folder: Input folder path
        generators: List of generator dictionaries
    """
    gen_file = folder / "supply" / "pGenDataInput.csv"

    # Define column headers
    headers = [
        "g", "z", "tech", "fuel", "Status", "StYr", "RetrYr", "Life",
        "Capacity", "UnitSize", "BuildLimitperYear", "DescreteCap",
        "HeatRate", "Capex", "FOMperMW", "VOM", "ReserveCost",
        "MinLimitShare", "RampUpRate", "RampDnRate", "OverLoadFactor",
        "ResLimShare", "MinGenCommitment", "HoursOn", "HoursOff",
        "minUT", "minDT", "StUpCost", "InitialOn", "fuel2", "HeatRate2"
    ]

    # Map from schema field names to CSV column names
    field_map = {
        "name": "g",
        "zone": "z",
        "technology": "tech",
        "fuel": "fuel",
        "capacity_mw": "Capacity",
        "status": "Status",
        "start_year": "StYr",
        "retirement_year": "RetrYr",
        "capex_per_mw": "Capex",
        "fixed_om_per_mw": "FOMperMW",
        "variable_om": "VOM",
        "heat_rate": "HeatRate",
    }

    with open(gen_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for gen in generators:
            row = {}
            for schema_field, csv_col in field_map.items():
                if schema_field in gen and gen[schema_field] is not None:
                    row[csv_col] = gen[schema_field]
            writer.writerow(row)


def write_demand_forecast_csv(folder: Path, demand_data: list[dict], years: list[int]):
    """
    Write pDemandForecast.csv with demand projections.

    Args:
        folder: Input folder path
        demand_data: List of demand dicts with zone, base values, and growth rate
        years: List of planning years
    """
    demand_file = folder / "load" / "pDemandForecast.csv"

    # Calculate demand for each year based on growth rate
    rows = []
    for demand in demand_data:
        zone = demand["zone"]
        base_energy = demand["base_year_energy_gwh"]
        base_peak = demand["base_year_peak_mw"]
        growth = demand.get("annual_growth_rate", 0.03)

        base_year = years[0]

        # Energy row
        energy_row = {"z": zone, "type": "Energy"}
        for year in years:
            years_from_base = year - base_year
            energy_row[str(year)] = round(base_energy * ((1 + growth) ** years_from_base), 2)
        rows.append(energy_row)

        # Peak row
        peak_row = {"z": zone, "type": "Peak"}
        for year in years:
            years_from_base = year - base_year
            peak_row[str(year)] = round(base_peak * ((1 + growth) ** years_from_base), 2)
        rows.append(peak_row)

    # Write CSV
    fieldnames = ["z", "type"] + [str(y) for y in years]
    with open(demand_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_carbon_price_csv(folder: Path, carbon_price: float, years: list[int]):
    """
    Write pCarbonPrice.csv with carbon price trajectory.

    Args:
        folder: Input folder path
        carbon_price: Carbon price in $/tCO2
        years: List of planning years
    """
    carbon_file = folder / "constraint" / "pCarbonPrice.csv"

    with open(carbon_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "CarbonPrice"])
        for year in years:
            writer.writerow([year, carbon_price])


def build_scenario_folder(scenario_data: dict) -> tuple[str, Path]:
    """
    Build a complete EPM input folder from scenario data.

    Args:
        scenario_data: Dictionary containing scenario configuration

    Returns:
        Tuple of (scenario_id, folder_path)
    """
    # Generate unique ID
    scenario_id = str(uuid.uuid4())[:8]

    # Create folder from template
    folder = create_input_folder(scenario_id)

    # Extract years
    start_year = scenario_data.get("start_year", 2025)
    end_year = scenario_data.get("end_year", 2040)
    years = list(range(start_year, end_year + 1, 5))
    if years[-1] != end_year:
        years.append(end_year)

    # Write years
    write_years_csv(folder, start_year, end_year)

    # Write zones if provided
    zones = scenario_data.get("zones", [])
    if zones:
        # Convert string list to dict list if needed
        if isinstance(zones[0], str):
            zones = [{"code": z, "country": z} for z in zones]
        write_zones_csv(folder, zones)

    # Build settings from features and economics
    settings = {}

    economics = scenario_data.get("economics", {})
    if economics:
        if "wacc" in economics:
            settings["WACC"] = economics["wacc"]
        if "discount_rate" in economics:
            settings["DR"] = economics["discount_rate"]
        if "voll" in economics:
            settings["VoLL"] = economics["voll"]

    features = scenario_data.get("features", {})
    if features:
        if "enable_capacity_expansion" in features:
            settings["fEnableCapacityExpansion"] = 1 if features["enable_capacity_expansion"] else 0
        if "enable_transmission_expansion" in features:
            settings["fAllowTransferExpansion"] = 1 if features["enable_transmission_expansion"] else 0
        if "enable_storage" in features:
            settings["fEnableStorage"] = 1 if features["enable_storage"] else 0
        if "enable_hydrogen" in features:
            settings["fEnableH2Production"] = 1 if features["enable_hydrogen"] else 0
        if "apply_carbon_price" in features:
            settings["fEnableCarbonPrice"] = 1 if features["apply_carbon_price"] else 0
        if "apply_co2_constraint" in features:
            settings["fApplySystemCo2Constraint"] = 1 if features["apply_co2_constraint"] else 0
        if "enable_economic_retirement" in features:
            settings["fEnableEconomicRetirement"] = 1 if features["enable_economic_retirement"] else 0

    emissions = scenario_data.get("emissions", {})
    if emissions:
        if emissions.get("min_renewable_share"):
            settings["sMinRenewableSharePct"] = emissions["min_renewable_share"]

    if settings:
        write_settings_csv(folder, settings)

    # Write generators if provided
    generators = scenario_data.get("generators", [])
    if generators:
        write_generators_csv(folder, generators)

    # Write demand if provided
    demand = scenario_data.get("demand", [])
    if demand:
        write_demand_forecast_csv(folder, demand, years)

    # Write carbon price if provided and enabled
    if features.get("apply_carbon_price") and emissions.get("carbon_price_per_ton"):
        write_carbon_price_csv(folder, emissions["carbon_price_per_ton"], years)

    # Copy uploaded files (overwriting template files)
    upload_session_id = scenario_data.get("upload_session_id")
    if upload_session_id:
        copy_session_files_to_scenario(upload_session_id, folder)

    return scenario_id, folder


def get_template_data() -> dict:
    """
    Read template data from data_test to provide defaults to frontend.

    Returns:
        Dictionary with zones, technologies, fuels, and default settings
    """
    data = {
        "zones": [],
        "technologies": [],
        "fuels": [],
        "default_years": [],
        "default_settings": {}
    }

    # Read zones
    zcmap_file = TEMPLATE_FOLDER / "zcmap.csv"
    if zcmap_file.exists():
        with open(zcmap_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["zones"].append({
                    "code": row["z"],
                    "name": row["z"],
                    "country": row["c"]
                })

    # Read years
    years_file = TEMPLATE_FOLDER / "y.csv"
    if years_file.exists():
        with open(years_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["default_years"].append(int(row["y"]))

    # Read technologies from resources
    tech_file = EPM_ROOT / "resources" / "pTechFuel.csv"
    if tech_file.exists():
        with open(tech_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            seen_techs = set()
            seen_fuels = set()
            for row in reader:
                tech = row.get("tech", "")
                fuel = row.get("fuel", "")

                if tech and tech not in seen_techs:
                    seen_techs.add(tech)
                    is_renewable = fuel in ["Solar", "Wind", "Water", "Biomass", "Geothermal"]
                    data["technologies"].append({
                        "code": tech,
                        "name": tech,
                        "fuel": fuel,
                        "is_renewable": is_renewable
                    })

                if fuel and fuel not in seen_fuels:
                    seen_fuels.add(fuel)
                    data["fuels"].append(fuel)

    # Read default settings
    settings_file = TEMPLATE_FOLDER / "pSettings.csv"
    if settings_file.exists():
        with open(settings_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3 and row[1]:
                    try:
                        value = float(row[2]) if row[2] else None
                    except ValueError:
                        value = row[2]
                    data["default_settings"][row[1]] = value

    return data
