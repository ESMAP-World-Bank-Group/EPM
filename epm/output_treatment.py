"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This Python script is part of the GAMS-based Electricity Planning Model (EPM),
    designed for post-processing of model outputs. It handles tasks such as:
    - Renaming columns in CSV files with wildcard domains
    - Calculating cumulative values over years
    - Other output transformations

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

License:
    Creative Commons Zero v1.0 Universal

Notes:
    - Can be run standalone or called from GAMS embedded Python
    - Default folder: simulations_test/baseline

Contact:
    Claire Nicolas - cnicolas@worldbank.org
**********************************************************************
"""

import argparse
import os
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

# Column renaming for parameters with wildcard (*) domains
RENAME_COLUMNS_MAP = {
    'pEnergyTechFuelComplete': {'z_0': 'z', 'uni_1': 'tech', 'uni_2': 'f', 'y_3': 'y'},
}

# Files to calculate cumulative values: (input_name, output_name)
CUMULATIVE_FILES = [
    ('pNewCapacityTechFuel', 'pNewCapacityTechFuelCumulated'),
    ('pCapexInvestmentComponent', 'pCapexInvestmentComponentCumulated'),
    ('pYearlyDiscountedWeightedCostsZone', 'pYearlyDiscountedWeightedCostsZoneCumulated'),
]

# Files to fill with all (tech, fuel) combinations and add Processing column
TECHFUEL_FILES = [
    'pNewCapacityTechFuel',
    'pNewCapacityTechFuelCumulated',
    'pCapacityTechFuel',
    'pEnergyTechFuel',
    'pEnergyTechFuelComplete',
    'pUtilizationTechFuel',
]

# Plant-level files that need to be merged with pGeneratorTechFuel to add tech/fuel columns
# After merging, these files can be processed like TechFuel files (techfuel column, renaming, etc.)
PLANT_FILES = [
    'pCapacityPlant',
    'pNewCapacityPlant',
    'pEnergyPlant',
    'pUtilizationPlant',
    'pPlantAnnualLCOE',
    'pCostsPlant',
]

# Path to pTechFuelProcessing.csv (relative to epm/ folder)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TECHFUEL_PROCESSING_PATH = os.path.join(_SCRIPT_DIR, 'resources', 'pTechFuelProcessing.csv')

# Files to fill with all cost components: (file_name, cost_component_column_name)
# The cost component column is either 'uni' or 'sumhdr' depending on the file
COST_COMPONENT_FILES = [
    ('pYearlyCostsZone', 'uni'),
    ('pYearlyCostsZonePerMWh', 'sumhdr'),
    ('pCostsZonePerMWh', 'sumhdr'),
    ('pYearlyCostsCountryPerMWh', 'sumhdr'),
    ('pCostsCountryPerMWh', 'sumhdr'),
    ('pYearlyDiscountedWeightedCostsZone', 'uni'),
    ('pCostsSystem', 'uni'),
    ('pCostsSystemPerMWh', 'uni'),
]

# All cost components from sumhdr set in generate_report.gms
ALL_COST_COMPONENTS = [
    "Generation costs: $m",
    "Fixed O&M: $m",
    "Variable O&M: $m",
    "Startup costs: $m",
    "Fuel costs: $m",
    "Transmission costs: $m",
    "Spinning reserve costs: $m",
    "Unmet demand costs: $m",
    "Unmet country spinning reserve costs: $m",
    "Unmet country planning reserve costs: $m",
    "Unmet country CO2 backstop cost: $m",
    "Unmet system planning reserve costs: $m",
    "Unmet system spinning reserve costs: $m",
    "Unmet system CO2 backstop cost: $m",
    "Excess generation: $m",
    "VRE curtailment: $m",
    "Import costs with external zones: $m",
    "Export revenues with external zones: $m",
    "Import costs with internal zones: $m",
    "Export revenues with internal zones: $m",
    "Trade shared benefits: $m",
    "Carbon costs: $m",
    "NPV of system cost: $m",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _default_log(message: str) -> None:
    """Default logging function that prints to stdout."""
    print(message)


def rename_columns(
    input_path: str,
    column_map: Dict[str, str],
    output_path: Optional[str] = None,
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Rename columns in a CSV file.

    Parameters
    ----------
    input_path : str
        Path to input CSV file
    column_map : dict
        Dictionary mapping old column names to new column names
    output_path : str, optional
        Path to output CSV file. If None, overwrites input file.
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = input_path

    file_name = os.path.basename(input_path)

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   {file_name}: WARNING - file not found")
        return False

    df = pd.read_csv(input_path)
    original_cols = list(df.columns)
    df.rename(columns=column_map, inplace=True)
    df.to_csv(output_path, index=False)

    # Show only the columns that changed
    changes = [f"{k}->{v}" for k, v in column_map.items() if k in original_cols]
    log_func(f"[output_treatment]   {file_name}: {', '.join(changes)}")

    return True


def calculate_cumulative(
    input_path: str,
    output_path: str,
    year_col: str = 'y',
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Calculate cumulative values over years, grouped by all other columns.

    Parameters
    ----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output CSV file with cumulated values
    year_col : str
        Name of the year column (default: 'y')
    value_col : str
        Name of the value column (default: 'value')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    input_name = os.path.basename(input_path)
    output_name = os.path.basename(output_path)

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   {input_name}: WARNING - file not found")
        return False

    df = pd.read_csv(input_path)

    # Identify grouping columns (all columns except year and value)
    group_cols = [col for col in df.columns if col not in [year_col, value_col]]

    # Sort by group columns and year to ensure correct cumulative order
    df = df.sort_values(by=group_cols + [year_col])

    # Calculate cumulative sum within each group
    df[value_col] = df.groupby(group_cols)[value_col].cumsum()

    # Save to output file
    df.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   {input_name} -> {output_name} ({len(df)} rows)")

    return True


def fill_techfuel_combinations(
    input_path: str,
    techfuel_df: pd.DataFrame,
    techfuel_mapping: Optional[Dict[str, str]] = None,
    tech_col: str = 'tech',
    fuel_col: str = 'f',
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Fill missing (tech, fuel) combinations in a TechFuel CSV file with value 0.
    Optionally adds a 'techfuel' column with Processing names.

    Parameters
    ----------
    input_path : str
        Path to input CSV file
    techfuel_df : pd.DataFrame
        DataFrame with unique (tech, fuel) pairs from pTechFuel
    techfuel_mapping : dict, optional
        Dictionary mapping 'tech-fuel' keys to Processing names.
        If provided, adds a 'techfuel' column with mapped values.
    tech_col : str
        Name of the technology column (default: 'tech')
    fuel_col : str
        Name of the fuel column (default: 'f')
    value_col : str
        Name of the value column (default: 'value')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    file_name = os.path.basename(input_path)

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   {file_name}: WARNING - file not found")
        return False

    if techfuel_df.empty:
        log_func(f"[output_treatment]   {file_name}: WARNING - no techfuel pairs provided")
        return False

    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Identify other dimension columns (e.g., zone, year)
    other_cols = [col for col in df.columns if col not in [tech_col, fuel_col, value_col, 'techfuel']]

    # Get unique values for other dimensions from existing data
    if other_cols:
        other_dims = df[other_cols].drop_duplicates()
    else:
        other_dims = pd.DataFrame([{}])

    # Cross join other_dims with techfuel_df to get all combinations
    other_dims['_cross'] = 1
    techfuel_df = techfuel_df.copy()
    techfuel_df['_cross'] = 1
    full_index_df = other_dims.merge(techfuel_df, on='_cross').drop(columns=['_cross'])
    other_dims = other_dims.drop(columns=['_cross'])

    # Merge with existing data to fill missing combinations with NaN
    merge_cols = other_cols + [tech_col, fuel_col]
    df_filled = full_index_df.merge(df, on=merge_cols, how='left')

    # Fill NaN values with 0
    df_filled[value_col] = df_filled[value_col].fillna(0)

    # Add techfuel column with Processing names if mapping provided
    if techfuel_mapping:
        df_filled['techfuel'] = (df_filled[tech_col] + '-' + df_filled[fuel_col]).map(techfuel_mapping)
        # Fallback to tech-fuel key if not in mapping
        df_filled['techfuel'] = df_filled['techfuel'].fillna(df_filled[tech_col] + '-' + df_filled[fuel_col])

    # Sort for consistent output (by techfuel order if available, otherwise by tech/fuel)
    if techfuel_mapping and 'techfuel' in df_filled.columns:
        # Create order based on techfuel_mapping values order
        techfuel_order = list(dict.fromkeys(techfuel_mapping.values()))
        df_filled['_sort_order'] = df_filled['techfuel'].apply(
            lambda x: techfuel_order.index(x) if x in techfuel_order else len(techfuel_order)
        )
        sort_cols = other_cols + ['_sort_order']
        df_filled = df_filled.sort_values(by=sort_cols)
        df_filled = df_filled.drop(columns=['_sort_order'])
    else:
        sort_cols = other_cols + [tech_col, fuel_col]
        df_filled = df_filled.sort_values(by=sort_cols)

    # Save back to file
    df_filled.to_csv(input_path, index=False)
    added_rows = len(df_filled) - original_rows
    log_func(f"[output_treatment]   {file_name}: {original_rows} -> {len(df_filled)} rows (+{added_rows})")

    return True


def fill_cost_components(
    input_path: str,
    all_cost_components: List[str],
    cost_col: str = 'uni',
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Fill missing cost components in a CSV file with value 0.

    This ensures all cost components from sumhdr are present in the file,
    which is required for proper visualization in Tableau.

    Parameters
    ----------
    input_path : str
        Path to input CSV file
    all_cost_components : list
        List of all cost component names (from sumhdr)
    cost_col : str
        Name of the cost component column (default: 'uni', can be 'sumhdr')
    value_col : str
        Name of the value column (default: 'value')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    file_name = os.path.basename(input_path)

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   {file_name}: WARNING - file not found")
        return False

    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Identify other dimension columns (e.g., zone, country, year)
    other_cols = [col for col in df.columns if col not in [cost_col, value_col]]

    # Get unique values for other dimensions from existing data
    if other_cols:
        other_dims = df[other_cols].drop_duplicates()
    else:
        other_dims = pd.DataFrame([{}])

    # Create all combinations of (other_dims, cost_component)
    all_combinations = []
    for _, other_row in other_dims.iterrows():
        for cost_comp in all_cost_components:
            row = other_row.to_dict()
            row[cost_col] = cost_comp
            all_combinations.append(row)

    full_index_df = pd.DataFrame(all_combinations)

    # Merge with existing data to fill missing combinations with NaN
    merge_cols = other_cols + [cost_col]
    df_filled = full_index_df.merge(df, on=merge_cols, how='left')

    # Fill NaN values with 0
    df_filled[value_col] = df_filled[value_col].fillna(0)

    # Sort for consistent output
    df_filled = df_filled.sort_values(by=merge_cols)

    # Save back to file
    df_filled.to_csv(input_path, index=False)
    added_rows = len(df_filled) - original_rows
    log_func(f"[output_treatment]   {file_name}: {original_rows} -> {len(df_filled)} rows (+{added_rows})")

    return True


def merge_plant_with_generator_techfuel(
    input_path: str,
    generator_techfuel_df: pd.DataFrame,
    generator_col: str = 'g',
    tech_col: str = 'tech',
    fuel_col: str = 'f',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Merge a plant-level CSV file with pGeneratorTechFuel to add tech and fuel columns.

    This enables plant-level files to be processed like TechFuel files, allowing
    the addition of the techfuel column and Processing names.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (plant-level data with 'g' column)
    generator_techfuel_df : pd.DataFrame
        DataFrame with (g, tech, f) mapping from pGeneratorTechFuel
    generator_col : str
        Name of the generator column in the input file (default: 'g')
    tech_col : str
        Name of the technology column to add (default: 'tech')
    fuel_col : str
        Name of the fuel column to add (default: 'f')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    file_name = os.path.basename(input_path)

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   {file_name}: WARNING - file not found")
        return False

    if generator_techfuel_df.empty:
        log_func(f"[output_treatment]   {file_name}: WARNING - no generator techfuel mapping provided")
        return False

    df = pd.read_csv(input_path)

    if generator_col not in df.columns:
        log_func(f"[output_treatment]   {file_name}: WARNING - column '{generator_col}' not found")
        return False

    original_cols = list(df.columns)

    # Merge with generator techfuel mapping
    df = df.merge(generator_techfuel_df, on=generator_col, how='left')

    # Reorder columns: put tech and f right after g
    g_idx = original_cols.index(generator_col)
    new_cols = original_cols[:g_idx + 1] + [tech_col, fuel_col] + original_cols[g_idx + 1:]
    # Only include columns that exist
    new_cols = [c for c in new_cols if c in df.columns]
    df = df[new_cols]

    # Save back to file
    df.to_csv(input_path, index=False)
    log_func(f"[output_treatment]   {file_name}: added {tech_col}, {fuel_col} columns")

    return True


# =============================================================================
# MAIN TREATMENT FUNCTIONS
# =============================================================================

def run_output_treatment(
    output_dir: str,
    rename_map: Optional[Dict[str, Dict[str, str]]] = None,
    cumulative_files: Optional[List[Tuple[str, str]]] = None,
    techfuel_files: Optional[List[str]] = None,
    cost_component_files: Optional[List[Tuple[str, str]]] = None,
    techfuel_df: Optional[pd.DataFrame] = None,
    techfuel_mapping: Optional[Dict[str, str]] = None,
    all_cost_components: Optional[List[str]] = None,
    plant_files: Optional[List[str]] = None,
    generator_techfuel_df: Optional[pd.DataFrame] = None,
    log_func: Callable[[str], None] = _default_log
) -> None:
    """
    Run all output treatments on CSV files in the specified directory.

    Parameters
    ----------
    output_dir : str
        Path to the output directory containing CSV files
    rename_map : dict, optional
        Dictionary of parameter names to column rename mappings.
        If None, uses RENAME_COLUMNS_MAP.
    cumulative_files : list, optional
        List of (input_name, output_name) tuples for cumulative calculations.
        If None, uses CUMULATIVE_FILES.
    techfuel_files : list, optional
        List of file names to fill with valid (tech, fuel) combinations.
        If None, uses TECHFUEL_FILES.
    cost_component_files : list, optional
        List of (file_name, cost_column_name) tuples for cost component filling.
        If None, uses COST_COMPONENT_FILES.
    techfuel_df : pd.DataFrame, optional
        DataFrame with unique (tech, fuel) pairs from pTechFuel. Required for techfuel filling.
    techfuel_mapping : dict, optional
        Dictionary mapping 'tech-fuel' keys to Processing names.
    all_cost_components : list, optional
        List of all cost component names. If None, uses ALL_COST_COMPONENTS.
    plant_files : list, optional
        List of plant-level file names to merge with generator techfuel data.
        If None, uses PLANT_FILES.
    generator_techfuel_df : pd.DataFrame, optional
        DataFrame with (g, tech, f) mapping from pGeneratorTechFuel. Required for plant file merging.
    log_func : callable
        Logging function (default: print)
    """
    if rename_map is None:
        rename_map = RENAME_COLUMNS_MAP
    if cumulative_files is None:
        cumulative_files = CUMULATIVE_FILES
    if techfuel_files is None:
        techfuel_files = TECHFUEL_FILES
    if cost_component_files is None:
        cost_component_files = COST_COMPONENT_FILES
    if all_cost_components is None:
        all_cost_components = ALL_COST_COMPONENTS
    if plant_files is None:
        plant_files = PLANT_FILES

    log_func("=" * 60)
    log_func("[output_treatment] Starting output treatment")
    log_func(f"[output_treatment] Output directory: {output_dir}")
    log_func("=" * 60)

    # ---------------------------------------------------------
    # 1. Rename columns for wildcard domain parameters
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 1: Renaming columns for wildcard domain parameters")
    log_func("-" * 60)

    for param_name, col_map in rename_map.items():
        csv_path = os.path.join(output_dir, f"{param_name}.csv")
        rename_columns(csv_path, col_map, log_func=log_func)

    # ---------------------------------------------------------
    # 2. Merge plant files with generator techfuel mapping
    # ---------------------------------------------------------
    if generator_techfuel_df is not None and not generator_techfuel_df.empty and plant_files:
        log_func("")
        log_func("[output_treatment] STEP 2: Merging plant files with generator techfuel mapping")
        log_func("-" * 60)

        for file_name in plant_files:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            merge_plant_with_generator_techfuel(csv_path, generator_techfuel_df, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 2: Skipping plant file merging (no generator_techfuel_df provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 3. Fill TechFuel combinations (if techfuel_df provided)
    # ---------------------------------------------------------
    if techfuel_df is not None and not techfuel_df.empty and techfuel_files:
        log_func("")
        log_func("[output_treatment] STEP 3: Filling TechFuel combinations and adding Processing column")
        log_func("-" * 60)

        # Process standard TechFuel files
        for file_name in techfuel_files:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            fill_techfuel_combinations(csv_path, techfuel_df, techfuel_mapping=techfuel_mapping, log_func=log_func)

        # Also process plant files that now have tech/f columns (from Step 2)
        for file_name in PLANT_FILES:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            fill_techfuel_combinations(csv_path, techfuel_df, techfuel_mapping=techfuel_mapping, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 3: Skipping TechFuel filling (no techfuel_df provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 4. Fill cost components with all sumhdr values
    # ---------------------------------------------------------
    if cost_component_files and all_cost_components:
        log_func("")
        log_func("[output_treatment] STEP 4: Filling cost components (sumhdr)")
        log_func("-" * 60)

        for file_name, cost_col in cost_component_files:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            fill_cost_components(csv_path, all_cost_components, cost_col=cost_col, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 4: Skipping cost component filling")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 5. Calculate cumulative values
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 5: Calculating cumulative values")
    log_func("-" * 60)

    for input_name, output_name in cumulative_files:
        input_path = os.path.join(output_dir, f"{input_name}.csv")
        output_path = os.path.join(output_dir, f"{output_name}.csv")
        calculate_cumulative(input_path, output_path, log_func=log_func)

    log_func("")
    log_func("=" * 60)
    log_func("[output_treatment] Output treatment complete")
    log_func("=" * 60)


def run_output_treatment_gams(gams, output_dir: str) -> None:
    """
    Run output treatment from GAMS embedded Python.

    Note: This function never raises exceptions - all errors are logged as warnings.
    These treatments are for Tableau compatibility and are non-fatal.

    Parameters
    ----------
    gams : object
        GAMS object with printLog method and db attribute for GAMS database access
    output_dir : str
        Path to the output directory containing CSV files
    """
    log_func = gams.printLog

    log_func("")
    log_func("=" * 60)
    log_func("[output_treatment] Starting output treatment for Tableau")
    log_func("[output_treatment] Note: Errors here are non-fatal warnings")
    log_func("=" * 60)

    try:
        import gams.transfer as gt

        # Validate output directory
        if not output_dir or not os.path.isdir(output_dir):
            log_func(f"[output_treatment] WARNING: Output directory not found: {output_dir}")
            log_func("[output_treatment] Skipping output treatment")
            return

        log_func(f"[output_treatment] Output directory: {output_dir}")

        # Extract tech-fuel pairs from GAMS database
        log_func("[output_treatment] Extracting tech-fuel pairs from GAMS database...")

        techfuel_df = pd.DataFrame(columns=['tech', 'f'])
        generator_techfuel_df = pd.DataFrame(columns=['g', 'tech', 'f'])

        try:
            db = gt.Container(gams.db)

            # Get unique (tech, fuel) pairs directly from pTechFuel
            if 'pTechFuel' in db.data:
                techfuel_data = db.data['pTechFuel']
                if techfuel_data.records is not None and len(techfuel_data.records) > 0:
                    techfuel_df = techfuel_data.records.iloc[:, :2].copy()
                    techfuel_df.columns = ['tech', 'f']
                    techfuel_df = techfuel_df.drop_duplicates()
                    log_func(f"[output_treatment]   Found {len(techfuel_df)} unique (tech, fuel) pairs from pTechFuel")

            # Get generator to (tech, fuel) mapping from pGeneratorTechFuel
            if 'pGeneratorTechFuel' in db.data:
                gen_techfuel_data = db.data['pGeneratorTechFuel']
                if gen_techfuel_data.records is not None and len(gen_techfuel_data.records) > 0:
                    generator_techfuel_df = gen_techfuel_data.records.iloc[:, :3].copy()
                    generator_techfuel_df.columns = ['g', 'tech', 'f']
                    generator_techfuel_df = generator_techfuel_df.drop_duplicates()
                    log_func(f"[output_treatment]   Found {len(generator_techfuel_df)} generator-techfuel mappings from pGeneratorTechFuel")

        except Exception as e:
            log_func(f"[output_treatment]   WARNING: Could not extract tech-fuel pairs from GAMS: {e}")

        # Load techfuel_mapping from pTechFuelProcessing.csv
        log_func("[output_treatment] Loading techfuel mapping from pTechFuelProcessing.csv...")

        techfuel_mapping = None

        try:
            if os.path.exists(TECHFUEL_PROCESSING_PATH):
                df_processing = pd.read_csv(TECHFUEL_PROCESSING_PATH, comment='#')
                # Create mapping: 'tech-fuel' -> Processing
                techfuel_mapping = dict(zip(
                    df_processing['tech'] + '-' + df_processing['fuel'],
                    df_processing['Processing']
                ))
                log_func(f"[output_treatment]   Loaded {len(techfuel_mapping)} Processing mappings")
            else:
                log_func(f"[output_treatment]   WARNING: pTechFuelProcessing.csv not found at {TECHFUEL_PROCESSING_PATH}")
        except Exception as e:
            log_func(f"[output_treatment]   WARNING: Could not load techfuel mapping: {e}")

        # Run the main treatment
        run_output_treatment(
            output_dir,
            techfuel_df=techfuel_df,
            techfuel_mapping=techfuel_mapping,
            generator_techfuel_df=generator_techfuel_df,
            log_func=log_func
        )

    except Exception as e:
        log_func(f"[output_treatment] WARNING: Output treatment failed: {e}")
        log_func("[output_treatment] This is non-fatal - model results are still valid")
        log_func("[output_treatment] Tableau outputs may need manual adjustment")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    # Default path when running from root folder (EPM_main/)
    DEFAULT_OUTPUT_DIR = os.path.join("epm", "output", "simulations_test", "baseline", "output_csv")

    parser = argparse.ArgumentParser(
        description="Post-process EPM output CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python epm/output_treatment.py
  python epm/output_treatment.py epm/output/simulations_test/baseline/output_csv
  python epm/output_treatment.py epm/output/simulations_run_20241201/baseline/output_csv
        """
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output directory containing CSV files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="Skip column renaming step"
    )
    parser.add_argument(
        "--no-cumulative",
        action="store_true",
        help="Skip cumulative calculation step"
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.output_dir):
        print(f"ERROR: Directory not found: {args.output_dir}")
        raise SystemExit(1)

    # Configure what to run
    rename_map = RENAME_COLUMNS_MAP if not args.no_rename else {}
    cumulative_files = CUMULATIVE_FILES if not args.no_cumulative else []

    # Run treatment
    run_output_treatment(
        args.output_dir,
        rename_map=rename_map,
        cumulative_files=cumulative_files,
        log_func=_default_log if args.verbose else lambda x: None
    )

    # Always print summary
    print(f"\nOutput treatment completed for: {args.output_dir}")
