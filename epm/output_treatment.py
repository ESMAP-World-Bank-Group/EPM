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

# Files to fill with all (tech, fuel) combinations
TECHFUEL_FILES = [
    'pNewCapacityTechFuel',
    'pCapacityTechFuel',
    'pEnergyTechFuel',
    'pUtilizationTechFuel',
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
    all_tech: List[str],
    all_fuel: List[str],
    tech_col: str = 'tech',
    fuel_col: str = 'f',
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Fill missing (tech, fuel) combinations in a TechFuel CSV file with value 0.

    Parameters
    ----------
    input_path : str
        Path to input CSV file
    all_tech : list
        List of all technology names (from pTechData)
    all_fuel : list
        List of all fuel names (from ftfindex)
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

    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Identify other dimension columns (e.g., zone, year)
    other_cols = [col for col in df.columns if col not in [tech_col, fuel_col, value_col]]

    # Get unique values for other dimensions from existing data
    if other_cols:
        other_dims = df[other_cols].drop_duplicates()
    else:
        other_dims = pd.DataFrame([{}])

    # Create all combinations of (other_dims, tech, fuel)
    from itertools import product

    all_combinations = []
    for _, other_row in other_dims.iterrows():
        for tech, fuel in product(all_tech, all_fuel):
            row = other_row.to_dict()
            row[tech_col] = tech
            row[fuel_col] = fuel
            all_combinations.append(row)

    full_index_df = pd.DataFrame(all_combinations)

    # Merge with existing data to fill missing combinations with NaN
    merge_cols = other_cols + [tech_col, fuel_col]
    df_filled = full_index_df.merge(df, on=merge_cols, how='left')

    # Fill NaN values with 0
    df_filled[value_col] = df_filled[value_col].fillna(0)

    # Sort for consistent output
    sort_cols = other_cols + [tech_col, fuel_col]
    df_filled = df_filled.sort_values(by=sort_cols)

    # Save back to file
    df_filled.to_csv(input_path, index=False)
    added_rows = len(df_filled) - original_rows
    log_func(f"[output_treatment]   {file_name}: {original_rows} -> {len(df_filled)} rows (+{added_rows})")

    return True


# =============================================================================
# MAIN TREATMENT FUNCTIONS
# =============================================================================

def run_output_treatment(
    output_dir: str,
    rename_map: Optional[Dict[str, Dict[str, str]]] = None,
    cumulative_files: Optional[List[Tuple[str, str]]] = None,
    techfuel_files: Optional[List[str]] = None,
    all_tech: Optional[List[str]] = None,
    all_fuel: Optional[List[str]] = None,
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
        List of file names to fill with all (tech, fuel) combinations.
        If None, uses TECHFUEL_FILES.
    all_tech : list, optional
        List of all technology names. Required for techfuel filling.
    all_fuel : list, optional
        List of all fuel names. Required for techfuel filling.
    log_func : callable
        Logging function (default: print)
    """
    if rename_map is None:
        rename_map = RENAME_COLUMNS_MAP
    if cumulative_files is None:
        cumulative_files = CUMULATIVE_FILES
    if techfuel_files is None:
        techfuel_files = TECHFUEL_FILES

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
    # 2. Fill TechFuel combinations (if tech/fuel lists provided)
    # ---------------------------------------------------------
    if all_tech and all_fuel and techfuel_files:
        log_func("")
        log_func("[output_treatment] STEP 2: Filling TechFuel combinations")
        log_func("-" * 60)

        for file_name in techfuel_files:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            fill_techfuel_combinations(csv_path, all_tech, all_fuel, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 2: Skipping TechFuel filling (no tech/fuel lists provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 3. Calculate cumulative values
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 3: Calculating cumulative values")
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

        # Extract tech and fuel lists from GAMS database
        log_func("[output_treatment] Extracting tech and fuel from GAMS database...")

        all_tech = []
        all_fuel = []

        try:
            db = gt.Container(gams.db)

            # Get tech from pTechData (first dimension)
            if 'pTechData' in db.data:
                tech_data = db.data['pTechData']
                if tech_data.records is not None and len(tech_data.records) > 0:
                    all_tech = tech_data.records.iloc[:, 0].unique().tolist()
                    log_func(f"[output_treatment]   Found {len(all_tech)} technologies from pTechData")

            # Get fuel from ftfindex (first dimension)
            if 'ftfindex' in db.data:
                ftf_data = db.data['ftfindex']
                if ftf_data.records is not None and len(ftf_data.records) > 0:
                    all_fuel = ftf_data.records.iloc[:, 0].unique().tolist()
                    log_func(f"[output_treatment]   Found {len(all_fuel)} fuels from ftfindex")

        except Exception as e:
            log_func(f"[output_treatment]   WARNING: Could not extract tech/fuel from GAMS: {e}")

        # Run the main treatment
        run_output_treatment(
            output_dir,
            all_tech=all_tech,
            all_fuel=all_fuel,
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
