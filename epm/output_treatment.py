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

    log_func(f"[output_treatment] Renaming columns in {os.path.basename(input_path)}...")

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   WARNING: File not found - {input_path}")
        return False

    df = pd.read_csv(input_path)
    original_cols = list(df.columns)
    log_func(f"[output_treatment]   Original columns: {original_cols}")

    df.rename(columns=column_map, inplace=True)
    new_cols = list(df.columns)
    log_func(f"[output_treatment]   New columns:      {new_cols}")

    df.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   SUCCESS: Saved to {os.path.basename(output_path)}")

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
    log_func(f"[output_treatment] Calculating cumulative values for {os.path.basename(input_path)}...")

    if not os.path.exists(input_path):
        log_func(f"[output_treatment]   WARNING: File not found - {input_path}")
        return False

    df = pd.read_csv(input_path)
    log_func(f"[output_treatment]   Loaded {len(df)} rows")
    log_func(f"[output_treatment]   Columns: {list(df.columns)}")

    # Identify grouping columns (all columns except year and value)
    group_cols = [col for col in df.columns if col not in [year_col, value_col]]
    log_func(f"[output_treatment]   Grouping by: {group_cols}")
    log_func(f"[output_treatment]   Year column: {year_col}")
    log_func(f"[output_treatment]   Value column: {value_col}")

    # Sort by group columns and year to ensure correct cumulative order
    df = df.sort_values(by=group_cols + [year_col])

    # Calculate cumulative sum within each group
    df[value_col] = df.groupby(group_cols)[value_col].cumsum()

    # Save to output file
    df.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   SUCCESS: Saved cumulative values to {os.path.basename(output_path)}")
    log_func(f"[output_treatment]   Output rows: {len(df)}")

    return True


# =============================================================================
# MAIN TREATMENT FUNCTIONS
# =============================================================================

def run_output_treatment(
    output_dir: str,
    rename_map: Optional[Dict[str, Dict[str, str]]] = None,
    cumulative_files: Optional[List[Tuple[str, str]]] = None,
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
    log_func : callable
        Logging function (default: print)
    """
    if rename_map is None:
        rename_map = RENAME_COLUMNS_MAP
    if cumulative_files is None:
        cumulative_files = CUMULATIVE_FILES

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
    # 2. Calculate cumulative values
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 2: Calculating cumulative values")
    log_func("-" * 60)

    for input_name, output_name in cumulative_files:
        input_path = os.path.join(output_dir, f"{input_name}.csv")
        output_path = os.path.join(output_dir, f"{output_name}.csv")
        calculate_cumulative(input_path, output_path, log_func=log_func)

    log_func("")
    log_func("=" * 60)
    log_func("[output_treatment] Output treatment complete")
    log_func("=" * 60)


def run_output_treatment_gams(gams) -> None:
    """
    Run output treatment from GAMS embedded Python.

    Parameters
    ----------
    gams : object
        GAMS object with printLog method and environment variables
    """
    # Get output directory from GAMS environment
    output_dir = gams.wsWorkingDir

    # Use GAMS printLog for logging
    run_output_treatment(output_dir, log_func=gams.printLog)


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
