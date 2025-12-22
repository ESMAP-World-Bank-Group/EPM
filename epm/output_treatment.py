"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    Post-processing script for EPM model outputs. Prepares CSV files for
    Tableau visualization by:

    1. Renaming columns (wildcard domain parameters)
    2. Adding tech/fuel columns to plant files (via pGeneratorTechFuel)
    3. Filling missing tech-fuel combinations with zeros
    4. Adding techfuel column to dispatch files
    5. Creating pDispatchComplete (pDispatch + pDispatchTechFuel)
    6. Filling cost components with all sumhdr values
    7. Calculating cumulative values over years
    8. Aggregating plant files to tech-fuel level
    9. Merging related files into consolidated CSVs (long format):
       - pTechFuelMerged
       - pPlantMerged
       - pYearlyCostsMerged
       - pTransmissionMerged
       - pYearlyZoneMerged
       - pCostsSystemMerged
    10. Adding country column to zone-based files
    11. Organizing files (essential in main dir, others in 'other/' subdir)

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

License:
    Creative Commons Zero v1.0 Universal

Notes:
    - Can be run standalone or called from GAMS embedded Python
    - Default folder: simulations_test/baseline/output_csv
    - All errors are non-fatal (logged as warnings)

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
# This section defines which files to process and how.
#
# PROCESSING PIPELINE:
# 1. RENAME_COLUMNS_MAP     -> Fix column names from wildcard domains
# 2. PLANT_FILES            -> Add tech/f columns via pGeneratorTechFuel
# 3. TECHFUEL_FILES         -> Fill missing (tech,fuel) combinations with zeros
# 4. DISPATCH_FUEL_FILES    -> Add techfuel column (tech-f combination)
# 5. COST_COMPONENT_FILES   -> Fill missing cost components with zeros
# 6. CUMULATIVE_FILES       -> Calculate cumulative values over years
# 7. PLANT_TO_TECHFUEL_AGGREGATIONS -> Aggregate plant data to tech-fuel level
# 8. *_MERGE_FILES          -> Merge related files into consolidated CSVs
# 9. Add country column to all zone-based files
# 10. Move non-essential files to 'other/' subdirectory
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

# Dispatch files that need techfuel column added (they already have tech and f columns)
# These will be merged with pDispatch into pDispatchComplete
DISPATCH_FUEL_FILES = [
    'pDispatchTechFuel',
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
# FILES TO MERGE INTO CONSOLIDATED CSVs
# =============================================================================

# TechFuel files to merge (dimensions: z, tech, f, y, value)
# These will be merged into pTechFuelMerged.csv in long format
# Note: Includes cumulated files, so merge must happen after cumulative calculation
TECHFUEL_MERGE_FILES = [
    'pCapacityTechFuel',
    'pNewCapacityTechFuel',
    'pNewCapacityTechFuelCumulated',
    'pEnergyTechFuelComplete',
    'pUtilizationTechFuel',
    'pReserveSpinningTechFuel',
]

# Plant files to aggregate to TechFuel level: (input_name, output_name)
# These are plant-level files that will be merged with pGeneratorTechFuel and grouped by sum
PLANT_TO_TECHFUEL_AGGREGATIONS = [
    ('pReserveSpinningPlantZone', 'pReserveSpinningTechFuel'),
]

# Plant files to merge (dimensions: z, g, y, value) - excludes pDispatch which has time dimensions
# These will be merged into pPlantMerged.csv with an 'attribute' column
PLANT_MERGE_FILES = [
    'pCapacityPlant',
    'pNewCapacityPlant',
    'pEnergyPlant',
    'pUtilizationPlant',
    'pPlantAnnualLCOE',
    'pCostsPlant',
    'pCapexInvestmentPlant',
]

# YearlyCosts zone files to merge (dimensions: z, uni, y, value)
# These will be merged into pYearlyCostsMerged.csv (long format)
YEARLY_COSTS_MERGE_FILES = [
    'pYearlyCostsZone',
    'pYearlyDiscountedWeightedCostsZone',
    'pCostsZonePerMWh',
    'pYearlyGenCostZonePerMWh',
]

# Transmission/interconnection files to merge (dimensions: z, z2, y, value)
# These will be merged into pTransmissionMerged.csv (long format)
TRANSMISSION_MERGE_FILES = [
    'pInterchange',
    'pInterconUtilization',
    'pAnnualTransmissionCapacity',
    'pAdditionalTransmissionCapacity',
    'pCongestionShare',
]

# Yearly zone files to merge (dimensions: z, y, value)
# Combines demand, emissions into pYearlyZoneMerged.csv (long format)
YEARLY_ZONE_MERGE_FILES = [
    'pDemandEnergyZone',
    'pDemandPeakZone',
    'pEmissionsZone',
    'pEmissionsIntensityZone',
]

# System-level cost files to merge (dimensions: uni, value or y, value)
# These will be merged into pCostsSystemMerged.csv (long format)
SYSTEM_COSTS_MERGE_FILES = [
    'pCostsSystem',
    'pCostsSystemPerMWh',
]

# =============================================================================
# FILES TO KEEP IN MAIN OUTPUT DIRECTORY
# =============================================================================
# These files are kept in the main output_csv directory for Tableau/analysis
# All other CSV files will be moved to the 'other' subdirectory

# Primary merged/consolidated files (created by output_treatment)
PRIMARY_OUTPUT_FILES = [
    'pTechFuelMerged',
    'pPlantMerged',
    'pYearlyCostsMerged',
    'pTransmissionMerged',
    'pYearlyZoneMerged',
    'pCostsSystemMerged',
    'pDispatchComplete',
]

# Essential standalone files (not merged but needed for analysis)
ESSENTIAL_FILES = [
    # Prices
    'pPrice',
    # Settings
    'pSettings',
    # Investment components
    'pCapexInvestmentComponent',
    'pCapexInvestmentComponentCumulated',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _default_log(message: str) -> None:
    """Default logging function that prints to stdout."""
    print(message)


# Preferred column order for merged files
COLUMN_ORDER = ['c', 'z', 'tech', 'f', 'g', 'y', 'uni', 'techfuel']


def _reorder_columns(df: pd.DataFrame, preferred_order: List[str] = COLUMN_ORDER) -> pd.DataFrame:
    """
    Reorder DataFrame columns according to preferred order.
    Columns not in preferred_order are placed at the end in their original order.
    """
    existing_cols = list(df.columns)
    ordered_cols = []

    # Add columns in preferred order (if they exist)
    for col in preferred_order:
        if col in existing_cols:
            ordered_cols.append(col)
            existing_cols.remove(col)

    # Add remaining columns at the end
    ordered_cols.extend(existing_cols)

    return df[ordered_cols]


def merge_csv_files_wide(
    output_dir: str,
    file_names: List[str],
    output_name: str,
    techfuel_df: Optional[pd.DataFrame] = None,
    techfuel_mapping: Optional[Dict[str, str]] = None,
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Merge multiple CSV files into one wide-format file with attributes as columns.

    Each source file becomes a column in the output, with the file name (without 'p'
    prefix) as the column name. Missing combinations are filled with 0.

    Parameters
    ----------
    output_dir : str
        Path to the directory containing CSV files
    file_names : list
        List of file names (without .csv extension) to merge
    output_name : str
        Name for the output merged file (without .csv extension)
    techfuel_df : pd.DataFrame, optional
        DataFrame with unique (tech, fuel) pairs to fill all combinations
    techfuel_mapping : dict, optional
        Dictionary mapping 'tech-fuel' keys to Processing names for techfuel column
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    dfs = {}

    for file_name in file_names:
        csv_path = os.path.join(output_dir, f"{file_name}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Rename cost component columns to 'uni' for consistency
        # (cost files use sumhdr or genCostCmp, others use uni)
        if 'sumhdr' in df.columns:
            df = df.rename(columns={'sumhdr': 'uni'})
        if 'genCostCmp' in df.columns:
            df = df.rename(columns={'genCostCmp': 'uni'})

        # Extract attribute name from file name (remove 'p' prefix)
        attr_name = file_name[1:] if file_name.startswith('p') else file_name

        # Skip files without 'value' column
        if 'value' not in df.columns:
            log_func(f"[output_treatment]   {file_name}: WARNING - no 'value' column found, skipping")
            continue

        # Rename value column to attribute name
        df = df.rename(columns={'value': attr_name})
        dfs[attr_name] = df

    if not dfs:
        log_func(f"[output_treatment]   {output_name}: WARNING - no files found to merge")
        return False

    # Merge all dataframes on common index columns
    attr_names = list(dfs.keys())
    merged_df = dfs[attr_names[0]]

    for attr_name in attr_names[1:]:
        df = dfs[attr_name]
        # Find common columns (excluding attribute columns and 'techfuel' which is added later)
        # Excluding 'techfuel' prevents duplicates when some files have it and others don't
        common_cols = [c for c in merged_df.columns if c in df.columns and c not in attr_names and c != 'techfuel']
        if common_cols:
            # Drop techfuel before merge to avoid column conflicts
            if 'techfuel' in merged_df.columns:
                merged_df = merged_df.drop(columns=['techfuel'])
            if 'techfuel' in df.columns:
                df = df.drop(columns=['techfuel'])
            merged_df = merged_df.merge(df, on=common_cols, how='outer')
        else:
            # If no common columns, just concatenate
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Fill missing tech-fuel combinations if techfuel_df provided
    if techfuel_df is not None and not techfuel_df.empty:
        if 'tech' in merged_df.columns and 'f' in merged_df.columns:
            # Identify dimension columns (not attribute columns)
            dim_cols = [c for c in merged_df.columns if c not in attr_names]
            other_dim_cols = [c for c in dim_cols if c not in ['tech', 'f']]

            if other_dim_cols:
                # Get unique values for other dimensions
                other_dims = merged_df[other_dim_cols].drop_duplicates()

                # Cross join with techfuel_df
                other_dims['_cross'] = 1
                techfuel_expanded = techfuel_df.copy()
                techfuel_expanded['_cross'] = 1
                full_index = other_dims.merge(techfuel_expanded, on='_cross').drop(columns=['_cross'])

                # Merge with existing data
                merged_df = full_index.merge(merged_df, on=dim_cols, how='left')

            # Fill NaN values with 0 for attribute columns
            for attr_name in attr_names:
                if attr_name in merged_df.columns:
                    merged_df[attr_name] = merged_df[attr_name].fillna(0)

            # Add techfuel column if mapping provided
            if techfuel_mapping:
                merged_df['techfuel'] = (merged_df['tech'] + '-' + merged_df['f']).map(techfuel_mapping)
                merged_df['techfuel'] = merged_df['techfuel'].fillna(merged_df['tech'] + '-' + merged_df['f'])

    # Reorder columns according to preferred order
    merged_df = _reorder_columns(merged_df)

    # Save merged file
    output_path = os.path.join(output_dir, f"{output_name}.csv")
    merged_df.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   {output_name}.csv: merged {len(dfs)} files -> wide format ({len(merged_df)} rows)")

    return True


def merge_csv_files_long(
    output_dir: str,
    file_names: List[str],
    output_name: str,
    attribute_col: str = 'attribute',
    normalize_cost_component_cols: bool = False,
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Merge multiple CSV files into one long-format file with an attribute column.

    Each source file contributes rows with an 'attribute' column indicating the source.
    The output has columns: [index_cols..., attribute, value]

    Parameters
    ----------
    output_dir : str
        Path to the directory containing CSV files
    file_names : list
        List of file names (without .csv extension) to merge
    output_name : str
        Name for the output merged file (without .csv extension)
    attribute_col : str
        Name for the attribute column (default: 'attribute')
    normalize_cost_component_cols : bool
        Rename cost component columns ('sumhdr' or 'genCostCmp') to 'uni' for consistency
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    dfs = []

    for file_name in file_names:
        csv_path = os.path.join(output_dir, f"{file_name}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        if normalize_cost_component_cols:
            if 'sumhdr' in df.columns:
                df = df.rename(columns={'sumhdr': 'uni'})
            if 'genCostCmp' in df.columns:
                df = df.rename(columns={'genCostCmp': 'uni'})
        if df.empty:
            continue

        # Identify if file has 'value' column
        if 'value' not in df.columns:
            log_func(f"[output_treatment]   {file_name}: WARNING - no 'value' column found, skipping")
            continue

        # Extract attribute name from file name (remove 'p' prefix)
        attr_name = file_name[1:] if file_name.startswith('p') else file_name

        # Add attribute column
        df[attribute_col] = attr_name

        dfs.append(df)

    if not dfs:
        log_func(f"[output_treatment]   {output_name}: WARNING - no files found to merge")
        return False

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns: put attribute before value
    cols = list(merged_df.columns)
    if attribute_col in cols and 'value' in cols:
        cols.remove(attribute_col)
        value_idx = cols.index('value')
        cols.insert(value_idx, attribute_col)
        merged_df = merged_df[cols]

    # Reorder columns according to preferred order
    merged_df = _reorder_columns(merged_df)

    # Save merged file
    output_path = os.path.join(output_dir, f"{output_name}.csv")
    merged_df.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   {output_name}.csv: merged {len(dfs)} files -> long format ({len(merged_df)} rows)")

    return True


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


def add_techfuel_column_to_dispatch(
    input_path: str,
    techfuel_mapping: Optional[Dict[str, str]] = None,
    tech_col: str = 'tech',
    fuel_col: str = 'f',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Add techfuel column to dispatch files that already have tech and fuel columns.

    This function is for files like pDispatchTechFuel that already have both
    'tech' and 'f' columns. It adds the 'techfuel' column with Processing names.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (dispatch data with 'tech' and 'f' columns)
    techfuel_mapping : dict, optional
        Dictionary mapping 'tech-fuel' keys to Processing names.
    tech_col : str
        Name of the technology column (default: 'tech')
    fuel_col : str
        Name of the fuel column (default: 'f')
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

    # Check if file already has tech and fuel columns
    if tech_col not in df.columns:
        log_func(f"[output_treatment]   {file_name}: WARNING - column '{tech_col}' not found, skipping")
        return False

    if fuel_col not in df.columns:
        log_func(f"[output_treatment]   {file_name}: WARNING - column '{fuel_col}' not found, skipping")
        return False

    # Skip if techfuel column already exists
    if 'techfuel' in df.columns:
        log_func(f"[output_treatment]   {file_name}: skipped (already has 'techfuel' column)")
        return True

    # Add techfuel column
    if techfuel_mapping:
        # Use mapping to get Processing names
        df['techfuel'] = (df[tech_col] + '-' + df[fuel_col]).map(techfuel_mapping)
        # Fallback to tech-fuel for unmapped values
        df['techfuel'] = df['techfuel'].fillna(df[tech_col] + '-' + df[fuel_col])
    else:
        # Just concatenate tech and fuel
        df['techfuel'] = df[tech_col] + '-' + df[fuel_col]

    # Reorder columns to have techfuel after tech and f
    cols = list(df.columns)
    cols.remove('techfuel')
    if fuel_col in cols:
        f_idx = cols.index(fuel_col)
        cols.insert(f_idx + 1, 'techfuel')
    df = df[cols]

    # Save back to file
    df.to_csv(input_path, index=False)
    log_func(f"[output_treatment]   {file_name}: added techfuel column ({len(df)} rows)")

    return True


def create_dispatch_complete(
    dispatch_path: str,
    dispatch_techfuel_path: str,
    output_path: str,
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Create pDispatchComplete by merging pDispatch with pDispatchTechFuel.

    pDispatch has: z, c, y, q, d, uni, t, value (where uni contains demand categories like Imports, Exports)
    pDispatchTechFuel has: z, c, y, q, d, tech, f, t, value (dispatch by technology and fuel)

    The output pDispatchComplete will have 'uni' containing both:
    - Demand categories (from pDispatch): Imports, Exports, etc.
    - Tech-fuel combinations (from pDispatchTechFuel): formatted as "tech-fuel" or using techfuel mapping

    Parameters
    ----------
    dispatch_path : str
        Path to pDispatch.csv
    dispatch_techfuel_path : str
        Path to pDispatchTechFuel.csv
    output_path : str
        Path to output pDispatchComplete.csv
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    dispatch_name = os.path.basename(dispatch_path)
    dispatch_techfuel_name = os.path.basename(dispatch_techfuel_path)
    output_name = os.path.basename(output_path)

    # Check file existence with verbose logging
    dispatch_exists = os.path.exists(dispatch_path)
    dispatch_techfuel_exists = os.path.exists(dispatch_techfuel_path)

    log_func(f"[output_treatment]     - {dispatch_name}: {'found' if dispatch_exists else 'NOT FOUND'}")
    log_func(f"[output_treatment]     - {dispatch_techfuel_name}: {'found' if dispatch_techfuel_exists else 'NOT FOUND'}")

    if not dispatch_exists:
        log_func(f"[output_treatment]   WARNING: Cannot create {output_name} - {dispatch_name} not found")
        return False

    if not dispatch_techfuel_exists:
        log_func(f"[output_treatment]   WARNING: Cannot create {output_name} - {dispatch_techfuel_name} not found")
        return False

    # Read dispatch files
    df_dispatch = pd.read_csv(dispatch_path)
    df_dispatch_techfuel = pd.read_csv(dispatch_techfuel_path)

    log_func(f"[output_treatment]     - {dispatch_name}: {len(df_dispatch)} rows, columns: {list(df_dispatch.columns)}")
    log_func(f"[output_treatment]     - {dispatch_techfuel_name}: {len(df_dispatch_techfuel)} rows, columns: {list(df_dispatch_techfuel.columns)}")

    # Create 'uni' column in dispatch_techfuel from tech-fuel combination
    # Check what columns are available
    if 'techfuel' in df_dispatch_techfuel.columns:
        # If techfuel column already exists (added by earlier processing), use it
        df_dispatch_techfuel['uni'] = df_dispatch_techfuel['techfuel']
        log_func(f"[output_treatment]     - Using existing 'techfuel' column for 'uni'")
    elif 'tech' in df_dispatch_techfuel.columns and 'f' in df_dispatch_techfuel.columns:
        # Create techfuel from tech and fuel columns
        df_dispatch_techfuel['uni'] = df_dispatch_techfuel['tech'] + '-' + df_dispatch_techfuel['f']
        log_func(f"[output_treatment]     - Created 'uni' from tech-f combination")
    elif 'f' in df_dispatch_techfuel.columns:
        # Fallback: use fuel column only
        df_dispatch_techfuel['uni'] = df_dispatch_techfuel['f']
        log_func(f"[output_treatment]     - Using 'f' column for 'uni' (fallback)")
    else:
        log_func(f"[output_treatment]   WARNING: Cannot create {output_name} - no tech/f/techfuel columns found")
        return False

    # Keep only the columns needed for merge (same structure as pDispatch)
    # Common structure: z, c, y, q, d, uni, t, value
    common_cols = ['z', 'c', 'y', 'q', 'd', 'uni', 't', 'value']

    # Select only columns that exist
    dispatch_cols = [c for c in common_cols if c in df_dispatch.columns]
    dispatch_techfuel_cols = [c for c in common_cols if c in df_dispatch_techfuel.columns]

    log_func(f"[output_treatment]     - Selecting columns from {dispatch_name}: {dispatch_cols}")
    log_func(f"[output_treatment]     - Selecting columns from {dispatch_techfuel_name}: {dispatch_techfuel_cols}")

    df_dispatch = df_dispatch[dispatch_cols]
    df_dispatch_techfuel = df_dispatch_techfuel[dispatch_techfuel_cols]

    # Concatenate both dataframes
    df_complete = pd.concat([df_dispatch, df_dispatch_techfuel], ignore_index=True)

    # Sort for consistent output
    sort_cols = [c for c in ['z', 'c', 'y', 'q', 'd', 'uni', 't'] if c in df_complete.columns]
    df_complete = df_complete.sort_values(by=sort_cols)

    # Save to output file
    df_complete.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   SUCCESS: {output_name} created ({len(df_complete)} rows)")

    return True


def aggregate_plant_to_techfuel(
    input_path: str,
    output_path: str,
    generator_techfuel_df: pd.DataFrame,
    generator_col: str = 'g',
    zone_col: str = 'z',
    year_col: str = 'y',
    value_col: str = 'value',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Aggregate plant-level data to tech-fuel level by merging with pGeneratorTechFuel and grouping.

    This transforms files like pReserveSpinningPlantZone (z, g, y, value) into
    tech-fuel aggregated files (z, tech, f, y, value) by summing values.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (plant-level data with 'g' column)
    output_path : str
        Path to output CSV file (tech-fuel aggregated data)
    generator_techfuel_df : pd.DataFrame
        DataFrame with (g, tech, f) mapping from pGeneratorTechFuel
    generator_col : str
        Name of the generator column (default: 'g')
    zone_col : str
        Name of the zone column (default: 'z')
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

    if generator_techfuel_df.empty:
        log_func(f"[output_treatment]   {input_name}: WARNING - no generator techfuel mapping provided")
        return False

    df = pd.read_csv(input_path)

    if generator_col not in df.columns:
        log_func(f"[output_treatment]   {input_name}: WARNING - column '{generator_col}' not found")
        return False

    # Merge with generator techfuel mapping to get tech and fuel columns
    df = df.merge(generator_techfuel_df, on=generator_col, how='left')

    # Group by zone, tech, fuel, year and sum the values
    group_cols = [zone_col, 'tech', 'f', year_col]
    # Only include columns that exist
    group_cols = [c for c in group_cols if c in df.columns]

    df_aggregated = df.groupby(group_cols, as_index=False)[value_col].sum()

    # Save to output file
    df_aggregated.to_csv(output_path, index=False)
    log_func(f"[output_treatment]   {input_name} -> {output_name}: aggregated to tech-fuel ({len(df_aggregated)} rows)")

    return True


def add_total_to_yearly_zone_merged(
    input_file_path: str,
    attribute_name: str,
    yearly_zone_merged_path: str,
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Calculate total per year and zone from a CSV file and add it to pYearlyZoneMerged.
    
    Sums across all component columns (e.g., uni, sumhdr, genCostCmp) grouped by zone and year,
    then appends the totals to pYearlyZoneMerged with the specified attribute name.
    
    Parameters
    ----------
    input_file_path : str
        Path to input CSV file (must have z, y, value columns)
    attribute_name : str
        Name for the attribute column in pYearlyZoneMerged
    yearly_zone_merged_path : str
        Path to pYearlyZoneMerged.csv file
    log_func : callable
        Logging function (default: print)
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if not os.path.exists(input_file_path):
        return False
    
    df = pd.read_csv(input_file_path)
    
    # Check for required columns
    if 'value' not in df.columns or 'z' not in df.columns or 'y' not in df.columns:
        return False
    
    # Sum across all component columns by zone and year
    df_total = df.groupby(['z', 'y'], as_index=False)['value'].sum()
    df_total['attribute'] = attribute_name
    # Reorder columns to match pYearlyZoneMerged format
    df_total = df_total[['z', 'y', 'attribute', 'value']]
    
    # Append to pYearlyZoneMerged or create it if it doesn't exist
    if os.path.exists(yearly_zone_merged_path):
        df_zone_merged = pd.read_csv(yearly_zone_merged_path)
        df_zone_merged = pd.concat([df_zone_merged, df_total], ignore_index=True)
        df_zone_merged = _reorder_columns(df_zone_merged)
        df_zone_merged.to_csv(yearly_zone_merged_path, index=False)
        log_func(f"[output_treatment]   pYearlyZoneMerged.csv: added {attribute_name} totals")
    else:
        # If pYearlyZoneMerged doesn't exist yet, create it
        df_total = _reorder_columns(df_total)
        df_total.to_csv(yearly_zone_merged_path, index=False)
        log_func(f"[output_treatment]   pYearlyZoneMerged.csv: created with {attribute_name} totals")
    
    return True


def add_country_to_zone_file(
    input_path: str,
    zone_country_df: pd.DataFrame,
    zone_col: str = 'z',
    country_col: str = 'c',
    log_func: Callable[[str], None] = _default_log
) -> bool:
    """
    Add country column to a zone-based CSV file by merging with pZoneCountry.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (zone-level data with 'z' column)
    zone_country_df : pd.DataFrame
        DataFrame with (z, c) mapping from pZoneCountry
    zone_col : str
        Name of the zone column in the input file (default: 'z')
    country_col : str
        Name of the country column to add (default: 'c')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    file_name = os.path.basename(input_path)

    if not os.path.exists(input_path):
        return False

    if zone_country_df.empty:
        log_func(f"[output_treatment]   {file_name}: WARNING - no zone-country mapping provided")
        return False

    df = pd.read_csv(input_path)

    # Skip if no zone column
    if zone_col not in df.columns:
        return False

    # Skip if country column already exists
    if country_col in df.columns:
        log_func(f"[output_treatment]   {file_name}: skipped (already has '{country_col}' column)")
        return False

    original_cols = list(df.columns)

    # Merge with zone-country mapping
    df = df.merge(zone_country_df, on=zone_col, how='left')

    # Reorder columns: put country right after zone
    z_idx = original_cols.index(zone_col)
    new_cols = original_cols[:z_idx + 1] + [country_col] + original_cols[z_idx + 1:]
    # Only include columns that exist
    new_cols = [c for c in new_cols if c in df.columns]
    df = df[new_cols]

    # Save back to file
    df.to_csv(input_path, index=False)
    log_func(f"[output_treatment]   {file_name}: added '{country_col}' column")

    return True


def organize_output_files(
    output_dir: str,
    keep_files: Optional[List[str]] = None,
    other_subdir: str = 'other',
    log_func: Callable[[str], None] = _default_log
) -> Tuple[int, int]:
    """
    Organize output files by moving non-essential CSVs to a subdirectory.

    Files in keep_files list stay in the main directory, all other CSV files
    are moved to the 'other' subdirectory.

    Parameters
    ----------
    output_dir : str
        Path to the output directory containing CSV files
    keep_files : list, optional
        List of file names (without .csv extension) to keep in main directory.
        If None, uses PRIMARY_OUTPUT_FILES + ESSENTIAL_FILES.
    other_subdir : str
        Name of subdirectory for other files (default: 'other')
    log_func : callable
        Logging function (default: print)

    Returns
    -------
    tuple
        (kept_count, moved_count) - number of files kept and moved
    """
    import shutil

    if keep_files is None:
        keep_files = PRIMARY_OUTPUT_FILES + ESSENTIAL_FILES

    # Create set of files to keep (with .csv extension)
    keep_set = {f"{name}.csv" for name in keep_files}

    # Create other subdirectory if needed
    other_dir = os.path.join(output_dir, other_subdir)

    # Find all CSV files in output directory
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

    kept_count = 0
    moved_count = 0

    for csv_file in sorted(csv_files):
        if csv_file in keep_set:
            kept_count += 1
            log_func(f"[output_treatment]   KEEP: {csv_file}")
        else:
            # Create other directory only when we have files to move
            if not os.path.exists(other_dir):
                os.makedirs(other_dir)
                log_func(f"[output_treatment]   Created subdirectory: {other_subdir}/")

            src_path = os.path.join(output_dir, csv_file)
            dst_path = os.path.join(other_dir, csv_file)
            shutil.move(src_path, dst_path)
            moved_count += 1
            log_func(f"[output_treatment]   MOVE: {csv_file} -> {other_subdir}/")

    return kept_count, moved_count


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
    zone_country_df: Optional[pd.DataFrame] = None,
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
    zone_country_df : pd.DataFrame, optional
        DataFrame with (z, c) mapping from pZoneCountry. Required for adding country to zone-based files.
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

        # Process standard TechFuel files (fill missing tech-fuel combinations)
        for file_name in techfuel_files:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            fill_techfuel_combinations(csv_path, techfuel_df, techfuel_mapping=techfuel_mapping, log_func=log_func)

        # Note: Plant files are NOT filled with all tech-fuel combinations
        # Each generator has exactly one tech-fuel, so no filling is needed

        # Process dispatch techfuel files (add techfuel column with Processing names)
        for file_name in DISPATCH_FUEL_FILES:
            csv_path = os.path.join(output_dir, f"{file_name}.csv")
            add_techfuel_column_to_dispatch(csv_path, techfuel_mapping=techfuel_mapping, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 3: Skipping TechFuel filling (no techfuel_df provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 3b. Create pDispatchComplete (independent of techfuel_df)
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 3b: Creating pDispatchComplete")
    log_func("-" * 60)

    dispatch_path = os.path.join(output_dir, "pDispatch.csv")
    dispatch_techfuel_path = os.path.join(output_dir, "pDispatchTechFuel.csv")
    dispatch_complete_path = os.path.join(output_dir, "pDispatchComplete.csv")
    create_dispatch_complete(dispatch_path, dispatch_techfuel_path, dispatch_complete_path, log_func=log_func)

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

    # ---------------------------------------------------------
    # 6. Aggregate plant files to TechFuel level
    # ---------------------------------------------------------
    if generator_techfuel_df is not None and not generator_techfuel_df.empty:
        log_func("")
        log_func("[output_treatment] STEP 6: Aggregating plant files to TechFuel level")
        log_func("-" * 60)

        for input_name, output_name in PLANT_TO_TECHFUEL_AGGREGATIONS:
            input_path = os.path.join(output_dir, f"{input_name}.csv")
            output_path = os.path.join(output_dir, f"{output_name}.csv")
            aggregate_plant_to_techfuel(input_path, output_path, generator_techfuel_df, log_func=log_func)
    else:
        log_func("")
        log_func("[output_treatment] STEP 6: Skipping plant-to-techfuel aggregation (no generator_techfuel_df provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 7. Merge related CSV files into consolidated files (long format)
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 7: Merging related CSV files into consolidated files (long format)")
    log_func("-" * 60)

    # Merge TechFuel files (z, tech, f, y) -> long format with attribute column
    merge_csv_files_long(output_dir, TECHFUEL_MERGE_FILES, 'pTechFuelMerged', log_func=log_func)

    # Merge Plant files (z, g, y) -> long format with attribute column
    merge_csv_files_long(output_dir, PLANT_MERGE_FILES, 'pPlantMerged', log_func=log_func)

    # Add techfuel column to pPlantMerged (if tech and f columns exist)
    plant_merged_path = os.path.join(output_dir, 'pPlantMerged.csv')
    if os.path.exists(plant_merged_path):
        df_plant_merged = pd.read_csv(plant_merged_path)
        if 'tech' in df_plant_merged.columns and 'f' in df_plant_merged.columns:
            if 'techfuel' not in df_plant_merged.columns:
                if techfuel_mapping:
                    df_plant_merged['techfuel'] = (df_plant_merged['tech'] + '-' + df_plant_merged['f']).map(techfuel_mapping)
                    df_plant_merged['techfuel'] = df_plant_merged['techfuel'].fillna(df_plant_merged['tech'] + '-' + df_plant_merged['f'])
                else:
                    df_plant_merged['techfuel'] = df_plant_merged['tech'] + '-' + df_plant_merged['f']
                # Reorder columns to have techfuel after f
                cols = list(df_plant_merged.columns)
                cols.remove('techfuel')
                if 'f' in cols:
                    f_idx = cols.index('f')
                    cols.insert(f_idx + 1, 'techfuel')
                df_plant_merged = df_plant_merged[cols]
                df_plant_merged = _reorder_columns(df_plant_merged)
                df_plant_merged.to_csv(plant_merged_path, index=False)
                log_func(f"[output_treatment]   pPlantMerged.csv: added techfuel column")

    # Merge YearlyCosts zone files (z, uni, y) -> long format
    merge_csv_files_long(
        output_dir, YEARLY_COSTS_MERGE_FILES, 'pYearlyCostsMerged',
        normalize_cost_component_cols=True, log_func=log_func
    )

    # Merge Transmission/interconnection files (z, z2, y) -> long format
    merge_csv_files_long(output_dir, TRANSMISSION_MERGE_FILES, 'pTransmissionMerged', log_func=log_func)

    # Merge Yearly zone files (demand, emissions) (z, y) -> long format
    merge_csv_files_long(output_dir, YEARLY_ZONE_MERGE_FILES, 'pYearlyZoneMerged', log_func=log_func)

    # Add totals from various cost and investment files to pYearlyZoneMerged
    yearly_zone_merged_path = os.path.join(output_dir, 'pYearlyZoneMerged.csv')
    
    # List of files to add: (file_name, attribute_name)
    files_to_add = [
        ('pYearlyCostsZone', 'YearlyCostsZone'),
        ('pYearlyCostsZonePerMWh', 'YearlyCostsZonePerMWh'),
        ('pYearlyGenCostZonePerMWh', 'YearlyGenCostZonePerMWh'),
        ('pCapexInvestmentComponent', 'CapexInvestmentComponent'),
        ('pCapexInvestmentComponentCumulated', 'CapexInvestmentComponentCumulated'),
    ]
    
    for file_name, attribute_name in files_to_add:
        file_path = os.path.join(output_dir, f"{file_name}.csv")
        add_total_to_yearly_zone_merged(file_path, attribute_name, yearly_zone_merged_path, log_func=log_func)

    # Merge System costs files -> long format
    merge_csv_files_long(output_dir, SYSTEM_COSTS_MERGE_FILES, 'pCostsSystemMerged', log_func=log_func)

    # ---------------------------------------------------------
    # 8. Add country column to all zone-based files
    # ---------------------------------------------------------
    if zone_country_df is not None and not zone_country_df.empty:
        log_func("")
        log_func("[output_treatment] STEP 8: Adding country column to zone-based files")
        log_func("-" * 60)

        # Find all CSV files in the output directory
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        processed_count = 0

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(output_dir, csv_file)
            if add_country_to_zone_file(csv_path, zone_country_df, log_func=log_func):
                processed_count += 1

        log_func(f"[output_treatment]   Processed {processed_count} zone-based files")
    else:
        log_func("")
        log_func("[output_treatment] STEP 8: Skipping country column addition (no zone_country_df provided)")
        log_func("-" * 60)

    # ---------------------------------------------------------
    # 9. Organize output files (move non-essential to 'other' subdir)
    # ---------------------------------------------------------
    log_func("")
    log_func("[output_treatment] STEP 9: Organizing output files")
    log_func("-" * 60)

    kept_count, moved_count = organize_output_files(output_dir, log_func=log_func)
    log_func(f"[output_treatment]   Summary: {kept_count} files kept, {moved_count} files moved to 'other/'")

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
        zone_country_df = pd.DataFrame(columns=['z', 'c'])

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

            # Get zone to country mapping from pZoneCountry
            if 'pZoneCountry' in db.data:
                zone_country_data = db.data['pZoneCountry']
                if zone_country_data.records is not None and len(zone_country_data.records) > 0:
                    zone_country_df = zone_country_data.records.iloc[:, :2].copy()
                    zone_country_df.columns = ['z', 'c']
                    zone_country_df = zone_country_df.drop_duplicates()
                    log_func(f"[output_treatment]   Found {len(zone_country_df)} zone-country mappings from pZoneCountry")

        except Exception as e:
            log_func(f"[output_treatment]   WARNING: Could not extract mappings from GAMS: {e}")

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
            zone_country_df=zone_country_df,
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
