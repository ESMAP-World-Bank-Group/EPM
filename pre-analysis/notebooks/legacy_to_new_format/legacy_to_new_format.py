"""Convert a legacy EPM GDX file into the newer CSV layout used by prepare-data.

How it works (summary):
- Reads a legacy GDX with the GAMS transfer API and a mapping table (`symbol_mapping.csv`)
  that aligns old GDX symbol names to the expected CSV names.
- For each symbol in `symbol_mapping.csv`, reads the GDX data and optionally unstacks a
  specified column to create wide-format CSV files. Writes to `{folder}/{csv_symbol}.csv`.
  Optional symbols missing in the GDX are written as empty stubs.
- Any extra GDX symbols not covered by the mapping are dumped to `output/data/extras/`
  for inspection.
- Run from the CLI (`python legacy_to_new_format.py --gdx ... --mapping ...`) to batch-convert
  a GDX; defaults point to the sample inputs/outputs in this folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import gams.transfer as gt
except ImportError as err:  # pragma: no cover - runtime dependency
    raise ImportError("Install the GAMS Python API before running this script.") from err

# --------------------------------------------------------------------------- #
# User-editable defaults
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
# Repo root two levels up from pre-analysis/notebooks/legacy_to_new_format
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_GDX_PATH = SCRIPT_DIR / "input" / "input_epm_Turkiye_v8.gdx"
DEFAULT_MAPPING_PATH = SCRIPT_DIR / "symbol_mapping.csv"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output"
DEFAULT_TARGET_FOLDER = "data"


# CSV layout is built from symbol_mapping.csv at runtime
CSV_LAYOUT: List[Dict] = []

OPTIONAL_SYMBOLS = {
    "pAvailability",
    "pAvailabilityDefault",
    "pAvailabilityH2",
    "pCSPData",
    "pCapexTrajectories",
    "pCapexTrajectoriesDefault",
    "pCapexTrajectoryH2",
    "pCarbonPrice",
    "pDemandData",
    "pDemandForecast",
    "pDemandProfile",
    "pEmissionsCountry",
    "pEmissionsTotal",
    "pEnergyEfficiencyFactor",
    "pExtTransferLimit",
    "pExternalH2",
    "pFuelCarbonContent",
    "pFuelDataH2",
    "pFuelPrice",
    "pGenDataInputDefault",
    "pH2DataExcel",
    "pHours",
    "pLossFactorInternal",
    "pMaxFuellimit",
    "pMaxPriceImportShare",
    "pMinImport",
    "pNewTransmission",
    "pPlanningReserveMargin",
    "pSpinningReserveReqCountry",
    "pSpinningReserveReqSystem",
    "pStorageDataInput",
    "pTradePrice",
    "pTransferLimit",
    "pVREProfile",
    "pVREgenProfile",
    "zext",
    "sRelevant",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_header_table(df_gdx_records: pd.DataFrame, spec: dict, csv_symbol: str = "", gdx_symbol: str = "") -> pd.DataFrame:
    """Transform GDX records into CSV format by unstacking a specific column.
    
    This function takes GDX records (which have all dimensions as columns plus a value column)
    and unstacks the column at the specified index to become CSV column headers (wide format).
    
    Args:
        df_gdx_records: DataFrame from GDX records (all dimensions + value column)
        spec: Layout specification dict with header (containing column index)
        csv_symbol: CSV symbol name (for error messages)
        gdx_symbol: GDX symbol name (for error messages)
    
    Returns:
        DataFrame in CSV format with index columns + unstacked header column
    """
    # Detect value column if present
    value_col = None
    for candidate in ("value", "Value"):
        if candidate in df_gdx_records.columns:
            value_col = candidate
            break
    
    # All columns except the value column are dimension columns from GDX
    gdx_dimension_cols = [col for col in df_gdx_records.columns if col != value_col]
    
    # Get the column index to unstack (should be a single index in spec["header"])
    if not spec["header"]:
        # No unstacking: return data as-is
        if value_col:
            return df_gdx_records.rename(columns={value_col: "value"}).reset_index(drop=True)
        return df_gdx_records.reset_index(drop=True)
    
    header_col_index = spec["header"][0]  # 0-indexed column index to unstack
    
    # Validate header_col_index is within bounds
    if header_col_index >= len(gdx_dimension_cols):
        symbol_info = f" (CSV: '{csv_symbol}', GDX: '{gdx_symbol}')" if csv_symbol or gdx_symbol else ""
        raise IndexError(
            f"header_cols index {header_col_index} is out of bounds for {len(gdx_dimension_cols)} dimension columns"
            f"{symbol_info}. "
            f"Available dimension columns (0-indexed): {list(range(len(gdx_dimension_cols)))} "
            f"with names: {gdx_dimension_cols}"
        )
    
    # Columns before header_col_index become CSV index columns
    csv_index_cols = gdx_dimension_cols[:header_col_index]
    # Column at header_col_index becomes header (unstacked)
    header_col = gdx_dimension_cols[header_col_index]
    
    if value_col:
        # Unstack: specific column → CSV column headers
        pivot = df_gdx_records.pivot_table(
            index=csv_index_cols if csv_index_cols else None,
            columns=header_col,
            values=value_col,
            aggfunc="first",
            observed=False,
        )
        pivot.columns = [col if isinstance(col, str) else "_".join(map(str, col)) for col in pivot.columns]
        return pivot.reset_index().reset_index(drop=True)
    
    # No value column: return data as-is
    return df_gdx_records.reset_index(drop=True)


def postprocess_pGenDataInput(csv_path: Path, extras_root: Path, repo_root: Path) -> None:
    """Apply optional post-processing rules to pGenDataInput CSV file.
    
    Rules applied:
    - Rename first column to "gen"
    - Find and rename fuel, tech, zone columns
    - Reorder columns: gen, zone, tech, fuel, then others
    - Apply lookup mappings from extras folder
    
    All operations are optional and fail gracefully.
    Reads from CSV file and writes back to the same file.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [pGenDataInput] Failed to read {csv_path}: {e}")
        return
    
    if df.empty or len(df.columns) == 0:
        return
    
    # Rename first column to "gen"
    first_col = df.columns[0]
    if first_col != "gen":
        df = df.rename(columns={first_col: "gen"})
    
    # Find fuel column (case-insensitive, could be fuel, fuel1, Fuel, etc.)
    fuel_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "fuel" or col_lower.startswith("fuel"):
            fuel_col = col
            break
    
    # Find tech column (could be tech or type)
    tech_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "tech" or col_lower == "type":
            tech_col = col
            break
    
    # Find zone column
    zone_col = None
    for col in df.columns:
        if col.lower() == "zone":
            zone_col = col
            break
    
    # Rename columns to standard names
    rename_dict = {}
    if zone_col and zone_col != "zone":
        rename_dict[zone_col] = "zone"
        zone_col = "zone"
    if tech_col and tech_col != "tech":
        rename_dict[tech_col] = "tech"
        tech_col = "tech"
    if fuel_col and fuel_col != "fuel":
        rename_dict[fuel_col] = "fuel"
        fuel_col = "fuel"
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Load expected column order from pGenDataInputHeader
    header_file = repo_root / "epm" / "resources" / "pGenDataInputHeader.csv"
    header_cols = []
    try:
        if header_file.exists():
            header_df = pd.read_csv(header_file)
            header_cols = header_df.iloc[:, 0].tolist()
            # Remove header row name if present
            if len(header_cols) > 0 and header_cols[0] == "pGenDataInputHeader":
                header_cols = header_cols[1:]
            print(f"  [pGenDataInput] Loaded {len(header_cols)} columns from pGenDataInputHeader.csv")
            
            # Remove columns not in header (keep gen, zone, tech, fuel)
            columns_to_keep = ["gen"]
            if zone_col:
                columns_to_keep.append(zone_col)
            if tech_col:
                columns_to_keep.append(tech_col)
            if fuel_col:
                columns_to_keep.append(fuel_col)
            columns_to_keep.extend(header_cols)
            
            columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                print(f"  [pGenDataInput] Removed {len(columns_to_remove)} columns not in pGenDataInputHeader: {', '.join(columns_to_remove)}")
        else:
            print(f"  [pGenDataInput] pGenDataInputHeader.csv not found at {header_file}")
    except Exception as e:
        print(f"  [pGenDataInput] Failed to load pGenDataInputHeader.csv: {e}")
    
    # Build column order: gen, zone, tech, fuel, then others in header order
    ordered_cols = ["gen"]
    if zone_col:
        if zone_col not in ordered_cols:
            ordered_cols.append(zone_col)
        print(f"  [pGenDataInput] Found zone column: {zone_col}")
    else:
        print(f"  [pGenDataInput] Zone column not found (searched for 'zone')")
    
    if tech_col:
        if tech_col not in ordered_cols:
            ordered_cols.append(tech_col)
        print(f"  [pGenDataInput] Found tech column: {tech_col}")
    else:
        print(f"  [pGenDataInput] Tech column not found (searched for 'tech' or 'type')")
    
    if fuel_col:
        if fuel_col not in ordered_cols:
            ordered_cols.append(fuel_col)
        print(f"  [pGenDataInput] Found fuel column: {fuel_col}")
    else:
        print(f"  [pGenDataInput] Fuel column not found (searched for 'fuel' or 'fuel*')")
    
    # Add remaining columns in header order, then any others
    for col in header_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    
    # Add any remaining columns not yet included
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    print(f"  [pGenDataInput] Column order: {', '.join(ordered_cols)}")
    
    # Reorder columns
    df = df[ordered_cols]
    
    # Apply lookup mappings from extras folder
    # Zone mapping from pZoneIndex
    if zone_col:
        zone_file = extras_root / "pZoneIndex.csv"
        try:
            if zone_file.exists():
                zone_lookup = pd.read_csv(zone_file)
                if len(zone_lookup.columns) >= 2:
                    zone_map = dict(zip(zone_lookup.iloc[:, 1], zone_lookup.iloc[:, 0]))
                    mapped_series = df[zone_col].map(zone_map)
                    mapped_count = mapped_series.notna().sum()
                    df[zone_col] = mapped_series.where(mapped_series.notna(), df[zone_col])
                    print(f"  [pGenDataInput] Applied zone mapping from pZoneIndex.csv ({mapped_count}/{len(df)} values mapped)")
                else:
                    print(f"  [pGenDataInput] pZoneIndex.csv has insufficient columns ({len(zone_lookup.columns)} < 2)")
            else:
                print(f"  [pGenDataInput] pZoneIndex.csv not found at {zone_file}")
        except Exception as e:
            print(f"  [pGenDataInput] Failed to apply zone mapping: {e}")
    
    # Tech mapping from pTechDataExcel
    if tech_col:
        tech_file = extras_root / "pTechDataExcel.csv"
        try:
            if tech_file.exists():
                tech_lookup = pd.read_csv(tech_file)
                # Only keep Assigned Value in the second column
                tech_lookup = tech_lookup[tech_lookup[tech_lookup.columns[1]] == "Assigned Value"]
                
                if len(tech_lookup.columns) >= 1:
                    tech_map = dict(zip(tech_lookup.iloc[:, 2], tech_lookup.iloc[:, 0]))
                    mapped_series = df[tech_col].map(tech_map)
                    mapped_count = mapped_series.notna().sum()
                    df[tech_col] = mapped_series.where(mapped_series.notna(), df[tech_col])
                    print(f"  [pGenDataInput] Applied tech mapping from pTechDataExcel.csv ({mapped_count}/{len(df)} values mapped)")
                else:
                    print(f"  [pGenDataInput] pTechDataExcel.csv has no columns")
            else:
                print(f"  [pGenDataInput] pTechDataExcel.csv not found at {tech_file}")
        except Exception as e:
            print(f"  [pGenDataInput] Failed to apply tech mapping: {e}")
    
    # Fuel mapping from ftfindex
    if fuel_col:
        fuel_file = extras_root / "ftfindex.csv"
        try:
            if fuel_file.exists():
                fuel_lookup = pd.read_csv(fuel_file)
                if len(fuel_lookup.columns) >= 1:
                    fuel_map = dict(zip(fuel_lookup.iloc[:, 2], fuel_lookup.iloc[:, 0]))
                    mapped_series = df[fuel_col].map(fuel_map)
                    mapped_count = mapped_series.notna().sum()
                    df[fuel_col] = mapped_series.where(mapped_series.notna(), df[fuel_col])
                    print(f"  [pGenDataInput] Applied fuel mapping from ftfindex.csv ({mapped_count}/{len(df)} values mapped)")
                else:
                    print(f"  [pGenDataInput] ftfindex.csv has no columns")
            else:
                print(f"  [pGenDataInput] ftfindex.csv not found at {fuel_file}")
        except Exception as e:
            print(f"  [pGenDataInput] Failed to apply fuel mapping: {e}")
    
    # Write back to CSV file
    try:
        df.to_csv(csv_path, index=False, na_rep="")
    except Exception as e:
        print(f"  [pGenDataInput] Failed to write {csv_path}: {e}")


def postprocess_csv(csv_path: Path, output_root: Path, extras_root: Path, repo_root: Path) -> None:
    """Apply post-processing to a single CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    
    if df.empty:
        return
    
    file_name = csv_path.name
    folder = csv_path.parent.name if csv_path.parent != output_root else ""
    file_path_str = f"{folder}/{file_name}" if folder else file_name
    
    changes = []
    
    # Remove element_text column
    if "element_text" in df.columns:
        df = df.drop(columns=["element_text"])
        changes.append("removed element_text")
    
    # Apply file-specific rules (rename columns by position)
    if file_name == "pCarbonPrice.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "year"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→year")
    
    elif file_name == "pEmissionsTotal.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "year"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→year")
    
    elif file_name == "pDemandForecast.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "type"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pDemandProfile.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "season"
        if len(df.columns) > 2:
            rename_dict[df.columns[2]] = "daytype"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pHours.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "season"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "daytype"
        if len(df.columns) > 2:
            rename_dict[df.columns[2]] = "year"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pSpinningReserveReqSystem.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "year"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→year")
    
    elif file_name == "pMaxAnnualExternalTradeShare.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "year"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→year")
    
    elif file_name == "pAvailabilityCustom.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "gen"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→gen")
    
    elif file_name == "pCapexTrajectoriesCustom.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "gen"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→gen")
    
    elif file_name == "pStorageDataInput.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "gen"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "Linked plants"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
        
        additional_renames = {}
        if "Capacity" in df.columns:
            additional_renames["Capacity"] = "CapacityMWh"
        if "Capex" in df.columns:
            additional_renames["Capex"] = "CapexMWh"
        if "FixedOM" in df.columns:
            additional_renames["FixedOM"] = "FixedOMMWh"
        if "VOMMWh" in df.columns:
            additional_renames["VOMMWh"] = "VOM"
        if additional_renames:
            df = df.rename(columns=additional_renames)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in additional_renames.items())}")
    
    elif file_name == "pVREProfile.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "tech"
        if len(df.columns) > 2:
            rename_dict[df.columns[2]] = "season"
        if len(df.columns) > 3:
            rename_dict[df.columns[3]] = "daytype"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pExtTransferLimit.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "zext"
        if len(df.columns) > 2:
            rename_dict[df.columns[2]] = "season"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pLossFactorInternal.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "From"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "To"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "pTradePrice.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "season"
        if len(df.columns) > 2:
            rename_dict[df.columns[2]] = "daytype"
        if len(df.columns) > 3:
            rename_dict[df.columns[3]] = "year"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    elif file_name == "y.csv":
        if len(df.columns) > 0:
            rename_dict = {df.columns[0]: "year"}
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {list(rename_dict.keys())[0]}→year")
    
    elif file_name == "zcmap.csv":
        rename_dict = {}
        if len(df.columns) > 0:
            rename_dict[df.columns[0]] = "zone"
        if len(df.columns) > 1:
            rename_dict[df.columns[1]] = "country"
        if rename_dict:
            df = df.rename(columns=rename_dict)
            changes.append(f"renamed: {', '.join(f'{k}→{v}' for k, v in rename_dict.items())}")
    
    # Write back
    try:
        df.to_csv(csv_path, index=False, na_rep="")
        if changes:
            print(f"  [{file_name}] {', '.join(changes)}")
    except Exception:
        pass


def merge_storage_from_gen_to_storage(output_root: Path, repo_root: Path) -> None:
    """Merge Storage rows from pGenDataInput into pStorageDataInput."""
    pGenDataInput_path = output_root / "supply" / "pGenDataInput.csv"
    pStorageDataInput_path = output_root / "supply" / "pStorageDataInput.csv"
    
    print(f"[Post-processing] Merging Storage rows from pGenDataInput to pStorageDataInput")
    
    if not pGenDataInput_path.exists():
        print(f"  [Storage merge] pGenDataInput.csv not found at {pGenDataInput_path}")
        return
    
    if not pStorageDataInput_path.exists():
        print(f"  [Storage merge] pStorageDataInput.csv not found at {pStorageDataInput_path}")
        return
    
    try:
        # Load both files
        df_gen = pd.read_csv(pGenDataInput_path)
        df_storage = pd.read_csv(pStorageDataInput_path)
        
        print(f"  [Storage merge] Loaded pGenDataInput: {len(df_gen)} rows, pStorageDataInput: {len(df_storage)} rows")
        
        if df_gen.empty:
            print(f"  [Storage merge] pGenDataInput is empty, nothing to merge")
            return
        
        if "tech" not in df_gen.columns:
            print(f"  [Storage merge] No 'tech' column in pGenDataInput")
            return
        
        # Find rows where tech contains "Storage" or "Sto" but exclude "STO HY" (reservoir)
        tech_lower = df_gen["tech"].astype(str).str.lower()
        storage_mask = tech_lower.str.contains("storage|sto", na=False, regex=True) & ~tech_lower.str.contains("sto hy|storage hy|reservoir", na=False, regex=True)
        storage_rows = df_gen[storage_mask].copy()
        
        if storage_rows.empty:
            print(f"  [Storage merge] No Storage rows found in pGenDataInput (searched for 'storage' or 'sto' in tech column)")
            return
        
        print(f"  [Storage merge] Found {len(storage_rows)} Storage rows in pGenDataInput")
        if "gen" in storage_rows.columns:
            storage_gens = storage_rows["gen"].astype(str).tolist()
            print(f"  [Storage merge] Storage gen values: {', '.join(storage_gens[:10])}{'...' if len(storage_gens) > 10 else ''}")
        
        # Merge: keep existing values in pStorageDataInput when gen matches
        if "gen" not in df_storage.columns:
            print(f"  [Storage merge] No 'gen' column in pStorageDataInput, cannot merge")
            return
        
        if "gen" not in storage_rows.columns:
            print(f"  [Storage merge] No 'gen' column in Storage rows, cannot merge")
            return
        
        # Convert gen to string for consistent matching
        df_storage["gen"] = df_storage["gen"].astype(str)
        storage_rows["gen"] = storage_rows["gen"].astype(str)
        
        # Get gens that already exist in pStorageDataInput
        existing_gens = set(df_storage["gen"])
        print(f"  [Storage merge] pStorageDataInput has {len(existing_gens)} existing gen values")
        
        # Separate existing and new rows
        existing_rows = storage_rows[storage_rows["gen"].isin(existing_gens)]
        new_rows = storage_rows[~storage_rows["gen"].isin(existing_gens)]
        
        # Merge existing rows: combine all columns, keep pStorageDataInput values when duplicate
        if len(existing_rows) > 0:
            existing_gens_list = existing_rows["gen"].tolist()
            print(f"  [Storage merge] Merging {len(existing_rows)} existing rows: {', '.join(existing_gens_list[:10])}{'...' if len(existing_gens_list) > 10 else ''}")
            
            # Set gen as index for merging
            df_storage_indexed = df_storage.set_index("gen")
            existing_rows_indexed = existing_rows.set_index("gen")
            
            # For each existing gen, add new columns from pGenDataInput
            # Keep existing values in pStorageDataInput (don't overwrite)
            for gen in existing_gens_list:
                if gen in existing_rows_indexed.index:
                    for col in existing_rows_indexed.columns:
                        if col not in df_storage_indexed.columns:
                            # New column - add it
                            df_storage_indexed[col] = None
                            df_storage_indexed.loc[gen, col] = existing_rows_indexed.loc[gen, col]
                        else:
                            # Column exists - only update if pStorageDataInput value is empty/null
                            if pd.isna(df_storage_indexed.loc[gen, col]) or df_storage_indexed.loc[gen, col] == "":
                                val = existing_rows_indexed.loc[gen, col]
                                if not pd.isna(val) and val != "":
                                    df_storage_indexed.loc[gen, col] = val
            
            df_storage = df_storage_indexed.reset_index()
        
        # Add new rows
        if len(new_rows) > 0:
            print(f"  [Storage merge] Adding {len(new_rows)} new rows to pStorageDataInput")
            new_gens = new_rows["gen"].tolist()
            print(f"  [Storage merge] New gen values: {', '.join(new_gens[:10])}{'...' if len(new_gens) > 10 else ''}")
            df_storage = pd.concat([df_storage, new_rows], ignore_index=True)
        
        print(f"  [Storage merge] Merged: pStorageDataInput now has {len(df_storage)} rows")
        
        # Load pStorageDataHeader for column ordering
        header_file = repo_root / "epm" / "resources" / "pStorageDataHeader.csv"
        header_cols = []
        try:
            if header_file.exists():
                header_df = pd.read_csv(header_file)
                header_cols = header_df.iloc[:, 0].tolist()
                if len(header_cols) > 0 and header_cols[0] == "pStorageDataHeader":
                    header_cols = header_cols[1:]
                print(f"  [Storage merge] Loaded {len(header_cols)} columns from pStorageDataHeader.csv")
            else:
                print(f"  [Storage merge] pStorageDataHeader.csv not found at {header_file}")
        except Exception as e:
            print(f"  [Storage merge] Failed to load pStorageDataHeader.csv: {e}")
        
        # Reorder columns: gen, zone, tech, fuel, Linked plants, then header order
        ordered_cols = []
        priority_cols = ["gen", "zone", "tech", "fuel", "Linked plants"]
        for col in priority_cols:
            if col in df_storage.columns and col not in ordered_cols:
                ordered_cols.append(col)
        
        for col in header_cols:
            if col in df_storage.columns and col not in ordered_cols:
                ordered_cols.append(col)
        
        for col in df_storage.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        df_storage = df_storage[ordered_cols]
        df_storage.to_csv(pStorageDataInput_path, index=False, na_rep="")
        print(f"  [Storage merge] Reordered columns in pStorageDataInput: {', '.join(ordered_cols[:10])}{'...' if len(ordered_cols) > 10 else ''}")
        
        # Remove Storage rows from pGenDataInput
        df_gen = df_gen[~storage_mask]
        df_gen.to_csv(pGenDataInput_path, index=False, na_rep="")
        print(f"  [Storage merge] Removed {len(storage_rows)} Storage rows from pGenDataInput (now has {len(df_gen)} rows)")
        
    except Exception as e:
        print(f"  [Storage merge] Failed: {e}")
        import traceback
        traceback.print_exc()


def apply_postprocessing(output_root: Path, extras_root: Path, repo_root: Path) -> None:
    """Apply post-processing rules to exported CSV files.
    
    This function is called after all CSV files have been exported.
    It applies symbol-specific post-processing rules.
    """
    # Post-process pGenDataInput (has special logic)
    pGenDataInput_path = output_root / "supply" / "pGenDataInput.csv"
    if pGenDataInput_path.exists():
        print(f"[Post-processing] Applying rules to pGenDataInput.csv")
        postprocess_pGenDataInput(pGenDataInput_path, extras_root, repo_root)
    
    # Post-process all other CSV files
    for csv_file in output_root.rglob("*.csv"):
        if csv_file.name == "pGenDataInput.csv":
            continue
        postprocess_csv(csv_file, output_root, extras_root, repo_root)
    
    # Merge Storage rows from pGenDataInput to pStorageDataInput (after all post-processing)
    merge_storage_from_gen_to_storage(output_root, repo_root)


def build_frame(container: gt.Container, gdx_symbol: str, csv_symbol: str, spec: dict) -> Optional[pd.DataFrame]:
    """Fetch GDX symbol and transform it to CSV format according to spec.
    
    This is the main transformation function that converts GDX data to CSV format.
    Sets and parameters are treated the same - no special handling.
    
    Args:
        container: GDX container with loaded symbols
        gdx_symbol: GDX symbol name to fetch from container
        csv_symbol: CSV symbol name (for error messages and reference)
        spec: Layout specification dict defining the transformation rules
    
    Returns:
        DataFrame in CSV format, or None if GDX symbol not found/empty
    """
    if gdx_symbol not in container:
        return None
    gdx_records = container[gdx_symbol].records
    if gdx_records is None:
        return None
    df_gdx_data = gdx_records.copy()
    
    # If header_cols is specified, unstack that column; otherwise return data as-is
    if spec.get("header"):
        return format_header_table(df_gdx_data, spec, csv_symbol=csv_symbol, gdx_symbol=gdx_symbol)
    
    # No unstacking: return data as-is (with value column renamed if present)
    for candidate in ("value", "Value"):
        if candidate in df_gdx_data.columns:
            return df_gdx_data.rename(columns={candidate: "value"}).reset_index(drop=True)
    return df_gdx_data.reset_index(drop=True)


def fallback_frame(container: gt.Container, gdx_symbol: str) -> Optional[pd.DataFrame]:
    """Export GDX symbol as-is (no transformation) for symbols not in CSV_LAYOUT.
    
    Used for extra GDX symbols that don't have a CSV layout specification.
    
    Args:
        container: GDX container with loaded symbols
        gdx_symbol: GDX symbol name to export
    
    Returns:
        DataFrame with GDX records as-is, or None if not found/empty
    """
    if gdx_symbol not in container:
        return None
    gdx_records = container[gdx_symbol].records
    if gdx_records is None:
        return None
    return gdx_records.copy().reset_index(drop=True)


def empty_frame_from_spec() -> pd.DataFrame:
    """Return an empty DataFrame used for optional placeholders."""
    return pd.DataFrame()


def load_symbol_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load symbol mapping from CSV file.
    
    The mapping file defines how CSV symbol names map to GDX symbol names,
    along with folder and header_cols configuration.
    
    Args:
        mapping_path: Path to symbol_mapping.csv file
    
    Returns:
        DataFrame with columns: csv_symbol, gdx_symbol, folder, header_cols
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping table: {mapping_path}. Populate it before continuing.")

    symbol_mapping_df = pd.read_csv(mapping_path)
    required_cols = {"csv_symbol", "gdx_symbol", "folder", "header_cols"}
    if required_cols - set(symbol_mapping_df.columns):
        raise ValueError(f"symbol_mapping.csv must contain columns: {', '.join(required_cols)}")

    symbol_mapping_df = symbol_mapping_df.fillna("").drop_duplicates(subset="csv_symbol", keep="last")
    # Use csv_symbol as fallback if gdx_symbol is empty
    symbol_mapping_df["gdx_symbol"] = symbol_mapping_df["gdx_symbol"].replace("", pd.NA)
    symbol_mapping_df["gdx_symbol"] = symbol_mapping_df["gdx_symbol"].fillna(symbol_mapping_df["csv_symbol"])
    
    return symbol_mapping_df


def build_csv_layout(symbol_mapping_df: pd.DataFrame) -> List[Dict]:
    """Build CSV_LAYOUT from symbol_mapping DataFrame.
    
    Args:
        symbol_mapping_df: DataFrame with csv_symbol, gdx_symbol, folder, header_cols
    
    Returns:
        List of layout specification dictionaries
    """
    layout = []
    for row in symbol_mapping_df.itertuples():
        # Construct path from folder and csv_symbol
        if row.folder:
            relative_path = f"{row.folder}/{row.csv_symbol}.csv"
        else:
            relative_path = f"{row.csv_symbol}.csv"
        
        # Parse header_cols (can be empty string, NaN, or integer)
        header_cols = row.header_cols
        if pd.isna(header_cols) or header_cols == "":
            header_index = None
        else:
            try:
                header_index = int(header_cols)
            except (ValueError, TypeError):
                header_index = None
        
        layout.append({
            "primary_symbol": row.csv_symbol,
            "gdx_symbol": row.gdx_symbol,
            "relative_path": relative_path,
            "header": [header_index] if header_index is not None else [],
        })
    
    return layout


def resolve_paths(
    gdx_path: Path,
    mapping_path: Path,
    output_base: Path,
    target_folder: str,
) -> Tuple[Path, Path, Path, Path]:
    """Validate and build output paths."""
    if not gdx_path.exists():
        raise FileNotFoundError(f"Update GDX path to point to your legacy file. Missing: {gdx_path}")

    export_root = (output_base / target_folder).resolve()
    expected_root = (SCRIPT_DIR / "output").resolve()
    try:
        export_root.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError("Choose TARGET_FOLDER inside the ./output directory.") from exc

    export_root.mkdir(parents=True, exist_ok=True)
    extras_root = export_root / "extras"
    extras_root.mkdir(parents=True, exist_ok=True)
    return gdx_path.resolve(), mapping_path.resolve(), export_root, extras_root


# --------------------------------------------------------------------------- #
# Core conversion
# --------------------------------------------------------------------------- #
def convert_legacy_gdx(
    gdx_path: Path,
    mapping_path: Path,
    output_root: Path,
    extras_root: Path,
    overwrite: bool = True,
) -> Dict[str, List]:
    """Convert a legacy GDX file to CSV outputs following CSV_LAYOUT specification.
    
    Main conversion function that:
    1. Loads GDX file and symbol mapping
    2. For each CSV_LAYOUT entry, transforms GDX symbol → CSV format
    3. Writes CSV files to output_root
    4. Exports extra GDX symbols (not in CSV_LAYOUT) to extras_root
    
    Args:
        gdx_path: Path to input GDX file
        mapping_path: Path to symbol_mapping.csv (CSV → GDX symbol mapping)
        output_root: Root directory for CSV outputs
        extras_root: Directory for extra GDX symbols not in CSV_LAYOUT
        overwrite: Whether to overwrite existing CSV files
    
    Returns:
        Dictionary with conversion summary, errors, and statistics
    """
    # Load symbol mapping and build CSV_LAYOUT
    symbol_mapping_df = load_symbol_mapping(mapping_path)
    csv_layout = build_csv_layout(symbol_mapping_df)

    # Load GDX file
    gdx_container = gt.Container()
    gdx_container.read(str(gdx_path))
    loaded_gdx_symbols = set(gdx_container.data.keys())

    summary: List[Dict] = []
    extras_written: List[Dict] = []
    skipped: List[Path] = []
    missing_in_gdx: List[Tuple[str, str]] = []
    optional_stubbed: List[Tuple[str, str]] = []
    empty_in_gdx: List[Tuple[str, str]] = []
    used_gdx_symbols: set[str] = set()

    # Process each CSV_LAYOUT entry
    for entry in csv_layout:
        csv_symbol = entry["primary_symbol"]
        gdx_symbol = entry["gdx_symbol"]

        # Transform GDX data to CSV format
        df_csv_data = build_frame(gdx_container, gdx_symbol, csv_symbol, entry)
        stubbed_optional = False
        if df_csv_data is None:
            # GDX symbol not found - check if optional
            if csv_symbol in OPTIONAL_SYMBOLS:
                df_csv_data = empty_frame_from_spec()
                stubbed_optional = True
                optional_stubbed.append((csv_symbol, gdx_symbol))
            else:
                missing_in_gdx.append((csv_symbol, gdx_symbol))
                continue
        else:
            df_csv_data = df_csv_data.copy()

        if not stubbed_optional and df_csv_data.empty:
            empty_in_gdx.append((csv_symbol, gdx_symbol))

        csv_target_path = output_root / Path(entry["relative_path"])
        csv_target_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip exporting pSettings.csv
        if csv_symbol == "pSettings":
            continue
        
        if not overwrite and csv_target_path.exists():
            skipped.append(csv_target_path)
            continue

        df_csv_data.to_csv(csv_target_path, index=False, na_rep="")
        
        # Add to summary (pSettings is already skipped above)
        summary.append(
            {
                "csv_symbol": csv_symbol,
                "gdx_symbol": gdx_symbol,
                "rows": len(df_csv_data),
                "path": csv_target_path.relative_to(output_root).as_posix(),
            }
        )

        if not stubbed_optional:
            used_gdx_symbols.add(gdx_symbol)

    # Export extra GDX symbols (not in CSV_LAYOUT) to extras folder
    extras_candidates = sorted(loaded_gdx_symbols - used_gdx_symbols)
    for gdx_symbol in extras_candidates:
        df_extra = fallback_frame(gdx_container, gdx_symbol)
        if df_extra is None:
            continue
        extras_target_path = extras_root / f"{gdx_symbol}.csv"
        if not overwrite and extras_target_path.exists():
            skipped.append(extras_target_path)
            continue
        df_extra.to_csv(extras_target_path, index=False, na_rep="")
        extras_written.append(
            {
                "csv_symbol": "",
                "gdx_symbol": gdx_symbol,
                "rows": len(df_extra),
                "path": extras_target_path.relative_to(output_root).as_posix(),
            }
        )

    return {
        "summary": summary,
        "extras_written": extras_written,
        "skipped": skipped,
        "missing_in_gdx": missing_in_gdx,
        "optional_stubbed": optional_stubbed,
        "empty_in_gdx": empty_in_gdx,
    }


def print_report(results: Dict[str, List], output_root: Path) -> None:
    """Emit a concise report mirroring the notebook prints."""
    missing_in_gdx = results["missing_in_gdx"]
    optional_stubbed = results["optional_stubbed"]
    empty_in_gdx = results["empty_in_gdx"]
    extras_written = results["extras_written"]
    skipped = results["skipped"]

    if missing_in_gdx:
        formatted = ", ".join(f"{csv} (expected '{gdx}')" for csv, gdx in sorted(missing_in_gdx))
        print("Symbols missing in GDX:", formatted)
    if optional_stubbed:
        formatted = ", ".join(f"{csv} (stubbed as '{gdx}')" for csv, gdx in sorted(optional_stubbed))
        print("Optional symbols absent in GDX; wrote empty CSV:", formatted)
    if empty_in_gdx:
        formatted = ", ".join(f"{csv} (mapped to '{gdx}')" for csv, gdx in sorted(empty_in_gdx))
        print("Symbols present in GDX but empty:", formatted)
    if extras_written:
        print("Extras written:")
        for item in extras_written:
            print(f"  - {item['gdx_symbol']} -> {item['path']}")
    if skipped:
        print("Skipped existing files (re-run with --overwrite to replace them):")
        for path in skipped:
            print(f"  - {path.relative_to(output_root)}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a legacy EPM GDX into the new CSV layout.")
    parser.add_argument("--gdx", type=Path, default=DEFAULT_GDX_PATH, help="Path to the legacy .gdx file.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="CSV mapping table (csv_symbol,gdx_symbol).")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE, help="Base output directory (default: ./output).")
    parser.add_argument("--target-folder", type=str, default=DEFAULT_TARGET_FOLDER, help="Subfolder under output-base for exports.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing CSVs (default: True).")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Skip existing files instead of replacing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gdx_path, mapping_path, export_root, extras_root = resolve_paths(
        args.gdx, args.mapping, args.output_base, args.target_folder
    )

    results = convert_legacy_gdx(
        gdx_path=gdx_path,
        mapping_path=mapping_path,
        output_root=export_root,
        extras_root=extras_root,
        overwrite=args.overwrite,
    )

    print_report(results, export_root)
    print(f"\nExports written under: {export_root}")
    
    # Apply post-processing to exported CSV files
    print("\n[Post-processing] Starting post-processing of exported files...")
    apply_postprocessing(export_root, extras_root, REPO_ROOT)


if __name__ == "__main__":
    main()
