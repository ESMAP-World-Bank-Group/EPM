"""Convert a legacy EPM GDX file into the newer CSV layout used by prepare-data.

How it works (summary):
- Reads a legacy GDX with the GAMS transfer API and a mapping table (`symbol_mapping.csv`)
  that aligns old GDX symbol names to the expected CSV names.
- For each symbol in `symbol_mapping.csv`, reads the GDX data and optionally unstacks a
  specified column to create wide-format CSV files. Writes to `{folder}/{csv_symbol}.csv`.
  Symbols missing in the GDX but present in `datapackage.json` are created as empty stubs
  with proper column structure from the schema.
- Any extra GDX symbols not covered by the mapping or `datapackage.json` are dumped to
  `output/data/.legacy/` for inspection.
- Run from the CLI (`python legacy_to_new_format.py --gdx ... --mapping ...`) to batch-convert
  a GDX; defaults point to the sample inputs/outputs in this folder.
"""

from __future__ import annotations

import argparse
import json
import difflib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
DEFAULT_GDX_PATH = SCRIPT_DIR / "input" / "WB_EPM_8_5_WAPP_251022.gdx"
DEFAULT_MAPPING_PATH = SCRIPT_DIR / "symbol_mapping.csv"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output"
DEFAULT_TARGET_FOLDER = "data"


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
    - Apply lookup mappings from .legacy folder
    
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
    
    # Apply lookup mappings from .legacy folder
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


def _rename_reference_files(output_root: Path) -> None:
    """Manually rename reference files for dimension mapping.
    
    Renames:
    - zcmap.csv: first column → "z", second column → "c"
    - y.csv: column → "y"
    - pHours.csv: ensure columns are named "q" and "d" (after its treatment)
    
    Args:
        output_root: Root directory for CSV outputs
    """
    # Rename zcmap.csv
    zcmap_path = output_root / "zcmap.csv"
    if zcmap_path.exists():
        try:
            df = pd.read_csv(zcmap_path)
            if not df.empty and len(df.columns) >= 2:
                rename_dict = {}
                if df.columns[0] != "z":
                    rename_dict[df.columns[0]] = "z"
                if df.columns[1] != "c":
                    rename_dict[df.columns[1]] = "c"
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    df.to_csv(zcmap_path, index=False, na_rep="")
                    print(f"  [Reference files] Renamed zcmap.csv: {', '.join(f'{old}→{new}' for old, new in rename_dict.items())}")
        except Exception as e:
            print(f"  [Reference files] Failed to rename zcmap.csv: {e}")
    
    # Rename y.csv
    y_path = output_root / "y.csv"
    if y_path.exists():
        try:
            df = pd.read_csv(y_path)
            if not df.empty and len(df.columns) >= 1:
                if df.columns[0] != "y":
                    df = df.rename(columns={df.columns[0]: "y"})
                    df.to_csv(y_path, index=False, na_rep="")
                    print(f"  [Reference files] Renamed y.csv: {df.columns[0]}→y")
        except Exception as e:
            print(f"  [Reference files] Failed to rename y.csv: {e}")
    
    # Ensure pHours.csv has "q" and "d" columns (after its treatment in postprocess_csv)
    # Note: pHours.csv is processed by postprocess_csv which removes YYYY columns and duplicates
    # We need to ensure the remaining columns are named "q" and "d"
    phours_path = output_root / "pHours.csv"
    if phours_path.exists():
        try:
            df = pd.read_csv(phours_path)
            if not df.empty and len(df.columns) >= 2:
                rename_dict = {}
                # First column should be "q" (season/quarter)
                if df.columns[0] != "q":
                    rename_dict[df.columns[0]] = "q"
                # Second column should be "d" (daytype)
                if len(df.columns) > 1 and df.columns[1] != "d":
                    rename_dict[df.columns[1]] = "d"
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    df.to_csv(phours_path, index=False, na_rep="")
                    print(f"  [Reference files] Renamed pHours.csv: {', '.join(f'{old}→{new}' for old, new in rename_dict.items())}")
        except Exception as e:
            print(f"  [Reference files] Failed to rename pHours.csv: {e}")


def postprocess_csv(csv_path: Path, output_root: Path, extras_root: Path, repo_root: Path) -> None:
    """Apply post-processing to a single CSV file."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    
    if df.empty:
        return
    
    file_name = csv_path.name
    
    changes = []
    
    # Remove element_text column
    if "element_text" in df.columns:
        df = df.drop(columns=["element_text"])
        changes.append("removed element_text")
    
    # Specific rule for pHours: remove columns with YYYY format values (year columns) and remove duplicates
    if file_name == "pHours.csv":
        # Find columns where values match YYYY format (4-digit year)
        year_cols = []
        for col in df.columns:
            # Check if all non-null values in the column are 4-digit years (YYYY format)
            col_values = df[col].dropna().astype(str).str.strip()
            if len(col_values) > 0:
                # Check if all values are exactly 4 digits
                all_yyyy = col_values.str.len().eq(4).all() and col_values.str.isdigit().all()
                if all_yyyy:
                    year_cols.append(col)
        
        if year_cols:
            df = df.drop(columns=year_cols)
            changes.append(f"removed year columns: {', '.join(year_cols)}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            changes.append(f"removed {removed_duplicates} duplicate row(s)")
    
    # Write back
    try:
        df.to_csv(csv_path, index=False, na_rep="")
        if changes:
            print(f"  [{file_name}] {', '.join(changes)}")
    except Exception:
        pass


def build_dimension_value_mapping(output_root: Path, repo_root: Path) -> Dict[str, Set[str]]:
    """Build dimension value mapping from reference files.
    
    Extracts dimension values from manually renamed reference files:
    - z, c from zcmap.csv
    - y from y.csv
    - q, d, t from pHours.csv (after treatment)
    - zext, f, tech from other files
    
    Args:
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
    
    Returns:
        Dictionary mapping dimension type to set of values found
    """
    dimension_mapping: Dict[str, Set[str]] = {
        "z": set(),
        "c": set(),
        "y": set(),
        "q": set(),
        "d": set(),
        "t": set(),
        "zext": set(),
        "f": set(),
        "tech": set(),
    }
    
    # Extract z and c from zcmap.csv
    zcmap_path = output_root / "zcmap.csv"
    if zcmap_path.exists():
        try:
            df = pd.read_csv(zcmap_path)
            if "z" in df.columns:
                dimension_mapping["z"].update(df["z"].dropna().astype(str).str.strip().unique())
            if "c" in df.columns:
                dimension_mapping["c"].update(df["c"].dropna().astype(str).str.strip().unique())
        except Exception as e:
            print(f"  [Dimension mapping] Failed to load zcmap.csv: {e}")
    
    # Extract y from y.csv
    y_path = output_root / "y.csv"
    if y_path.exists():
        try:
            df = pd.read_csv(y_path)
            if "y" in df.columns:
                dimension_mapping["y"].update(df["y"].dropna().astype(str).str.strip().unique())
        except Exception as e:
            print(f"  [Dimension mapping] Failed to load y.csv: {e}")
    
    # Extract q, d, and t from pHours.csv (after treatment)
    phours_path = output_root / "pHours.csv"
    if phours_path.exists():
        try:
            df = pd.read_csv(phours_path)
            if "q" in df.columns:
                dimension_mapping["q"].update(df["q"].dropna().astype(str).str.strip().unique())
            if "d" in df.columns:
                dimension_mapping["d"].update(df["d"].dropna().astype(str).str.strip().unique())
            if "t" in df.columns:
                dimension_mapping["t"].update(df["t"].dropna().astype(str).str.strip().unique())
        except Exception as e:
            print(f"  [Dimension mapping] Failed to load pHours.csv: {e}")
    
    # Extract tech and fuel from pTechFuel.csv using datapackage.json path
    resources_map = load_datapackage_resources(repo_root)
    if "pTechFuel" in resources_map:
        pTechFuel_rel_path = resources_map["pTechFuel"]["path"]
        # Resolve path relative to epm/input/ (where datapackage.json is)
        pTechFuel_path = (repo_root / "epm" / "input" / pTechFuel_rel_path).resolve()
        
        if pTechFuel_path.exists():
            try:
                df = pd.read_csv(pTechFuel_path)
                if "tech" in df.columns:
                    dimension_mapping["tech"].update(df["tech"].dropna().astype(str).str.strip().unique())
                if "fuel" in df.columns:
                    dimension_mapping["f"].update(df["fuel"].dropna().astype(str).str.strip().unique())
                print(f"  [Dimension mapping] Loaded pTechFuel.csv: {len(dimension_mapping['tech'])} tech, {len(dimension_mapping['f'])} fuel values")
            except Exception as e:
                print(f"  [Dimension mapping] Failed to load pTechFuel.csv: {e}")
    
    # Scan other files for zext, f, tech columns
    for csv_file in output_root.rglob("*.csv"):
        # Skip reference files already processed
        if csv_file.name in ("zcmap.csv", "y.csv", "pHours.csv"):
            continue
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
            
            # Check for zext column
            if "zext" in df.columns:
                dimension_mapping["zext"].update(df["zext"].dropna().astype(str).str.strip().unique())
            
            # Check for f or fuel column
            if "f" in df.columns:
                dimension_mapping["f"].update(df["f"].dropna().astype(str).str.strip().unique())
            elif "fuel" in df.columns:
                dimension_mapping["f"].update(df["fuel"].dropna().astype(str).str.strip().unique())
            
            # Check for tech column
            if "tech" in df.columns:
                dimension_mapping["tech"].update(df["tech"].dropna().astype(str).str.strip().unique())
        except Exception:
            # Skip files that can't be read
            continue
    
    # Remove empty sets and log results
    non_empty = {k: v for k, v in dimension_mapping.items() if v}
    if non_empty:
        print(f"  [Dimension mapping] Extracted dimension values:")
        for dim_type, values in sorted(non_empty.items()):
            print(f"    {dim_type}: {len(values)} unique value(s)")
    
    return dimension_mapping


def validate_column_rename(
    old_name: str,
    new_name: str,
    column_values: pd.Series,
    dimension_mapping: Dict[str, Set[str]],
    dimension_type: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Validate if a column rename is consistent with expected dimension values.
    
    Args:
        old_name: Current column name
        new_name: Proposed new column name
        column_values: Actual values in the column
        dimension_mapping: Dimension value mapping from build_dimension_value_mapping()
        dimension_type: Expected dimension type (e.g., "z", "d", "t", "q", "y", "zext", "f", "tech")
        verbose: If True, print validation details
    
    Returns:
        True if rename is valid, False otherwise
    """
    # Special handling for "value" column: must be numeric
    if new_name.lower() == "value":
        try:
            # Try to convert to numeric - if it fails, it's not a value column
            numeric_values = pd.to_numeric(column_values.dropna(), errors='coerce')
            if numeric_values.notna().any():
                # At least some values are numeric
                non_numeric_count = numeric_values.isna().sum()
                if non_numeric_count == 0:
                    if verbose:
                        print(f"      [Validation] '{old_name}'→'{new_name}': ✓ Valid (all values are numeric)")
                    return True
                else:
                    if verbose:
                        print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed ({non_numeric_count} non-numeric values found)")
                    return False
            else:
                if verbose:
                    print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed (no numeric values found)")
                return False
        except Exception:
            if verbose:
                print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed (cannot validate as numeric)")
            return False
    
    # Special handling for "y" column: must match XXXX format (4-digit year) only
    if new_name.lower() in ("y", "year"):
        actual_values = column_values.dropna().astype(str).str.strip()
        # Check if all values are 4-digit years
        all_yyyy = actual_values.str.len().eq(4).all() and actual_values.str.isdigit().all()
        if all_yyyy:
            if verbose:
                sample_values = sorted(actual_values.unique())[:5]
                print(f"      [Validation] '{old_name}'→'{new_name}': ✓ Valid (YYYY format: {', '.join(sample_values)}{'...' if len(actual_values.unique()) > 5 else ''})")
            return True
        else:
            if verbose:
                sample_values = sorted(actual_values.unique())[:5]
                print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed (not all values are 4-digit years, found: {', '.join(sample_values)}{'...' if len(actual_values.unique()) > 5 else ''})")
            return False
    
    # If dimension_type is provided, check if column values match expected dimension values
    if dimension_type and dimension_type in dimension_mapping:
        expected_values = dimension_mapping[dimension_type]
        if expected_values:
            # Get unique non-null values from column
            actual_values = set(column_values.dropna().astype(str).str.strip().unique())
            # Check if all actual values are in expected values (or at least most of them)
            if actual_values:
                match_ratio = len(actual_values & expected_values) / len(actual_values)
                # Allow some tolerance - at least 80% of values should match
                is_valid = match_ratio >= 0.8
                if verbose:
                    matched = actual_values & expected_values
                    unmatched = actual_values - expected_values
                    print(f"      [Validation] '{old_name}'→'{new_name}': {'✓ Valid' if is_valid else '✗ Failed'} (match: {len(matched)}/{len(actual_values)}, expected set: {sorted(list(expected_values))[:10]}{'...' if len(expected_values) > 10 else ''})")
                    if unmatched and verbose:
                        print(f"        Unmatched values: {sorted(list(unmatched))[:5]}{'...' if len(unmatched) > 5 else ''}")
                return is_valid
    
    # If renaming to a known dimension field, validate values match that dimension's set
    dimension_field_map = {
        "z": "z",
        "zone": "z",
        "c": "c",
        "country": "c",
        "y": "y",
        "year": "y",
        "q": "q",
        "season": "q",
        "quarter": "q",
        "d": "d",
        "daytype": "d",
        "day": "d",
        "zext": "zext",
        "f": "f",
        "fuel": "f",
        "tech": "tech",
        "technology": "tech",
    }
    
    if new_name.lower() in dimension_field_map:
        dim_type = dimension_field_map[new_name.lower()]
        if dim_type in dimension_mapping:
            expected_values = dimension_mapping[dim_type]
            if expected_values:
                actual_values = set(column_values.dropna().astype(str).str.strip().unique())
                if actual_values:
                    match_ratio = len(actual_values & expected_values) / len(actual_values)
                    is_valid = match_ratio >= 0.8
                    if verbose:
                        matched = actual_values & expected_values
                        unmatched = actual_values - expected_values
                        print(f"      [Validation] '{old_name}'→'{new_name}': {'✓ Valid' if is_valid else '✗ Failed'} (match: {len(matched)}/{len(actual_values)}, expected set: {sorted(list(expected_values))[:10]}{'...' if len(expected_values) > 10 else ''})")
                        if unmatched and verbose:
                            print(f"        Unmatched values: {sorted(list(unmatched))[:5]}{'...' if len(unmatched) > 5 else ''}")
                    return is_valid
            else:
                if verbose:
                    print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed (no expected values found for dimension '{dim_type}')")
                return False
    
    # For other fields without specific validation, reject by default (safer)
    if verbose:
        print(f"      [Validation] '{old_name}'→'{new_name}': ✗ Failed (no validation rule for this field)")
    return False


def rename_columns_from_datapackage(output_root: Path, repo_root: Path, dimension_mapping: Dict[str, Set[str]]) -> None:
    """Rename columns in all CSV files to match datapackage.json schema field names.
    
    Only renames columns that are explicitly defined in datapackage.json schema.
    Does not rename additional columns that aren't in the schema.
    
    For wide format files, only renames index columns (not dimension value columns).
    Validates renames using dimension mapping before applying.
    
    Args:
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
        dimension_mapping: Dimension value mapping from build_dimension_value_mapping()
    """
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Renaming columns to match datapackage.json schema")
    print(f"{'=' * 70}")
    
    resources_map = load_datapackage_resources(repo_root)
    if not resources_map:
        print("  [Column rename] No resources found in datapackage.json")
        return
    
    renamed_count = 0
    
    for resource_name, resource_info in resources_map.items():
        # Skip pSettings (handled separately)
        if resource_name == "pSettings":
            continue
        
        # Get path from datapackage
        datapackage_path = resource_info["path"]
        target_path = output_root / datapackage_path
        
        if not target_path.exists():
            continue
        
        try:
            df = pd.read_csv(target_path)
            if df.empty:
                continue
            
            # Get field names from schema
            field_names = resource_info.get("field_names", [])
            if not field_names:
                continue
            
            format_type = resource_info.get("format", "long")
            dimensions = resource_info.get("dimensions", [])
            foreign_key_fields = resource_info.get("foreign_key_fields", set())  # Get fields with foreignKeys
            foreign_key_ref_map = resource_info.get("foreign_key_ref_map", {})  # Maps field name -> reference field name
            
            # Build rename mapping - only rename columns that are in the defined set
            rename_dict = {}
            current_cols = list(df.columns)
            field_names_set = set(field_names)
            
            if format_type == "wide":
                # For wide format: only rename index columns (not dimension value columns or value field)
                dimensions_set = set(dimensions)
                # Exclude "value" from index fields - it's created during transformation, not renamed
                index_field_names = [fname for fname in field_names 
                                    if fname not in dimensions_set and fname != "value"]
                index_field_names_set = set(index_field_names)
                
                # Filter foreignKey fields to only those that are index fields
                index_foreign_key_fields = foreign_key_fields & index_field_names_set
                
                # Helper function to check if a column is a dimension value column
                def is_dimension_value_column(col_name: str, dimensions: List[str]) -> bool:
                    """Check if column name represents a dimension value by checking against dimension mapping."""
                    col_str = str(col_name)
                    # Check if it's exactly a dimension name (shouldn't happen, but check anyway)
                    if col_str in dimensions:
                        return True
                    # Check if column name matches dimension values from reference resources
                    for dim in dimensions:
                        # Get dimension values from mapping
                        dim_values = dimension_mapping.get(dim, set())
                        # Check if column name is in the dimension values
                        if col_str in dim_values:
                            return True
                        # For "year" dimension, also check if it's a 4-digit integer
                        if dim == "year" and col_str.isdigit() and len(col_str) == 4:
                            return True
                    return False
                
                # Helper function to score a potential match
                def score_column_match(col_name: str, expected_name: str, col_values: set, ref_values: set) -> float:
                    """Score how well a column matches an expected field name.
                    
                    Returns:
                        Score from 0.0 to 1.0, where 1.0 is perfect match
                    """
                    # Exact name match gets highest score
                    if col_name.lower() == expected_name.lower():
                        return 1.0
                    
                    # If no reference values, can't score based on values
                    if not ref_values:
                        return 0.0
                    
                    # Score based on value matching
                    if not col_values:
                        return 0.0
                    
                    # Perfect match: all column values are in reference values
                    if col_values.issubset(ref_values):
                        return 0.9
                    
                    # Partial match: calculate ratio
                    intersection = col_values & ref_values
                    if intersection:
                        match_ratio = len(intersection) / len(col_values)
                        # Require at least 80% match to be considered
                        if match_ratio >= 0.8:
                            return 0.7 + (match_ratio - 0.8) * 0.2  # Scale 0.8-1.0 to 0.7-0.9
                    
                    return 0.0
                
                # For wide format: iterate through index field names in schema order
                print(f"  [Column rename] {resource_name}: Processing {len(index_field_names)} index field(s) in schema order")
                
                # Show foreignKey fields found and their reference value counts (for index fields only)
                if index_foreign_key_fields:
                    fk_list = sorted(index_foreign_key_fields)
                    ref_counts = []
                    for fk_field in fk_list:
                        ref_values = dimension_mapping.get(fk_field, set())
                        ref_counts.append(f"{fk_field}({len(ref_values)})")
                    print(f"  [Column rename] {resource_name}: Found {len(index_foreign_key_fields)} foreignKey index field(s): {', '.join(fk_list)}")
                    print(f"  [Column rename] {resource_name}: Reference value counts: {', '.join(ref_counts)}")
                
                for expected_name in index_field_names:
                    # Check if column already exists (case-insensitive)
                    existing_col = None
                    for col in current_cols:
                        if col.lower() == expected_name.lower():
                            existing_col = col
                            break
                    
                    if existing_col:
                        # Already exists with correct name (case may differ), skip
                        continue
                    
                    # First, try exact name match (case-insensitive)
                    exact_match = None
                    for col_name in current_cols:
                        if col_name.lower() == expected_name.lower():
                            if col_name not in index_field_names_set and col_name not in rename_dict.values():
                                if not is_dimension_value_column(col_name, dimensions):
                                    exact_match = col_name
                                    break
                    
                    if exact_match:
                        # Only validate if it's a foreignKey field
                        if expected_name in index_foreign_key_fields:
                            if validate_column_rename(exact_match, expected_name, df[exact_match], dimension_mapping, verbose=True):
                                rename_dict[exact_match] = expected_name
                                print(f"    ✓ Matched '{exact_match}'→'{expected_name}' (exact name match)")
                                continue
                        else:
                            # Non-foreignKey field, just rename without validation
                            rename_dict[exact_match] = expected_name
                            print(f"    ✓ Matched '{exact_match}'→'{expected_name}' (exact name match)")
                            continue
                    
                    # Only try value-based matching for fields with foreignKeys (GAMS sets)
                    if expected_name not in index_foreign_key_fields:
                        # Non-foreignKey field with no exact match, silently skip
                        continue
                    
                    # For foreignKey fields, try value-based matching
                    # Use reference field name for dimension lookup (e.g., z1 -> z)
                    ref_field_name = foreign_key_ref_map.get(expected_name, expected_name)
                    ref_values = dimension_mapping.get(ref_field_name, set())
                    ref_info = f" ({len(ref_values)} reference values)" if ref_values else ""
                    print(f"    Looking for '{expected_name}'{ref_info}...")
                    best_match = None
                    best_score = 0.0
                    
                    for col_name in current_cols:
                        # Skip dimension value columns
                        if is_dimension_value_column(col_name, dimensions):
                            continue
                        
                        # Skip if column already renamed or already matches a field name
                        if col_name in index_field_names_set or col_name in rename_dict.values():
                            continue
                        
                        # Calculate match score
                        col_values = set(df[col_name].dropna().astype(str).str.strip().unique())
                        score = score_column_match(col_name, expected_name, col_values, ref_values)
                        
                        if score > best_score:
                            # Validate before considering it
                            if validate_column_rename(col_name, expected_name, df[col_name], dimension_mapping, verbose=False):
                                best_match = col_name
                                best_score = score
                    
                    if best_match and best_score >= 0.7:
                        rename_dict[best_match] = expected_name
                        match_type = "exact name" if best_score == 1.0 else f"value match (score: {best_score:.2f})"
                        print(f"    ✓ Matched '{best_match}'→'{expected_name}' ({match_type})")
                    else:
                        print(f"    ✗ No matching column found for '{expected_name}'")
            else:
                # For long format: iterate through field names in schema order
                # For each field name, try to find a matching column
                print(f"  [Column rename] {resource_name}: Processing {len(field_names)} field(s) in schema order")
                
                # Show foreignKey fields found and their reference value counts
                if foreign_key_fields:
                    fk_list = sorted(foreign_key_fields)
                    ref_counts = []
                    for fk_field in fk_list:
                        ref_values = dimension_mapping.get(fk_field, set())
                        ref_counts.append(f"{fk_field}({len(ref_values)})")
                    print(f"  [Column rename] {resource_name}: Found {len(foreign_key_fields)} foreignKey field(s): {', '.join(fk_list)}")
                    print(f"  [Column rename] {resource_name}: Reference value counts: {', '.join(ref_counts)}")
                
                # Helper function to score a potential match
                def score_column_match(col_name: str, expected_name: str, col_values: set, ref_values: set) -> float:
                    """Score how well a column matches an expected field name.
                    
                    Returns:
                        Score from 0.0 to 1.0, where 1.0 is perfect match
                    """
                    # Exact name match gets highest score
                    if col_name.lower() == expected_name.lower():
                        return 1.0
                    
                    # If no reference values, can't score based on values
                    if not ref_values:
                        return 0.0
                    
                    # Score based on value matching
                    if not col_values:
                        return 0.0
                    
                    # Perfect match: all column values are in reference values
                    if col_values.issubset(ref_values):
                        return 0.9
                    
                    # Partial match: calculate ratio
                    intersection = col_values & ref_values
                    if intersection:
                        match_ratio = len(intersection) / len(col_values)
                        # Require at least 80% match to be considered
                        if match_ratio >= 0.8:
                            return 0.7 + (match_ratio - 0.8) * 0.2  # Scale 0.8-1.0 to 0.7-0.9
                    
                    return 0.0
                
                for expected_name in field_names:
                    # Check if column already exists (case-insensitive)
                    existing_col = None
                    for col in current_cols:
                        if col.lower() == expected_name.lower():
                            existing_col = col
                            break
                    
                    if existing_col:
                        # Already exists with correct name (case may differ), skip
                        continue
                    
                    # First, try exact name match (case-insensitive)
                    exact_match = None
                    for col_name in current_cols:
                        if col_name.lower() == expected_name.lower():
                            if col_name not in field_names_set and col_name not in rename_dict.values():
                                exact_match = col_name
                                break
                    
                    if exact_match:
                        # Only validate if it's a foreignKey field
                        if expected_name in foreign_key_fields:
                            if validate_column_rename(exact_match, expected_name, df[exact_match], dimension_mapping, verbose=True):
                                rename_dict[exact_match] = expected_name
                                print(f"    ✓ Matched '{exact_match}'→'{expected_name}' (exact name match)")
                                continue
                        else:
                            # Non-foreignKey field, just rename without validation
                            rename_dict[exact_match] = expected_name
                            print(f"    ✓ Matched '{exact_match}'→'{expected_name}' (exact name match)")
                            continue
                    
                    # Only try value-based matching for fields with foreignKeys (GAMS sets)
                    if expected_name not in foreign_key_fields:
                        # Non-foreignKey field with no exact match, silently skip
                        continue
                    
                    # For foreignKey fields, try value-based matching
                    # Use reference field name for dimension lookup (e.g., z1 -> z)
                    ref_field_name = foreign_key_ref_map.get(expected_name, expected_name)
                    ref_values = dimension_mapping.get(ref_field_name, set())
                    ref_info = f" ({len(ref_values)} reference values)" if ref_values else ""
                    print(f"    Looking for '{expected_name}'{ref_info}...")
                    best_match = None
                    best_score = 0.0
                    
                    for col_name in current_cols:
                        # Skip if column already renamed or already matches a field name
                        if col_name in field_names_set or col_name in rename_dict.values():
                            continue
                        
                        # Calculate match score
                        col_values = set(df[col_name].dropna().astype(str).str.strip().unique())
                        score = score_column_match(col_name, expected_name, col_values, ref_values)
                        
                        if score > best_score:
                            # Validate before considering it
                            if validate_column_rename(col_name, expected_name, df[col_name], dimension_mapping, verbose=False):
                                best_match = col_name
                                best_score = score
                    
                    if best_match and best_score >= 0.7:
                        rename_dict[best_match] = expected_name
                        match_type = "exact name" if best_score == 1.0 else f"value match (score: {best_score:.2f})"
                        print(f"    ✓ Matched '{best_match}'→'{expected_name}' ({match_type})")
                    else:
                        print(f"    ✗ No matching column found for '{expected_name}'")
            
            # Apply renaming
            if rename_dict:
                df = df.rename(columns=rename_dict)
                df.to_csv(target_path, index=False, na_rep="")
                renamed_count += 1
                renamed_cols = ', '.join(f"{old}→{new}" for old, new in list(rename_dict.items())[:5])
                if len(rename_dict) > 5:
                    renamed_cols += f" ... ({len(rename_dict)} total)"
                print(f"  [Column rename] {resource_name}: {renamed_cols}")
        
        except Exception as e:
            # Fail silently for individual files
            continue
    
    if renamed_count > 0:
        print(f"  [Column rename] Renamed columns in {renamed_count} file(s)")
    else:
        print(f"  [Column rename] No columns needed renaming ✓")


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
        
        # Rename storage-specific columns in storage_rows before merging
        storage_rename_map = {}
        if "Capacity" in storage_rows.columns:
            storage_rename_map["Capacity"] = "CapacityMWh"
        if "Capex" in storage_rows.columns:
            storage_rename_map["Capex"] = "CapexMWh"
        if "VOM" in storage_rows.columns:
            storage_rename_map["VOM"] = "VOMMWh"
        if "FOM" in storage_rows.columns:
            storage_rename_map["FOM"] = "FOMMWh"
        
        if storage_rename_map:
            storage_rows = storage_rows.rename(columns=storage_rename_map)
            renamed_cols = ', '.join(f"{old}→{new}" for old, new in storage_rename_map.items())
            print(f"  [Storage merge] Renamed columns in storage rows: {renamed_cols}")
        
        # Ensure pStorageDataInput has 'gen' column - rename first column if needed
        if "gen" not in df_storage.columns:
            if len(df_storage.columns) > 0:
                first_col = df_storage.columns[0]
                print(f"  [Storage merge] Renaming first column '{first_col}' to 'gen' in pStorageDataInput")
                df_storage = df_storage.rename(columns={first_col: "gen"})
            else:
                print(f"  [Storage merge] No columns in pStorageDataInput, cannot merge")
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


def copy_cplex_folder(output_root: Path, repo_root: Path) -> None:
    """Copy CPLEX folder from data_test to output directory."""
    source_cplex_dir = repo_root / "epm" / "input" / "data_test" / "cplex"
    target_cplex_dir = output_root / "cplex"
    
    if not source_cplex_dir.exists():
        print(f"  [CPLEX] Source folder not found at {source_cplex_dir}")
        return
    
    try:
        if target_cplex_dir.exists():
            shutil.rmtree(target_cplex_dir)
        shutil.copytree(source_cplex_dir, target_cplex_dir)
        cplex_files = list(target_cplex_dir.glob("*.opt"))
        print(f"  [CPLEX] Copied {len(cplex_files)} CPLEX option files to {target_cplex_dir}")
    except Exception as e:
        print(f"  [CPLEX] Failed to copy CPLEX folder: {e}")
        import traceback
        traceback.print_exc()


def copy_psettings_files(output_root: Path, repo_root: Path) -> None:
    """Copy pSettings.csv from data_test to output directory."""
    source_file = repo_root / "epm" / "input" / "data_test" / "pSettings.csv"
    target_file = output_root / "pSettings.csv"
    
    if not source_file.exists():
        print(f"  [pSettings] pSettings.csv not found at {source_file}")
        return
    
    try:
        shutil.copy2(source_file, target_file)
        print(f"  [pSettings] Copied pSettings.csv to {output_root}")
    except Exception as e:
        print(f"  [pSettings] Failed to copy pSettings.csv: {e}")


def copy_and_expand_default_files(output_root: Path, repo_root: Path) -> None:
    """Copy default files from data_test and expand for all zones in zcmap."""
    data_test_dir = repo_root / "epm" / "input" / "data_test" / "supply"
    output_supply_dir = output_root / "supply"
    
    # Read zcmap to get all zones
    zcmap_path = output_root / "zcmap.csv"
    if not zcmap_path.exists():
        print(f"  [Default files] zcmap.csv not found at {zcmap_path}")
        return
    
    try:
        zcmap_df = pd.read_csv(zcmap_path)
        if "zone" not in zcmap_df.columns:
            print(f"  [Default files] No 'zone' column in zcmap.csv")
            return
        all_zones = zcmap_df["zone"].astype(str).tolist()
        print(f"  [Default files] Found {len(all_zones)} zones in zcmap: {', '.join(all_zones[:10])}{'...' if len(all_zones) > 10 else ''}")
    except Exception as e:
        print(f"  [Default files] Failed to read zcmap.csv: {e}")
        return
    
    # Files to copy and expand
    default_files = [
        "pCapexTrajectoriesDefault.csv",
        "pAvailabilityDefault.csv",
        "pGenDataInputDefault.csv"
    ]
    
    for file_name in default_files:
        source_path = data_test_dir / file_name
        target_path = output_supply_dir / file_name
        
        if not source_path.exists():
            print(f"  [Default files] {file_name} not found at {source_path}")
            continue
        
        try:
            df = pd.read_csv(source_path)
            if df.empty:
                print(f"  [Default files] {file_name} is empty")
                continue
            
            if "zone" not in df.columns:
                print(f"  [Default files] No 'zone' column in {file_name}")
                continue
            
            # Get first zone as reference
            reference_zone = df["zone"].iloc[0]
            print(f"  [Default files] Using '{reference_zone}' as reference zone for {file_name}")
            
            # Get all rows for reference zone
            ref_rows = df[df["zone"] == reference_zone].copy()
            if ref_rows.empty:
                print(f"  [Default files] No rows found for reference zone '{reference_zone}' in {file_name}")
                continue
            
            # Create expanded dataframe
            expanded_rows = []
            for zone in all_zones:
                zone_rows = ref_rows.copy()
                zone_rows["zone"] = zone
                expanded_rows.append(zone_rows)
            
            expanded_df = pd.concat(expanded_rows, ignore_index=True)
            expanded_df.to_csv(target_path, index=False, na_rep="")
            print(f"  [Default files] Expanded {file_name}: {len(ref_rows)} rows × {len(all_zones)} zones = {len(expanded_df)} rows")
            
        except Exception as e:
            print(f"  [Default files] Failed to process {file_name}: {e}")
            import traceback
            traceback.print_exc()


def validate_tech_fuel_combinations(output_root: Path, repo_root: Path) -> None:
    """Validate tech-fuel combinations in pGenDataInput and pStorageDataInput against pTechFuel.csv.
    
    Checks unique combinations and suggests corrections based on string similarity.
    """
    pTechFuel_path = repo_root / "epm" / "resources" / "pTechFuel.csv"
    
    if not pTechFuel_path.exists():
        print(f"  [Tech-Fuel validation] pTechFuel.csv not found")
        return
    
    try:
        # Load valid tech-fuel combinations
        pTechFuel_df = pd.read_csv(pTechFuel_path)
        if len(pTechFuel_df.columns) < 2:
            return
        
        valid_techs = set(pTechFuel_df.iloc[:, 0].astype(str).str.strip())
        valid_fuels = set(pTechFuel_df.iloc[:, 1].astype(str).str.strip())
        valid_combinations = set(
            (str(row.iloc[0]).strip(), str(row.iloc[1]).strip())
            for _, row in pTechFuel_df.iterrows()
        )
        
        # Files to check
        files_to_check = [
            ("pGenDataInput", output_root / "supply" / "pGenDataInput.csv"),
            ("pStorageDataInput", output_root / "supply" / "pStorageDataInput.csv"),
        ]
        
        all_invalid = []
        
        for file_name, file_path in files_to_check:
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path)
                if df.empty or "tech" not in df.columns or "fuel" not in df.columns:
                    continue
                
                # Get unique tech-fuel combinations
                df_clean = df[["tech", "fuel"]].copy()
                df_clean["tech"] = df_clean["tech"].astype(str).str.strip()
                df_clean["fuel"] = df_clean["fuel"].astype(str).str.strip()
                df_clean = df_clean[(df_clean["tech"] != "") & (df_clean["tech"] != "nan") & 
                                   (df_clean["fuel"] != "") & (df_clean["fuel"] != "nan")]
                
                unique_combinations = set(zip(df_clean["tech"], df_clean["fuel"]))
                invalid_combinations = [(tech, fuel) for tech, fuel in unique_combinations 
                                       if (tech, fuel) not in valid_combinations]
                
                if invalid_combinations:
                    all_invalid.append((file_name, invalid_combinations))
                    
            except Exception:
                continue
        
        # Report and suggest corrections
        if all_invalid:
            for file_name, invalid_combinations in all_invalid:
                for tech, fuel in sorted(invalid_combinations):
                    # Find closest tech match
                    closest_tech = None
                    if tech:
                        matches = difflib.get_close_matches(tech, valid_techs, n=1, cutoff=0.6)
                        if matches:
                            closest_tech = matches[0]
                    
                    # Find closest fuel match
                    closest_fuel = None
                    if fuel:
                        matches = difflib.get_close_matches(fuel, valid_fuels, n=1, cutoff=0.6)
                        if matches:
                            closest_fuel = matches[0]
                    
                    # Check if fuel is valid
                    fuel_is_valid = fuel in valid_fuels
                    
                    # Build suggestion message
                    suggestion_parts = []
                    
                    # First, try to find a valid combination with suggested tech
                    suggested_tech = closest_tech
                    suggested_fuel = None
                    if suggested_tech:
                        # Check if suggested tech + current fuel is valid
                        if fuel_is_valid and (suggested_tech, fuel) in valid_combinations:
                            suggested_fuel = fuel
                        else:
                            # Try to find a valid fuel for the suggested tech
                            valid_fuels_for_tech = [f for t, f in valid_combinations if t == suggested_tech]
                            if valid_fuels_for_tech:
                                closest_valid_fuel = difflib.get_close_matches(fuel, valid_fuels_for_tech, n=1, cutoff=0.6)
                                if closest_valid_fuel:
                                    suggested_fuel = closest_valid_fuel[0]
                    
                    # Build suggestion text
                    if suggested_tech:
                        suggestion_parts.append(f"tech: '{suggested_tech}'")
                        
                        if suggested_fuel:
                            if suggested_fuel == fuel:
                                suggestion_parts.append("and fuel is valid")
                            else:
                                suggestion_parts.append(f"and fuel should be replaced by '{suggested_fuel}'")
                        else:
                            if fuel_is_valid:
                                suggestion_parts.append("and fuel is valid")
                            elif closest_fuel:
                                suggestion_parts.append(f"and fuel should be replaced by '{closest_fuel}'")
                            else:
                                suggestion_parts.append("and fuel is invalid (no suggestion)")
                    else:
                        if fuel_is_valid:
                            suggestion_parts.append("fuel is valid but tech is invalid (no suggestion)")
                        elif closest_fuel:
                            suggestion_parts.append(f"fuel should be replaced by '{closest_fuel}' (tech suggestion unavailable)")
                        else:
                            suggestion_parts.append("no suggestion available")
                    
                    suggestion = ". Suggest replacement of " + " and ".join(suggestion_parts) if suggestion_parts else ". No suggestion available"
                    
                    print(f"  Found in {file_name} tech: '{tech}' - fuel: '{fuel}'{suggestion}.")
        else:
            print(f"  [Tech-Fuel validation] All combinations valid ✓")
            
    except Exception:
        pass


def apply_postprocessing(output_root: Path, extras_root: Path, repo_root: Path, created_csv_symbols: set[str], rename_columns: bool = True) -> None:
    """Apply post-processing rules to exported CSV files.
    
    This function is called after all CSV files have been exported.
    It applies symbol-specific post-processing rules and ensures all files match
    the datapackage.json schema.
    
    Args:
        output_root: Root directory for CSV outputs
        extras_root: Directory for extra GDX symbols (saved to .legacy folder)
        repo_root: Repository root path
        rename_columns: If True, rename columns to match datapackage.json schema (default: True)
        created_csv_symbols: Set of CSV symbol names that were created during conversion
    """
    # Manually rename reference files for dimension mapping
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Renaming reference files for dimension mapping")
    print(f"{'=' * 70}")
    _rename_reference_files(output_root)
    
    # Build dimension value mapping from renamed reference files
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Building dimension value mapping")
    print(f"{'=' * 70}")
    dimension_mapping = build_dimension_value_mapping(output_root, repo_root)
    
    # Copy CPLEX folder from data_test
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Copying CPLEX folder")
    print(f"{'=' * 70}")
    copy_cplex_folder(output_root, repo_root)
    
    # Copy pSettings files from data_test
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Copying pSettings files")
    print(f"{'=' * 70}")
    copy_psettings_files(output_root, repo_root)
    
    # Copy and expand default files
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Copying and expanding default files")
    print(f"{'=' * 70}")
    copy_and_expand_default_files(output_root, repo_root)
    
    # Validate tech-fuel combinations (final validation step)
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Validating tech-fuel combinations")
    print(f"{'=' * 70}")
    validate_tech_fuel_combinations(output_root, repo_root)
    
    # Ensure all datapackage.json files exist
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Ensuring all datapackage.json files exist")
    print(f"{'=' * 70}")
    created_stubs = ensure_all_datapackage_files_exist(output_root, repo_root, created_csv_symbols)
    if created_stubs:
        print(f"  Created {len(created_stubs)} missing files:")
        for stub in created_stubs:
            format_info = f" [{stub.get('format', 'unknown')} format"
            if stub.get('rows', 0) > 0:
                format_info += f", {stub['rows']} row(s)"
            format_info += f", {stub['fields']} column(s)]"
            print(f"    - {stub['resource_name']} -> {stub['path']}{format_info}")
    else:
        print(f"  All datapackage.json files already exist ✓")
    
    # Post-process pGenDataInput (has special logic)
    pGenDataInput_path = output_root / "supply" / "pGenDataInput.csv"
    if pGenDataInput_path.exists():
        print(f"\n{'=' * 70}")
        print(f"[Post-processing] Applying rules to pGenDataInput.csv")
        print(f"{'=' * 70}")
        postprocess_pGenDataInput(pGenDataInput_path, extras_root, repo_root)
    
    # Post-process all other CSV files (generic cleanup only)
    for csv_file in output_root.rglob("*.csv"):
        if csv_file.name == "pGenDataInput.csv":
            continue
        postprocess_csv(csv_file, output_root, extras_root, repo_root)
    
    # Merge Storage rows from pGenDataInput to pStorageDataInput (AFTER column renaming and pGenDataInput processing)
    print(f"\n{'=' * 70}")
    print(f"[Post-processing] Merging Storage rows from pGenDataInput to pStorageDataInput")
    print(f"{'=' * 70}")
    merge_storage_from_gen_to_storage(output_root, repo_root)
    
    # Rename columns to match datapackage.json schema (LAST STEP - after all other processing)
    if rename_columns:
        rename_columns_from_datapackage(output_root, repo_root, dimension_mapping)
    else:
        print(f"\n{'=' * 70}")
        print(f"[Post-processing] Skipping column renaming (--no-rename-columns specified)")
        print(f"{'=' * 70}")


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


# Cache for datapackage resources mapping
_DATAPACKAGE_RESOURCES_CACHE: Optional[Dict[str, Dict]] = None


def load_datapackage_resources(repo_root: Path) -> Dict[str, Dict]:
    """Load full datapackage.json and return resource name -> resource dict mapping.
    
    Uses caching to avoid reloading the file multiple times.
    
    Returns:
        Dictionary mapping resource name to dict with 'path', 'field_names', 'format', 'dimensions', 'encoding' (empty for wide resources), 'schema', and 'foreign_key_fields'
    """
    global _DATAPACKAGE_RESOURCES_CACHE
    if _DATAPACKAGE_RESOURCES_CACHE is not None:
        return _DATAPACKAGE_RESOURCES_CACHE
    
    datapackage_path = repo_root / "epm" / "input" / "datapackage.json"
    if not datapackage_path.exists():
        return {}
    
    try:
        with open(datapackage_path, "r") as f:
            datapackage = json.load(f)
        
        resources_map = {}
        for resource in datapackage.get("resources", []):
            resource_name = resource.get("name", "")
            if not resource_name:
                continue
            
            schema = resource.get("schema", {})
            fields = schema.get("fields", [])
            field_names = [field.get("name") for field in fields if field.get("name")]
            
            # Extract foreignKeys to identify which fields are GAMS sets
            foreign_keys = schema.get("foreignKeys", [])
            # Build set of field names that have foreignKeys and map to reference field names
            foreign_key_fields = set()
            foreign_key_ref_map = {}  # Maps field name -> reference field name (for dimension lookup)
            for fk in foreign_keys:
                fk_fields = fk.get("fields", [])
                foreign_key_fields.update(fk_fields)
                # Get reference field names for dimension lookup
                ref_fields = fk.get("reference", {}).get("fields", [])
                if fk_fields and ref_fields:
                    for i, field_name in enumerate(fk_fields):
                        # Map field name to reference field name (e.g., z1 -> z)
                        ref_field = ref_fields[i] if i < len(ref_fields) else ref_fields[0]
                        foreign_key_ref_map[field_name] = ref_field
            
            custom = resource.get("custom", {})
            format_type = custom.get("format", "long")  # Default to long if not specified
            dimensions = custom.get("dimensions", [])
            # For wide resources, encoding is removed in new structure
            # For long resources, keep encoding if present
            encoding = {} if format_type == "wide" else custom.get("encoding", {})
            
            resources_map[resource_name] = {
                "path": resource.get("path", f"{resource_name}.csv"),
                "field_names": field_names,
                "format": format_type,
                "dimensions": dimensions,
                "encoding": encoding,
                "schema": schema,  # Keep full schema for field type info
                "foreign_key_fields": foreign_key_fields,  # Fields that have foreignKeys
                "foreign_key_ref_map": foreign_key_ref_map,  # Maps field name -> reference field name for dimension lookup
            }
        
        _DATAPACKAGE_RESOURCES_CACHE = resources_map
        return resources_map
    except Exception as e:
        print(f"  [Datapackage check] Failed to load datapackage.json: {e}")
        return {}


def get_years_from_output(output_root: Path) -> List[int]:
    """Get years from y.csv file if it exists in output directory.
    
    Args:
        output_root: Root directory for CSV outputs
    
    Returns:
        List of years, or empty list if y.csv not found
    """
    y_file = output_root / "y.csv"
    if not y_file.exists():
        return []
    
    try:
        df_y = pd.read_csv(y_file)
        if df_y.empty:
            return []
        
        # Get first column (year column)
        year_col = df_y.iloc[:, 0]
        years = pd.to_numeric(year_col, errors="coerce").dropna().astype(int).tolist()
        return sorted(years)
    except Exception:
        return []




def extract_dimension_values_from_source(output_root: Path, dimension: str, repo_root: Path) -> List[str]:
    """Extract dimension values directly from reference resources.
    
    Reads dimension values directly from reference CSV files without pattern formatting.
    Uses dimension mapping: z/c -> zcmap, y -> y, q/d/t -> pHours, tech/fuel -> pTechFuel.
    
    Args:
        output_root: Root directory for CSV outputs
        dimension: Name of the dimension (e.g., 't', 'q', 'd', 'year', 'z', 'c', 'tech', 'fuel')
        repo_root: Repository root path (to load datapackage resources)
    
    Returns:
        List of dimension values as strings
    """
    # Map dimensions to their source resources
    dimension_to_resource = {
        "z": ("zcmap", "z"),
        "c": ("zcmap", "c"),
        "y": ("y", "y"),
        "year": ("y", "y"),
        "q": ("pHours", "q"),
        "d": ("pHours", "d"),
        "t": ("pHours", "t"),
        "tech": ("pTechFuel", "tech"),
        "fuel": ("pTechFuel", "fuel"),
        "f": ("pTechFuel", "fuel"),
    }
    
    if dimension not in dimension_to_resource:
        return []
    
    source_resource, source_field = dimension_to_resource[dimension]
    resources_map = load_datapackage_resources(repo_root)
    
    if source_resource not in resources_map:
        return []
    
    source_path = output_root / resources_map[source_resource]["path"]
    if not source_path.exists():
        return []
    
    try:
        df_source = pd.read_csv(source_path)
        if source_field in df_source.columns:
            values = df_source[source_field].dropna().astype(str).str.strip().unique().tolist()
            return sorted(values)
    except Exception:
        pass
    
    return []


def _print_stub_verbose_output(
    resource_name: str,
    datapackage_path: str,
    df: pd.DataFrame,
    resource_info: Dict,
    log_prefix: str,
    gdx_symbol: Optional[str] = None,
) -> None:
    """Print verbose output for created stub files (shared by datapackage-based stubs).
    
    Args:
        resource_name: Resource name (CSV symbol name)
        datapackage_path: Path from datapackage.json
        df: Created DataFrame
        resource_info: Resource information dict from datapackage
        log_prefix: Prefix for log messages (e.g., "[Created from datapackage]", "[Empty GDX]", "[Datapackage stub]")
        gdx_symbol: Optional GDX symbol name (for logging context)
    """
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    row_count = len(df)
    col_count = len(df.columns)
    dims_str = f", dimensions: {', '.join(dimensions)}" if dimensions else ""
    
    # Identify index columns and time columns
    if dimensions:
        field_names = resource_info.get("field_names", [])
        index_cols = [col for col in df.columns if col in field_names and col not in dimensions]
        time_cols = [col for col in df.columns if col not in index_cols]
    else:
        index_cols = list(df.columns)
        time_cols = []
    
    # Build the main message
    if gdx_symbol:
        main_msg = f"  {log_prefix} Created {resource_name} (from '{gdx_symbol}') -> {datapackage_path} ({format_type} format{dims_str}, {row_count} row(s), {col_count} column(s))"
    else:
        main_msg = f"  {log_prefix} Created {resource_name} -> {datapackage_path} ({format_type} format{dims_str}, {row_count} row(s), {col_count} column(s))"
    
    print(main_msg)
    if index_cols:
        print(f"    Index columns ({len(index_cols)}): {', '.join(index_cols[:10])}{'...' if len(index_cols) > 10 else ''}")
    if time_cols:
        print(f"    Time columns ({len(time_cols)}): {', '.join(time_cols[:10])}{'...' if len(time_cols) > 10 else ''}")


def _create_empty_dataframe_from_datapackage(csv_symbol: str, gdx_symbol: str, output_root: Path, log_prefix: str) -> pd.DataFrame:
    """Create empty dataframe from datapackage.json for missing or empty GDX symbols.
    
    Args:
        csv_symbol: CSV symbol name
        gdx_symbol: GDX symbol name (for logging)
        output_root: Root directory for CSV outputs
        log_prefix: Prefix for log messages (e.g., "[Created from datapackage]" or "[Empty GDX]")
    
    Returns:
        Empty DataFrame with proper structure from datapackage.json
    """
    resources_map = load_datapackage_resources(REPO_ROOT)
    if csv_symbol in resources_map:
        resource_info = resources_map[csv_symbol]
        df_csv_data = create_empty_dataframe_for_resource(resource_info, output_root, REPO_ROOT)
        
        # Use the same verbose output function as ensure_all_datapackage_files_exist
        datapackage_path = resource_info.get("path", f"{csv_symbol}.csv")
        _print_stub_verbose_output(csv_symbol, datapackage_path, df_csv_data, resource_info, log_prefix, gdx_symbol)
        
        return df_csv_data
    else:
        # Fallback to simple field names if resource not found
        resources_map = load_datapackage_resources(REPO_ROOT)
        if csv_symbol in resources_map:
            field_names = resources_map[csv_symbol].get("field_names", [])
            if field_names:
                df_csv_data = pd.DataFrame(columns=field_names)
                print(f"  {log_prefix} {csv_symbol} (from '{gdx_symbol}') - created with field names from datapackage.json")
                return df_csv_data
        else:
            df_csv_data = pd.DataFrame()
            if log_prefix == "[Created from datapackage]":
                print(f"  {log_prefix} {csv_symbol} (from '{gdx_symbol}') - created empty stub (no schema found)")
            return df_csv_data


def create_empty_dataframe_for_resource(resource_info: Dict, output_root: Path, repo_root: Path) -> pd.DataFrame:
    """Create an empty DataFrame with appropriate structure based on resource format.
    
    Args:
        resource_info: Resource information dict from datapackage
        output_root: Root directory for CSV outputs (to read source files)
        repo_root: Repository root path (to load datapackage resources)
    
    Returns:
        DataFrame with appropriate structure, with at least one row
    """
    field_names = resource_info.get("field_names", [])
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    schema = resource_info.get("schema", {})
    
    if not field_names:
        return pd.DataFrame()
    
    # Get field types from schema
    fields_dict = {f["name"]: f.get("type", "string") for f in schema.get("fields", [])}
    
    if format_type == "long":
        # Long format: create one row with placeholder values
        row_data = {}
        for field_name in field_names:
            field_type = fields_dict.get(field_name, "string")
            if field_type in ("integer", "number"):
                row_data[field_name] = [0]
            else:
                row_data[field_name] = [""]
        return pd.DataFrame(row_data)
    
    elif format_type == "wide":
        # Wide format: create columns for dimensions specified in datapackage
        # Dimensions can be ANYTHING (time, spatial, technology, etc.) - not just time
        dimensions_set = set(dimensions)
        
        # Get index columns (all fields except dimensions and value columns)
        index_cols = []
        for field_name in field_names:
            if field_name in dimensions_set:
                # This dimension will become columns (not an index column)
                continue
            elif field_name.lower() in ("value", "val"):
                # Value column - skip (values will be in the dimension columns)
                continue
            else:
                index_cols.append(field_name)
        
        # Create DataFrame with index columns
        if index_cols:
            # Create one empty row to show structure (ensures first column is full)
            df = pd.DataFrame(columns=index_cols)
            # Add one row with empty values
            df.loc[0] = [""] * len(index_cols)
        else:
            df = pd.DataFrame()
            # Even if no index cols, add one row to ensure structure
            df.loc[0] = {}
        
        # Add dimension columns - treat ALL dimensions uniformly
        # (year, t, q, d, zone, tech, generator, etc.)
        for dim in dimensions:
            dim_values = extract_dimension_values_from_source(output_root, dim, repo_root)
            if dim_values:
                # Add columns for each dimension value
                for dim_value in dim_values:
                    df[str(dim_value)] = None
            # If extraction fails, log warning but continue
            # (This should rarely happen if encoding is properly configured)
        
        return df
    
    # Default: just headers with one row
    df = pd.DataFrame(columns=field_names)
    df.loc[0] = [""] * len(field_names)
    return df


def ensure_all_datapackage_files_exist(output_root: Path, repo_root: Path, created_files: set[str]) -> List[Dict]:
    """Ensure all files from datapackage.json exist, creating empty ones with headers if missing.
    
    For long format: creates a DataFrame with one placeholder row.
    For wide format with time dimensions: creates columns for time periods in chronological order.
    
    Args:
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
        created_files: Set of resource names that were created during conversion
    
    Returns:
        List of dictionaries with info about created files
    """
    resources_map = load_datapackage_resources(repo_root)
    if not resources_map:
        print("  [Datapackage check] No resources found in datapackage.json")
        return []
    
    # Get years from y.csv if available
    years = get_years_from_output(output_root)
    if years:
        print(f"  [Datapackage check] Found {len(years)} years in y.csv: {min(years)}-{max(years)}")
    else:
        print(f"  [Datapackage check] y.csv not found, using default year range for wide format files")
    
    created_stubs = []
    
    for resource_name, resource_info in resources_map.items():
        # Skip pSettings (handled separately)
        if resource_name == "pSettings":
            continue
        
        # Skip if already created
        if resource_name in created_files:
            continue
        
        # Get path from datapackage
        datapackage_path = resource_info["path"]
        target_path = output_root / datapackage_path
        
        # Skip if file already exists (might have been created by post-processing)
        if target_path.exists():
            continue
        
        # Create empty DataFrame with appropriate structure
        df_empty = create_empty_dataframe_for_resource(resource_info, output_root, repo_root)
        
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write empty CSV with headers
        df_empty.to_csv(target_path, index=False, na_rep="")
        
        # Use the same verbose output function as datapackage-based stubs
        _print_stub_verbose_output(resource_name, datapackage_path, df_empty, resource_info, "[Datapackage stub]")
        
        format_type = resource_info.get("format", "long")
        row_count = len(df_empty)
        col_count = len(df_empty.columns)
        
        created_stubs.append({
            "resource_name": resource_name,
            "path": datapackage_path,
            "format": format_type,
            "rows": row_count,
            "fields": col_count,
        })
    
    return created_stubs


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
    extras_root = export_root / ".legacy"
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
    4. Exports extra GDX symbols (not in CSV_LAYOUT) to .legacy folder
    
    Args:
        gdx_path: Path to input GDX file
        mapping_path: Path to symbol_mapping.csv (CSV → GDX symbol mapping)
        output_root: Root directory for CSV outputs
        extras_root: Directory for extra GDX symbols not in CSV_LAYOUT (saved to .legacy folder)
        overwrite: Whether to overwrite existing CSV files
    
    Returns:
        Dictionary with conversion summary, errors, and statistics
    """
    # Load symbol mapping and build CSV_LAYOUT
    symbol_mapping_df = load_symbol_mapping(mapping_path)
    csv_layout = build_csv_layout(symbol_mapping_df)

    # Load datapackage resources to check which symbols should be created from schema
    resources_map = load_datapackage_resources(REPO_ROOT)

    # Load GDX file
    gdx_container = gt.Container()
    gdx_container.read(str(gdx_path))
    loaded_gdx_symbols = set(gdx_container.data.keys())

    summary: List[Dict] = []
    extras_written: List[Dict] = []
    skipped: List[Path] = []
    missing_in_gdx: List[Tuple[str, str]] = []
    created_from_datapackage: List[Tuple[str, str]] = []
    empty_in_gdx: List[Tuple[str, str]] = []
    used_gdx_symbols: set[str] = set()
    created_csv_symbols: set[str] = set()

    # Process each CSV_LAYOUT entry
    for entry in csv_layout:
        csv_symbol = entry["primary_symbol"]
        gdx_symbol = entry["gdx_symbol"]

        # Transform GDX data to CSV format
        df_csv_data = build_frame(gdx_container, gdx_symbol, csv_symbol, entry)
        created_from_schema = False
        if df_csv_data is None:
            # GDX symbol not found - check if it's in datapackage.json
            if csv_symbol in resources_map:
                # Create empty dataframe from datapackage.json schema
                df_csv_data = _create_empty_dataframe_from_datapackage(csv_symbol, gdx_symbol, output_root, "[Created from datapackage]")
                created_from_schema = True
                created_from_datapackage.append((csv_symbol, gdx_symbol))
            else:
                missing_in_gdx.append((csv_symbol, gdx_symbol))
                continue
        else:
            df_csv_data = df_csv_data.copy()

        if not created_from_schema and df_csv_data.empty:
            empty_in_gdx.append((csv_symbol, gdx_symbol))
        
        # Ensure empty DataFrames have correct structure from datapackage
        if df_csv_data.empty and not created_from_schema:
            df_csv_data = _create_empty_dataframe_from_datapackage(csv_symbol, gdx_symbol, output_root, "[Empty GDX]")

        csv_target_path = output_root / Path(entry["relative_path"])
        csv_target_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip exporting pSettings.csv
        if csv_symbol == "pSettings":
            continue
        
        if not overwrite and csv_target_path.exists():
            skipped.append(csv_target_path)
            continue

        df_csv_data.to_csv(csv_target_path, index=False, na_rep="")
        created_csv_symbols.add(csv_symbol)  # Track created symbol
        
        # Add to summary (pSettings is already skipped above)
        summary.append(
            {
                "csv_symbol": csv_symbol,
                "gdx_symbol": gdx_symbol,
                "rows": len(df_csv_data),
                "path": csv_target_path.relative_to(output_root).as_posix(),
            }
        )

        if not created_from_schema:
            used_gdx_symbols.add(gdx_symbol)

    # Export extra GDX symbols (not in CSV_LAYOUT) to .legacy folder
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
        "created_from_datapackage": created_from_datapackage,
        "empty_in_gdx": empty_in_gdx,
        "created_csv_symbols": created_csv_symbols,
    }


def print_report(results: Dict[str, List], output_root: Path) -> None:
    """Emit a concise report mirroring the notebook prints."""
    missing_in_gdx = results["missing_in_gdx"]
    created_from_datapackage = results.get("created_from_datapackage", [])
    empty_in_gdx = results["empty_in_gdx"]
    extras_written = results["extras_written"]
    skipped = results["skipped"]

    if missing_in_gdx:
        formatted = ", ".join(f"{csv} (expected '{gdx}')" for csv, gdx in sorted(missing_in_gdx))
        print("Symbols missing in GDX:", formatted)
    if created_from_datapackage:
        formatted = ", ".join(f"{csv} (from '{gdx}')" for csv, gdx in sorted(created_from_datapackage))
        print("Symbols absent in GDX; created empty CSV from datapackage.json:", formatted)
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
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True, help="Skip existing files instead of replacing (default: overwrite).")
    parser.add_argument("--no-rename-columns", dest="rename_columns", action="store_false", default=True, help="Skip column renaming step (default: columns are renamed).")
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
    created_csv_symbols = results.get("created_csv_symbols", set())
    apply_postprocessing(export_root, extras_root, REPO_ROOT, created_csv_symbols, rename_columns=args.rename_columns)


if __name__ == "__main__":
    main()
