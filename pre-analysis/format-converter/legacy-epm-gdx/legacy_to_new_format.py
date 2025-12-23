"""Convert a legacy EPM GDX file into the newer CSV layout using datapackage.json schema.

Minimal, schema-driven implementation. All logic is driven by datapackage.json.
No hard-coded variable-specific logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import gams.transfer as gt
except ImportError as err:
    raise ImportError("Install the GAMS Python API before running this script.") from err

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_GDX_PATH = SCRIPT_DIR / "input" / "input_epm_Turkiye_v8.gdx"
DEFAULT_MAPPING_PATH = SCRIPT_DIR / "symbol_mapping.csv"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output"
DEFAULT_TARGET_FOLDER = "data"

# Cache for datapackage resources
_DATAPACKAGE_CACHE: Optional[Dict[str, Dict]] = None


# --------------------------------------------------------------------------- #
# Core Functions
# --------------------------------------------------------------------------- #

def parse_datapackage(repo_root: Path) -> Dict[str, Dict]:
    """Load and parse datapackage.json into resource map.
    
    Returns:
        Dict mapping resource name to resource info (path, format, dimensions, encoding, field_names, schema)
    """
    global _DATAPACKAGE_CACHE
    if _DATAPACKAGE_CACHE is not None:
        return _DATAPACKAGE_CACHE
    
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
            
            custom = resource.get("custom", {})
            format_type = custom.get("format", "long")
            dimensions = custom.get("dimensions", [])
            encoding = custom.get("encoding", {})
            
            resources_map[resource_name] = {
                "path": resource.get("path", f"{resource_name}.csv"),
                "format": format_type,
                "dimensions": dimensions,
                "encoding": encoding,
                "field_names": field_names,
                "schema": schema,
            }
        
        _DATAPACKAGE_CACHE = resources_map
        return resources_map
    except Exception as e:
        print(f"Failed to load datapackage.json: {e}")
        return {}


def format_values_with_pattern(values: List, pattern: str) -> List[str]:
    """Format extracted values according to pattern.
    
    Args:
        values: Raw values from source
        pattern: Pattern string (e.g., "tN", "QN", "YYYY")
    
    Returns:
        Formatted values as strings
    """
    if not pattern:
        return sorted([str(v) for v in values])
    
    if "tN" in pattern or ("t" in pattern.lower() and "N" in pattern.upper()):
        numeric_values = []
        for v in values:
            v_str = str(v)
            if v_str.startswith("t") and v_str[1:].isdigit():
                numeric_values.append(int(v_str[1:]))
            elif v_str.isdigit():
                numeric_values.append(int(v_str))
        if numeric_values:
            return [f"t{t}" for t in sorted(numeric_values)]
    
    elif "Q" in pattern.upper() and "N" in pattern.upper():
        numeric_values = [int(v) for v in values if str(v).isdigit() or isinstance(v, (int, float))]
        if numeric_values:
            return [f"Q{q}" for q in sorted(numeric_values)]
    
    return sorted([str(v) for v in values])


def extract_dimension_values(encoding: Dict, dimension: str, output_root: Path, repo_root: Path) -> List[str]:
    """Extract dimension values from source CSV files based on encoding.
    
    Args:
        encoding: Encoding dict from datapackage.json
        dimension: Name of the dimension
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
    
    Returns:
        List of dimension values as strings
    """
    dim_encoding = encoding.get(dimension, {})
    if not dim_encoding:
        return []
    
    pattern = dim_encoding.get("pattern", "")
    source = dim_encoding.get("source", "")
    
    if source:
        parts = source.split(".")
        if len(parts) == 2:
            source_resource, source_field = parts
            resources_map = parse_datapackage(repo_root)
            
            if source_resource in resources_map:
                source_path = output_root / resources_map[source_resource]["path"]
                if source_path.exists():
                    try:
                        df_source = pd.read_csv(source_path)
                        if source_field in df_source.columns:
                            values = df_source[source_field].dropna().unique().tolist()
                            return format_values_with_pattern(values, pattern)
                    except Exception:
                        pass
    
    # Fallback: try to infer from dimension name
    resources_map = parse_datapackage(repo_root)
    potential_resources = [dimension]
    if dimension == "year":
        potential_resources = ["y", "year"]
    
    for resource_name in potential_resources:
        if resource_name in resources_map:
            source_path = output_root / resources_map[resource_name]["path"]
            if source_path.exists():
                try:
                    df_source = pd.read_csv(source_path)
                    if dimension in df_source.columns:
                        values = df_source[dimension].dropna().unique().tolist()
                        return format_values_with_pattern(values, pattern)
                    elif len(df_source.columns) > 0:
                        if dimension == "year" and "y" in df_source.columns:
                            values = df_source["y"].dropna().unique().tolist()
                            return format_values_with_pattern(values, pattern)
                        else:
                            values = df_source.iloc[:, 0].dropna().unique().tolist()
                            return format_values_with_pattern(values, pattern)
                except Exception:
                    pass
    
    return []


def create_empty_dataframe(resource_info: Dict, output_root: Path, repo_root: Path) -> pd.DataFrame:
    """Create empty DataFrame with structure from datapackage.json schema.
    
    Args:
        resource_info: Resource information from datapackage
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
    
    Returns:
        DataFrame with appropriate structure
    """
    field_names = resource_info.get("field_names", [])
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    encoding = resource_info.get("encoding", {})
    schema = resource_info.get("schema", {})
    
    if not field_names:
        return pd.DataFrame()
    
    fields_dict = {f["name"]: f.get("type", "string") for f in schema.get("fields", [])}
    
    if format_type == "long":
        row_data = {}
        for field_name in field_names:
            field_type = fields_dict.get(field_name, "string")
            row_data[field_name] = [0] if field_type in ("integer", "number") else [""]
        return pd.DataFrame(row_data)
    
    elif format_type == "wide":
        # Identify which dimensions become columns (wide format) vs rows (index columns)
        column_dimensions = set()
        dimension_to_field = {}  # Map dimension name to field name
        
        for dim, dim_encoding in encoding.items():
            if dim_encoding.get("type") == "columns":
                column_dimensions.add(dim)
            # Map dimension to its field name
            field_name = dim_encoding.get("field")
            if field_name:
                dimension_to_field[dim] = field_name
        
        # Also check if dimension name directly matches a field name
        for dim in dimensions:
            if dim in field_names:
                dimension_to_field[dim] = dim
        
        # Index columns: fields that are NOT mapped to column-type dimensions
        index_cols = []
        for fname in field_names:
            # Skip value columns
            if fname.lower() in ("value", "val"):
                continue
            
            # Check if this field maps to a column dimension
            is_column_dim = False
            for dim in column_dimensions:
                # Check if dimension maps to this field
                if dimension_to_field.get(dim) == fname:
                    is_column_dim = True
                    break
            
            if not is_column_dim:
                index_cols.append(fname)
        
        df = pd.DataFrame(columns=index_cols)
        if index_cols:
            df.loc[0] = [""] * len(index_cols)
        else:
            df.loc[0] = {}
        
        # Add wide columns only for column-type dimensions
        for dim in dimensions:
            dim_encoding = encoding.get(dim, {})
            if dim_encoding.get("type") == "columns":
                dim_values = extract_dimension_values(encoding, dim, output_root, repo_root)
                for dim_value in dim_values:
                    df[str(dim_value)] = None
        
        return df
    
    df = pd.DataFrame(columns=field_names)
    df.loc[0] = [""] * len(field_names)
    return df


def transform_gdx_to_dataframe(
    gdx_records: pd.DataFrame,
    resource_info: Dict,
    gdx_symbol: str,
    header_col_index: Optional[int] = None,
    csv_symbol: str = ""
) -> pd.DataFrame:
    """Transform GDX records to CSV format based on datapackage.json schema or header_cols.
    
    Args:
        gdx_records: DataFrame from GDX records
        resource_info: Resource information from datapackage
        gdx_symbol: GDX symbol name (for error messages)
        header_col_index: 0-indexed column index to unstack (from mapping file, takes priority)
        csv_symbol: CSV symbol name (for error messages)
    
    Returns:
        DataFrame in CSV format
    """
    value_col = None
    for candidate in ("value", "Value"):
        if candidate in gdx_records.columns:
            value_col = candidate
            break
    
    gdx_dimension_cols = [col for col in gdx_records.columns if col != value_col]
    
    # Priority 1: Use header_cols from mapping file if provided (for non-empty files)
    if header_col_index is not None:
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
            pivot = gdx_records.pivot_table(
                index=csv_index_cols if csv_index_cols else None,
                columns=header_col,
                values=value_col,
                aggfunc="first",
                observed=False,
            )
            pivot.columns = [str(col) for col in pivot.columns]
            return pivot.reset_index().reset_index(drop=True)
        else:
            return gdx_records.reset_index(drop=True)
    
    # Priority 2: Use datapackage.json schema (fallback for empty files or when header_cols not specified)
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    field_names = resource_info.get("field_names", [])
    
    if format_type == "wide" and dimensions:
        # Determine which dimension to unstack based on field order
        # Find the first dimension in field_names that appears in dimensions
        unstack_dim = None
        for fname in field_names:
            if fname in dimensions:
                unstack_dim = fname
                break
        
        if unstack_dim and unstack_dim in gdx_dimension_cols:
            # Unstack this dimension
            index_cols = [col for col in gdx_dimension_cols if col != unstack_dim]
            if value_col:
                pivot = gdx_records.pivot_table(
                    index=index_cols if index_cols else None,
                    columns=unstack_dim,
                    values=value_col,
                    aggfunc="first",
                    observed=False,
                )
                pivot.columns = [str(col) for col in pivot.columns]
                return pivot.reset_index()
            else:
                return gdx_records.reset_index(drop=True)
    
    # Long format or no unstacking needed
    if value_col:
        return gdx_records.rename(columns={value_col: "value"}).reset_index(drop=True)
    return gdx_records.reset_index(drop=True)


def populate_wide_dimensions(
    df: pd.DataFrame,
    resource_info: Dict,
    output_root: Path,
    repo_root: Path
) -> pd.DataFrame:
    """Fill missing wide format columns from declared sources.
    
    Args:
        df: DataFrame to populate
        resource_info: Resource information from datapackage
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
    
    Returns:
        DataFrame with missing columns added
    """
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    encoding = resource_info.get("encoding", {})
    
    if format_type != "wide" or not dimensions:
        return df
    
    for dim in dimensions:
        dim_values = extract_dimension_values(encoding, dim, output_root, repo_root)
        for dim_value in dim_values:
            dim_value_str = str(dim_value)
            if dim_value_str not in df.columns:
                df[dim_value_str] = None
    
    return df


def rename_columns_to_schema(df: pd.DataFrame, resource_info: Dict) -> pd.DataFrame:
    """Rename columns to match datapackage.json field names.
    
    For wide format, only renames index columns (not dimension value columns).
    
    Args:
        df: DataFrame to rename
        resource_info: Resource information from datapackage
    
    Returns:
        DataFrame with renamed columns
    """
    field_names = resource_info.get("field_names", [])
    format_type = resource_info.get("format", "long")
    dimensions = resource_info.get("dimensions", [])
    encoding = resource_info.get("encoding", {})
    
    if not field_names:
        return df
    
    rename_dict = {}
    current_cols = list(df.columns)
    
    if format_type == "wide":
        # Identify which dimensions become columns vs rows (same logic as create_empty_dataframe)
        column_dimensions = set()
        dimension_to_field = {}
        
        for dim, dim_encoding in encoding.items():
            if dim_encoding.get("type") == "columns":
                column_dimensions.add(dim)
            field_name = dim_encoding.get("field")
            if field_name:
                dimension_to_field[dim] = field_name
        
        for dim in dimensions:
            if dim in field_names:
                dimension_to_field[dim] = dim
        
        # Index field names: fields that are NOT mapped to column-type dimensions
        index_field_names = []
        for fname in field_names:
            if fname.lower() in ("value", "val"):
                continue
            is_column_dim = False
            for dim in column_dimensions:
                if dimension_to_field.get(dim) == fname:
                    is_column_dim = True
                    break
            if not is_column_dim:
                index_field_names.append(fname)
        
        def is_dimension_value(col_name: str) -> bool:
            """Check if column name represents a dimension value (e.g., 't1', '2024', 'Q1')."""
            col_str = str(col_name)
            # Check if it's exactly a dimension name
            if col_str in dimensions:
                return True
            # Check if it matches dimension value patterns
            for dim in dimensions:
                dim_encoding = encoding.get(dim, {})
                if dim_encoding.get("type") == "columns":
                    # This is a column dimension, check if col_name matches its pattern
                    if dim == "year" and col_str.isdigit():
                        return True
                    if dim == "t" and col_str.startswith("t") and col_str[1:].isdigit():
                        return True
                    if dim in ("q", "quarter") and col_str.startswith("Q") and col_str[1:].isdigit():
                        return True
                    if col_str.startswith(dim) and len(col_str) > len(dim):
                        suffix = col_str[len(dim):]
                        if suffix.isdigit() or suffix.isalpha():
                            return True
            return False
        
        index_col_idx = 0
        for col_name in current_cols:
            if col_name in index_field_names:
                continue
            if is_dimension_value(col_name):
                continue
            if index_col_idx < len(index_field_names):
                expected_name = index_field_names[index_col_idx]
                if col_name != expected_name:
                    rename_dict[col_name] = expected_name
                index_col_idx += 1
    else:
        for col_idx, col_name in enumerate(current_cols):
            if col_idx < len(field_names):
                expected_name = field_names[col_idx]
                if col_name != expected_name:
                    rename_dict[col_name] = expected_name
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df


def load_resource_csv(
    resource_name: str,
    resource_info: Dict,
    gdx_container: gt.Container,
    gdx_symbol: str,
    output_root: Path,
    repo_root: Path,
    header_col_index: Optional[int] = None
) -> pd.DataFrame:
    """Load CSV from GDX or create empty from schema.
    
    Args:
        resource_name: Resource name (CSV symbol)
        resource_info: Resource information from datapackage
        gdx_container: GDX container
        gdx_symbol: GDX symbol name
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
        header_col_index: 0-indexed column index to unstack (from mapping file)
    
    Returns:
        DataFrame in CSV format
    """
    if gdx_symbol in gdx_container:
        gdx_records = gdx_container[gdx_symbol].records
        if gdx_records is not None and not gdx_records.empty:
            df = transform_gdx_to_dataframe(
                gdx_records.copy(), 
                resource_info, 
                gdx_symbol,
                header_col_index=header_col_index,
                csv_symbol=resource_name
            )
            df = populate_wide_dimensions(df, resource_info, output_root, repo_root)
            return df
    
    # Create empty from schema
    return create_empty_dataframe(resource_info, output_root, repo_root)


def write_csv(df: pd.DataFrame, resource_info: Dict, output_root: Path) -> None:
    """Write DataFrame to CSV at resource path.
    
    Args:
        df: DataFrame to write
        resource_info: Resource information from datapackage
        output_root: Root directory for CSV outputs
    """
    csv_path = output_root / resource_info["path"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = rename_columns_to_schema(df, resource_info)
    df.to_csv(csv_path, index=False, na_rep="")


def load_symbol_mapping(mapping_path: Path) -> Dict[str, Dict]:
    """Load GDX symbol to CSV symbol mapping with header_cols information.
    
    Args:
        mapping_path: Path to symbol_mapping.csv
    
    Returns:
        Dict mapping CSV symbol to dict with 'gdx_symbol' and 'header_cols' (0-indexed, or None)
    """
    if not mapping_path.exists():
        return {}
    
    df = pd.read_csv(mapping_path)
    if "csv_symbol" not in df.columns or "gdx_symbol" not in df.columns:
        return {}
    
    mapping = {}
    for _, row in df.iterrows():
        csv_symbol = str(row["csv_symbol"]).strip()
        if not csv_symbol:
            continue
        
        gdx_symbol = str(row.get("gdx_symbol", csv_symbol)).strip()
        
        # Parse header_cols (can be empty string, NaN, or integer)
        header_cols = row.get("header_cols", "")
        header_index = None
        if pd.notna(header_cols) and str(header_cols).strip() != "":
            try:
                header_index = int(header_cols)
            except (ValueError, TypeError):
                header_index = None
        
        mapping[csv_symbol] = {
            "gdx_symbol": gdx_symbol,
            "header_cols": header_index,
        }
    
    return mapping


def convert_gdx_to_csv(
    gdx_path: Path,
    mapping_path: Path,
    output_root: Path,
    repo_root: Path,
    overwrite: bool = True
) -> Dict:
    """Main conversion function.
    
    Args:
        gdx_path: Path to input GDX file
        mapping_path: Path to symbol_mapping.csv
        output_root: Root directory for CSV outputs
        repo_root: Repository root path
        overwrite: Whether to overwrite existing CSV files
    
    Returns:
        Dictionary with conversion summary
    """
    resources_map = parse_datapackage(repo_root)
    if not resources_map:
        print("No resources found in datapackage.json")
        return {}
    
    symbol_mapping = load_symbol_mapping(mapping_path)
    
    gdx_container = gt.Container()
    gdx_container.read(str(gdx_path))
    
    summary = []
    created_resources = set()
    
    for resource_name, resource_info in resources_map.items():
        if resource_name == "pSettings":
            continue
        
        csv_path = output_root / resource_info["path"]
        if not overwrite and csv_path.exists():
            continue
        
        # Get mapping info (gdx_symbol and header_cols)
        mapping_info = symbol_mapping.get(resource_name, {})
        if isinstance(mapping_info, str):
            # Backward compatibility: if it's just a string, treat as gdx_symbol
            gdx_symbol = mapping_info
            header_col_index = None
        else:
            gdx_symbol = mapping_info.get("gdx_symbol", resource_name)
            header_col_index = mapping_info.get("header_cols")
        
        df = load_resource_csv(
            resource_name,
            resource_info,
            gdx_container,
            gdx_symbol,
            output_root,
            repo_root,
            header_col_index=header_col_index
        )
        
        write_csv(df, resource_info, output_root)
        created_resources.add(resource_name)
        
        summary.append({
            "resource": resource_name,
            "gdx_symbol": gdx_symbol,
            "rows": len(df),
            "columns": len(df.columns),
            "path": resource_info["path"],
        })
    
    # Export extra GDX symbols
    used_gdx_symbols = set()
    for mapping_info in symbol_mapping.values():
        if isinstance(mapping_info, dict):
            used_gdx_symbols.add(mapping_info.get("gdx_symbol", ""))
        else:
            # Backward compatibility
            used_gdx_symbols.add(mapping_info)
    extras_root = output_root / ".legacy"
    extras_root.mkdir(parents=True, exist_ok=True)
    
    extras = []
    for gdx_symbol in gdx_container.data.keys():
        if gdx_symbol not in used_gdx_symbols:
            try:
                gdx_records = gdx_container[gdx_symbol].records
                if gdx_records is not None:
                    extras_path = extras_root / f"{gdx_symbol}.csv"
                    gdx_records.to_csv(extras_path, index=False)
                    extras.append(gdx_symbol)
            except Exception:
                pass
    
    return {
        "summary": summary,
        "extras": extras,
        "created_resources": created_resources,
    }


def print_report(results: Dict, output_root: Path) -> None:
    """Print conversion summary report."""
    summary = results.get("summary", [])
    extras = results.get("extras", [])
    
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Total resources converted: {len(summary)}")
    print(f"Extra GDX symbols exported: {len(extras)}")
    print(f"\nOutput directory: {output_root}")
    
    if summary:
        print(f"\nConverted resources:")
        for item in summary[:10]:
            print(f"  {item['resource']:30} {item['rows']:6} rows, {item['columns']:3} cols -> {item['path']}")
        if len(summary) > 10:
            print(f"  ... and {len(summary) - 10} more")
    
    if extras:
        print(f"\nExtra GDX symbols (exported to extras/):")
        for symbol in extras[:10]:
            print(f"  {symbol}")
        if len(extras) > 10:
            print(f"  ... and {len(extras) - 10} more")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert legacy GDX to CSV using datapackage.json schema")
    parser.add_argument("--gdx", type=Path, default=DEFAULT_GDX_PATH, help="Path to input GDX file")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to symbol_mapping.csv")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE, help="Base output directory")
    parser.add_argument("--target-folder", type=str, default=DEFAULT_TARGET_FOLDER, help="Target folder name")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if not args.gdx.exists():
        raise FileNotFoundError(f"GDX file not found: {args.gdx}")
    
    output_root = (args.output_base / args.target_folder).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = convert_gdx_to_csv(
        gdx_path=args.gdx,
        mapping_path=args.mapping,
        output_root=output_root,
        repo_root=REPO_ROOT,
        overwrite=args.overwrite,
    )
    
    print_report(results, output_root)


if __name__ == "__main__":
    main()

