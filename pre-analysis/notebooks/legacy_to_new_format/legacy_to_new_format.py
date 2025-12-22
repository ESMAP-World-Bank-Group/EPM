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

        if not overwrite and csv_target_path.exists():
            skipped.append(csv_target_path)
            continue

        df_csv_data.to_csv(csv_target_path, index=False, na_rep="")
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

    df = pd.DataFrame(results["summary"] + results["extras_written"]).sort_values("path")
    print(df)
    print_report(results, export_root)
    print(f"\nExports written under: {export_root}")


if __name__ == "__main__":
    main()
