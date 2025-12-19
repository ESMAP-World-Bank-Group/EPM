"""Convert a legacy EPM GDX file into the newer CSV layout used by prepare-data.

How it works (summary):
- Reads a legacy GDX with the GAMS transfer API and a mapping table (`symbol_mapping.csv`)
  that aligns old GDX symbol names to the expected CSV names.
- For each symbol defined in `CSV_LAYOUT`, reshapes records (sets vs. parameters with
  header/value columns) into a normalized DataFrame and writes it to the corresponding
  `data_test/...` CSV path. Optional symbols missing in the GDX are written as empty stubs.
- Any extra GDX symbols not covered by the layout are dumped to `output/data_test/extras/`
  for inspection.
- Run from the CLI (`python legacy_to_new_format.py --gdx ... --mapping ...`) to batch-convert
  a GDX; defaults point to the sample inputs/outputs in this folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
DEFAULT_MAPPING_PATH = SCRIPT_DIR / "input" / "symbol_mapping.csv"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "output"
DEFAULT_TARGET_FOLDER = "data"


def _entry(
    csv_symbol: str,
    path: str,
    gdx_dim: int,
    header_cols: int = 0,
    symbols: Optional[List[str]] = None,
    typ: str = "par",
    value_cols: Optional[List[int]] = None,
) -> Dict:
    """Create a CSV layout entry specification.
    
    Args:
        csv_symbol: The CSV symbol name (primary identifier in CSV_LAYOUT)
        path: Relative path where the CSV file should be written
        gdx_dim: GDX dimension (number of dimensions in the GDX parameter/set)
        header_cols: Number of GDX dimensions to pivot to CSV column headers
                    (typically the last dimension like years, hours, seasons).
                    Remaining dimensions become index columns.
        symbols: Alternative GDX symbol names to check (for aliases)
        typ: Type of symbol - "par" for parameter, "set" for set
        value_cols: Explicit value column indices (rarely used)
    
    Returns:
        Dictionary specification for the CSV layout entry.
    
    Note:
        indexColumns count = gdx_dim - header_cols
        For sets, header_cols is ignored (all dimensions become index columns).
    """
    # Calculate number of index columns: total GDX dimensions minus those pivoted to headers
    num_index_cols = gdx_dim - header_cols if typ != "set" else gdx_dim
    return {
        "primary_symbol": csv_symbol,
        "relative_path": path,
        "symbols": symbols or [csv_symbol],
        "type": typ,
        "indexColumns": list(range(1, num_index_cols + 1)) if num_index_cols > 0 else [],
        "header": list(range(1, header_cols + 1)) if header_cols > 0 else [],
        "valueColumns": value_cols or [],
    }


# GDX dimension mapping: Maps GDX symbol names to their dimensionality (1D, 2D, 3D, etc.)
# This is used to determine how many index columns vs header columns the CSV should have.
# Key insight: indexColumns count = GDX_DIM - header_cols
# Example: pFuelPrice has GDX dim=3, header_cols=1 → 2 index columns (zone, fuel) + years as headers
GDX_DIM = {
    "hh": 1,
    "MapGG": 2,
    "pAnnualMaxBuildC": 3,
    "pAnnualMaxBuildZ": 3,
    "pAvailability": 2,
    "pAvailabilityH2": 2,
    "pCapexTrajectory": 2,
    "pCapexTrajectoryH2": 2,
    "pCarbonPrice": 1,
    "pCSPData": 3,
    "pDemandData": 5,
    "pDemandForecast": 3,
    "pDemandProfile": 4,
    "peak": 1,
    "pEmissionsCountry": 2,
    "pEmissionsTotal": 1,
    "pEnergyEfficiencyFactor": 2,
    "pExternalH2": 4,
    "pExtTransferLimit": 5,
    "pFuelData": 1,
    "pFuelPrice": 3,
    "pFuelTypeCarbonContent": 1,
    "pGenDataExcel": 2,
    "pH2DataExcel": 2,
    "pHours": 4,
    "pLossFactor": 3,
    "pMaxCap": 3,
    "pMaxExchangeShare": 2,
    "pMaxFuelLimit": 3,
    "pMinCap": 3,
    "pNewTransmission": 3,
    "pPlanningReserveMargin": 1,
    "pScalars": 1,
    "pSpinningReserveReqCountry": 2,
    "pSpinningReserveReqSystem": 1,
    "pStorDataExcel": 3,
    "pTechDataExcel": 2,
    "pTradePrice": 5,
    "pTranspferLimit": 4,
    "pVREgenProfile": 5,
    "pVREprofile": 5,
    "pZoneIndex": 2,
    "Relevant": 1,
    "sRelevant": 1,
    "sTopology": 2,
    "y": 1,
    "zcnexpExcel": 2,
    "zcmapExcel": 2,
    "zext": 1,
    # Note: GDX_DIM only contains GDX symbol names, not CSV symbol names.
    # Use symbol_mapping.csv to map CSV symbols to GDX symbols, then look up dimension here.
}


def get_gdx_dimension(csv_symbol: str, gdx_symbol: str) -> int:
    """Get GDX dimension for a symbol, looking up by GDX symbol name.
    
    Args:
        csv_symbol: CSV symbol name (for error messages)
        gdx_symbol: GDX symbol name to look up in GDX_DIM
    
    Returns:
        GDX dimension (number of dimensions)
    
    Raises:
        KeyError: If gdx_symbol is not found in GDX_DIM
    """
    if gdx_symbol not in GDX_DIM:
        raise KeyError(
            f"GDX dimension not found for '{gdx_symbol}' (CSV symbol: '{csv_symbol}'). "
            f"Add it to GDX_DIM dictionary."
        )
    return GDX_DIM[gdx_symbol]


# CSV layout specification: Defines how each CSV symbol should be converted from GDX format.
# 
# Structure:
# - Each entry specifies a CSV symbol name, output path, and transformation rules
# - gdx_dim: Number of dimensions in the GDX parameter/set (from GDX_DIM)
# - header_cols: Number of GDX dimensions to pivot to CSV column headers
#   * header_cols=0: All dimensions become index columns (e.g., pGenDataInput, pStorageDataInput)
#   * header_cols=1: Last dimension becomes column headers (e.g., years, hours, seasons)
#   * indexColumns count = gdx_dim - header_cols
#
# Examples:
# - pCarbonPrice: GDX dim=1, header_cols=0 → 1 index column (year) + value column
# - pFuelPrice: GDX dim=3, header_cols=1 → 2 index columns (zone, fuel) + years as headers
# - pGenDataInput: GDX dim=2, header_cols=0 → All dimensions as columns (no pivot)
# - pHours: GDX dim=4, header_cols=1 → 3 index columns (season, daytype, ?) + hours as headers
#
# Note: GDX symbol names may differ from CSV symbol names (see symbol_mapping.csv)
CSV_LAYOUT: List[Dict] = [
    # Constraint parameters
    _entry("pCarbonPrice", "constraint/pCarbonPrice.csv", gdx_dim=1, header_cols=0),  # GDX: pCarbonPrice, CSV: year,value
    _entry("pEmissionsCountry", "constraint/pEmissionsCountry.csv", gdx_dim=2, header_cols=1),  # GDX: pEmissionsCountry, CSV: country index, years as headers
    _entry("pEmissionsTotal", "constraint/pEmissionsTotal.csv", gdx_dim=1, header_cols=0),  # GDX: pEmissionsTotal, CSV: year,value
    _entry("pMaxFuellimit", "constraint/pMaxFuellimit.csv", gdx_dim=3, header_cols=1),  # GDX: pMaxFuellimit, CSV: zone,fuel index, years as headers
    
    # H2 parameters
    _entry("pAvailabilityH2", "h2/pAvailabilityH2.csv", gdx_dim=2, header_cols=1),  # GDX: pAvailabilityH2, CSV: gen index, seasons as headers
    _entry("pCapexTrajectoryH2", "h2/pCapexTrajectoryH2.csv", gdx_dim=2, header_cols=1),  # GDX: pCapexTrajectoryH2, CSV: gen index, years as headers
    _entry("pExternalH2", "h2/pExternalH2.csv", gdx_dim=4, header_cols=1),  # GDX: pExternalH2, CSV: ZONE,Season index, years as headers
    _entry("pFuelDataH2", "h2/pFuelDataH2.csv", gdx_dim=1, header_cols=0),  # GDX: pFuelDataH2, CSV: Type of fuel, Hydrogen index (all columns)
    _entry("pH2DataExcel", "h2/pH2DataExcel.csv", gdx_dim=2, header_cols=1),  # GDX: pH2DataExcel, CSV: gen index, attributes as headers
    
    # Load parameters
    _entry("pDemandData", "load/pDemandData.csv", gdx_dim=5, header_cols=1),  # GDX: pDemandData, CSV: zone,q,d,y index, hours as headers
    _entry("pDemandForecast", "load/pDemandForecast.csv", gdx_dim=3, header_cols=1),  # GDX: pDemandForecast, CSV: zone,type index, years as headers
    _entry("pDemandProfile", "load/pDemandProfile.csv", gdx_dim=4, header_cols=1),  # GDX: pDemandProfile, CSV: zone,season,daytype index, hours as headers
    _entry("pEnergyEfficiencyFactor", "load/pEnergyEfficiencyFactor.csv", gdx_dim=2, header_cols=1),  # GDX: pEnergyEfficiencyFactor, CSV: years as headers (1D pivot)
    _entry("sRelevant", "load/sRelevant.csv", gdx_dim=1, typ="set"),  # GDX: sRelevant, CSV: set elements
    
    # General parameters
    _entry("pHours", "pHours.csv", gdx_dim=4, header_cols=1),  # GDX: pHours, CSV: season,daytype index, hours as headers
    _entry("pSettings", "pSettings.csv", gdx_dim=1, header_cols=0),  # GDX: pScalars, CSV: Parameter,Abbreviation,Value (special format)
    
    # Reserve parameters
    _entry("pPlanningReserveMargin", "reserve/pPlanningReserveMargin.csv", gdx_dim=1, header_cols=0),  # GDX: pPlanningReserveMargin, CSV: year,value
    _entry("pSpinningReserveReqCountry", "reserve/pSpinningReserveReqCountry.csv", gdx_dim=2, header_cols=1),  # GDX: pSpinningReserveReqCountry, CSV: country index, years as headers
    _entry("pSpinningReserveReqSystem", "reserve/pSpinningReserveReqSystem.csv", gdx_dim=1, header_cols=0),  # GDX: pSpinningReserveReqSystem, CSV: year,value
    
    # Supply parameters
    _entry("pAvailability", "supply/pAvailabilityCustom.csv", gdx_dim=2, header_cols=1),  # GDX: pAvailability, CSV: gen index, seasons as headers
    _entry("pCSPData", "supply/pCSPData.csv", gdx_dim=3, header_cols=1),  # GDX: pCSPData, CSV: gen,attribute index, years as headers
    _entry("pCapexTrajectories", "supply/pCapexTrajectoriesCustom.csv", gdx_dim=2, header_cols=1),  # GDX: pCapexTrajectory, CSV: gen index, years as headers
    _entry("pFuelPrice", "supply/pFuelPrice.csv", gdx_dim=3, header_cols=1),  # GDX: pFuelPrice, CSV: zone,fuel index, years as headers
    _entry("pGenDataInput", "supply/pGenDataInput.csv", gdx_dim=2, header_cols=0, symbols=["gmap", "pGenDataInput"]),  # GDX: pGenDataExcel, CSV: all dimensions as columns (no pivot)
    _entry("pStorageDataInput", "supply/pStorageDataInput.csv", gdx_dim=3, header_cols=0),  # GDX: pStorDataExcel, CSV: all dimensions as columns (no pivot)
    _entry("pVREProfile", "supply/pVREProfile.csv", gdx_dim=5, header_cols=1),  # GDX: pVREProfile, CSV: zone,tech,season,daytype index, hours as headers
    _entry("pVREgenProfile", "supply/pVREgenProfile.csv", gdx_dim=5, header_cols=1),  # GDX: pVREgenProfile, CSV: zone,tech,season,daytype index, hours as headers
    
    # Trade parameters
    _entry("pExtTransferLimit", "trade/pExtTransferLimit.csv", gdx_dim=5, header_cols=1),  # GDX: pExtTransferLimit, CSV: Internal zone,External zone,Seasons,Import-Export index, years as headers
    _entry("pLossFactorInternal", "trade/pLossFactorInternal.csv", gdx_dim=3, header_cols=1),  # GDX: pLossFactor, CSV: zone1,zone2 index, years as headers
    _entry("pMaxAnnualExternalTradeShare", "trade/pMaxAnnualExternalTradeShare.csv", gdx_dim=2, header_cols=1),  # GDX: pMaxAnnualExternalTradeShare, CSV: y index, zones as headers
    _entry("pMaxPriceImportShare", "trade/pMaxPriceImportShare.csv", gdx_dim=2, header_cols=1),  # GDX: pMaxPriceImportShare, CSV: y index, zones as headers
    _entry("pMinImport", "trade/pMinImport.csv", gdx_dim=2, header_cols=1),  # GDX: pMinImport, CSV: zone1,zone2 index, years as headers
    _entry("pNewTransmission", "trade/pNewTransmission.csv", gdx_dim=3, header_cols=1),  # GDX: pNewTransmission, CSV: From,To index, years as headers
    _entry("pTradePrice", "trade/pTradePrice.csv", gdx_dim=5, header_cols=1),  # GDX: pTradePrice, CSV: zext,q,daytype,y index, hours as headers
    _entry("pTransferLimit", "trade/pTransferLimit.csv", gdx_dim=4, header_cols=1),  # GDX: pTranspferLimit, CSV: From,To,q index, years as headers
    
    # Sets
    _entry("zext", "trade/zext.csv", gdx_dim=1, typ="set"),  # GDX: zext, CSV: set elements
    _entry("y", "y.csv", gdx_dim=1, typ="set"),  # GDX: y, CSV: set elements
    _entry("zcmap", "zcmap.csv", gdx_dim=2, typ="set"),  # GDX: zcmapExcel, CSV: set elements (2D)
]

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

# Hard-coded column layouts mirroring epm/input/data_test (no runtime discovery)
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_value_column(df_gdx_records: pd.DataFrame) -> Optional[str]:
    """Find the value column in GDX records DataFrame.
    
    GDX parameter records typically have a 'value' or 'Value' column
    containing the numeric values, plus dimension columns.
    
    Args:
        df_gdx_records: DataFrame from GDX parameter records
    
    Returns:
        Name of the value column, or None if not found
    """
    for candidate in ("value", "Value"):
        if candidate in df_gdx_records.columns:
            return candidate
    return None


def format_set(df_gdx_set: pd.DataFrame) -> pd.DataFrame:
    """Format GDX set records into CSV format (deduplicate and sort).
    
    Sets in GDX are simple lists of elements. This function ensures
    the CSV output has unique, sorted set elements.
    
    Args:
        df_gdx_set: DataFrame from GDX set records
    
    Returns:
        DataFrame with unique, sorted set elements
    """
    if not list(df_gdx_set.columns):
        return df_gdx_set.copy().reset_index(drop=True)
    return df_gdx_set.drop_duplicates().sort_values(list(df_gdx_set.columns)).reset_index(drop=True)


def format_header_table(df_gdx_records: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Transform GDX parameter records into CSV format by pivoting header dimensions.
    
    This function takes GDX records (which have all dimensions as columns plus a value column)
    and pivots the specified header dimensions to become CSV column headers.
    
    Args:
        df_gdx_records: DataFrame from GDX records (all dimensions + value column)
        spec: Layout specification dict with indexColumns, header, etc.
    
    Returns:
        DataFrame in CSV format with index columns + pivoted header columns
    """
    value_col = find_value_column(df_gdx_records)
    # All columns except the value column are dimension columns from GDX
    gdx_dimension_cols = [col for col in df_gdx_records.columns if col != value_col]
    # First N dimension columns become CSV index columns
    csv_index_cols = gdx_dimension_cols[: len(spec["indexColumns"])]
    # Next M dimension columns become CSV header columns (pivoted)
    gdx_header_dimensions = gdx_dimension_cols[len(csv_index_cols) : len(csv_index_cols) + len(spec["header"])]
    base_cols = csv_index_cols + gdx_header_dimensions + ([value_col] if value_col else [])
    data = df_gdx_records[base_cols] if base_cols else df_gdx_records.copy()
    if gdx_header_dimensions and value_col:
        # Pivot: GDX header dimensions → CSV column headers
        pivot = data.pivot_table(
            index=csv_index_cols,
            columns=gdx_header_dimensions,
            values=value_col,
            aggfunc="first",
            observed=False,
        )
        pivot.columns = [col if isinstance(col, str) else "_".join(map(str, col)) for col in pivot.columns]
        return pivot.reset_index().reset_index(drop=True)
    if value_col:
        return data.rename(columns={value_col: "value"}).reset_index(drop=True)
    return data.reset_index(drop=True)


def format_value_table(df_gdx_records: pd.DataFrame, spec: dict, csv_symbol: str, gdx_symbol: str) -> pd.DataFrame:
    """Transform GDX parameter records to CSV format with explicit value columns.
    
    Args:
        df_gdx_records: DataFrame from GDX records
        spec: Layout specification dict
        csv_symbol: CSV symbol name (for error messages)
        gdx_symbol: GDX symbol name (for error messages)
    
    Returns:
        DataFrame with CSV index columns + value column
    """
    value_col = find_value_column(df_gdx_records)
    if value_col is None:
        raise KeyError(f"No value column found for GDX symbol '{gdx_symbol}' (CSV: '{csv_symbol}')")
    # All columns except value are GDX dimension columns
    gdx_dimension_cols = [col for col in df_gdx_records.columns if col != value_col]
    # First N dimensions become CSV index columns
    csv_index_cols = gdx_dimension_cols[: len(spec["indexColumns"])]
    columns = csv_index_cols + [value_col]
    return df_gdx_records[columns].reset_index(drop=True)


def build_frame(container: gt.Container, gdx_symbol: str, csv_symbol: str, spec: dict) -> Optional[pd.DataFrame]:
    """Fetch GDX symbol and transform it to CSV format according to spec.
    
    This is the main transformation function that converts GDX data to CSV format.
    
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
    if spec["type"] == "set":
        # Sets: just format and deduplicate
        return format_set(df_gdx_data)
    if spec["valueColumns"]:
        # Explicit value columns specified
        return format_value_table(df_gdx_data, spec, csv_symbol, gdx_symbol)
    # Default: pivot header dimensions to CSV column headers
    return format_header_table(df_gdx_data, spec)


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


def load_symbol_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load CSV symbol → GDX symbol mapping from CSV file.
    
    The mapping file defines how CSV symbol names map to GDX symbol names.
    Many symbols have the same name in both formats, but some differ
    (e.g., pGenDataInput → pGenDataExcel, pSettings → pScalars).
    
    Args:
        mapping_path: Path to symbol_mapping.csv file
    
    Returns:
        Dictionary mapping csv_symbol → gdx_symbol
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping table: {mapping_path}. Populate it before continuing.")

    symbol_mapping_df = pd.read_csv(mapping_path)
    if {"csv_symbol", "gdx_symbol"} - set(symbol_mapping_df.columns):
        raise ValueError("symbol_mapping.csv must contain 'csv_symbol' and 'gdx_symbol' columns")

    symbol_mapping_df = symbol_mapping_df.fillna("").drop_duplicates(subset="csv_symbol", keep="last")
    # Return mapping: csv_symbol -> gdx_symbol (use csv_symbol as fallback if gdx_symbol is empty)
    return {row.csv_symbol: (row.gdx_symbol or row.csv_symbol) for row in symbol_mapping_df.itertuples()}


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
    # Load mapping: csv_symbol -> gdx_symbol
    csv_to_gdx_mapping = load_symbol_mapping(mapping_path)
    missing_mapping_rows = [
        entry["primary_symbol"] for entry in CSV_LAYOUT if entry["primary_symbol"] not in csv_to_gdx_mapping
    ]

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
    for entry in CSV_LAYOUT:
        csv_symbol = entry["primary_symbol"]
        # Map CSV symbol name to GDX symbol name
        gdx_symbol = csv_to_gdx_mapping.get(csv_symbol, csv_symbol)

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
            # Track aliases too
            for alias in entry["symbols"]:
                used_gdx_symbols.add(csv_to_gdx_mapping.get(alias, alias))

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
        "missing_mapping_rows": missing_mapping_rows,
        "missing_in_gdx": missing_in_gdx,
        "optional_stubbed": optional_stubbed,
        "empty_in_gdx": empty_in_gdx,
    }


def print_report(results: Dict[str, List], output_root: Path) -> None:
    """Emit a concise report mirroring the notebook prints."""
    missing_mapping_rows = results["missing_mapping_rows"]
    missing_in_gdx = results["missing_in_gdx"]
    optional_stubbed = results["optional_stubbed"]
    empty_in_gdx = results["empty_in_gdx"]
    extras_written = results["extras_written"]
    skipped = results["skipped"]

    if missing_mapping_rows:
        print("Mapping rows missing for:", ", ".join(sorted(missing_mapping_rows)))
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
