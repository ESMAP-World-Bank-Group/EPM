"""
Generate single-country input files from a regional model run.

This module creates the necessary input files to run a single-country model
using border prices extracted from a full regional run. Neighboring countries
become external zones (zext) with fixed prices from pHourlyPrice.

Usage:
    After a regional run with --focus_country <country>, this generates:
    - config_<country>.csv: Modified config pointing to single-country files
    - single_country_<country>/: Folder with modified trade files
        - zcmap_<country>.csv: Only zones belonging to focus country
        - zext_<country>.csv: Neighbors as external zones
        - pTradePrice_<country>.csv: Border prices from pHourlyPrice
        - pExtTransferLimit_<country>.csv: Border capacities
        - pTransferLimit_<country>.csv: Internal transmission only
"""

import pandas as pd
from pathlib import Path


def identify_country_zones(zcmap_df: pd.DataFrame, country: str) -> list:
    """Get list of zones belonging to a country (a country can have multiple zones)."""
    return zcmap_df[zcmap_df['c'] == country]['z'].tolist()


def identify_neighbors(folder_input: Path, country_zones: list) -> list:
    """
    Identify neighboring zones from both pTransferLimit AND pNewTransmission.

    A neighbor is a zone that has a transmission link (existing or planned)
    to a country zone but is not itself a country zone.

    Args:
        folder_input: Path to input folder containing trade/ subfolder
        country_zones: List of zones belonging to the focus country

    Returns:
        List of neighboring zone names
    """
    neighbors = set()

    # Helper function to extract neighbors from a dataframe
    def extract_neighbors_from_df(df: pd.DataFrame, from_col: str, to_col: str):
        for _, row in df.iterrows():
            from_zone = row[from_col]
            to_zone = row[to_col]
            # If one side is in country and the other is not, it's a neighbor
            if from_zone in country_zones and to_zone not in country_zones:
                neighbors.add(to_zone)
            elif to_zone in country_zones and from_zone not in country_zones:
                neighbors.add(from_zone)

    # Check pTransferLimit (existing transmission)
    transfer_path = folder_input / "trade" / "pTransferLimit.csv"
    if transfer_path.exists():
        transfer_df = pd.read_csv(transfer_path)
        extract_neighbors_from_df(transfer_df, 'From', 'To')

    # Check pNewTransmission (candidate/committed new lines)
    new_trans_path = folder_input / "trade" / "pNewTransmission.csv"
    if new_trans_path.exists():
        new_trans_df = pd.read_csv(new_trans_path)
        extract_neighbors_from_df(new_trans_df, 'From', 'To')

    return list(neighbors)


def generate_single_country_inputs(
    folder_input: Path,
    folder_output: Path,
    country: str,
    hourly_price_df: pd.DataFrame = None,
    scenario_reference: str = 'baseline'
) -> None:
    """
    Generate single-country input files from regional run results.

    Args:
        folder_input: Path to the regional input folder
        folder_output: Path to the regional output folder (with pHourlyPrice)
        country: Focus country code (e.g., 'DRC')
        hourly_price_df: Optional DataFrame with pHourlyPrice data (if already loaded)
        scenario_reference: Reference scenario to use for prices (default: 'baseline')
    """
    country_lower = country.lower()
    single_country_dir = folder_input / f"single_country_{country_lower}"
    single_country_dir.mkdir(exist_ok=True)

    # Read source files
    zcmap_df = pd.read_csv(folder_input / "zcmap.csv")
    transfer_limit_path = folder_input / "trade" / "pTransferLimit.csv"
    transfer_limit_df = pd.read_csv(transfer_limit_path) if transfer_limit_path.exists() else pd.DataFrame()

    # Identify country zones and neighbors
    country_zones = identify_country_zones(zcmap_df, country)
    if not country_zones:
        raise ValueError(f"No zones found for country '{country}' in zcmap.csv")

    neighbors = identify_neighbors(folder_input, country_zones)
    print(f"Country zones for {country}: {country_zones}")
    print(f"Neighboring zones: {neighbors}")

    # 1. Generate zcmap_<country>.csv - all zones belonging to focus country
    zcmap_country = zcmap_df[zcmap_df['c'] == country]
    zcmap_country.to_csv(
        single_country_dir / f"zcmap_{country_lower}.csv",
        index=False
    )

    # 2. Generate zext_<country>.csv - neighbors + existing external zones
    existing_zext_path = folder_input / "trade" / "zext.csv"
    if existing_zext_path.exists():
        existing_zext = pd.read_csv(existing_zext_path)['z'].tolist()
    else:
        existing_zext = []

    all_zext = list(set(neighbors + existing_zext))
    zext_df = pd.DataFrame({'z': all_zext})
    zext_df.to_csv(
        single_country_dir / f"zext_{country_lower}.csv",
        index=False
    )

    # 3. Generate pTransferLimit_<country>.csv - internal transmission only
    if not transfer_limit_df.empty:
        internal_transfer = transfer_limit_df[
            (transfer_limit_df['From'].isin(country_zones)) &
            (transfer_limit_df['To'].isin(country_zones))
        ]
        internal_transfer.to_csv(
            single_country_dir / f"pTransferLimit_{country_lower}.csv",
            index=False
        )

    # 4. Generate pExtTransferLimit_<country>.csv - border capacities
    _generate_ext_transfer_limit(
        folder_input, country_zones, single_country_dir, country_lower
    )

    # 5. Generate pTradePrice_<country>.csv from pHourlyPrice
    _generate_trade_price(folder_output, neighbors, single_country_dir, country_lower, hourly_price_df, scenario_reference)

    # 6. Generate config_<country>.csv
    _generate_config(folder_input, single_country_dir, country_lower)

    print(f"Single-country inputs generated in: {single_country_dir}")
    print(f"Config file: {folder_input / f'config_{country_lower}.csv'}")


def _generate_ext_transfer_limit(
    folder_input: Path,
    country_zones: list,
    output_dir: Path,
    country_lower: str
) -> None:
    """Generate pExtTransferLimit from cross-border pTransferLimit entries."""
    transfer_limit_path = folder_input / "trade" / "pTransferLimit.csv"
    if not transfer_limit_path.exists():
        print("Warning: pTransferLimit.csv not found. pExtTransferLimit not generated.")
        return

    transfer_limit_df = pd.read_csv(transfer_limit_path)

    # Extract cross-border transmission
    cross_border = transfer_limit_df[
        ((transfer_limit_df['From'].isin(country_zones)) & (~transfer_limit_df['To'].isin(country_zones))) |
        ((~transfer_limit_df['From'].isin(country_zones)) & (transfer_limit_df['To'].isin(country_zones)))
    ]

    if cross_border.empty:
        print("Warning: No cross-border transmission found.")
        # Create empty file with headers
        pd.DataFrame(columns=['Internal zone', 'External zone', 'Seasons', 'Import-Export']).to_csv(
            output_dir / f"pExtTransferLimit_{country_lower}.csv",
            index=False
        )
        return

    ext_transfer_rows = []
    year_cols = [c for c in cross_border.columns if c not in ['From', 'To', 'q']]

    for _, row in cross_border.iterrows():
        from_zone = row['From']
        to_zone = row['To']
        season = row['q']

        if from_zone in country_zones:
            # Export from country zone to neighbor
            internal_zone = from_zone
            external_zone = to_zone
            direction = 'Export'
        else:
            # Import to country zone from neighbor
            internal_zone = to_zone
            external_zone = from_zone
            direction = 'Import'

        ext_row = {
            'Internal zone': internal_zone,
            'External zone': external_zone,
            'Seasons': season,
            'Import-Export': direction
        }
        for yr in year_cols:
            ext_row[yr] = row[yr]

        ext_transfer_rows.append(ext_row)

    ext_transfer_df = pd.DataFrame(ext_transfer_rows)
    ext_transfer_df.to_csv(
        output_dir / f"pExtTransferLimit_{country_lower}.csv",
        index=False
    )


def _generate_trade_price(
    folder_output: Path,
    neighbors: list,
    output_dir: Path,
    country_lower: str,
    hourly_price_df: pd.DataFrame = None,
    scenario_reference: str = 'baseline'
) -> None:
    """
    Convert pHourlyPrice to pTradePrice format for neighbor zones.

    pHourlyPrice format (from GDX): zone,season,day,hour,year,value (long format)
    pTradePrice format: zext,q,d,y,t1,t2,...,t24 (wide format)

    Args:
        folder_output: Path to output folder (used if hourly_price_df not provided)
        neighbors: List of neighbor zone names
        output_dir: Directory to write the output file
        country_lower: Lowercase country name for file naming
        hourly_price_df: Optional DataFrame with pHourlyPrice (if already loaded from GDX)
        scenario_reference: Reference scenario to use for prices (default: 'baseline')
    """
    # Use provided DataFrame or try to read from CSV
    if hourly_price_df is None:
        possible_paths = [
            folder_output / "pHourlyPrice.csv",
            folder_output / "csv" / "pHourlyPrice.csv",
            folder_output / "output_csv" / "pHourlyPrice.csv",
        ]

        hourly_price_path = None
        for path in possible_paths:
            if path.exists():
                hourly_price_path = path
                break

        if hourly_price_path is None:
            print(f"Warning: pHourlyPrice.csv not found in {folder_output}. pTradePrice not generated.")
            return

        hourly_price_df = pd.read_csv(hourly_price_path)

    # Check the structure of the file
    print(f"pHourlyPrice columns: {hourly_price_df.columns.tolist()}")

    # Determine zone column name (could be 'z' or 'zone')
    zone_col = 'zone' if 'zone' in hourly_price_df.columns else 'z'
    if zone_col not in hourly_price_df.columns:
        print(f"Warning: zone column not found in pHourlyPrice. Columns: {hourly_price_df.columns.tolist()}")
        return

    # Determine other column names (could vary between 'q'/'season', 'd'/'day', 'y'/'year')
    season_col = 'season' if 'season' in hourly_price_df.columns else 'q'
    day_col = 'day' if 'day' in hourly_price_df.columns else 'd'
    year_col = 'year' if 'year' in hourly_price_df.columns else 'y'

    # Filter to scenario_reference (or first scenario if not found)
    if 'scenario' in hourly_price_df.columns:
        scenarios = hourly_price_df['scenario'].unique()
        if scenario_reference in scenarios:
            hourly_price_df = hourly_price_df[hourly_price_df['scenario'] == scenario_reference]
            print(f"Filtered to '{scenario_reference}' scenario")
        else:
            first_scenario = scenarios[0]
            hourly_price_df = hourly_price_df[hourly_price_df['scenario'] == first_scenario]
            print(f"'{scenario_reference}' not found, using first scenario: '{first_scenario}'")

    # Filter to only neighbor zones
    neighbor_prices = hourly_price_df[hourly_price_df[zone_col].isin(neighbors)].copy()

    if neighbor_prices.empty:
        print(f"Warning: No hourly prices found for neighbors {neighbors}")
        return

    # Pivot from long format to wide format
    # Input columns: zone, season, day, t, year, value
    # Output columns: zext, q, d, y, t1, t2, ..., t24
    try:
        # Pivot the data
        pivoted = neighbor_prices.pivot_table(
            index=[zone_col, season_col, day_col, year_col],
            columns='t',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Rename columns to match pTradePrice format
        col_rename = {
            zone_col: 'zext',
            season_col: 'q',
            day_col: 'd',
            year_col: 'y'
        }
        # Handle t columns (already t1, t2, etc.)
        for col in pivoted.columns:
            if col not in [zone_col, season_col, day_col, year_col, 'zext', 'q', 'd', 'y']:
                if str(col).startswith('t'):
                    col_rename[col] = col
                else:
                    col_rename[col] = f't{col}'

        pivoted = pivoted.rename(columns=col_rename)

        # Reorder columns to match pTradePrice format
        base_cols = ['zext', 'q', 'd', 'y']
        t_cols = sorted(
            [c for c in pivoted.columns if str(c).startswith('t')],
            key=lambda x: int(str(x)[1:])
        )
        pivoted = pivoted[base_cols + t_cols]

        pivoted.to_csv(
            output_dir / f"pTradePrice_{country_lower}.csv",
            index=False
        )
        print(f"Generated pTradePrice with {len(pivoted)} rows for neighbors: {neighbors}")

    except Exception as e:
        print(f"Warning: Could not pivot pHourlyPrice to pTradePrice format: {e}")
        print("Saving raw filtered data instead.")
        neighbor_prices.to_csv(
            output_dir / f"pTradePrice_{country_lower}.csv",
            index=False
        )


def _generate_config(
    folder_input: Path,
    single_country_dir: Path,
    country_lower: str
) -> None:
    """Generate modified config file pointing to single-country files."""
    config_path = folder_input / "config.csv"
    config_df = pd.read_csv(config_path)

    # Relative path from folder_input to single_country_dir
    rel_path = f"single_country_{country_lower}"

    # Update file paths for trade-related parameters
    path_updates = {
        'zcmap': f'{rel_path}/zcmap_{country_lower}.csv',
        'zext': f'{rel_path}/zext_{country_lower}.csv',
        'pTransferLimit': f'{rel_path}/pTransferLimit_{country_lower}.csv',
        'pExtTransferLimit': f'{rel_path}/pExtTransferLimit_{country_lower}.csv',
        'pTradePrice': f'{rel_path}/pTradePrice_{country_lower}.csv',
    }

    for param, new_path in path_updates.items():
        mask = config_df['paramNames'] == param
        if mask.any():
            config_df.loc[mask, 'file'] = new_path

    config_df.to_csv(
        folder_input / f"config_{country_lower}.csv",
        index=False
    )


if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) < 4:
        print("Usage: python single_country.py <folder_input> <folder_output> <country>")
        print("Example: python single_country.py epm/input/data_test epm/output/data_test DRC")
        sys.exit(1)

    folder_input = Path(sys.argv[1])
    folder_output = Path(sys.argv[2])
    country = sys.argv[3]

    generate_single_country_inputs(folder_input, folder_output, country)
