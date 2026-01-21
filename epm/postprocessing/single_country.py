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
    """Get list of zones belonging to a country."""
    return zcmap_df[zcmap_df['c'] == country]['z'].tolist()


def identify_neighbors(transfer_limit_df: pd.DataFrame, country_zones: list) -> list:
    """
    Identify neighboring zones from pTransferLimit.

    A neighbor is a zone that has a transmission link to a country zone
    but is not itself a country zone.
    """
    neighbors = set()

    for _, row in transfer_limit_df.iterrows():
        from_zone = row['From']
        to_zone = row['To']

        # If one side is in country and the other is not, it's a neighbor
        if from_zone in country_zones and to_zone not in country_zones:
            neighbors.add(to_zone)
        elif to_zone in country_zones and from_zone not in country_zones:
            neighbors.add(from_zone)

    return list(neighbors)


def generate_single_country_inputs(
    folder_input: Path,
    folder_output: Path,
    country: str
) -> None:
    """
    Generate single-country input files from regional run results.

    Args:
        folder_input: Path to the regional input folder
        folder_output: Path to the regional output folder (with pHourlyPrice)
        country: Focus country code (e.g., 'DRC')
    """
    country_lower = country.lower()
    single_country_dir = folder_input / f"single_country_{country_lower}"
    single_country_dir.mkdir(exist_ok=True)

    # Read source files
    zcmap_df = pd.read_csv(folder_input / "zcmap.csv")
    transfer_limit_df = pd.read_csv(folder_input / "trade" / "pTransferLimit.csv")

    # Identify country zones and neighbors
    country_zones = identify_country_zones(zcmap_df, country)
    if not country_zones:
        raise ValueError(f"No zones found for country '{country}' in zcmap.csv")

    neighbors = identify_neighbors(transfer_limit_df, country_zones)
    print(f"Country zones for {country}: {country_zones}")
    print(f"Neighboring zones: {neighbors}")

    # 1. Generate zcmap_<country>.csv - only focus country zones
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
    internal_transfer = transfer_limit_df[
        (transfer_limit_df['From'].isin(country_zones)) &
        (transfer_limit_df['To'].isin(country_zones))
    ]
    internal_transfer.to_csv(
        single_country_dir / f"pTransferLimit_{country_lower}.csv",
        index=False
    )

    # 4. Generate pExtTransferLimit_<country>.csv - border capacities
    # Extract cross-border transmission and convert to external zone format
    cross_border = transfer_limit_df[
        ((transfer_limit_df['From'].isin(country_zones)) & (~transfer_limit_df['To'].isin(country_zones))) |
        ((~transfer_limit_df['From'].isin(country_zones)) & (transfer_limit_df['To'].isin(country_zones)))
    ]

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
        single_country_dir / f"pExtTransferLimit_{country_lower}.csv",
        index=False
    )

    # 5. Generate pTradePrice_<country>.csv from pHourlyPrice
    # TODO: Read pHourlyPrice from output and format as pTradePrice
    hourly_price_path = folder_output / "pHourlyPrice.csv"
    if hourly_price_path.exists():
        _generate_trade_price(hourly_price_path, neighbors, single_country_dir, country_lower)
    else:
        print(f"Warning: {hourly_price_path} not found. pTradePrice not generated.")

    # 6. Generate config_<country>.csv
    _generate_config(folder_input, single_country_dir, country_lower)

    print(f"Single-country inputs generated in: {single_country_dir}")
    print(f"Config file: {folder_input / f'config_{country_lower}.csv'}")


def _generate_trade_price(
    hourly_price_path: Path,
    neighbors: list,
    output_dir: Path,
    country_lower: str
) -> None:
    """Convert pHourlyPrice to pTradePrice format for neighbor zones."""
    # pHourlyPrice format: z,q,d,t,y,value (or similar - need to check actual format)
    # pTradePrice format: zext,q,d,y,t1,t2,...,t24

    hourly_price_df = pd.read_csv(hourly_price_path)

    # Filter to only neighbor zones
    neighbor_prices = hourly_price_df[hourly_price_df['z'].isin(neighbors)]

    if neighbor_prices.empty:
        print(f"Warning: No hourly prices found for neighbors {neighbors}")
        return

    # Pivot to wide format (t1-t24 columns)
    # TODO: Implement proper pivot based on actual pHourlyPrice structure

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
