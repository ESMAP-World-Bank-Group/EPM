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

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _filter_to_scenario(df: pd.DataFrame, scenario_reference: str) -> pd.DataFrame:
    """Filter DataFrame to reference scenario if scenario column exists."""
    if df is None or df.empty:
        return df
    if 'scenario' not in df.columns:
        return df

    scenarios = df['scenario'].unique()
    if scenario_reference in scenarios:
        return df[df['scenario'] == scenario_reference].copy()
    elif len(scenarios) > 0:
        return df[df['scenario'] == scenarios[0]].copy()
    return df


def identify_country_zones(zcmap_df: pd.DataFrame, country: str) -> list:
    """Get list of zones belonging to a country (a country can have multiple zones)."""
    return zcmap_df[zcmap_df['country'] == country]['zone'].tolist()


def _convert_zone_country_to_zcmap(zone_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pZoneCountry GDX data to zcmap format.

    pZoneCountry from GDX has columns: z, c, value (where value=1 indicates mapping)
    zcmap format needs: z, c
    """
    df = zone_country_df.copy()

    # Filter to rows where value=1 (the mapping exists)
    if 'value' in df.columns:
        df = df[df['value'] == 1]

    # Return only z and c columns
    return df[['zone', 'country']].drop_duplicates().reset_index(drop=True)


def identify_neighbors_from_gdx(
    transmission_capacity_df: pd.DataFrame,
    country_zones: list,
    scenario_reference: str = 'baseline'
) -> list:
    """
    Identify neighboring zones from pTransmissionCapacity (GDX results).

    A neighbor is a zone that has a transmission link to a country zone
    but is not itself a country zone. Using pTransmissionCapacity ensures
    we capture both existing and new (committed/candidate) transmission.

    Args:
        transmission_capacity_df: DataFrame with pTransmissionCapacity from GDX
        country_zones: List of zones belonging to the focus country
        scenario_reference: Scenario to filter on

    Returns:
        List of neighboring zone names
    """
    if transmission_capacity_df is None or transmission_capacity_df.empty:
        return []

    df = _filter_to_scenario(transmission_capacity_df, scenario_reference)

    neighbors = set()
    for _, row in df.iterrows():
        from_zone = row['zone']
        to_zone = row['z2']
        # If one side is in country and the other is not, it's a neighbor
        if from_zone in country_zones and to_zone not in country_zones:
            neighbors.add(to_zone)
        elif to_zone in country_zones and from_zone not in country_zones:
            neighbors.add(from_zone)

    return list(neighbors)


def generate_single_country_inputs(
    folder_input: Path,
    folder_output: Path,
    country: str,
    hourly_price_df: pd.DataFrame = None,
    transmission_capacity_df: pd.DataFrame = None,
    ext_transfer_limit_df: pd.DataFrame = None,
    trade_price_df: pd.DataFrame = None,
    zone_country_df: pd.DataFrame = None,
    scenario_reference: str = 'baseline'
) -> None:
    """
    Generate single-country input files from regional run results.

    Args:
        folder_input: Path to the regional input folder
        folder_output: Path to the regional output folder (with pHourlyPrice)
        country: Focus country code (e.g., 'DRC')
        hourly_price_df: DataFrame with pHourlyPrice data from GDX (for neighbor zones)
        transmission_capacity_df: DataFrame with pTransmissionCapacity from GDX
        ext_transfer_limit_df: DataFrame with pExtTransferLimit from GDX
        trade_price_df: DataFrame with pTradePrice from GDX (for existing external zones)
        zone_country_df: DataFrame with pZoneCountry from GDX (zone to country mapping)
        scenario_reference: Reference scenario to use for prices (default: 'baseline')
    """
    country_lower = country.lower()
    single_country_dir = folder_input / f"single_country_{country_lower}"
    single_country_dir.mkdir(exist_ok=True)

    # Get zone-country mapping from GDX (source of truth)
    if zone_country_df is None or zone_country_df.empty:
        raise ValueError("pZoneCountry data from GDX is required")

    zcmap_df = _convert_zone_country_to_zcmap(zone_country_df)

    # Identify country zones
    country_zones = identify_country_zones(zcmap_df, country)
    if not country_zones:
        raise ValueError(f"No zones found for country '{country}' in zcmap.csv")

    # Identify neighbors from GDX transmission capacity data
    neighbors = identify_neighbors_from_gdx(
        transmission_capacity_df, country_zones, scenario_reference
    )
    logger.info(f"Country zones for {country}: {country_zones}")
    logger.info(f"Neighboring zones (from pTransmissionCapacity): {neighbors}")

    # 1. Generate zcmap_<country>.csv - all zones belonging to focus country
    zcmap_country = zcmap_df[zcmap_df['country'] == country].copy()
    # Rename to output format (short names for GAMS input)
    zcmap_country = zcmap_country.rename(columns={'zone': 'z', 'country': 'c'})
    zcmap_country.to_csv(
        single_country_dir / f"zcmap_{country_lower}.csv",
        index=False
    )

    # 2. Generate zext_<country>.csv - neighbors + existing external zones
    existing_zext_path = folder_input / "trade" / "zext.csv"
    if existing_zext_path.exists():
        existing_zext_df = pd.read_csv(existing_zext_path)
        # Handle both 'z' (legacy) and 'zext' (correct) column names
        col = 'zext' if 'zext' in existing_zext_df.columns else 'z'
        existing_zext = existing_zext_df[col].tolist()
    else:
        existing_zext = []

    all_zext = list(set(neighbors + existing_zext))
    zext_df = pd.DataFrame({'zext': all_zext})
    zext_df.to_csv(
        single_country_dir / f"zext_{country_lower}.csv",
        index=False
    )

    # 3. Generate pTransferLimit_<country>.csv - internal transmission only
    _generate_internal_transfer_limit(
        transmission_capacity_df, country_zones, single_country_dir, country_lower, scenario_reference
    )

    # 4. Generate pExtTransferLimit_<country>.csv - border capacities
    _generate_ext_transfer_limit_from_gdx(
        transmission_capacity_df,
        ext_transfer_limit_df,
        hourly_price_df,
        country_zones,
        single_country_dir,
        country_lower,
        scenario_reference
    )

    # 5. Generate pTradePrice_<country>.csv
    # - Use pHourlyPrice for neighbors (zones that were internal in regional model)
    # - Use pTradePrice for existing external zones (like SAPP)
    _generate_trade_price(
        folder_output, neighbors, existing_zext, single_country_dir, country_lower,
        hourly_price_df, trade_price_df, scenario_reference
    )

    # 6. Generate config_<country>.csv
    _generate_config(folder_input, single_country_dir, country_lower)

    logger.info(f"Single-country inputs generated in: {single_country_dir}")
    logger.info(f"Config file: {folder_input / f'config_{country_lower}.csv'}")


def _generate_internal_transfer_limit(
    transmission_capacity_df: pd.DataFrame,
    country_zones: list,
    output_dir: Path,
    country_lower: str,
    scenario_reference: str = 'baseline'
) -> None:
    """Generate pTransferLimit for internal transmission only (within country zones)."""
    if transmission_capacity_df is None or transmission_capacity_df.empty:
        pd.DataFrame(columns=['z', 'z2', 'q']).to_csv(
            output_dir / f"pTransferLimit_{country_lower}.csv",
            index=False
        )
        return

    df = _filter_to_scenario(transmission_capacity_df, scenario_reference)

    # Filter to internal links only (both zones in country)
    internal = df[
        (df['zone'].isin(country_zones)) &
        (df['z2'].isin(country_zones))
    ].copy()

    if internal.empty:
        pd.DataFrame(columns=['z', 'z2', 'q']).to_csv(
            output_dir / f"pTransferLimit_{country_lower}.csv",
            index=False
        )
        return

    # pTransmissionCapacity is (zone, z2, year) - need to pivot to wide format with year columns
    pivoted = internal.pivot_table(
        index=['zone', 'z2'],
        columns='year',
        values='value',
        aggfunc='first',
        observed=True
    ).reset_index()

    # Add season column - use 'q1' as placeholder (capacity is constant across seasons)
    pivoted['q'] = 'q1'

    # Rename to output format (short names for GAMS input)
    pivoted = pivoted.rename(columns={'zone': 'z'})

    # Reorder columns
    year_cols = [c for c in pivoted.columns if c not in ['z', 'z2', 'q']]
    pivoted = pivoted[['z', 'z2', 'q'] + year_cols]

    pivoted.to_csv(
        output_dir / f"pTransferLimit_{country_lower}.csv",
        index=False
    )


def _generate_ext_transfer_limit_from_gdx(
    transmission_capacity_df: pd.DataFrame,
    ext_transfer_limit_df: pd.DataFrame,
    hourly_price_df: pd.DataFrame,
    country_zones: list,
    output_dir: Path,
    country_lower: str,
    scenario_reference: str = 'baseline'
) -> None:
    """
    Generate pExtTransferLimit from GDX data.

    Combines two sources:
    1. pTransmissionCapacity - for cross-border internal transmission (z to z2)
    2. pExtTransferLimit - for existing external zone connections (z to zext)
    """
    ext_transfer_rows = []

    # Extract seasons from pHourlyPrice
    if hourly_price_df is None or hourly_price_df.empty or 'season' not in hourly_price_df.columns:
        raise ValueError("pHourlyPrice data with 'season' column is required to extract seasons")
    seasons = sorted(hourly_price_df['season'].unique().tolist())

    # 1. Process cross-border internal transmission from pTransmissionCapacity
    if transmission_capacity_df is not None and not transmission_capacity_df.empty:
        df = _filter_to_scenario(transmission_capacity_df, scenario_reference)

        # Extract cross-border links (one zone in country, one not)
        cross_border = df[
            ((df['zone'].isin(country_zones)) & (~df['z2'].isin(country_zones))) |
            ((~df['zone'].isin(country_zones)) & (df['z2'].isin(country_zones)))
        ]

        # Pivot to get year columns
        if not cross_border.empty:
            pivoted = cross_border.pivot_table(
                index=['zone', 'z2'],
                columns='year',
                values='value',
                aggfunc='first',
                observed=True
            ).reset_index()

            year_cols = [c for c in pivoted.columns if c not in ['zone', 'z2']]

            for _, row in pivoted.iterrows():
                from_zone = row['zone']
                to_zone = row['z2']

                if from_zone in country_zones:
                    # Export: from country zone to neighbor
                    internal_zone = from_zone
                    external_zone = to_zone
                    direction = 'Export'
                else:
                    # Import: from neighbor to country zone
                    internal_zone = to_zone
                    external_zone = from_zone
                    direction = 'Import'

                # pTransmissionCapacity is yearly - replicate for all seasons
                for season in seasons:
                    ext_row = {
                        'z': internal_zone,
                        'zext': external_zone,
                        'q': season,
                        'type': direction
                    }
                    for yr in year_cols:
                        ext_row[yr] = row[yr]

                    ext_transfer_rows.append(ext_row)

    # 2. Process existing external zone connections from pExtTransferLimit
    if ext_transfer_limit_df is not None and not ext_transfer_limit_df.empty:
        df = _filter_to_scenario(ext_transfer_limit_df, scenario_reference)

        # Find the direction column (could be 'attribute', 'uni_0', etc.)
        direction_col = None
        for col in ['attribute', 'uni_0', 'Import-Export', 'direction', 'type']:
            if col in df.columns:
                direction_col = col
                break

        # Filter to links connected to country zones
        country_ext = df[df['zone'].isin(country_zones)]

        if not country_ext.empty:
            # GDX data is in long format - need to pivot to wide format with year columns
            # Group by zone, zext, season, direction and pivot on year
            pivot_cols = ['zone', 'zext', 'season']
            if direction_col:
                pivot_cols.append(direction_col)

            pivoted = country_ext.pivot_table(
                index=pivot_cols,
                columns='year',
                values='value',
                aggfunc='first',
                observed=True
            ).reset_index()

            year_cols = [c for c in pivoted.columns if c not in pivot_cols]

            for _, row in pivoted.iterrows():
                ext_row = {
                    'z': row['zone'],
                    'zext': row['zext'],
                    'q': row['season'],
                    'type': row[direction_col] if direction_col else 'Import'
                }
                for yr in year_cols:
                    ext_row[yr] = row[yr]

                ext_transfer_rows.append(ext_row)

    # Write output
    if not ext_transfer_rows:
        logger.warning("No cross-border transmission found for pExtTransferLimit.")
        pd.DataFrame(columns=['z', 'zext', 'q', 'type']).to_csv(
            output_dir / f"pExtTransferLimit_{country_lower}.csv",
            index=False
        )
        return

    ext_transfer_df = pd.DataFrame(ext_transfer_rows)
    ext_transfer_df.to_csv(
        output_dir / f"pExtTransferLimit_{country_lower}.csv",
        index=False
    )
    logger.info(f"Generated pExtTransferLimit with {len(ext_transfer_df)} rows")


def _generate_trade_price(
    folder_output: Path,
    neighbors: list,
    existing_zext: list,
    output_dir: Path,
    country_lower: str,
    hourly_price_df: pd.DataFrame = None,
    trade_price_df: pd.DataFrame = None,
    scenario_reference: str = 'baseline'
) -> None:
    """
    Generate pTradePrice for the single-country model.

    Combines two sources:
    1. pHourlyPrice - for neighbors (zones that were internal in regional model)
    2. pTradePrice - for existing external zones (like SAPP)

    Args:
        folder_output: Path to output folder (used if hourly_price_df not provided)
        neighbors: List of neighbor zone names (from pTransmissionCapacity)
        existing_zext: List of existing external zones (from zext.csv)
        output_dir: Directory to write the output file
        country_lower: Lowercase country name for file naming
        hourly_price_df: DataFrame with pHourlyPrice from GDX (for neighbor zones)
        trade_price_df: DataFrame with pTradePrice from GDX (for existing external zones)
        scenario_reference: Reference scenario to use for prices (default: 'baseline')
    """
    result_dfs = []

    # 1. Process neighbors from pHourlyPrice
    if neighbors and hourly_price_df is not None and not hourly_price_df.empty:
        neighbor_df = _process_hourly_price_for_neighbors(
            hourly_price_df, neighbors, scenario_reference
        )
        if neighbor_df is not None:
            result_dfs.append(neighbor_df)

    # 2. Process existing external zones from pTradePrice
    if existing_zext and trade_price_df is not None and not trade_price_df.empty:
        existing_df = _process_trade_price_for_existing_zext(
            trade_price_df, existing_zext, scenario_reference
        )
        if existing_df is not None:
            result_dfs.append(existing_df)

    # Combine and write output
    if not result_dfs:
        logger.warning("No trade prices found for any external zones")
        pd.DataFrame(columns=['zext', 'q', 'd', 'y']).to_csv(
            output_dir / f"pTradePrice_{country_lower}.csv",
            index=False
        )
        return

    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df.to_csv(
        output_dir / f"pTradePrice_{country_lower}.csv",
        index=False
    )
    logger.info(f"Generated pTradePrice with {len(combined_df)} rows")


def _process_hourly_price_for_neighbors(
    hourly_price_df: pd.DataFrame,
    neighbors: list,
    scenario_reference: str
) -> pd.DataFrame:
    """Convert pHourlyPrice to pTradePrice format for neighbor zones."""
    df = _filter_to_scenario(hourly_price_df, scenario_reference)

    # Filter to neighbor zones
    neighbor_prices = df[df['zone'].isin(neighbors)].copy()

    if neighbor_prices.empty:
        logger.warning(f"No hourly prices found for neighbors {neighbors}")
        return None

    try:
        # Pivot to wide format
        pivoted = neighbor_prices.pivot_table(
            index=['zone', 'season', 'day', 'year'],
            columns='t',
            values='value',
            aggfunc='first',
            observed=True
        ).reset_index()

        # Rename to output format (short names for GAMS input)
        pivoted = pivoted.rename(columns={'zone': 'zext', 'season': 'q', 'day': 'd', 'year': 'y'})

        # Ensure t columns have 't' prefix
        col_rename = {}
        for col in pivoted.columns:
            if col not in ['zext', 'q', 'd', 'y']:
                if not str(col).startswith('t'):
                    col_rename[col] = f't{col}'
        if col_rename:
            pivoted = pivoted.rename(columns=col_rename)

        # Reorder columns
        base_cols = ['zext', 'q', 'd', 'y']
        t_cols = sorted(
            [c for c in pivoted.columns if str(c).startswith('t')],
            key=lambda x: int(str(x)[1:])
        )
        return pivoted[base_cols + t_cols]

    except Exception as e:
        logger.warning(f"Could not process pHourlyPrice for neighbors: {e}")
        return None


def _process_trade_price_for_existing_zext(
    trade_price_df: pd.DataFrame,
    existing_zext: list,
    scenario_reference: str
) -> pd.DataFrame:
    """Extract pTradePrice entries for existing external zones."""
    df = _filter_to_scenario(trade_price_df, scenario_reference)

    # Filter to existing external zones
    existing_prices = df[df['zext'].isin(existing_zext)].copy()

    if existing_prices.empty:
        logger.warning(f"No trade prices found for existing zext {existing_zext}")
        return None

    try:
        # Pivot to wide format (same as pHourlyPrice)
        pivoted = existing_prices.pivot_table(
            index=['zext', 'season', 'day', 'year'],
            columns='t',
            values='value',
            aggfunc='first',
            observed=True
        ).reset_index()

        # Rename to output format (short names for GAMS input)
        pivoted = pivoted.rename(columns={'season': 'q', 'day': 'd', 'year': 'y'})

        # Ensure t columns have 't' prefix
        col_rename = {}
        for col in pivoted.columns:
            if col not in ['zext', 'q', 'd', 'y']:
                if not str(col).startswith('t'):
                    col_rename[col] = f't{col}'
        if col_rename:
            pivoted = pivoted.rename(columns=col_rename)

        # Reorder columns
        base_cols = ['zext', 'q', 'd', 'y']
        t_cols = sorted(
            [c for c in pivoted.columns if str(c).startswith('t')],
            key=lambda x: int(str(x)[1:])
        )
        return pivoted[base_cols + t_cols]

    except Exception as e:
        logger.warning(f"Could not process pTradePrice for existing zext: {e}")
        return None


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
        print("\nNote: This script now expects GDX data to be passed programmatically.")
        print("For standalone testing, use the postprocessing module with --focus_country.")
        sys.exit(1)

    folder_input = Path(sys.argv[1])
    folder_output = Path(sys.argv[2])
    country = sys.argv[3]

    # Note: For standalone use, GDX data would need to be loaded separately
    generate_single_country_inputs(folder_input, folder_output, country)
