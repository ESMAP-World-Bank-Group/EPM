"""
**********************************************************************
* ELECTRICITY PLANNING MODEL (EPM)
* Developed at the World Bank
**********************************************************************
Description:
    This Python script is part of the GAMS-based Electricity Planning Model (EPM),
    designed for electricity system planning. It supports tasks such as capacity
    expansion, generation dispatch, and the enforcement of policy constraints,
    including renewable energy targets and emissions limits.

Author(s):
    ESMAP Modelling Team

Organization:
    World Bank

Version:
    (Specify version here)

License:
    Creative Commons Zero v1.0 Universal

Key Features:
    - Optimization of electricity generation and capacity planning
    - Inclusion of renewable energy integration and storage technologies
    - Multi-period, multi-region modeling framework
    - CO₂ emissions constraints and policy instruments

Notes:
    - Ensure GAMS is installed and the model has completed execution
      before running this script.
    - The model generates output files in the working directory
      which will be organized by this script.

Contact:
    Claire Nicolas — c.nicolas@worldbank.org
**********************************************************************
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

# Suppress warning about centroid on geographic CRS - acceptable for visualization
warnings.filterwarnings('ignore', message=".*Geometry is in a geographic CRS.*", category=UserWarning)

# If this script runs directly from `epm/postprocessing`, make sure the
# repository root is on `sys.path` so package imports succeed.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing utility functions for data processing using the package path
from epm.postprocessing.maps import get_json_data, create_zonemap
from epm.postprocessing.utils import log_warning, log_info


def _build_zones_gdf(zone_map_gdf, geojson_to_epm_dict, zcmap_df, zone_col, country_col):
    """Build polygon GeoDataFrame with z, ISO_A3, c, geometry columns for EPM View."""
    zones = zone_map_gdf.copy()
    zones['z'] = zones['ADMIN'].map(geojson_to_epm_dict)
    zones = zones[zones['z'].notna()].copy()
    zcmap_lookup = zcmap_df.set_index(zone_col)[country_col]
    zones['c'] = zones['z'].map(zcmap_lookup)
    return gpd.GeoDataFrame(
        zones[['z', 'ISO_A3', 'c', 'geometry']].reset_index(drop=True),
        geometry='geometry',
        crs=zone_map_gdf.crs
    )


def create_geojson_for_tableau(geojson_to_epm, zcmap, selected_zones, folder='tableau',
                               zone_map=None, output_path=None, dict_specs=None,
                               output_stem=None):
    """
    Generate linestring and zones GeoJSON files for selected EPM zones.

    Produces two files:
    - linestring_{output_stem}.geojson (or linestring_countries.geojson by default):
      LineString geometries connecting zone centroids, for Tableau / NTC visualization.
    - zones_{output_stem}.geojson (or zones.geojson by default):
      Polygon geometries per EPM zone, for MapLibre GL fill layers.

    Parameters
    ----------
    geojson_to_epm : str or dict
        Either a filename (within ../output/{folder}/) of the CSV mapping GeoJSON zone names to EPM zone
        identifiers, OR a dict mapping GeoJSON names to EPM zone names (for pipeline integration).

    zcmap : str or pd.DataFrame
        Either a filename (within ../output/{folder}/) of the CSV mapping EPM zone names to countries,
        OR a DataFrame with columns ['zone', 'country'] (for pipeline integration).

    selected_zones : list of str
        List of EPM zone identifiers to include in the visualization (e.g., ['ETH_North', 'KEN', 'TZA']).

    folder : str, optional
        Name of the output folder where processed data and the GeoJSON file should be saved.
        Used when geojson_to_epm and zcmap are filenames. Default is 'tableau'.

    zone_map : str, optional
        User-specific geojson file path (default: None, uses built-in zones.geojson).

    output_path : str, optional
        If provided, use this exact path for output instead of constructing from folder.
        Useful for pipeline integration.

    dict_specs : dict, optional
        If provided, use for loading zone data via get_json_data (for pipeline integration).

    output_stem : str, optional
        Stem for output file names. If provided, outputs are named
        'linestring_{output_stem}.geojson' and 'zones_{output_stem}.geojson'.
        Defaults to 'linestring_countries' and 'zones' respectively.

    Returns
    -------
    result_df : geopandas.GeoDataFrame
        A GeoDataFrame containing pairwise LineStrings between selected zones.
    """

    # Handle geojson_to_epm: either a file path (str) or already loaded path for dict_specs
    if isinstance(geojson_to_epm, str):
        # Check if it's an absolute path or just a filename
        if os.path.isabs(geojson_to_epm) or os.path.exists(geojson_to_epm):
            geojson_to_epm_path = geojson_to_epm
        else:
            geojson_to_epm_path = os.path.join('..', 'output', folder, geojson_to_epm)
    else:
        geojson_to_epm_path = None

    # Handle zone_map parameter
    if zone_map is not None and isinstance(zone_map, str):
        if not os.path.isabs(zone_map) and not os.path.exists(zone_map):
            zone_map = os.path.join('..', 'output', folder, zone_map)

    # Load zone map and geojson_to_epm mapping
    if dict_specs is not None:
        # Pipeline mode: use dict_specs for loading
        zone_map_gdf, geojson_to_epm_dict = get_json_data(
            selected_zones=selected_zones,
            dict_specs=dict_specs
        )
    else:
        # Standalone mode: use default resources or custom file path if provided
        if geojson_to_epm_path is not None and os.path.exists(geojson_to_epm_path):
            zone_map_gdf, geojson_to_epm_dict = get_json_data(
                selected_zones=selected_zones,
                geojson_to_epm=geojson_to_epm_path,
                zone_map=zone_map
            )
        else:
            # Use default resources from read_plot_specs()
            zone_map_gdf, geojson_to_epm_dict = get_json_data(
                selected_zones=selected_zones,
                zone_map=zone_map
            )

    zone_map_gdf, centers = create_zonemap(zone_map_gdf, map_geojson_to_epm=geojson_to_epm_dict)

    # Zone mapping diagnostics - only warn if there are issues
    zones_missing_geometry = [z for z in selected_zones if z not in centers]
    if zones_missing_geometry:
        log_warning(
            f"Linestring GeoJSON: {len(zones_missing_geometry)} zones missing map geometry:\n"
            f"  {zones_missing_geometry}\n"
            f"  To fix: Add entries to epm/resources/postprocess/geojson_to_epm.csv"
        )

    # Determine output file names
    ls_name = f'linestring_{output_stem}.geojson' if output_stem else 'linestring_countries.geojson'
    zones_name = f'zones_{output_stem}.geojson' if output_stem else 'zones.geojson'

    if output_path is not None:
        output_file = os.path.join(output_path, ls_name)
        zones_file = os.path.join(output_path, zones_name)
    else:
        output_file = os.path.join('..', 'output', folder, ls_name)
        zones_file = os.path.join('..', 'output', folder, zones_name)

    if not centers:
        log_warning(f"No zones have map geometry - creating empty GeoJSON files.")
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf.to_file(output_file, driver='GeoJSON')
        empty_gdf.to_file(zones_file, driver='GeoJSON')
        return empty_gdf

    # Build GeoDataFrame from centers (already has EPM zone names as keys)
    countries_shapefile = gpd.GeoDataFrame(
        {'z': list(centers.keys())},
        geometry=[Point(coords) for coords in centers.values()],
        crs="EPSG:4326"
    )

    countries_shapefile = countries_shapefile.reset_index(drop=True)

    # Create pairwise combinations (excluding self) to generate lines between all zones
    results = []
    for i, row1 in countries_shapefile.iterrows():
        for j, row2 in countries_shapefile.iterrows():
            if i != j:  # exclude self-comparison
                # Combine the rows as needed
                combined = {**row1.to_dict(), **{f'{k}_other': v for k, v in row2.to_dict().items()}}
                results.append(combined)

    result_df = pd.DataFrame(results)

    # Extract coordinates for the starting zone
    result_df['country_ini_lat'] = result_df['geometry'].apply(lambda x: x.y)
    result_df['country_ini_lon'] = result_df['geometry'].apply(lambda x: x.x)

    # Create LineString geometries between zone centroids
    result_df['geometry'] = result_df.apply(
        lambda row: LineString([row['geometry'], row['geometry_other']]),
        axis=1
    )
    result_df = gpd.GeoDataFrame(result_df, geometry='geometry')
    result_df.crs = countries_shapefile.crs
    result_df.drop(columns=['geometry_other'], inplace=True)

    # Compute the centroid of each line (used in Tableau for labeling or tooltips)
    result_df['centroid'] = result_df['geometry'].centroid
    result_df['lat_linestring'] = result_df['centroid'].apply(lambda x: x.y)
    result_df['lon_linestring'] = result_df['centroid'].apply(lambda x: x.x)
    result_df.drop(columns=['centroid'], inplace=True)

    # Handle zcmap: either a file path (str) or already loaded DataFrame
    if isinstance(zcmap, str):
        if os.path.isabs(zcmap) or os.path.exists(zcmap):
            zcmap_df = pd.read_csv(zcmap)
        else:
            zcmap_df = pd.read_csv(os.path.join('..', 'output', folder, zcmap))
    else:
        # Already a DataFrame
        zcmap_df = zcmap.copy()

    # Support both 'z'/'zone' and 'c'/'country' column names
    zone_col = 'zone' if 'zone' in zcmap_df.columns else 'z'
    country_col = 'country' if 'country' in zcmap_df.columns else 'c'

    # Build polygon zones GeoDataFrame before zcmap_df index is modified
    zones_gdf = _build_zones_gdf(zone_map_gdf, geojson_to_epm_dict, zcmap_df, zone_col, country_col)

    # Add country codes for both zones (start and end)
    zcmap_df = zcmap_df.set_index(zone_col)
    result_df = result_df.set_index('z')
    result_df['c'] = zcmap_df[country_col]
    result_df = result_df.reset_index().set_index('z_other')
    result_df['c2'] = zcmap_df[country_col]

    result_df.to_file(output_file, driver='GeoJSON')
    zones_gdf.to_file(zones_file, driver='GeoJSON')
    log_info(f"Linestring GeoJSON written: {output_file} ({len(result_df)} lines)")
    log_info(f"Zones GeoJSON written: {zones_file} ({len(zones_gdf)} zones)")
    return result_df


def regen_zones_from_run(run_folder, dict_specs=None):
    """
    Generate zones.geojson for an existing run folder from its linestring_countries.geojson.

    Reads zone names from the linestring file, rebuilds polygon geometries via
    get_json_data, and writes zones.geojson alongside. Use this for runs that
    predate automatic zones.geojson generation.

    Parameters
    ----------
    run_folder : str or Path
        Path to the run folder (e.g. epm/output_view/RETRADE_0626/).
    dict_specs : dict, optional
        If provided, use for loading zone data. Otherwise read_plot_specs() is called.
    """
    ls_path = Path(run_folder) / 'linestring_countries.geojson'
    if not ls_path.exists():
        raise FileNotFoundError(f"linestring_countries.geojson not found in {run_folder}")

    with open(ls_path, encoding='utf-8') as f:
        ls_data = json.load(f)

    zone_set = set()
    for feature in ls_data['features']:
        p = feature.get('properties', {})
        if p.get('z'):
            zone_set.add(p['z'])
        if p.get('z_other'):
            zone_set.add(p['z_other'])

    if not zone_set:
        raise ValueError("No zone identifiers found in linestring_countries.geojson properties")

    selected_zones = sorted(zone_set)
    log_info(f"regen_zones_from_run: rebuilding {len(selected_zones)} zones from {run_folder}")

    if dict_specs is None:
        from epm.postprocessing.postprocessing import read_plot_specs
        dict_specs = read_plot_specs()

    zone_map_gdf, geojson_to_epm_dict = get_json_data(
        selected_zones=selected_zones,
        dict_specs=dict_specs
    )
    zone_map_gdf, _ = create_zonemap(zone_map_gdf, map_geojson_to_epm=geojson_to_epm_dict)

    zones = zone_map_gdf.copy()
    zones['z'] = zones['ADMIN'].map(geojson_to_epm_dict)
    zones = zones[zones['z'].notna()].copy()
    zones_gdf = gpd.GeoDataFrame(
        zones[['z', 'ISO_A3', 'geometry']].reset_index(drop=True),
        geometry='geometry',
        crs=zone_map_gdf.crs
    )

    out_path = Path(run_folder) / 'zones.geojson'
    zones_gdf.to_file(str(out_path), driver='GeoJSON')
    log_info(f"zones.geojson written: {out_path} ({len(zones_gdf)} zones)")
    return zones_gdf


if __name__ == '__main__':
    HELP_TEXT = """
Generate GeoJSON files (linestring + zones) for an EPM input data folder.

Usage (from EPM_main directory):
    python epm/postprocessing/create_geojson.py --folder data_sapp --zcmap zcmap.csv

Generates in epm/input/{folder}/:
    linestring_{zcmap_stem}.geojson   — LineStrings between zone centroids
    zones_{zcmap_stem}.geojson        — Polygon per EPM zone

To regenerate zones.geojson for an existing run folder without re-running GAMS:
    python epm/postprocessing/create_geojson.py --regen-zones --run-folder epm/output_view/RETRADE_0626

Note: Runs automatically during postprocessing.py for multi-zone models.
"""

    parser = argparse.ArgumentParser(
        description="Generate GeoJSON files for EPM zone visualization.",
        epilog=HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zones", nargs="+", default=None,
                        help="List of EPM zone names (default: all zones from zcmap.csv)")
    parser.add_argument("--folder", type=str, default='data_test',
                        help="Folder name in epm/input/ where zcmap.csv is located (default: data_test)")
    parser.add_argument("--zcmap", type=str, default="zcmap.csv",
                        help="Filename of zone-country mapping CSV (default: zcmap.csv)")
    parser.add_argument("--regen-zones", action="store_true",
                        help="Regenerate zones.geojson for an existing run folder from its linestring file")
    parser.add_argument("--run-folder", type=str, default=None,
                        help="Run folder path for --regen-zones mode (e.g. epm/output_view/RETRADE_0626)")

    args = parser.parse_args()

    if args.regen_zones:
        if not args.run_folder:
            print("Error: --regen-zones requires --run-folder to be specified.")
            sys.exit(1)
        print(f"Regenerating zones.geojson for: {os.path.abspath(args.run_folder)}")
        zones = regen_zones_from_run(args.run_folder)
        print(f"Done: zones.geojson written with {len(zones)} zones.")
        sys.exit(0)

    # Normal mode: generate linestring + zones for an input data folder
    zcmap_stem = Path(args.zcmap).stem  # e.g. 'zcmap' or 'zcmap_alt'

    if os.path.isabs(args.zcmap) or os.path.exists(args.zcmap):
        zcmap_path = args.zcmap
    else:
        zcmap_path = os.path.join('epm', 'input', args.folder, args.zcmap)

    selected_zones = args.zones
    if selected_zones is None:
        if os.path.exists(zcmap_path):
            zcmap_df = pd.read_csv(zcmap_path)
            zone_col = 'zone' if 'zone' in zcmap_df.columns else 'z'
            selected_zones = zcmap_df[zone_col].unique().tolist()
            print(f"Using all zones from {args.zcmap}: {selected_zones}")
        else:
            print(f"Error: zcmap not found at {os.path.abspath(zcmap_path)}. Provide --zones or valid --zcmap path.")
            sys.exit(1)
    else:
        print(f"Generating GeoJSON for zones: {selected_zones}")

    output_path = os.path.join('epm', 'input', args.folder)

    if not os.path.exists(output_path):
        print(f"Error: Output folder does not exist: {os.path.abspath(output_path)}")
        sys.exit(1)

    print(f"Output folder: {os.path.abspath(output_path)}")

    linestring = create_geojson_for_tableau(
        selected_zones=selected_zones,
        geojson_to_epm=None,  # Use defaults from resources
        zcmap=zcmap_path,
        folder=args.folder,
        output_path=output_path,
        output_stem=zcmap_stem
    )

    print(f"linestring_{zcmap_stem}.geojson: {len(linestring)} lines")
    print(f"zones_{zcmap_stem}.geojson: written")
