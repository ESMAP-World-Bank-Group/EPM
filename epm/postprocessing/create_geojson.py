
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
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

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


def create_geojson_for_tableau(geojson_to_epm, zcmap, selected_zones, folder='tableau',
                               zone_map=None, output_path=None, dict_specs=None):
    """
    Generate a GeoJSON file representing lines between selected EPM zones for use in Tableau visualizations.

    This function creates a GeoDataFrame with LineString geometries connecting the centroids of selected zones.
    Each pair of zones is represented as a directed line with associated metadata, allowing users to map
    inter-zone connections in Tableau. The resulting file is saved in the output folder as
    'linestring_countries.geojson'.

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

    Returns
    -------
    result_df : geopandas.GeoDataFrame
        A GeoDataFrame containing pairwise LineStrings between selected zones, with the following columns:
        - 'z': EPM zone identifier of the starting point.
        - 'c': Country of the starting zone.
        - 'z_other': EPM zone identifier of the destination point.
        - 'c2': Country of the destination zone.
        - 'country_ini_lat', 'country_ini_lon': Latitude and longitude of the starting zone.
        - 'lat_linestring', 'lon_linestring': Latitude and longitude of the LineString midpoint.
        - 'geometry': LineString geometry from the centroid of 'z' to 'z_other'.

    Output
    ------
    A GeoJSON file is written to:
        - {output_path}/linestring_countries.geojson (if output_path is provided)
        - ../output/{folder}/linestring_countries.geojson (otherwise)

    Notes
    -----
    - The function uses centroids of the input geometries to create lines, simplifying visualization.
    - Self-pairs (i.e., lines from a zone to itself) are excluded.
    - Designed for visualizing zone-to-zone relations (e.g., trade, transmission) in Tableau.
    - Can be called from command line (standalone) or from postprocessing pipeline.
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
        # If geojson_to_epm_path is None or doesn't exist, get_json_data will use defaults
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

    # Processing for Tableau use
    countries_shapefile = zone_map_gdf.copy()
    countries_shapefile['geometry'] = countries_shapefile.centroid
    countries_shapefile = countries_shapefile.set_index('ADMIN')

    # Assign EPM zone names to geometries using the mapping
    # geojson_to_epm_dict is {epm_name: geojson_name}, we need reverse mapping
    geojson_to_epm_reverse = {v: k for k, v in geojson_to_epm_dict.items()}
    countries_shapefile['z'] = countries_shapefile.index.map(geojson_to_epm_reverse)
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

    # Add country codes for both zones (start and end)
    zcmap_df = zcmap_df.set_index(zone_col)
    result_df = result_df.set_index('z')
    result_df['c'] = zcmap_df[country_col]
    result_df = result_df.reset_index().set_index('z_other')
    result_df['c2'] = zcmap_df[country_col]

    # Determine output file path
    if output_path is not None:
        output_file = os.path.join(output_path, 'linestring_countries.geojson')
    else:
        output_file = os.path.join('..', 'output', folder, 'linestring_countries.geojson')

    result_df.to_file(output_file, driver='GeoJSON')
    return result_df



if __name__ == '__main__':
    HELP_TEXT = """
Generate Tableau-ready GeoJSON with LineStrings connecting zone centroids.

Usage (from EPM_main directory):
    python epm/postprocessing/create_geojson.py --folder my_scenario

Reads zcmap.csv from ../input/{folder}/ and saves GeoJSON there.
Optional: --zones (defaults to all zones in zcmap)

Note: Runs automatically in postprocessing.py for multi-zone models.
"""

    parser = argparse.ArgumentParser(
        description="Generate Tableau-ready GeoJSON for selected zones.",
        epilog=HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zones", nargs="+", default=None,
                        help="List of EPM zone names (default: all zones from zcmap.csv)")
    parser.add_argument("--folder", type=str, default='data_test',
                        help="Folder name in ../input/ where zcmap.csv is located (default: data_test)")
    parser.add_argument("--zcmap", type=str, default="zcmap.csv",
                        help="Filename of zone-country mapping CSV (default: zcmap.csv)")

    args = parser.parse_args()

    # Build path to zcmap in ../input/{folder}/
    if os.path.isabs(args.zcmap) or os.path.exists(args.zcmap):
        zcmap_path = args.zcmap
    else:
        zcmap_path = os.path.join( '..', 'input', args.folder, args.zcmap)

    # If zones not specified, read them from zcmap
    selected_zones = args.zones
    if selected_zones is None:
        if os.path.exists(zcmap_path):
            zcmap_df = pd.read_csv(zcmap_path)
            # Support both 'z'/'zone' column names
            zone_col = 'zone' if 'zone' in zcmap_df.columns else 'z'
            selected_zones = zcmap_df[zone_col].unique().tolist()
            print(f"Using all zones from zcmap: {selected_zones}")
        else:
            print(f"Error: zcmap not found at {os.path.abspath(zcmap_path)}. Provide --zones or valid --zcmap path.")
            sys.exit(1)
    else:
        print(f"Generating Tableau GeoJSON for zones: {selected_zones}")

    # Output path is same as input folder
    output_path = os.path.join( '..', 'input', args.folder)

    # Ensure output directory exists
    if not os.path.exists(output_path):
        print(f"Error: Output folder does not exist: {os.path.abspath(output_path)}")
        sys.exit(1)

    print(f"Output folder: {os.path.abspath(output_path)}")

    linestring = create_geojson_for_tableau(
        selected_zones=selected_zones,
        geojson_to_epm=None,  # Use defaults from resources
        zcmap=zcmap_path,
        folder=args.folder,
        output_path=output_path
    )

    print(f"GeoJSON created with {len(linestring)} lines.")
