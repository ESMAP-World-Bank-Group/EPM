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
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

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
        # Standalone mode: use file path
        zone_map_gdf, geojson_to_epm_dict = get_json_data(
            selected_zones=selected_zones,
            geojson_to_epm=geojson_to_epm_path,
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

    # Add country codes for both zones (start and end)
    zcmap_df = zcmap_df.set_index('zone')
    result_df = result_df.set_index('z')
    result_df['c'] = zcmap_df['country']
    result_df = result_df.reset_index().set_index('z_other')
    result_df['c2'] = zcmap_df['country']

    # Determine output file path
    if output_path is not None:
        output_file = os.path.join(output_path, 'linestring_countries.geojson')
    else:
        output_file = os.path.join('..', 'output', folder, 'linestring_countries.geojson')

    result_df.to_file(output_file, driver='GeoJSON')
    return result_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Tableau-ready GeoJSON for selected zones.")
    parser.add_argument("--zones", nargs="+",
                        help="List of EPM zone names to include (e.g., Angola Botswana Zambia).")
    parser.add_argument("--folder", type=str, default='tableau',
                        help="Output folder containing CSVs result - which will be used in Tableau - and where the GeoJSON will be saved.")
    parser.add_argument("--geojson", type=str, default="geojson_to_epm.csv",
                        help="Filename of GeoJSON to EPM mapping (default: geojson_to_epm.csv).")
    parser.add_argument("--zcmap", type=str, default="zcmap.csv",
                        help="Filename of zone-to-country mapping (default: zcmap.csv).")
    parser.add_argument("--zonemap", type=str, default=None,
                        help="User-specific geojson file (default: None).")

    args = parser.parse_args()

    linestring = create_geojson_for_tableau(
        selected_zones=args.zones,
        geojson_to_epm=args.geojson,
        zcmap=args.zcmap,
        folder=args.folder,
        zone_map=args.zonemap
    )

    print("GeoJSON created with", len(linestring), "lines.")
