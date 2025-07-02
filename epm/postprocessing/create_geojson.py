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
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
# Importing utility functions for data processing
from utils import get_json_data, create_zonemap


def create_geojson_for_tableau(geojson_to_epm, zcmap, selected_zones, folder='tableau'):
    """
    Generate a GeoJSON file representing lines between selected EPM zones for use in Tableau visualizations.

    This function creates a GeoDataFrame with LineString geometries connecting the centroids of selected zones.
    Each pair of zones is represented as a directed line with associated metadata, allowing users to map
    inter-zone connections in Tableau. The resulting file is saved in the output folder as
    'linestring_countries_2.geojson'.

    Parameters
    ----------
    selected_zones : list of str
        List of EPM zone identifiers to include in the visualization (e.g., ['ETH_North', 'KEN', 'TZA']).

    geojson_to_epm : str
        Filename (within ../output/{folder}/) of the CSV mapping GeoJSON zone names to EPM zone identifiers.

    zcmap : str
        Filename (within ../output/{folder}/) of the CSV mapping EPM zone names (`z`) to countries.

    folder : str
        Name of the output folder where processed data and the GeoJSON file should be saved.

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
        ../output/{folder}/linestring_countries_2.geojson

    Notes
    -----
    - The function uses centroids of the input geometries to create lines, simplifying visualization.
    - Self-pairs (i.e., lines from a zone to itself) are excluded.
    - Designed for visualizing zone-to-zone relations (e.g., trade, transmission) in Tableau.
    """

    # Load and process zone geometries for the selected zones
    geojson_to_epm_path = os.path.join('..', 'output', folder, geojson_to_epm)
    # Creating zone map for desired zones
    zone_map, geojson_to_epm_dict = get_json_data(selected_zones=selected_zones, geojson_to_epm=geojson_to_epm_path)

    zone_map, centers = create_zonemap(zone_map, map_geojson_to_epm=geojson_to_epm)

    # Processing for Tableau use
    countries_shapefile = zone_map
    countries_shapefile['geometry'] = countries_shapefile.centroid
    countries_shapefile = countries_shapefile.set_index('ADMIN')

    # Load mapping file and join it to assign EPM zone names to geometries
    geojson_to_epm = os.path.join('..', 'output', folder, geojson_to_epm)  # loading again geojson_to_epm
    geojson_to_epm = pd.read_csv(geojson_to_epm)
    geojson_to_epm = geojson_to_epm.set_index('Geojson')
    countries_shapefile['z'] = geojson_to_epm.EPM
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

    # Add country codes for both zones (start and end)
    zcmap = os.path.join('..', 'output', folder, zcmap)
    zcmap = pd.read_csv(zcmap)

    zcmap = zcmap.set_index('Zone')
    result_df = result_df.set_index('z')
    result_df['c'] = zcmap['Country']
    result_df = result_df.reset_index().set_index('z_other')
    result_df['c2'] = zcmap['Country']

    result_df.to_file(os.path.join('..', 'output', folder, 'linestring_countries.geojson'), driver='GeoJSON')
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

    args = parser.parse_args()

    linestring = create_geojson_for_tableau(
        selected_zones=args.zones,
        geojson_to_epm=args.geojson,
        zcmap=args.zcmap,
        folder=args.folder
    )

    print("GeoJSON created with", len(linestring), "lines.")

