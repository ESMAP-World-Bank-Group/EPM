"""Cutting EPM zones out of reference polygons.

The geometry half of what used to live at the top of
`epm/postprocessing/maps.py`: reading the admin-area-to-zone mapping, loading
the reference polygons, splitting a country into subregions, and assembling the
polygons of one model's zones. Moved here because none of it is
post-processing -- it reads no model result and can run before a solve.

`epm/postprocessing/maps.py` re-exports these names, so callers that import
them from there keep working.

This module deliberately imports geopandas and pandas only. Pulling in
`epm.postprocessing.utils` costs a further ~3 s (gams.transfer, matplotlib,
seaborn, PIL) for a logger and two file paths, so the few names it needs from
there are imported inside the functions that use them.
"""

import io
import os
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = REPO_ROOT / 'epm' / 'resources' / 'postprocess'

#: admin-0 polygons every zone that is a whole country is cut from
ZONE_MAP = str(RESOURCES_DIR / 'zones.geojson')
#: hand-drawn areas no admin-0 polygon can supply, in the same ADMIN schema
ZONES_CUSTOM = str(RESOURCES_DIR / 'zones_custom.geojson')
#: shared admin area -> zone mapping, used when a data folder has no override
GEOJSON_TO_EPM = str(RESOURCES_DIR / 'geojson_to_epm.csv')


def _warn(message):
    """Warn through the post-processing logger when one is set, else print.

    Imported lazily: during a run `epm.postprocessing.utils` is already loaded,
    so this costs nothing, and a standalone build never pays for it.
    """
    from epm.postprocessing.utils import log_warning
    log_warning(message)


_GEOJSON_HEADER = "epm_zone,source_name,subregion,split"


def read_geojson_mapping(path):
    """
    Read a GeoJSON-to-EPM mapping CSV.

    Required format: epm_zone,source_name,subregion,split
    - epm_zone: Zone name in the EPM model (required)
    - source_name: Zone/country name in GeoJSON file matching ADMIN field (required)
    - subregion: For split zones only - north/south/east/west/center
    - split: For split zones only - NS/EW/NSE/NCS

    Some exported CSVs append the header again without a newline, causing pandas
    to fail when parsing. This helper inserts the missing newline and removes
    duplicate header rows so that `pd.read_csv` can succeed.
    """
    with open(path, encoding='utf-8-sig') as fp:
        raw_text = fp.read()

    # Check for required header format
    if _GEOJSON_HEADER not in raw_text:
        raise ValueError(
            f"Invalid geojson_to_epm.csv format in {path}\n"
            f"  Required header: {_GEOJSON_HEADER}\n"
            f"  Columns:\n"
            f"    - epm_zone: Zone name in your EPM model\n"
            f"    - source_name: Zone name in GeoJSON file (matches ADMIN field)\n"
            f"    - subregion: Optional, for split zones (north/south/east/west/center)\n"
            f"    - split: Optional, split pattern (NS/EW/NSE/NCS)"
        )

    pattern = r'(?<=.)(?<![\r\n])' + re.escape(_GEOJSON_HEADER)
    normalized_text = re.sub(pattern, '\n' + _GEOJSON_HEADER, raw_text)

    clean_lines = []
    header_seen = False
    for line in normalized_text.splitlines():
        if line.strip() == _GEOJSON_HEADER:
            if header_seen:
                continue
            header_seen = True
        clean_lines.append(line)

    clean_text = "\n".join(clean_lines)
    if not clean_text.endswith("\n"):
        clean_text += "\n"

    return pd.read_csv(io.StringIO(clean_text))


def load_zone_map(zone_map=None, zones_custom=None):
    """Admin-0 polygons plus the hand-drawn zones that no admin area can supply.

    A zone such as an industrial off-taker (Mozal) or a sub-national market
    (Trakia) matches no entry of the admin polygon file, so it is drawn from an
    overlay whose features carry the same ADMIN/ISO_A3 columns and therefore
    resolve through `geojson_to_epm.csv` like any country.

    The overlay is appended here rather than at each call site, so that the
    standalone regeneration of the map layers and the plotting pipeline always
    see the same map. `zones_custom` lets a data folder override the shared
    overlay with its own.
    """
    zones = gpd.read_file(zone_map or ZONE_MAP)
    overlay = zones_custom or ZONES_CUSTOM
    if os.path.exists(overlay):
        zones = pd.concat([zones, gpd.read_file(overlay)], ignore_index=True)
    return zones


def divide(geodf, country, division):
    """
    Divide a country's geometry into two or more subzones using North-South (NS), East-West (EW), or
    three-way splits.

    This function overlays the country geometry with a dividing polygon and extracts
    the two subregions.

    Parameters
    ----------
    geodf : gpd.GeoDataFrame
        GeoDataFrame containing geometries of all countries.
    country : str
        Name of the country to divide.
    division : str
        Type of division:
        - 'NS' (North-South) splits along the latitude midpoint.
        - 'EW' (East-West) splits along the longitude midpoint.
        - 'NSE' (North-South-East) splits into three quadrants.
        - 'NCS' (North-Center-South) splits into three horizontal bands.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the divided subregions with the correct CRS.
    """
    # Get the country geometry
    crs = geodf.crs
    country_geometry = geodf.loc[geodf['ADMIN'] == country, 'geometry'].values[0]

    # Get bounds
    minx, miny, maxx, maxy = country_geometry.bounds

    if division == 'NS':
        median_latitude = (miny + maxy) / 2
        south_polygon = Polygon([(minx, miny), (minx, median_latitude), (maxx, median_latitude), (maxx, miny)])
        north_polygon = Polygon([(minx, median_latitude), (minx, maxy), (maxx, maxy), (maxx, median_latitude)])

        # Convert to GeoDataFrame with the correct CRS
        south_gdf = gpd.GeoDataFrame(geometry=[south_polygon], crs=crs)
        north_gdf = gpd.GeoDataFrame(geometry=[north_polygon], crs=crs)

        south_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], south_gdf, how='intersection')
        north_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], north_gdf, how='intersection')
        south_part = south_part.to_crs(crs)
        north_part = north_part.to_crs(crs)
        south_part['region'] = 'south'
        north_part['region'] = 'north'

        return pd.concat([south_part, north_part])

    elif division == 'EW':
        median_longitude = (minx + maxx) / 2
        west_polygon = Polygon([(minx, miny), (minx, maxy), (median_longitude, maxy), (median_longitude, miny)])
        east_polygon = Polygon([(median_longitude, miny), (median_longitude, maxy), (maxx, maxy), (maxx, miny)])

        # Convert to GeoDataFrame with the correct CRS
        west_gdf = gpd.GeoDataFrame(geometry=[west_polygon], crs=crs)
        east_gdf = gpd.GeoDataFrame(geometry=[east_polygon], crs=crs)

        west_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country],west_gdf, how='intersection')
        east_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], east_gdf, how='intersection')
        west_part['region'] = 'west'
        east_part['region'] = 'east'
        
        return pd.concat([west_part, east_part])

    elif division == 'NCS':
        third_latitude = (maxy - miny) / 3
        south_limit = miny + third_latitude
        north_limit = maxy - third_latitude

        south_polygon = Polygon([(minx, miny), (minx, south_limit), (maxx, south_limit), (maxx, miny)])
        center_polygon = Polygon([(minx, south_limit), (minx, north_limit), (maxx, north_limit), (maxx, south_limit)])
        north_polygon = Polygon([(minx, north_limit), (minx, maxy), (maxx, maxy), (maxx, north_limit)])

        south_gdf = gpd.GeoDataFrame(geometry=[south_polygon], crs=crs)
        center_gdf = gpd.GeoDataFrame(geometry=[center_polygon], crs=crs)
        north_gdf = gpd.GeoDataFrame(geometry=[north_polygon], crs=crs)

        south_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], south_gdf, how='intersection')
        center_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], center_gdf, how='intersection')
        north_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], north_gdf, how='intersection')

        south_part['region'] = 'south'
        center_part['region'] = 'center'
        north_part['region'] = 'north'

        return pd.concat([north_part, center_part, south_part])

    elif division == 'NSE':
        median_latitude = (miny + maxy) / 2
        median_longitude = (minx + maxx) / 2
        north_polygon = Polygon([(minx, median_latitude), (minx, maxy), (median_longitude, maxy), (median_longitude, median_latitude)])
        south_polygon = Polygon([(minx, miny), (minx, median_latitude), (median_longitude, median_latitude), (median_longitude, miny)])
        east_polygon = Polygon([(median_longitude, miny), (median_longitude, median_latitude), (maxx, median_latitude), (maxx, miny)])
        west_polygon = Polygon([(minx, median_latitude), (minx, maxy), (median_longitude, maxy), (median_longitude, median_latitude)])
        # Convert to GeoDataFrame with the correct CRS
        north_gdf = gpd.GeoDataFrame(geometry=[north_polygon], crs=crs)
        south_gdf = gpd.GeoDataFrame(geometry=[south_polygon], crs= crs)
        east_gdf = gpd.GeoDataFrame(geometry=[east_polygon], crs= crs)
        west_gdf = gpd.GeoDataFrame(geometry=[west_polygon], crs= crs)
        north_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], north_gdf, how='intersection')
        south_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], south_gdf, how='intersection')
        east_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], east_gdf, how='intersection')
        west_part = gpd.overlay(geodf.loc[geodf['ADMIN'] == country], west_gdf, how='intersection')
        north_part['region'] = 'north'
        south_part['region'] = 'south'
        east_part['region'] = 'east'
        west_part['region'] = 'west'

        return pd.concat([east_part, north_part, south_part])

    else:
        raise ValueError("Invalid division type. Use 'NS', 'EW', 'NSE', or 'NCS'.")


def get_json_data(epm_results=None, selected_zones=None, dict_specs=None, geojson_to_epm=None, geo_add=None,
                  zone_map=None, zones_custom=None):
    """
    Extract and process zone map data, handling divisions for sub-national regions.

    This function retrieves the zone map, identifies zones that need to be divided
    (e.g., North-South or East-West split), applies the `divide` function, and
    returns a processed GeoDataFrame ready for visualization.

    Parameters
    ----------
    epm_results : dict
        Dictionary containing EPM results, including transmission capacity data.
    dict_specs : dict
        Dictionary with mapping specifications, including:
        - `geojson_to_epm`: Mapping from GeoJSON names to EPM zone names.
        - `map_zones`: GeoDataFrame of all countries.

    Returns
    -------
    tuple
        - zone_map (gpd.GeoDataFrame): Processed zone map including divided regions.
        - geojson_to_epm (dict): Updated mapping of GeoJSON names to EPM zones.
    """
    # If neither dict_specs nor geojson_to_epm is provided, load default specs
    if dict_specs is None:
        # Falling back to the plotting specs pulls in the whole post-processing
        # stack; a caller that passes its own sources never gets here.
        from epm.postprocessing.utils import read_plot_specs
        dict_specs = read_plot_specs()
    if geojson_to_epm is None:
        geojson_to_epm = dict_specs['geojson_to_epm']
    else:
        if not os.path.exists(geojson_to_epm):
            raise FileNotFoundError(f"GeoJSON to EPM mapping file not found: {os.path.abspath(geojson_to_epm)}")
        geojson_to_epm = read_geojson_mapping(geojson_to_epm)
    # Build reverse mapping: epm_zone -> source_name
    epm_to_source = {v: k for k, v in
                     geojson_to_epm.set_index('source_name')['epm_zone'].to_dict().items()}
    # Separate zones that need splitting from complete zones
    zones_to_split = geojson_to_epm.loc[geojson_to_epm.subregion.notna()]
    zones_complete = geojson_to_epm.loc[~geojson_to_epm.subregion.notna()]
    if selected_zones is None:
        selected_zones_epm = geojson_to_epm['epm_zone'].unique()
    else:
        selected_zones_epm = selected_zones
    selected_zones_to_split = [z for z in selected_zones_epm if z in zones_to_split['epm_zone'].values]
    selected_sources = [
        epm_to_source[key] for key in selected_zones_epm if
        ((key not in selected_zones_to_split) and (key in epm_to_source))
    ]

    if zone_map is None and zones_custom is None:
        zone_map = dict_specs['map_zones']  # getting json data on all countries
    else:
        # Reading a map explicitly must still pick up the hand-drawn zones, or a
        # standalone regeneration would silently drop every zone that has no
        # admin polygon.
        zone_map = load_zone_map(zone_map, zones_custom)

    zone_map = zone_map[zone_map['ADMIN'].isin(selected_sources)]

    if geo_add is not None:
        zone_map_add = gpd.read_file(geo_add)
        zone_map = pd.concat([zone_map, zone_map_add])

    divided_parts = []
    # Filter zones_to_split to only include zones that are actually selected
    zones_to_split_selected = zones_to_split[zones_to_split['epm_zone'].isin(selected_zones_to_split)]
    # source_name column contains the polygon name (matches ADMIN in world GeoJSON)
    for (source_name, split_type), subset in zones_to_split_selected.groupby(['source_name', 'split']):
        # Apply division function to split the polygon
        divided_parts.append(divide(dict_specs['map_zones'], source_name, split_type))

    if divided_parts:
        zone_map_divide = pd.concat(divided_parts)

        # Merge divided parts with mapping info
        # Use source_name as ADMIN for the merge (matches divide() output)
        merge_df = zones_to_split_selected.copy()
        merge_df['ADMIN'] = merge_df['source_name']
        # divide() returns 'region' column, so rename subregion to match
        merge_df = merge_df.rename(columns={'subregion': 'region'})
        # The custom zone layer carries epm_zone/epm_country of its own, so the
        # divided parts inherit them and the merge would suffix both sides away.
        # The mapping is the authority on those, so drop the layer's copy.
        clash = [c for c in merge_df.columns
                 if c in zone_map_divide.columns and c not in ('region', 'ADMIN')]
        zone_map_divide = zone_map_divide.drop(columns=clash)
        zone_map_divide = merge_df.merge(zone_map_divide, on=['region', 'ADMIN'])[
            ['epm_zone', 'ISO_A3', 'ISO_A2', 'geometry']]
        # Use epm_zone as the final ADMIN (unique identifier for each zone)
        zone_map_divide = zone_map_divide.rename(columns={'epm_zone': 'ADMIN'})
        # Convert zone_map_divide back to a GeoDataFrame
        zone_map_divide = gpd.GeoDataFrame(zone_map_divide, geometry='geometry', crs=zone_map.crs)

        # Ensure final zone_map is in EPSG:4326
        zone_map = pd.concat([zone_map, zone_map_divide]).to_crs(epsg=4326)

    # Build the mapping dict for create_zonemap()
    # For complete zones: source_name -> epm_zone
    # For split zones: epm_zone -> epm_zone (identity, since ADMIN is now the epm_zone name)
    geojson_to_epm_dict = zones_complete.set_index('source_name')['epm_zone'].to_dict()
    for epm_zone in zones_to_split_selected['epm_zone'].values:
        geojson_to_epm_dict[epm_zone] = epm_zone

    return zone_map, geojson_to_epm_dict


def create_zonemap(zone_map, map_geojson_to_epm):
    """
    Convert zone map to the correct coordinate reference system (CRS) and extract centroids.

    This function ensures that the provided `zone_map` is in EPSG:4326 (latitude/longitude),
    extracts the centroid coordinates of each zone, and maps them to the EPM zone names.

    Parameters
    ----------
    zone_map : gpd.GeoDataFrame
        A GeoDataFrame containing zone geometries and attributes.
    map_geojson_to_epm : dict
        Dictionary mapping GeoJSON zone names to EPM zone names.

    Returns
    -------
    tuple
        - zone_map (gpd.GeoDataFrame): The zone map converted to EPSG:4326.
        - centers (dict): Dictionary mapping EPM zone names to their centroid coordinates [longitude, latitude].
    """
    if zone_map.crs is not None and zone_map.crs.to_epsg() != 4326:
        zone_map = zone_map.to_crs(epsg=4326)  # Convert to EPSG:4326 for folium

    # Get the coordinates of the centers of the zones
    centers = {
        row['ADMIN']: [row.geometry.centroid.x, row.geometry.centroid.y]
        for _, row in zone_map.iterrows()
    }

    # Report unmapped zones
    all_geojson_zones = set(centers.keys())
    mapped_zones = set(map_geojson_to_epm.keys())
    unmapped = all_geojson_zones - mapped_zones

    if unmapped:
        _warn(f"Zones in GeoJSON but not in mapping (will be skipped): {list(unmapped)}")

    centers = {map_geojson_to_epm[c]: v for c, v in centers.items() if c in map_geojson_to_epm}

    if not centers:
        _warn(
            f"No zone centers could be extracted. Map visualization will be empty.\n"
            f"  - GeoJSON zones available: {list(all_geojson_zones)}\n"
            f"  - Mapping expects: {list(map_geojson_to_epm.keys())}"
        )

    return zone_map, centers
