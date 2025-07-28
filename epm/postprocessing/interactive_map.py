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

import os 
import folium
import geopandas as gpd
from .utils import *


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

    centers = {map_geojson_to_epm[c]: v for c, v in centers.items() if c in map_geojson_to_epm}

    return zone_map, centers


def get_json_data(epm_results=None, selected_zones=None, dict_specs=None, geojson_to_epm=None, geo_add=None,
                  zone_map=None):
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
        - `map_countries`: GeoDataFrame of all countries.

    Returns
    -------
    tuple
        - zone_map (gpd.GeoDataFrame): Processed zone map including divided regions.
        - geojson_to_epm (dict): Updated mapping of GeoJSON names to EPM zones.
    """
    assert ((dict_specs is not None) or (geojson_to_epm is not None)), "Mapping zone names from geojson to EPM must be provided either under dict_specs or under geojson_to_epm"

    if dict_specs is None:
        if 'postprocessing' in os.getcwd():
            dict_specs = read_plot_specs(folder='')
        else:
            dict_specs = read_plot_specs(folder='postprocessing')
    if geojson_to_epm is None:
        geojson_to_epm = dict_specs['geojson_to_epm']
    else:
        if not os.path.exists(geojson_to_epm):
            raise FileNotFoundError(f"GeoJSON to EPM mapping file not found: {os.path.abspath(geojson_to_epm)}")
        geojson_to_epm = pd.read_csv(geojson_to_epm)
    epm_to_geojson = {v: k for k, v in
                      geojson_to_epm.set_index('Geojson')['EPM'].to_dict().items()}  # Reverse dictionary
    geojson_to_divide = geojson_to_epm.loc[geojson_to_epm.region.notna()]
    geojson_complete = geojson_to_epm.loc[~geojson_to_epm.region.notna()]
    if selected_zones is None:
        selected_zones_epm = geojson_to_epm['EPM'].unique()
    else:
        selected_zones_epm = selected_zones
    selected_zones_to_divide = [e for e in selected_zones_epm if e in geojson_to_divide['EPM'].values]
    selected_countries_geojson = [
        epm_to_geojson[key] for key in selected_zones_epm if
        ((key not in selected_zones_to_divide) and (key in epm_to_geojson))
    ]

    if zone_map is None:
        zone_map = dict_specs['map_countries']  # getting json data on all countries
    else:
        zone_map = gpd.read_file(zone_map)

    zone_map = zone_map[zone_map['ADMIN'].isin(selected_countries_geojson)]

    if geo_add is not None:
        zone_map_add = gpd.read_file(geo_add)
        zone_map = pd.concat([zone_map, zone_map_add])

    divided_parts = []
    for (country, division), subset in geojson_to_divide.groupby(['country', 'division']):
        # Apply division function
        divided_parts.append(divide(dict_specs['map_countries'], country, division))

    if divided_parts:
        zone_map_divide = pd.concat(divided_parts)

        zone_map_divide = \
        geojson_to_divide.rename(columns={'country': 'ADMIN'}).merge(zone_map_divide, on=['region', 'ADMIN'])[
            ['Geojson', 'ISO_A3', 'ISO_A2', 'geometry']]
        zone_map_divide = zone_map_divide.rename(columns={'Geojson': 'ADMIN'})
        # Convert zone_map_divide back to a GeoDataFrame
        zone_map_divide = gpd.GeoDataFrame(zone_map_divide, geometry='geometry', crs=zone_map.crs)

        # Ensure final zone_map is in EPSG:4326
        zone_map = pd.concat([zone_map, zone_map_divide]).to_crs(epsg=4326)
    geojson_to_epm = geojson_to_epm.set_index('Geojson')['EPM'].to_dict()  # get only relevant info
    return zone_map, geojson_to_epm


def divide(geodf, country, division):
    """
    Divide a country's geometry into two subzones using North-South (NS) or East-West (EW) division.

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
        raise ValueError("Invalid division type. Use 'NS' (North-South) or 'EW' (East-West).")


def plot_zone_map_on_ax(ax, zone_map):
    zone_map.plot(ax=ax, color='white', edgecolor='black')

    # Adjusting the limits to better center the zone_map on the region
    ax.set_xlim(zone_map.bounds.minx.min() - 1, zone_map.bounds.maxx.max() + 1)
    ax.set_ylim(zone_map.bounds.miny.min() - 1, zone_map.bounds.maxy.max() + 1)


def make_overall_map(zone_map, dict_colors, centers, year, region, scenario, filename, map_epm_to_geojson,
                     df_capacity=None, df_transmission=None, column_lines='value', min_lines=0,
                     min_line_width=1, max_line_width=5, index_pie='fuel',
                     figsize=(10, 6), percent_cap=25, bbox_to_anchor=(0.5, -0.1), loc='center left', min_size=0.5,
                     max_size =2.5, pie_sizing=True, show_arrows=False, arrow_style='-|>', arrow_size = 20,
                     arrow_offset_ratio=0.1, plot_colored_countries=True, plot_lines=True, offset=0.5,
                     arrow_linewidth=1, mutation_scale=3, predefined_colors=None):

    # Define consistent colors for each country
    if predefined_colors is None:
        unique_countries = zone_map['ADMIN'].unique()
        colors = get_extended_pastel_palette(len(unique_countries))
        predefined_colors = {country: colors[i] for i, country in enumerate(unique_countries)}
        # predefined_colors = {country: plt.cm.Pastel1(i % 9) for i, country in enumerate(unique_countries)}

    # Filter data for the given year and scenario
    transmission_data = df_transmission[
        (df_transmission['year'] == year) &
        (df_transmission['scenario'] == scenario) &
        (df_transmission[column_lines] > min_lines)
        ]

    capacity_data = df_capacity[
        (df_transmission['year'] == year) &
        (df_transmission['scenario'] == scenario)
        ]

    # Compute capacity range for scaling line width
    if not transmission_data.empty:
        min_cap = transmission_data[column_lines].min()
        max_cap = transmission_data[column_lines].max()
    else:
        min_cap = max_cap = 1  # Avoid division by zero

    # Function to scale line width
    def scale_line_width(capacity):
        if max_cap == min_cap:
            return min_line_width
        return min_line_width + (capacity - min_cap) / (max_cap - min_cap) * (max_line_width - min_line_width)

    def calculate_pie_size(zone, capacity_data):
        """Calculate pie chart size based on region area."""
        # area = region_sizes.loc[region_sizes['Name'] == zone, 'area'].values[0]
        # normalized_area = (area - region_sizes['area'].min()) / (region_sizes['area'].max() - region_sizes['area'].min())
        area = capacity_data[(capacity_data['zone'] == zone) ].value.sum()
        normalized_area = (area - capacity_data.groupby('zone').value.sum().min()) / (capacity_data.groupby('zone').value.sum().max() - capacity_data.groupby('zone').value.sum().min())
        return min_size + normalized_area * (max_size - min_size)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot the base zone map with predefined colors for each country
    if isinstance(plot_colored_countries, bool):
        if plot_colored_countries:
            zone_map['color'] = zone_map['ADMIN'].map(predefined_colors)
            zone_map.plot(ax=ax, color=zone_map['color'], edgecolor='black')
        else:
            zone_map.plot(ax=ax, color='white', edgecolor='black')
    else:  # plot_colored_countries is a list of countries
        assert isinstance(plot_colored_countries, list), 'plot_colored_countries must be a list or a bool'
        zone_map['color'] = zone_map['ADMIN'].apply(
            lambda c: predefined_colors[c] if c in plot_colored_countries else 'white'
        )
        zone_map.plot(ax=ax, color=zone_map['color'], edgecolor='black')

    handles, labels = [], []
    # Plot pie charts for each zone
    for zone in capacity_data['zone'].unique():
        # Extract capacity mix for the given zone and year
        CapacityMix_plot = (capacity_data[(capacity_data['zone'] == zone)]
                            .set_index(index_pie)['value']
                            .fillna(0)).reset_index()

        # Skip empty plots
        if CapacityMix_plot['value'].sum() == 0:
            continue

        # Get map coordinates
        coordinates = centers.get(zone, (0, 0))
        loc = fig.transFigure.inverted().transform(ax.transData.transform(coordinates))

        # Pie chart positioning and size
        size = [0.03, 0.07]
        if pie_sizing:
            pie_size = calculate_pie_size(zone, df_capacity)
        else:
            pie_size = None

        # Create inset pie chart
        ax_pie = fig.add_axes([loc[0] - 0.45 * size[0], loc[1] - 0.5 * size[1], size[0], size[1]])
        colors = [dict_colors[f] for f in CapacityMix_plot[index_pie]]
        h, l = plot_pie_on_ax(ax_pie, CapacityMix_plot, index_pie, percent_cap, colors, None, radius= pie_size)
        ax_pie.set_axis_off()

        for handle, label in zip(h, l):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc=loc, frameon=False, ncol=1,
               bbox_to_anchor=bbox_to_anchor)

    # Save and show figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return 0


def make_capacity_mix_map(zone_map, pCapacityByFuel, dict_colors, centers, year, region, scenario, filename,
                          map_epm_to_geojson, index='fuel', list_reduced_size=None, figsize=(10, 6), percent_cap=25,
                          bbox_to_anchor=(0.5, -0.1), loc='center left', min_size=0.5, max_size =2.5, pie_sizing=True):
    """
    Plots a capacity mix map with pie charts overlaid on a regional map.

    Parameters:
    - zone_map: GeoDataFrame containing the map regions.
    - CapacityMix_scen: DataFrame containing the capacity mix data per zone.
    - fuels_list: List of fuels to include in the plot.
    - centers: Dictionary mapping zones to their center coordinates.
    - year: The target year for the plot.
    - region_name: Name of the region for the title.
    - scenario: Scenario name for the title.
    - graphs_folder: Path where the plot will be saved.
    - selected_scenario: The specific scenario being plotted.
    - geojson_names: List of country names in the GeoJSON file.
    - model_names: List of country names used in the model.
    - list_reduced_size: List of zones where pie size should be reduced.
    - colorf: Function mapping fuel names to colors.
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot the base zone map
    plot_zone_map_on_ax(ax, zone_map)

    # Remove axes for a clean map
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'Capacity mix - {region} \n {scenario} - {year}', loc='center')

    # Compute pie sizes for each zone
    region_sizes = zone_map.copy()
    region_sizes['area'] = region_sizes.geometry.area
    region_sizes['Name'] = region_sizes['ADMIN'].replace(map_epm_to_geojson)

    def calculate_pie_size(zone, CapacityByFuel):
        """Calculate pie chart size based on region area."""
        # area = region_sizes.loc[region_sizes['Name'] == zone, 'area'].values[0]
        # normalized_area = (area - region_sizes['area'].min()) / (region_sizes['area'].max() - region_sizes['area'].min())
        area = pCapacityByFuel[(pCapacityByFuel['zone'] == zone) & (pCapacityByFuel['year'] == year)].value.sum()
        normalized_area = (area - CapacityByFuel.groupby('zone').value.sum().min()) / (CapacityByFuel.groupby('zone').value.sum().max() - CapacityByFuel.groupby('zone').value.sum().min())
        return min_size + normalized_area * (max_size - min_size)

    handles, labels = [], []
    # Plot pie charts for each zone
    for zone in pCapacityByFuel['zone'].unique():
        # Extract capacity mix for the given zone and year
        CapacityMix_plot = (pCapacityByFuel[(pCapacityByFuel['zone'] == zone) & (pCapacityByFuel['year'] == year)  & (pCapacityByFuel['scenario'] == scenario)]
                            .set_index(index)['value']
                            .fillna(0)).reset_index()

        # Skip empty plots
        if CapacityMix_plot['value'].sum() == 0:
            continue

        # Get map coordinates
        coordinates = centers.get(zone, (0, 0))
        loc = fig.transFigure.inverted().transform(ax.transData.transform(coordinates))

        # Pie chart positioning and size
        size = [0.03, 0.07]
        if pie_sizing:
            if list_reduced_size is not None:
                pie_size = 0.7 if zone in list_reduced_size else calculate_pie_size(zone, pCapacityByFuel)
            else:
                pie_size = calculate_pie_size(zone, pCapacityByFuel)
        else:
            pie_size = None

        # Create inset pie chart
        ax_pie = fig.add_axes([loc[0] - 0.45 * size[0], loc[1] - 0.5 * size[1], size[0], size[1]])
        colors = [dict_colors[f] for f in CapacityMix_plot[index]]
        h, l = plot_pie_on_ax(ax_pie, CapacityMix_plot, index, percent_cap, colors, None, radius= pie_size)
        ax_pie.set_axis_off()

        for handle, label in zip(h, l):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc=loc, frameon=False, ncol=1,
               bbox_to_anchor=bbox_to_anchor)

    # Save and show figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_interconnection_map(zone_map, pAnnualTransmissionCapacity, centers, year, scenario, column='value', color_col=None, filename=None,
                             min_capacity=0.1, figsize=(12, 8), show_labels=True, label_yoffset=0.02, label_xoffset=0.02,
                             label_fontsize=12, predefined_colors=None, min_display_value=100,
                             min_line_width=1, max_line_width=5, format_y=lambda y, _: '{:.0f} MW'.format(y),
                             title='Transmission capacity', show_arrows=False,
                             arrow_style='-|>', arrow_size = 20,
                             arrow_offset_ratio=0.1, plot_colored_countries=True, plot_lines=True, offset=0.5,
                             arrow_linewidth=1, mutation_scale=3):
    """
    Plots an interconnection map showing transmission capacities between different zones.

    Parameters:
    - zone_map: pd.DataFrame
    GeoDataFrame containing the map regions.
    - pAnnualTransmissionCapacity: pd.DataFrame
     Dataframe containing transmission capacities (zone_from, zone_to, value).
    - centers: dict
    Dictionary mapping zones to their center coordinates.
    - year: int
    The target year for the plot.
    - scenario: str
    Scenario name for the title.
    - filename: Path
    Path where the plot will be saved (optional).
    - min_capacity: float
    Minimum capacity threshold for plotting lines (default 0.1 GW).
    - figsize: tuple
    Tuple defining figure size (default (12, 8)).
    - show_labels: bool
    Whether to display country names on the map (default True).
    - label_yoffset: float
    Proportion of figure height to shift labels vertically (default 0.02, normalized value).
    - label_xoffset: float
    Proportion of figure width to shift labels horizontally (default 0.02, normalized value).
    - label_fontsize: int
    Font size for country labels (default 12).
    - predefined_colors: dict
    Dictionary mapping country names to predefined colors to ensure consistency across plots.
    - min_display_capacity: float
    Minimum capacity value required to display text on the transmission line (default 0.5 GW).
    - min_line_width: float
    Minimum line width for transmission lines (default 1).
    - max_line_width: float
    Maximum line width for the largest transmission capacity (default 5).
    """
    # Define consistent colors for each country
    if predefined_colors is None:
        unique_countries = zone_map['ADMIN'].unique()
        colors = get_extended_pastel_palette(len(unique_countries))
        predefined_colors = {country: colors[i] for i, country in enumerate(unique_countries)}
        # predefined_colors = {country: plt.cm.Pastel1(i % 9) for i, country in enumerate(unique_countries)}

    # Filter data for the given year and scenario
    transmission_data = pAnnualTransmissionCapacity[
        (pAnnualTransmissionCapacity['year'] == year) &
        (pAnnualTransmissionCapacity['scenario'] == scenario) &
        (pAnnualTransmissionCapacity[column] > min_capacity)
        ]

    # Compute capacity range for scaling line width
    if not transmission_data.empty:
        min_cap = transmission_data[column].min()
        max_cap = transmission_data[column].max()
    else:
        min_cap = max_cap = 1  # Avoid division by zero

    # Function to scale line width
    def scale_line_width(capacity):
        if max_cap == min_cap:
            return min_line_width
        return min_line_width + (capacity - min_cap) / (max_cap - min_cap) * (max_line_width - min_line_width)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot the base zone map with predefined colors for each country
    if isinstance(plot_colored_countries, bool):
        if plot_colored_countries:
            zone_map['color'] = zone_map['ADMIN'].map(predefined_colors)
            zone_map.plot(ax=ax, color=zone_map['color'], edgecolor='black')
        else:
            zone_map.plot(ax=ax, color='white', edgecolor='black')
    else:  # plot_colored_countries is a list of countries
        assert isinstance(plot_colored_countries, list), 'plot_colored_countries must be a list or a bool'
        zone_map['color'] = zone_map['ADMIN'].apply(
            lambda c: predefined_colors[c] if c in plot_colored_countries else 'white'
        )
        zone_map.plot(ax=ax, color=zone_map['color'], edgecolor='black')


    # Remove axes for a clean map
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f'{title} - {scenario} - {year}', loc='center')

    # Get vertical and horizontal extent of the figure to normalize offsets
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    y_offset = (ymax - ymin) * label_yoffset
    x_offset = (xmax - xmin) * label_xoffset

    min_color_value = transmission_data[column].min()
    max_color_value = transmission_data[column].max()

    if color_col is not None:
        min_color_value_arrow = transmission_data[color_col].min()
        max_color_value_arrow = transmission_data[color_col].max()

    # Plot interconnections
    for _, row in transmission_data.iterrows():
        zone_from, zone_to, value = row['zone_from'], row['zone_to'], row[column]

        if zone_from in centers and zone_to in centers:
            coord_from, coord_to = centers[zone_from], centers[zone_to]
            coor_mid = [(coord_from[0] + coord_to[0]) / 2, (coord_from[1] + coord_to[1]) / 2]

            line_width = scale_line_width(value)

            color = calculate_color_gradient(value, min_color_value, max_color_value)

            if plot_lines:  # plotting transmission lines
                ax.plot([coord_from[0], coord_to[0]], [coord_from[1], coord_to[1]], color=color,
                        linewidth=3)

                # Optional arrow
                if show_arrows:
                    dx = coord_to[0] - coord_from[0]
                    dy = coord_to[1] - coord_from[1]
                    start_x = coord_from[0] + dx * (0.5 - arrow_offset_ratio)
                    start_y = coord_from[1] + dy * (0.5 - arrow_offset_ratio)
                    end_x = coord_from[0] + dx * (0.5 + arrow_offset_ratio)
                    end_y = coord_from[1] + dy * (0.5 + arrow_offset_ratio)

                    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                            arrowstyle=arrow_style,
                                            color=color,
                                            mutation_scale=arrow_size,
                                            linewidth=0)
                    ax.add_patch(arrow)

                if value >= min_display_value:
                    ax.text(coor_mid[0], coor_mid[1], format_y(value, None), ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'), fontsize=10)

            else:   # Plotting flows

                dx = coord_to[0] - coord_from[0]
                dy = coord_to[1] - coord_from[1]

                # Midpoint for label placement
                mid_x = (coord_from[0] + coord_to[0]) / 2
                mid_y = (coord_from[1] + coord_to[1]) / 2

                # Shorten the arrow to avoid overlapping arrowheads
                arrow_offset = arrow_offset_ratio
                start_x = coord_from[0] + dx * arrow_offset
                start_y = coord_from[1] + dy * arrow_offset
                end_x = coord_to[0] - dx * arrow_offset
                end_y = coord_to[1] - dy * arrow_offset

                # Compute direction for label offset
                norm = np.hypot(dx, dy)
                unit_dx, unit_dy = dx / norm, dy / norm
                norm_dx = -unit_dy
                norm_dy = unit_dx

                if color_col is not None:
                    color_value = row[color_col]
                    color = calculate_color_gradient(color_value, min_color_value_arrow, max_color_value_arrow)
                else:
                    color = 'black'

                # Arrow width scaling with value
                arrow_linewidth = scale_line_width(value)  # or define your own scaling logic
                arrowstyle = f"simple,head_length={0.5},head_width={0.5},tail_width={0.2*arrow_linewidth}"

                # Plot arrow
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                        arrowstyle=arrowstyle,
                                        color=color,
                                        edgecolor='black',  # Optional: to match the example style
                                        linewidth=0,  # outline thickness (not width of the arrow)
                                        mutation_scale=mutation_scale,
                                        zorder=5)
                ax.add_patch(arrow)

                # Add text at midpoint with perpendicular offset
                if value >= min_display_value:
                    text_x = mid_x + offset * norm_dx
                    text_y = mid_y + offset * norm_dy

                    ax.text(text_x, text_y, format_y(value, None),
                            ha='center', va='center', fontsize=label_fontsize,
                            fontweight='bold', color='black')

    # Optionally plot zone labels with a normalized offset
    if show_labels:
        for zone, coord in centers.items():
            ax.text(coord[0] + x_offset, coord[1] + y_offset, zone.replace('_', ' '), fontsize=label_fontsize,
                    ha='center', va='bottom')

    # Save or show the figure
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_extended_pastel_palette(n):
    # Get base pastel colormaps
    pastel1 = [plt.cm.Pastel1(i) for i in range(9)]
    pastel2 = [plt.cm.Pastel2(i) for i in range(8)]

    # Combine and repeat if needed
    base_colors = pastel1 + pastel2
    if n <= len(base_colors):
        return base_colors[:n]

    # Generate extra soft pastel colors if needed
    import colorsys
    extra_needed = n - len(base_colors)
    extra_colors = []
    for i in range(extra_needed):
        h = (i / extra_needed)
        s = 0.4  # low saturation = pastel
        v = 0.9  # high brightness = pastel
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        extra_colors.append((r, g, b))

    return base_colors + extra_colors


def create_interactive_map(zone_map, centers, transmission_data, energy_data, year, scenario, filename,
                           dict_specs, pCapacityByFuel, pEnergyByFuel, pDispatch, pPlantDispatch, pPrice, label_size=14):
    """
    Create an interactive HTML map displaying energy capacity, dispatch, and interconnections.

    Parameters:
    - zone_map (GeoDataFrame): Geospatial data for regions
    - centers (dict): Mapping of zone names to coordinates
    - transmission_data (DataFrame): Transmission line capacities and utilization rates
    - energy_data (DataFrame): Energy-related data including capacity, generation, and demand
    - graphs_folder (str): Folder path for saving generated plots
    - year (int): Year of the analysis
    - scenario (str): Scenario name
    - filename (str): Output HTML file name
    - dict_specs (dict): Specifications for plotting
    - pCapacityByFuel (DataFrame): Capacity mix data
    - pDispatch (DataFrame): Dispatch data
    """
    # Focus the map on the bounding box of the region
    bounds = zone_map.total_bounds  # [minx, miny, maxx, maxy]
    region_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]  # Center latitude, longitude
    energy_map = folium.Map(location=region_center, zoom_start=6, tiles='CartoDB positron')

    # Add country zones
    folium.GeoJson(zone_map, style_function=lambda feature: {
        'fillColor': '#ffffff', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.3
    }).add_to(energy_map)

    # Plotting transmission information
    transmission_data = transmission_data.copy()
    transmission_data = transmission_data.loc[(transmission_data.year == year) & (transmission_data.scenario == scenario)]
    if not transmission_data.empty:  # ie, there is transmission data to plot
        transmission_data.drop(columns=['scenario'], inplace=True)
        transmission_data = transmission_data.set_index(['zone_from', 'zone_to'])

        # Getting the topology of lines, counting each line a unique time
        topology = set(transmission_data.index.unique())
        final_topology = set()
        for (z, z2) in topology:
            if (z2, z) not in final_topology:
                final_topology.add((z, z2))

        for (z, z2) in final_topology:
            row1 = transmission_data.loc[(z, z2)]
            row2 = transmission_data.loc[(z2, z)]
            zone1, zone2 = row1.name[0], row1.name[1]
            capacity, utilization_1to2, utilization_2to1 = max(row1.fillna(0)['capacity'], row2.fillna(0)['capacity']), row1['utilization'], row2['utilization']

            if zone1 in centers and zone2 in centers:
                coords = [[centers[zone1][1], centers[zone1][0]],  # Lat, Lon
                          [centers[zone2][1], centers[zone2][0]]]  # Lat, Lon
                color = calculate_color_gradient(utilization_1to2 + utilization_2to1, 0, 100)

                tooltip_text = f"""
                <div style="font-size: {label_size}px;">
                <b>Capacity:</b> {capacity:.0f} MW <br>
                <b>Utilization {zone1} - {zone2}:</b> {utilization_1to2:.0f}% <br>
                <b>Utilization {zone2} - {zone1}:</b> {utilization_2to1:.0f}%
                </div>
                """

                folium.PolyLine(
                    locations=coords, color=color, weight=4,
                    tooltip=tooltip_text
                ).add_to(energy_map)

    # Add zone markers with popup information and dynamically generated images
    for zone, coords in centers.items():
        if zone in energy_data['zone'].unique():
            coords = [coords[1], coords[0]]  # changing to Lat,Long as required by Folium
            popup_content = f"""
            <b>{zone}</b><br>
            Generation: {get_value(energy_data, zone, year, scenario, 'Total production: GWh'):,.0f} GWh<br>
            Demand: {get_value(energy_data, zone, year, scenario, 'Demand: GWh'):,.0f} GWh<br>
            Imports: {get_value(energy_data, zone, year, scenario, 'Imports exchange: GWh'):,.0f} GWh<br>
            Exports: {get_value(energy_data, zone, year, scenario, 'Exports exchange: GWh'):,.0f} GWh
            """

            # Generate and embed capacity mix and dispatch plots
            popup_content += generate_zone_plots(zone, year, scenario, dict_specs, pCapacityByFuel, pEnergyByFuel, pDispatch,
                                                pPlantDispatch, pPrice, scale_factor=0.8)

            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, min_width=800, max_height=700),
                icon=folium.Icon(color='blue', icon="")
            ).add_to(energy_map)

    # Fit map to bounds
    energy_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Save the map
    energy_map.save(filename)
    print(f"Interactive map saved to {filename}")


def make_automatic_map(epm_results, dict_specs, GRAPHS_FOLDER, selected_scenarios=None):
    # TODO: ongoing work
    if selected_scenarios is None:
        selected_scenarios = list(epm_results['pPlantDispatch'].scenario.unique())

    pAnnualTransmissionCapacity = epm_results['pAnnualTransmissionCapacity'].copy()
    pInterconUtilization = epm_results['pInterconUtilization'].copy()
    pInterconUtilization['value'] = pInterconUtilization['value'] * 100  # percentage
    years = epm_results['pAnnualTransmissionCapacity']['year'].unique()

    for selected_scenario in selected_scenarios:
        print(f'Automatic map for scenario {selected_scenario}')
        folder = f'{GRAPHS_FOLDER}/{selected_scenario}/map'
        if not os.path.exists(folder):
            os.mkdir(folder)
        # Select first and last years
        years = [min(years), max(years)]
        #years = [y for y in [2025, 2030, 2035, 2040] if y in years]

        try:
            zone_map, geojson_to_epm = get_json_data(epm_results=epm_results, dict_specs=dict_specs)

            zone_map, centers = create_zonemap(zone_map, map_geojson_to_epm=geojson_to_epm)

        except Exception as e:
            print(
                'Error when creating zone geojson for automated map graphs. This may be caused by a problem when specifying a mapping between EPM zone names, and GEOJSON zone names.\n Edit the `geojson_to_epm.csv` file in the `static` folder.')
            raise  # Re-raise the exception for debuggings

        capa_transmission = epm_results['pAnnualTransmissionCapacity'].copy()
        utilization_transmission = epm_results['pInterconUtilization'].copy()
        utilization_transmission['value'] = utilization_transmission['value'] * 100  # percentage
        utilization_transmission_max = keep_max_direction(utilization_transmission)  # we sum utilization across both directions, and keep the direction with the maximum utilization (for arrows on graph)
        transmission_data = capa_transmission.rename(columns={'value': 'capacity'}).merge(
            utilization_transmission_max.rename
            (columns={'value': 'utilization'}), on=['scenario', 'zone', 'z2', 'year'])
        transmission_data = transmission_data.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

        for year in years:
            filename = f'{folder}/TransmissionCapacity_{selected_scenario}_{year}.png'

            if not transmission_data.loc[transmission_data.scenario == selected_scenario].empty:  # only plotting transmission when there is information to plot
                make_interconnection_map(zone_map, transmission_data, centers, year=year, scenario=selected_scenario, column='capacity',
                                         label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=True,
                                         min_display_value=200, filename=filename, title='Transmission capacity (MW)')

                filename = f'{folder}/TransmissionUtilization_{selected_scenario}_{year}.png'

                make_interconnection_map(zone_map, transmission_data, centers, year=year, scenario=selected_scenario, column='utilization',
                                         min_capacity=0.01, label_yoffset=0.01, label_xoffset=-0.05,
                                         label_fontsize=10, show_labels=False, min_display_value=50,
                                         format_y=lambda y, _: '{:.0f} %'.format(y), filename=filename, title='Transmission utilization (%)')

                make_interconnection_map(zone_map, transmission_data, centers, year=year, scenario=selected_scenario,
                                         column='utilization',
                                         min_capacity=0.01, label_yoffset=0.01, label_xoffset=-0.05,
                                         label_fontsize=10, show_labels=False, min_display_value=50,
                                         format_y=lambda y, _: '{:.0f} %'.format(y), filename=filename,
                                         title='Transmission utilization (%)', show_arrows=True, arrow_offset_ratio=0.4,
                                         arrow_size=25, plot_colored_countries=False)

                tmp = epm_results['pInterchange'].copy()
                df_congested = epm_results['pCongested'].copy().rename(columns={'value': 'congestion'})
                tmp = tmp.merge(df_congested, on=['scenario', 'year', 'zone', 'z2'], how='left')
                # Fill only numerical columns with 0
                tmp = tmp.fillna({
                    col: 0 for col in tmp.select_dtypes(include=["number"]).columns
                })
                tmp_rev = tmp.copy().rename(columns={'zone': 'z2', 'z2': 'zone'})
                tmp_rev['value'] = - tmp_rev['value']
                df_combined = pd.concat([tmp, tmp_rev], ignore_index=True)
                df_combined = df_combined.groupby(['scenario', 'year', 'zone', 'z2'])[['value', 'congestion']].sum().reset_index()
                df_net = df_combined[df_combined['value'] > 0]
                df_net = df_net.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

                filename = f'{folder}/NetExports_{selected_scenario}_{year}.png'

                make_interconnection_map(zone_map, df_net, centers, filename=filename, year=year, scenario=selected_scenario,
                                         title='Net Exports (GWh)',
                                         label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=False,
                                         plot_colored_countries=False,
                                         min_display_value=100, column='value', plot_lines=False,
                                         format_y=lambda y, _: '{:.0f}'.format(y), offset=-1.5,
                                         min_line_width=0.7, max_line_width=1.5, arrow_linewidth=0.1, mutation_scale=20,
                                         color_col='congestion')

            if len(epm_results['pDemandSupply'].loc[(epm_results['pDemandSupply'].scenario == selected_scenario)].zone.unique()) > 1:  # only plotting on interactive map when more than one zone
                    energy_data = epm_results['pDemandSupply'].copy()
                    pCapacityByFuel = epm_results['pCapacityByFuel'].copy()
                    pEnergyByFuel = epm_results['pEnergyByFuel'].copy()
                    pDispatch = epm_results['pDispatch'].copy()
                    pPlantDispatch = epm_results['pPlantDispatch'].copy()
                    pPrice = epm_results['pPrice'].copy()
                    filename = f'{folder}/InteractiveMap_{selected_scenario}_{year}.html'

                    transmission_data = capa_transmission.rename(columns={'value': 'capacity'}).merge(
                        utilization_transmission.rename
                        (columns={'value': 'utilization'}), on=['scenario', 'zone', 'z2', 'year'])
                    transmission_data = transmission_data.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

                    create_interactive_map(zone_map, centers, transmission_data, energy_data, year, selected_scenario, filename,
                                           dict_specs, pCapacityByFuel, pEnergyByFuel, pDispatch, pPlantDispatch, pPrice)
