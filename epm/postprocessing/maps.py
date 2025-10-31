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
from .plots import subplot_pie, make_fuel_dispatchplot

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


def get_value(df, zone, year, scenario, attribute, column_to_select='attribute'):
    """Safely retrieves an energy value for a given zone, year, scenario, and attribute."""
    value = df.loc[
        (df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario) & (df[column_to_select] == attribute),
        'value'
    ]
    return value.values[0] if not value.empty else 0


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


def make_capacity_mix_map(zone_map, pCapacityFuel, dict_colors, centers, year, region, scenario, filename,
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
        area = pCapacityFuel[(pCapacityFuel['zone'] == zone) & (pCapacityFuel['year'] == year)].value.sum()
        normalized_area = (area - CapacityByFuel.groupby('zone').value.sum().min()) / (CapacityByFuel.groupby('zone').value.sum().max() - CapacityByFuel.groupby('zone').value.sum().min())
        return min_size + normalized_area * (max_size - min_size)

    handles, labels = [], []
    # Plot pie charts for each zone
    for zone in pCapacityFuel['zone'].unique():
        # Extract capacity mix for the given zone and year
        CapacityMix_plot = (pCapacityFuel[(pCapacityFuel['zone'] == zone) & (pCapacityFuel['year'] == year)  & (pCapacityFuel['scenario'] == scenario)]
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
                pie_size = 0.7 if zone in list_reduced_size else calculate_pie_size(zone, pCapacityFuel)
            else:
                pie_size = calculate_pie_size(zone, pCapacityFuel)
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


def calculate_color_gradient(value, min_val, max_val, start_color=(135, 206, 250), end_color=(139, 0, 0)):
    """Generates a color gradient based on a value range."""
    ratio = (value - min_val) / (max_val - min_val)
    ratio = min(max(ratio, 0), 1)  # Clamp between 0 and 1
    ratio = ratio**2.5  # Exponential scaling
    r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
    g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
    b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


def make_complete_value_dispatch_plot(df_dispatch, zone, year, scenario, unit_value, title,
                                      filename=None, select_time=None, legend_loc='bottom', bottom=0, figsize=(20,6), fontsize=12):
    """
    Generates and saves a dispatch plot for a specific value (e.g., Imports, Exports, Demand).

    Parameters
    ----------
    dfs_value : dict
        Dictionary containing dataframes for the selected value plot.
    zone : str
        The zone to visualize.
    year : int
        The target year.
    scenario : str
        The scenario to visualize.
    value : str
        The specific attribute to visualize (e.g., 'Imports', 'Exports', 'Demand').
    unit_value : str
        Unit of the displayed value (e.g., 'GWh', 'MW').
    title : str
        Title of the plot.
    filename : str, optional
        Path to save the figure, default is None.
    select_time : dict, optional
        Time selection parameters for filtering the data.
    legend_loc : str, optional
        Location of the legend (default is 'bottom').
    bottom : float, optional
        Adjusts bottom margin for better layout (default is 0).
    figsize : tuple, optional
        Size of the figure, default is (10,6).

    Returns
    -------
    None
    """

    df_dispatch_value = df_dispatch.loc[(df_dispatch['zone']==zone)&(df_dispatch['scenario']==scenario)&(df_dispatch['year']==year)]

    # Extracting unique seasons and representative days
    dispatch_seasons = list(df_dispatch['season'].unique())
    n_rep_days = len(list(df_dispatch['day'].unique()))

    # Selecting

    # Plot
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

    # Plot the selected value dispatch
    df_dispatch_value = df_dispatch_value.set_index(['scenario', 'year', 'season', 'day', 't'])
    df_dispatch_value.plot(ax=ax, color='steelblue')

    ax.legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adding the representative days as vertical lines
    m = 0
    d_m = 0
    x_ds = []
    for d in range(len(dispatch_seasons) * n_rep_days):
        if d != 0:
            m = m + 1
            d_m = d_m + 1
            x_d = 24 * d - 1
            if m == n_rep_days:
                ax.axvline(x=x_d, color='slategrey', linestyle='-')
                ax.text(x=x_d-12, y=(ax.get_ylim()[1]) * 0.99, s=f'd{str(int(d_m))}', ha='center')
                m = 0
                d_m = 0
            else:
                ax.axvline(x=x_d, color='slategrey', linestyle='--')
                ax.text(x=x_d-12, y=(ax.get_ylim()[1]) * 0.99, s=f'd{str(int(d_m))}', ha='center')
            x_ds = x_ds + [x_d]

    # Adding the last day label
    ax.text(x=x_d+12, y=(ax.get_ylim()[1]) * 0.9, s=f'd{str(int(d_m+1))}', ha='center')
    ax.set_xlabel("")
    ax.set_ylabel(unit_value, fontsize=fontsize, fontweight='bold')
    ax.text(0, 1.2, title, fontsize=fontsize, fontweight='bold', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_xticks([24 * n_rep_days * s - 24 * n_rep_days / 2 for s in range(len(dispatch_seasons) + 1)])
    ax.set_xticklabels([''] + [str(s) for s in dispatch_seasons])
    ax.set_xlim(left=0)

    fig.text(0.5, 0.05, 'Hours', ha='center', fontsize=fontsize, fontweight='bold')

    # Save plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_zone_plots(zone, year, scenario, dict_specs, pCapacityFuel, pEnergyFuel, pDispatch, pDispatchPlant, pPrice, scale_factor=0.8):
    """Generate capacity mix and dispatch plots for a given zone and return them as base64 strings."""
    # Generate capacity mix pie chart using existing function
    df1 = pCapacityFuel.copy()
    df1['attribute'] = 'Capacity'
    df2 = pEnergyFuel.copy()
    df2['attribute'] = 'Energy'
    df = pd.concat([df1, df2])
    capacity_plot = make_pie_chart_interactive(
        df=df, zone=zone, year=year, scenario=scenario,
        dict_colors=dict_specs['colors'], index='fuel'
    )

    df_exchanges = pDispatch.loc[pDispatch['attribute'].isin(['Imports', 'Exports'])]
    df_exchanges_piv = df_exchanges.pivot(index= ['scenario', 'year', 'season', 'day', 't', 'zone'], columns = 'attribute', values = 'value').reset_index()
    df_exchanges_piv[['Exports', 'Imports']] = df_exchanges_piv[['Exports', 'Imports']].fillna(0)
    df_exchanges_piv['Net imports'] =  df_exchanges_piv['Imports'] + df_exchanges_piv['Exports']
    time_index = df_exchanges_piv[['year', 'season', 'day', 't']].drop_duplicates()
    zone_scenario_index = df_exchanges_piv[['zone', 'scenario']].drop_duplicates()
    full_index = zone_scenario_index.merge(time_index, how='cross')
    df_exchanges_piv = full_index.merge(df_exchanges_piv, on=['scenario', 'year', 'season', 'day', 't', 'zone'], how='left')
    df_exchanges_piv['Net imports'] = df_exchanges_piv['Net imports'].fillna(0)
    df_net_imports = df_exchanges_piv.drop(columns=['Imports', 'Exports']).copy()

    df_price = pPrice.copy()

    dfs_to_plot_area = {
        'pDispatchPlant': filter_dataframe(pDispatchPlant, {'attribute': ['Generation']}),
        'pDispatch': filter_dataframe(pDispatch, {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']})
    }
    
    net_exchange = filter_dataframe(pDispatch, {'attribute': ['Exports', 'Imports']})
    net_exchange = net_exchange.set_index(['scenario', 'zone', 'attribute', 'year', 'season', 'day', 't']).squeeze().unstack('attribute')
    # Remove col name
    net_exchange.columns.name = None
    net_exchange['value'] = net_exchange['Exports'] + net_exchange['Imports']
    net_exchange = net_exchange.reset_index()
    net_exchange['attribute'] = 'Net exchange'
    net_exchange = net_exchange.loc[:, pDispatch.columns]

    dfs_to_plot_line = {
        'pDispatch': filter_dataframe(pDispatch, {'attribute': ['Demand']}),
        'pNetExchange': net_exchange,
    }

    seasons = pDispatchPlant.season.unique()
    days = pDispatchPlant.day.unique()

    select_time = {'season': seasons, 'day': days}

    dispatch_plot =  make_dispatch_plot_interactive(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'], zone, year, scenario,
                                                    select_time=select_time, stacked=True,
                                                    reorder_dispatch=['Hydro', 'Solar', 'Wind', 'Nuclear', 'Coal', 'Oil', 'Gas', 'Imports', 'Battery Storage'])

    dfs_to_plot_area = {
    }

    dfs_to_plot_line = {
        'pPrice': df_price.rename(columns={'value': 'price'})
    }

    price_plot = make_dispatch_plot_interactive(dfs_to_plot_area, dfs_to_plot_line, dict_colors=None, zone=zone,
                                                    year=year, scenario=scenario, select_time=select_time, stacked=False,
                                                    ylabel='Price (US $/MWh)')

    dfs_to_plot_area = {
    }

    dfs_to_plot_line = {
        'pNetImportsZoneEvolution': df_net_imports[['year', 'season', 'day', 't', 'zone', 'scenario', 'Net imports']]
    }

    imports_zero = dfs_to_plot_line['pNetImportsZoneEvolution']
    imports_zero = imports_zero.loc[(imports_zero.scenario == scenario) & ((imports_zero.zone == zone)) & (imports_zero.year == year)]
    imports_zero = (imports_zero['Net imports'] == 0).all().all()
    if not imports_zero:  # plotting net imports only when there is some variation
        net_imports_plots = make_dispatch_plot_interactive(dfs_to_plot_area, dfs_to_plot_line, dict_colors=None, zone=zone,
                                                        year=year, scenario=scenario, select_time=select_time, stacked=False,
                                                        ylabel='Net imports (MWh)')

        final_image = combine_and_resize_images([capacity_plot, dispatch_plot, price_plot, net_imports_plots], scale_factor=scale_factor)
    else:
        final_image = combine_and_resize_images([capacity_plot, dispatch_plot, price_plot],
                                                scale_factor=scale_factor)

    # Convert images to base64 and embed in popup
    return f'<br>{final_image}'


def combine_and_resize_images(image_list, scale_factor=0.6):
    """
    Takes a list of base64-encoded images, resizes them to the same width,
    and vertically stacks them before returning as a base64-encoded image.

    Parameters:
    - image_list: List of base64-encoded images
    - scale_factor: Factor to scale down images

    Returns:
    - base64-encoded combined image
    """
    images = []

    # Decode base64 images into PIL images
    for img_str in image_list:
        if img_str:
            img_data = base64.b64decode(img_str.split(",")[1])
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

    if not images:
        return ""

    # Resize all images to the same width
    target_width = max(img.width for img in images)
    resized_images = [img.resize((target_width, int(img.height * (target_width / img.width)))) for img
                      in images]

    # Stack images vertically
    total_height = sum(img.height for img in resized_images)
    final_img = Image.new("RGB", (target_width, total_height), (255, 255, 255))  # White background

    y_offset = 0
    for img in resized_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height

    # Resize the entire combined image
    new_width = int(final_img.width * scale_factor)
    new_height = int(final_img.height * scale_factor)
    final_img = final_img.resize((new_width, new_height))

    # Convert back to base64
    img_io = io.BytesIO()
    final_img.save(img_io, format="PNG")
    img_io.seek(0)
    encoded_str = base64.b64encode(img_io.getvalue()).decode()

    return f'<img src="data:image/png;base64,{encoded_str}" width="{new_width}">'


def make_complete_dispatch_plot_for_interactive(pDispatchFuel, pDispatch, dict_colors, zone, year, scenario,
                                filename=None, BESS_included=True, Hydro_stor_included=True,title='Dispatch',
                                select_time=None, legend_loc='bottom', bottom=0, figsize=(20,6), fontsize=12):
    """
    Generates and saves a dispatch plot for fuel-based generation in a given zone, year, and scenario.

    Parameters
    ----------
    pDispatchFuel : DataFrame
        Dataframe containing dispatch data by fuel type.
    pDispatch : DataFrame
        Dataframe containing total demand and other key dispatch attributes.
    dict_colors : dict
        Dictionary mapping fuel types to colors.
    zone : str
        The zone to visualize.
    year : int
        The target year.
    scenario : str
        The scenario to visualize.
    filename : str, optional
        Path to save the figure, default is None.
    BESS_included : bool, optional
        Whether to include Battery Storage in the dispatch, default is True.
    Hydro_stor_included : bool, optional
        Whether to include Pumped-Hydro Storage, default is True.
    select_time : dict, optional
        Time selection parameters for filtering the data.
    legend_loc : str, optional
        Location of the legend (default is 'bottom').
    bottom : float, optional
        Adjusts bottom margin for better layout (default is 0).
    figsize : tuple, optional
        Size of the figure, default is (20,6).
    fontsize : int, optional
        Font size for labels and titles.

    Returns
    -------
    None
    """

       # Extracting unique seasons and representative days
    dispatch_seasons = list(pDispatchFuel['season'].unique())
    n_rep_days = len(list(pDispatchFuel['day'].unique()))

    # Filtrer les données de production
    pDispatchFuel_zone = pDispatchFuel.loc[
        (pDispatchFuel['zone'] == zone) & (pDispatchFuel['year'] == year) & (pDispatchFuel['scenario'] == scenario)
    ]

    # Exclure les stockages si nécessaire
    if not BESS_included:
        pDispatchFuel_zone = pDispatchFuel_zone[pDispatchFuel_zone['fuel'] != 'Battery Storage']
    if not Hydro_stor_included:
        pDispatchFuel_zone = pDispatchFuel_zone[pDispatchFuel_zone['fuel'] != 'Pumped-Hydro']
    y_max_dispatch = float(pDispatchFuel_zone['value'].max())

    # Mise en forme pour le stacked area plot
    pDispatchFuel_pivot = pDispatchFuel_zone.pivot_table(index=['season', 'day', 't'],
                                                          columns='fuel', values='value', aggfunc='sum')

    # Récupérer la demande
    pDemand_zone = pDispatch.loc[
        (pDispatch['zone'] == zone) & (pDispatch['year'] == year) & (pDispatch['scenario'] == scenario) & (pDispatch['attribute'] == 'Demand')
    ]
    y_max_demand = float(pDemand_zone['value'].max())

    pDemand_pivot = pDemand_zone.pivot_table(index=['season', 'day', 't'], values='value')

    # Extraire les saisons et jours représentatifs
    dispatch_seasons = list(pDispatchFuel['season'].unique())
    n_rep_days = len(list(pDispatchFuel['day'].unique()))

    # Créer le graphique
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

    # Tracer la production en stacked area
    if not pDispatchFuel_pivot.empty:
        pDispatchFuel_pivot.plot.area(ax=ax, stacked=True, linewidth=0, color=[dict_colors.get(fuel, 'gray') for fuel in pDispatchFuel_pivot.columns])

    # Tracer la demande
    if not pDemand_pivot.empty:
        pDemand_pivot.plot(ax=ax, linewidth=1.5, color='darkred', linestyle='-', label='Demand')

    ax.legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adding the representative days as vertical lines
    m = 0
    d_m = 0
    x_ds = []
    for d in range(len(dispatch_seasons) * n_rep_days):
        if d != 0:
            m = m + 1
            d_m = d_m + 1
            x_d = 24 * d - 1
            if m == n_rep_days:
                ax.axvline(x=x_d, color='slategrey', linestyle='-')
                ax.text(x=x_d-12, y=(ax.get_ylim()[1]) * 0.99, s=f'd{str(int(d_m))}', ha='center')
                m = 0
                d_m = 0
            else:
                ax.axvline(x=x_d, color='slategrey', linestyle='--')
                ax.text(x=x_d-12, y=(ax.get_ylim()[1]) * 0.99, s=f'd{str(int(d_m))}', ha='center')
            x_ds = x_ds + [x_d]

    # Adding the last day label
    ax.text(x=x_d+12, y=(ax.get_ylim()[1]) * 0.9, s=f'd{str(int(d_m+1))}', ha='center')
    ax.set_xlabel("")
    ax.set_ylabel('GWh', fontsize=fontsize, fontweight='bold')
    ax.text(0, 1.2, title, fontsize=fontsize, fontweight='bold', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_xticks([24 * n_rep_days * s - 24 * n_rep_days / 2 for s in range(len(dispatch_seasons) + 1)])
    ax.set_xticklabels([''] + [str(s) for s in dispatch_seasons])
    ax.set_xlim(left=0)
    ax.set_ylim(top=max(y_max_dispatch, y_max_demand))

    fig.text(0.5, 0.05, 'Hours', ha='center', fontsize=fontsize, fontweight='bold')

    # Save plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_pie_chart_interactive(df, zone, year, scenario, dict_colors, index='fuel'):
    """
    Generates a pie chart using the existing subplot_pie function and returns it as a base64 image string.
    """

    temp_df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
    if temp_df.empty:
        return ""

    img = BytesIO()

    fig_width = 12
    fig_height = 2  # Shorter height for better fit

    subplot_pie(
        df=temp_df, index=index, dict_colors=dict_colors, title=f'Power mix - {zone} - {year}',
        filename=img, figsize=(fig_width, fig_height), column_subplot='attribute', legend_ncol=1, legend_fontsize=8,
        bbox_to_anchor=(0.9, 0.5), legend=False
    )

    img.seek(0)
    encoded_str = base64.b64encode(img.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded_str}" width="300">'


def make_dispatch_plot_interactive(dfs_area, dfs_line, dict_colors, zone, year, scenario, select_time, stacked=True,
                                       ylabel=None, bottom=None, reorder_dispatch=None):
    """Generates a dispatch plot and returns it as a base64 image string."""
    img = BytesIO()

    fig_width = 14
    fig_height = 4  # Shorter height for better fit

    make_fuel_dispatchplot(
        dfs_area=dfs_area, dfs_line=dfs_line, dict_colors=dict_colors,
        zone=zone, year=year, scenario=scenario, select_time=select_time, filename=img, figsize=(fig_width,fig_height),
        stacked=stacked, ylabel=ylabel, bottom=bottom, reorder_dispatch=reorder_dispatch,
    )

    img.seek(0)
    encoded_str = base64.b64encode(img.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded_str}" width="400">'


def encode_image_from_memory(img):
    """Encodes an in-memory image (BytesIO) to base64 for embedding in HTML."""
    if img is None:
        return ""
    encoded_str = base64.b64encode(img.read()).decode()
    return f'<img src="data:image/png;base64,{encoded_str}" width="300">'


def _resolve_predefined_colors(zone_map, predefined_colors):
    if predefined_colors is not None:
        return predefined_colors
    unique_countries = zone_map['ADMIN'].unique()
    colors = get_extended_pastel_palette(len(unique_countries))
    return {country: colors[i] for i, country in enumerate(unique_countries)}


def _plot_interconnection_map_on_axis(
        ax,
        zone_map,
        transmission_data,
        centers,
        column='value',
        color_col=None,
        show_labels=True,
        label_yoffset=0.02,
        label_xoffset=0.02,
        label_fontsize=12,
        predefined_colors=None,
        min_display_value=100,
        format_y=lambda y, _: '{:.0f} MW'.format(y),
        subplot_title='Transmission capacity',
        show_arrows=False,
        arrow_style='-|>',
        arrow_size=20,
        arrow_offset_ratio=0.1,
        plot_colored_countries=True,
        plot_lines=True,
        offset=0.5,
        mutation_scale=3,
        scale_line_width=lambda _: 1,
        line_color_range=None,
        arrow_color_range=None):
    predefined_colors = _resolve_predefined_colors(zone_map, predefined_colors)

    if isinstance(plot_colored_countries, bool):
        if plot_colored_countries:
            zone_map_plot = zone_map.assign(color=zone_map['ADMIN'].map(predefined_colors))
            zone_map_plot.plot(ax=ax, color=zone_map_plot['color'], edgecolor='black')
        else:
            zone_map.plot(ax=ax, color='white', edgecolor='black')
    else:
        assert isinstance(plot_colored_countries, list), 'plot_colored_countries must be a list or a bool'
        zone_map_plot = zone_map.assign(
            color=zone_map['ADMIN'].apply(
                lambda c: predefined_colors.get(c, 'white') if c in plot_colored_countries else 'white'
            )
        )
        zone_map_plot.plot(ax=ax, color=zone_map_plot['color'], edgecolor='black')

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(subplot_title, loc='center')

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    y_offset = (ymax - ymin) * label_yoffset
    x_offset = (xmax - xmin) * label_xoffset

    if line_color_range is None:
        line_color_range = (0, 1)
    min_color_value, max_color_value = line_color_range
    if max_color_value == min_color_value:
        max_color_value = min_color_value + 1

    if arrow_color_range is not None:
        min_color_value_arrow, max_color_value_arrow = arrow_color_range
        if max_color_value_arrow == min_color_value_arrow:
            max_color_value_arrow = min_color_value_arrow + 1
    else:
        min_color_value_arrow = max_color_value_arrow = None

    if show_labels:
        for zone, coord in centers.items():
            ax.text(
                coord[0] + x_offset,
                coord[1] + y_offset,
                zone.replace('_', ' '),
                fontsize=label_fontsize,
                ha='center',
                va='bottom'
            )

    if transmission_data.empty:
        return

    for _, row in transmission_data.iterrows():
        zone_from, zone_to, value = row['zone_from'], row['zone_to'], row[column]

        if zone_from not in centers or zone_to not in centers:
            continue

        coord_from, coord_to = centers[zone_from], centers[zone_to]
        coor_mid = [(coord_from[0] + coord_to[0]) / 2, (coord_from[1] + coord_to[1]) / 2]

        line_width = scale_line_width(value)
        color = calculate_color_gradient(value, min_color_value, max_color_value)

        if plot_lines:
            ax.plot(
                [coord_from[0], coord_to[0]],
                [coord_from[1], coord_to[1]],
                color=color,
                linewidth=3
            )

            if show_arrows:
                dx = coord_to[0] - coord_from[0]
                dy = coord_to[1] - coord_from[1]
                start_x = coord_from[0] + dx * (0.5 - arrow_offset_ratio)
                start_y = coord_from[1] + dy * (0.5 - arrow_offset_ratio)
                end_x = coord_from[0] + dx * (0.5 + arrow_offset_ratio)
                end_y = coord_from[1] + dy * (0.5 + arrow_offset_ratio)

                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle=arrow_style,
                    color=color,
                    mutation_scale=arrow_size,
                    linewidth=0
                )
                ax.add_patch(arrow)

            if value >= min_display_value:
                ax.text(
                    coor_mid[0],
                    coor_mid[1],
                    format_y(value, None),
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                    fontsize=10
                )
        else:
            dx = coord_to[0] - coord_from[0]
            dy = coord_to[1] - coord_from[1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue

            mid_x = (coord_from[0] + coord_to[0]) / 2
            mid_y = (coord_from[1] + coord_to[1]) / 2

            arrow_offset = arrow_offset_ratio
            start_x = coord_from[0] + dx * arrow_offset
            start_y = coord_from[1] + dy * arrow_offset
            end_x = coord_to[0] - dx * arrow_offset
            end_y = coord_to[1] - dy * arrow_offset

            unit_dx, unit_dy = dx / norm, dy / norm
            norm_dx = -unit_dy
            norm_dy = unit_dx

            if color_col is not None and min_color_value_arrow is not None and max_color_value_arrow is not None:
                color_value = row[color_col]
                color = calculate_color_gradient(color_value, min_color_value_arrow, max_color_value_arrow)
            else:
                color = 'black'

            arrow_linewidth = scale_line_width(value)
            arrowstyle = f"simple,head_length=0.5,head_width=0.5,tail_width={0.2 * arrow_linewidth}"

            arrow = FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle=arrowstyle,
                facecolor=color,
                edgecolor='black',
                linewidth=0,
                mutation_scale=mutation_scale,
                zorder=5
            )
            ax.add_patch(arrow)

            if value >= min_display_value:
                text_x = mid_x + offset * norm_dx
                text_y = mid_y + offset * norm_dy

                ax.text(
                    text_x,
                    text_y,
                    format_y(value, None),
                    ha='center',
                    va='center',
                    fontsize=label_fontsize,
                    fontweight='bold',
                    color='black'
                )


def make_dispatch_value_plot_interactive(df_dispatch, zone, year, scenario, unit_value, title, select_time=None):
 
    img = BytesIO()

    fig_width = 12
    fig_height = 2  # Shorter height for better fit
    fontsize=6

    make_complete_value_dispatch_plot(
        df_dispatch=df_dispatch, zone=zone, year=year, scenario=scenario, 
        unit_value=unit_value, title=title, filename=img, select_time=select_time, 
        figsize=(fig_width, fig_height),fontsize=fontsize
    )

    img.seek(0)
    encoded_str = base64.b64encode(img.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded_str}" width="400">'


def make_interconnection_map(zone_map, df, centers, column='value', color_col=None, filename=None,
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
    predefined_colors = _resolve_predefined_colors(zone_map, predefined_colors)

    transmission_data = df[df[column] > min_capacity]

    if not transmission_data.empty:
        min_cap = transmission_data[column].min()
        max_cap = transmission_data[column].max()
        line_color_range = (min_cap, max_cap)
    else:
        min_cap = max_cap = 1
        line_color_range = (0, 1)

    def scale_line_width(capacity):
        if max_cap == min_cap:
            return min_line_width
        return min_line_width + (capacity - min_cap) / (max_cap - min_cap) * (max_line_width - min_line_width)

    arrow_color_range = None
    if color_col is not None and not transmission_data.empty:
        arrow_values = transmission_data[color_col].dropna()
        if not arrow_values.empty:
            arrow_color_range = (arrow_values.min(), arrow_values.max())

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    _plot_interconnection_map_on_axis(
        ax=ax,
        zone_map=zone_map,
        transmission_data=transmission_data,
        centers=centers,
        column=column,
        color_col=color_col,
        show_labels=show_labels,
        label_yoffset=label_yoffset,
        label_xoffset=label_xoffset,
        label_fontsize=label_fontsize,
        predefined_colors=predefined_colors,
        min_display_value=min_display_value,
        format_y=format_y,
        subplot_title=title,
        show_arrows=show_arrows,
        arrow_style=arrow_style,
        arrow_size=arrow_size,
        arrow_offset_ratio=arrow_offset_ratio,
        plot_colored_countries=plot_colored_countries,
        plot_lines=plot_lines,
        offset=offset,
        mutation_scale=mutation_scale,
        scale_line_width=scale_line_width,
        line_color_range=line_color_range,
        arrow_color_range=arrow_color_range
    )

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_interconnection_map_faceted(
        zone_map,
        df,
        centers,
        column='value',
        color_col=None,
        filename=None,
        min_capacity=0.1,
        figsize=(12, 8),
        show_labels=True,
        label_yoffset=0.02,
        label_xoffset=0.02,
        label_fontsize=12,
        predefined_colors=None,
        min_display_value=100,
        min_line_width=1,
        max_line_width=5,
        format_y=lambda y, _: '{:.0f} MW'.format(y),
        title='Transmission capacity',
        show_arrows=False,
        arrow_style='-|>',
        arrow_size=20,
        arrow_offset_ratio=0.1,
        plot_colored_countries=True,
        plot_lines=True,
        offset=0.5,
        arrow_linewidth=1,
        mutation_scale=3,
        subplotcolumn='year',
        col_wrap=3,
        subplot_order=None):
    """
    Create multiple interconnection maps faceted by the values of `subplotcolumn`.

    Parameters extend `make_interconnection_map` with:
    - subplotcolumn (str): Column used to create subplots.
    - col_wrap (int): Maximum number of subplot columns per row.
    - subplot_order (Sequence, optional): Explicit ordering for facet panels.

    Returns:
    - None
    """
    if subplotcolumn not in df.columns:
        raise ValueError(f"Column '{subplotcolumn}' not found in DataFrame.")

    predefined_colors = _resolve_predefined_colors(zone_map, predefined_colors)

    transmission_data_all = df[df[column] > min_capacity]

    if not transmission_data_all.empty:
        min_cap = transmission_data_all[column].min()
        max_cap = transmission_data_all[column].max()
        line_color_range = (min_cap, max_cap)
    else:
        min_cap = max_cap = 1
        line_color_range = (0, 1)

    def scale_line_width(capacity):
        if max_cap == min_cap:
            return min_line_width
        return min_line_width + (capacity - min_cap) / (max_cap - min_cap) * (max_line_width - min_line_width)

    arrow_color_range = None
    if color_col is not None and not transmission_data_all.empty and color_col in transmission_data_all.columns:
        arrow_values = transmission_data_all[color_col].dropna()
        if not arrow_values.empty:
            arrow_color_range = (arrow_values.min(), arrow_values.max())

    facet_values = df[subplotcolumn].dropna().unique()

    def sort_facet_values(values, reference_series):
        ref_dtype = reference_series.dtype
        if pd.api.types.is_categorical_dtype(ref_dtype):
            return [value for value in reference_series.cat.categories if value in values]

        numeric_coerced = pd.to_numeric(pd.Series(values), errors='coerce')
        if not numeric_coerced.isna().any():
            return [val for _, val in sorted(zip(numeric_coerced.tolist(), values))]

        datetime_coerced = pd.to_datetime(pd.Series(values), errors='coerce')
        if not datetime_coerced.isna().any():
            return [val for _, val in sorted(zip(datetime_coerced.tolist(), values))]

        try:
            return sorted(values)
        except TypeError:
            return list(values)

    if subplot_order is not None:
        missing = [value for value in subplot_order if value not in facet_values]
        if missing:
            raise ValueError(f"Values {missing} in 'subplot_order' not found in '{subplotcolumn}'.")
        ordered_values = [value for value in subplot_order if value in facet_values]
        leftover = [value for value in facet_values if value not in ordered_values]
        if leftover:
            ordered_values.extend(sort_facet_values(list(leftover), df[subplotcolumn]))
    elif pd.api.types.is_categorical_dtype(df[subplotcolumn]):
        ordered_values = [value for value in df[subplotcolumn].cat.categories if value in facet_values]
    else:
        ordered_values = pd.unique(facet_values)
        if len(ordered_values) > 1:
            ordered_values = sort_facet_values(list(ordered_values), df[subplotcolumn])

    if len(ordered_values) == 0:
        raise ValueError(f"No data available to facet by '{subplotcolumn}'.")

    if col_wrap is None or col_wrap <= 0:
        col_wrap = len(ordered_values)

    ncols = min(col_wrap, len(ordered_values))
    nrows = int(np.ceil(len(ordered_values) / ncols))

    width_total, height_total = figsize
    base_cols = max(1, min(col_wrap, len(ordered_values)))
    width_per_subplot = width_total / base_cols
    height_per_subplot = height_total / max(1, nrows)

    fig_width = width_per_subplot * ncols
    fig_height = height_per_subplot * nrows

    use_constrained_layout = True
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        constrained_layout=use_constrained_layout
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, value in enumerate(ordered_values):
        ax = axes[idx]
        subset = df[df[subplotcolumn] == value]
        transmission_subset = subset[subset[column] > min_capacity]

        _plot_interconnection_map_on_axis(
            ax=ax,
            zone_map=zone_map,
            transmission_data=transmission_subset,
            centers=centers,
            column=column,
            color_col=color_col,
            show_labels=show_labels,
            label_yoffset=label_yoffset,
            label_xoffset=label_xoffset,
            label_fontsize=label_fontsize,
            predefined_colors=predefined_colors,
            min_display_value=min_display_value,
            format_y=format_y,
            subplot_title=str(value),
            show_arrows=show_arrows,
            arrow_style=arrow_style,
            arrow_size=arrow_size,
            arrow_offset_ratio=arrow_offset_ratio,
            plot_colored_countries=plot_colored_countries,
            plot_lines=plot_lines,
            offset=offset,
            mutation_scale=mutation_scale,
            scale_line_width=scale_line_width,
            line_color_range=line_color_range,
            arrow_color_range=arrow_color_range
        )

    for idx in range(len(ordered_values), len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, y=0.98)
        if not use_constrained_layout:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        if not use_constrained_layout:
            fig.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches='tight')
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
                           dict_specs, pCapacityFuel, pEnergyFuel, pDispatch, pDispatchPlant, pPrice, label_size=14):
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
    - pCapacityFuel (DataFrame): Capacity mix data
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
            popup_content += generate_zone_plots(zone, year, scenario, dict_specs, pCapacityFuel, pEnergyFuel, pDispatch,
                                                pDispatchPlant, pPrice, scale_factor=0.8)

            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, min_width=800, max_height=700),
                icon=folium.Icon(color='blue', icon="")
            ).add_to(energy_map)

    # Fit map to bounds
    energy_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Save the map
    energy_map.save(filename)


def make_automatic_map(epm_results, dict_specs, folder, FIGURES_ACTIVATED, selected_scenarios=None):
    # TODO: ongoing work
    def keep_max_direction(df):
        # Make sure zone names are consistent strings
        df_grouped = df.copy()
        df_grouped['zone'] = df_grouped['zone'].astype(str)
        df_grouped['z2'] = df_grouped['z2'].astype(str)

        # Create a canonical pair identifier (sorted zones)
        df_grouped['zone_pair'] = df_grouped.apply(lambda row: tuple(sorted([row['zone'], row['z2']])), axis=1)

        df_grouped = df_grouped.sort_values('value', ascending=False)
        # Group by scenario, year, and zone_pair
        # df_grouped = df_grouped.sort_values('value', ascending=False).groupby(['scenario', 'year', 'zone_pair'], as_index=False)['value'].sum()
        df_sum = df_grouped.groupby(['scenario', 'year', 'zone_pair'], as_index=False)['value'].sum()

        df_grouped = df_grouped.groupby(['scenario', 'year', 'zone_pair'], as_index=False).first()  # this line is used to keep the direction corresponding to the maximum utilization

        df_sum = df_sum.merge(df_grouped[['scenario', 'year', 'zone_pair', 'zone', 'z2']], on=['scenario', 'year', 'zone_pair'], how='left')
        # Drop the helper column
        df_sum = df_sum.drop(columns='zone_pair')

        return df_sum
 
    if selected_scenarios is None:
        selected_scenarios = list(epm_results['pAnnualTransmissionCapacity'].scenario.unique())

    years = epm_results['pAnnualTransmissionCapacity']['year'].unique()

    # One figure per scenario
    for selected_scenario in selected_scenarios:
        print(f'Generating map for scenario {selected_scenario}')
        # Select first and last years
        #years = [min(years), max(years)]
        years = [y for y in [2025, 2030, 2035, 2040] if y in years]

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
        # We sum utilization across both directions, and keep the direction with the maximum utilization (for arrows on graph)
        utilization_transmission_max = keep_max_direction(utilization_transmission)
        transmission_data = capa_transmission.rename(columns={'value': 'capacity'}).merge(
            utilization_transmission_max.rename
            (columns={'value': 'utilization'}), on=['scenario', 'zone', 'z2', 'year'])
        transmission_data = transmission_data.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

        figure_name = 'TransmissionCapacityMapEvolution'
        if FIGURES_ACTIVATED.get(figure_name, False):
            title = f'Evolution Transmission Capacity [MW] - {selected_scenario}'
            filename = os.path.join(folder, f'{figure_name}_{selected_scenario}.pdf')
            
            selected_years = [2025, 2035, 2040]
            df = transmission_data[
                    (transmission_data['scenario'] == selected_scenario) &
                    (transmission_data['year'].isin(selected_years))
                    ]
            
            make_interconnection_map_faceted(zone_map, df, centers, title=title, column='capacity',
                                    label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=False,
                                    min_display_value=50, filename=filename, subplotcolumn='year', col_wrap=3)
        
        figure_name = 'TransmissionUtilizationMapEvolution'
        if FIGURES_ACTIVATED.get(figure_name, False):
            title = f'Evolution Transmission Utilization [%] - {selected_scenario}'
            filename = os.path.join(folder, f'{figure_name}_{selected_scenario}.pdf')
            
            selected_years = [2025, 2035, 2040]
            df = transmission_data[
                    (transmission_data['scenario'] == selected_scenario) &
                    (transmission_data['year'].isin(selected_years))
                    ]
            
            make_interconnection_map_faceted(zone_map, df, centers, title=title, column='utilization',
                                    label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=False,
                                    min_display_value=10, filename=filename, subplotcolumn='year', col_wrap=3,
                                    format_y=lambda y, _: '{:.0f} %'.format(y), show_arrows=True, arrow_offset_ratio=0.4,
                                    arrow_size=25)
        
        
        
        for year in years:

            if not transmission_data.loc[transmission_data.scenario == selected_scenario].empty: 
   
                # Filter data for the given year and scenario
                df = transmission_data[
                    (transmission_data['year'] == year) &
                    (transmission_data['scenario'] == selected_scenario)
                    ]
   
                figure_name = 'TransmissionCapacityMap'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    
                    title = f'Transmission capacity [MW] - {selected_scenario} - {year}'
                    filename = os.path.join(folder, f'{figure_name}_{selected_scenario}_{year}.pdf')
                    make_interconnection_map(zone_map, df, centers, title=title, column='capacity',
                                            label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=False,
                                            min_display_value=50, filename=filename)

                figure_name = 'TransmissionUtilizationMap'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    title = f'Transmission utilization [%] - {selected_scenario} - {year}'
                    filename = os.path.join(folder, f'{figure_name}_{selected_scenario}_{year}.pdf')
                    make_interconnection_map(zone_map, df, centers,
                                            column='utilization',
                                            min_capacity=0.01, label_yoffset=0.01, label_xoffset=-0.05,
                                            label_fontsize=10, show_labels=False, min_display_value=20,
                                            format_y=lambda y, _: '{:.0f} %'.format(y), filename=filename,
                                            title=title, show_arrows=True, arrow_offset_ratio=0.4,
                                            arrow_size=25, plot_colored_countries=True)
             
                figure_name = 'NetExportsMap'
                if FIGURES_ACTIVATED.get(figure_name, False):
                    title = f'Net Exports [GWh] - {selected_scenario} - {year}'
                    filename = os.path.join(folder, f'{figure_name}_{selected_scenario}_{year}.pdf')

                    
                    tmp = epm_results['pInterchange'].copy()
                    df_congested = epm_results['pCongestionShare'].copy().rename(columns={'value': 'congestion'})
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

                    make_interconnection_map(zone_map, df_net, centers, filename=filename,
                                            title=title,
                                            label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=False,
                                            plot_colored_countries=True,
                                            min_display_value=100, column='value', plot_lines=False,
                                            format_y=lambda y, _: '{:.0f}'.format(y), offset=-1.5,
                                            min_line_width=0.7, max_line_width=1.5, arrow_linewidth=0.1, mutation_scale=20,
                                            color_col='congestion')

            if len(epm_results['pEnergyBalance'].loc[(epm_results['pEnergyBalance'].scenario == selected_scenario)].zone.unique()) > 1:  # only plotting on interactive map when more than one zone
                    
                    figure_name = 'InteractiveMap'
                    if FIGURES_ACTIVATED.get(figure_name, False):
                        filename = os.path.join(folder, f'{figure_name}_{selected_scenario}_{year}.html')

                        # Add utilization rate
                        capa_transmission = capa_transmission.rename(columns={'value': 'capacity'})
                        utilization_transmission = utilization_transmission.rename(columns={'value': 'utilization'})
                        transmission_data = capa_transmission.merge(utilization_transmission, 
                                                                    on=['scenario', 'zone', 'z2', 'year'],
                                                                    how='left').fillna({'utilization': 0})
                        
                        transmission_data = transmission_data.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

                        create_interactive_map(zone_map, centers, transmission_data, epm_results['pEnergyBalance'], year, selected_scenario, filename,
                                            dict_specs, epm_results['pCapacityFuel'], epm_results['pEnergyFuel'], epm_results['pDispatch'], epm_results['pDispatchPlant'], epm_results['pPrice'])

