"""
This module contains utility functions for the project.
"""
import cartopy.io.shapereader as shpreader
import pandas as pd
import xarray as xr
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import GeoJson, GeoJsonTooltip
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_bbox(ISO_A2):
    """
    Get the bounding box of a country based on its ISO 3166-1 alpha-2 code.

    Args:
        ISO_A2 (str): The ISO 3166-1 alpha-2 code of the country.

    """
    shp = shpreader.Reader(
        shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
    )
    de_record = list(filter(lambda c: c.attributes["ISO_A2"] == ISO_A2, shp.records()))[0]
    de = pd.Series({**de_record.attributes, "geometry": de_record.geometry})
    x_west, y_south, x_east, y_north = de["geometry"].bounds
    return x_west, y_south, x_east, y_north


def read_grib_file(grib_path, step_type=None):
    """
    Reads a GRIB file and returns an xarray dataset.
    This function uses the cfgrib engine to read the GRIB file and filter by stepType.
    It merges the datasets for 'avgid' and 'avgas' step types.
    :param grib_path: Path to the GRIB file.
    :return: An xarray dataset containing the data from the GRIB file.
    """
    print(f"Opening GRIB file: {grib_path}")
    try:
        if step_type:
            ds_avgid = xr.open_dataset(grib_path, engine='cfgrib',
                                       backend_kwargs={"filter_by_keys": {"stepType": "avgid"}},
                                       decode_timedelta=True)
            ds_avgas = xr.open_dataset(grib_path, engine="cfgrib",
                                       backend_kwargs={"filter_by_keys": {"stepType": "avgas"}},
                                       decode_timedelta=True)
            dataset = xr.merge([ds_avgid, ds_avgas], compat='override')
        else:
            dataset = xr.open_dataset(grib_path, engine='cfgrib',
                                       decode_timedelta=True)

        #print(dataset)  # or process the dataset as needed
    except Exception as e:
        print(f"Failed to open {grib_path}: {e}")

    return dataset


def extract_data(zip_path, step_type=None, extract_to='era5_extracted_files'):
    """
    Extracts GRIB files from a ZIP file and reads them into xarray datasets.

    :param zip_path: Path to the ZIP file containing GRIB files.
    :return: A list of xarray datasets created from the GRIB files.
    """

    # Check if the ZIP file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Create the extraction directory if it does not exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Extract files from zip file
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]

    # Extract all files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Loop through extracted files and open GRIB files
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.grib') or file.endswith('.grb'):
                grib_path = os.path.join(root, file)

                # Rename GRIB file using ZIP file name
                new_grib_name = f"{zip_name}.grib"
                new_grib_path = os.path.join(root, new_grib_name)

                # Avoid overwriting if already renamed
                if grib_path != new_grib_path:
                    os.rename(grib_path, new_grib_path)
                    grib_path = new_grib_path  # update path to the renamed file

                dataset = read_grib_file(grib_path, step_type=step_type)

    return dataset


def convert_dataset_units(ds):
    """
    Convert ERA5-Land variables in a Dataset to commonly used units.

    Parameters:
        ds (xarray.Dataset): The input dataset with ERA5-Land variables.
        output_md_path (str): File path to save the Markdown table.

    Returns:
        ds_converted (xarray.Dataset): Dataset with converted units.
        output_md_path (str): Path to the Markdown file.
    """
    conversions = {
        "t2m": {"description": "2m temperature", "original_unit": "K", "factor": 1, "offset": -273.15, "new_unit": "degC"},
        "tp": {"description": "Total precipitation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "ro": {"description": "Runoff", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sro": {"description": "Surface runoff", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "pev": {"description": "Potential evaporation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "e": {"description": "Total evaporation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sd": {"description": "Snow depth water equivalent", "original_unit": "m", "factor": 1000, "offset": 0, "new_unit": "mm"}
    }

    ds_converted = ds.copy()
    for var, conv in conversions.items():
        if var in ds_converted.data_vars:
            ds_converted[var] = ds_converted[var] * conv["factor"] + conv["offset"]
            ds_converted[var].attrs["units"] = conv["new_unit"]
    return ds_converted


def plot_mean_map(ds, var, folder=None):
    """
    Plot the mean of a variable over time.
    :param ds: xarray dataset
    :param var: variable name to plot
    :return: None
    """
    mean_field = ds[var].mean(dim='time')
    plt.figure(figsize=(8, 6))
    mean_field.plot()
    plt.title(f"Mean {var.upper()} Over Time")
    plt.xlabel("lon_name")
    plt.ylabel("lat_name")
    if folder:
        plt.savefig(os.path.join(folder, f"mean_{var}.jpg"), dpi=300)
        plt.close()

    else:
        plt.show()


def plot_monthly_climatology_grid(ds, var, filename=None):
    """
    Plot the monthly climatology of a variable in a grid format.
    :param ds:
    :param var:
    :return:
    """
    # Group by month and average over years
    monthly_clim = ds[var].groupby('time.month').mean(dim='time')

    # Calculate shared color scale
    vmin = monthly_clim.min().item()
    vmax = monthly_clim.max().item()

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        im = monthly_clim.sel(month=i+1).plot(
            ax=ax,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Month {i+1}")
        ax.set_xlabel("")
        ax.set_ylabel("")

        #ax.coastlines()
        #ax.add_feature(cfeature.BORDERS)

    # Add a single shared colorbar for all axes
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.9)
    cbar.set_label(ds[var].attrs.get('units', var))



    fig.suptitle(f"Interannual Monthly Mean of {var.upper()}", fontsize=16)
    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for suptitle and colorbar
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_spatial_mean_timeseries_all_vars(ds, lat_name='latitude', lon_name='longitude', folder=None):
    """
    Plot the spatial mean time series for all variables in the dataset.
    :param ds:
    :return:
    """
    plt.figure(figsize=(12, 6))

    for var in ds.data_vars:
        # Calculate spatial average over lat/lon
        spatial_mean = ds[var].mean(dim=[lat_name, lon_name])

        # Get units if available
        units = ds[var].attrs.get('units', '')
        label = f"{var.upper()} ({units})" if units else var.upper()

        # Plot it
        plt.plot(ds.time, spatial_mean, label=label)

    plt.title("Spatial Mean Time Series for All Variables")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if folder:
        plt.savefig(os.path.join(folder, "spatial_mean_timeseries_all_vars.png.jpg"), dpi=300)
        plt.close()
    else:
        plt.show()

def scatter_annual_spatial_means(data, var_x='t2m', var_y='tp', lat_name='latitude', lon_name='longitude', folder=None):
    """
    Scatter plot of annual spatial means for two variables across multiple datasets.
    """
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X']
    colors = plt.cm.tab10.colors  # 10 distinct colors

    plt.figure(figsize=(10, 6))

    for i, (iso, ds) in enumerate(data.items()):
        # Compute spatial mean for each time step
        x_spatial = ds[var_x].mean(dim=[lat_name, lon_name])
        y_spatial = ds[var_y].mean(dim=[lat_name, lon_name])

        # Group by year and compute annual means
        x_annual = x_spatial.groupby('time.year').mean()
        y_annual = y_spatial.groupby('time.year').mean()

        # Plot each year for this ISO
        plt.scatter(
            x_annual.values,
            y_annual.values,
            label=iso,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            edgecolor='black'
        )

    # Axis labels
    plt.xlabel(f"{var_x.upper()} ({ds[var_x].attrs.get('units', '')})")
    plt.ylabel(f"{var_y.upper()} ({ds[var_y].attrs.get('units', '')})")
    plt.title(f"Annual Spatial Mean: {var_x.upper()} vs {var_y.upper()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if folder:
        plt.savefig(os.path.join(folder, f'scatter_annual_spatial_means_{var_x}_{var_y}.pdf'), dpi=300)
        plt.close()
    else:
        plt.show()


def plot_spatial_mean_timeseries_all_iso(dataset, var='tp', agg=None, folder=None):
    """
    Plot the temporal evolution of the spatial mean for a given variable across multiple datasets.
    This function takes a dictionary of datasets, extracts the specified variable, computes the spatial mean,
    and plots the temporal evolution of that mean for each dataset.


    :param dataset:
        A dictionary where keys are ISO A2 codes and values are xarray datasets containing the variable of interest.
    :param var:
        The variable to be plotted. Default is 'tp' (total precipitation).
    :param agg: None (monthly), 'avg' (annual mean), or 'sum' (annual sum)
    :param folder:
        The folder where the output plot will be saved. Default is 'output'.
    :return:
    """


    lat_name, lon_name = 'latitude', 'longitude'
    plt.figure(figsize=(12, 6))

    for iso, ds in dataset.items():
        da = ds[var]

        # Mean over space
        spatial_mean = da.mean(dim=[lat_name, lon_name])

        # Optional temporal aggregation
        if agg == 'avg':
            # Average by year
            spatial_mean = spatial_mean.groupby('time.year').mean()
            time_axis = spatial_mean['year'].values
        elif agg == 'sum':
            # Sum by year
            spatial_mean = spatial_mean.groupby('time.year').sum()
            time_axis = spatial_mean['year'].values
        else:
            # Keep original time resolution
            time_axis = ds['time'].values

        # Plot
        plt.plot(time_axis, spatial_mean, label=f"{iso} - mean")

    plt.title(f"Temporal Evolution of Spatial Mean of {var.upper()}")
    plt.xlabel("Time")
    plt.ylabel(ds[var].attrs.get("units", var))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    if folder is not None:
        plt.savefig(os.path.join(folder, f'spatial_mean_{var}_{agg}.pdf'))
        plt.close()
    else:
        plt.show()

def plot_monthly_mean(dataset, var, lat_name='latitude', lon_name='longitude', folder=None):
 # Example variable to plot
    n = len(dataset)
    fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, sharey=True)

    if n == 1:
        axs = [axs]

    for ax, (iso, ds) in zip(axs, dataset.items()):
        da = ds[var]
        spatial_mean = da.mean(dim=[lat_name, lon_name])

        # Count months per year
        months_per_year = spatial_mean.groupby('time.year').count()

        # Keep only years with 12 months
        complete_years = months_per_year.where(months_per_year == 12, drop=True)['year'].values

        # Filter the time series to keep only complete years
        spatial_mean_complete = spatial_mean.where(spatial_mean['time.year'].isin(complete_years), drop=True)

        # Recompute group-by with only complete years
        by_year = spatial_mean_complete.groupby('time.year')
        annual_totals = by_year.sum()

        # Find year with max and min total
        max_year = annual_totals['year'].values[annual_totals.argmax().item()]
        min_year = annual_totals['year'].values[annual_totals.argmin().item()]

        # Group by month and take mean across all years
        monthly_mean = spatial_mean.groupby('time.month').mean()

        # Extract monthly series for max and min year
        max_year_monthly = spatial_mean.where(spatial_mean['time.year'] == max_year, drop=True).groupby(
            'time.month').mean()
        min_year_monthly = spatial_mean.where(spatial_mean['time.year'] == min_year, drop=True).groupby(
            'time.month').mean()

        # Plotting
        ax.plot([pd.Timestamp(f'2025-{m:02d}-01').strftime('%b') for m in monthly_mean['month'].values], monthly_mean,
                label='Mean (all years)', marker='o')
        ax.plot([pd.Timestamp(f'2025-{m:02d}-01').strftime('%b') for m in max_year_monthly['month'].values],
                max_year_monthly, label=f'Max Year ({max_year})', linestyle='--', marker='^')
        ax.plot([pd.Timestamp(f'2025-{m:02d}-01').strftime('%b') for m in min_year_monthly['month'].values],
                min_year_monthly, label=f'Min Year ({min_year})', linestyle=':', marker='v')

        ax.set_title(f"{iso} - Monthly Average of {var.upper()}")
        ax.set_ylabel(ds[var].attrs.get('units', var))
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Month")
    plt.tight_layout()
    if folder is not None:
        plt.savefig(os.path.join(folder, f'monthly_mean_{var}.pdf'))
        plt.close()
    else:
        plt.show()

def calculate_resolution_netcdf(dataset, lon_name='X', lat_name='Y'):
    """
    Calculate spatial resolution (in degrees and km) from a netCDF-like xarray.Dataset.

    Parameters:
        dataset: xarray.Dataset
            The dataset to analyze
        lon_name: str
            Name of the lon_name coordinate
        lat_name: str
            Name of the lat_name coordinate

    Returns:
        dict: resolution information including degrees and approximate km
    """
    # Spatial resolution in degrees
    dx_deg = float(dataset[lon_name][1] - dataset[lon_name][0])
    dy_deg = float(dataset[lat_name][1] - dataset[lat_name][0])

    # Approximate midpoint lat_name for scaling lon_name to km
    mid_lat = float(dataset[lat_name].mean())


    # Constants
    km_per_deg_lat = 111  # approximate
    km_per_deg_lon = 111 * np.cos(np.deg2rad(mid_lat))

    # Convert to km
    dx_km = dx_deg * km_per_deg_lon
    dy_km = dy_deg * km_per_deg_lat

    print(f"Spatial resolution: {dx_deg}° lon x {dy_deg}° lat")
    print(f"Approximate spatial resolution:")
    print(f"{dx_km:.2f} km (lon_name) x {dy_km:.2f} km (lat_name) at {mid_lat:.2f}° lat")

    # Temporal resolution (assuming consistent intervals)
    dt = dataset.time[1] - dataset.time[0]
    temporal_resolution_days = pd.to_timedelta(dt.values).days
    print(f"Temporal resolution: {temporal_resolution_days} days")

    return {
        "dx_deg": dx_deg,
        "dy_deg": dy_deg,
        "dx_km": dx_km,
        "dy_km": dy_km,
        "mid_lat": mid_lat
    }


def check_spatial_overlap(data_clim, geometry, lon_name='X', lat_name='Y', mode='total'):
    """
    Check if a shapefile or a single geometry overlaps with a climate DataArray.

    :param data_clim: xarray DataArray
    :param geometry: GeoDataFrame (for mode='total') or shapely geometry (for mode='row')
    :param lon_name: str, name of lon_name coordinate in DataArray
    :param lat_name: str, name of lat_name coordinate in DataArray
    :param mode: 'total' for full shapefile, 'row' for a single geometry
    :return: bool, True if overlap exists
    """
    # Get bounds from DataArray
    da_lon_min = float(data_clim[lon_name].min())
    da_lon_max = float(data_clim[lon_name].max())
    da_lat_min = float(data_clim[lat_name].min())
    da_lat_max = float(data_clim[lat_name].max())

    print("\n📦 DataArray bounds:")
    print(f"  lon_name ({lon_name}): {da_lon_min:.4f} to {da_lon_max:.4f}")
    print(f"  lat_name ({lat_name}):  {da_lat_min:.4f} to {da_lat_max:.4f}")

    # Get geometry bounds
    if mode == 'total':
        bounds = geometry.total_bounds  # (minx, miny, maxx, maxy)
    elif mode == 'row':
        bounds = geometry.bounds  # shapely object
    else:
        raise ValueError("mode must be either 'total' or 'row'")

    g_lon_min, g_lat_min, g_lon_max, g_lat_max = bounds

    print(f"  lon_name: {g_lon_min:.4f} to {g_lon_max:.4f}")
    print(f"  lat_name:  {g_lat_min:.4f} to {g_lat_max:.4f}")

    # Check overlap
    overlap_lon = (g_lon_min <= da_lon_max) and (g_lon_max >= da_lon_min)
    overlap_lat = (g_lat_min <= da_lat_max) and (g_lat_max >= da_lat_min)

    if overlap_lon and overlap_lat:
        print("\n✅ Overlap detected.\n")
        return True
    else:
        print("\n❌ No overlap detected.\n")
        return False


def calculate_spatial_mean_annual(data_climatic, gdf_regions, lat_name='Y', lon_name='X'):
    """
    Calculate spatial and yearly mean of a climate variable per region.

    :param data_climatic: xarray.DataArray
        A 3D DataArray (time, lat, lon) containing the climate variable.
    :param gdf_regions: GeoDataFrame
        A GeoDataFrame with polygons (geometry) and unique ID per region (e.g., "region", "station", etc.)
    :param var_name: str
        Name to assign to the climate variable in output
    :return: DataFrame
        Multi-index DataFrame with time and region ID, showing the monthly and annual mean
    """

    # Ensure spatial metadata is set
    data_climatic = data_climatic.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    data_climatic = data_climatic.rio.write_crs("EPSG:4326", inplace=False)

    if not gdf_regions.crs:
        gdf_regions = gdf_regions.set_crs("EPSG:4326")
    else:
        gdf_regions = gdf_regions.to_crs("EPSG:4326")


    check_spatial_overlap(data_climatic, gdf_regions, lon_name=lon_name, lat_name=lat_name, mode='total')
    results = []

    for i, row in gdf_regions.iterrows():
        region_id = row.get("station", f"region_{i}")
        print(f'Region {region_id}')
        if check_spatial_overlap(data_climatic, row.geometry, mode='row', lon_name=lon_name, lat_name=lat_name):
            # proceed with clipping
            clipped = data_climatic.rio.clip([row.geometry.__geo_interface__], gdf_regions.crs, all_touched=True)
        else:
            print(f'No overlap for region: {region_id}')

        # Mean over space (lat, lon) -> results in time series
        regional_mean = clipped.mean(dim=[lat_name, lon_name])
        regional_mean = regional_mean.expand_dims({"region": [region_id]})
        results.append(regional_mean)

    # Combine into one dataset (dims: time, region)
    combined = xr.concat(results, dim="region")

    # Convert to DataFrame
    df = combined.to_dataframe().reset_index()

    return df

def convert_to_yearly_mm_year(df, var_name="Runoff", unit_init="mm/day"):
    """
    Convert monthly values in mm/day to yearly total in mm/year.

    :param df: DataFrame with columns ['time', 'region', var_name] where time is datetime-like.
    :param var_name: Name of the variable column (e.g., 'Runoff').
    :return: DataFrame with yearly runoff totals in mm/year per region.
    """

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    if unit_init == "mm/day":
        df["days_in_month"] = df["time"].dt.days_in_month
        # mm/month = mm/day * days in month
        df["monthly_total_mm"] = df[var_name] * df["days_in_month"]
    elif unit_init == "mm/month":
        df["monthly_total_mm"] = df[var_name]
    else:
        raise ValueError("unit_init must be either 'mm/day' or 'mm/month'")

    # Sum over year per region
    df_yearly = (
        df.groupby(["year", "region"])["monthly_total_mm"]
        .sum()
        .reset_index()
        .rename(columns={"monthly_total_mm": f"{var_name}_mm_per_year"})
    )

    return df_yearly


def map_grdc_stationbasins_and_subregions(folder, file_stationbasins, file_subregions=None):
    # Load your files
    stationbasins = gpd.read_file(file_stationbasins)
    if file_subregions is not None:
        subregions = gpd.read_file(file_subregions)

        # Ensure CRS is consistent
        if stationbasins.crs != subregions.crs:
            stationbasins = stationbasins.to_crs(subregions.crs)

    # Calculate map center
    map_center = stationbasins.geometry.unary_union.centroid.coords[0][::-1]

    # Initialize folium map
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
    if file_subregions is not None:
        # Add subregions layer
        folium.GeoJson(
            subregions,
            name="Subregions",
            style_function=lambda feature: {
                "color": "blue",
                "weight": 1.5,
                "fillOpacity": 0.1,
            },
            tooltip=GeoJsonTooltip(fields=subregions.columns[:2].tolist(), aliases=["SUBREGNAME", "RIVERBASIN"])
            # customize fields
        ).add_to(m)

    # Add stations (points or polygons) layer
    folium.GeoJson(
        stationbasins,
        name="Station Catchments",
        style_function=lambda feature: {
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.3,
        },
        tooltip=GeoJsonTooltip(
            fields=["station", "river", "area"],
            aliases=["Station ID", "River", "Area (km²)"],
            sticky=True
        )
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save and display
    m.save(os.path.join(folder, "map_stations_and_subregions.html"))
    print("Map saved to map_stations_and_subregions.html")



def get_common_dataframes(data_dict):
    """
    Align multiple DataFrames on shared rows (index) and columns.

    Parameters:
        data_dict (dict): Dictionary with names as keys and pandas DataFrames as values.
                          Each DataFrame should have the same structure: rows = years, columns = stations.

    Returns:
        - aligned_data (dict): Dictionary of filtered DataFrames with only common years and stations.
        - unmatched_columns (dict): Dictionary of unmatched station names per source.
    """
    # Get the set of common years (index) and common stations (columns)
    common_index = set.intersection(*[set(df.index) for df in data_dict.values()])
    common_columns = set.intersection(*[set(df.columns) for df in data_dict.values()])

    print(f"✅ Common years: {len(common_index)}")
    print(f"✅ Common stations: {len(common_columns)}")

    # Filter all dataframes
    aligned_data = {
        name: df.loc[sorted(common_index), sorted(common_columns)]
        for name, df in data_dict.items()
    }

    # Report stations that are missing in each dataset
    unmatched_columns = {
        name: set(df.columns) - common_columns
        for name, df in data_dict.items()
    }

    for name, unmatched in unmatched_columns.items():
        print(f"📌 Stations only in {name}: {unmatched}")

    return aligned_data, unmatched_columns


def plot_station_comparison_dict(data_dict, station, save_path=None):
    """
    Plot runoff time series from multiple datasets for a single station.

    Parameters:
        data_dict (dict): keys are dataset names (e.g., 'GRUN', 'GRDC'),
                          values are DataFrames with years as index and station names as columns
        station (str): station name to plot
        save_path (str or None): if provided, saves plot to file; else displays it
    """
    plt.figure(figsize=(10, 4))

    # Loop through all datasets and plot the station
    for label, df in data_dict.items():
        if station in df.columns:
            plt.plot(df.index, df[station], label=label, marker='o')
        else:
            print(f"⚠️ Station '{station}' not found in dataset '{label}' — skipping.")

    plt.title(f"Runoff Comparison for Station: {station}")
    plt.xlabel("Year")
    plt.ylabel("Runoff (mm/year)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_river_discharge_by_year(gdf, year, cmap='viridis_r', figsize=(12, 10), folder=None, bbox=None):
    """
    Plot river discharge values for a given year from a GeoDataFrame.

    Used with VegDischarge data.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame with river geometries and annual discharge columns.
        year (int or str): The year to visualize (must match column name, e.g. 2020).
        cmap (str): Colormap to use (default 'viridis').
        figsize (tuple): Figure size.
    """
    year = str(year)
    file_name = f"discharge_{year}"

    if year not in gdf.columns:
        raise ValueError(f"Year {year} not found in columns: {gdf.columns.tolist()}")

    if bbox:
        long_west, lat_south, long_east, lat_north = bbox
        gdf = gdf.cx[long_west:long_east, lat_south:lat_north]
        file_name = f"{file_name}_{long_west}_{lat_south}_{long_east}_{lat_north}"

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(column=year, cmap=cmap, linewidth=1.2, legend=True, ax=ax)

    ax.set_title(f"River Discharge - {year}", fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()
    if folder:
        plt.savefig(os.path.join(folder, f"{file_name}.png"), dpi=300)
        plt.close()
    else:
        plt.show()