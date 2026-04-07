"""
Script to download PV generation data from PVGIS and wind generation data from ERA5 via CDS API
and format it for EPM representative days pipeline.

Alternative to Renewables.ninja data source.

Requirements:
pip install pvlib cdsapi xarray netcdf4 requests openpyxl

For CDS API:
1. Register at https://cds.climate.copernicus.eu/
2. Get API key and save to ~/.cdsapirc or set environment variables:
   export CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
   export CDSAPI_KEY=your_key_here

For Windows, create .cdsapirc file in user home directory with:
url: https://cds.climate.copernicus.eu/api/v2
key: your_key_here

Usage:
python generate_pvgis_era5_data.py --countries "Spain,France,Germany" --year 2018
"""

import argparse
import os
import calendar
import tempfile
from typing import List, Tuple

import warnings
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
import cdsapi
import xarray as xr
import requests

sadc_countries = {
    "South Africa":  {"lat": -28.48, "lon": 24.67, "tilt": 28},
    "Mozambique":    {"lat": -18.67, "lon": 35.53, "tilt": 19},
    "Tanzania":      {"lat":  -6.37, "lon": 34.89, "tilt":  6},
    "Zambia":        {"lat": -13.13, "lon": 27.85, "tilt": 13},
    "Zimbabwe":      {"lat": -19.02, "lon": 29.15, "tilt": 19},
    "Namibia":       {"lat": -22.96, "lon": 18.49, "tilt": 23},
    "Botswana":      {"lat": -22.33, "lon": 24.68, "tilt": 22},
    "Angola":        {"lat": -11.20, "lon": 17.87, "tilt": 11},
    "DRC":           {"lat":  -4.04, "lon": 21.76, "tilt":  4},
    "Madagascar":    {"lat": -18.77, "lon": 46.87, "tilt": 19},
    "Malawi":        {"lat": -13.25, "lon": 34.30, "tilt": 13},
    "Lesotho":       {"lat": -29.61, "lon": 28.23, "tilt": 30},
    "Eswatini":      {"lat": -26.52, "lon": 31.47, "tilt": 27},
    "Mauritius":     {"lat": -20.28, "lon": 57.57, "tilt": 20},
    "Seychelles":    {"lat":  -4.68, "lon": 55.49, "tilt":  5},
}

# PV system parameters for the PVGIS simulation.
# These settings describe a reference 1 MW fixed-tilt array.
PV_CAPACITY = 1.0  # MW
AZIMUTH = 0  # north-facing
MODULE = 'Canadian_Solar_Inc__CS5P_220M'  # Verified module name in SAM CEC library
INVERTER = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_'  # Verified inverter in SAM CEC library

# Wind turbine parameters for the simple wind power curve.
# This is not a detailed turbine model, but it produces a basic MW profile.
WIND_CAPACITY = 1.0  # MW
CUT_IN = 3.0  # m/s
CUT_OUT = 25.0  # m/s
RATED_SPEED = 12.0  # m/s


def fetch_pvgis_data(lat: float, lon: float, tilt: float, year: int) -> pd.DataFrame:
    """Fetch PV generation data from PVGIS via pvlib.

    The function uses the PVGIS TMY interface to retrieve weather and irradiance
    data for the specified location. It then simulates a PV system and returns
    hourly AC electricity output in MW.
    """
    print(f"Fetching PVGIS data for {lat:.4f}°N, {lon:.4f}°E, tilt={tilt:.1f}°, {year}...")
    
    # Create a cache file name based on location and year for potential reuse
    cache_file = f'pvgis_pv_{lat:.4f}_{lon:.4f}_{year}.csv'
    if os.path.exists(cache_file):
        print(f"Loading cached PV data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = 'time'
        df.columns = ['electricity']
        return df
    else:
        # Use a common SADC timezone for simplicity (could be improved by using country-specific timezones).
        location = Location(lat, lon, tz='Africa/Johannesburg')  
        # Download TMY data for the location. Output is hourly weather data.
        tmy_data, inputs = pvlib.iotools.get_pvgis_tmy(lat, lon, outputformat='json')

        # Retrieve modules and inverters from SAM CEC library.
        # Note: These need to exist in the SAM database; check availability.
        modules = pvlib.pvsystem.retrieve_sam('cecmod')
        inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        module_params = modules.get(MODULE)
        inverter_params = inverters.get(INVERTER)
        inverter_params['pdc0'] = inverter_params.pop('Pdco')  # Rename for pvlib compatibility
        temperature_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        
        system = PVSystem(
            module_parameters=module_params,
            inverter_parameters=inverter_params,
            temperature_model_parameters=temperature_params,
            modules_per_string=1,
            strings_per_inverter=1,
            surface_tilt=tilt,  # Tilt towards the equator
            surface_azimuth=AZIMUTH,  # north-facing
        )

        # Build the PV model chain and simulate with clearsky conditions.
        mc = ModelChain(system, 
                        location, 
                        aoi_model='physical',  # best for CEC modules
                        spectral_model='no_loss',  # sufficient for regional studies
                        temperature_model='sapm',  # handles high SADC temperatures
                        ac_model='pvwatts')  # simple and robust for regional analysis
        weather = location.get_clearsky(tmy_data.index)

        # Filter out nighttime / near-zero irradiance rows
        #weather = weather[weather['ghi'] > 10]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mc.run_model(weather)

        # Convert output from watts to MW and scale to the reference capacity.
        power = mc.results.ac / 1000.0
        power = power * PV_CAPACITY
        
        # Ensure power is a Series with proper index
        if isinstance(power, pd.DataFrame):
            power = power.iloc[:, 4]  # take the fourth column

        # Create DataFrame with consistent format
        result_df = power.to_frame('electricity')
        result_df.index.name = 'time'
        
        # Save the generated data to cache for future runs
        result_df.to_csv(cache_file)
        
        return result_df



def fetch_era5_wind_data(lat: float, lon: float, year: int, tmpfile: str) -> pd.DataFrame:
    """Fetch ERA5 wind data via CDS API and compute wind power output."""
    print(f"Fetching ERA5 wind data for {lat:.4f}°N, {lon:.4f}°E, {year}...")
    
    # Create cache file name
    cache_file = f'era5_wind_{lat:.4f}_{lon:.4f}_{year}.csv'
    if os.path.exists(cache_file):
        print(f"Loading cached wind data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index.name = 'time'
        df.columns = ['electricity']
        return df
    
    c = cdsapi.Client()
    
    # Process all months
    all_u, all_v = [], []

    for month in range(1, 2):
        print(f"Fetching {year}-{month:02d}...")
        month_cache = f'era5_wind_{lat:.4f}_{lon:.4f}_{year}_{month:02d}.nc'
        
        if os.path.exists(month_cache):
            ds = xr.open_dataset(month_cache, engine='netcdf4')
        else:
            # Get the correct number of days for the month
            num_days = calendar.monthrange(year, month)[1]
            
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                    'year': str(year),
                    'month': f'{month:02d}',
                    'day': [f'{d:02d}' for d in range(1, num_days + 1)],
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5],
                    'format': 'netcdf',
                    'grid': [0.25, 0.25]
                },
                month_cache
            )
            
            ds = xr.open_dataset(month_cache, engine='netcdf4')
        
        # Extract point data
        ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
        all_u.append(ds_point['u10'].values.flatten())
        all_v.append(ds_point['v10'].values.flatten())
        ds.close()

    # Combine all months
    print("Combining monthly data...")
    u10 = np.concatenate(all_u)
    v10 = np.concatenate(all_v)
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    # Create time index for the full year
    times = pd.date_range(start=f'{year}-01-01', periods=len(wind_speed), freq='h')

    def wind_power_curve(speed: float) -> float:
        # Simple cubic power curve with cut-in, rated, and cut-out speeds.
        if speed < CUT_IN:
            return 0.0
        elif speed < RATED_SPEED:
            return (speed**3 / RATED_SPEED**3) * WIND_CAPACITY
        elif speed < CUT_OUT:
            return WIND_CAPACITY
        return 0.0

    power = np.array([wind_power_curve(speed) for speed in wind_speed])
    df = pd.DataFrame({'electricity': power}, index=times)
    df.index.name = 'time'
    
    # Save to cache
    df.to_csv(cache_file)
    
    return df


def format_for_epm(df: pd.DataFrame, country: str, technology: str) -> pd.DataFrame:
    """Format PV or wind time series for EPM usage.

    This helper converts the raw hourly series into the standard EPM
    format with columns for country, zone, technology, month, day, hour, and value.
    """
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'time'})
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df = df.rename(columns={'electricity': 'value'})
    df['country'] = country
    df['technology'] = technology
    df['zone'] = country
    df = df[['country', 'zone', 'technology', 'month', 'day', 'hour', 'value']]
    return df.sort_values(['country', 'technology', 'month', 'day', 'hour'])


def geocode_country(country_name: str) -> Tuple[float, float]:
    """Resolve country name to latitude and longitude using Nominatim."""
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': country_name,
        'format': 'json',
        'limit': 1,
    }
    headers = {"User-Agent": "MyEnergyResearchApp/1.0 (your@email.com)"}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f'Could not geocode country: {country_name}')
    return float(results[0]['lat']), float(results[0]['lon'])


def get_country_coordinates(country_name: str, use_geocode: bool = True) -> Tuple[float, float, float]:
    """Resolve coordinates either from the SADC dictionary or via geocoding."""
    if not use_geocode:
        coords = sadc_countries.get(country_name)
        if coords is None:
            raise ValueError(f"Country '{country_name}' is not available in sadc_countries.")
        return coords['lat'], coords['lon'], coords['tilt']

    try:
        lat, lon = geocode_country(country_name)
        # Estimate tilt based on latitude (simplified approach)
        tilt = abs(lat)
        return lat, lon, tilt
    except ValueError:
        if country_name in sadc_countries:
            coords = sadc_countries[country_name]
            print(f"Geocode failed for {country_name}; using sadc_countries coordinates instead.")
            return coords['lat'], coords['lon'], coords['tilt']
        raise


def build_country_list(countries: str) -> List[str]:
    """Normalize user-supplied comma-separated country names."""
    return [country.strip() for country in countries.split(',') if country.strip()]


def generate_country_files(countries: List[str], year: int, output_dir: str, use_geocode: bool = True) -> None:
    """Generate PV and wind files for all requested countries.

    If use_geocode is False, coordinates are pulled from the sadc_countries
    dictionary only; otherwise geocoding is attempted first with a fallback to
    the dictionary for known SADC countries.
    """
    pv_frames = []
    wind_frames = []

    for country in countries:
        print(f"\nProcessing country: {country}")
        try:
            lat, lon, tilt = get_country_coordinates(country, use_geocode=use_geocode)
        except ValueError as e:
            print(f"Error resolving coordinates for {country}: {e}")
            continue
        print(f"Resolved {country} to lat={lat:.4f}, lon={lon:.4f}, tilt={tilt:.4f}")

        # Generate PV data and format it for EPM.
        pv_df = fetch_pvgis_data(lat, lon, tilt, year)
        pv_formatted = format_for_epm(pv_df, country, 'PV')
        print (f"Sample PV data for {country} completed")
        pv_frames.append(pv_formatted)

        # Generate wind data and format it for EPM.
        wind_df = fetch_era5_wind_data(lat, lon, year, tmpfile=os.path.join(output_dir, f'era5_wind_{country}.nc'))
        wind_formatted = format_for_epm(wind_df, country, 'Wind')
        wind_frames.append(wind_formatted)

    all_pv = pd.concat(pv_frames, ignore_index=True)
    all_wind = pd.concat(wind_frames, ignore_index=True)



    pv_csv = os.path.join(output_dir, f'pv_countries_{year}.csv')
    wind_csv = os.path.join(output_dir, f'wind_countries_{year}.csv')
    excel_file = os.path.join(output_dir, f'renewables_countries_{year}.xlsx')

    # Save combined outputs for all countries in CSV and Excel formats.
    all_pv.to_csv(pv_csv, index=False)
    all_wind.to_csv(wind_csv, index=False)

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        all_pv.to_excel(writer, sheet_name='PV', index=False)
        all_wind.to_excel(writer, sheet_name='Wind', index=False)

    print(f"Saved combined PV CSV: {pv_csv}")
    print(f"Saved combined Wind CSV: {wind_csv}")
    print(f"Saved combined Excel workbook: {excel_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate PV and wind data for a list of countries using PVGIS and ERA5.'
    )
    parser.add_argument(
        '--countries',
        required=True,
        help='Comma-separated list of country names, e.g. "Spain,France,Germany"',
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2019,
        help='Year for data generation',
    )
    parser.add_argument(
        '--output-dir',
        default='pre-analysis/representative_days/',
        help='Output directory for generated files',
    )
    parser.add_argument(
        '--no-geocode',
        action='store_false',
        dest='use_geocode',
        help='Use only the sadc_countries dictionary and skip Nominatim geocoding',
    )
    args = parser.parse_args()

    countries = build_country_list(args.countries)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        generate_country_files(countries, args.year, args.output_dir, use_geocode=args.use_geocode)
        print('Data generation complete.')
    except Exception as exc:
        print(f'Error: {exc}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
