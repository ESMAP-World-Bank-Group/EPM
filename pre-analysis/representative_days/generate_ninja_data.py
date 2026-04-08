"""
Script to download PV and wind generation data from Renewables.ninja API
and format it for EPM representative days pipeline.

Usage:
1. Get API token from https://www.renewables.ninja/register
2. Set TOKEN variable below
3. Run: python generate_ninja_data.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Configuration
TOKEN = "your_api_token_here"  # Get from https://www.renewables.ninja/register
LAT = 40.0  # Latitude (example: Spain)
LON = -4.0  # Longitude (example: Spain)
YEAR = 2018  # Year for data
OUTPUT_DIR = "pre-analysis/representative_days/input"

def fetch_solar_data(lat, lon, year, token):
    """Fetch solar PV data from Renewables.ninja"""
    url = "https://www.renewables.ninja/api/data/pv"

    params = {
        'lat': lat,
        'lon': lon,
        'date_from': f'{year}-01-01',
        'date_to': f'{year}-12-31',
        'dataset': 'merra2',
        'capacity': 1,  # 1 MW capacity
        'system_loss': 0.1,  # 10% system loss
        'tracking': 0,  # Fixed tilt
        'tilt': 35,  # 35° tilt
        'azim': 180,  # South-facing
        'format': 'json',
        'header': True,
        'local_time': True
    }

    headers = {'Authorization': f'Bearer {token}'}

    print(f"Fetching solar data for {lat}°N, {lon}°E, {year}...")
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")

    data = response.json()
    df = pd.DataFrame(data['data'], columns=['time', 'electricity'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    return df

def fetch_wind_data(lat, lon, year, token):
    """Fetch wind data from Renewables.ninja"""
    url = "https://www.renewables.ninja/api/data/wind"

    params = {
        'lat': lat,
        'lon': lon,
        'date_from': f'{year}-01-01',
        'date_to': f'{year}-12-31',
        'dataset': 'merra2',
        'capacity': 1,  # 1 MW capacity
        'height': 100,  # 100m hub height
        'turbine': 'Vestas_V80_2000kW',  # Example turbine
        'format': 'json',
        'header': True,
        'local_time': True
    }

    headers = {'Authorization': f'Bearer {token}'}

    print(f"Fetching wind data for {lat}°N, {lon}°E, {year}...")
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")

    data = response.json()
    df = pd.DataFrame(data['data'], columns=['time', 'electricity'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    return df

def format_for_epm(df, technology, zone="ZONE1"):
    """Convert Renewables.ninja data to EPM format"""
    # Reset index to get time as column
    df = df.reset_index()

    # Extract date components
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour  # 0-23 format

    # Rename electricity column
    df = df.rename(columns={'electricity': 'value'})

    # Add zone column
    df['zone'] = zone

    # Select and reorder columns
    df = df[['month', 'day', 'hour', 'zone', 'value']]

    # Sort by time
    df = df.sort_values(['month', 'day', 'hour'])

    return df

def main():
    if TOKEN == "your_api_token_here":
        print("ERROR: Please set your API token in the TOKEN variable")
        print("Get one at: https://www.renewables.ninja/register")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Fetch solar data
        solar_df = fetch_solar_data(LAT, LON, YEAR, TOKEN)
        solar_formatted = format_for_epm(solar_df, "PV")

        # Save solar data
        solar_path = os.path.join(OUTPUT_DIR, f"vre_rninja_solar_{YEAR}.csv")
        solar_formatted.to_csv(solar_path, index=False)
        print(f"Saved solar data to {solar_path}")

        # Fetch wind data
        wind_df = fetch_wind_data(LAT, LON, YEAR, TOKEN)
        wind_formatted = format_for_epm(wind_df, "Wind")

        # Save wind data
        wind_path = os.path.join(OUTPUT_DIR, f"vre_rninja_wind_{YEAR}.csv")
        wind_formatted.to_csv(wind_path, index=False)
        print(f"Saved wind data to {wind_path}")

        print("Data generation complete!")
        print(f"Files ready for EPM pipeline in {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()