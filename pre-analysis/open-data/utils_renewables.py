import os
import time
import requests
import pandas as pd
import numpy as np
from time import sleep
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

# API token for Renewables Ninja
API_TOKEN = '2deb093cdece49f3b316c98687150421ee425566'
nb_days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))


def get_renewable_data(power_type, locations, start_date, end_date, api_token=API_TOKEN, dataset="merra2", capacity=1,
                       system_loss=0.1, height=100, tracking=0, tilt=35, azim=180,
                       turbine='Gamesa+G114+2000', local_time='true'):
    """Fetch solar or wind power data for multiple locations using the Renewables Ninja API.

    Args:
    - api_token (str): Your Renewables Ninja API token.
    - power_type (str): 'solar' or 'wind'.
    - locations (list of tuples): List of locations as (latitude, longitude).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - dataset (str): Dataset to use ('merra2' or 'sarah' for solar). Defaults to 'merra2'.
    - capacity (float): Capacity of the power system. Defaults to 1.
    - system_loss (float): System losses (for solar). Defaults to 0.1.
    - height (float): Turbine height (for wind). Defaults to 100.
    - tracking (int): Tracking type for solar (0 for fixed, 1 for single-axis). Defaults to 0.
    - tilt (float): Tilt angle for solar panels. Defaults to 35.
    - azim (float): Azimuth angle for solar panels. Defaults to 180.
    - turbine (str): Turbine type for wind. Defaults to 'Vestas+V80+2000'.
    - local_time (bool): Whether to return data in local time. Defaults to True.

    Returns:
    - dict: A dictionary of results for each location.
    """
    base_url = 'https://www.renewables.ninja/api/data/'

    # Set headers for the API request
    headers = {
        'Authorization': f'Token {api_token}'
    }

    # Track requests per minute
    requests_made_minute = 0
    requests_tot = 0
    minute_start_time = time.time()

    all_data = []

    # Iterate over locations
    for zone, (lat, lon) in locations.items():
        print(f'Fetching data for: {zone}')
    # Build the request URL based on the power type (solar or wind)
        if power_type == 'solar':
            url = f"{base_url}pv?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&system_loss={system_loss}&tracking={tracking}&tilt={tilt}&azim={azim}&local_time={local_time}&format=json"
        elif power_type == 'wind':
            url = f"{base_url}wind?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&height={height}&turbine={turbine}&local_time={local_time}&format=json"
        else:
            raise ValueError("Invalid power_type. Choose either 'solar' or 'wind'.")

        # Send the request
        response = requests.get(url, headers=headers, verify=True)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            # If empty data
            if not data['data']:
                print(f"No data available for {zone})")
                continue
            # Append the data along with location details
            for timestamp, values in data['data'].items():
                row = {
                    'zone': zone,
                    # 'latitude': lat,
                    # 'longitude': lon,
                    'timestamp': pd.to_datetime(int(timestamp) / 1000, unit='s'),
                    # 'local_time': pd.to_datetime(int(local_time) / 1000, unit='s'),
                    **values  # Unpack all the power generation data at that timestamp
                }
                all_data.append(row)
        else:
            print(f"Error fetching data for location ({lat}, {lon}): {response.status_code}")

        # Track the request
        requests_made_minute += 1
        requests_tot += 1

        # If we hit 6 requests within a minute, we wait for the remaining time of that minute
        if requests_made_minute >= 6:
            print('Waiting for a minute to not hit the API rate limit...')
            elapsed_time = time.time() - minute_start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the counter and timer
            requests_made_minute = 0
            minute_start_time = time.time()

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(all_data)
    df.rename(columns={'electricity': power_type}, inplace=True)
    return df, requests_tot


def get_years_renewables(locations, power_type, start_year, end_year, start_day='01-01', end_day='12-31',
                         turbine='Gamesa+G114+2000', name_data='data', output='data', local_time=True):
    """Get the renewable data for multiple years.

    Parameters
    ----------
    locations: list
        Dict of locations as (latitude, longitude).
    power_type: str
        'solar' or 'wind'.
    start_year: int
        Start year.
    end_year: int
        End year.
    start_day: str, optional, default '01-01'
        Start day.
    end_day: str, optional, default '12-31'
        End day.
    turbine: str, optional, default 'Gamesa+G114+2000'
        Turbine type for wind.
    name_data: str, optional, default 'data'
        Name of the data.

    Returns
    -------
    results_concat: pd.DataFrame
        DataFrame with the energy data.
    """

    results = {}
    hour_start_time = time.time()
    requests_tot = 0
    for year in range(start_year, end_year):

        start_date = '{}-{}'.format(year, start_day)
        end_date = '{}-{}'.format(year, end_day)

        # Call the function to get data
        data, requests_made = get_renewable_data(power_type, locations, start_date, end_date, turbine=turbine,
                                                 local_time=str(local_time).lower())
        requests_tot += requests_made

        if data.empty:
            print("No data for year {}".format(year))
            continue
        else:
            print("Getting data {} for year {}".format(power_type, year))
            data.set_index(['zone'], inplace=True)
            data['local_time'] = pd.to_datetime(data['local_time'], utc=True)
            data['season'] = data['local_time'].dt.month
            data['day'] = data['local_time'].dt.day
            data['hour'] = data['local_time'].dt.hour
            # data['time_only'] = data['local_time'].dt.strftime('%m-%d %H:%M:%S')
            data.set_index(['season', 'day', 'hour'], inplace=True, append=True)
            data = data.loc[:, power_type]

            results.update({year: data})

        # If we hit 50 requests within an hour, we wait for the remaining time of that hour before continuing the requests
        if requests_tot >= 36:
            print('Waiting for an hour to not hit the API rate limit...')
            elapsed_time = time.time() - hour_start_time
            if elapsed_time < 3600:
                sleep_time = 3600 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the counter and timer
            requests_tot = 0
            hour_start_time = time.time()

    results_concat = pd.concat(results, axis=1)
    results_concat.to_csv(os.path.join(output, 'data_{}_{}.csv'.format(name_data, power_type)))
    print(f'Export data to {os.path.join(output, "data_{}_{}.csv".format(name_data, power_type))}')
    return results_concat


def find_representative_year(df, method='average_profile'):
    """Find the representative year.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the energy data.
    method: str, optional, default 'average_profile'
        Method to find the representative year.

    Returns
    -------
    repr_year: int
        Representative year.
    """
    dict_info, repr_year = {}, None
    if method == 'average_cf':
        # Find representative year
        temp = df.sum().round(3)
        # get index closest (not equal) to the median in the temp
        repr_year = (np.abs(temp - temp.median())).idxmin()
        print('Representative year {}'.format(repr_year), 'with production of {:.0%}'.format(temp[repr_year] / 8760))
        print('Standard deviation of {:.3%}'.format(temp.std() / 8760))
        dict_info.update({'year_repr': repr_year, 'std': temp.std(), 'median': temp.median()})

    elif method == 'average_profile':
        average_profile = df.mean(axis=1)
        deviations = {}
        for year in df.columns:
            deviations[year] = np.sum(np.abs(df[year] - average_profile))
        repr_year = min(deviations, key=deviations.get)

    else:
        raise ValueError("Invalid method. Choose either 'representative_year' or 'average_profile'.")

    return repr_year


def format_data_energy(filenames, locations):
    """Format the data for the energy from .csv files.

    Parameters
    ----------
    filenames: dict
        Dictionary with the filenames.

    Returns
    -------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    """
    # Find the representative year

    df_energy = {}
    for key, item in filenames.items():
        filename, reading = item[0], item[1]

        # Extract from the data results
        if reading == 'renewable_ninja':
            df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3, 4, 5])
            df = df.reset_index()
            df['tech'] = key
            df = df.drop(columns=['zone']).set_index(['zone', 'season', 'day', 'hour', 'tech'])

        elif reading == 'standard':
            df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3])
            df = df.reset_index()
            df['tech'] = key
            df = df.set_index(['zone', 'season', 'day', 'hour', 'tech'])

        else:
            raise ValueError('Unknown reading. Only implemented for: renewable_ninja, standard.')

        df_energy.update({key: df})

    df_energy = pd.concat(df_energy.values(), ignore_index=False)

    repr_year = find_representative_year(df_energy)
    print('Representative year {}'.format(repr_year))
    df_energy = df_energy.loc[:, repr_year].unstack('tech').reset_index()
    df_energy = df_energy.sort_values(by=['zone', 'season', 'day', 'hour'], ascending=True)

    # If 2/29, remove it
    if len(df_energy.season.unique()) == 12:  # season expressed as months
        df_energy = df_energy[~((df_energy['season'] == 2) & (df_energy['day'] == 29))]

    if df_energy.isna().any().any():
        print('Warning: NaN values in the DataFrame')

    keys_to_merge = ['PV', 'Wind', 'Load', 'ROR']
    keys_to_merge = [i for i in keys_to_merge if i in df_energy.keys()]

    print('Annual capacity factor (%):', df_energy.groupby('zone')[keys_to_merge].mean().reset_index())
    if len(df_energy.zone.unique()) > 1:  # handling the case with multiple zones to rename columns
        df_energy = df_energy.set_index(['zone', 'season', 'day', 'hour']).unstack('zone')
        df_energy.columns = ['_'.join([idx0, idx1]) for idx0, idx1 in df_energy.columns]
        df_energy = df_energy.reset_index()
    else:
        df_energy = df_energy.drop('zone', axis=1)
    return df_energy


def make_heatmap(df, tech, path=None):
    df_energy = df.copy()
    daily_df = df_energy.groupby(['season', 'day'], as_index=False).mean(numeric_only=True).drop(columns='hour')
    daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)
    tmp = daily_df.sort_values(['season', 'day']).copy()

    heatmap_data = tmp.copy()
    heatmap_data.index = tmp['season-day']

    # Get tick positions where the season changes
    season_labels = tmp['season'].values
    x_positions = []
    x_labels = []
    for i in range(1, len(season_labels)):
        if season_labels[i] != season_labels[i - 1]:
            x_positions.append(i)
            x_labels.append(season_labels[i])
    # Add the first season manually
    x_positions = [0] + x_positions
    x_labels = [season_labels[0]] + x_labels

    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data.T, cmap='YlGnBu', xticklabels=False)

    # Custom x-ticks with season only
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=0)
    plt.xlabel("Season")
    plt.ylabel("")
    plt.title(f"Heatmap of Daily {tech} Capacity Factor")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_boxplot(df, tech, path=None):
    df_energy = df.copy()
    #df_energy['season'] = df_energy['season'].apply(lambda x: calendar.month_abbr[int(x)])
    daily_df = df_energy.groupby(['season', 'day'], as_index=False).mean(numeric_only=True).drop(columns='hour')
    daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)

    tmp = daily_df.sort_values(['season', 'day']).copy()

    melted = tmp.melt(id_vars=['season', 'day'], value_vars=[col for col in tmp.columns if col not in ['zone', 'season', 'day', 'hour']],
                      var_name='zone', value_name='daily_mean')

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted, x="zone", y="daily_mean", hue="season")
    plt.xticks(rotation=45)

    plt.legend(
        title='Season',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )
    plt.ylabel("")
    plt.xlabel("")

    # Set ymin to 0
    plt.ylim(bottom=0)

    # Format x-ticks to show only the zone name
    xtick_labels = [col.split('_')[1] for col in melted['zone'].unique()]
    plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=45)

    plt.title(f"Distribution of Daily {tech} Capacity Factor by Season and Zone")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
