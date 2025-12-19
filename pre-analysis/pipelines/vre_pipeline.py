"""Variable renewable energy (VRE) pipeline for GAP -> RNinja/IRENA extraction and exports.

Main entry points:
- `run_renewables_ninja_workflow`: orchestrate RNinja downloads and plotting.
- `run_irena_workflow`: fetch IRENA capacity factors and export representative CSVs.
"""

import os
import re
import time
import warnings
from configparser import ConfigParser
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

TOKEN_SECTION = "api_tokens"
TOKEN_PATH_ENV_VAR = "API_TOKENS_PATH"
# Default to the repo's config/api_tokens.ini (two levels up from this file).
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_PATH = BASE_DIR / "config" / "api_tokens.ini"
nb_days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _slugify(text):
    """Simple slug for filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _flatten_year_columns(df):
    """Convert MultiIndex columns like (year, tech) -> simple year ints."""

    def _to_year(col):
        if isinstance(col, tuple):
            col = col[0]
        try:
            return int(col)
        except (TypeError, ValueError):
            return col

    df.columns = [_to_year(c) for c in df.columns]
    return df


def require_file(path, hint=None):
    """Validate that a required external file exists; raise with a helpful message if missing."""
    path = Path(path)
    if not path.exists():
        sharepoint = "https://worldbankgroup.sharepoint.com/:f:/r/teams/PowerSystemPlanning-WBGroup/Shared%20Documents/2.%20Knowledge%20Products/19.%20Databases/EPM%20Prepare%20Data?csf=1&web=1&e=wig2nC"
        extra = f" {hint}" if hint else f" Place the file in the expected directory. All files can be downloaded from WBG SharePoint: {sharepoint}"
        raise FileNotFoundError(f"File not found: {path}.{extra}")
    return path


def resolve_country_name(input_name, available_names, threshold=0.75, verbose=True, allow_missing=False):
    """Resolve a user-provided country name against available names with fuzzy matching.

    When ``allow_missing`` is True, returns None (with a message) instead of raising when no match is found.
    """
    available = [name for name in available_names if isinstance(name, str)]
    normalized = {name.strip().lower(): name for name in available}
    candidate = input_name.strip().lower()

    if candidate in normalized:
        match = normalized[candidate]
        if verbose:
            print(f"[country-resolver] Exact match: '{input_name}' -> '{match}'")
        return match

    best, best_score = None, 0
    for name in available:
        score = SequenceMatcher(None, candidate, name.strip().lower()).ratio()
        if score > best_score:
            best, best_score = name, score

    if best_score >= threshold:
        if verbose:
            pct = f"{best_score*100:.1f}%"
            print(f"[country-resolver] Using closest match ({pct}): '{input_name}' -> '{best}'")
        return best

    if allow_missing:
        suggestions = ", ".join(available[:5])
        print(
            f"[country-resolver] Missing '{input_name}'. Closest='{best}' ({best_score:.2f}). "
            f"Available examples: {suggestions}. Skipping."
        )
        return None

    suggestions = ", ".join(available[:5])
    raise ValueError(
        f"Could not match country '{input_name}'. Top suggestion: '{best}' "
        f"(score {best_score:.2f}). Available examples: {suggestions}"
    )


def load_api_token(token_name, env_var=None, config_path=None):
    """Resolve an API token from the environment first, then from a config file."""
    env_var_name = env_var or f"API_TOKEN_{token_name.upper()}"
    token = os.getenv(env_var_name)
    if token:
        return token

    candidate_path = Path(config_path) if config_path else Path(os.getenv(TOKEN_PATH_ENV_VAR, DEFAULT_TOKEN_PATH))
    if candidate_path.exists():
        parser = ConfigParser()
        parser.read(candidate_path)
        if parser.has_option(TOKEN_SECTION, token_name):
            value = parser.get(TOKEN_SECTION, token_name).strip()
            if value:
                return value

    raise RuntimeError(
        f"Missing API token for '{token_name}'. Set {env_var_name} or add it under [{TOKEN_SECTION}] "
        f"in {candidate_path} (set a custom path via {TOKEN_PATH_ENV_VAR})."
    )


def vre_output_filename(source, label, tech):
    """Standard output filename: vre_<source>_<label?>_<tech>.csv."""
    label_part = f"_{label}" if label else ""
    return f"vre_{source}{label_part}_{tech}.csv"


def rninja_output_filename(dataset_label, tech):
    """Build the RNinja CSV filename using the unified VRE naming scheme."""
    return vre_output_filename("rninja", dataset_label, tech)


def irena_output_filename(dataset_label, tech):
    """Build the IRENA CSV filename using the unified VRE naming scheme."""
    return vre_output_filename("irena", dataset_label, tech)


def _plot_capacity_factor_by_zone(df: pd.DataFrame, tech_label: str, output_path: Path, plot_label: str, source_suffix: str):
    """Render per-zone heatmap/boxplot files with country names in titles and filenames."""
    try:
        from plots import make_boxplot, make_heatmap  # local import to avoid hard dependency
    except Exception:
        return {}

    season_label = "season" if "season" in df.columns else "month" if "month" in df.columns else None
    required_cols = {"zone", "day", "hour"}
    if season_label:
        required_cols.add(season_label)
    if not season_label or not required_cols.issubset(df.columns):
        return {}

    value_cols = [
        col for col in df.columns
        if col not in required_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not value_cols:
        return {}

    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for zone, group in df.groupby("zone"):
        zone_label = str(zone)
        zone_slug = _slugify(zone_label)
        plot_df = group[[season_label, "day", "hour"]].reset_index(drop=True).copy()
        numeric = group[value_cols].apply(pd.to_numeric, errors="coerce")

        if numeric.shape[1] == 1:
            plot_df[zone_label] = numeric.iloc[:, 0].reset_index(drop=True)
        else:
            for col in numeric.columns:
                plot_df[f"{zone_label} ({col})"] = numeric[col].reset_index(drop=True)

        heatmap_path = output_path / f"heatmap_{tech_label}_{zone_slug}{plot_label}_{source_suffix}.pdf"
        boxplot_path = output_path / f"boxplot_{tech_label}_{zone_slug}{plot_label}_{source_suffix}.pdf"
        make_heatmap(
            plot_df,
            tech=f"{tech_label} - {zone_label}",
            path=heatmap_path,
            season_label=season_label,
            vmin=0,
            vmax=1,
        )
        make_boxplot(plot_df, tech=f"{tech_label} - {zone_label}", path=boxplot_path, season_label=season_label)
        paths[zone_label] = {"heatmap": heatmap_path, "boxplot": boxplot_path}

    return paths


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def extract_time_zone(countries, name_map):
    """
    Extracts the time zone (IANA tz database name, e.g., 'Africa/Luanda') for each country in a given list.

    This function takes a list of country names used in SPLAT (which may use non-standard naming conventions)
    and a mapping (`name_map`) from SPLAT-style names to standard country names (as recognized by geocoding services).
    It uses the `geopy` package to geocode each standard country name and `timezonefinder` to identify the time zone
    at the country centroid.

    Parameters:
    ----------
    countries : list of str
        List of SPLAT-style country names (e.g., ['SouthAfrica', 'DemocraticRepublicoftheCongo']).

    name_map : dict
        Dictionary mapping SPLAT-style names to standard country names
        (e.g., {'SouthAfrica': 'South Africa'}).

    Returns:
    -------
    dict
        Dictionary mapping original SPLAT-style names to their IANA timezone name
        (e.g., {'SouthAfrica': 'Africa/Johannesburg'}).
    """
    standard_names = [name_map.get(c, c) for c in countries]

    # Initialize timezone and geolocation tools
    tf = TimezoneFinder()
    geolocator = Nominatim(user_agent="splat_timezones")

    # Build timezone dictionary
    country_timezones = {}

    for name, std_name in zip(countries, standard_names):
        try:
            location = geolocator.geocode(std_name, timeout=10)
            if location:
                tz = tf.timezone_at(lat=location.latitude, lng=location.longitude)
                country_timezones[name] = tz
            else:
                print(f"Could not geocode: {std_name}")
        except Exception as e:
            print(f"Error with {std_name}: {e}")

        time.sleep(1)  # avoid overloading the API
    return country_timezones


def convert_to_utc(row, country_timezones):
    """
    Converts a local timestamp to UTC based on the country's time zone.

    This function is designed to be used within a pandas `.apply()` call
    to convert a 'timestamp' column (assumed local time) into UTC, using
    a dictionary that maps each country name to its IANA time zone string.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing at least 'CtryName' and 'timestamp'.
    country_timezones : dict
        Dictionary mapping country names (as in 'CtryName') to their IANA time zone names,
        e.g., {'SouthAfrica': 'Africa/Johannesburg'}.

    Returns
    -------
    datetime
        The timestamp converted to UTC timezone.
    """
    ctry = row['CtryName']
    local_zone = pytz.timezone(country_timezones[ctry])
    local_time = local_zone.localize(row['timestamp'], is_dst=None)
    return local_time.astimezone(pytz.utc)


# ---------------------------------------------------------------------------
# GAP / RNinja workflows
# ---------------------------------------------------------------------------


def gap_rninja_coordinates(xlsx_path, countries, tech_types=('solar', 'wind'), sheet_name='Power facilities',
                           output_dir='output', encoding='utf-8'):
    """Run the Global Atlas Power -> RNinja workflow: load, filter, pick top projects, export, and return locations."""

    def gap_top_projects(df, countries_map, tech_types):
        """Pick the most relevant Global Atlas Power project per country/tech and return RNinja locations mapping."""
        # df columns: ['Country/area', 'Type', 'Status', 'Capacity (MW)', 'Plant / Project name', 'Latitude', 'Longitude', 'City']

        def gap_rninja_locations(df_relevant):
            """Convert Global Atlas Power rows into the RNinja ``locations`` mapping `{tech: {country: (lat, lon)}}`."""
            result = {}
            for _, row in df_relevant.iterrows():
                tech = row['Type'].lower()  # e.g., 'solar', 'wind'
                if tech not in result:
                    result[tech] = {}
                result[tech][row['Country']] = (row['Latitude'], row['Longitude'])
            return result

        results = []

        status_priority = ['operating', 'construction', 'pre-construction', 'announced']

        for country, resolved_country in countries_map.items():
            for tech in tech_types:
                filtered = df[(df['Country/area'] == resolved_country) & (df['Type'] == tech)]

                found = False
                for status in status_priority:
                    sub = filtered[filtered['Status'] == status]
                    if not sub.empty:
                        top_project = sub.loc[sub['Capacity (MW)'].idxmax()]
                        results.append({
                            'Country': country,
                            'Type': tech,
                            'Plant / Project name': top_project['Plant / Project name'],
                            'Capacity (MW)': top_project['Capacity (MW)'],
                            'Latitude': top_project['Latitude'],
                            'Longitude': top_project['Longitude'],
                            'City': top_project['City']
                        })
                        found = True
                        break  # stop at the first status with a result
                        # Fallback: search all remaining statuses

                if not found:
                    remaining_statuses = set(df['Status'].unique()) - set(status_priority)
                    sub = filtered[filtered['Status'].isin(remaining_statuses)]
                    if not sub.empty:
                        top_project = sub.loc[sub['Capacity (MW)'].idxmax()]
                        results.append({
                            'Country': country,
                            'Type': tech,
                            'Plant / Project name': top_project['Plant / Project name'],
                            'Capacity (MW)': top_project['Capacity (MW)'],
                            'Latitude': top_project['Latitude'],
                            'Longitude': top_project['Longitude'],
                            'City': top_project['City']
                        })

        projects_df = pd.DataFrame(results)
        rninja_locations = gap_rninja_locations(projects_df)
        return projects_df, rninja_locations

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown extension is not supported and will be removed",
            category=UserWarning,
            module="openpyxl",
        )
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=None, header=0)

    available_countries = df['Country/area'].unique()
    resolved_map = {}
    for country in countries:
        resolved = resolve_country_name(country, available_countries, verbose=True)
        resolved_map[country] = resolved

    df = df[df['Country/area'].isin(resolved_map.values())]
    df = df[df['Type'].isin(tech_types)]

    projects_df, rninja_locations = gap_top_projects(df, resolved_map, list(tech_types))

    tech_slug = "_".join(tech_types)
    csv_path = output_path / f"most_relevant_projects_{tech_slug}.csv"
    projects_df.to_csv(csv_path, index=False, encoding=encoding)
    print(f"Saved selected projects to {csv_path}")

    for tech in tech_types:
        temp = projects_df[projects_df['Type'] == tech]
        for country in countries:
            if country not in temp['Country'].unique():
                print(f"Warning: No projects found for {country} for {tech}.")

    return projects_df, rninja_locations, csv_path


def get_renewables_ninja(locations, tech, start_year, end_year, start_day='01-01', end_day='12-31',
                         turbine='Gamesa+G114+2000', dataset_label=None, output='data', export_csv=True, local_time=False,
                         api_token=None, dataset="merra2", capacity=1, system_loss=0.1,
                         height=100, tracking=0, tilt=35, azim=180):
    """Get Renewables Ninja data across multiple years (main entry point)."""

    def get_rninja_yearly(tech, locations, start_date, end_date, api_token=None, dataset="merra2", capacity=1,
                          system_loss=0.1, height=100, tracking=0, tilt=35, azim=180,
                          turbine='Gamesa+G114+2000', local_time=False):
        """Fetch solar or wind power data for a single year/date span using the Renewables Ninja API."""
        base_url = 'https://www.renewables.ninja/api/data/'
        api_token = api_token or load_api_token("renewables_ninja")

        headers = {
            'Authorization': f'Token {api_token}'
        }

        requests_made_minute = 0
        requests_tot = 0
        minute_start_time = time.time()

        all_data = []

        for zone, (lat, lon) in locations.items():
            print(f'Fetching data for: {zone}')
            if tech == 'solar':
                url = f"{base_url}pv?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&system_loss={system_loss}&tracking={tracking}&tilt={tilt}&azim={azim}&local_time={local_time}&format=json"
            elif tech == 'wind':
                url = f"{base_url}wind?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&height={height}&turbine={turbine}&local_time={local_time}&format=json"
            else:
                raise ValueError("Invalid tech. Choose either 'solar' or 'wind'.")

            response = requests.get(url, headers=headers, verify=True)

            if response.status_code == 200:
                data = response.json()
                if not data['data']:
                    print(f"No data available for {zone})")
                    continue
                for timestamp, values in data['data'].items():
                    row = {
                        'zone': zone,
                        'timestamp_utc': pd.to_datetime(int(timestamp) / 1000, unit='s', utc=True),
                        **values
                    }
                    all_data.append(row)
            else:
                print(f"Error fetching data for location ({lat}, {lon}): {response.status_code}")

            requests_made_minute += 1
            requests_tot += 1

            if requests_made_minute >= 6:
                print('Waiting for a minute to not hit the API rate limit...')
                elapsed_time = time.time() - minute_start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)

                requests_made_minute = 0
                minute_start_time = time.time()

        df = pd.DataFrame(all_data)
        df.rename(columns={'electricity': tech}, inplace=True)
        return df, requests_tot

    results = {}
    hour_start_time = time.time()
    requests_tot = 0
    for year in range(start_year, end_year):

        start_date = '{}-{}'.format(year, start_day)
        end_date = '{}-{}'.format(year, end_day)

        data, requests_made = get_rninja_yearly(
            tech,
            locations,
            start_date,
            end_date,
            api_token=api_token,
            dataset=dataset,
            capacity=capacity,
            system_loss=system_loss,
            height=height,
            tracking=tracking,
            tilt=tilt,
            azim=azim,
            turbine=turbine,
            local_time=str(local_time).lower()
        )
        requests_tot += requests_made

        if data.empty:
            print("No data for year {}".format(year))
            continue
        else:
            print("Getting data {} for year {}".format(tech, year))
            data.set_index(['zone'], inplace=True)
            data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'], utc=True)
            data['month'] = data['timestamp_utc'].dt.month
            data['day'] = data['timestamp_utc'].dt.day
            data['hour'] = data['timestamp_utc'].dt.hour
            data.set_index(['month', 'day', 'hour'], inplace=True, append=True)
            data = data.loc[:, tech]

            results.update({year: data})

        if requests_tot >= 36:
            print('Waiting for an hour to not hit the API rate limit...')
            elapsed_time = time.time() - hour_start_time
            if elapsed_time < 3600:
                sleep_time = 3600 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            requests_tot = 0
            hour_start_time = time.time()

    if not results:
        print("No Renewables Ninja data retrieved; nothing to export.")
        return pd.DataFrame()

    results_concat = pd.concat(results, axis=1)
    results_concat = _flatten_year_columns(results_concat)
    results_concat.index = results_concat.index.set_names(['zone', 'month', 'day', 'hour'])

    flat_output = results_concat.reset_index()
    if export_csv:
        output_csv = os.path.join(output, rninja_output_filename(dataset_label, tech))
        flat_output.to_csv(output_csv, index=False)
        print(f'Export data to {output_csv}')
    return results_concat


def run_renewables_ninja_workflow(
    locations,
    start_year,
    end_year,
    dataset_label=None,
    extract_renewables=None,
    input_dir="input",
    output_dir="output",
    generate_plots=True,
    local_time=True,
    **rninja_kwargs,
):
    """Run the Renewables.ninja extraction pipeline and optionally plot the results."""
    if not locations:
        raise ValueError("locations must map technologies to zone coordinate dictionaries.")

    extract_flags = extract_renewables or {ptype: True for ptype in locations}
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    techs = [ptype for ptype, enabled in extract_flags.items() if enabled and ptype in locations]
    for tech in techs:
        get_renewables_ninja(
            locations=locations[tech],
            tech=tech,
            start_year=start_year,
            end_year=end_year,
            dataset_label=dataset_label,
            output=str(input_path),
            local_time=local_time,
            **rninja_kwargs,
        )

    from plots import make_boxplot, make_heatmap  # local import to avoid hard dependency for consumers

    tech_labels = {"solar": "PV", "wind": "Wind"}
    loaded = {}
    plot_label = f"_{dataset_label}" if dataset_label else ""
    for tech, tech_label in tech_labels.items():
        csv_path = input_path / rninja_output_filename(dataset_label, tech)
        if not csv_path.exists():
            print(f"File {csv_path} does not exist. Skipping {tech}.")
            continue

        df = pd.read_csv(csv_path, index_col=None, header=[0])
        loaded[tech_label] = df

        if generate_plots:
            _plot_capacity_factor_by_zone(df, tech_label, output_path, plot_label, source_suffix="rninja")

    return loaded


# ---------------------------------------------------------------------------
# IRENA workflows
# ---------------------------------------------------------------------------


def get_renewables_irena_data(
    countries,
    input_dir="input",
    output_dir="output",
    dataset_label="irena",
    input_files=None,
    include_capacity_factors=False,
    profile_year=2023,
    country_name_map=None,
    timezone_map=None,
):
    """Aggregate IRENA MSR solar/wind CSVs into SPLAT-style hourly load profiles."""
    if not countries:
        raise ValueError("countries must be provided to filter the IRENA MSR data.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    name_map = country_name_map or {}
    tz_map = timezone_map or extract_time_zone(countries, name_map)

    hourly_cols = [f'H{i}' for i in range(1, 8761)]
    date_index = pd.date_range(start=f'{profile_year}-01-01', periods=8760, freq='h')  # non-leap year

    def weighted_hourly_profile(group):
        weights = group['CapacityMW'].values.reshape(-1, 1)
        hourly_data = group[hourly_cols].values
        weighted_avg = (hourly_data * weights).sum(axis=0) / weights.sum()
        return pd.Series(weighted_avg, index=hourly_cols)

    def weighted_capacity_factor(group, cf_column):
        return pd.Series({
            'capacity_factor': (group[cf_column] * group['CapacityMW']).sum() / group['CapacityMW'].sum()
        })

    default_files = {
        'solar': 'SolarPV_BestMSRsToCover5%CountryArea.csv',
        'wind': 'Wind_BestMSRsToCover5%CountryArea.csv',
    }
    filenames = {**default_files, **(input_files or {})}
    tech_config = {
        'solar': {'filename': filenames['solar'], 'cf_col': 'CF'},
        'wind': {'filename': filenames['wind'], 'cf_col': 'CF100m'},
    }

    profiles, capacity_factors, paths = {}, {}, {}

    for tech, cfg in tech_config.items():
        file_path = require_file(
            input_path / cfg['filename'],
            hint=f"Download the IRENA MSR data and place {cfg['filename']} in {input_path}."
        )
        file_out = output_path / irena_output_filename(dataset_label, tech)

        use_columns = ['CtryName', 'CapacityMW', cfg['cf_col']] + hourly_cols
        data_msr = pd.read_csv(file_path, usecols=use_columns, header=0)
        available_countries = data_msr['CtryName'].unique()
        resolved_map = {}
        for ctry in countries:
            match = resolve_country_name(ctry, available_countries, verbose=True, allow_missing=True)
            if match:
                resolved_map[ctry] = match
            else:
                print(f"No IRENA {tech} data for '{ctry}' in {cfg['filename']}; skipping.")
        reverse_map = {v: k for k, v in resolved_map.items()}

        if not resolved_map:
            missing = ", ".join(sorted(countries))
            warning = (
                f"No IRENA {tech} rows matched any requested countries ({missing}) in {cfg['filename']}. "
                "The profile file will contain a warning summary instead of hourly data."
            )
            print(f"[WARNING] {warning}")
            Path(file_out).parent.mkdir(parents=True, exist_ok=True)
            Path(file_out).write_text(warning, encoding="utf-8")
            profiles[tech] = pd.DataFrame()
            paths[tech] = file_out
            continue

        data_msr = data_msr[data_msr['CtryName'].isin(resolved_map.values())].copy()
        data_msr['CtryName'] = data_msr['CtryName'].map(reverse_map).fillna(data_msr['CtryName'])

        if data_msr.empty:
            warning = (
                f"No {tech} rows found for the requested countries in {file_path}. "
                "Ensure the MSR file is present and contains the configured country names; "
                "writing a warning file instead."
            )
            print(f"[WARNING] {warning}")
            Path(file_out).parent.mkdir(parents=True, exist_ok=True)
            Path(file_out).write_text(warning, encoding="utf-8")
            profiles[tech] = pd.DataFrame()
            paths[tech] = file_out
            continue

        if include_capacity_factors:
            cf_stats = (
                data_msr.groupby('CtryName')
                .apply(weighted_capacity_factor, cf_column=cfg['cf_col'])
                .reset_index()
                .rename(columns={'CtryName': 'zone'})
            )
            capacity_factors[tech] = cf_stats

        hourly_input = data_msr.set_index(['CtryName', 'CapacityMW'])[hourly_cols].reset_index()
        hourly_profile = hourly_input.groupby('CtryName').apply(weighted_hourly_profile).reset_index()

        df_long = hourly_profile.melt(id_vars='CtryName', var_name='Hour', value_name='value')
        df_long['hour_index'] = df_long['Hour'].str.extract('H(\\d+)').astype(int) - 1
        df_long['timestamp'] = df_long['hour_index'].map(lambda i: date_index[i])

        df_long['CtryName'] = df_long['CtryName'].astype(str)
        df_long['timestamp_utc'] = df_long.apply(lambda row: convert_to_utc(row, tz_map), axis=1)
        df_long['month'] = df_long['timestamp_utc'].dt.month
        df_long['day'] = df_long['timestamp_utc'].dt.day
        df_long['hour'] = df_long['timestamp_utc'].dt.hour

        df_final = (
            df_long.rename(columns={'CtryName': 'zone'})
            .set_index(['zone', 'month', 'day', 'hour'])['value']
            .to_frame()
            .rename(columns={'value': profile_year})
            .sort_index()
        )
        df_final = _flatten_year_columns(df_final)
        df_final.index = df_final.index.set_names(['zone', 'month', 'day', 'hour'])
        df_out = df_final.reset_index()
        if df_out.empty:
            warning = (
                f"No hourly rows generated for the requested IRENA {tech} profiles ({countries}); "
                "writing a warning file and skipping CSV export."
            )
            print(f"[WARNING] {warning}")
            Path(file_out).parent.mkdir(parents=True, exist_ok=True)
            Path(file_out).write_text(warning, encoding="utf-8")
        else:
            df_out.to_csv(file_out, index=False)

        profiles[tech] = df_final
        paths[tech] = file_out

    return {
        'profiles': profiles,
        'capacity_factors': capacity_factors if include_capacity_factors else {},
        'paths': paths,
        'timezones': tz_map,
    }


def run_irena_workflow(
    dataset_label,
    countries,
    input_dir="input",
    output_dir="output",
    input_files=None,
    include_capacity_factors=False,
    profile_year=2023,
    country_name_map=None,
    timezone_map=None,
    generate_plots=True,
):
    """End-to-end get IRENA data workflow."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    result = get_renewables_irena_data(
        countries=countries,
        input_dir=input_path,
        output_dir=output_path,
        dataset_label=dataset_label,
        input_files=input_files,
        include_capacity_factors=include_capacity_factors,
        profile_year=profile_year,
        country_name_map=country_name_map,
        timezone_map=timezone_map,
    )

    plot_paths = {}
    if generate_plots:
        from plots import make_boxplot, make_heatmap  # local import to avoid hard dependency

        tech_labels = {'solar': 'PV', 'wind': 'Wind'}
        plot_label = f"_{dataset_label}" if dataset_label else ""
        for tech, df in result['profiles'].items():
            tech_label = tech_labels.get(tech, tech.title())
            df_plot = df.reset_index()
            per_zone_paths = _plot_capacity_factor_by_zone(
                df_plot,
                tech_label,
                output_path,
                plot_label,
                source_suffix="irena",
            )
            plot_paths[tech] = {'per_zone': per_zone_paths}

    result['plots'] = plot_paths
    return result


# ---------------------------------------------------------------------------
# Profile helpers and exports
# ---------------------------------------------------------------------------


def find_representative_year(df, method='average_profile'):
    """Find the representative year from a wide hourly profile DataFrame."""
    dict_info, repr_year = {}, None
    if method == 'average_cf':
        temp = df.sum().round(3)
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


def export_epm_full_timeseries(
    vre_profiles,
    load_profile=None,
    output_dir="output",
    year_column=None,
    load_zone=None,
    demand_filename="pDemandProfile_full.csv",
):
    """Export full-hour EPM inputs (pHours=1) for VRE and optional load before representative-day reduction."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_cols = {'zone', 'month', 'day', 'hour'}
    hours_cols = [f"t{i}" for i in range(1, 25)]

    def _load_df(obj):
        return pd.read_csv(obj) if isinstance(obj, (str, Path)) else obj.copy()

    def _month_label(val):
        try:
            return f"m{int(val):02d}"
        except (TypeError, ValueError):
            return str(val)

    def _pick_value_column(df, tech_key):
        override = year_column.get(tech_key) if isinstance(year_column, dict) else year_column
        if override is not None:
            if override in df.columns:
                return override
            raise ValueError(f"Column '{override}' not found in {tech_key} profile.")
        candidates = [c for c in df.columns if c not in index_cols]
        if not candidates:
            raise ValueError(f"No value columns found for {tech_key}.")
        year_like = [c for c in candidates if (isinstance(c, (int, float))) or (isinstance(c, str) and c.isdigit())]
        return year_like[0] if year_like else candidates[0]

    def _prepare_hourly(df, label, zone_fallback=None):
        df = df.copy()
        if 'timestamp' in df.columns and {'month', 'day', 'hour'}.difference(df.columns):
            ts = pd.to_datetime(df['timestamp'])
            df['month'] = ts.dt.month
            df['day'] = ts.dt.day
            df['hour'] = ts.dt.hour
        if 'hour' in df.columns and df['hour'].min() >= 1 and df['hour'].max() <= 24:
            df['hour'] = df['hour'] - 1  # normalize 1-24 to 0-23

        missing = {'month', 'day', 'hour'}.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} for {label}.")

        if 'zone' not in df.columns:
            df['zone'] = zone_fallback or 'zone_1'

        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['hour'] = df['hour'].astype(int)
        return df

    vre_long = []
    calendar_parts = []
    for tech, src in vre_profiles.items():
        df_raw = _load_df(src)
        df_norm = _prepare_hourly(df_raw, label=tech)
        value_col = _pick_value_column(df_norm, tech)
        df_subset = df_norm[['zone', 'month', 'day', 'hour', value_col]].rename(columns={value_col: 'value'})
        df_subset['fuel'] = {'solar': 'PV', 'wind': 'Wind'}.get(str(tech).lower(), tech)
        vre_long.append(df_subset)
        calendar_parts.append(df_subset[['month', 'day']].drop_duplicates())

    if not vre_long:
        raise ValueError("At least one VRE profile is required.")

    load_df = None
    if load_profile is not None:
        load_raw = _load_df(load_profile)
        load_norm = _prepare_hourly(load_raw, label='load', zone_fallback=load_zone or 'load')

        load_candidates = [c for c in load_norm.columns if c not in index_cols.union({'timestamp'})]
        if not load_candidates:
            raise ValueError("No value column found in load_profile.")
        load_value_col = load_candidates[0]

        load_df = load_norm[['zone', 'month', 'day', 'hour', load_value_col]].rename(columns={load_value_col: 'Load'})
        calendar_parts.append(load_df[['month', 'day']].drop_duplicates())

    if not calendar_parts:
        raise ValueError("No profiles provided to export.")

    calendar = pd.concat(calendar_parts, ignore_index=True).drop_duplicates().sort_values(['month', 'day'])
    calendar['month_label'] = calendar['month'].apply(_month_label)
    calendar['daytype'] = calendar.groupby('month_label').cumcount() + 1
    calendar['daytype'] = calendar['daytype'].apply(lambda x: f'd{x}')

    phours = calendar[['month_label', 'daytype']].drop_duplicates().rename(columns={'month_label': 'month'})
    for col in hours_cols:
        phours[col] = 1
    phours_path = output_path / "pHours_full.csv"
    phours.to_csv(phours_path, index=False)

    day_lookup = calendar[['month', 'day', 'month_label', 'daytype']]

    def _pivot_profile(df, index_fields, value_column):
        df = df.merge(day_lookup, on=['month', 'day'], how='left')
        if df['daytype'].isna().any():
            missing = df[df['daytype'].isna()][['month', 'day']].drop_duplicates()
            raise ValueError(f"Missing daytype mapping for rows:\n{missing}")
        wide = (
            df.pivot_table(index=index_fields + ['month_label', 'daytype'], columns='hour', values=value_column)
            .reindex(columns=range(24), fill_value=0)
        )
        wide.columns = hours_cols
        wide = wide.reset_index().rename(columns={'month_label': 'month'})
        return wide

    vre_concat = pd.concat(vre_long, ignore_index=True)
    vre_wide = _pivot_profile(vre_concat, ['zone', 'fuel'], value_column='value')
    vre_path = output_path / "pVREProfile_full.csv"
    vre_wide.to_csv(vre_path, index=False, float_format='%.5f')

    demand_path = None
    if load_df is not None:
        max_per_zone = load_df.groupby('zone')['Load'].transform('max')
        zeros = max_per_zone == 0
        if zeros.any():
            raise ValueError(f"Cannot normalize demand: zero peak load for zones {load_df.loc[zeros, 'zone'].unique()}.")
        load_df['Load'] = load_df['Load'] / max_per_zone

        load_wide = _pivot_profile(load_df, ['zone'], value_column='Load')
        demand_filename = demand_filename or "pDemandProfile_full.csv"
        demand_path = Path(demand_filename)
        demand_path = demand_path if demand_path.is_absolute() else output_path / demand_path
        load_wide.to_csv(demand_path, index=False, float_format='%.5f')

    return {
        'pHours': phours_path,
        'pVREProfile': vre_path,
        'pDemandProfile': demand_path,
    }


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    sample_output = BASE_DIR / "output_workflow" / "vre_standalone"
    sample_output.mkdir(parents=True, exist_ok=True)
    sample_excel = BASE_DIR / "dataset" / "Global-Integrated-Power-April-2025.xlsx"
    print(f"[vre-standalone] Preparing sample GAP extraction to {sample_output}")
    try:
        _, locations, csv_path = gap_rninja_coordinates(
            xlsx_path=sample_excel,
            countries=["Albania"],
            output_dir=sample_output,
        )
        print(f"[vre-standalone] Exported GAP subset to {csv_path}")
        print(f"[vre-standalone] Locations for RNinja: {locations}")
        rninja_outdir = sample_output / "rninja"
        print(f"[vre-standalone] Requesting RNinja sample year to {rninja_outdir}")
        try:
            run_renewables_ninja_workflow(
                locations=locations,
                start_year=2023,
                end_year=2024,
                dataset_label="sample",
                input_dir=rninja_outdir,
                output_dir=rninja_outdir,
                generate_plots=True,
            )
            print(f"[vre-standalone] RNinja CSVs written under {rninja_outdir}")
        except RuntimeError as api_err:
            print(f"[vre-standalone] RNinja skipped: {api_err}")
    except FileNotFoundError as exc:
        print(f"[vre-standalone] Skipped: {exc}")
