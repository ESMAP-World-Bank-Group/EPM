"""Representative days pipeline helpers (clustering, optimization I/O, EPM formatting).

Key entry points
----------------
- `run_representative_days_pipeline`: high-level orchestration from raw inputs to repr-days outputs.
- `prepare_input_timeseries`: normalize raw inputs (month/day/hour) to season/day/hour CSVs.
- `cluster_data_new` and `get_special_days_clustering`: identify representative/special days before optimization.
"""

import requests
import pandas as pd
import time
import os
import numpy as np
import gams.transfer as gt
import subprocess
import shutil
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Tuple, Union
import matplotlib.pyplot as plt
import calendar
import warnings
import logging
import sys
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from IPython.display import display
try:
    from representative_days.utils import (
        _log_nan_time_index,
        month_to_season,
        load_and_clean_timeseries,
    )
except ImportError:  # pragma: no cover - fallback when run as script
    from utils import (
        _log_nan_time_index,
        month_to_season,
        load_and_clean_timeseries,
    )


logging.basicConfig(level=logging.WARNING)  # Configure logging level
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Time-series preparation helpers
# --------------------------------------------------------------------------- #

def prepare_input_timeseries(
    input_path: Union[str, Path],
    seasons_map: dict,
    output_dir: Union[str, Path],
    zones_to_exclude=None,
    value_column: str = 'value',
    year_label: Union[int, str] = 2018,
    verbose: bool = True,
) -> Tuple[Path, pd.DataFrame]:
    """Clean and season-aggregate a raw time-series CSV, mirroring the notebook pre-processing.

    Parameters
    ----------
    input_path : str or Path
        Path to the raw CSV with columns ``zone``, ``month`` (or ``season``), ``day``, ``hour`` and a value column.
    seasons_map : dict
        Mapping month -> season used to regroup months into broader seasons.
    output_dir : str or Path
        Directory where the season-aggregated CSV will be written.
    zones_to_exclude : list, optional
        Zones to drop from the dataset.
    value_column : str, default 'value'
        Name of the column holding the time-series values. If not present, the first non-index column is used.
    year_label : int or str, default 2018
        Column name to use for the value series (e.g., rename ``value`` to ``2018`` to match historical year convention).
    verbose : bool, default True
        Whether to warn when the year is incomplete at zone/month/day/hour level.

    Returns
    -------
    Tuple[Path, pd.DataFrame]
        - Path to the cleaned CSV (named ``<original>_season.csv`` inside ``output_dir``).
        - Cleaned DataFrame used to create that file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    cleaned_df, value_col = load_and_clean_timeseries(
        input_path=input_path,
        zones_to_exclude=zones_to_exclude,
        value_column=value_column,
        rename_value_to=year_label,
        normalize=True,
        drop_feb_29=True,
        check_complete_year=True,
        verbose=verbose,
    )

    _log_nan_time_index(
        cleaned_df,
        time_cols=('month', 'day', 'hour'),
        value_cols=[value_col],
        label=f'{input_path.name} before month_to_season',
    )

    # Transform monthly to seasonal data
    df = month_to_season(cleaned_df, seasons_map, other_columns=['zone'])

    _log_nan_time_index(
        df,
        time_cols=('season', 'day', 'hour'),
        value_cols=[value_col],
        label=f'{input_path.name} after month_to_season',
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{input_path.stem}_season{input_path.suffix}'
    df.to_csv(output_path, float_format='%.4f', index=False)
    logger.info('Processed %s -> %s', input_path, output_path)
    return output_path, df


# --------------------------------------------------------------------------- #
# Representative year selection and profile prep
# --------------------------------------------------------------------------- #

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


def prepare_energy_profiles(filenames, verbose: bool = True):
    """Format and validate the energy data read from CSV inputs.

    Parameters
    ----------
    filenames: dict
        Dictionary with the filenames.
    verbose: bool, default True
        Whether to print informative messages about the representative year and
        validation checks.

    Returns
    -------
    pd.DataFrame
        DataFrame with the representative year aggregated per tech.
    """
    # Find the representative year

    energy_dfs = []
    for tech, filename in filenames.items():
        df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3])
        df = df.reset_index()
        df['tech'] = tech
        df = df.set_index(['zone', 'season', 'day', 'hour', 'tech'])
        energy_dfs.append(df)

    if not energy_dfs:
        raise ValueError('No energy files were provided to prepare energy profiles.')

    combined_energy = pd.concat(energy_dfs, ignore_index=False)
    repr_year = find_representative_year(combined_energy)
    if verbose:
        print(f'Representative year {repr_year}')

    repr_year_data = combined_energy.loc[:, repr_year].unstack('tech').reset_index()

    energy_profiles = repr_year_data.sort_values(by=['zone', 'season', 'day', 'hour'], ascending=True)

    keys_to_merge = ['PV', 'Wind', 'Load', 'ROR']
    keys_to_merge = [i for i in keys_to_merge if i in energy_profiles.keys()]

    if verbose:
        display('Annual capacity factor (%):', energy_profiles.groupby('zone')[keys_to_merge].mean().reset_index())
    if len(energy_profiles.zone.unique()) > 1:  # handling the case with multiple zones to rename columns
        energy_profiles = energy_profiles.set_index(['zone', 'season', 'day', 'hour']).unstack('zone')
        energy_profiles.columns = ['_'.join([idx0, idx1]) for idx0, idx1 in energy_profiles.columns]
        energy_profiles = energy_profiles.reset_index()
    else:
        energy_profiles = energy_profiles.drop('zone', axis=1)
    return energy_profiles


# --------------------------------------------------------------------------- #
# Clustering helpers (k-means and special days)
# --------------------------------------------------------------------------- #

def cluster_data_new(df_energy, n_clusters=10, columns=None):
    """
    Perform KMeans clustering on daily energy data to identify representative clusters by season.

    This function applies KMeans clustering to energy data grouped by season, where each data point
    represents the sum of values for a single day. It enables identification of clustered daily profiles
    (e.g., for PV, Wind, Load) separately for each season.

    Parameters:
        df_energy (pd.DataFrame):
            DataFrame containing energy-related features with columns ['season', 'day', 'hour', 'PV', 'Wind', 'Load'].
        n_clusters (int, optional):
            Number of clusters to create. Defaults to 10.
        columns (list, optional):
            List of feature columns used for clustering. If None, all columns except ['season', 'day', 'hour'] are used.

    Returns:
        tuple:
            - pd.DataFrame: Original DataFrame with assigned cluster labels.
            - pd.DataFrame: Representative days closest to cluster centroids.
            - pd.DataFrame: Cluster centroids with associated probabilities.
    """

    # Select relevant columns for clustering
    if columns is None:
        columns = [i for i in df_energy.columns if i not in ['season', 'day', 'hour']]

    # Find closest days to centroids
    def find_closest_days(df, centroids, features):
        closest_days = []
        for i, centroid in enumerate(centroids):
            # Compute Euclidean distance between each row and the centroid
            distances = np.linalg.norm(df[features] - centroid, axis=1)
            closest_idx = distances.argmin()  # Find index of closest day
            closest_days.append(df.iloc[closest_idx])  # Store closest day
        closest_days = pd.DataFrame(closest_days)
        closest_days.index = range(len(centroids))
        return closest_days

    # Compute cluster probabilities
    def assign_probabilities(labels: np.ndarray, n_clusters: int) -> pd.Series:
        """
        Calculate probabilities for each cluster based on frequency of occurrence.
        """
        unique, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return pd.Series(probabilities, index=range(n_clusters))

    df_closest_days = []
    centroids_df = []
    df_tot = df_energy.copy()
    for season, df_season in df_tot.groupby('season'):
        df_season_cluster = df_season.copy()
        df_season_cluster = df_season_cluster.groupby(['season', 'day'])[columns].sum().reset_index()

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_season_cluster['Cluster'] = kmeans.fit_predict(df_season_cluster[columns])
        df_tot = df_tot.merge(df_season_cluster[['season', 'day', 'Cluster']], on=['season', 'day'], how='left',
                              suffixes=('', '_temp'))

        # Fill only missing values in df_tot['Cluster']
        if 'Cluster_temp' in df_tot.columns:
            df_tot['Cluster'] = df_tot['Cluster'].combine_first(df_tot['Cluster_temp'])
            df_tot = df_tot.drop(['Cluster_temp'], axis=1)

        # Extract cluster centers
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns, index=range(n_clusters))
        cluster_probabilities = assign_probabilities(df_season_cluster['Cluster'].values, n_clusters)
        centroids = pd.concat([cluster_probabilities.to_frame().rename(columns={0: 'probability'}), cluster_centers],
                              axis=1)
        centroids['season'] = season
        centroids = centroids.reset_index().rename(columns={'index': 'Cluster'})
        centroids_df.append(centroids)

        df_closest_days.append(find_closest_days(df_season_cluster, cluster_centers.values, columns))

    df_closest_days, centroids_df = pd.concat(df_closest_days, axis=0, ignore_index=True), pd.concat(centroids_df,
                                                                                                     axis=0,
                                                                                                     ignore_index=True)
    df_closest_days = df_closest_days.merge(centroids_df[['Cluster', 'probability', 'season']],
                                            on=['Cluster', 'season'], how='left')
    return df_tot, df_closest_days, centroids_df


def select_representative_series_hierarchical(path_data_file: str, n: int, method: str = 'ward',
                                              metric: str = 'euclidean', scale: bool = True,
                                              scale_method: str = 'standard'):
    """
    Reduce dimensionality of time series features using hierarchical clustering, keeping only real series.

    Parameters:
        path_data_file (str):
            Path to the dataframe with columns as different time series (e.g., 'Wind_ZM', 'PV_BW', 'Corr_Load_TZ__PV_MZ', etc.)
            and rows as time steps (e.g., hours in the season).
        n (int):
            Number of representative time series to keep.
        method (str):
            Linkage method for hierarchical clustering (e.g., 'ward', 'average', 'complete').
        metric (str):
            Distance metric (e.g., 'euclidean', 'cosine', 'correlation').

    Returns:
        list:
            Names of selected time series (columns) to retain.
        pd.DataFrame:
            Reduced DataFrame with only the selected columns.
    """
    path_data_file = os.path.join(os.getcwd(), path_data_file)
    df = pd.read_csv(path_data_file, index_col=[0,1,2])

    # Transpose to get features as rows
    series_matrix = df.T.values  # shape: (n_series, n_timesteps)
    series_names = df.columns.tolist()

    # Optional scaling
    if scale:
        if scale_method == 'standard':
            scaler = StandardScaler()
        elif scale_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("scale_method must be 'standard' or 'minmax'")
        series_matrix = scaler.fit_transform(series_matrix)

    # Compute pairwise distance
    distance_matrix = pdist(series_matrix, metric=metric)

    # Hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)

    # Cut the dendrogram into n clusters
    cluster_labels = fcluster(linkage_matrix, t=n, criterion='maxclust')

    # For each cluster, find the series closest to the centroid
    selected_series = []
    for cluster_id in range(1, n + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = series_matrix[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)

        # Compute distance to centroid
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        selected_series.append(series_names[closest_idx])

    base, ext = os.path.splitext(path_data_file)
    path_data_file_selection = f'{base}_selection{ext}'

    df[selected_series].to_csv(path_data_file_selection, index=True)
    print(f'File saved at {path_data_file_selection}')

    return selected_series, df[selected_series], path_data_file_selection


# --------------------------------------------------------------------------- #
# Special day identification
# --------------------------------------------------------------------------- #

def get_special_days_clustering(df_closest_days, df_tot, threshold=None):
    """
    Identify special days based on clustering results.

    This function selects extreme days (e.g., lowest PV, lowest Wind, highest Load) as centroids of clusters,
    while ensuring that clusters representing a large share of the data are excluded.

    Parameters:
        df_closest_days (pd.DataFrame):
            DataFrame of closest representative days to cluster centroids.
        df_tot (pd.DataFrame):
            Original dataset with all time series data.
        threshold (float, optional):
            Probability threshold for excluding large clusters. If None, a data-driven threshold is
            computed per season (75th percentile of cluster probabilities).

    Returns:
        tuple:
            - pd.DataFrame: Special days with associated weights.
            - pd.DataFrame: Updated df_tot with special days removed.
    """
    df_tot = df_tot.copy()
    special_days = []
    indices_to_remove = set()
    for season, df_season in df_closest_days.groupby('season'):
        seasonal_threshold = threshold
        if seasonal_threshold is None:
            seasonal_threshold = float(df_season['probability'].quantile(0.75))
            if pd.isna(seasonal_threshold):
                seasonal_threshold = 1.0

        # first special day is based on minimum PV production across all zones
        add_special_days(feature='PV', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='min', threshold=seasonal_threshold)

        # second special day is based on minimum Wind production across all zones
        add_special_days(feature='Wind', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='min', threshold=seasonal_threshold)

        # third special day is based on maximum peak demand across all zones
        add_special_days(feature='Load', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='max', threshold=seasonal_threshold)


    # Convert special days to DataFrame
    df_special_days = pd.DataFrame(special_days)
    df_special_days = df_special_days.drop_duplicates()
    df_tot = df_tot.drop(index=indices_to_remove)  # we remove the days corresponding to clusters which have been included as special days
    df_tot = df_tot.drop(columns=['Cluster'])  # no longer needed

    return df_special_days, df_tot


def add_special_days(feature, df, df_days, season, special_days, indices_to_remove, rule='min', threshold=0.07):
    """
    Identifies and adds a representative 'special day' from a given season based on extreme values
    of a selected feature (e.g., PV, Load, Wind) across multiple zones.

    The function looks for the day within a season that corresponds to the most extreme value
    (minimum or maximum depending on the `rule`) of the summed feature across all zones. It excludes
    high-probability clusters (based on the `threshold`) because those frequent profiles are already
    captured by the representative clusters; this step isolates rare-but-important extremes (e.g.,
    very low PV or very high load) so they are forced into the optimization as special days. If no
    clusters remain below the threshold, it falls back to using all clusters to ensure a special day
    is still selected. When an extreme (min/max) occurs in a high-probability cluster, that cluster
    is added back so the true extreme is not lost due to filtering.
    If the most extreme day is already present in `special_days`, the function iteratively selects
    the next most extreme day that hasn’t been used.

    Parameters:
        feature (str):
            The feature to analyze (e.g., 'PV', 'Wind', 'Load').
        df (pd.DataFrame):
            DataFrame containing cluster probabilities and feature values.
        df_days (pd.DataFrame):
            Complete dataset with all time series data.
        season (str):
            Current season being analyzed.
        special_days (list):
            List of previously identified special days.
        indices_to_remove (set):
            Set to track days to be removed from df_days.
        rule (str, optional):
            Whether to select the 'min' or 'max' extreme value. Defaults to 'min'.
        threshold (float, optional):
            Probability threshold for excluding large clusters. Defaults to 0.07.

    Returns:
        list: Updated list of special days.
    """
    assert rule in ['min', 'max'], "Rule for selecting cluster should be either 'min' or 'max'."
    columns = [col for col in df.columns if feature in col]
    df = df.copy()

    if len(columns) > 0:
        df_filtered = df[df['probability'] < threshold].copy()
        target_df = df_filtered if not df_filtered.empty else df.copy()
        target_df.loc[:, feature] = target_df.loc[:, columns].sum(axis=1)

        # Ensure the global extreme (min/max) is still eligible even if it sits in a high-probability cluster.
        full_extreme = df.copy()
        full_extreme.loc[:, feature] = full_extreme.loc[:, columns].sum(axis=1)
        extreme_val = full_extreme[feature].min() if rule == 'min' else full_extreme[feature].max()
        extreme_rows = full_extreme[full_extreme[feature] == extreme_val]
        if not extreme_rows['Cluster'].isin(target_df['Cluster']).any():
            target_df = pd.concat([target_df, extreme_rows], ignore_index=True)
            target_df = target_df.drop_duplicates(subset=['Cluster'])

        df = target_df

        def find_next_special_day(df, special_days, rule):
            for i in range(len(df)):  # Iterate over all ranked rows
                cluster = df.nsmallest(i + 1, feature).iloc[-1:] if rule == 'min' else df.nlargest(i + 1, feature).iloc[-1:]
                special_day = tuple(cluster[['season', 'day']].values[0])
                if special_day not in {entry['days'] for entry in special_days}:
                    return cluster
            return None  # This should never be reached unless all days are special (unlikely)

        # Find the most extreme day based on rule
        cluster = df.nsmallest(1, feature) if rule == 'min' else df.nlargest(1, feature)
        special_day = tuple(cluster[['season', 'day']].values[0])

        if special_day in {entry['days'] for entry in special_days}:
            # If already included, find the next available extreme day
            cluster = find_next_special_day(df, special_days, rule)

        if cluster is not None:
            cluster_id = cluster['Cluster'].values[0]
            special_day = tuple(cluster[['season', 'day']].values[0])
            cluster_weight = (df_days[(df_days['season'] == season) & (df_days['Cluster'] == cluster_id)].shape[0]) // 24
            rule_label = f'{rule}_{feature}'
            special_days.append({'days': special_day, 'weight': cluster_weight, 'rule': rule_label})
            indices_to_remove.update(df_days[(df_days['season'] == season) & (df_days['Cluster'] == cluster_id)].index)
        else:
            raise ValueError(f"All days are already special days.")

    return special_days


def find_special_days(df_energy, columns=None):
    """Find special days within the representative year.

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.

    Returns
    -------
    special_days: pd.DataFrame
        DataFrame with the special days.
    """
    # Find the special days within the representative year

    df = df_energy.copy()

    def get_special_day(df, feature, rule):
        if rule == 'min':
            min_prod = df.groupby(['season', 'day'])[feature].sum().unstack().idxmin(axis=1)
            min_prod = list(min_prod.items())
            return min_prod
        else:  # rule = 'max'
            max_load = df.groupby(['season', 'day'])[feature].sum().unstack().idxmax(axis=1)
            max_load = list(max_load.items())
            return max_load

    special_days = {}

    feature = 'PV'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='min')
        special_days.update({feature: special_day})

    feature = 'Wind'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='min')
        special_days.update({feature: special_day})

    feature = 'Load'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='max')
        special_days.update({feature: special_day})

    # Format special days
    special_days = sorted([item for sublist in special_days.values() for item in sublist])
    special_days = pd.Series(special_days)
    special_days = pd.concat((special_days, pd.Series(1, index=special_days.index)), keys=['days', 'weight'], axis=1)
    special_days = special_days.set_index('days').groupby('days').first().reset_index()

    return special_days


def removed_special_days(df_energy, special_days):
    # Remove all lines in dict_info
    for special_day in special_days['days']:
        df_energy = df_energy[~((df_energy['season'] == special_day[0]) & (df_energy['day'] == special_day[1]))]

    return df_energy


def calculate_pairwise_correlation(df):
    """ Calculate correlation between all columns in a DataFrame on a row-by-row basis,
    and store the result in a new column for each pair.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the energy data.
    """
    columns = [i for i in df.columns if i not in ['season', 'day', 'hour']]
    new_columns = {}

    # Precompute means for efficiency
    means = {col: df[col].mean() for col in columns}

    for col1, col2 in combinations(columns, 2):
        corr_col_name = f"{col1}_{col2}_corr"
        new_columns[corr_col_name] = (df[col1] - means[col1]) * (df[col2] - means[col2])

    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df


def format_optim_repr_days(df_energy, folder_process_data, data_dir=None):
    """Format the data for the optimization.

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    name_data: str
        Name of the zone.
    folder_process_data: str
        Path to save optimization artifacts.
    data_dir: str, optional
        Directory to save the formatted optimization CSV. Defaults to ``folder_process_data``.
    """
    df_formatted_optim = df_energy.copy()
    # Add correlation
    df_formatted_optim = calculate_pairwise_correlation(df_formatted_optim)
    df_formatted_optim.set_index(['season', 'day', 'hour'], inplace=True)
    df_formatted_optim.index.names = [''] * 3
    # TODO: check this, currently removing zone
    # # Add header to the DataFrame with the name of the zone
    # df_formatted_optim = pd.concat([df_formatted_optim], keys=[name_data], axis=1)
    # # Invert the order of levels
    # df_formatted_optim = df_formatted_optim.swaplevel(0, 1, axis=1)

    target_dir = data_dir or folder_process_data
    os.makedirs(target_dir, exist_ok=True)
    path_data_file = os.path.join(target_dir, 'data_formatted_optim.csv')
    df_formatted_optim.to_csv(path_data_file, index=True)
    print('File saved at:', path_data_file)

    return df_formatted_optim, path_data_file


# --------------------------------------------------------------------------- #
# GAMS optimization launcher and parsing
# --------------------------------------------------------------------------- #

def launch_optim_repr_days(path_data_file, folder_process_data, nbr_days=3,
                           main_file='OptimizationModelZone.gms', nbr_bins=10):
    """Launch the representative days optimization through GAMS.

    Parameters
    ----------
    path_data_file: str
        Path to the data file.
    folder_process_data: str
        Path to save the .gms file.
    nbr_days: int
        Number of representative days to target.
    main_file: str
        Name (or absolute path) of the GAMS model to run.
    nbr_bins: int
        Number of bins to use in the GAMS settings helper.
    """

    def generate_bins_file(n_bins, save_dir="gams"):
        """
        Generates a CSV file with bin names (b1, b2, ..., bn).

        Parameters:
            n_bins (int): Number of bins
            save_dir (str): Directory where to save the file
            filename (str): Name of the output file

        Returns:
            str: Full path to the saved file
        """
        # Create list of bin names
        bin_names = [f"b{i}" for i in range(1, n_bins + 1)]

        # Create DataFrame
        df = pd.DataFrame({"BINS": bin_names})

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        filename = f"bins_settings_{n_bins}.csv"  # Append number of bins to filename

        # Full path
        file_path = os.path.join(save_dir, filename)

        # Save to CSV
        df.to_csv(file_path, index=False)

        print(f"File saved to: {file_path}")
        return filename

    if shutil.which("gams") is None:
        raise FileNotFoundError("GAMS executable not found in PATH. Please install GAMS or update PATH.")

    # Generate bins settings file
    path_main_file = Path(main_file)
    if not path_main_file.is_absolute():
        path_main_file = Path(os.getcwd()) / 'gams' / main_file

    setting_filename = generate_bins_file(n_bins=nbr_bins)
    path_setting_file = Path(os.getcwd()) / 'gams' / setting_filename

    path_data_file = Path(path_data_file)
    if not path_data_file.is_absolute():
        path_data_file = Path(os.getcwd()) / path_data_file

    if not path_main_file.exists():
        raise ValueError(f'Gams file: {path_main_file} not found.')
    elif not path_setting_file.exists():
        raise ValueError(f'Settings file: {path_setting_file} not found.')
    elif not path_data_file.exists():
        raise ValueError(f'Data file: {path_data_file} not found.')
    else:
        command = ["gams", str(path_main_file),
                   f"--data {path_data_file}",
                   f"--settings {path_setting_file}",
                   f"--N {nbr_days}"]

    # Print the command
    cwd = Path(os.getcwd()) / folder_process_data
    print('Launch GAMS code')
    if sys.platform.startswith("win"):  # If running on Windows
        print("Command to execute:", ' '.join(command))
        subprocess.run(' '.join(command), cwd=str(cwd), shell=True, stdout=subprocess.DEVNULL)
    else:  # For Linux or macOS
        subprocess.run(command, cwd=str(cwd), stdout=subprocess.DEVNULL)
    print('End GAMS code')

    # TODO: Check if the results exist


def parse_repr_days(folder_process_data, special_days):
    """Parse the results of the optimization.

    Parameters
    ----------
    folder_process_data: str
        Path to the folder with the results.
    special_days: pd.DataFrame
        DataFrame with the special days.

    Returns
    -------
    repr_days: pd.DataFrame
    """

    def extract_gdx(file):
        """
        Extract information as pandas DataFrame from a gdx file.

        Parameters
        ----------
        file: str
            Path to the gdx file

        Returns
        -------
        epm_result: dict
            Dictionary containing the extracted information
        """
        df = {}
        container = gt.Container(file)
        for param in container.getVariables():
            if container.data[param.name].records is not None:
                df[param.name] = container.data[param.name].records.copy()

        return df

    # Extract the results
    results_optim = extract_gdx(os.path.join(folder_process_data, 'Results.gdx'))

    # From gdx result to pandas DataFrame
    weight = results_optim['w'].copy()
    weight = weight[~np.isclose(weight['level'], 0, atol=1e-6)]

    repr_days = pd.concat((
        weight.apply(lambda x: (x['s'], x['d']), axis=1),
        weight['level']), keys=['days', 'weight'], axis=1)
    repr_days['rule'] = 'representative'

    repr_days['weight'] = repr_days['weight'].round().astype(int)

    # Add special days
    if special_days is not None and not special_days.empty:
        if 'rule' not in special_days.columns:
            special_days = special_days.assign(rule='special')
    repr_days = pd.concat((special_days, repr_days), axis=0, ignore_index=True)

    print('Number of days: {}'.format(repr_days.shape[0]))
    print('Total weight: {}'.format(repr_days['weight'].sum()))

    # Format the data
    repr_days['season'] = repr_days['days'].apply(lambda x: x[0])
    repr_days['day'] = repr_days['days'].apply(lambda x: x[1])
    repr_days.drop(columns=['days'], inplace=True)
    repr_days = repr_days.loc[:, ['season', 'day', 'weight', 'rule']]
    repr_days = repr_days.astype({'season': int, 'day': int, 'weight': float})
    repr_days.sort_values(['season'], inplace=True)
    repr_days['daytype'] = repr_days.groupby('season').cumcount() + 1

    repr_days['season'] = repr_days['season'].apply(lambda x: f'Q{x}')

    # Update daytype naming to d1, d2...
    repr_days['daytype'] = repr_days['daytype'].apply(lambda x: f'd{x}')

    print(repr_days.groupby('season')['weight'].sum())

    return repr_days


def export_repr_days_summary(df_energy, repr_days, special_days, output_dir):
    """Export per-day capacity factor summary and benchmarks to CSV.

    The output includes selected days (representative or special) with average capacity factors per tech,
    plus benchmark rows with min/avg/max capacity factors across the full dataset. Wrapped in try/except
    to avoid interrupting the pipeline on bad inputs.
    """
    try:
        output_path = Path(output_dir) / 'representative_days_summary.csv'
        df_energy_local = df_energy.copy()
        repr_days_local = repr_days.copy()

        # Identify numeric tech columns (exclude time/weight identifiers).
        numeric_cols = df_energy_local.select_dtypes(include=[np.number]).columns
        tech_cols = [c for c in numeric_cols if c not in {'season', 'day', 'hour', 'weight'}]
        if not tech_cols:
            raise ValueError('No technology columns found to summarize.')

        # Build a lookup for special days using integer season/day tuples.
        special_set = set()
        if special_days is not None and not special_days.empty:
            if 'days' in special_days.columns:
                special_set = {tuple(map(int, d)) for d in special_days['days']}
            elif {'season', 'day'}.issubset(special_days.columns):
                special_set = {tuple(map(int, t)) for t in special_days[['season', 'day']].itertuples(index=False, name=None)}

        # Normalize season for repr_days (stored as Q1/Q2...); keep original label for readability.
        repr_days_local['season_int'] = repr_days_local['season'].astype(str).str.extract(r'(\d+)').astype(int)
        total_weight = repr_days_local['weight'].sum()

        rows = []
        for _, row in repr_days_local.iterrows():
            season_int = int(row['season_int'])
            day_val = int(row['day'])
            if 'rule' in repr_days_local.columns and pd.notna(row.get('rule', None)):
                category = row['rule']
            else:
                category = 'special' if (season_int, day_val) in special_set else 'representative'
            day_slice = df_energy_local[(df_energy_local['season'] == season_int) & (df_energy_local['day'] == day_val)]
            if day_slice.empty:
                continue
            avg_cf = day_slice[tech_cols].mean()
            weight_pct = (float(row['weight']) / total_weight) if total_weight else np.nan
            row_data = {
                'season': row['season'],
                'day': day_val,
                'category': category,
                'weight_pct': weight_pct,
            }
            row_data.update({f'{tech}_avg_cf': avg_cf[tech] for tech in tech_cols})
            rows.append(row_data)

        # Benchmark rows (min/avg/max across full dataset for each tech).
        benchmarks = []
        tech_stats = {
            'benchmark_min': df_energy_local[tech_cols].min(),
            'benchmark_avg': df_energy_local[tech_cols].mean(),
            'benchmark_max': df_energy_local[tech_cols].max(),
        }
        for label, series in tech_stats.items():
            bench_row = {'season': '-', 'day': '-', 'category': label, 'weight_pct': '-'}
            bench_row.update({f'{tech}_avg_cf': series[tech] for tech in tech_cols})
            benchmarks.append(bench_row)

        summary_df = pd.DataFrame(rows + benchmarks)
        summary_df.to_csv(output_path, index=False)
        return output_path
    except Exception as exc:  # noqa: BLE001 - intentional catch-all to keep pipeline running
        print(f'[repr-days] Warning: failed to export representative_days_summary.csv ({exc})')
        return None


def format_epm_phours(repr_days, folder):
    """Format pHours EPM like.

    Parameters
    ----------
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    folder: str
        Path to save the file.
    """
    repr_days_formatted_epm = repr_days.copy()
    repr_days_formatted_epm = repr_days_formatted_epm.set_index(['season', 'daytype'])['weight'].squeeze()
    repr_days_formatted_epm = pd.concat([repr_days_formatted_epm] * 24,
                                        keys=['t{}'.format(i) for i in range(1, 25)], names=['hour'], axis=1)

    path_file = os.path.join(folder, 'pHours.csv')
    repr_days_formatted_epm.to_csv(path_file)
    print('pHours file saved at:', path_file)
    print('Number of hours: {:.0f}'.format(repr_days_formatted_epm.sum().sum() / len(repr_days_formatted_epm.columns)))


def format_epm_pvreprofile(df_energy, repr_days, folder, name_data=''):
    """Format pVREProfile EPM like

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    name_data: str
        Name of the zone.
    """
    pVREProfile = df_energy.copy()
    pVREProfile['season'] = pVREProfile['season'].apply(lambda x: f'Q{x}')

    pVREProfile = pVREProfile.set_index(['season', 'day', 'hour'])
    pVREProfile = pVREProfile[[col for col in df_energy.columns if (('PV' in col) or ('Wind' in col))]]
    pVREProfile.columns = pd.MultiIndex.from_tuples([tuple([col.split('_')[0], '_'.join(col.split('_')[1:])]) for col in pVREProfile.columns])
    # TODO: check why name_data is used there?
    if pVREProfile.columns.nlevels == 1:
        pVREProfile.columns = pd.MultiIndex.from_tuples([(col[0], name_data) for col in pVREProfile.columns])
    pVREProfile.columns.names = ['fuel', 'zone']

    t = repr_days.copy()
    t = t.drop(columns=[c for c in ['rule'] if c in t.columns])
    t = t.set_index(['season', 'day'])

    pVREProfile = pVREProfile.unstack('hour')
    # select only the representative days
    pVREProfile = pVREProfile.loc[t.index, :]
    pVREProfile = pVREProfile.stack(level=['fuel', 'zone'], future_stack=True)
    pVREProfile = pd.merge(pVREProfile.reset_index(), t.reset_index(), on=['season', 'day']).set_index(
        ['zone', 'season', 'daytype', 'fuel'])
    pVREProfile.drop(['day', 'weight'], axis=1, inplace=True)

    pVREProfile.columns = ['t{}'.format(i + 1) for i in pVREProfile.columns]

    # Reorder index names
    pVREProfile = pVREProfile.reorder_levels(['zone', 'fuel', 'season', 'daytype'], axis=0)

    pVREProfile.to_csv(os.path.join(folder, 'pVREProfile.csv'), float_format='%.5f')
    print('VRE Profile file saved at:', os.path.join(folder, 'pVREProfile.csv'))


def format_epm_demandprofile(df_energy, repr_days, folder, name_data=''):
    """Format pDemandProfile EPM like

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    name_data: str
        Name of the zone.
    """
    pDemandProfile = df_energy.copy()
    pDemandProfile['season'] = pDemandProfile['season'].apply(lambda x: f'Q{x}')

    pDemandProfile = pDemandProfile.set_index(['season', 'day', 'hour'])
    pDemandProfile = pDemandProfile[[col for col in pDemandProfile.columns if 'Load' in col]]

    pDemandProfile.columns = pd.MultiIndex.from_tuples(
        [tuple([col.split('_')[0], '_'.join(col.split('_')[1:])]) for col in pDemandProfile.columns])
    pDemandProfile.columns.names = ['load', 'zone']

    # Drop the 'Load' level if it exists
    pDemandProfile = pDemandProfile.droplevel('load', axis=1)

    # pVREProfile.index.names = ['season', 'day', 'hour']
    t = repr_days.copy()
    t = t.drop(columns=[c for c in ['rule'] if c in t.columns])
    t = t.set_index(['season', 'day'])
    pDemandProfile = pDemandProfile.unstack('hour')
    # select only the representative days
    pDemandProfile = pDemandProfile.loc[t.index, :]
    pDemandProfile = pDemandProfile.stack('zone', future_stack=True)

    pDemandProfile = pd.merge(pDemandProfile.reset_index(), t.reset_index(), on=['season', 'day']).set_index(
        ['zone', 'season', 'daytype'])
    pDemandProfile.drop(['day', 'weight'], axis=1, inplace=True)

    pDemandProfile.columns = ['t{}'.format(i + 1) for i in pDemandProfile.columns]

    pDemandProfile = pDemandProfile.reorder_levels(['zone', 'season', 'daytype'], axis=0)

    pDemandProfile.to_csv(os.path.join(folder, 'pDemandProfile.csv'))
    print('pDemandProfile file saved at:', os.path.join(folder, 'pDemandProfile.csv'))


def run_representative_days_pipeline(
    seasons_map: dict,
    input_files: dict,
    output_dir: Union[str, Path],
    data_dir: Union[str, Path] = None,
    epm_output_dir: Union[str, Path] = None,
    summary_dir: Union[str, Path] = None,
    zones_to_exclude=None,
    value_column: str = 'value',
    year_label: Union[int, str] = 2018,
    n_representative_days: int = 2,
    n_clusters: int = 20,
    n_bins: int = 10,
    feature_selection_count: int = None,
    feature_selection_method: str = 'ward',
    feature_selection_metric: str = 'euclidean',
    feature_selection_scale: bool = True,
    special_day_threshold: float = 0.1,
    gams_main_file: str = 'OptimizationModelZone.gms',
    verbose: bool = True,
) -> dict:
    """Run the full representative day pipeline (matches the notebook flow) end-to-end.

    Parameters
    ----------
    seasons_map : dict
        Mapping month -> season (e.g., {1: 2, 2: 2, 5: 1, ...}).
    input_files : dict
        Keys are technology names (e.g., {'PV': 'input/data_capp_solar.csv', 'Wind': ...}).
        Values are paths to raw CSV inputs with columns ``zone``, ``month`` (or ``season``), ``day``, ``hour``
        and one value column (named by ``value_column`` or the only non-index column).
    output_dir : str or Path
        Directory to store all outputs (seasonal CSVs, optimization inputs, and EPM-formatted files).
    data_dir : str or Path, optional
        Directory to store intermediate optimization inputs (e.g., data_formatted_optim.csv). Defaults to output_dir.
    epm_output_dir : str or Path, optional
        Directory to store EPM-formatted outputs (pHours, pVREProfile, pDemandProfile). Defaults to output_dir.
    summary_dir : str or Path, optional
        Directory to store representative_days_summary.csv. Defaults to output_dir.
    zones_to_exclude : list, optional
        Zones to drop before processing.
    value_column : str, default 'value'
        Name of the value column in raw inputs. If absent and exactly one non-index column exists, it is used.
    year_label : int or str, default 2018
        Column name to assign to the value series (used to pick the representative year).
    n_representative_days : int, default 2
        Number of representative days to compute via the Poncelet optimization.
    n_clusters : int, default 20
        Number of clusters used to identify extreme/special days before optimization.
    n_bins : int, default 10
        Number of bins to pass to the GAMS model.
    feature_selection_count : int, optional
        If provided, runs hierarchical feature selection to keep only ``n`` series before optimization.
    feature_selection_method : str, default 'ward'
        Linkage method for feature selection clustering.
    feature_selection_metric : str, default 'euclidean'
        Distance metric for feature selection clustering.
    feature_selection_scale : bool, default True
        Whether to scale features before feature selection.
    special_day_threshold : float, default 0.1
        Probability cutoff when extracting special days from clusters.
    gams_main_file : str, default 'OptimizationModelZone.gms'
        Name (or absolute path) of the GAMS model to execute.

    Returns
    -------
    dict
        Dictionary with DataFrames and key output paths:
        {
            'df_energy': formatted full-year data,
            'df_energy_no_special': data after removing special days,
            'special_days': special days DataFrame,
            'repr_days': representative days with weights,
            'paths': {
                'seasonal_inputs': {tech: <path>},
                'data_formatted': <path>,
                'data_feature_selection': <path or None>,
                'pHours': <path>,
                'pVREProfile': <path>,
                'pDemandProfile': <path or None>,
                'repr_days': <path>,
                'df_energy': <path>,
            }
        }
    """
    zones_to_exclude = zones_to_exclude or []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    epm_output_dir = Path(epm_output_dir) if epm_output_dir else output_dir
    epm_output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(summary_dir) if summary_dir else output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 - Prepare raw inputs (month -> season, drop zones, rename value column)
    if verbose:
        print('[repr-days] Step 1: preparing raw inputs')
    seasonal_inputs = {}
    for tech, path in input_files.items():
        seasonal_path, _ = prepare_input_timeseries(
            input_path=path,
            seasons_map=seasons_map,
            output_dir=output_dir,
            zones_to_exclude=zones_to_exclude,
            value_column=value_column,
            year_label=year_label,
            verbose=verbose,
        )
        seasonal_inputs[tech] = str(seasonal_path)
        if verbose:
            print(f'[repr-days]   prepared {tech} -> {seasonal_path}')

    # Step 2 - Combine tech profiles and pick representative year
    if verbose:
        print('[repr-days] Step 2: combine tech profiles and pick representative year')
    df_energy = prepare_energy_profiles(seasonal_inputs)
    df_energy = df_energy.dropna(axis=1, how='all')

    # Step 3 - Cluster and extract special days
    if verbose:
        print(f'[repr-days] Step 3: clustering with n_clusters={n_clusters}')
    df_energy_cluster, df_closest_days, centroids_df = cluster_data_new(df_energy, n_clusters=n_clusters)
    special_days, df_energy_no_special = get_special_days_clustering(
        df_closest_days, df_energy_cluster, threshold=special_day_threshold
    )
    if verbose:
        print(f'[repr-days]   special days extracted: {len(special_days)}')

    # Step 4 - Build optimization input (compute correlations)
    if verbose:
        print('[repr-days] Step 4: building optimization input')
    _, path_data_file = format_optim_repr_days(
        df_energy_no_special,
        output_dir,
        data_dir=str(data_dir),
    )

    # Optional feature selection to shrink the feature set
    path_data_file_for_optim = path_data_file
    selection_path = None
    if feature_selection_count:
        if verbose:
            print(f'[repr-days]   feature selection to top {feature_selection_count} series')
        _, _, selection_path = select_representative_series_hierarchical(
            path_data_file,
            n=feature_selection_count,
            method=feature_selection_method,
            metric=feature_selection_metric,
            scale=feature_selection_scale,
        )
        path_data_file_for_optim = selection_path

    # Step 5 - Run optimization and parse representative days
    if verbose:
        print(f'[repr-days] Step 5: running optimization for {n_representative_days} representative days')
    launch_optim_repr_days(
        path_data_file_for_optim,
        folder_process_data=str(output_dir),
        nbr_days=n_representative_days,
        main_file=gams_main_file,
        nbr_bins=n_bins,
    )
    repr_days = parse_repr_days(str(output_dir), special_days)
    if verbose:
        print(f'[repr-days]   parsed {len(repr_days)} representative days')

    # Step 6 - Format outputs for EPM
    if verbose:
        print('[repr-days] Step 6: formatting outputs for EPM')
    format_epm_phours(repr_days, str(epm_output_dir))
    format_epm_pvreprofile(df_energy, repr_days, str(epm_output_dir))
    demand_profile_path = None
    if any('Load' in c for c in df_energy.columns):
        format_epm_demandprofile(df_energy, repr_days, str(epm_output_dir))
        demand_profile_path = epm_output_dir / 'pDemandProfile.csv'

    repr_days_path = output_dir / 'repr_days.csv'
    df_energy_path = output_dir / 'df_energy.csv'
    summary_path = export_repr_days_summary(df_energy, repr_days, special_days, summary_dir)
    repr_days.to_csv(repr_days_path, index=False)
    df_energy.to_csv(df_energy_path, index=False)

    return {
        'df_energy': df_energy,
        'df_energy_no_special': df_energy_no_special,
        'special_days': special_days,
        'repr_days': repr_days,
        'paths': {
            'seasonal_inputs': seasonal_inputs,
            'data_formatted': path_data_file,
            'data_feature_selection': selection_path,
            'pHours': str(epm_output_dir / 'pHours.csv'),
            'pVREProfile': str(epm_output_dir / 'pVREProfile.csv'),
            'pDemandProfile': str(demand_profile_path) if demand_profile_path else None,
            'repr_days': str(repr_days_path),
            'df_energy': str(df_energy_path),
            'repr_days_summary': str(summary_path) if summary_path else None,
        },
    }


# --------------------------------------------------------------------------- #
# Reduced scenarios helpers (alternative clustering/selection)
# --------------------------------------------------------------------------- #

def cluster_data(data: pd.DataFrame, n_clusters: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Cluster the historical data into specified number of clusters.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        n_clusters (int): Number of clusters to divide the data into.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Cluster labels and cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data.T)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.index, index=range(n_clusters))
    return labels, cluster_centers


# TODO: consolidate representative-year selection helpers or expose stable API
def _select_representative_year_real(
        data: pd.DataFrame, labels: np.ndarray, cluster_centers: pd.DataFrame
) -> pd.Series:
    """
    Select a real historical year that best represents each cluster based on the minimum distance to the cluster center.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        labels (np.ndarray): Cluster labels for each year.
        cluster_centers (pd.DataFrame): Cluster centers.

    Returns:
        pd.Series: Representative year for each cluster.
    """
    representative_years = {}
    for cluster in range(len(cluster_centers)):
        cluster_members = data.columns[labels == cluster]
        distances = [
            np.linalg.norm(data[year].values - cluster_centers.loc[cluster].values)
            for year in cluster_members
        ]
        best_year = cluster_members[np.argmin(distances)]
        representative_years[cluster] = best_year
    return pd.Series(representative_years)


def _select_representative_year_synthetic(cluster_centers: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic years as the cluster centers.

    Parameters:
        cluster_centers (pd.DataFrame): Cluster centers.

    Returns:
        pd.DataFrame: Synthetic years for each cluster.
    """
    return cluster_centers.T


# TODO: scope this helper to the reduced-scenario workflow only
def _assign_probabilities(labels: np.ndarray, n_clusters: int) -> pd.Series:
    """
    Calculate probabilities for each cluster based on the frequency of occurrence.

    Parameters:
        labels (np.ndarray): Cluster labels for each year.
        n_clusters (int): Number of clusters.

    Returns:
        pd.Series: Probabilities for each cluster.
    """
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return pd.Series(probabilities, index=range(n_clusters))


def run_reduced_scenarios(data: pd.DataFrame, n_clusters: int, method: str) -> pd.DataFrame:
    """
    Run the reduced scenarios algorithm to select representative years.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        n_clusters (int): Number of clusters to divide the data into.
        method (str): Method to select representative years.

    Returns:
        pd.DataFrame: Representative years for each cluster.
    """

    labels, cluster_centers = cluster_data(data, n_clusters)
    probabilities = _assign_probabilities(labels, n_clusters)

    if method == "real":
        representatives_years = _select_representative_year_real(data, labels, cluster_centers)
        representatives = data.loc[:, representatives_years]
        representatives.columns = ['{} - {:.2f}'.format(i, probabilities[k]) for k, i in representatives_years.items()]
    elif method == "synthetic":
        representatives = _select_representative_year_synthetic(cluster_centers)
        representatives.columns = ['{} - {:.2f}'.format(i, probabilities[i]) for i in representatives.columns]
    else:
        raise ValueError("Invalid method")

    return representatives


def plot_uncertainty(df, df2=None, title="Uncertainty Range Plot", ylabel="Values", xlabel="Month", ymin=0, ymax=None,
                     filename=None,
                     convert_months=True):
    """
    Plots the range of values (min to max) as a transparent grey area and highlights
    the interquartile range (25th to 75th percentile) in darker grey.

    Parameters:
    - df: DataFrame with months as the index and multiple years as columns.
    - title: Title of the plot.
    - ylabel: Label for the y-axis.
    - xlabel: Label for the x-axis.
    """
    # Calculate min, max, and interquartile range
    min_vals = df.min(axis=1)
    max_vals = df.max(axis=1)
    q10_vals = df.quantile(0.10, axis=1)
    q90_vals = df.quantile(0.90, axis=1)
    q25_vals = df.quantile(0.25, axis=1)
    q75_vals = df.quantile(0.75, axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the full uncertainty range (min to max) as light grey
    ax.fill_between(df.index, min_vals, max_vals, color='grey', alpha=0.3, label='Min-Max Range')

    # Plot 10th-90th percentile in blue
    ax.fill_between(df.index, q10_vals, q90_vals, color='grey', alpha=0.5, label='10th-90th Percentile')

    # Plot the interquartile range (25th to 75th percentile) as darker grey
    ax.fill_between(df.index, q25_vals, q75_vals, color='grey', alpha=0.9, label='25th-75th Percentile')

    if df2 is not None:
        df2.plot(ax=ax)

    # Convert x-axis labels to month names if requested
    if convert_months:
        ax.set_xticks(df.index)
        ax.set_xticklabels([calendar.month_abbr[m] for m in df.index], rotation=0)

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper right')

    ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    # Show the plot
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# TODO: keep private; dispatch plotting uses only within this module
def _format_dispatch_ax(ax, pd_index, day='day', season='season', display_day=True):
    # Adding the representative days and seasons
    n_rep_days = len(pd_index.get_level_values(day).unique())
    dispatch_seasons = pd_index.get_level_values(season).unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

    if display_day:
        for d in range(total_days):
            x_d = 24 * d

            # Add vertical lines to separate days
            is_end_of_season = d % n_rep_days == 0
            linestyle = '-' if is_end_of_season else '--'
            ax.axvline(x=x_d, color='slategrey', linestyle=linestyle, linewidth=0.8)

            # Add day labels (d1, d2, ...)
            ax.text(
                x=x_d + 12,  # Center of the day (24 hours per day)
                y=y_max * 0.99,
                s=f'd{(d % n_rep_days) + 1}',
                ha='center',
                fontsize=7
            )

    # Add season labels
    season_x_positions = [24 * n_rep_days * s + 12 * n_rep_days for s in range(len(dispatch_seasons))]
    ax.set_xticks(season_x_positions)
    ax.set_xticklabels(dispatch_seasons, fontsize=8)
    ax.set_xlim(left=0, right=24 * total_days)
    ax.set_xlabel('Time')
    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)


def select_time_period(df, select_time):
    """Select a specific time period in a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Columns contain season and day
    select_time: dict
        For each key, specifies a subset of the dataframe

    Returns
    -------
    pd.DataFrame: Dataframe with the selected time period
    str: String with the selected time period
    """
    temp = df.copy()
    for k, i in select_time.items():
        temp = temp.loc[temp.index.get_level_values(k).isin(i), :]
    return temp


# TODO: evaluate if this helper is still needed outside notebooks
def create_season_day_index() -> pd.Series:
    """Create a MultiIndex with all combinations of seasons and days.

    Returns:
    -------
    pd.Series: Series with a MultiIndex of seasons and days.
    """

    # Create the levels for the MultiIndex
    months = np.arange(1, 13)  # Months 1-12
    days = np.arange(1, 32)  # Days 1-31

    # Generate all combinations of months and days
    month_day_combinations = [(month, day) for month in months for day in days]

    # Filter invalid days (e.g., February 30, April 31)
    valid_combinations = [
        (month, day)
        for month, day in month_day_combinations
        if day <= pd.Timestamp(f"2025-{month:02d}-01").days_in_month
    ]

    # Create the MultiIndex
    multi_index = pd.MultiIndex.from_tuples(valid_combinations, names=["season", "day"])

    # Create the Series with 1 as the value everywhere
    series = pd.Series(1, index=multi_index)

    return series


def plot_dispatch(df, day_level='daytype', season_level='season', title=None):
    fig, ax = plt.subplots()

    df.plot(ax=ax)

    _format_dispatch_ax(ax, df.index)

    ax.set_xlabel('Hours')
    ax.set_ylim(bottom=0)
    if title is not None:
        ax.set_title(title)
    plt.show()


def plot_vre_repdays(input_file, vre_profile, pHours, season_colors, countries=None,
                     fontsize_legend=8, alpha_background=0.1, min_alpha=0.3, max_alpha=1,
                     path=None):
    """
    Plot time series of renewable generation (e.g., PV, Wind) for all days and highlight representative days.

    This function displays hourly renewable generation profiles for all days (as background lines)
    and overlays the representative days used in the Poncelet algorithm, with line opacity
    proportional to their weight in the optimization.

    Parameters
    ----------
    input_file : pd.DataFrame
        Hourly renewable generation data with a multi-index (season, day, hour) and multi-index columns (fuel, zone).

    vre_profile : pd.DataFrame
        Output from the representative day selection, formatted with representative generation profiles.

    pHours : pd.DataFrame
        Hourly weights associated with each representative day, used to scale the line opacity.

    season_colors : dict
        Dictionary mapping each season to a color.

    countries : list, optional
        List of zones to include. If None, all available zones are used.

    fontsize_legend : int, default 8
        Font size for the legend.

    alpha_background : float, default 0.1
        Transparency for the background lines representing all days.

    min_alpha : float, default 0.3
        Minimum transparency for the representative day lines (lowest weight).

    max_alpha : float, default 1
        Maximum opacity for the representative day lines (highest weight).

    Notes
    -----
    - One subplot is generated per fuel and season.
    - Representative days are plotted with darker lines when their weight in the optimization is higher.
    - All background days are plotted in a lighter color for context.
    """

    input_file = input_file.copy()
    input_file.index = input_file.index.set_levels(
        ['Q' + str(s) for s in input_file.index.levels[0]],
        level='season'
    )
    input_file.index = input_file.index.set_levels(
        ['d' + str(s) for s in input_file.index.levels[1]],
        level='day'
    )

    # Remove correlation columns
    columns = [c for c in input_file.columns if 'corr' not in c]
    input_file = input_file[columns]

    # Reconstruct MultiIndex if needed
    if input_file.columns.nlevels == 1:
        input_file.columns = pd.MultiIndex.from_tuples(
            [tuple([col.split('_')[0], '_'.join(col.split('_')[1:])]) for col in input_file.columns])
        input_file.columns.names = ['fuel', 'zone']

    if countries is None:
        countries = input_file.columns.get_level_values('zone').unique()

    input_file = input_file.loc[:, input_file.columns.get_level_values('zone').isin(countries)]

    fuels = input_file.columns.get_level_values('fuel').unique()
    seasons = input_file.index.get_level_values('season').unique()

    df_rep = vre_profile.loc[vre_profile.index.get_level_values('zone').isin(countries), :]
    df_rep.columns = df_rep.columns.astype(str).str.extract(r't(\d+)')[0].astype(int) - 1
    df_rep = df_rep.stack(future_stack=True)
    df_rep.index.names = ['zone', 'fuel', 'season', 'day', 'hour']
    df_rep = df_rep.unstack(['zone', 'fuel'])

    df_hours = pHours.copy()
    df_hours.columns = df_hours.columns.astype(str).str.extract(r't(\d+)')[0].astype(int) - 1
    df_hours = df_hours.stack(future_stack=True)
    df_hours.index.names = ['season', 'day', 'hour']

    # Get min/max weights for scaling alpha
    wmin, wmax = df_hours.min(), df_hours.max()

    def scale_alpha(w):
        if wmax == wmin:
            return (min_alpha + max_alpha) / 2
        return min_alpha + (w - wmin) / (wmax - wmin) * (max_alpha - min_alpha)

    fig, axs = plt.subplots(len(fuels), len(seasons), figsize=(5 * len(seasons), 3.5 * len(fuels)), sharey=True, sharex=True)

    # Always use 2D indexing for axs
    if len(fuels) == 1:
        axs = axs[np.newaxis, :]
    if len(seasons) == 1:
        axs = axs[:, np.newaxis]

    for i, fuel in enumerate(fuels):
        subset_days = input_file.loc[:, input_file.columns.get_level_values('fuel') == fuel]
        subset_repdays = df_rep.loc[:, df_rep.columns.get_level_values('fuel') == fuel]

        for j, season in enumerate(seasons):
            ax = axs[i][j]
            season_data = subset_days.loc[season]
            season_repdays = subset_repdays.loc[season]

            for day in season_data.index.get_level_values('day').unique():
                day_data = season_data.loc[day]
                ax.plot(
                    day_data.index.get_level_values('hour'),
                    day_data.mean(axis=1),
                    color=season_colors.get(season, 'orange'),
                    alpha=alpha_background
                )

            # TODO: Correct to add pDemandProfile
            if not season_repdays.empty:
                for repday in season_repdays.index.get_level_values('day').unique():
                    day_data = season_repdays.loc[repday]
                    weight = df_hours.xs(season, level='season').xs(repday, level='day').mean()
                    label = f'{season}-{repday} ({int(weight)})'

                    alpha = scale_alpha(weight)
                    ax.plot(
                        day_data.index.get_level_values('hour'),
                        day_data.mean(axis=1),
                        color=season_colors.get(season, 'grey'),
                        alpha=alpha,
                        linewidth = 2,
                        label=label,
                        zorder=2
                    )

            ax.set_title(f"{fuel} - {season}")
            ax.legend(loc='upper right', fontsize=fontsize_legend, frameon=False)
            if j == 0:
                ax.set_ylabel("Capacity factor")
            ax.grid(True)

    axs[-1][-1].set_xlabel("Hour of day")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# TODO: confirm usage and move to a dedicated module if kept
def run_smoothing_reservoir(config):
    """
    Smooths the energy dispatch of a hydro reservoir by solving a linear optimization problem.

    The function models a hydro power plant with a reservoir and optimizes its hourly dispatch
    to meet inflow and storage constraints while minimizing variability in dispatch across time
    and across seasons. It ensures the system operates within physical limits such as reservoir capacity,
    minimum seasonal storage, and seasonal balance in dispatch.

    Args:
        config (dict): Dictionary containing model parameters. Required keys are:
            - 'hours' (int): Total number of time steps (typically hours).
            - 'inflow' (np.ndarray): Array of inflows into the reservoir [GWh] of shape (hours,).
            - 'reservoir_size' (float): Maximum storage capacity of the reservoir [GWh].
            - 'reservoir_min' (float): Minimum storage level as a fraction of reservoir_size
              (e.g., 0.2 for 20%). Enforced only during specified hours.
            - 'hours_reservoir_min_constraints' (np.ndarray): Array of time indices (ints)
              where the minimum storage level must be enforced.

    Returns:
        res (scipy.optimize.OptimizeResult): Result object from `scipy.optimize.linprog` containing:
            - res.x: Optimal values of the decision variables (inflow, outflow, storage, dispatch, abs_diff, D_j, U_j).
            - res.fun: Value of the objective function at the optimum (total smoothing cost).
            - res.success: Boolean indicating whether optimization succeeded.
            - res.message: modeltype status message.
    """

    # Parameters
    hours = config['hours']
    reservoir_max = config['reservoir_size']  # in GWh
    min_storage = config['reservoir_min'] * reservoir_max
    seasonal_hours = config['hours_reservoir_min_constraints']  # must be a numpy array containing the t for which the reservoir constraint should be enforced
    inflow = config['inflow']  # should be of shape (hours,)

    # Variables: [i_t, o_t, s_t, d_t, u_t, D_j, U_j] for each hour, plus D1-D4, plus U1-U3
    n = hours
    n_var = 5 * n + 4 + 3  # hourly vars + seasonal dispatch sums + abs diffs

    # Objective: minimize sum of U_j
    c = np.zeros(n_var)
    c[5 * n + 4:] = 1  # U1, U2, U3
    c[4: 5 * n: 5] = 1  # u_t terms

    # Bounds
    bounds = [(0, None), (0, None), (0, reservoir_max), (0, None), (0, None)] * n
    bounds += [(0, None)] * 4  # D1-D4
    bounds += [(0, None)] * 3  # U1-U3

    # Equality constraints: storage + dispatch + seasonal sums
    A_eq = lil_matrix((2 * n + 4, n_var))
    b_eq = np.zeros(2 * n + 4)

    for t in range(n):
        # Storage equation: s_t - s_{t-1} - i_t + o_t = 0
        if t > 0:
            A_eq[t - 1, 5 * t + 2] = 1  # s_t
            A_eq[t - 1, 5 * t + 0] = -1  # i_t
            A_eq[t - 1, 5 * t + 1] = 1  # o_t
            A_eq[t - 1, 5 * (t - 1) + 2] = -1  # s_{t-1}

        # Dispatch equation: d_t + i_t - o_t = inflow_t
        A_eq[n - 1 + t, 5 * t + 3] = 1  # d_t
        A_eq[n - 1 + t, 5 * t + 0] = 1  # i_t
        A_eq[n - 1 + t, 5 * t + 1] = -1  # o_t
        b_eq[n - 1 + t] = inflow[t]

    # Equation to ensure storage level is the same at beginning and end
    A_eq[2 * n - 1, 2] = 1
    A_eq[2 * n - 1, 5 * (n - 1) + 2] = -1

    # Seasonal sum constraints: D_j = sum(d_t in season j)
    season_length = n // 4
    for j in range(4):
        for t in range(j * season_length, (j + 1) * season_length):
            A_eq[2 * n + j, 5 * t + 3] = 1  # coefficients for each d_t in the season
        A_eq[2 * n + j, 5 * n + j] = -1  # subtract D_j

    # Inequality constraints
    row_max = 2 * (n - 1) + len(seasonal_hours) + 6
    A_ub = lil_matrix((row_max, n_var))
    b_ub = np.zeros(row_max)

    row = 0

    # Absolute value for difference in outcomes
    for t in range(1, n):
        # u_t >= |d_t - d_{t-1}|
        A_ub[row, 5 * t + 4] = -1
        A_ub[row, 5 * t + 3] = 1
        A_ub[row, 5 * (t - 1) + 3] = -1
        row += 1

        A_ub[row, 5 * t + 4] = -1
        A_ub[row, 5 * t + 3] = -1
        A_ub[row, 5 * (t - 1) + 3] = 1
        row += 1

    # Seasonal storage constraint: s_t >= min → -s_t <= -min
    for t in seasonal_hours:
        A_ub[row, 5 * t + 2] = -1
        b_ub[row] = -min_storage
        row += 1

    # U_j >= |D_{j+1} - D_j|
    for j in range(3):
        # D_{j+1} - D_j - U_j <= 0
        A_ub[row, 5 * n + j + 1] = 1
        A_ub[row, 5 * n + j] = -1
        A_ub[row, 5 * n + 4 + j] = -1
        row += 1

        # D_{j} - D_{j+1} - U_j <= 0
        A_ub[row, 5 * n + j + 1] = -1
        A_ub[row, 5 * n + j] = 1
        A_ub[row, 5 * n + 4 + j] = -1
        row += 1

    # Solve
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub[:row],
                   A_eq=A_eq.tocsr(), b_eq=b_eq,
                   bounds=bounds, method='highs')

    # Output result status
    print(res.success, res.fun)
    return res, A_eq, b_eq


if __name__ == "__main__":
    sample_base = Path(__file__).resolve().parent
    sample_input_dir = sample_base / "input"
    sample_output_dir = sample_base / "output"
    sample_input_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_files = {
        "PV": os.path.join(sample_input_dir, "vre_rninja_solar.csv"),
        "Wind": os.path.join(sample_input_dir, "vre_rninja_wind.csv"),
        "Load": os.path.join(sample_input_dir, "load_profiles.csv"),
    }

    seasons_map = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 1,
        10: 1,
        11: 1,
        12: 2,
    }
    gams_model = sample_base / "gams" / "OptimizationModelZone.gms"
    print(f"[repr-days example] Using sample inputs in {sample_input_dir} and output {sample_output_dir}")
    if not gams_model.exists():
        print(f"[repr-days example] GAMS model not found at {gams_model}; update the script with your local GAMS path.")
    else:
        result = run_representative_days_pipeline(
            seasons_map=seasons_map,
            input_files=sample_files,
            output_dir=sample_output_dir,
            gams_main_file=str(gams_model),
            n_representative_days=1,
            n_clusters=4,
            n_bins=4,
            special_day_threshold=0.2,
        )
        print("[repr-days example] Pipeline completed; outputs:")
        for key, value in result["paths"].items():
            print(f"  {key}: {value}")
