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

# General packages to process and plot EPM results
# Docstring formatting should follow the NumPy/SciPy format: https://numpydoc.readthedocs.io/en/latest/format.html
# The first line should be a short and concise summary of the function's purpose.
# The second line should be blank, and any further lines should be wrapped at 72 characters.
# The docstring should describe the function's purpose, arguments, and return values, as well as any exceptions that the function may raise.
# The docstring should also provide examples of how to use the function.

# Importing necessary libraries
import argparse
import base64
import io
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional
import colorsys

import gams.transfer as gt
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import FixedLocator, MaxNLocator
from PIL import Image
from shapely.geometry import LineString, Point, Polygon


_DEFAULT_LOGGER = None


def set_default_logger(logger):
    """Set shared post-processing logger for all helper modules."""
    global _DEFAULT_LOGGER
    _DEFAULT_LOGGER = logger


def get_default_logger():
    """Return the current shared logger instance."""
    return _DEFAULT_LOGGER


def set_utils_logger(logger):
    """Backwards compatible logger setter used by legacy code."""
    set_default_logger(logger)


def _get_logger(logger=None):
    return logger or get_default_logger()


def _log(level, message, logger=None):
    log = _get_logger(logger)
    if log:
        getattr(log, level)(message)
    else:
        print(message)


def log_info(message, logger=None):
    _log('info', message, logger=logger)


def log_warning(message, logger=None):
    _log('warning', message, logger=logger)


def log_error(message, logger=None):
    _log('error', message, logger=logger)


# Backwards compatibility for older imports relying on underscored helpers
_log_info = log_info
_log_warning = log_warning
_log_error = log_error


FUELS = os.path.join('static', 'fuels.csv')
TECHS = os.path.join('static', 'technologies.csv')
COLORS = os.path.join('static', 'colors.csv')
GEOJSON = os.path.join('static', 'zones.geojson')
GEOJSON_TO_EPM = os.path.join('static', 'geojson_to_epm.csv')

TOLERANCE = 1e-2

# TODO: fix that because only used in dispatch
NAME_COLUMNS = {
    'pDispatchTechFuel': 'fuel',
    'pDispatchPlant': 'fuel',
    'pDispatch': 'attribute',
    'pYearlyCostsZone': 'attribute',
    'pCapacityTechFuel': 'fuel',
    'pEnergyTechFuel': 'fuel',
    'pDispatchReserve':'attribute',
    'pNetExchange':'attribute'
}

UNIT = {
    'Capex: $m': 'M$/year',
    'Unmet demand costs: : $m': 'M$'
}

RENAME_COLUMNS = {'c': 'country', 'c_0': 'country', 'y': 'year', 'v': 'value', 's': 'scenario', 'uni': 'attribute',
                  'z': 'zone', 'g': 'generator', 'gen': 'generator',
                  'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't', 'sumhdr': 'attribute', 'genCostCmp': 'attribute'}
TYPE_COLUMNS  = {'year': int, 'season': str, 'day': str, 'tech': str, 'fuel': str}


_FLOAT_PATTERN = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_OBJECTIVE_REGEX = re.compile(rf"(?:Objective|Final\s+Solve):\s*({_FLOAT_PATTERN})")
_ELAPSED_REGEX = re.compile(r"elapsed\s+(\d+):(\d+):(\d+(?:\.\d+)?)", re.IGNORECASE)
_MEMORY_REGEX = re.compile(rf"({_FLOAT_PATTERN})\s*M[Bb]\b")


def parse_gams_solver_log(log_text: str) -> Dict[str, Optional[float]]:
    """Extract objective, elapsed time, and peak memory metrics from a GAMS log.

    Parameters
    ----------
    log_text : str
        Full text content of a GAMS solver log.

    Returns
    -------
    dict
        Dictionary with keys ``objective_billion_usd``, ``elapsed_time_seconds``
        and ``peak_memory_mb`` populated when the corresponding metric is
        present in the log; otherwise the values are ``None``.
    """

    objective_value = _extract_objective_value(log_text)
    elapsed_seconds = _extract_elapsed_time_seconds(log_text)
    peak_memory_mb = _extract_peak_memory_mb(log_text)

    objective_billion_usd = None
    if objective_value is not None:
        objective_billion_usd = objective_value / 1e9

    return {
        "Objective (Billion USD)": objective_billion_usd,
        "Time (s)": elapsed_seconds,
        "Peak Memory (Mb)": peak_memory_mb,
    }


def parse_gams_solver_log_file(log_path: Path) -> Dict[str, Optional[float]]:
    """Read a GAMS log file and parse key metrics.

    Parameters
    ----------
    log_path : pathlib.Path
        Path to the log file.

    Returns
    -------
    dict
        Parsed metrics as returned by :func:`parse_gams_solver_log`.
    """

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        content = log_file.read()
    return parse_gams_solver_log(content)


def _extract_objective_value(log_text: str) -> Optional[float]:
    matches = _OBJECTIVE_REGEX.findall(log_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _extract_elapsed_time_seconds(log_text: str) -> Optional[float]:
    matches = _ELAPSED_REGEX.findall(log_text)
    if not matches:
        return None
    hours, minutes, seconds = matches[-1]
    try:
        total_seconds = (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    except ValueError:
        return None
    return total_seconds


def _extract_peak_memory_mb(log_text: str) -> Optional[float]:
    peak_value: Optional[float] = None
    for match in _MEMORY_REGEX.findall(log_text):
        try:
            value = float(match)
        except ValueError:
            continue
        if peak_value is None or value > peak_value:
            peak_value = value
    return peak_value


def read_plot_specs(folder='postprocessing'):
    """
    Read the specifications for the plots from the static files.
    
    Returns:
    -------
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    colors = pd.read_csv(os.path.join(folder, COLORS), skiprows=1)
    colors = colors.dropna(subset=['Processing'])
    colors['Processing'] = colors['Processing'].astype(str).str.strip()
    colors = colors[~colors['Processing'].str.startswith('#')]
    colors = colors.dropna(subset=['Color'])
    colors['Color'] = colors['Color'].astype(str).str.strip()
    colors = colors[colors['Color'] != '']
    
    fuel_df = pd.read_csv(os.path.join(folder, FUELS), skiprows=1)
    tech_mapping = pd.read_csv(os.path.join(folder, TECHS))
    zones = gpd.read_file(os.path.join(folder, GEOJSON))
    geojson_to_epm = pd.read_csv(os.path.join(folder, GEOJSON_TO_EPM))

    fuel_df = fuel_df.dropna(subset=['Processing'])
    fuel_df['Processing'] = fuel_df['Processing'].astype(str).str.strip()
    fuel_df = fuel_df[fuel_df['Processing'] != '']
    fuel_df['EPM_Fuel'] = fuel_df['EPM_Fuel'].astype(str).str.strip()
    fuel_order = list(dict.fromkeys(fuel_df['Processing']))
    fuel_mapping = fuel_df.set_index('EPM_Fuel')['Processing'].to_dict()

    dict_specs = {
        'colors': colors.set_index('Processing')['Color'].dropna().to_dict(),
        'fuel_mapping': fuel_mapping,
        'fuel_order': fuel_order,
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict(),
        'map_zones': zones,
        'geojson_to_epm': geojson_to_epm
    }
    return dict_specs


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
    for param in container.getParameters():
        if container.data[param.name].records is not None:
            df[param.name] = container.data[param.name].records.copy()

    for s in container.getSets():
        if container.data[s.name].records is not None:
            df[s.name] = container.data[s.name].records.copy()

    return df


def extract_epm_folder(results_folder, file='epmresults.gdx'):
    """
    Extract information from a folder containing multiple scenarios.
    
    Parameters
    ----------
    results_folder: str
        Path to the folder containing the scenarios
    file: str, optional, default='epmresults.gdx'
        Name of the gdx file to extract
        
    Returns
    -------
    inverted_dict: dict
        Dictionary containing the extracted information for each scenario
    """
    # Dictionary to store the extracted information for each scenario
    dict_df = {}
    for scenario in [i for i in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, i))]:
        if file in os.listdir(os.path.join(results_folder, scenario)):
            dict_df.update({scenario: extract_gdx(os.path.join(results_folder, scenario, file))})

    inverted_dict = {
        k: {outer: inner[k] for outer, inner in dict_df.items() if k in inner}
        for k in {key for inner in dict_df.values() for key in inner}
    }

    inverted_dict = {k: pd.concat(v, names=['scenario']).reset_index('scenario') for k, v in inverted_dict.items()}
    return inverted_dict


def extract_epm_folder_by_scenario(FOLDER, file='epmresults.gdx', save_to_csv=False,
                                   folder_csv='csv'):
    """
    Extract information from a folder containing multiple scenarios,
    keeping the results separate for each scenario.

    Parameters
    ----------
    FOLDER: str
        Path to the folder containing the scenarios
    file: str, optional, default='epmresults.gdx'
        Name of the gdx file to extract

    Returns
    -------
    scenario_dict: dict
        Dictionary with scenario names as keys, each containing a dict of DataFrames
    """

    if 'postprocessing' in os.getcwd():  # code is launched from postprocessing folder
        assert 'output' not in FOLDER, 'FOLDER name is not specified correctly'
        # RESULTS_FOLDER = FOLDER
        RESULTS_FOLDER = os.path.join('..', 'output', FOLDER)
    else:  # code is launched from main root
        if 'output' not in FOLDER:
            RESULTS_FOLDER = os.path.join('output', FOLDER)
        else:
            RESULTS_FOLDER = FOLDER

    scenario_dict = {}
    for scenario in [i for i in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, i))]:
        gdx_path = os.path.join(RESULTS_FOLDER, scenario, file)
        if os.path.exists(gdx_path):
            scenario_dict[scenario] = extract_gdx(gdx_path)

    if save_to_csv:
        save_csv = Path(RESULTS_FOLDER) / Path(folder_csv)
        if not os.path.exists(save_csv):
            os.mkdir(save_csv)
        for key in scenario_dict.keys():
            path_scenario = Path(save_csv) / Path(f'{key}')
            if not os.path.exists(path_scenario):
                os.mkdir(path_scenario)
            for name, dataframe in scenario_dict[key].items():
                dataframe.to_csv(path_scenario / f'{name}.csv', index=False)

    return scenario_dict


def gdx_to_csv(gdx_file, output_csv_folder):
    try:
        container = gt.Container(gdx_file)
    except Exception as e:
        log_error(f"Error while loading gdx file : {e}")
        exit()

    parameters = [p.name for p in container.getParameters()]

    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)

    for param in parameters:
        try:
            df = container.data[param].records  # Récupérer les données
            if df is not None:
                output_csv = os.path.join(output_csv_folder, f"{param}.csv")
                df.to_csv(output_csv, index=False)  # Sauvegarder en CSV
                log_info(f"Folder downloaded : {output_csv}")
            else:
                log_warning(f"No data for param : {param}")
        except Exception as e:
            log_error(f"Error with param {param}: {e}")

    log_info("Conversion over! All files saved to folder 'data/'.")


def standardize_names(dict_df, key, mapping, column='fuel'):
    """
    Standardize the names of fuels in the dataframes.

    Only works when dataframes have fuel and value (with numerical values) columns.
    
    Parameters
    ----------
    dict_df: dict
        Dictionary containing the dataframes
    key: str
        Key of the dictionary to modify
    mapping: dict
        Dictionary mapping the original fuel names to the standardized names
    column: str, optional, default='fuel'
        Name of the column containing the fuels
    """

    if key in dict_df.keys():
        temp = dict_df[key].copy()
        temp[column] = temp[column].replace(mapping)
        temp = temp.groupby([i for i in temp.columns if i != 'value'], observed=False).sum().reset_index()

        new_fuels = [f for f in temp[column].unique() if f not in mapping.values()]
        if new_fuels:
            raise ValueError(f'New fuels found in {key}: {new_fuels}. '
                             f'Add fuels to the mapping in the /static folder and add in the colors.csv file.')

        dict_df[key] = temp.copy()
    else:
        log_warning(f'{key} not found in epm_dict')


def process_epm_inputs(epm_input, dict_specs, scenarios_rename=None):
    """
    Processing EPM inputs to use in plots.  
    
    Parameters
    ----------
    epm_input: dict
        Dictionary containing the input data
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    keys = ['pGenDataInput', 'ftfindex', 'pTechData', 'pZoneIndex', 'pDemandProfile', 'pDemandForecast', 'pSettings',
            'zcmap']
    rename_keys = {}

    dimension_aliases = {'pDemandForecast': 'year', 'pDemandProfile': 't'}
    for key, new_label in dimension_aliases.items():
        if key in epm_input and 'uni' in epm_input[key].columns:
            epm_input[key] = epm_input[key].rename(columns={'uni': new_label})

    epm_dict = {k: i.rename(columns=RENAME_COLUMNS) for k, i in epm_input.items() if k in keys and k in epm_input.keys()}

    if rename_keys is not None:
        epm_dict.update({rename_keys[k]: i for k, i in epm_dict.items() if k in rename_keys.keys()})

    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    for column, type in TYPE_COLUMNS.items():
        for key in epm_dict.keys():
            if column in epm_dict[key].columns:
                epm_dict[key][column] = epm_dict[key][column].astype(type)

    # Modify pGenDataInput
    df = epm_dict['pGenDataInput'].pivot(index=['scenario', 'zone', 'generator', 'tech', 'fuel'], columns='attribute', values='value').reset_index(['tech', 'fuel'])
    #df = df.loc[:, ['tech', 'fuel']]
    df['fuel'] = df['fuel'].replace(dict_specs['fuel_mapping'])
    # Test if all new fuel values are in dict_specs['fuel_mapping'].values
    for k in df['fuel'].unique():
        if k not in dict_specs['fuel_mapping'].values():
            log_warning(f'{k} not defined as accepted fuels. Please add it to `postprocessing/static/fuels.csv`.')

    df['tech'] = df['tech'].replace(dict_specs['tech_mapping'])
    # Test if all new fuel values are in dict_specs['fuel_mapping'].values
    for k in df['tech'].unique():
        if k not in dict_specs['tech_mapping'].values():
            log_warning(f'{k} not defined as accepted techs. Please add it to `postprocessing/static/technologies.csv`.')

    epm_dict['pGenDataInput'] = df.reset_index()

    return epm_dict


def filter_dataframe(df, conditions):
    """
    Filters a DataFrame based on a dictionary of conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be filtered.
    - conditions (dict): Dictionary specifying filtering conditions.
      - Keys: Column names in the DataFrame.
      - Values: Either a single value (exact match) or a list of values (keeps only matching rows).

    Returns:
    - pd.DataFrame: The filtered DataFrame.

    Example Usage:
    ```
    conditions = {'scenario': 'Baseline', 'year': 2050}
    filtered_df = filter_dataframe(df, conditions)
    ```
    """
    for col, value in conditions.items():
        if isinstance(value, list):
            df = df[df[col].isin(value)]
        else:
            df = df[df[col] == value]
    return df


def filter_dataframe_by_index(df, conditions):
    """
    Filters a DataFrame based on a dictionary of conditions applied to the index level.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be filtered, where conditions apply to index levels.
    - conditions (dict): Dictionary specifying filtering conditions.
      - Keys: Index level names.
      - Values: Either a single value (exact match) or a list of values (keeps only matching rows).

    Returns:
    - pd.DataFrame: The filtered DataFrame.

    Example Usage:
    ```
    conditions = {'scenario': 'Baseline', 'year': 2050}
    filtered_df = filter_dataframe_by_index(df, conditions)
    ```
    """
    for level, value in conditions.items():
        if isinstance(value, list):
            df = df[df.index.get_level_values(level).isin(value)]
        else:
            df = df[df.index.get_level_values(level) == value]
    return df


def process_epm_results(epm_results, dict_specs, keys=None, scenarios_rename=None):
    """
    Processing EPM results to use in plots.
    
    Parameters
    ----------
    epm_results: dict
        Dictionary containing the results
    dict_specs: dict
        Dictionary containing the specifications for the plots
    scenarios_rename: dict, optional
        Dictionary mapping the original scenario names to the standardized names
    mapping_gen_fuel: pd.DataFrame, optional
        DataFrame containing the mapping between generators and fuels
        
    Returns
    -------
    epm_dict: dict
        Dictionary containing the processed results    
    """

    def remove_unused_tech(epm_dict, list_keys):
        """
        Remove rows that correspond to technologies that are never used across the whole time horizon

        Parameters
        ----------
        epm_dict: dict
            Dictionary containing the dataframes
        list_keys: list
            List of keys to remove unused technologies

        Returns
        -------
        epm_dict: dict
            Dictionary containing the dataframes with unused technologies removed
        """
        for key in list_keys:
            epm_dict[key] = epm_dict[key].where((epm_dict[key]['value'] > 2e-6) | (epm_dict[key]['value'] < -2e-6),
                                                np.nan)  # get rid of small values to avoid unneeded labels
            epm_dict[key] = epm_dict[key].dropna(subset=['value'])

        return epm_dict

    if keys is None:  # default keys to process in output
        raise ValueError('Please provide a list of keys to process in output.')

    rename_keys = {}
    
    #-----------------------------
    
    # Check if all keys are in epm_results
    for k in keys:
        if k not in epm_results.keys():
            log_warning(f'{k} not in epm_results.keys().')
        
    # Rename columns
    epm_dict = {k: i.rename(columns=RENAME_COLUMNS) for k, i in epm_results.items() if
                k in keys and k in epm_results.keys()}
    
    # Zero out near-zero values across processed outputs to avoid spurious noise.
    for df in epm_dict.values():
        if 'value' in df.columns:
            mask = df['value'].abs() < TOLERANCE
            if mask.any():
                df.drop(df.index[mask], inplace=True)

    # Rename variables if needed (could be used to convert legacy names)
    if rename_keys is not None:
        epm_dict.update({rename_keys[k]: i for k, i in epm_dict.items() if k in rename_keys.keys()})

    # Rename scenarios
    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    # Remove unused technologies/plants that are lower to simplify plots
    list_keys = ['pReserveSpinningPlantCountry', 'pPlantReserve', 'pCapacityPlant']
    list_keys = [i for i in list_keys if i in epm_dict.keys()]
    epm_dict = remove_unused_tech(epm_dict, list_keys)

    # Convert columns to the right type
    for k, i in epm_dict.items():
        if 'year' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'year': 'int'})
        if 'value' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'value': 'float'})
        if 'zone' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'zone': 'str'})
        if 'country' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'country': 'str'})
        if 'generator' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'generator': 'str'})
        if 'fuel' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'fuel': 'str'})
        if 'scenario' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'scenario': 'str'})
        if 'attribute' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'attribute': 'str'})

    # Align naming across every fuel-based output now shipped with EPM
    tech_fuel_outputs = [
        'pEnergyTechFuel',
        'pEnergyTechFuelCountry',
        'pCapacityTechFuel',
        'pCapacityTechFuelCountry',
        'pNewCapacityTechFuel',
        'pNewCapacityTechFuelCountry',
        'pUtilizationTechFuel',
        'pDispatchTechFuel',
        'pGeneratorTechFuel'
        
    ]
    # Transform tech-fuel in one column
    for key in tech_fuel_outputs:
        if key not in epm_dict:
            continue
        df = epm_dict[key]
        if 'tech' not in df.columns or 'fuel' not in df.columns:
            continue
        
        df['f'] = df['tech'].astype(str) + '-' + df['fuel'].astype(str)
        df.drop(columns=['tech', 'fuel'], inplace=True)
        df.rename({'f': 'fuel'}, inplace=True, axis=1)

    fuel_outputs = [
        'pFuelCosts',
        'pFuelCostsCountry',
        'pFuelConsumption',
        'pFuelConsumptionCountry'
    ]
    for key in fuel_outputs + tech_fuel_outputs:
        standardize_names(epm_dict, key, dict_specs['fuel_mapping'])

    # Add fuel type to Plant-based results
    mapping_gen_fuel = epm_dict['pGeneratorTechFuel'].loc[:, ['scenario', 'generator', 'fuel']]
    # Plant-based results
    plant_result = ['pReserveSpinningPlantZone', 'pPlantAnnualLCOE', 'pEnergyPlant', 'pCapacityPlant',
                    'pDispatchPlant', 'pCostsPlant', 'pUtilizationPlant']
    for key in [k for k in plant_result if k in epm_dict.keys()]:
        epm_dict[key] = epm_dict[key].merge(mapping_gen_fuel, on=['scenario', 'generator'], how='left')

    # Add country to some Zone-based results
    mapping_zone_country = epm_dict['pZoneCountry'].loc[:, ['scenario', 'country', 'zone']]
    # Add country to the results
    zone_result = ['pEnergyPlant', 'pCapacityPlant', 'pDispatchPlant', 'pCostsPlant', 'pUtilizationPlant']
    for key in [k for k in zone_result if k in epm_dict.keys()]:
        epm_dict[key] = epm_dict[key].merge(mapping_zone_country, on=['scenario', 'zone'], how='left')

    return epm_dict


def path_to_extract_results(folder):
    if 'postprocessing' in os.getcwd():  # code is launched from postprocessing folder
        assert 'output' not in folder, 'folder name is not specified correctly'
        RESULTS_FOLDER = os.path.join('..', 'output', folder)
    else:  # code is launched from main root
        if 'output' not in folder:
            RESULTS_FOLDER = os.path.join('output', folder)
        else:
            RESULTS_FOLDER = folder
    return RESULTS_FOLDER


def generate_summary(epm_results, folder):
    """
    Generate a summary of the EPM results.
    
    Parameters
    ----------
    epm_results: dict
        Dictionary containing the EPM results
    folder: str
        Path to the folder where the summary will be saved
        
    Returns
    -------
    summary: pd.DataFrame
        DataFrame containing the summary of the EPM results
    """

    summary = {}
    
    # 1. Costs
        
    if 'pCostsSystem' in epm_results.keys():
        t = epm_results['pCostsSystem'].copy()
        summary.update({'pCostsSystem': t})
    else:
        log_warning('No pCostsSystem in epm_results')
        
    if 'pCostsSystemPerMWh' in epm_results.keys():
        t = epm_results['pCostsSystemPerMWh'].copy()
        t['attribute'] = t['attribute'].str.replace('$m', '$/MWh', regex=False)
        summary.update({'pCostsSystemPerMWh': t})
    else:
        log_warning('No pCostsZonePerMWh in epm_results')

    if 'pCostsZonePerMWh' in epm_results.keys():
        t = epm_results['pCostsZonePerMWh'].copy()
        t['attribute'] = t['attribute'].str.replace('$m', '$/MWh', regex=False)
        summary.update({'pCostsZonePerMWh': t})
    else:
        log_warning('No pCostsZonePerMWh in epm_results')

    if 'pCostsCountryPerMWh' in epm_results.keys():
        t = epm_results['pCostsCountryPerMWh'].copy()
        t['attribute'] = t['attribute'].str.replace('$m', '$/MWh', regex=False)
        summary.update({'pCostsCountryPerMWh': t})
    else:
        log_warning('No pCostsCountryPerMWh in epm_results')

    if 'pYearlyCostsCountry' in epm_results.keys():
        t = epm_results['pYearlyCostsCountry'].copy()
        summary.update({'pYearlyCostsCountry': t})
    else:
        log_warning('No pYearlyCostsCountry in epm_results')

    if 'pYearlyCostsZone' in epm_results.keys():
        t = epm_results['pYearlyCostsZone'].copy()
        t = t[t['value'] > 1e-2]
        summary.update({'pYearlyCostsZone': t})
    else:
        log_warning('No pYearlyCostsZone in epm_results')

    # 2. Capacity
    
    if 'pCapacityTechFuel' in epm_results.keys():
        t = epm_results['pCapacityTechFuel'].copy()
        t['attribute'] = 'Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pCapacityTechFuel': t})
    else:
        log_warning('No pCapacityTechFuel in epm_results')

    if 'pCapacityTechFuelCountry' in epm_results.keys():
        t = epm_results['pCapacityTechFuelCountry'].copy()
        t['attribute'] = 'Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pCapacityTechFuelCountry': t})
    else:
        log_warning('No pCapacityTechFuelCountry in epm_results')

    if 'pNewCapacityTechFuel' in epm_results.keys():
        t = epm_results['pNewCapacityTechFuel'].copy()
        t['attribute'] = 'New Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pNewCapacityTechFuel': t})
    else:
        log_warning('No pNewCapacityTechFuel in epm_results')

    if 'pNewCapacityTechFuelCountry' in epm_results.keys():
        t = epm_results['pNewCapacityTechFuelCountry'].copy()
        t['attribute'] = 'New Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pNewCapacityTechFuelCountry': t})
    else:
        log_warning('No pNewCapacityTechFuelCountry in epm_results')

    if 'pAnnualTransmissionCapacity' in epm_results.keys():
        t = epm_results['pAnnualTransmissionCapacity'].copy()
        t['attribute'] = 'Annual Transmission Capacity: MW'
        t.rename(columns={'z2': 'resolution'}, inplace=True)
        summary.update({'pAnnualTransmissionCapacity': t})
    else:
        log_warning('No pAnnualTransmissionCapacity in epm_results')
        
    if 'pNewTransmissionCapacity' in epm_results.keys():
        t = epm_results['pNewTransmissionCapacity'].copy()
        t['attribute'] = 'New Transmission Capacity: MW'
        t.rename(columns={'z2': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pNewTransmissionCapacity': t})
    else:
        log_warning('No pNewTransmissionCapacity in epm_results')

    # 3. Energy balance

    if 'pEnergyBalance' in epm_results.keys():
        t = epm_results['pEnergyBalance'].copy()
        t = t[t['value'] > 1e-2]
        t.replace({'Total production: GWh': 'Generation: GWh'}, inplace=True)
        summary.update({'pEnergyBalance': t})
    else:
        log_warning('No pEnergyBalance in epm_results')

    if 'pEnergyTechFuel' in epm_results.keys():
        t = epm_results['pEnergyTechFuel'].copy()
        t['attribute'] = 'Energy: GWh'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pEnergyTechFuel': t})
    else:
        log_warning('No pEnergyTechFuel in epm_results')
    
    if 'pEnergyTechFuelCountry' in epm_results.keys():
        t = epm_results['pEnergyTechFuelCountry'].copy()
        t['attribute'] = 'Energy: GWh'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'pEnergyTechFuelCountry': t})
    else:
        log_warning('No pEnergyTechFuelCountry in epm_results')
       
    # 5. Reserves
    
    if 'pReserveSpinningPlantZone' in epm_results.keys():
        t = epm_results['pReserveSpinningPlantZone'].copy()
        t = t.groupby(['scenario', 'zone', 'year'])['value'].sum().reset_index()
        t['attribute'] = 'Spinning Reserve: GWh'
        summary.update({'pReserveSpinningPlantZone': t})
    else:
        log_warning('No pReserveSpinningPlantZone in epm_results')

    if 'pReserveMarginCountry' in epm_results.keys():
        t = epm_results['pReserveMarginCountry'].copy()
        t.replace({'TotalFirmCapacity': 'Firm Capacity: MW', 'ReserveMargin': 'Planning Reserve: MW'}, inplace=True)
        summary.update({'pReserveMarginResCountry': t})
    else:
        log_warning('No pReserveMarginCountry in epm_results')

    # 6. Interconnections
    
    if 'pInterchange' in epm_results.keys():
        t = epm_results['pInterchange'].copy()
        t['attribute'] = 'Annual Energy Exchanges: GWh'
        t.rename(columns={'z2': 'resolution'}, inplace=True)
        summary.update({'pInterchange': t})
    else:
        log_warning('No pInterchange in epm_results')
            
    if 'pInterchangeExternalExports' in epm_results.keys():
        t = epm_results['pInterchangeExternalExports'].copy()
        t['attribute'] = 'Annual Energy Exports External: GWh'
        t.rename(columns={'zext': 'resolution'}, inplace=True)
        summary.update({'pInterchangeExternalExports': t})
    else:
        log_warning('No pInterchangeExternalExports in epm_results')
        
    if 'pInterchangeExternalImports' in epm_results.keys():
        t = epm_results['pInterchangeExternalImports'].copy()
        t['attribute'] = 'Annual Energy Imports External: GWh'
        t.rename(columns={'zext': 'resolution'}, inplace=True)
        summary.update({'pInterchangeExternalImports': t})
    else:
        log_warning('No pInterchangeExternalImports in epm_results')

    # 7. Emissions

    if 'pEmissionsZone' in epm_results.keys():
        t = epm_results['pEmissionsZone'].copy()
        t['attribute'] = 'Emissions: MtCO2'
        summary.update({'pEmissionsZone': t})
    else:
        log_warning('No pEmissionsZone in epm_results')

    if 'pEmissionsIntensityZone' in epm_results.keys():
        t = epm_results['pEmissionsIntensityZone'].copy()
        t['attribute'] = 'Emissions: tCO2/GWh'
        summary.update({'pEmissionsIntensityZone': t})
    else:
        log_warning('No pEmissionsIntensityZone in epm_results')

    # 8. Prices
    
    if 'pYearlyPrice' in epm_results.keys():
        t = epm_results['pYearlyPrice'].copy()
        t['attribute'] = 'Price: $/MWh'
        summary.update({'pYearlyPrice': t})
    else:
        log_warning('No pYearlyPrice in epm_results')


    # Concatenate all dataframes in the summary dictionary

    summary = pd.concat(summary)

    # Define the order that will appear in the summary.csv file
    if False:
        order = ['NPV of system cost: $m',
                "Generation costs: $m",
                "Fixed O&M: $m",
                "Variable O&M: $m",
                "Fuel costs: $m",
                "Transmission costs: $m",
                "Spinning reserve costs: $m",
                "Unmet demand costs: $m",
                "Unmet country spinning reserve costs: $m",
                "Unmet country planning reserve costs: $m",
                "Unmet country CO2 backstop cost: $m",
                "Unmet system planning reserve costs: $m",
                "Unmet system spinning reserve costs: $m",
                "Unmet system CO2 backstop cost: $m",
                "Excess generation: $m",
                "VRE curtailment: $m",
                "Import costs with external zones: $m",
                "Export revenues with external zones: $m",
                "Import costs with internal zones: $m",
                "Export revenues with internal zones: $m",
                "Trade shared benefits: $m",
                "Carbon costs: $m",
                'Demand: GWh', 
                'Generation: GWh', 
                'Unmet demand: GWh',
                'Surplus generation: GWh',
                'Peak demand: MW', 
                'Firm Capacity: MW', 
                'Planning Reserve: MW',
                'Spinning Reserve: GWh',
                'Average Cost: $/MWh',
                'Capex: $m'
                ]
        order = [i for i in order if i in summary['attribute'].unique()]
        order = order + [i for i in summary['attribute'].unique() if i not in order]

    summary.reset_index(drop=True, inplace=True)
    summary = summary.set_index(['scenario', 'country', 'zone', 'attribute', 'resolution', 'year']).squeeze().unstack('scenario')
    summary.reset_index(inplace=True)
    #summary = summary.sort_values()
    
    if False:
        # Create a mapping of attributes to their position in the list
        order_dict = {attr: index for index, attr in enumerate(order)}
        summary = summary.sort_values(by="attribute", key=lambda x: x.map(order_dict))

    zone_to_country = epm_results['pZoneCountry'].set_index('zone')['country'].to_dict()    
    summary['country'] = summary['country'].fillna(summary['zone'].map(zone_to_country))
    # Remove duplicates
    
    def drop_redundant_country_rows(df):
        idx_cols = ["country", "attribute", "resolution", "year"]

        # mark entries that already sit at the zone level
        zone_level = df["zone"].notna() & df["zone"].ne("")

        # countries with a single zone have redundant country-level rows
        single_zone_countries = (
            df.loc[zone_level]
              .groupby("country")["zone"]
              .nunique()
              .pipe(lambda s: set(s[s == 1].index))
        )

        # capture attribute/resolution/year combos available at zone level
        zone_keys = (
            df.loc[zone_level, idx_cols]
              .drop_duplicates()
              .apply(tuple, axis=1)
        )

        # drop country-level counterparts when they duplicate zone-level info
        mask = (
            df["country"].isin(single_zone_countries)
            & ~zone_level
            & df[idx_cols].apply(tuple, axis=1).isin(zone_keys)
        )

        return df.loc[~mask].copy()
    
    summary = drop_redundant_country_rows(summary)
    
    summary.round(1).to_csv(os.path.join(folder, 'summary.csv'), index=False)


def generate_plants_summary(epm_results, folder):
    summary_detailed = {}
    if 'pCapacityPlant' in epm_results.keys():
        temp = epm_results['pCapacityPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Capacity: MW': temp.copy()})
    else:
        log_warning('No pCapacityPlan in epm_results')

    if 'pUtilizationPlant' in epm_results.keys():
        temp = epm_results['pUtilizationPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Utilization: percent': temp.copy()})
    else:
        log_warning('No pUtilizationPlant in epm_results')

    if 'pEnergyPlant' in epm_results.keys():
        temp = epm_results['pEnergyPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Energy: GWh': temp.copy()})
    else:
        log_warning('No pEnergyPlant in epm_results')

    if 'pReserveSpinningPlantZone' in epm_results.keys():
        temp = epm_results['pReserveSpinningPlantZone'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Spinning Reserve: GWh': temp.copy()})
    else:
        log_warning('No pReserveSpinningPlantZone in epm_results')

    if 'pCostsPlant' in epm_results.keys():
        temp = epm_results['pCostsPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year', 'attribute']).squeeze().unstack(
            'scenario')
        temp.reset_index(inplace=True)
        temp = temp.sort_index()
        grouped_dfs = {key: group.drop(columns=['attribute']) for key, group in temp.groupby('attribute')}
        summary_detailed.update(grouped_dfs)
    else:
        log_warning('No pCostsPlant in epm_results')

    if 'pPlantAnnualLCOE' in epm_results.keys():
        temp = epm_results['pPlantAnnualLCOE'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'LCOE: $/MWH': temp.copy()})
    else:
        log_warning('No pPlantAnnualLCOE in epm_results')

    summary_detailed = pd.concat(summary_detailed).round(2)
    summary_detailed.index.names = ['Variable', '']
    summary_detailed = summary_detailed.droplevel('', axis=0)
    summary_detailed.to_csv(os.path.join(folder, 'summary_generators.csv'), index=True)


def calculate_npv(data, discount_rate, start_year=None, end_year=None):
    """
    Calculate the net present value interpolation and discounting.

    Parameters:
    ----------
    data: pd.DataFrame
        DataFrame with columns ['scenario', 'year', 'value']
    discount_rate: float
        Discount rate (e.g., 0.05 for 5%)
    start_year: int
        Starting year for cost calculation
    end_year: int
        Ending year for cost calculation

    Returns:
    -------
        pd.Series: Total system cost for each scenario
    """
    if start_year is None:
        start_year = data['year'].min()

    if end_year is None:
        end_year = data['year'].max()

    results = []

    # Group data by scenario
    for scenario, group in data.groupby('scenario'):
        # Sort data by year
        group = group.sort_values(by='year')

        # Create a full range of years
        years = np.arange(start_year, end_year + 1)

        # Interpolate costs
        interpolated_costs = np.interp(years, group['year'], group['value'])

        # Discount costs
        discounted_costs = [
            cost / ((1 + discount_rate) ** (year - start_year))
            for year, cost in zip(years, interpolated_costs)
        ]

        # Sum up discounted costs
        total_cost = sum(discounted_costs)

        # Store result
        results.append({'scenario': scenario, 'cost': total_cost})

    return pd.DataFrame(results).set_index('scenario').squeeze()


def process_simulation_results(FOLDER, SCENARIOS_RENAME=None, keys_results=None):
    # Create the folder path
    def adjust_color(color, factor=0.1):
        """Adjusts the color slightly by modifying its HSL components."""
        rgb = mcolors.to_rgb(color)  # Convert to RGB
        h, l, s = colorsys.rgb_to_hls(*rgb)  # Convert to HLS

        # Adjust lightness slightly to differentiate (factor controls how much)
        l = min(1, max(0, l + factor * (0.5 - l)))

        # Convert back to RGB
        new_rgb = colorsys.hls_to_rgb(h, l, s)
        return mcolors.to_hex(new_rgb)

    RESULTS_FOLDER = path_to_extract_results(FOLDER)
    log_info(f"Processing simulation results from {RESULTS_FOLDER}")

    # Read the plot specifications
    dict_specs = read_plot_specs()
    try:
        from .plots import set_default_fuel_order  # Lazy import to avoid circular dependency
        set_default_fuel_order(dict_specs.get('fuel_order'))
    except (ImportError, AttributeError):
        pass

    # Extract and process EPM inputs
    if False:
        epm_input = extract_epm_folder(RESULTS_FOLDER, file='input.gdx')
        epm_input = process_epm_inputs(epm_input, dict_specs, scenarios_rename=SCENARIOS_RENAME)

    # Extract and process EPM results
    epm_results = extract_epm_folder(RESULTS_FOLDER, file='epmresults.gdx')
    epm_results = process_epm_results(epm_results, dict_specs, scenarios_rename=SCENARIOS_RENAME, keys=keys_results)
    log_info(f"Loaded {len(epm_results)} processed result tables")

    # Update color dict with plant colors
    if True:
        if 'pCapacityPlant' in epm_results.keys():
            temp = epm_results['pCapacityPlant'].copy()
            plant_fuel_pairs = temp[['generator', 'fuel']].drop_duplicates()

            # Map base colors from fuel types
            plant_fuel_pairs['colors'] = plant_fuel_pairs['fuel'].map(dict_specs['colors'])

            # Generate slightly varied colors for each generator
            plant_fuel_pairs['colors'] = plant_fuel_pairs.apply(
                lambda row: adjust_color(row['colors'], factor=0.2 * hash(row['generator']) % 5), axis=1
            )

            # Create the mapping
            plant_to_color = dict(zip(plant_fuel_pairs['generator'], plant_fuel_pairs['colors']))

            # Update dict_specs with the new colors
            dict_specs['colors'].update(plant_to_color)

    return RESULTS_FOLDER, dict_specs, epm_results

