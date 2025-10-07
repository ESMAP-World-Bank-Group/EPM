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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import gams.transfer as gt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from matplotlib.ticker import MaxNLocator, FixedLocator
import colorsys
import matplotlib.colors as mcolors
from PIL import Image
import base64
from io import BytesIO
import io
from shapely.geometry import Point, Polygon
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString, Point, LinearRing
import argparse
import shutil


FUELS = os.path.join('static', 'fuels.csv')
TECHS = os.path.join('static', 'technologies.csv')
COLORS = os.path.join('static', 'colors.csv')
GEOJSON = os.path.join('static', 'countries.geojson')
GEOJSON_TO_EPM = os.path.join('static', 'geojson_to_epm.csv')


KEYS_RESULTS = {
    # 1. Capacity expansion
    'pCapacityPlant', 'pCapacityFuel',
    'pNewCapacityFuel', 'pNewCapacityFuelCountry',
    'pAnnualTransmissionCapacity', 'pAdditionalCapacity',
    # 2. Cost
    'pPrice',
    'pCostsPlant', 'pYearlyCostsZone', 'pYearlyCostsCountry', 
    'pCostsZone','pCostsSystem',
    # 3. Energy balance
    'pEnergyPlant', 'pEnergyFuel',
    'pEnergyBalance',
    'pUtilizationPlant', 'pUtilizationFuel',
    # 4. Energy dispatch
    'pDispatchPlant', 'pDispatch', 'pDispatchFuel',
    # 5. Reserves
    'pReserveSpinningPlantZone', 'pReserveSpinningPlantCountry',
    'pReserveMarginCountry',
    # 6. Interconnections
    'pInterchange', 'pInterconUtilization',
    # 7. Emissions
    'pEmissionsZone',
    # 10. Metrics
    'pPlantAnnualLCOE',
    'pZonalAverageCost', 'pSystemAverageCost',
    # 11. Other
    'pSolverParameters'
}

NAME_COLUMNS = {
    'pDispatchFuel': 'fuel',
    'pDispatchPlant': 'fuel',
    'pDispatch': 'attribute',
    'pYearlyCostsZone': 'attribute',
    'pCapacityFuel': 'fuel',
    'pEnergyFuel': 'fuel',
    'pDispatchReserve':'attribute',
    'pNetExchange':'attribute'
}

UNIT = {
    'Capex: $m': 'M$/year',
    'Unmet demand costs: : $m': 'M$'
}

RENAME_COLUMNS = {'c': 'country', 'c_0': 'country', 'y': 'year', 'v': 'value', 's': 'scenario', 'uni': 'attribute',
                  'z': 'zone', 'g': 'generator', 'gen': 'generator',
                  'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't'}
TYPE_COLUMNS  = {'year': int, 'season': str, 'day': str, 'tech': str, 'fuel': str}


def read_plot_specs(folder=''):
    """
    Read the specifications for the plots from the static files.
    
    Returns:
    -------
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    colors = pd.read_csv(os.path.join(folder, COLORS))
    fuel_mapping = pd.read_csv(os.path.join(folder, FUELS))
    tech_mapping = pd.read_csv(os.path.join(folder, TECHS))
    countries = gpd.read_file(os.path.join(folder, GEOJSON))
    geojson_to_epm = pd.read_csv(os.path.join(folder, GEOJSON_TO_EPM))

    dict_specs = {
        'colors': colors.set_index('Processing')['Color'].to_dict(),
        'fuel_mapping': fuel_mapping.set_index('EPM_Fuel')['Processing'].to_dict(),
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict(),
        'map_countries': countries,
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
        print(f"Error while loading gdx file : {e}")
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
                print(f"Folder downloaded : {output_csv}")
            else:
                print(f"No data for param : {param}")
        except Exception as e:
            print(f"Error with param {param}: {e}")

    print("Conversion over! All files saved to folder 'data/'.")


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
        print(f'{key} not found in epm_dict')


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

    """epm_input['ftfindex'].rename(columns={'f': 'fuel'}, inplace=True)
    mapping_fuel = epm_input['ftfindex'].loc[:, ['fuel', 'value']].drop_duplicates().set_index(
        'value').squeeze().to_dict()

    # TODO: this may be extracted from pGenDataInput now if we get rid of pTechDataExcel and pZoneIndex in future versions
    temp = epm_input['pTechData']
    temp['uni'] = temp['uni'].astype(str)
    temp = temp[temp['uni'] == 'Assigned Value']
    mapping_tech = temp.loc[:, ['Abbreviation', 'value']].drop_duplicates().set_index(
        'value').squeeze()
    mapping_tech.replace(dict_specs['tech_mapping'], inplace=True)"""

    # Modify pGenDataInput
    df = epm_dict['pGenDataInput'].pivot(index=['scenario', 'zone', 'generator', 'tech', 'fuel'], columns='pGenDataInputHeader', values='value').reset_index(['tech', 'fuel'])
    #df = df.loc[:, ['tech', 'fuel']]
    df['fuel'] = df['fuel'].replace(dict_specs['fuel_mapping'])
    # Test if all new fuel values are in dict_specs['fuel_mapping'].values
    for k in df['fuel'].unique():
        if k not in dict_specs['fuel_mapping'].values():
            print(f'{k} not defined as accepted fuels. Please add it to `postprocessing/static/fuels.csv`.')

    df['tech'] = df['tech'].replace(dict_specs['tech_mapping'])
    # Test if all new fuel values are in dict_specs['fuel_mapping'].values
    for k in df['tech'].unique():
        if k not in dict_specs['tech_mapping'].values():
            print(f'{k} not defined as accepted techs. Please add it to `postprocessing/static/technologies.csv`.')

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


def process_epm_results(epm_results, dict_specs, keys=None, scenarios_rename=None, mapping_gen_fuel=None,
                        mapping_zone_country=None):
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
            print(f'{k} not in epm_results.keys().')
    
    # Rename columns
    epm_dict = {k: i.rename(columns=RENAME_COLUMNS) for k, i in epm_results.items() if
                k in keys and k in epm_results.keys()}

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

    # Standardize names for outputs
    standardize_names(epm_dict, 'pEnergyFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pCapacityFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pNewCapacityFuelCountry', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pUtilizationFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pDispatchFuel', dict_specs['fuel_mapping'])

    # Add fuel type to Plant-based results
    if mapping_gen_fuel is not None:
        # Plant-based results
        plant_result = ['pReserveSpinningPlantZone', 'pPlantAnnualLCOE', 'pEnergyPlant', 'pCapacityPlant',
                        'pDispatchPlant', 'pCostsPlant', 'pUtilizationPlant']
        for key in [k for k in plant_result if k in epm_dict.keys()]:
            epm_dict[key] = epm_dict[key].merge(mapping_gen_fuel, on=['scenario', 'generator'], how='left')

    # Add country to some Zone-based results
    if mapping_zone_country is not None:
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


def generate_summary(epm_results, folder, epm_input):
    """
    Generate a summary of the EPM results.
    
    Parameters
    ----------
    epm_results: dict
        Dictionary containing the EPM results
    folder: str
        Path to the folder where the summary will be saved
    epm_input: dict
        Dictionary containing the EPM input data
        
    Returns
    -------
    summary: pd.DataFrame
        DataFrame containing the summary of the EPM results
    """

    summary = {}
    
    # 1. Costs

    if 'pSystemAverageCost' in epm_results.keys():
        t = epm_results['pSystemAverageCost'].copy()
        t['attribute'] = 'Average Cost: $/MWh'
        summary.update({'SystemAverageCost': t})
    else:
        print('No pSystemAverageCost in epm_results')

    if False:
        if 'pZonalAverageCost' in epm_results.keys():
            t = epm_results['pZonalAverageCost'].copy()
            t.rename(columns={'uni_1': 'attribute'}, inplace=True)
            t.drop('uni_2', axis=1, inplace=True)
            summary.update({'pZonalAverageCost': t})
        else:
            print('No pZonalAverageCost in epm_results')

    if 'pYearlyCostsCountry' in epm_results.keys():
        t = epm_results['pYearlyCostsCountry'].copy()
        summary.update({'pYearlyCostsCountry': t})
    else:
        print('No pYearlyCostsCountry in epm_results')

    if 'pYearlyCostsZone' in epm_results.keys():
        t = epm_results['pYearlyCostsZone'].copy()
        t = t[t['value'] > 1e-2]
        summary.update({'pYearlyCostsZone': t})
    else:
        print('No pYearlyCostsZone in epm_results')

    # 2. Capacity
    
    if 'pCapacityFuel' in epm_results.keys():
        t = epm_results['pCapacityFuel'].copy()
        t['attribute'] = 'Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'Capacity: MW': t})
    else:
        print('No pCapacityFuel in epm_results')

    if 'pNewCapacityFuel' in epm_results.keys():
        t = epm_results['pNewCapacityFuel'].copy()
        t['attribute'] = 'New Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'NewCapacity: MW': t})
    else:
        print('No pNewCapacityFuel in epm_results')

    if 'pAnnualTransmissionCapacity' in epm_results.keys():
        t = epm_results['pAnnualTransmissionCapacity'].copy()
        t['attribute'] = 'Annual Transmission Capacity: MW'
        t.rename(columns={'z2': 'resolution'}, inplace=True)
        summary.update({'Annual Transmission Capacity: MW': t})
    else:
        print('No pAnnualTransmissionCapacity in epm_results')
        
    if 'pAdditionalCapacity' in epm_results.keys():
        t = epm_results['pAdditionalCapacity'].copy()
        t['attribute'] = 'Additional Capacity: MW'
        t.rename(columns={'z2': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'Additional Capacity: MW': t})
    else:
        print('No pAdditionalCapacity in epm_results')

    # 3. Energy balance

    if 'pEnergyBalance' in epm_results.keys():
        t = epm_results['pEnergyBalance'].copy()
        t = t[t['value'] > 1e-2]
        t.replace({'Total production: GWh': 'Generation: GWh'}, inplace=True)
        summary.update({'pEnergyBalance': t})
    else:
        print('No pEnergyBalance in epm_results')

    if 'pEnergyFuel' in epm_results.keys():
        t = epm_results['pEnergyFuel'].copy()
        t['attribute'] = 'Energy: GWh'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'Energy: GWh': t})
    else:
        print('No pEnergyFuel in epm_results')
        
    # 4. Reserves
    
    if 'pReserveSpinningPlantZone' in epm_results.keys():
        t = epm_results['pReserveSpinningPlantZone'].copy()
        t = t.groupby(['scenario', 'zone', 'year'])['value'].sum().reset_index()
        t['attribute'] = 'Spinning Reserve: GWh'
        summary.update({'pReserveSpinningPlantZone': t})
    else:
        print('No pReserveSpinningPlantZone in epm_results')

    if 'pReserveMarginCountry' in epm_results.keys():
        t = epm_results['pReserveMarginCountry'].copy()
        t.replace({'TotalFirmCapacity': 'Firm Capacity: MW', 'ReserveMargin': 'Planning Reserve: MW'}, inplace=True)
        summary.update({'pReserveMarginResCountry': t})
    else:
        print('No pReserveMarginCountry in epm_results')

    # 5. Emissions

    if 'pEmissionsZone' in epm_results.keys():
        t = epm_results['pEmissionsZone'].copy()
        t['attribute'] = 'Emissions: MtCO2'
        summary.update({'pEmissionsZone': t})
    else:
        print('No pEmissionsZone in epm_results')

    # Concatenate all dataframes in the summary dictionary

    summary = pd.concat(summary)

    # Define the order that will appear in the summary.csv file
    order = ['NPV of system cost: $m',
            "Annualized capex: $m",
            "Fixed O&M: $m",
            "Variable O&M: $m",
            "Fuel costs: $m",
            "Transmission additions: $m",
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
    # Create a mapping of attributes to their position in the list
    order_dict = {attr: index for index, attr in enumerate(order)}
    summary = summary.sort_values(by="attribute", key=lambda x: x.map(order_dict))

    zone_to_country = epm_input['zcmap'].set_index('zone')['country'].to_dict()
    summary['country'] = summary['country'].fillna(summary['zone'].map(zone_to_country))

    summary.round(1).to_csv(os.path.join(folder, 'summary.csv'), index=False)


def generate_plants_summary(epm_results, folder):
    summary_detailed = {}
    if 'pCapacityPlant' in epm_results.keys():
        temp = epm_results['pCapacityPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Capacity: MW': temp.copy()})
    else:
        print('No pCapacityPlan in epm_results')

    if 'pUtilizationPlant' in epm_results.keys():
        temp = epm_results['pUtilizationPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Utilization: percent': temp.copy()})
    else:
        print('No pUtilizationPlant in epm_results')

    if 'pEnergyPlant' in epm_results.keys():
        temp = epm_results['pEnergyPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Energy: GWh': temp.copy()})
    else:
        print('No pEnergyPlant in epm_results')

    if 'pReserveSpinningPlantZone' in epm_results.keys():
        temp = epm_results['pReserveSpinningPlantZone'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Spinning Reserve: GWh': temp.copy()})
    else:
        print('No pReserveSpinningPlantZone in epm_results')

    if 'pCostsPlant' in epm_results.keys():
        temp = epm_results['pCostsPlant'].copy()
        temp = temp.set_index(['scenario', 'country', 'zone', 'generator', 'fuel', 'year', 'attribute']).squeeze().unstack(
            'scenario')
        temp.reset_index(inplace=True)
        temp = temp.sort_index()
        grouped_dfs = {key: group.drop(columns=['attribute']) for key, group in temp.groupby('attribute')}
        summary_detailed.update(grouped_dfs)
    else:
        print('No pCostsPlant in epm_results')

    if 'pPlantAnnualLCOE' in epm_results.keys():
        temp = epm_results['pPlantAnnualLCOE'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'LCOE: $/MWH': temp.copy()})
    else:
        print('No pPlantAnnualLCOE in epm_results')

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


def process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='postprocessing',
                               graphs_folder = 'img', keys_results=None):
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

        GRAPHS_FOLDER = os.path.join(RESULTS_FOLDER, graphs_folder)
        if not os.path.exists(GRAPHS_FOLDER):
            os.makedirs(GRAPHS_FOLDER)
            print(f'Created folder {GRAPHS_FOLDER}')

        # Read the plot specifications
        dict_specs = read_plot_specs(folder=folder)

        # Extract and process EPM inputs
        epm_input = extract_epm_folder(RESULTS_FOLDER, file='input.gdx')
        epm_input = process_epm_inputs(epm_input, dict_specs, scenarios_rename=SCENARIOS_RENAME)
        mapping_gen_fuel = epm_input['pGenDataInput'].loc[:, ['scenario', 'generator', 'fuel']]
        mapping_zone_country = epm_input['zcmap'].loc[:, ['scenario', 'zone', 'country']]

        # Extract and process EPM results
        epm_results = extract_epm_folder(RESULTS_FOLDER, file='epmresults.gdx')
        epm_results = process_epm_results(epm_results, dict_specs, scenarios_rename=SCENARIOS_RENAME,
                                          mapping_gen_fuel=mapping_gen_fuel, mapping_zone_country=mapping_zone_country,
                                          keys=KEYS_RESULTS)

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

        return RESULTS_FOLDER, GRAPHS_FOLDER, dict_specs, epm_input, epm_results, mapping_gen_fuel


def generate_summary_excel(results_folder, template_file="epm_results_summary_dis_template.xlsx"):

    # Get the data

    results_folder, graphs_folder, dict_specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results(
    results_folder, folder='')

    tabs_to_update=['pEnergyBalance','pCapacityFuel','pEnergyFuel','pYearlyCostsZone','pYearlyCostsZoneCountry','pEmissions','pInterchange']

    output_file = f"{results_folder}_results_summary_dis.xlsx"

    # Create the file from the template
    shutil.copyfile(template_file, output_file)

    # Charge data
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for tab in tabs_to_update:
            if tab in epm_results.keys():
                df_temp = epm_results[tab].copy()
                col_order = [col for col in df_temp.columns if col != "scenario"] + ["scenario"]
                df_temp = df_temp[col_order]
                df_temp.to_excel(writer, sheet_name=tab, index=False)
            else:
                print(f"No data for '{tab}' — ignored")

    print(f"Excel generated : {output_file}")