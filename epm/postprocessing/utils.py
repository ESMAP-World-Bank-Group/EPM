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
import folium
import base64
from io import BytesIO
import io
from shapely.geometry import Point, Polygon

FUELS = os.path.join('static', 'fuels.csv')
TECHS = os.path.join('static', 'technologies.csv')
COLORS = os.path.join('static', 'colors.csv')
GEOJSON = os.path.join('static', 'countries.geojson')
GEOJSON_TO_EPM = os.path.join('static', 'geojson_to_epm.csv')

NAME_COLUMNS = {
    'pFuelDispatch': 'fuel',
    'pPlantDispatch': 'fuel',
    'pDispatch': 'attribute',
    'pCostSummary': 'attribute',
    'pCapacityByFuel': 'fuel',
    'pEnergyByFuel': 'fuel',
    'pDispatchReserve':'attribute'
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


def extract_epm_folder_by_scenario(results_folder, file='epmresults.gdx', save_to_csv=False):
    """
    Extract information from a folder containing multiple scenarios,
    keeping the results separate for each scenario.

    Parameters
    ----------
    results_folder: str
        Path to the folder containing the scenarios
    file: str, optional, default='epmresults.gdx'
        Name of the gdx file to extract

    Returns
    -------
    scenario_dict: dict
        Dictionary with scenario names as keys, each containing a dict of DataFrames
    """
    scenario_dict = {}
    for scenario in [i for i in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, i))]:
        gdx_path = os.path.join(results_folder, scenario, file)
        if os.path.exists(gdx_path):
            scenario_dict[scenario] = extract_gdx(gdx_path)

    if save_to_csv:
        save_csv = Path(results_folder) / Path('csv')
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
        temp = temp.groupby([i for i in temp.columns if i != 'value']).sum().reset_index()

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

    keys = ['pGenDataExcel', 'ftfindex', 'pTechData', 'pZoneIndex', 'pDemandProfile', 'pDemandForecast', 'pSettings',
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

    # TODO: this may be extracted from pGenDataExcel now if we get rid of pTechDataExcel and pZoneIndex in future versions
    temp = epm_input['pTechData']
    temp['uni'] = temp['uni'].astype(str)
    temp = temp[temp['uni'] == 'Assigned Value']
    mapping_tech = temp.loc[:, ['Abbreviation', 'value']].drop_duplicates().set_index(
        'value').squeeze()
    mapping_tech.replace(dict_specs['tech_mapping'], inplace=True)"""

    # Modify pGenDataExcel
    df = epm_dict['pGenDataExcel'].pivot(index=['scenario', 'zone', 'generator', 'tech', 'fuel'], columns='attribute', values='value').reset_index(['tech', 'fuel'])
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

    epm_dict['pGenDataExcel'] = df.reset_index()

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


def process_epm_results(epm_results, dict_specs, scenarios_rename=None, mapping_gen_fuel=None,
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

    keys = {'pDemandSupplyCountry', 'pDemandSupply', 'pPeakCapacity', 'pEnergyByPlant', 'pEnergyByFuel', 'pCapacityByFuel', 'pCapacityPlan',
            'pPlantUtilization', 'pFuelUtilization', 'pCostSummary', 'pCostSummaryCountry', 'pEmissions', 'pPrice', 'pHourlyFlow',
            'pDispatch', 'pFuelDispatch', 'pPlantFuelDispatch', 'pInterconUtilization',
            'pSpinningReserveByPlantCountry', 'InterconUtilization', 'pInterchange', 'Interchange', 'interchanges', 'pInterconUtilizationExtImp',
            'pInterconUtilizationExtExp', 'pInterchangeExtExp', 'InterchangeExtImp', 'annual_line_capa', 'pAnnualTransmissionCapacity',
            'AdditiononalCapacity_trans', 'pDemandSupplySeason', 'pCurtailedVRET', 'pCurtailedStoHY',
            'pNewCapacityFuelCountry', 'pPlantAnnualLCOE', 'pStorageComponents', 'pNPVByYear',
            'pSpinningReserveByPlantCountry', 'pPlantDispatch', 'pSummary', 'pSystemAverageCost', 'pNewCapacityFuel',
            'pCostSummaryWeightedAverageCountry', 'pReserveMarginResCountry', 'pSpinningReserveByPlantZone',
            'pCostsbyPlant', 'pYearlyTrade', 'pSolverParameters'}

    rename_keys = {}
    for k in keys:
        if k not in epm_results.keys():
            print(f'{k} not in epm_results.keys().')
    # Rename columns
    epm_dict = {k: i.rename(columns=RENAME_COLUMNS) for k, i in epm_results.items() if
                k in keys and k in epm_results.keys()}

    # pSpinningReserveByPlantCountry, pSpinningReserveByPlantZone

    if rename_keys is not None:
        epm_dict.update({rename_keys[k]: i for k, i in epm_dict.items() if k in rename_keys.keys()})

    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    list_keys = ['pSpinningReserveByPlantCountry', 'pPlantReserve', 'pCapacityPlan']
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

    # Standardize names
    standardize_names(epm_dict, 'pEnergyByFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pCapacityByFuel', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pNewCapacityFuelCountry', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pFuelUtilization', dict_specs['fuel_mapping'])

    standardize_names(epm_dict, 'pFuelDispatch', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pPlantFuelDispatch', dict_specs['tech_mapping'])

    # Add fuel type to the results
    if mapping_gen_fuel is not None:
        # Add fuel type to the results
        plant_result = ['pSpinningReserveByPlantZone', 'pPlantAnnualLCOE', 'pEnergyByPlant', 'pCapacityPlan',
                        'pPlantDispatch', 'pCostsbyPlant', 'pPlantUtilization']
        for key in [k for k in plant_result if k in epm_dict.keys()]:
            epm_dict[key] = epm_dict[key].merge(mapping_gen_fuel, on=['scenario', 'generator'], how='left')

    # Add country to the results
    if mapping_zone_country is not None:
        # Add country to the results
        plant_result = ['pEnergyByPlant', 'pCapacityPlan', 'pPlantDispatch', 'pCostsbyPlant', 'pPlantUtilization']
        for key in [k for k in plant_result if k in epm_dict.keys()]:
            epm_dict[key] = epm_dict[key].merge(mapping_zone_country, on=['scenario', 'zone'], how='left')

    return epm_dict


def process_simulation_results(FOLDER, SCENARIOS_RENAME=None, folder='postprocessing'):
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

    # TODO: Clean that
    if 'postprocessing' in os.getcwd():  # code is launched from postprocessing folder
        assert 'output' not in FOLDER, 'FOLDER name is not specified correctly'
        # RESULTS_FOLDER = FOLDER
        RESULTS_FOLDER = os.path.join('..', 'output', FOLDER)
    else:  # code is launched from main root
        if 'output' not in FOLDER:
            RESULTS_FOLDER = os.path.join('output', FOLDER)
        else:
            RESULTS_FOLDER = FOLDER

    GRAPHS_FOLDER = 'img'
    GRAPHS_FOLDER = os.path.join(RESULTS_FOLDER, GRAPHS_FOLDER)
    if not os.path.exists(GRAPHS_FOLDER):
        os.makedirs(GRAPHS_FOLDER)
        print(f'Created folder {GRAPHS_FOLDER}')

    # Read the plot specifications
    dict_specs = read_plot_specs(folder=folder)

    # Extract and process EPM inputs
    epm_input = extract_epm_folder(RESULTS_FOLDER, file='input.gdx')
    epm_input = process_epm_inputs(epm_input, dict_specs, scenarios_rename=SCENARIOS_RENAME)
    mapping_gen_fuel = epm_input['pGenDataExcel'].loc[:, ['scenario', 'generator', 'fuel']]
    mapping_zone_country = epm_input['zcmap'].loc[:, ['scenario', 'zone', 'country']]

    # Extract and process EPM results
    epm_results = extract_epm_folder(RESULTS_FOLDER, file='epmresults.gdx')
    epm_results = process_epm_results(epm_results, dict_specs, scenarios_rename=SCENARIOS_RENAME,
                                      mapping_gen_fuel=mapping_gen_fuel, mapping_zone_country=mapping_zone_country)

    # Update color dict with plant colors
    if True:
        # Copy results
        temp = epm_results['pCapacityPlan'].copy()
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


def generate_summary(epm_results, folder, epm_input):

    summary = {}

    if 'pSystemAverageCost' in epm_results.keys():
        t = epm_results['pSystemAverageCost'].copy()
        t['attribute'] = 'Average Cost: $/MWh'
        summary.update({'SystemAverageCost': t})
    else:
        print('No pSystemAverageCost in epm_results')

    if 'pCostSummaryWeightedAverageCountry' in epm_results.keys():
        t = epm_results['pCostSummaryWeightedAverageCountry'].copy()
        t.rename(columns={'uni_1': 'attribute'}, inplace=True)
        t.drop('uni_2', axis=1, inplace=True)
        summary.update({'pCostSummaryWeightedAverageCountry': t})
    else:
        print('No pCostSummaryWeightedAverageCountry in epm_results')

    if 'pSummary' in epm_results.keys():
        t = epm_results['pSummary'].copy()
        t = t[t['value'] > 1e-2]
        summary.update({'pSummary': t})
    else:
        print('No pSummary in epm_results')

    if 'pDemandSupply' in epm_results.keys():
        t = epm_results['pDemandSupply'].copy()
        t = t[t['value'] > 1e-2]
        t.replace({'Total production: GWh': 'Generation: GWh'}, inplace=True)
        summary.update({'pDemandSupply': t})
    else:
        print('No pDemandSupply in epm_results')

    if 'pReserveMarginResCountry' in epm_results.keys():
        t = epm_results['pReserveMarginResCountry'].copy()
        t.replace({'TotalFirmCapacity': 'Firm Capacity: MW', 'ReserveMargin': 'Planning Reserve: MW'}, inplace=True)
        summary.update({'pReserveMarginResCountry': t})
    else:
        print('No pDemandSupply in epm_results')

    if 'pEmissions' in epm_results.keys():
        t = epm_results['pEmissions'].copy()
        t['attribute'] = 'Emissions: MtCO2'
        summary.update({'pEmissions': t})
    else:
        print('No pEmissions in epm_results')


    if 'pSpinningReserveByPlantZone' in epm_results.keys():
        t = epm_results['pSpinningReserveByPlantZone'].copy()
        t = t.groupby(['scenario', 'zone', 'year'])['value'].sum().reset_index()
        t['attribute'] = 'Spinning Reserve: GWh'
        summary.update({'pSpinningReserveByPlantZone': t})
    else:
        print('No pSpinningReserveByPlantZone in epm_results')

    if 'pCostSummaryCountry' in epm_results.keys():
        t = epm_results['pCostSummaryCountry'].copy()
        summary.update({'pCostSummaryCountry': t})
    else:
        print('No pCostSummaryCountry in epm_results')

    if 'pCapacityByFuel' in epm_results.keys():
        t = epm_results['pCapacityByFuel'].copy()
        t['attribute'] = 'Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'Capacity: MW': t})
    else:
        print('No pCapacityByFuel in epm_results')

    if 'pNewCapacityFuel' in epm_results.keys():
        t = epm_results['pNewCapacityFuel'].copy()
        t['attribute'] = 'New Capacity: MW'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'NewCapacity: MW': t})
    else:
        print('No pNewCapacityFuel in epm_results')

    if 'pEnergyByFuel' in epm_results.keys():
        t = epm_results['pEnergyByFuel'].copy()
        t['attribute'] = 'Energy: GWh'
        t.rename(columns={'fuel': 'resolution'}, inplace=True)
        t = t[t['value'] > 1e-2]
        summary.update({'Energy: GWh': t})
    else:
        print('No pEnergyByFuel in epm_results')

    summary = pd.concat(summary)

    # Define the order that will appear in the summary.csv file
    order = ['NPV of system cost: $m',
             'Total Demand: GWh', 'Total Generation: GWh', 'Total USE: GWh',
             'Total Capacity Added: MW', 'Total Investment: $m',
             'Total Emission: mt', 'Sys Plan Reserve violation: $m',
             'Zonal Plan Reserve violation: $m', 'Climate backstop cost: $m',
             'Average Total Annual Cost: $m',
             'Average Capex: $m', 'Average Annualized capex: $m', 'Average Fixed O&M: $m',
             'Average Variable O&M: $m',
             'Average Spinning Reserve costs: $m', 'Average Spinning Reserve violation: $m',
             'Average Planning Reserve violation: $m', 'Average Excess generation: $m',
             'Average Unmet demand costs: $m', 'Zonal Spin Reserve violation: $m',
             'Average CO2 backstop cost by Country: $m',
              'Demand: GWh', 'Generation: GWh', 'Unmet demand: GWh',
             'Surplus generation: GWh',
             'Peak demand: MW', 'Firm Capacity: MW', 'Planning Reserve: MW',
             'Spinning Reserve: GWh',
             'Average Cost: $/MWh',
             'Capex: $m', 'Total Annual Cost by Zone: $m', 'Annualized capex: $m', 'Fixed O&M: $m',
             'Variable O&M: $m', 'Total fuel Costs: $m', 'Unmet demand costs: $m',
             'Excess generation: $m', 'VRE curtailment: $m', 'Country Spinning Reserve violation: $m'
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


def generate_summary_detailed(epm_results, folder):
    summary_detailed = {}
    if 'pCapacityPlan' in epm_results.keys():
        temp = epm_results['pCapacityPlan'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Capacity: MW': temp.copy()})
    else:
        print('No pCapacityPlan in epm_results')

    if 'pPlantUtilization' in epm_results.keys():
        temp = epm_results['pPlantUtilization'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Utilization: percent': temp.copy()})
    else:
        print('No pPlantUtilization in epm_results')

    if 'pEnergyByPlant' in epm_results.keys():
        temp = epm_results['pEnergyByPlant'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Energy: GWh': temp.copy()})
    else:
        print('No pEnergyByPlant in epm_results')

    if 'pSpinningReserveByPlantZone' in epm_results.keys():
        temp = epm_results['pSpinningReserveByPlantZone'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year']).squeeze().unstack('scenario')
        temp.reset_index(inplace=True)
        summary_detailed.update({'Spinning Reserve: GWh': temp.copy()})
    else:
        print('No pSpinningReserveByPlantZone in epm_results')

    if 'pCostsbyPlant' in epm_results.keys():
        temp = epm_results['pCostsbyPlant'].copy()
        temp = temp.set_index(['scenario', 'zone', 'generator', 'fuel', 'year', 'attribute']).squeeze().unstack(
            'scenario')
        temp.reset_index(inplace=True)

        grouped_dfs = {key: group.drop(columns=['attribute']) for key, group in temp.groupby('attribute')}
        summary_detailed.update(grouped_dfs)
    else:
        print('No pCostsByPlant in epm_results')

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
    summary_detailed.to_csv(os.path.join(folder, 'summary_detailed.csv'), index=True)


def postprocess_output(FOLDER, reduced_output=False, plot_all=False, folder='', selected_scenario='baseline'):

    # Process results
    RESULTS_FOLDER, GRAPHS_FOLDER, dict_specs, epm_input, epm_results, mapping_gen_fuel = process_simulation_results(
        FOLDER, SCENARIOS_RENAME=None, folder=folder)

    # TODO: Make smth to only select some scenarios that should appear in the Figures

    # Generate summary
    generate_summary(epm_results, RESULTS_FOLDER, epm_input)

    # Generate detailed by plant to debug
    if not reduced_output:

        # Define selected scenario
        if selected_scenario not in epm_results['pEnergyByPlant']['scenario'].unique():
            print(f'No {selected_scenario} in epm_results')
            selected_scenario = epm_results['pEnergyByPlant']['scenario'].unique()[0]
            print(f'Selected scenario is set to: {selected_scenario}')

        # Generate a detailed summary by Power Plant
        generate_summary_detailed(epm_results, RESULTS_FOLDER)

        # Make New Capacity Installed Timeline Figures
        df = epm_results['pCapacityPlan'].copy()

        if len(df.zone.unique()) == 1:
            filename = f'{GRAPHS_FOLDER}/NewCapacityInstalledTimeline-{selected_scenario}.png'
            df = df[df['scenario'] == selected_scenario]
            make_annotated_stacked_area_plot(df, filename, dict_colors=dict_specs['colors'])
        else:
            for zone in df.zone.unique():
                filename = f'{GRAPHS_FOLDER}/NewCapacityInstalledTimeline-{selected_scenario}-{zone}.png'
                df_zone = df.copy()
                df_zone = df_zone[(df_zone['scenario'] == selected_scenario) & (df_zone['zone'] == zone)]
                make_annotated_stacked_area_plot(df_zone, filename, dict_colors=dict_specs['colors'])

        if len(df.zone.unique()) > 1:  # multiple zones
            df = epm_results['pCapacityByFuel'].copy()

            filename = f'{GRAPHS_FOLDER}/CapacityEvolutionPerZone-{selected_scenario}.png'

            make_stacked_bar_subplots(df, filename, dict_specs['colors'], selected_zone=None, selected_year=None,
                                      column_xaxis='zone',
                                      column_stacked='fuel', column_multiple_bars='year',
                                      column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None,
                                      dict_scenarios=None,
                                      format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2,
                                      annotate=False,
                                      show_total=False, fonttick=12, rotation=45, title=None)

        # Make EnergyPlant Figures
        df = epm_results['pEnergyByPlant'].copy()
        if len(df.zone.unique()) == 1:  # single zone model
            if len(epm_results['pEnergyByPlant']['generator'].unique()) < 20:
                df = df[df['scenario'] == selected_scenario]
                filename = f'{GRAPHS_FOLDER}/EnergyPlantsStackedAreaPlot-{selected_scenario}.png'
                stacked_area_plot(df, filename, dict_specs['colors'], x_column='year',
                                  y_column='value',
                                  stack_column='generator', title='Energy Generation by Plant',
                                  y_label='Generation (GWh)',
                                  legend_title='Energy sources', figsize=(10, 6), selected_scenario=selected_scenario,
                                  sorting_column='fuel')
        else:
            for zone in df.zone.unique():
                filename = f'{GRAPHS_FOLDER}/EnergyPlantsStackedAreaPlot-{selected_scenario}-{zone}.png'
                df_zone = df.copy()
                df_zone = df_zone[(df_zone['scenario'] == selected_scenario) & (df_zone['zone'] == zone)]
                if len(df_zone['generator'].unique()) < 20:
                    stacked_area_plot(df_zone, filename, dict_specs['colors'], x_column='year',
                                      y_column='value',
                                      stack_column='generator', title='Energy Generation by Plant',
                                      y_label='Generation (GWh)',
                                      legend_title='Energy sources', figsize=(10, 6),
                                      selected_scenario=selected_scenario,
                                      sorting_column='fuel')


                # make_annotated_stacked_area_plot(df, filename, dict_colors=dict_specs['colors'])

        if len(epm_results['pEnergyByPlant']['generator'].unique()) < 20:
            filename = f'{GRAPHS_FOLDER}/EnergyPlantsStackedAreaPlot_baseline-{selected_scenario}.png'
            stacked_area_plot(epm_results['pEnergyByPlant'], filename, dict_specs['colors'], x_column='year',
                              y_column='value',
                              stack_column='generator', title='Energy Generation by Plant', y_label='Generation (GWh)',
                              legend_title='Energy sources', figsize=(10, 6), selected_scenario=selected_scenario,
                              sorting_column='fuel')

        # Scenario comparison
        if len(epm_results['pEnergyByPlant']['scenario'].unique()) < 8:

            df = epm_results['pCapacityByFuel'].copy()
            df['value'] = df['value'] / 1e3
            filename = f'{GRAPHS_FOLDER}/CapacityMixClusteredStackedAreaPlot.png'
            make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel', column_xaxis='year',
                                      column_value='value', column_multiple_bars='scenario',
                                      select_xaxis=[df['year'].min(), df['year'].max()],
                                      format_y=lambda y, _: '{:.0f} GW'.format(y), rotation=45)

            df = epm_results['pEnergyByFuel'].copy()
            df['value'] = df['value'] / 1e3
            filename = f'{GRAPHS_FOLDER}/EnergyMixClusteredStackedAreaPlot.png'
            make_stacked_bar_subplots(df, filename, dict_specs['colors'], column_stacked='fuel', column_xaxis='year',
                                      column_value='value', column_multiple_bars='scenario',
                                      select_xaxis=[df['year'].min(), df['year'].max()],
                                      format_y=lambda y, _: '{:.0f} TWh'.format(y), rotation=45)

        if 'pAnnualTransmissionCapacity' in epm_results.keys():
            if len(epm_results['pAnnualTransmissionCapacity'].zone.unique()) > 0:  # we have multiple zones
                make_automatic_map(epm_results, dict_specs, GRAPHS_FOLDER, plot_all)

        # Perform automatic Energy DispatchFigures
        make_automatic_dispatch(epm_results, dict_specs, GRAPHS_FOLDER, plot_all=plot_all,
                                selected_scenario=selected_scenario)




def make_automatic_map(epm_results, dict_specs, GRAPHS_FOLDER, plot_all):
    # TODO: ongoing work
    if not plot_all:  # we only plot the baseline scenario
        selected_scenarios = ['baseline']
    else:  # we plot all scenarios
        selected_scenarios = list(epm_results['pPlantDispatch'].scenario.unique())

    pAnnualTransmissionCapacity = epm_results['pAnnualTransmissionCapacity'].copy()
    pInterconUtilization = epm_results['pInterconUtilization'].copy()
    pInterconUtilization['value'] = pInterconUtilization['value'] * 100  # percentage
    years = epm_results['pAnnualTransmissionCapacity']['year'].unique()

    for selected_scenario in list(epm_results['pPlantDispatch'].scenario.unique()):
        print(f'Automatic map for scenario {selected_scenario}')
        folder = f'{GRAPHS_FOLDER}/{selected_scenario}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        # Select first and last years
        years = [min(years), max(years)]

        try:
            zone_map, geojson_to_epm = get_json_data(epm_results, dict_specs)

            zone_map, centers = create_zonemap(zone_map, map_geojson_to_epm=geojson_to_epm)

        except Exception as e:
            print(
                'Error when creating zone geojson for automated map graphs. This may be caused by a problem when specifying a mapping between EPM zone names, and GEOJSON zone names.\n Edit the `geojson_to_epm.csv` file in the `static` folder.')
            raise  # Re-raise the exception for debuggings

        capa_transmission = epm_results['pAnnualTransmissionCapacity'].copy()
        utilization_transmission = epm_results['pInterconUtilization'].copy()
        utilization_transmission['value'] = utilization_transmission['value'] * 100  # update to percentage value
        transmission_data = capa_transmission.rename(columns={'value': 'capacity'}).merge(
            utilization_transmission.rename(columns={'value': 'utilization'}),
            on=['scenario', 'zone', 'z2', 'year'], how='outer')  # removes connections with zero utilization
        transmission_data = transmission_data.rename(columns={'zone': 'zone_from', 'z2': 'zone_to'})

        for year in years:
            filename = f'{GRAPHS_FOLDER}/{selected_scenario}/TransmissionCapacity_{selected_scenario}_{year}.png'

            if not transmission_data.loc[transmission_data.scenario == selected_scenario].empty:  # only plotting transmission when there is information to plot
                make_interconnection_map(zone_map, transmission_data, centers, year=year, scenario=selected_scenario, column='capacity',
                                         label_yoffset=0.01, label_xoffset=-0.05, label_fontsize=10, show_labels=True,
                                         min_display_capacity=200, filename=filename, title='Transmission capacity (MW)')

                filename = f'{GRAPHS_FOLDER}/{selected_scenario}/TransmissionUtilization_{selected_scenario}_{year}.png'

                make_interconnection_map(zone_map, transmission_data, centers, year=year, scenario=selected_scenario, column='utilization',
                                         min_capacity=0.01, label_yoffset=0.01, label_xoffset=-0.05,
                                         label_fontsize=10, show_labels=False, min_display_capacity=50,
                                         format_y=lambda y, _: '{:.0f} %'.format(y), filename=filename, title='Transmission utilization (%)')

            if len(epm_results['pDemandSupply'].loc[(epm_results['pDemandSupply'].scenario == selected_scenario)].zone.unique()) > 1:  # only plotting on interactive map when more than one zone
                    energy_data = epm_results['pDemandSupply'].copy()
                    pCapacityByFuel = epm_results['pCapacityByFuel'].copy()
                    pEnergyByFuel = epm_results['pEnergyByFuel'].copy()
                    pDispatch = epm_results['pDispatch'].copy()
                    pPlantDispatch = epm_results['pPlantDispatch'].copy()
                    pPrice = epm_results['pPrice'].copy()
                    filename = f'{GRAPHS_FOLDER}/{selected_scenario}/InteractiveMap_{selected_scenario}_{year}.html'

                    create_interactive_map(zone_map, centers, transmission_data, energy_data, year, selected_scenario, filename,
                                           dict_specs, pCapacityByFuel, pEnergyByFuel, pDispatch, pPlantDispatch, pPrice)


def make_automatic_dispatch(epm_results, dict_specs, GRAPHS_FOLDER, plot_all=False, selected_scenario='baseline'):

    if not plot_all:  # we only plot the baseline scenario
        selected_scenarios = [selected_scenario]
    else:  # we plot all scenarios
        selected_scenarios = list(epm_results['pPlantDispatch'].scenario.unique())

    dfs_to_plot_area = {
        'pPlantDispatch': filter_dataframe(epm_results['pPlantDispatch'], {'attribute': ['Generation']}),
        'pDispatch': filter_dataframe(epm_results['pDispatch'], {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']})
    }

    dfs_to_plot_line = {
        'pDispatch': filter_dataframe(epm_results['pDispatch'], {'attribute': ['Demand']})
    }

    for selected_scenario in selected_scenarios:
        folder = f'{GRAPHS_FOLDER}/{selected_scenario}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        for zone in epm_results['pDispatch'].loc[epm_results['pDispatch'].scenario == selected_scenario]['zone'].unique():
            years = epm_results['pDispatch']['year'].unique()

            # Select first and last years
            years = [min(years), max(years)]
            for year in years:
                filename = f'{GRAPHS_FOLDER}/{selected_scenario}/Dispatch_{selected_scenario}_{zone}_{year}.png'

                # Select season min and max
                conditions = {'scenario': 'baseline', 'zone': zone, 'year': year, 'attribute': 'Demand'}
                temp = epm_results['pDispatch'].copy()
                temp = filter_dataframe(temp, conditions)
                t = temp.groupby('season', observed=False)['value'].sum()
                s_max, s_min = t.idxmax(), t.idxmin()
                temp = filter_dataframe(temp, {'season': [s_min, s_max]})

                # Select the day with max demand
                d = temp.groupby(['day'], observed=False)['value'].sum().idxmax()

                select_time = {'season': [s_min, s_max], 'day': [d]}
                make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'],
                                                 zone=zone, year=year, scenario=selected_scenario,
                                                 fuel_grouping=None, select_time=select_time, filename=filename,
                                                 bottom=None, legend_loc='bottom')
                select_time = {'season': [s_max]}
                make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'],
                                                 zone=zone, year=year, scenario=selected_scenario,
                                                 fuel_grouping=None, select_time=select_time, filename=filename,
                                                 bottom=None, legend_loc='bottom')



def format_ax(ax, linewidth=True):
    """
    Format the axis of a plot.
    

    Parameters:
    ----------
    ax: plt.Axes
        Axis to format
    linewidth: bool, optional, default=True
        If True, set the linewidth of the spines to 1
    """
    # Remove the background
    ax.set_facecolor('none')

    # Remove grid lines
    ax.grid(False)

    # Optionally, make spines more prominent (if needed)
    if linewidth:
        for spine in ax.spines.values():
            spine.set_color('black')  # Ensure spines are visible
            spine.set_linewidth(1)  # Adjust spine thickness

    # Remove ticks if necessary (optional, can comment out)
    ax.tick_params(top=False, right=False, left=True, bottom=True, direction='in', width=0.8)

    # Ensure the entire frame is displayed
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


def calculate_total_system_cost(data, discount_rate, start_year=None, end_year=None):
    """
    Calculate the total system cost with interpolation and discounting.

    Parameters:
    ----------
    data: pd.DataFrame
        DataFrame with columns ['Scenario', 'Year', 'Cost']
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


def line_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(10, 6)):
    """Makes a line plot.

    Parameters:
    ----------
    df: pd.DataFrame
    x: str
        Column name for x-axis
    y: str
        Column name for y-axis
    xlabel: str, optional
        Label for x-axis
    ylabel: str, optional
        Label for y-axis
    title: str, optional
        Title of the plot
    filename: str, optional
        Path to save the plot
    figsize: tuple, optional, default=(10, 6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[x], df[y])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if filename is not None:
        plt.savefig(filename)
        plt.close()

    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return None


def bar_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(8, 5)):
    """Makes a bar plot.

    Parameters:
    ----------
    df: pd.DataFrame
    x: str
        Column name for x-axis
    y: str
        Column name for y-axis
    xlabel: str, optional
        Label for x-axis
    ylabel: str, optional
        Label for y-axis
    title: str, optional
        Title of the plot
    filename: str, optional
        Path to save the plot
    figsize: tuple, optional, default=(10, 6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[x], df[y], width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}', va='bottom', ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return None


def make_demand_plot(pDemandSupplyCountry, folder, years=None, plot_option='bar', selected_scenario=None, unit='MWh'):
    """
    Depreciated. Makes a plot of demand for all countries.
    
    Parameters:
    ----------
    pDemandSupplyCountry: pd.DataFrame
        Contains demand data for all countries
    folder: str
        Path to folder where the plot will be saved
    years: list, optional
        List of years to include in the plot
    plot_option: str, optional, default='bar'
        Type of plot. Choose between 'line' and 'bar'
    selected_scenario: str, optional
        Name of the scenario
    unit: str, optional, default='GWh'
        Unit of the demand. Choose between 'GWh' and 'TWh'
    """
    # TODO: add scenario grouping, currently only works when selected_scenario is not None

    df_tot = pDemandSupplyCountry.loc[pDemandSupplyCountry['attribute'] == 'Demand: GWh']
    if selected_scenario is not None:
        df_tot = df_tot[df_tot['scenario'] == selected_scenario]
    df_tot = df_tot.groupby(['year']).agg({'value': 'sum'}).reset_index()

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000
    elif unit == '000 TWh':
        df_tot['value'] = df_tot['value'] / 1000000

    if years is not None:
        df_tot = df_tot.loc[df_tot['year'].isin(years)]

    if plot_option == 'line':
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Demand {unit}',
                       title=f'Total demand - {selected_scenario} scenario',
                       filename=f'{folder}/TotalDemand_{plot_option}_{selected_scenario}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Demand {unit}',
                      title=f'Total demand - {selected_scenario} scenario',
                      filename=f'{folder}/TotalDemand_{plot_option}_{selected_scenario}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def make_generation_plot(pEnergyByFuel, folder, years=None, plot_option='bar', selected_scenario=None, unit='GWh',
                         BESS_included=True, Hydro_stor_included=True):
    """
    Makes a plot of demand for all countries.

    Parameters:
    ----------
    pDemandSupplyCountry: pd.DataFrame
        Contains demand data for all countries
    folder: str
        Path to folder where the plot will be saved
    years: list, optional
        List of years to include in the plot
    plot_option: str, optional, default='bar'
        Type of plot. Choose between 'line' and 'bar'
    selected_scenario: str, optional
        Name of the scenario
    unit: str, optional, default='GWh'
        Unit of the demand. Choose between 'GWh' and 'TWh'
    """
    # TODO: add scenario grouping, currently only works when selected_scenario is not None
    if selected_scenario is not None:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['scenario'] == selected_scenario]

    if not BESS_included:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['fuel'] != 'Battery Storage']

    if not Hydro_stor_included:
        pEnergyByFuel = pEnergyByFuel[pEnergyByFuel['fuel'] != 'Pumped-Hydro Storage']

    df_tot = pEnergyByFuel.groupby('year').agg({'value': 'sum'}).reset_index()

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000
    elif unit == '000 TWh':
        df_tot['value'] = df_tot['value'] / 1000000

    if years is not None:
        df_tot = df_tot.loc[df_tot['year'].isin(years)]

    if plot_option == 'line':
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Generation {unit}',
                       title=f'Total generation - {selected_scenario} scenario',
                       filename=f'{folder}/TotalGeneration_{plot_option}_{selected_scenario}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Generation {unit}',
                      title=f'Total generation - {selected_scenario} scenario',
                      filename=f'{folder}/TotalGeneration_{plot_option}_{selected_scenario}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def subplot_pie(df, index, dict_colors, subplot_column=None, title='', figsize=(16, 4), ax=None,
                percent_cap=1, filename=None, rename=None, bbox_to_anchor=(0.5, -0.1), loc='lower center',
                legend_fontsize=16, legend_ncol=1, legend=True):
    """
    Creates pie charts for data grouped by a column, or a single pie chart if no grouping is specified.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    index: str
        Column to use for the pie chart
    dict_colors: dict
        Dictionary mapping the index values to colors
    subplot_column: str, optional
        Column to use for subplots. If None, a single pie chart is created.
    title: str, optional
        Title of the plot
    figsize: tuple, optional, default=(16, 4)
        Size of the figure
    percent_cap: float, optional, default=1
        Minimum percentage to show in the pie chart
    filename: str, optional
        Path to save the plot
    bbox_to_anchor: tuple
        Position of the legend compared to the figure
    loc: str
        Localization of the legend
    """
    if rename is not None:
        df[index] = df[index].replace(rename)
    if subplot_column is not None:
        # Group by the column for subplots
        groups = df.groupby(subplot_column)

        # Calculate the number of subplots
        num_subplots = len(groups)
        ncols = min(3, num_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(num_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1]*nrows))
        axes = np.array(axes).flatten()  # Ensure axes is iterable 1D array


        all_labels = set()  # Collect all labels for the combined legend
        for ax, (name, group) in zip(axes, groups):
            colors = [dict_colors[f] for f in group[index]]
            handles, labels = plot_pie_on_ax(ax, group, index, percent_cap, colors, title=f"{title} - {subplot_column}: {name}")
            all_labels.update(group[index])  # Collect unique labels

        # Hide unused subplots
        for j in range(len(groups), len(axes)):
            fig.delaxes(axes[j])

        if legend:
            # Create a shared legend below the graphs
            all_labels = sorted(all_labels)  # Sort labels for consistency
            handles = [plt.Line2D([0], [0], marker='o', color=dict_colors[label], linestyle='', markersize=10)
                       for label in all_labels]
            fig.legend(
                handles,
                all_labels,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                ncol=legend_ncol,  # Adjust number of columns based on subplots
                frameon=False, fontsize=legend_fontsize
            )

        # Add title for the whole figure
        fig.suptitle(title, fontsize=16)

    else:  # Create a single pie chart if no subplot column is specified
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        colors = [dict_colors[f] for f in df[index]]
        handles, labels = plot_pie_on_ax(ax, df, index, percent_cap, colors, title)

    # Save the figure if filename is provided
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pie_on_ax(ax, df, index, percent_cap, colors, title, radius=None, annotation_size=8):
    """Pie plot on a single axis."""
    if radius is not None:
        df.plot.pie(
            ax=ax,
            y='value',
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None,
            radius=radius
        )
    else:
        df.plot.pie(
            ax=ax,
            y='value',
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None
        )
    ax.set_ylabel('')
    ax.set_title(title)

    # Adjust annotation font sizes
    for text in ax.texts:
        if text.get_text().endswith('%'):  # Check if the text is a percentage annotation
            text.set_fontsize(annotation_size)

    # Generate legend handles and labels manually
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, df[index])]
    labels = list(df[index])
    return handles, labels


def stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value', stack_column='fuel',
                      df_2=None, title='', x_label='Years', y_label='',
                      legend_title='', y2_label='', figsize=(10, 6), selected_scenario=None,
                      annotate=None, sorting_column=None):
    """
    Generate a stacked area chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    filename : str
        Path to save the plot.
    dict_colors : dict, optional
        Dictionary mapping fuel types to colors.
    x_column : str, default 'year'
        Column for x-axis.
    y_column : str, default 'value'
        Column for y-axis.
    stack_column : str, default 'fuel'
        Column for stacking.
    legend_title : str
        Title for the legend.
    df_2 : pd.DataFrame, optional
        DataFrame containing data for the secondary y-axis.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the primary y-axis.
    y2_label : str
        Label for the secondary y-axis.
    figsize : tuple, default (10, 6)
        Size of the figure.
    selected_scenario : str, optional
        Name of the scenario.
    annotate : dict, optional
        Dictionary containing the annotations.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    if selected_scenario is not None:
        df = df[df['scenario'] == selected_scenario]

    if sorting_column is not None:
        # Create a mapping to control order
        sorting_column = df.groupby(stack_column)[sorting_column].first().sort_values().index

    # Plot stacked area for generation
    temp = df.groupby([x_column, stack_column])[y_column].sum().unstack(stack_column)

    if sorting_column is not None:
        temp = temp[sorting_column]

    temp.plot.area(ax=ax1, stacked=True, alpha=0.8, color=dict_colors)

    if annotate is not None:
        for key, value in annotate.items():
            x = key - 2
            y = temp.loc[key].sum() / 2
            ax1.annotate(value, xy=(x, y), xytext=(x, y * 1.2))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    format_ax(ax1)
    
    years = temp.index  # Assuming the x-axis data corresponds to years
    ticks = [year for year in years if year % 5 == 0]
    ax1.xaxis.set_major_locator(FixedLocator(ticks))

    # Secondary y-axis
    if df_2 is not None:
        # Remove legend ax1
        ax1.get_legend().remove()

        temp = df_2.groupby([x_column])[y_column].sum()
        ax2 = ax1.twinx()
        line, = ax2.plot(temp.index, temp, color='brown', label=y2_label)
        ax2.set_ylabel(y2_label, color='brown')
        format_ax(ax2, linewidth=False)

        # Combine legends for ax1 and ax2
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()  # Collect from ax1
        handles_ax2, labels_ax2 = [line], [y2_label]  # Collect from ax2
        handles = handles_ax1 + handles_ax2  # Combine handles
        labels = labels_ax1 + labels_ax2  # Combine labels
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.1, 0.5),  # Right side, centered vertically
            frameon=False,
        )
    else:
        ax1.legend(
            loc='center left',
            bbox_to_anchor=(1.1, 0.5),  # Right side, centered vertically
            title=legend_title,
            frameon=False,
        )
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def make_stacked_area_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, selected_scenario=None, column_xaxis='year',
                              column_stacked='fuel', column_subplots='scenario', format_y=lambda y, _: '{:.0f} MW'.format(y),
                              column_value='value', select_xaxis=None, rotation=0):
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if selected_scenario is not None:
        df = df[(df['scenario'] == selected_scenario)]
        df = df.drop(columns=['scenario'])

    if column_subplots is not None:
        df = (df.groupby([column_xaxis, column_stacked, column_subplots], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_subplots, column_xaxis]).squeeze().unstack(column_subplots)
    else:  # no subplots in this case
        df = (df.groupby([column_stacked, column_xaxis], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_xaxis])

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    stacked_area_subplots(df, column_stacked, filename, dict_colors, format_y=format_y,
                        rotation=rotation)


def stacked_area_subplots(df, column_group, filename, dict_colors=None, order_scenarios=None, order_columns=None,
                        dict_scenarios=None, rotation=0, fonttick=14, legend=True, format_y=lambda y, _: '{:.0f} GW'.format(y),
                        title=None, figsize=(10,6)):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_scenarios : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """

    list_keys = list(df.columns)
    num_subplots = len(list_keys)
    n_columns = min(3, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1] * n_rows), sharey='all')
    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array

    handles, labels = None, None
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_group)

            if order_columns is not None:
                df_temp = df_temp[order_columns]

            df_temp.plot.area(ax=ax, stacked=True, alpha=0.8, color=dict_colors if dict_colors else None)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')

            if len(list_keys) > 1:
                title = key
                if isinstance(key, tuple):
                    title = '{}-{}'.format(key[0], key[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
            else:
                if title is not None:
                    if isinstance(title, tuple):
                        title = '{}-{}'.format(title[0], title[1])
                    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k % n_columns != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.get_legend().remove()

            # Grid settings
            ax.axhline(0, color='black', linewidth=0.5)

        except IndexError:
            ax.axis('off')

    if legend:
        fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1, bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def make_annotated_stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value',
                                     stack_column='fuel', annotate_column='generator'):
    df.sort_values(stack_column, inplace=True)
    # complete year with 0 capacity when no data
    years = df[x_column].unique()

    result = {}
    for n, g in df.groupby([annotate_column]):
        g.set_index(x_column, inplace=True)
        g = g.loc[:, y_column]
        g = g.reindex(years, fill_value=0)
        g.sort_index(inplace=True)
        g = g.diff()
        g = g[g > 1].to_dict()
        g = {k: '{} - {:.0f}'.format(n[0], i) for k, i in g.items()}
        # if k in result.keys() add values to the existing dictionary
        for k, i in g.items():
            if k in result.keys():
                result[k] += '\n' + i
            else:
                result[k] = i

    stacked_area_plot(df, filename, dict_colors, x_column='year', y_column='value', stack_column='fuel',
                      annotate=result)


def format_dispatch_ax(ax, pd_index):

    # Adding the representative days and seasons
    n_rep_days = len(pd_index.get_level_values('day').unique())
    dispatch_seasons = pd_index.get_level_values('season').unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

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
    ax.set_xlabel('')
    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)


def dispatch_plot(df_area=None, filename=None, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom',
                  bottom=0, ylabel=None, title=None):
    """
    Generate and display or save a dispatch plot with area and line plots.
    
    
    Parameters
    ----------
    df_area : pandas.DataFrame, optional
        DataFrame containing data for the area plot. If provided, the area plot will be stacked.
    filename : str, optional
        Path to save the plot image. If not provided, the plot will be displayed.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the plot.
    df_line : pandas.DataFrame, optional
        DataFrame containing data for the line plot. If provided, the line plot will be overlaid on the area plot.
    figsize : tuple, default (10, 6)
        Size of the figure in inches.
    legend_loc : str, default 'bottom'
        Location of the legend. Options are 'bottom' or 'right'.
    ymin : int or float, default 0
        Minimum value for the y-axis.
    Raises
    ------
    ValueError
        If neither `df_area` nor `df_line` is provided.
    Notes
    -----
    The function will raise an assertion error if `df_area` and `df_line` are provided but do not share the same index.
    
    Examples
    --------
    >>> dispatch_plot(df_area=df_area, df_line=df_line, dict_colors=color_dict, filename='dispatch_plot.png')
    """    

    fig, ax = plt.subplots(figsize=figsize)

    if df_area is not None:
        df_area.plot.area(ax=ax, stacked=True, color=dict_colors, linewidth=0)
        pd_index = df_area.index
    if df_line is not None:
        if df_area is not None:
            assert df_area.index.equals(
                df_line.index), 'Dataframes used for area and line do not share the same index. Update the input dataframes.'
        df_line.plot(ax=ax, color=dict_colors)
        pd_index = df_line.index

    if (df_area is None) and (df_line is None):
        raise ValueError('No dataframes provided for the plot. Please provide at least one dataframe.')

    format_dispatch_ax(ax, pd_index)

    # Add axis labels and title
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold')
    else:
        ax.set_ylabel('Generation (MW)', fontsize=8.5)
    # ax.text(0, 1.2, f'Dispatch', fontsize=9, fontweight='bold', transform=ax.transAxes)
    # set ymin to 0
    if bottom is not None:
        ax.set_ylim(bottom=bottom)

    # Add legend bottom center
    if legend_loc == 'bottom':
        if df_area is not None:
            # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df_area.columns), frameon=False)
            fig.subplots_adjust(bottom=0.25)  # Adds space for the legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_area.columns), frameon=False)

        else:
            # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df_line.columns), frameon=False)
            fig.subplots_adjust(bottom=0.25)  # Adds space for the legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_line.columns), frameon=False)

    # TODO: needs to be fixed (dispatch plot was updated to work with interactive map, so that the legend is now inside the plot. Not working anymore when legend on the right)
    elif legend_loc == 'right':
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=False)
    
    if title is not None:
        ax.text(
        y=ax.get_ylim()[1] * 1.2,
        x = sum(ax.get_xlim()) / 2,
        s=title,
        ha='center',
        fontsize=8.5
        )

    if filename is not None:
        # fig.savefig(filename, bbox_inches='tight')
        fig.savefig(filename, bbox_inches=None, pad_inches=0.1, dpi=100)

        plt.close()
    else:
        plt.show()


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
    temp = ''
    if 'season' in select_time.keys():
        df = df.loc[df.season.isin(select_time['season'])]
        temp += '_'.join(select_time['season'])
    if 'day' in select_time.keys():
        df = df.loc[df.day.isin(select_time['day'])]
        temp += '_'.join(select_time['day'])
    return df, temp


def clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=None, select_time=None):
    """
    Transforms a dataframe from the results GDX into a dataframe with season, day, and time as the index, and format ready for plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the results.
    zone : str
        The zone to filter the data for.
    year : int
        The year to filter the data for.
    scenario : str
        The scenario to filter the data for.
    column_stacked : str
        Column to use for stacking values in the transformed dataframe.
    fuel_grouping : dict, optional
        A dictionary mapping fuels to their respective groups and to sum values over those groups.
    select_time : dict or None, optional
        Specific time filter to apply (e.g., "summer").

    Returns
    -------
    pd.DataFrame
        A transformed dataframe with multi-level index (season, day, time).

    Example
    -------
    df = epm_dict['FuelDispatch']
    column_stacked = 'fuel'
    select_time = {'season': ['m1'], 'day': ['d21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30']}
    df = clean_dataframe(df, zone='Liberia', year=2025, scenario='Baseline', column_stacked='fuel', fuel_grouping=None, select_time=select_time)
    """
    if 'zone' in df.columns:
        df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['zone', 'year', 'scenario'])
    else:
        df = df[(df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['year', 'scenario'])

    if column_stacked == 'fuel':
        if fuel_grouping is not None:
            df['fuel'] = df['fuel'].replace(
                fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

    if column_stacked is not None:
        df = (df.groupby(['season', 'day', 't', column_stacked], observed=False)['value'].sum().reset_index())

    if select_time is not None:
        df, temp = select_time_period(df, select_time)
    else:
        temp = None

    if column_stacked is not None:
        df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    else:
        df = df.set_index(['season', 'day', 't'])
    return df, temp


def remove_na_values(df):
    """Removes na values from a dataframe, to avoind unnecessary labels in plots."""
    df = df.where((df > 1e-6) | (df < -1e-6),
                                    np.nan)
    df = df.dropna(axis=1, how='all')
    return df


def make_complete_fuel_dispatch_plot(dfs_area, dfs_line, dict_colors, zone, year, scenario, stacked=True,
                                    filename=None, fuel_grouping=None, select_time=None, reorder_dispatch=None,
                                    legend_loc='bottom', bottom=None, figsize=(10,6), ylabel=None, title=None):
    """
    Generates and saves a fuel dispatch plot, including only generation plants.

    Parameters
    ----------
    dfs_area : dict
        Dictionary containing dataframes for area plots.
    dfs_line : dict
        Dictionary containing dataframes for line plots.
    graph_folder : str
        Path to the folder where the plot will be saved.
    dict_colors : dict
        Dictionary mapping fuel types to colors.
    fuel_grouping : dict
        Mapping to create aggregate fuel categories, e.g.,
        {'Battery Storage 4h': 'Battery Storage'}.
    select_time : dict
        Time selection parameters for filtering the data.
    dfs_line_2 : dict, optional
        Optional dictionary containing dataframes for a secondary line plot.

    Returns
    -------
    None

    Example
    -------
    Generate and save a fuel dispatch plot:
    dfs_to_plot_area = {
        'pFuelDispatch': epm_dict['pFuelDispatch'],
        'pCurtailedVRET': epm_dict['pCurtailedVRET'],
        'pDispatch': subset_dispatch
    }
    subset_demand = epm_dict['pDispatch'].loc[epm_dict['pDispatch'].attribute.isin(['Demand'])]
    dfs_to_plot_line = {
        'pDispatch': subset_demand
    }
    fuel_grouping = {
        'Battery Storage 4h': 'Battery Discharge',
        'Battery Storage 8h': 'Battery Discharge',
        'Battery Storage 2h': 'Battery Discharge',
        'Battery Storage 3h': 'Battery Discharge',
    }
    make_complete_fuel_dispatch_plot(dfs_to_plot_area, dfs_to_plot_line, folder_results / Path('images'), dict_specs['colors'],
                                 zone='Liberia', year=2030, scenario=scenario, fuel_grouping=fuel_grouping,
                                 select_time=select_time, reorder_dispatch=['MtCoffee', 'Oil', 'Solar'], season=False)
    """
    # TODO: Add ax2 to show other data. For example prices would be interesting to show in the same plot.

    tmp_concat_area = []
    for key in dfs_area:
        df = dfs_area[key]
        if stacked:  # we want to group data by a given column (eg, fuel for dispatch)
            column_stacked = NAME_COLUMNS[key]
        else:
            column_stacked = None
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]
        if stacked:  # we want to group data by a given column (eg, fuel for dispatch)
            column_stacked = NAME_COLUMNS[key]
        else:
            column_stacked = None
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_line.append(df)

    if len(tmp_concat_area) > 0:
        df_tot_area = pd.concat(tmp_concat_area, axis=1)
        df_tot_area = df_tot_area.droplevel(0, axis=1)
        df_tot_area = remove_na_values(df_tot_area)
    else:
        df_tot_area = None

    if len(tmp_concat_line) > 0:
        df_tot_line = pd.concat(tmp_concat_line, axis=1)
        if df_tot_line.columns.nlevels > 1:
            df_tot_line = df_tot_line.droplevel(0, axis=1)
        # df_tot_line = remove_na_values(df_tot_line)
    else:
        df_tot_line = None

    if reorder_dispatch is not None:
        new_order = [col for col in reorder_dispatch if col in df_tot_area.columns] + [col for col in df_tot_area.columns if col not in reorder_dispatch]
        df_tot_area = df_tot_area[new_order]

    if select_time is None:
        temp = 'all'
    temp = f'{year}_{temp}'
    if filename is not None and isinstance(filename, str):  # Only modify filename if it's a string
        filename = filename.split('.png')[0] + f'_{temp}.png'

    dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors, legend_loc=legend_loc, bottom=bottom,
                  figsize=figsize, ylabel=ylabel, title=title)


def stacked_bar_subplot(df, column_group, filename, dict_colors=None, year_ini=None,order_scenarios=None, order_columns=None,
                        dict_scenarios=None, rotation=0, fonttick=14, legend=True, format_y=lambda y, _: '{:.0f} GW'.format(y),
                        cap=6, annotate=True, show_total=False, title=None, figsize=(10,6), fontsize_label=10, format_label="{:.1f}"):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_scenarios : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """
    
    list_keys = list(df.columns)
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_group][0]).unique()
    num_subplots = int(len(list_keys))
    n_columns = min(3, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1]*n_rows), sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})
    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array

    handles, labels = None, None
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_group)

            if key == year_ini:
                df_temp = df_temp.iloc[0, :]
                df_temp = df_temp.to_frame().T
                df_temp.index = ['Initial']
            else:
                if dict_scenarios is not None:  # Renaming scenarios for plots
                    df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
                if order_scenarios is not None:  # Reordering scenarios
                    df_temp = df_temp.loc[[c for c in order_scenarios if c in df_temp.index], :]
                if order_columns is not None:
                    new_order = [c for c in order_columns if c in df_temp.columns] + [c for c in df_temp.columns if c not in order_columns]
                    df_temp = df_temp.loc[:,new_order]

            df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0,
                         color=dict_colors if dict_colors is not None else None)

            # Annotate each bar
            if annotate:
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > cap:  # Only annotate bars with a height
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
                                bar.get_y() + height / 2,  # Y position: middle of the bar
                                format_label.format(height),  # Annotation text (formatted value)
                                ha="center", va="center",  # Center align the text
                                fontsize=fontsize_label, color="black"  # Font size and color
                            )

            if show_total:
                df_total = df_temp.sum(axis=1)
                for x, y in zip(df_temp.index, df_total.values):
                    # Put the total at the y-position equal to the total
                    ax.text(x, y * (1 + 0.02), f"{y:,.0f}", ha='center', va='bottom', fontsize=10,
                            color='black', fontweight='bold')
                ax.scatter(df_temp.index, df_total, color='black', s=20)

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelrotation=rotation) 

            if len(list_keys) > 1:
                title = key
                if isinstance(key, tuple):
                    title = '{}-{}'.format(key[0], key[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
            else:
                if title is not None:
                    if isinstance(title, tuple):
                        title = '{}-{}'.format(title[0], title[1])
                    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k % n_columns != 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.get_legend().remove()


            # Add a horizontal line at 0
            ax.axhline(0, color='black', linewidth=0.5)

        except IndexError:
            ax.axis('off')

        if legend:
            fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
                       bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def make_stacked_bar_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, column_xaxis='year',
                              column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None, dict_scenarios=None,
                              format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None, fontsize_label=10, format_label="{:.1f}", figsize=(10,6)):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results.
    filename : str
        Path to save the figure.
    dict_colors : dict
        Dictionary with color arguments.
    selected_zone : str
        Zone to select.
    column_xaxis : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_multiple_bars : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_xaxis : list, optional
        Select a subset of subplots (e.g., a number of years).
    dict_grouping : dict, optional
        Dictionary for grouping variables and summing over a given group.
    order_scenarios : list, optional
        Order of scenarios for plotting.
    dict_scenarios : dict, optional
        Dictionary for renaming scenarios.
    format_y : function, optional
        Function for formatting y-axis labels.
    order_stacked : list, optional
        Reordering the variables that will be stacked.
    cap : int, optional
        Under this cap, no annotation will be displayed.
    annotate : bool, optional
        Whether to annotate the bars.
    show_total : bool, optional
        Whether to show the total value on top of each bar.

    Example
    -------
    Stacked bar subplots for capacity (by fuel) evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('CapacityEvolution.png')
    fuel_grouping = {
        'Battery Storage 4h': 'Battery',
        'Battery Storage 8h': 'Battery',
        'Hydro RoR': 'Hydro',
        'Hydro Storage': 'Hydro'
    }
    scenario_names = {
        'baseline': 'Baseline',
        'HydroHigh': 'High Hydro',
        'DemandHigh': 'High Demand',
        'LowImport_LowThermal': 'LowImport_LowThermal'
    }
    make_stacked_bar_subplots(epm_dict['pCapacityByFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=fuel_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} MW'.format(y))

    Stacked bar subplots for reserve evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
    make_stacked_bar_subplots(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], selected_zone='Liberia',
                              column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario',
                              select_xaxis=[2025, 2028, 2030], dict_grouping=dict_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} GWh'.format(y),
                              order_stacked=['Hydro', 'Oil'], cap=2)
    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if dict_grouping is not None:
        for key, grouping in dict_grouping.items():
            assert key in df.columns, f'Grouping parameter with key {key} is used but {key} is not in the columns.'
            df[key] = df[key].replace(grouping)  # case-specific, according to level of preciseness for dispatch plot

    if column_xaxis is not None:
        df = (df.groupby([column_xaxis, column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)
    else:  # no subplots in this case
        df = (df.groupby([column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
        df = df.set_index([column_stacked, column_multiple_bars])

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    stacked_bar_subplot(df, column_stacked, filename, dict_colors, format_y=format_y,
                        rotation=rotation, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                        order_columns=order_stacked, cap=cap, annotate=annotate, show_total=show_total, fonttick=fonttick, title=title, fontsize_label=fontsize_label, 
                        format_label=format_label, figsize=figsize)


def scatter_plot_with_colors(df, column_xaxis, column_yaxis, column_color, color_dict, ymax=None, xmax=None, title='',
                             legend=None, filename=None, size_scale=None, annotate_thresh=None):
    """
    Creates a scatter plot with points colored based on the values in a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column_xaxis : str
        Column name for x-axis values.
    column_yaxis : str
        Column name for y-axis values.
    column_color : str
        Column name for categorical values determining color.
    color_dict : dict
        Dictionary mapping values in column_color to specific colors.
    size_proportional : bool, optional
        Whether to size points proportionally to x-axis values.
    size_scale : float, optional
        Scaling factor for point sizes if size_proportional is True.
    ymax : float, optional
        Maximum y-axis value.
    title : str, optional
        Title of the plot.
    legend_title : str, optional
        Title for the legend.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.

    Returns
    -------
    None
        Displays the scatter plot.
    """
    # Ensure all values in value_col have a defined color
    unique_values = df[column_color].unique()
    for val in unique_values:
        if val not in color_dict:
            raise ValueError(f"No color specified for value '{val}' in {column_color}")
    color_dict = {val: color_dict[val] for val in unique_values}

    # Determine sizes of points
    sizes = 50
    if size_scale is not None:
        sizes = df[column_xaxis] * size_scale

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    handles = []  # To store legend handles
    labels = []

    for value, color in color_dict.items():
        subset = df[df[column_color] == value]
        scatter = plt.scatter(
            subset[column_xaxis],
            subset[column_yaxis],
            label=value,
            color=color,
            alpha=0.7,
            s=sizes[subset.index] if size_scale else sizes)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=8))
        labels.append(value)  # Add the label for each unique group

        # Add the name of the 'generator' for the points with a value above the threshold
        if annotate_thresh is not None:
            for i, txt in enumerate(subset['generator']):
                if subset[column_xaxis].iloc[i] > annotate_thresh:
                    plt.annotate(txt, (subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]), color='black')

    if ymax is not None:
        plt.ylim(0, ymax)

    if xmax is not None:
        plt.xlim(0, xmax)

    # Add labels and legend
    plt.xlabel(column_xaxis)
    plt.ylabel(column_yaxis)
    plt.title(title)

    # remove legend
    plt.legend().remove()
    if legend is not None:
        plt.legend(handles=handles, labels=labels, title=legend or column_color, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def subplot_scatter(df, column_xaxis, column_yaxis, column_color, color_dict, figsize=(12,8),
                             ymax=None, xmax=None, title='', legend=None, filename=None,
                             size_scale=None, annotate_thresh=None, subplot_column=None):
    """
    Creates scatter plots with points colored based on the values in a specific column.
    Supports optional subplots based on a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column_xaxis : str
        Column name for x-axis values.
    column_yaxis : str
        Column name for y-axis values.
    column_color : str
        Column name for categorical values determining color.
    color_dict : dict
        Dictionary mapping values in column_color to specific colors.
    ymax : float, optional
        Maximum y-axis value.
    xmax : float, optional
        Maximum x-axis value.
    title : str, optional
        Title of the plot.
    legend : str, optional
        Title for the legend.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.
    size_scale : float, optional
        Scaling factor for point sizes.
    annotate_thresh : float, optional
        Threshold for annotating points with generator names.
    subplot_column : str, optional
        Column name to split the data into subplots.

    Returns
    -------
    None
        Displays the scatter plots.
    """
    # If subplots are required
    if subplot_column is not None:
        unique_values = df[subplot_column].unique()
        n_subplots = len(unique_values)
        ncols = min(3, n_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(n_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharex=True, sharey=True)
        axes = np.array(axes).flatten()  # Ensure axes is an iterable 1D array

        for i, val in enumerate(unique_values):
            ax = axes[i]
            subset_df = df[df[subplot_column] == val]

            scatter_plot_on_ax(ax, subset_df, column_xaxis, column_yaxis, column_color, color_dict,
                               ymax, xmax, title=f"{title} - {subplot_column}: {val}",
                               legend=legend, size_scale=size_scale, annotate_thresh=annotate_thresh)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
    else:
        # If no subplots, plot normally
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                           ymax, xmax, title=title, legend=legend,
                           size_scale=size_scale, annotate_thresh=annotate_thresh)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, color_dict,
                       ymax=None, xmax=None, title='', legend=None,
                       size_scale=None, annotate_thresh=None):
    """
    Helper function to create a scatter plot on a given matplotlib Axes.
    """
    unique_values = df[column_color].unique()
    for val in unique_values:
        if val not in color_dict:
            raise ValueError(f"No color specified for value '{val}' in {column_color}")

    color_dict = {val: color_dict[val] for val in unique_values}

    # Determine sizes of points
    sizes = 50
    if size_scale is not None:
        sizes = df[column_xaxis] * size_scale

    # Plot each category separately
    for value, color in color_dict.items():
        subset = df[df[column_color] == value]
        scatter = ax.scatter(subset[column_xaxis], subset[column_yaxis],
                             label=value, color=color, alpha=0.7,
                             s=sizes[subset.index] if size_scale else sizes)

        # Annotate points above a certain threshold
        if annotate_thresh is not None:
            for i, txt in enumerate(subset['generator']):
                if subset[column_xaxis].iloc[i] > annotate_thresh:
                    x_value, y_value = subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]
                    ax.annotate(
                        txt,
                        (x_value, y_value),  # Point location
                        xytext=(5, 10),  # Offset in points (x, y)
                        textcoords='offset points',  # Use an offset from the data point
                        fontsize=9,
                        color='black',
                        ha='left'
                    )
                    # ax.annotate(txt, (subset[column_xaxis].iloc[i], subset[column_yaxis].iloc[i]), color='black')

    if ymax is not None:
        ax.set_ylim(0, ymax)

    if xmax is not None:
        ax.set_xlim(0, xmax)

    ax.set_xlabel(column_xaxis)
    ax.set_ylabel(column_yaxis)
    ax.set_title(title)

    # Remove legend from each subplot to avoid redundancy
    if legend is not None:
        ax.legend(title=legend, frameon=False)

    ax.grid(True, linestyle='--', alpha=0.5)


def heatmap_plot(data, filename=None, percentage=False, baseline='Baseline'):
    """
    Plots a heatmap showing differences from baseline with color scales defined per column.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with scenarios as rows and metrics as columns.
    filename : str
        Path to save the plot.
    percentage : bool, optional
        Whether to show differences as percentages.
    """

    # Calculate differences from baseline
    baseline_values = data.loc[baseline, :]
    diff_from_baseline = data.subtract(baseline_values, axis=1)

    # Combine differences and baseline values for annotations
    annotations = data.map(lambda x: f"{x:,.0f}")  # Format baseline values
    # Format differences in percentage
    if percentage:
        diff_from_baseline = diff_from_baseline / baseline_values
        diff_annotations = diff_from_baseline.map(lambda x: f" ({x:+,.0%})")
    else:
        diff_annotations = diff_from_baseline.map(lambda x: f" ({x:+,.0f})")
    combined_annotations = annotations + diff_annotations  # Combine both

    # Normalize the color scale by column
    diff_normalized = diff_from_baseline.copy()
    for column in diff_from_baseline.columns:
        col_min = diff_from_baseline[column].min()
        col_max = diff_from_baseline[column].max()
        diff_normalized[column] = (diff_from_baseline[column] - col_min) / (col_max - col_min)

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap
    sns.heatmap(
        diff_normalized,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=combined_annotations,  # Show baseline values and differences
        fmt="",  # Disable default formatting
        linewidths=0.5,
        ax=ax,
        cbar=False  # Remove color bar
    )

    # Customize the axes
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def rename_and_reoder(df, rename_index=None, rename_columns=None, order_index=None, order_columns=None):
    if rename_index is not None:
        df.index = df.index.map(lambda x: rename_index.get(x, x))
    if rename_columns is not None:
        df.columns = df.columns.map(lambda x: rename_columns.get(x, x))
    if order_index is not None:
        df = df.loc[order_index, :]
    if order_columns is not None:
        df = df.loc[:, order_columns]
    return df


def make_heatmap_plot(epm_results, filename, percentage=False, scenario_order=None,
                       discount_rate=0, year=2050, required_keys=None, fuel_capa_list=None,
                       fuel_gen_list=None, summary_metrics_list=None, zone_list=None, rows_index='zone',
                       rename_columns=None):
    """
    Make a heatmap plot for the results of the EPM model.


    Parameters
    ----------
    epm_results: dict
    filename: str
    percentage: bool, optional, default is False
    scenario_order
    discount_rate
    """
    summary = []

    if required_keys is None:
        required_keys = ['pCapacityByFuel', 'pEnergyByFuel', 'pEmissions', 'pDemandSupplyCountry', 'pCostSummary',
                         'pNPVByYear']

    assert all(
        key in epm_results for key in required_keys), "Required keys for the summary are not included in epm_results"

    if fuel_capa_list is None:
        fuel_capa_list = ['Hydro', 'Solar', 'Wind']

    if fuel_gen_list is None:
        fuel_gen_list = ['Hydro', 'Oil']

    if summary_metrics_list is None:
        summary_metrics_list = ['Capex: $m']

    if 'pCapacityByFuel' in required_keys:
        temp = epm_results['pCapacityByFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pCapacityByFuel'], values='value')
        temp = temp.loc[:, fuel_capa_list]
        temp = rename_and_reoder(temp, rename_columns=RENAME_COLUMNS)
        temp.columns = [f'{col} (MW)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEnergyByFuel' in required_keys:
        temp = epm_results['pEnergyByFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.loc[:, fuel_gen_list]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pEnergyByFuel'], values='value')
        temp.columns = [f'{col} (GWh)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEmissions' in required_keys:
        temp = epm_results['pEmissions'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.set_index(['scenario'])['value']
        temp = temp * 1e3
        temp.rename('ktCO2', inplace=True).to_frame()
        summary.append(temp)

    if 'pDemandSupplyCountry' in required_keys:
        temp = epm_results['pDemandSupplyCountry'].copy()
        temp = temp[temp['attribute'] == 'Unmet demand: GWh']
        temp = temp.groupby(['scenario'])['value'].sum()

        t = epm_results['pDemandSupplyCountry'].copy()
        t = t[t['attribute'] == 'Demand: GWh']
        t = t.groupby(['scenario'])['value'].sum()
        temp = (temp / t) * 1e3
        temp.rename('Unmet (‰)', inplace=True).to_frame()
        summary.append(temp)

        temp = epm_results['pDemandSupplyCountry'].copy()
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp[temp['attribute'] == 'Surplus generation: GWh']
        temp = temp.groupby(['scenario'])['value'].sum()

        t = epm_results['pDemandSupplyCountry'].copy()
        if zone_list is not None:
            t = t[t['zone'].isin(zone_list)]
        t = t[t['attribute'] == 'Demand: GWh']
        t = t.groupby(['scenario'])['value'].sum()
        temp = (temp / t) * 1000

        temp.rename('Surplus (‰)', inplace=True).to_frame()
        summary.append(temp)

    if 'pCostSummary' in required_keys:
        temp = epm_results['pCostSummary'].copy()
        temp = temp[temp['attribute'] == 'Total Annual Cost by Zone: $m']
        temp = calculate_total_system_cost(temp, discount_rate)

        t = epm_results['pDemandSupply'].copy()
        if zone_list is not None:
            t = t[t['zone'].isin(zone_list)]
        t = t[t['attribute'] == 'Demand: GWh']
        t = calculate_total_system_cost(t, discount_rate)

        temp = (temp * 1e6) / (t * 1e3)

        if isinstance(temp, (float, int)):
            temp = pd.Series(temp, index=[epm_results['pNPVByYear']['scenario'][0]])
        temp.rename('NPV ($/MWh)', inplace=True).to_frame()
        summary.append(temp)

    summary = pd.concat(summary, axis=1)

    if scenario_order is not None:
        scenario_order = [i for i in scenario_order if i in summary.index] + [i for i in summary.index if
                                                                              i not in scenario_order]
        summary = summary.loc[scenario_order]

    heatmap_plot(summary, filename, percentage=percentage, baseline=summary.index[0])


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


def get_json_data(epm_results, dict_specs, geo_add=None):
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
    geojson_to_epm = dict_specs['geojson_to_epm']
    epm_to_geojson = {v: k for k, v in
                      geojson_to_epm.set_index('Geojson')['EPM'].to_dict().items()}  # Reverse dictionary
    geojson_to_divide = geojson_to_epm.loc[geojson_to_epm.region.notna()]
    geojson_complete = geojson_to_epm.loc[~geojson_to_epm.region.notna()]
    selected_zones_epm = epm_results['pAnnualTransmissionCapacity'].zone.unique()
    selected_zones_to_divide = [e for e in selected_zones_epm if e in geojson_to_divide['EPM'].values]
    selected_countries_geojson = [
        epm_to_geojson[key] for key in selected_zones_epm if
        ((key not in selected_zones_to_divide) and (key in epm_to_geojson))
    ]

    zone_map = dict_specs['map_countries']  # getting json data on all countries
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

    else:
        raise ValueError("Invalid division type. Use 'NS' (North-South) or 'EW' (East-West).")



def plot_zone_map_on_ax(ax, zone_map):
    zone_map.plot(ax=ax, color='white', edgecolor='black')

    # Adjusting the limits to better center the zone_map on the region
    ax.set_xlim(zone_map.bounds.minx.min() - 1, zone_map.bounds.maxx.max() + 1)
    ax.set_ylim(zone_map.bounds.miny.min() - 1, zone_map.bounds.maxy.max() + 1)


def make_capacity_mix_map(zone_map, pCapacityByFuel, dict_colors, centers, year, region, scenario, filename,
                          map_epm_to_geojson, index='fuel', list_reduced_size=None, figsize=(10, 6), bbox_to_anchor=(0.5, -0.1),
                          loc='center left', min_size=0.5, max_size =2.5, pie_sizing=True):
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
        h, l = plot_pie_on_ax(ax_pie, CapacityMix_plot, index, 25, colors, None, radius= pie_size)
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


def make_interconnection_map(zone_map, pAnnualTransmissionCapacity, centers, year, scenario, column='value', filename=None,
                             min_capacity=0.1, figsize=(12, 8), show_labels=True, label_yoffset=0.02, label_xoffset=0.02,
                             label_fontsize=12, predefined_colors=None, min_display_capacity=100,
                             min_line_width=1, max_line_width=5, format_y=lambda y, _: '{:.0f} MW'.format(y), title='Transmission capacity'):
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
        predefined_colors = {country: plt.cm.Pastel1(i % 9) for i, country in enumerate(unique_countries)}

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
    zone_map['color'] = zone_map['ADMIN'].map(predefined_colors)
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

    # Plot interconnections
    for _, row in transmission_data.iterrows():
        zone_from, zone_to, capacity = row['zone_from'], row['zone_to'], row[column]

        if zone_from in centers and zone_to in centers:
            coord_from, coord_to = centers[zone_from], centers[zone_to]
            coor_mid = [(coord_from[0] + coord_to[0]) / 2, (coord_from[1] + coord_to[1]) / 2]

            line_width = scale_line_width(capacity)
            ax.plot([coord_from[0], coord_to[0]], [coord_from[1], coord_to[1]], 'r-', linewidth=line_width)

            if capacity >= min_display_capacity:
                ax.text(coor_mid[0], coor_mid[1], format_y(capacity, None), ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'), fontsize=10)

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
            # TODO: needs to be solved in EPM code, current problem in how new capacity is handled !
            capacity, utilization_1to2, utilization_2to1 = max(row1.fillna(0)['capacity'], row2.fillna(0)['capacity']), row1['utilization'], row2['utilization']

            if zone1 in centers and zone2 in centers:
                coords = [[centers[zone1][1], centers[zone1][0]],  # Lat, Lon
                          [centers[zone2][1], centers[zone2][0]]]  # Lat, Lon
                color = calculate_color_gradient(max(utilization_1to2, utilization_2to1), 0, 100)

                tooltip_text = f"""
                <div style="font-size: {label_size}px;">
                <b>Capacity:</b> {capacity:.2f} GW <br>
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
        coords = [coords[1], coords[0]]  # changing to Lat,Long as required by Folium
        popup_content = f"""
        <b>{zone}</b><br>
        Generation: {get_value(energy_data, zone, year, scenario, 'Total production: GWh'):.1f} GWh<br>
        Demand: {get_value(energy_data, zone, year, scenario, 'Demand: GWh'):.1f} GWh<br>
        Imports: {get_value(energy_data, zone, year, scenario, 'Imports exchange: GWh'):.1f} GWh<br>
        Exports: {get_value(energy_data, zone, year, scenario, 'Exports exchange: GWh'):.1f} GWh
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


def get_value(df, zone, year, scenario, attribute, column_to_select='attribute'):
    """Safely retrieves an energy value for a given zone, year, scenario, and attribute."""
    value = df.loc[
        (df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario) & (df[column_to_select] == attribute),
        'value'
    ]
    return value.values[0] if not value.empty else 0


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

def generate_zone_plots(zone, year, scenario, dict_specs, pCapacityByFuel, pEnergyByFuel, pDispatch, pPlantDispatch, pPrice, scale_factor=0.8):
    """Generate capacity mix and dispatch plots for a given zone and return them as base64 strings."""
    # Generate capacity mix pie chart using existing function
    df1 = pCapacityByFuel.copy()
    df1['attribute'] = 'Capacity'
    df2 = pEnergyByFuel.copy()
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
        'pPlantDispatch': filter_dataframe(pPlantDispatch, {'attribute': ['Generation']}),
        'pDispatch': filter_dataframe(pDispatch, {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']})
    }

    dfs_to_plot_line = {
        'pDispatch': filter_dataframe(pDispatch, {'attribute': ['Demand']})
    }

    seasons = pPlantDispatch.season.unique()
    days = pPlantDispatch.day.unique()

    select_time = {'season': seasons, 'day': days}

    dispatch_plot =  make_dispatch_plot_interactive(dfs_to_plot_area, dfs_to_plot_line, dict_specs['colors'], zone, year, scenario,
                                                        select_time=select_time, stacked=True)

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
        'pNetImports': df_net_imports[['year', 'season', 'day', 't', 'zone', 'scenario', 'Net imports']]
    }

    imports_zero = dfs_to_plot_line['pNetImports']
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

def make_complete_dispatch_plot_for_interactive(pFuelDispatch, pDispatch, dict_colors, zone, year, scenario,
                                filename=None, BESS_included=True, Hydro_stor_included=True,title='Dispatch',
                                select_time=None, legend_loc='bottom', bottom=0, figsize=(20,6), fontsize=12):
    """
    Generates and saves a dispatch plot for fuel-based generation in a given zone, year, and scenario.

    Parameters
    ----------
    pFuelDispatch : DataFrame
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
    dispatch_seasons = list(pFuelDispatch['season'].unique())
    n_rep_days = len(list(pFuelDispatch['day'].unique()))

    # Filtrer les données de production
    pFuelDispatch_zone = pFuelDispatch.loc[
        (pFuelDispatch['zone'] == zone) & (pFuelDispatch['year'] == year) & (pFuelDispatch['scenario'] == scenario)
    ]

    # Exclure les stockages si nécessaire
    if not BESS_included:
        pFuelDispatch_zone = pFuelDispatch_zone[pFuelDispatch_zone['fuel'] != 'Battery Storage']
    if not Hydro_stor_included:
        pFuelDispatch_zone = pFuelDispatch_zone[pFuelDispatch_zone['fuel'] != 'Pumped-Hydro']
    y_max_dispatch = float(pFuelDispatch_zone['value'].max())

    # Mise en forme pour le stacked area plot
    pFuelDispatch_pivot = pFuelDispatch_zone.pivot_table(index=['season', 'day', 't'],
                                                          columns='fuel', values='value', aggfunc='sum')

    # Récupérer la demande
    pDemand_zone = pDispatch.loc[
        (pDispatch['zone'] == zone) & (pDispatch['year'] == year) & (pDispatch['scenario'] == scenario) & (pDispatch['attribute'] == 'Demand')
    ]
    y_max_demand = float(pDemand_zone['value'].max())

    pDemand_pivot = pDemand_zone.pivot_table(index=['season', 'day', 't'], values='value')

    # Extraire les saisons et jours représentatifs
    dispatch_seasons = list(pFuelDispatch['season'].unique())
    n_rep_days = len(list(pFuelDispatch['day'].unique()))

    # Créer le graphique
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

    # Tracer la production en stacked area
    if not pFuelDispatch_pivot.empty:
        pFuelDispatch_pivot.plot.area(ax=ax, stacked=True, linewidth=0, color=[dict_colors.get(fuel, 'gray') for fuel in pFuelDispatch_pivot.columns])

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
        filename=img, figsize=(fig_width, fig_height), subplot_column='attribute', legend_ncol=1, legend_fontsize=8,
        bbox_to_anchor=(0.9, 0.5), legend=False
    )

    img.seek(0)
    encoded_str = base64.b64encode(img.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded_str}" width="300">'


def make_dispatch_plot_interactive(dfs_area, dfs_line, dict_colors, zone, year, scenario, select_time, stacked=True,
                                       ylabel=None, bottom=None):
    """Generates a dispatch plot and returns it as a base64 image string."""
    img = BytesIO()

    fig_width = 16
    fig_height = 5  # Shorter height for better fit

    make_complete_fuel_dispatch_plot(
        dfs_area=dfs_area, dfs_line=dfs_line, dict_colors=dict_colors,
        zone=zone, year=year, scenario=scenario, select_time=select_time, filename=img, figsize=(fig_width,fig_height),
        stacked=stacked, ylabel=ylabel, bottom=bottom
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


# def calculate_color_gradient(value, min_val, max_val, cmap_name='coolwarm'):
#     norm_val = (value - min_val) / (max_val - min_val)
#     norm_val = min(max(norm_val, 0), 1)
#     cmap = plt.get_cmap(cmap_name)
#     r, g, b, _ = cmap(norm_val)
#     return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def calculate_color_gradient(value, min_val, max_val, start_color=(135, 206, 250), end_color=(139, 0, 0)):
    """Generates a color gradient based on a value range."""
    ratio = (value - min_val) / (max_val - min_val)
    ratio = min(max(ratio, 0), 1)  # Clamp between 0 and 1
    ratio = ratio**2.5  # Exponential scaling
    r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
    g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
    b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


def make_multiple_lines_subplots(df, filename, dict_colors, selected_zone=None, selected_year=None, column_subplots='scenario',
                              column_multiple_lines='competition', column_xaxis='t',
                              column_value='value', select_subplots=None, order_index=None,
                              dict_scenarios=None, figsize=(10,6),
                              format_y=lambda y, _: '{:.0f} MW'.format(y),  annotation_format="{:.0f}",
                              order_stacked=None, max_ticks=10, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with results.
    filename : str
        Path to save the figure.
    dict_colors : dict
        Dictionary with color arguments.
    selected_zone : str
        Zone to select.
    column_xaxis : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_multiple_bars : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_xaxis : list, optional
        Select a subset of subplots (e.g., a number of years).
    dict_grouping : dict, optional
        Dictionary for grouping variables and summing over a given group.
    order_index : list, optional
        Order of scenarios for plotting.
    dict_scenarios : dict, optional
        Dictionary for renaming scenarios.
    format_y : function, optional
        Function for formatting y-axis labels.
    order_stacked : list, optional
        Reordering the variables that will be stacked.
    cap : int, optional
        Under this cap, no annotation will be displayed.
    annotate : bool, optional
        Whether to annotate the bars.
    show_total : bool, optional
        Whether to show the total value on top of each bar.

    Example
    -------

    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])

    if column_subplots is not None:
        df = (df.groupby([column_subplots, column_multiple_lines, column_xaxis], observed=False)[
                  column_value].mean().reset_index())
        df = df.set_index([column_multiple_lines, column_xaxis, column_subplots]).squeeze().unstack(column_subplots)
    else:  # no subplots in this case
        df = (df.groupby([column_multiple_lines, column_xaxis], observed=False)[column_value].mean().reset_index())
        df = df.set_index([column_multiple_lines, column_xaxis])

    # TODO: change select_axis name
    if select_subplots is not None:
        df = df.loc[:, [i for i in df.columns if i in select_subplots]]

    multiple_lines_subplot(df, column_multiple_lines, filename, figsize=figsize, dict_colors=dict_colors,  format_y=format_y,
                           annotation_format=annotation_format,  rotation=rotation, order_index=order_index, dict_scenarios=dict_scenarios,
                           order_columns=order_stacked, max_ticks=max_ticks, annotate=annotate, show_total=show_total,
                           fonttick=fonttick, title=title)


def multiple_lines_subplot(df, column_multiple_lines, filename, figsize=(10,6), dict_colors=None, order_index=None,
                            order_columns=None, dict_scenarios=None, rotation=0, fonttick=14, legend=True,
                           format_y=lambda y, _: '{:.0f} GW'.format(y), annotation_format="{:.0f}",
                           max_ticks=10, annotate=True, show_total=False, title=None, ylim_bottom=None):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_group : str
        Column name to group by for the stacked bars.
    filename : str
        Path to save the plot image. If None, the plot is shown instead.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the bars. Default is None.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (10, 6).
    year_ini : str, optional
        Initial year to highlight in the plot. Default is None.
    order_index : list, optional
        List of scenario names to order the bars. Default is None.
    order_columns : list, optional
        List of column names to order the stacked bars. Default is None.
    dict_scenarios : dict, optional
        Dictionary mapping scenario names to new names for the plot. Default is None.
    rotation : int, optional
        Rotation angle for x-axis labels. Default is 0.
    fonttick : int, optional
        Font size for tick labels. Default is 14.
    legend : bool, optional
        Whether to display the legend. Default is True.
    format_y : function, optional
        Function to format y-axis labels. Default is a lambda function formatting as '{:.0f} GW'.
    cap : int, optional
        Minimum height of bars to annotate. Default is 6.
    annotate : bool, optional
        Whether to annotate each bar with its height. Default is True.
    show_total : bool, optional
        Whether to show the total value on top of each bar. Default is False.
    Returns
    -------
    None
    """

    valid_keys = []
    df_temps = {}

    for key in df.columns:
        try:
            df_temp = df[key].unstack(column_multiple_lines)

            # Apply filters before checking emptiness
            if dict_scenarios is not None:
                df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
            if order_index is not None:
                df_temp = df_temp.loc[[c for c in order_index if c in df_temp.index], :]
            if order_columns is not None:
                df_temp = df_temp[[c for c in order_columns if c in df_temp.columns]]

            df_temp = df_temp.dropna(axis=1, how='all')  # drop columns with all NaNs

            if not df_temp.empty:
                valid_keys.append(key)
                df_temps[key] = df_temp

        except Exception:
            continue

    num_subplots = len(valid_keys)
    if num_subplots == 0:
        print("No data available to plot.")
        return

    n_columns = min(3, num_subplots)
    n_rows = int(np.ceil(num_subplots / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1] * n_rows), sharey='all',
                             gridspec_kw={'width_ratios': [1] * n_columns})

    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array


    handles, labels = None, None
    all_handles, all_labels = [], []
    for k, key in enumerate(valid_keys):
        ax = axes[k]
        df_temp = df_temps[key]

        plot = df_temp.plot(ax=ax, kind='line', marker='o',
                     color=dict_colors if dict_colors is not None else None)

        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels

        num_xticks = min(len(df_temp.index), max_ticks)  # Set a reasonable max number of ticks
        xticks_positions = np.linspace(0, len(df_temp.index) - 1, num_xticks, dtype=int)
        ax.set_xticks(xticks_positions)  # Set tick positions
        ax.set_xticklabels(df_temp.index[xticks_positions], rotation=rotation)

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
        # put tick label in bold
        ax.tick_params(axis='both', which=u'both', length=0)
        ax.set_xlabel('')

        if len(valid_keys) > 1:
            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
        else:
            if title is not None:
                if isinstance(title, tuple):
                    title = '{}-{}'.format(title[0], title[1])
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

        if k == 0:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        if k % n_columns != 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax.get_legend().remove()

        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)

        # Add a horizontal line at 0
        # ax.axhline(0, color='black', linewidth=0.5)


    if legend:
        seen = set()
        unique = [(h, l) for h, l in zip(all_handles, all_labels) if not (l in seen or seen.add(l))]
        fig.legend(
            [h for h, _ in unique],
            [l.replace('_', ' ') for _, l in unique],
            loc='center left',
            frameon=False,
            ncol=1,
            bbox_to_anchor=(1, 0.5)
        )

        # fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
        #            bbox_to_anchor=(1, 0.5))

    # Hide unused subplots
    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_line_subplots(df, filename, x_column, y_column, subplot_column,
                       group_column=None, dict_colors=None, format_y=None,
                       figsize=(10, 5), rotation=0, fonttick=12, title=None,
                       xlabel=None, ylabel=None):
    """
    Create multiple line subplots from a DataFrame, sliced by a given column.

    Parameters
    ----------
    df : pd.DataFrame
        The data to be plotted.
    filename : str
        Path to save the resulting figure.
    x_column : str
        Name of the column for the x-axis.
    y_column : str
        Name of the column for the y-axis.
    subplot_column : str
        Column used to create one subplot per unique value (e.g., 'zone', 'attribute').
    group_column : str, optional
        If specified, plots one line per value of this column inside each subplot.
    dict_colors : dict, optional
        Dictionary mapping group_column values to colors.
    format_y : function, optional
        A function for formatting the y-axis ticks.
    figsize : tuple, default=(10, 5)
        Size of each subplot (width, height).
    rotation : int, default=0
        Rotation of the x-axis tick labels.
    fonttick : int, default=12
        Font size for tick labels.
    title : str, optional
        Title for the entire figure.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    """

    unique_subplots = df[subplot_column].unique()
    ncols = min(3, len(unique_subplots))
    nrows = int(np.ceil(len(unique_subplots) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, key in enumerate(unique_subplots):
        ax = axes[i]
        subset = df[df[subplot_column] == key]

        if group_column:
            for g, data in subset.groupby(group_column):
                color = dict_colors[g] if dict_colors and g in dict_colors else None
                ax.plot(data[x_column], data[y_column], label=str(g), color=color)
        else:
            ax.plot(subset[x_column], subset[y_column], color='steelblue')

        ax.set_title(str(key), fontsize=fonttick, fontweight='bold')
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, linestyle='--', alpha=0.5)

        if format_y:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

        if i % ncols == 0:
            ax.set_ylabel(ylabel if ylabel else y_column, fontsize=fonttick)

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel if xlabel else x_column, fontsize=fonttick)

        if group_column:
            ax.legend(frameon=False, fontsize=fonttick - 2)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=fonttick + 2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
def make_line_subplots(df, filename, x_column, y_column, subplot_column,
                       group_column=None, dict_colors=None, format_y=None,
                       figsize=(10, 5), rotation=0, fonttick=12, title=None,
                       xlabel=None, ylabel=None):
    """
    Create multiple line subplots from a DataFrame, sliced by a given column.

    Parameters
    ----------
    df : pd.DataFrame
        The data to be plotted.
    filename : str
        Path to save the resulting figure.
    x_column : str
        Name of the column for the x-axis.
    y_column : str
        Name of the column for the y-axis.
    subplot_column : str
        Column used to create one subplot per unique value (e.g., 'zone', 'attribute').
    group_column : str, optional
        If specified, plots one line per value of this column inside each subplot.
    dict_colors : dict, optional
        Dictionary mapping group_column values to colors.
    format_y : function, optional
        A function for formatting the y-axis ticks.
    figsize : tuple, default=(10, 5)
        Size of each subplot (width, height).
    rotation : int, default=0
        Rotation of the x-axis tick labels.
    fonttick : int, default=12
        Font size for tick labels.
    title : str, optional
        Title for the entire figure.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    """

    unique_subplots = df[subplot_column].unique()
    ncols = min(3, len(unique_subplots))
    nrows = int(np.ceil(len(unique_subplots) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, key in enumerate(unique_subplots):
        ax = axes[i]
        subset = df[df[subplot_column] == key]

        if group_column:
            for g, data in subset.groupby(group_column):
                color = dict_colors[g] if dict_colors and g in dict_colors else None
                ax.plot(data[x_column], data[y_column], label=str(g), color=color)
        else:
            ax.plot(subset[x_column], subset[y_column], color='steelblue')

        ax.set_title(str(key), fontsize=fonttick, fontweight='bold')
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, linestyle='--', alpha=0.5)

        if format_y:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

        if i % ncols == 0:
            ax.set_ylabel(ylabel if ylabel else y_column, fontsize=fonttick)

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel if xlabel else x_column, fontsize=fonttick)

        if group_column:
            ax.legend(frameon=False, fontsize=fonttick - 2)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=fonttick + 2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print(0)

