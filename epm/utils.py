# General packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import os
import math
import gams.transfer as gt
import seaborn as sns
import openpyxl
from collections import OrderedDict
import ast

# For the maps
import matplotlib.pyplot as plt
# import geopandas as gpd
# import folium
import json

import time
from PIL import Image
import io
import base64
# import cv2

from pathlib import Path

COUNTRIES = 'static/countries.csv'
FUELS = 'static/fuels.csv'
TECHS = 'static/technologies.csv'
GEN = 'static/generation.csv'
EXTERNAL_ZONES = 'static/external_zones.csv'


def create_folders(graphs_results_folder, scenario):
    """Creates output folders."""
    results_scenario = Path(graphs_results_folder) / Path(f'Graphs_results_{scenario}')
    if not results_scenario.exists():
        results_scenario.mkdir()

    if not os.path.exists(results_scenario):
        os.makedirs(results_scenario)

    if not os.path.exists(f'{results_scenario}/Mix_Zone'):
        os.makedirs(f'{results_scenario}/Mix_Zone')

    if not os.path.exists(f'{results_scenario}/Dispatch_Zone'):
        os.makedirs(f'{results_scenario}/Dispatch_Zone')

    if not os.path.exists(f'{results_scenario}/NetImports_Zone'):
        os.makedirs(f'{results_scenario}/NetImports_Zone')

    if not os.path.exists(f'{results_scenario}/MC_Zone'):
        os.makedirs(f'{results_scenario}/MC_Zone')

    if not os.path.exists(f'{results_scenario}/Concat_Zone'):
        os.makedirs(f'{results_scenario}/Concat_Zone')

    if not os.path.exists(f'{results_scenario}/MixEvolution_Country'):
        os.makedirs(f'{results_scenario}/MixEvolution_Country')

    if not os.path.exists(f'{results_scenario}/MixPie_Country'):
        os.makedirs(f'{results_scenario}/MixPie_Country')


def read_plot_specs(excel_spec, mode='csv'):
    """Extracts specifications for plots from excel.
    excel_spec: str
        Path to excel file
    """
    if mode == 'excel':
        correspondence_Co = pd.read_excel(excel_spec, sheet_name='Zones_Countries')
        correspondence_Fu = pd.read_excel(excel_spec, sheet_name='Fuels')
        correspondence_Te = pd.read_excel(excel_spec, sheet_name='Techs')
        correspondence_Gen = pd.read_excel(excel_spec, sheet_name='Gen')
    elif mode == 'csv':
        correspondence_Co = pd.read_csv(COUNTRIES)
        correspondence_Fu = pd.read_csv(FUELS)
        correspondence_Te = pd.read_csv(TECHS)
        correspondence_Gen = pd.read_csv(GEN)
        external_zones_locations = pd.read_csv(EXTERNAL_ZONES)
    else:
        raise ValueError('Mode should be either "excel" or "csv"')

    correspondence_Gen.columns = ['generator', 'fuel', 'tech', 'zone']
    external_zones_locations.columns = ['zone', 'location']

    if external_zones_locations.shape[0] == 0:
        external_zones_included = False
    else:
        external_zones_included = True
        external_zones = list(external_zones_locations['zone'].unique())

    # Creating mappings
    #TODO:  Filter for the zones needed
    fuel_mapping = correspondence_Fu.set_index('EPM_Fuel')['Processing'].to_dict()
    tech_mapping = correspondence_Te.set_index('EPM_Tech')['Processing'].to_dict()
    # fuels_list = list(set(fuel_mapping.values()))
    # techs_list = list(set(tech_mapping.values()))

    dict_specs = {
        'correspondence_Co': correspondence_Co,
        'correspondence_Fu': correspondence_Fu,
        'correspondence_Te': correspondence_Te,
        'correspondence_Gen': correspondence_Gen,
        'external_zones_locations': external_zones_locations,
        'external_zones_included': external_zones_included,
        'fuel_mapping': fuel_mapping,
        'tech_mapping': tech_mapping
    }
    return dict_specs


def read_input_data(excel_input):
    """Reads input data required for the plots.
    excel_input: str
        Path to excel file containing inputs. Eg, WB_EPM_8_5.xlsx
    """
    # Read generation data
    sheet_name = 'Generator Data'
    names = ['Plants', 'Zone', 'Type', 'fuel1']  # Columns to select
    new_column_names = {'Plants': 'EPM_Gen', 'Zone': 'EPM_Zone', 'fuel1': 'EPM_Fuel', 'Type': 'EPM_Tech'}

    # Read the entire sheet starting from A6
    generation_data = pd.read_excel(excel_input, sheet_name=sheet_name, skiprows=5)

    generation_data = generation_data.loc[:, names]
    generation_data.rename(columns=new_column_names, inplace=True)
    generation_data = generation_data.dropna(how='all')

    gen_to_fuel_mapping = generation_data.set_index('EPM_Gen')['EPM_Fuel'].to_dict()

    sheet_name = 'Duration'
    duration_data = pd.read_excel(excel_input, sheet_name=sheet_name, skiprows=5)
    duration_data = duration_data.dropna(how='all')
    duration_data.columns = ['season', 'day', 'year', *duration_data.columns[3:]]
    pDuration = duration_data.melt(id_vars=['season', 'day', 'year'], var_name='t', value_name='value')
    pDuration = pDuration.sort_values(by=['season', 'day', 'year', 't'])
    pDuration.columns = ['season', 'day', 'year', 't', 'duration']
    pDuration['year'] = pDuration['year'].astype(str)

    dict_inputs = {
        'generation_data': generation_data,
        'gen_to_fuel_mapping': gen_to_fuel_mapping,
        'duration_data': duration_data,
        'pDuration': pDuration
    }
    return dict_inputs


def calculate_pRR(discount_rate, y, years_mapping):
    """
    # Discount factor to apply when estimating total system costs.
    # When years have a weight larger than 1, uses the middle time point (e.g., 2027 when time range is 2025-2030)
    :param discount_rate: float
        Yearly discount rate
    :param y: int
        Given year
    :param years_mapping: dict
        Maps year to a given weight, for use when compiling total cost
    :return:
    """
    # Calcul du facteur de discount pour une année donnée

    years = list(years_mapping.keys())
    # years_int = [int(y) for y in years_mapping.values()]

    if y == years[0]:
        pRR = 1.0
    else:
        pRR = 1 / (
                (1 + discount_rate) ** (sum(years_mapping[y2] for y2 in years_mapping if y2 < y) - 1 +
                            sum(years_mapping[y2] / 2 for y2 in years_mapping if y2 == y))
        )
    return pRR


def country_to_color(epm_country, color_country_mapping):
    """Gets color to represent a country.
    epm_country: str
        A given country such as 'Liberia'
    color_country_mapping: dict
        Dictionary mapping country names to colours. For instance, {'Liberia': 'tan'}
    """
    return color_country_mapping.get(epm_country)

def zone_to_country(zone, zone_mapping):
    """Gets country corresponding to zone.
    zone: str
        A given zone such as 'Liberia'
    zone_mapping: dict
        Dictionary mapping zones to countries. For instance, {'Liberia': 'Liberia'}
    """
    return zone_mapping.get(zone)


def correspondence_fuel_epm(epm_fuel, fuel_mapping):
    """Gets processed name for a given fuel in gdx output.
    epm_fuel: str
        A given fuel such as 'HydroMC'
    fuel_mapping: dict
        Dictionary mapping fuel names from EPM gdx to values for plots. For instance, {'HydroMC': 'Hydro'}
    """
    return fuel_mapping.get(epm_fuel)


def correspondence_tech_epm(epm_tech, tech_mapping):
    """Gets processed name for a given fuel in gdx output.
    epm_tech: str
        A given tech such as 'ROR'
    tech_mapping: dict
        Dictionary mapping tech names from EPM gdx to values for plots. For instance, {'ROR': 'Hydro'}
    """
    return tech_mapping.get(epm_tech)


def generation_to_fuel(generation, gen_to_fuel_mapping):
    """Gets fuel corresponding to a given generation power plant.
    generation: str
        A given generation plant such as 'MTCoffee'
    gen_to_fuel_mapping: dict
        Dictionary mapping generation name to fuel. For instance, {'MTCoffee': 'HydroMC'}
    """
    return gen_to_fuel_mapping.get(generation)


def fuel_to_color(fuel, color_fuel_mapping):
    """Gets color corresponding to fuel.
    fuel: str
        A given fuel such as 'Hydro'
    color_fuel_mapping: dict
        Dictionary mapping fuels to colors. For instance, {'Hydro': 'lightskyblue'}
    """
    color_fuel = color_fuel_mapping.get(fuel)
    if color_fuel.startswith('('):
        color_rgb = eval(color_fuel)
        color_fuel = [comp for comp in color_rgb]
    return color_fuel


def tech_to_color(tech, color_tech_mapping):
    """Gets color corresponding to tech.
    tech: str
        A given tech such as 'Hydro'
    color_tech_mapping: dict
        Dictionary mapping techs to colors. For instance, {'Coal': 'darkgray'}
    """
    color_tech = color_tech_mapping.get(tech)
    if color_tech.startswith('('):
        color_rgb = eval(color_tech)
        color_tech = [comp for comp in color_rgb]
    return color_tech


def extract_epm_results(results_folder, scenario):
    """Extracts all information from the gdx files outputed by EPM."""
    # Getting all the epmresults.gdx of the different cases
    containers = {}

    for all_path in [file.path for file in os.scandir(results_folder) if file.is_dir()]:
        containers[all_path[12:]] = gt.Container(f'{all_path}/epmresults.gdx')

    scenarios = [all_path[12:] for all_path in [file.path for file in os.scandir(results_folder) if file.is_dir()]]
    print(f' Scenarios in the folder: {scenarios}')
    print('')

    # Get only the selected scenario
    scenarios = [scenario]

    epmresults = {}
    parameters = [p.name for p in containers[scenarios[0]].getParameters()]

    # noinspection PyUnboundLocalVariable
    for parameter in parameters:
        df_parameter_all = []

        for scenario in scenarios:
            if containers[scenario].data[parameter].records is not None:
                df_parameter = containers[scenario].data[parameter].records.copy()
                df_parameter['scenario'] = scenario
                df_parameter_all.append(df_parameter)

            if not df_parameter_all == []:
                epmresults[parameter] = pd.concat(df_parameter_all)
            else:
                # print(f'Empty parameter for {parameter}')
                continue


    return epmresults


def process_epmresults(epmresults, correspondence_Gen, fuel_mapping, years):
    """Processing EPM results to use in plots."""
    pSolverParameters = epmresults['pSolverParameters']

    pSummary = epmresults['pSummary']
    pDemandSupplyCountry = epmresults['pDemandSupplyCountry']
    if years is None:
        years = list(pDemandSupplyCountry['y'].unique())

    #TODO: Make dict {'c': 'country', ...} to rename columns and use method
    pDemandSupplyCountry.columns = ['country', 'attribute', 'year', 'value', 'scenario']

    pDemandSupply = epmresults['pDemandSupply']
    pDemandSupply.columns = ['zone', 'attribute', 'year', 'value', 'scenario']

    pEnergyByPlant = epmresults['pEnergyByPlant']
    pEnergyByPlant.columns = ['zone', 'generator', 'year', 'value', 'scenario']

    pEnergyByFuel = epmresults['pEnergyByFuel']
    pEnergyByFuel.columns = ['zone', 'fuel', 'year', 'value', 'scenario']
    pEnergyByFuel['fuel'] = pEnergyByFuel['fuel'].copy().apply(lambda x:correspondence_fuel_epm(x, fuel_mapping))

    pCapacityByFuel = epmresults['pCapacityByFuel']
    pCapacityByFuel.columns = ['zone', 'fuel', 'year', 'value', 'scenario']
    pCapacityByFuel['fuel'] = pCapacityByFuel['fuel'].copy().apply(lambda x:correspondence_fuel_epm(x, fuel_mapping))

    pPlantUtilization = epmresults['pPlantUtilization']
    pPlantUtilization.columns = ['zone', 'generator', 'year', 'value', 'scenario']

    pCostSummary = epmresults['pCostSummary']
    pCostSummary.columns = ['zone', 'uni', 'year', 'value', 'scenario']

    pCostSummaryCountry = epmresults['pCostSummaryCountry']
    pCostSummaryCountry.columns = ['country', 'uni', 'year', 'value', 'scenario']

    pEmissions = epmresults['pEmissions']
    if 'uni' in list(pEmissions.columns):
        pEmissions = pEmissions.drop(columns=['uni'])
    pEmissions.columns = ['zone', 'year', 'value', 'scenario']

    pPrice = epmresults['pPrice']
    pPrice.columns = ['zone', 'season', 'day', 't', 'year', 'value', 'scenario']

    if 'pHourlyFlow' in list(epmresults.keys()):
        no_pHourlyFlow = False
        pHourlyFlow = epmresults['pHourlyFlow']
        pHourlyFlow.columns = ['zone_from', 'zone_to', 'season', 'day', 't', 'year', 'value', 'scenario']
    else:
        no_pHourlyFlow = True
        pHourlyFlow = None

    pDispatch = epmresults['pDispatch']
    pDispatch.columns = ['zone', 'year', 'season', 'day', 'uni', 't', 'value', 'scenario']

    if 'pFuelDispatch' in list(epmresults.keys()):
        fuel_dispatch = True
        pFuelDispatch = epmresults['pFuelDispatch']
        pFuelDispatch.columns = ['zone', 'season', 'day', 't', 'year', 'fuel', 'value', 'scenario']
        pPlantFuelDispatch = None

    else:
        fuel_dispatch = False
        pFuelDispatch = None
        pPlantDispatch = epmresults['pPlantDispatch']
        if 'uni' in list(pPlantDispatch.columns):
            pPlantDispatch = pPlantDispatch.drop(columns=['uni'])
        pPlantDispatch.columns = ['zone', 'year', 'season', 'day', 'generator', 't', 'value', 'scenario']
        pPlantFuelDispatch = pPlantDispatch.merge(correspondence_Gen, on=['generator'], how='left')

    if 'pInterconUtilization' in list(epmresults.keys()):

        interchanges = True

        pInterconUtilization = epmresults['pInterconUtilization']
        pInterconUtilization.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        InterconUtilization = pInterconUtilization.pivot(index=['zone_from', 'zone_to', 'scenario'], columns='year',
                                                         values='value')
        InterconUtilization.reset_index(inplace=True)
        InterconUtilization[years] = InterconUtilization[years] * 100  # %

        pInterchange = epmresults['pInterchange']
        pInterchange.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        Interchange = pInterchange.pivot(index=['zone_from', 'zone_to', 'scenario'], columns='year', values='value')
        Interchange.reset_index(inplace=True)
        Interchange[years] = Interchange[years]

    else:
        interchanges = False
        pInterconUtilization = None
        InterconUtilization = None
        pInterchange = None
        Interchange = None

    if 'pInterconUtilizationExtExp' in list(epmresults.keys()):

        interchanges_ext = True
        # Getting the imports and exmports and concatening them
        pInterconUtilizationExtExp = epmresults['pInterconUtilizationExtExp']
        pInterconUtilizationExtExp.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        pInterconUtilizationExtImp = epmresults['pInterconUtilizationExtImp']
        pInterconUtilizationExtImp.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        pInterconUtilizationExt = pd.concat([pInterconUtilizationExtExp, pInterconUtilizationExtImp])

        InterconUtilizationExt = pInterconUtilizationExt.pivot(index=['zone_from', 'zone_to', 'scenario'],
                                                               columns='year', values='value')
        InterconUtilizationExt.reset_index(inplace=True)
        InterconUtilizationExt[years] = InterconUtilizationExt[years] * 100  # %
        InterconUtilizationExt[years] = InterconUtilizationExt[years].fillna(0)

        pInterchangeExtExp = epmresults['pInterchangeExtExp']
        pInterchangeExtExp.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        pInterchangeExtImp = epmresults['pInterchangeExtImp']
        pInterchangeExtImp.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        pInterchangeExt = pd.concat([pInterchangeExtExp, pInterchangeExtImp])

        InterchangeExt = pInterchangeExt.pivot(index=['zone_from', 'zone_to', 'scenario'], columns='year',
                                               values='value')
        InterchangeExt.reset_index(inplace=True)
        InterchangeExt[years] = InterchangeExt[years].fillna(0)

    else:
        interchanges = False
        pInterconUtilizationExt = None
        InterconUtilizationExt = None
        pInterchangeExt = None
        InterchangeExt = None

    if 'pAnnualTransmissionCapacity' in list(epmresults.keys()):
        annual_line_capa = True
        pAnnualTransmissionCapacity = epmresults['pAnnualTransmissionCapacity']
        pAnnualTransmissionCapacity.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
    else:
        annual_line_capa = False
        pAnnualTransmissionCapacity = None

    if 'pAdditionalCapacity' in list(epmresults.keys()):
        add_line_capa = True
        pAdditiononalCapacity_trans = epmresults['pAdditionalCapacity']
        pAdditiononalCapacity_trans.columns = ['zone_from', 'zone_to', 'year', 'value', 'scenario']
        AdditiononalCapacity_trans = pAdditiononalCapacity_trans.pivot(index=['zone_from', 'zone_to', 'scenario'],
                                                                       columns='year', values='value')
        AdditiononalCapacity_trans.reset_index(inplace=True)
        #AdditiononalCapacity_trans[years[7:]] = AdditiononalCapacity_trans[years[7:]] / 1000  # To GW
        AdditiononalCapacity_trans = AdditiononalCapacity_trans / 1000
    else:
        add_line_capa = False
        AdditiononalCapacity_trans = None

    epm_dict = {
        'pSolverParameters': pSolverParameters,
        'pSummary': pSummary,
        'pDemandSupplyCountry': pDemandSupplyCountry,
        'pDemandSupply': pDemandSupply,
        'pEnergyByPlant': pEnergyByPlant,
        'pEnergyByFuel': pEnergyByFuel,
        'pCapacityByFuel': pCapacityByFuel,
        'pPlantUtilization': pPlantUtilization,
        'pCostSummary': pCostSummary,
        'pCostSummaryCountry': pCostSummaryCountry,
        'pEmissions': pEmissions,
        'pPrice': pPrice,
        'pHourlyFlow': pHourlyFlow,
        'no_pHourlyFlow': no_pHourlyFlow,
        'pDispatch': pDispatch,
        'pFuelDispatch': pFuelDispatch,
        'fuel_dispatch': fuel_dispatch,
        'pPlantFuelDispatch': pPlantFuelDispatch,
        'pInterconUtilization': pInterconUtilization,
        'InterconUtilization': InterconUtilization,
        'pInterchange': pInterchange,
        'Interchange': Interchange,
        'interchanges': interchanges,
        'pInterconUtilizationExt': pInterconUtilizationExt,
        'InterconUtilizationExt': InterconUtilizationExt,
        'pInterchangeExt': pInterchangeExt,
        'InterchangeExt': InterchangeExt,
        'pAnnualTransmissionCapacity': pAnnualTransmissionCapacity,
        'annual_line_capa': annual_line_capa,
        'AdditiononalCapacity_trans': AdditiononalCapacity_trans,
        'add_line_capa': add_line_capa
    }
    return epm_dict


def make_line_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(10, 6)):
    """Makes a line plot.

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
    figsize: tuple, optional, default=(8, 5)
        Size of the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[x], df[y])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if filename is not None:
        plt.savefig(filename)

    plt.show()
    return None


def make_bar_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(8, 5), round_tot=0):
    """Makes a bar plot.

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
    figsize: tuple, optional, default=(8, 5)
        Size of the figure
    round_tot: int
        Number of decimal places to round the values

    """
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df[x], df[y], width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, round_tot), va='bottom', ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None


def make_demand_plot(pDemandSupplyCountry, folder, years=None, plot_option='bar', selected_scenario=None, unit='GWh'):
    """Makes a plot of demand for all countries.

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
    df_tot = pDemandSupplyCountry.loc[pDemandSupplyCountry['attribute'] == 'Demand: GWh'].groupby(['year']).agg(
        {'country': 'first', 'attribute': 'first', 'value': 'sum'}).reset_index()
    df_tot['country'] = 'all'

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000

    if years is not None:
        df_tot = df_tot.loc[df_tot['year'].isin(years)]

    if plot_option == 'line':
        make_line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Demand {unit}',
                       title=f'Total demand - {selected_scenario} scenario',
                       filename=f'{folder}/TotalDemand_{plot_option}.png')
    elif plot_option == 'bar':
        make_bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Demand {unit}',
                      title=f'Total demand - {selected_scenario} scenario',
                      filename=f'{folder}/TotalDemand_{plot_option}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


if __name__ == '__main__':
    print(0)