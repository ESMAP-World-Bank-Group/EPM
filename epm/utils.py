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
COLORS = 'static/colors.csv'


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


def read_plot_specs():
    """Extracts specifications for plots from excel.
    excel_spec: str
        Path to excel file
    """

    #correspondence_Co = pd.read_csv(COUNTRIES)
    #correspondence_Fu = pd.read_csv(FUELS)
    #correspondence_Te = pd.read_csv(TECHS)
    #correspondence_Gen = pd.read_csv(GEN)
    colors = pd.read_csv(COLORS)
    external_zones_locations = pd.read_csv(EXTERNAL_ZONES)
    fuel_mapping = pd.read_csv(FUELS)
    tech_mapping = pd.read_csv(TECHS)


    """correspondence_Gen.columns = ['generator', 'fuel', 'tech', 'zone']
    external_zones_locations.columns = ['zone', 'location']

    if external_zones_locations.shape[0] == 0:
        external_zones_included = False
    else:
        external_zones_included = True
        external_zones = list(external_zones_locations['zone'].unique())"""

    dict_specs = {
        #'correspondence_Co': correspondence_Co,
        #'correspondence_Fu': correspondence_Fu,
        #'correspondence_Te': correspondence_Te,
        #'correspondence_Gen': correspondence_Gen,
        #'external_zones_locations': external_zones_locations,
        #'external_zones_included': external_zones_included,
        'colors': colors.set_index('Processing')['Color'].to_dict(),
        'fuel_mapping': fuel_mapping.set_index('EPM_Fuel')['Processing'].to_dict(),
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict()
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


def process_epmresults(epmresults, dict_specs):
    """Processing EPM results to use in plots."""

    # TODO: 'zone_from', 'zone_to'
    # TODO: 'uni' is sometimes replaced by attribute sometimes not. Now always attribute. Change code accordingly.
    # TODO: remove correspondence_fuel_epm

    rename_columns = {'c': 'country', 'y': 'year', 'v': 'value', 's': 'scenario', 'uni': 'attribute',
                      'z': 'zone', 'g': 'generator', 'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't'}

    keys = {'pDemandSupplyCountry', 'pDemandSupply', 'pEnergyByPlant', 'pEnergyByFuel', 'pCapacityByFuel',
            'pPlantUtilization', 'pCostSummary', 'pCostSummaryCountry', 'pEmissions', 'pPrice', 'pHourlyFlow',
            'pDispatch', 'pFuelDispatch', 'pPlantFuelDispatch', 'pInterconUtilization',
            'InterconUtilization', 'pInterchange', 'Interchange', 'interchanges', 'pInterconUtilizationExt',
            'InterconUtilizationExt', 'pInterchangeExt', 'InterchangeExt', 'annual_line_capa', 'pAnnualTransmissionCapacity',
            'AdditiononalCapacity_trans'}

    # Rename columns
    epm_dict = {k: i.rename(columns=rename_columns) for k, i in epmresults.items() if k in keys and k in epmresults.keys()}

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

    # Standardize names
    def standardize_names(key, mapping):
        if key in epm_dict.keys():
            epm_dict[key].replace(mapping, inplace=True)
            epm_dict[key].groupby(
                [i for i in epm_dict[key].columns if i != 'value']).sum().reset_index()

    standardize_names('pEnergyByFuel', dict_specs['fuel_mapping'])
    standardize_names('pCapacityByFuel', dict_specs['fuel_mapping'])
    standardize_names('pPlantFuelDispatch', dict_specs['tech_mapping'])

    return epm_dict

def for_later(epmresults, years):
    # TODO: no need to set False or None. Just check if in epm_dict.keys().
    # TODO: best practice


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
    return None


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
    df_tot = pDemandSupplyCountry.loc[pDemandSupplyCountry['attribute'] == 'Demand: GWh']
    df_tot = df_tot.groupby(['year']).agg({'value': 'sum'}).reset_index()

    if unit == 'TWh':
        df_tot['value'] = df_tot['value'] / 1000
    elif unit == '000 TWh':
        df_tot['value'] = df_tot['value'] / 1000000

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


def make_generation_plot(pEnergyByFuel, folder, years=None, plot_option='bar', selected_scenario=None, unit='GWh',
                         BESS_included=True, Hydro_stor_included=True):
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
        make_line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Generation {unit}',
                       title=f'Total generation - {selected_scenario} scenario',
                       filename=f'{folder}/TotalGeneration_{plot_option}.png')
    elif plot_option == 'bar':
        make_bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Generation {unit}',
                      title=f'Total generation - {selected_scenario} scenario',
                      filename=f'{folder}/TotalGeneration_{plot_option}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def make_subplot_pie(df, index, dict_colors, subplot_column, title='', figsize=(16, 4), percent_cap=6, filename=None):

    # Group by the column for subplots
    groups = df.groupby(subplot_column)

    # Calculate the number of subplots
    num_subplots = len(groups)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize, constrained_layout=True)
    if num_subplots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Plot each group as a pie chart
    all_labels = set()  # Collect all labels for the combined legend
    for ax, (name, group) in zip(axes, groups):
        colors = [dict_colors[f] for f in group[index]]
        group.plot.pie(
            ax=ax,
            y='value',
            autopct=lambda p: f'{p:.0f}%' if p > percent_cap else '',
            startangle=140,
            legend=False,
            colors=colors,
            labels=None
        )
        ax.set_ylabel('')
        ax.set_title(name)
        all_labels.update(group[index])  # Collect unique labels

    # Create a shared legend below the graphs
    all_labels = sorted(all_labels)  # Sort labels for consistency
    handles = [plt.Line2D([0], [0], marker='o', color=dict_colors[label], linestyle='', markersize=10) for label in all_labels]
    fig.legend(
        handles,
        all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(handles),  # Adjust number of columns based on subplots
        title="Legend"
    )

    # Add title for the whole figure
    fig.suptitle(title, fontsize=16)

    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def make_fuel_mix_pie_plot(df, years, graphs_folder, dict_colors, BESS_included=False, Hydro_stor_included=False,
                           figsize=(16, 4), percent_cap=6, selected_scenario=None):
    if not BESS_included:
        df = df[df['fuel'] != 'Battery Storage']
    if not Hydro_stor_included:
        df = df[df['fuel'] != 'Pumped-Hydro Storage']

    df = df.loc[df['year'].isin(years)].groupby(['year', 'fuel']).agg({'value': 'sum'}).reset_index()
    df['value'] = df['value'].apply(lambda x: 0 if x < 0 else x)

    title = f'Energy mix - {selected_scenario} scenario'
    temp = '_'.join([str(y) for y in years])
    filename = f'{graphs_folder}/EnergyMixPie_{temp}_{selected_scenario}.png'

    make_subplot_pie(df, 'fuel', dict_colors, 'year', title=title, figsize=figsize,
                     percent_cap=percent_cap, filename=filename)



if __name__ == '__main__':
    print(0)