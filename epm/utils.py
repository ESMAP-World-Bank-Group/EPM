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
GENERATION = 'static/generation.csv'
EXTERNAL_ZONES = 'static/external_zones.csv'
COLORS = 'static/colors.csv'

NAME_COLUMNS = {
    'pFuelDispatch': 'fuel',
    'pDispatch': 'attribute'
}

def create_folders_imgs(folder):
    """Creating folders for images"""
    for p in [path for path in Path(folder).iterdir() if path.is_dir()]:
        if not (p / Path('images')).is_dir():
            os.mkdir(p / Path('images'))
    if not (Path(folder) / Path('images')).is_dir():
        os.mkdir(Path(folder) / Path('images'))


def read_plot_specs():
    """Extracts specifications for plots from excel.
    excel_spec: str
        Path to excel file
    """
    # TODO: generation_mapping should be extracted from EPM results automatically, not specified by each user
    #correspondence_Co = pd.read_csv(COUNTRIES)
    #correspondence_Fu = pd.read_csv(FUELS)
    #correspondence_Te = pd.read_csv(TECHS)
    generation_mapping = pd.read_csv(GENERATION)
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
        'generation_mapping': generation_mapping.set_index('EPM_Gen')['EPM_Fuel'].to_dict(),
        #'external_zones_locations': external_zones_locations,
        #'external_zones_included': external_zones_included,
        'colors': colors.set_index('Processing')['Color'].to_dict(),
        'fuel_mapping': fuel_mapping.set_index('EPM_Fuel')['Processing'].to_dict(),
        'tech_mapping': tech_mapping.set_index('EPM_Tech')['Processing'].to_dict()
    }
    return dict_specs


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

    return df


def extract_epm_folder(results_folder, file='epmresults.gdx'):
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


def standardize_names(dict_df, key, mapping, column='fuel'):
    """Standardize the names of fuels in the dataframes.

    Only works when dataframes have fuel and value (with numerical values) columns.

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
    """Processing EPM inputs to use in plots.

    epm_input: dict
        Dictionary containing the input data
    dict_specs: dict
        Dictionary containing the specifications for the plots
    """

    keys = ['ftfindex', 'pGenDataExcel', 'pTechDataExcel', 'pZoneIndex']
    epm_input = {k: i for k, i in epm_input.items() if k in keys}
    if scenarios_rename is not None:
        for k, i in epm_input.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    epm_input['ftfindex'].rename(columns={'uni_0': 'fuel1', 'uni_1': 'fuel2'}, inplace=True)
    mapping_fuel1 = epm_input['ftfindex'].loc[:, ['fuel1', 'value']].drop_duplicates().set_index(
        'value').squeeze().to_dict()
    mapping_fuel2 = epm_input['ftfindex'].loc[:, ['fuel2', 'value']].drop_duplicates().set_index(
        'value').squeeze().to_dict()

    temp = epm_input['pTechDataExcel']
    temp['uni'] = temp['uni'].astype(str)
    temp = temp[temp['uni'] == 'Assigned Value']
    mapping_tech = temp.loc[:, ['Abbreviation', 'value']].drop_duplicates().set_index(
        'value').squeeze()
    mapping_tech.replace(dict_specs['tech_mapping'], inplace=True)

    mapping_zone = epm_input['pZoneIndex'].loc[:, ['value', 'ZONE']].set_index('value').loc[:, 'ZONE'].to_dict()
    epm_input.update({'mapping_zone': mapping_zone})

    # Modify pGenDataExcel
    df = epm_input['pGenDataExcel'].pivot(index=['scenario', 'Plants'], columns='uni', values='value')
    df = df.loc[:, ['Type', 'fuel1', 'fuel2']]
    df['fuel1'] = df['fuel1'].replace(mapping_fuel1)
    df['fuel2'] = df['fuel2'].replace(mapping_fuel2)

    df['fuel1'] = df['fuel1'].replace(dict_specs['fuel_mapping'])
    df['fuel2'] = df['fuel2'].replace(dict_specs['fuel_mapping'])

    df['Type'] = df['Type'].replace(mapping_tech)

    epm_input['pGenDataExcel'] = df.reset_index()

    epm_input['pGenDataExcel'].rename(columns={'Plants': 'generator'}, inplace=True)

    return epm_input


def remove_unused_tech(epm_dict, list_keys):
    """
    Remove rows that correspond to technologies that are never used across the whole time horizon
    :param epm_dict: dict
        Containing all the outputs of interest
    :param list_keys: list
        List containing the dataframes we want to process
    :return:
    """
    for key in list_keys:
        epm_dict[key] = epm_dict[key].where((epm_dict[key]['value'] > 2e-6) | (epm_dict[key]['value'] < -2e-6),np.nan)  # get rid of small values to avoid unneeded labels
        epm_dict[key] = epm_dict[key].dropna(subset=['value'])

    return epm_dict



def process_epm_results(epm_results, dict_specs, scenarios_rename=None):
    """Processing EPM results to use in plots."""

    # TODO: 'zone_from', 'zone_to'
    # TODO: 'uni' is sometimes replaced by attribute sometimes not. Now always attribute. Change code accordingly.
    # TODO: remove correspondence_fuel_epm
    rename_columns = {'c': 'zone', 'country': 'zone', 'y': 'year', 'v': 'value', 's': 'scenario', 'uni': 'attribute',
                      'z': 'zone', 'g': 'generator', 'f': 'fuel', 'q': 'season', 'd': 'day', 't': 't'}

    keys = {'pDemandSupplyCountry', 'pDemandSupply', 'pEnergyByPlant', 'pEnergyByFuel', 'pCapacityByFuel', 'pCapacityPlan',
            'pPlantUtilization', 'pCostSummary', 'pCostSummaryCountry', 'pEmissions', 'pPrice', 'pHourlyFlow',
            'pDispatch', 'pFuelDispatch', 'pPlantFuelDispatch', 'pInterconUtilization', 'pReserveByPlant',
            'InterconUtilization', 'pInterchange', 'Interchange', 'interchanges', 'pInterconUtilizationExt',
            'InterconUtilizationExt', 'pInterchangeExt', 'InterchangeExt', 'annual_line_capa', 'pAnnualTransmissionCapacity',
            'AdditiononalCapacity_trans', 'pPlantReserve', 'pDemandSupplySeason', 'pCurtailedVRET', 'pCurtailedStoHY', 'pNewCapacityFuelCountry', 'pPlantAnnualLCOE'}

    # Rename columns
    epm_dict = {k: i.rename(columns=rename_columns) for k, i in epm_results.items() if k in keys and k in epm_results.keys()}

    if scenarios_rename is not None:
        for k, i in epm_dict.items():
            if 'scenario' in i.columns:
                i['scenario'] = i['scenario'].replace(scenarios_rename)

    list_keys = ['pReserveByPlant', 'pPlantReserve', 'pCapacityPlan']
    list_keys = [i for i in list_keys if i in epm_dict.keys()]
    epm_dict = remove_unused_tech(epm_dict, list_keys)

    #generation_mapping = read_input_data(input=input)

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

    standardize_names(epm_dict, 'pFuelDispatch', dict_specs['fuel_mapping'])
    standardize_names(epm_dict, 'pPlantFuelDispatch', dict_specs['tech_mapping'])

    # epm_dict['pPlantDispatch'].replace(generation_mapping, inplace=True)  # map generator to fuel

    #epm_dict['pReserveByPlant'].replace(generation_mapping, inplace=True)  # map generator to fuel
    #epm_dict['pReserveByPlant'] = epm_dict['pReserveByPlant'].rename(columns={'generator': 'fuel'}).groupby(['zone', 'year', 'scenario', 'fuel'], observed=False).sum().reset_index()
    #standardize_names('pReserveByPlant', dict_specs['fuel_mapping'])

    return epm_dict


def format_ax(ax, linewidth=True):
    """
    Format the given Matplotlib axis:
    - Removes the background.
    - Removes the grid.
    - Ensures the entire frame is displayed.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to format.
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


def line_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(10, 6)):
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
        plt.close()

    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return None


def bar_plot(df, x, y, xlabel=None, ylabel=None, title=None, filename=None, figsize=(8, 5), round_tot=0):
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


def subplot_pie(df, index, dict_colors, subplot_column, title='', figsize=(16, 4), percent_cap=1, filename=None):

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
            #autopct=lambda p: f'{p:.0f}',
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
        frameon=False, fontsize=16
    )

    # Add title for the whole figure
    fig.suptitle(title, fontsize=16)

    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def make_fuel_energy_mix_pie_plot(df, years, folder, dict_colors, BESS_included=False, Hydro_stor_included=False,
                                  figsize=(16, 4), percent_cap=6, selected_scenario=None):
    # TODO: add scenario grouping, currently only works when selected_scenario is not None
    if selected_scenario is not None:
        df = df[df['scenario'] == selected_scenario]
    if not BESS_included:
        df = df[df['fuel'] != 'Battery Storage']
    if not Hydro_stor_included:
        df = df[df['fuel'] != 'Pumped-Hydro Storage']

    df = df.loc[df['year'].isin(years)].groupby(['year', 'fuel']).agg({'value': 'sum'}).reset_index()
    df['value'] = df['value'].apply(lambda x: 0 if x < 0 else x)

    title = f'Energy generation mix - {selected_scenario} scenario'
    temp = '_'.join([str(y) for y in years])
    filename = f'{folder}/EnergyGenerationMixPie_{temp}_{selected_scenario}.png'

    subplot_pie(df, 'fuel', dict_colors, 'year', title=title, figsize=figsize,
                     percent_cap=percent_cap, filename=filename)


def make_fuel_capacity_mix_pie_plot(df, years, folder, dict_colors, figsize=(16, 4), percent_cap=6,
                                    selected_scenario=None):

    df = df.loc[df['year'].isin(years)].groupby(['year', 'fuel']).agg({'value': 'sum'}).reset_index()
    df['value'] = df['value'].apply(lambda x: 0 if x < 0 else x)

    title = f'Energy capacity mix - {selected_scenario} scenario'
    temp = '_'.join([str(y) for y in years])
    filename = f'{folder}/EnergyCapacityMixPie_{temp}_{selected_scenario}.png'

    subplot_pie(df, 'fuel', dict_colors, 'year', title=title, figsize=figsize,
                     percent_cap=percent_cap, filename=filename)


def stacked_area_plot(df, filename, dict_colors=None, x_column='year', y_column='value', stack_column='fuel',
                      df_2=None, title='', x_label='Years', y_label='',
                      legend_title='', y2_label='', figsize=(10, 6)):
    """
    Generate a stacked area chart with a secondary y-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        filename (str): Path to save the plot
        dict_colors (dict): Dictionary mapping fuel types to colors, optional
        x_column (str): Column for x-axis, default is 'year'
        y_column (str): Column for y-axis, default is 'value'
        stack_column (str): Column for stacking, default is 'fuel'
        legend_title (str): Title for the legend
        df_2 (pd.DataFrame): DataFrame containing data for the secondary y-axis, optional
        title (str): Title of the plot
        x_label (str): Label for the x-axis
        y_label (str): Label for the primary y-axis
        y2_label (str): Label for the secondary y-axis
        figsize (tuple): Size of the figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot stacked area for generation
    temp = df.groupby([x_column, stack_column])[y_column].sum().unstack(stack_column)
    temp.plot.area(ax=ax1, stacked=True, alpha=0.8, color=dict_colors)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    format_ax(ax1)

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


def dispatch_plot(df_area, filename, dict_colors=None, df_line=None, figsize=(10, 6)):
    """

    :param df_area: pd.DataFrame
        Data to be displayed as stacked areas
    :param filename:
    :param dict_colors:
    :param df_line: pd.DataFrame
        Optional, data to be displayed as a line (e.g., demand)
    :param figsize:
    :return:
    """

    if df_line is not None:
        assert df_area.index.equals(df_line.index), 'Dataframes used for area and line do not share the same index. Update the input dataframes.'
    fig, ax = plt.subplots(figsize=figsize)
    df_area.plot.area(ax=ax, stacked=True, color=dict_colors, linewidth=0)
    if df_line is not None:
        df_line.plot(ax=ax, color=dict_colors)

    # Adding the representative days and seasons
    n_rep_days = len(df_area.index.get_level_values('day').unique())
    dispatch_seasons = df_area.index.get_level_values('season').unique()
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

    # Add axis labels and title
    ax.set_xlabel('Hours')
    ax.set_ylabel('Generation (MWh)', fontweight='bold')
    # ax.text(0, 1.2, f'Dispatch', fontsize=9, fontweight='bold', transform=ax.transAxes)

    # Add legend bottom center
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df_area.columns), frameon=False)

    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plt.close(fig)


def make_fuel_dispatch_plot(pFuelDispatch, folder, dict_colors, zone, year, scenario, column_stacked='fuel',
                            fuel_grouping=None, select_time=None):
    """Returns fuel dispatch plot, including only generation plants.
    fuel_grouping: dict
        A mapping to create aggregate fuel categories. E.g., {'Battery Storage 4h': 'Battery Storage'}
    select_time: dict

    """
    df = pFuelDispatch
    df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
    df = df.drop(columns=['zone', 'year', 'scenario'])

    if fuel_grouping is not None:
        df['fuel'] = df['fuel'].replace(
            fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

    df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())

    if select_time is not None:
        df, temp = select_time_period(df, select_time)

    df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    df = df.droplevel(0, axis=1)

    df = df.where((df > 1e-6) | (df < -1e-6), np.nan)  # get rid of small values to avoid unneeded labels
    df = df.dropna(axis=1, how='all')

    if select_time is None:
        temp = 'all'
    temp = f'{year}_{temp}'

    filename = f'{folder}/FuelDispatch_{scenario}_{temp}.png'
    dispatch_plot(df, filename, dict_colors)


def select_time_period(df, select_time):
    """Select a specific time period in a dataframe.
    df: pd.DataFrame
        Columns contain season and day
    select_time: dict
        For each key, specifies a subset of the dataframe
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
    Args:
        df (pd.DataFrame): Input dataframe containing the results.
        zone (str): The zone to filter the data for.
        year (int): The year to filter the data for.
        scenario (str): The scenario to filter the data for.
        column_stacked (str): Column to use for stacking values in the transformed dataframe.
        fuel_grouping (dict): A dictionary mapping fuels to their respective groups and to sum values over those groups.
        select_time (dict or None): Specific time filter to apply (e.g., "summer").

    Returns:
        pd.DataFrame: A transformed dataframe with multi-level index (season, day, time).
    Example:
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

    df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())

    if select_time is not None:
        df, temp = select_time_period(df, select_time)
    else:
        temp = None

    df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    return df, temp


def remove_na_values(df):
    """Removes na values from a dataframe, to avoind unnecessary labels in plots."""
    df = df.where((df > 1e-6) | (df < -1e-6),
                                    np.nan)
    df = df.dropna(axis=1, how='all')
    return df


def make_complete_fuel_dispatch_plot(dfs_area, dfs_line, folder, dict_colors, zone, year, scenario,
                                fuel_grouping=None, select_time=None, dfs_line_2=None, reorder_dispatch=None, season=True):
    """
    Generates and saves a fuel dispatch plot, including only generation plants.

    Args:
        dfs_area (dict):
            Dictionary containing dataframes for area plots.
        dfs_line (dict):
            Dictionary containing dataframes for line plots.
        graph_folder (str):
            Path to the folder where the plot will be saved.
        dict_colors (dict):
            Dictionary mapping fuel types to colors.
        fuel_grouping (dict):
            Mapping to create aggregate fuel categories, e.g.,
            {'Battery Storage 4h': 'Battery Storage'}.
        select_time (dict):
            Time selection parameters for filtering the data.
        dfs_line_2 (dict, optional):
            Optional dictionary containing dataframes for a secondary line plot.

    Returns:
        None

    Example:
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
        column_stacked = NAME_COLUMNS[key]
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]
        column_stacked = NAME_COLUMNS[key]
        df, temp = clean_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_line.append(df)

    df_tot_area = pd.concat(tmp_concat_area, axis=1)
    df_tot_area = df_tot_area.droplevel(0, axis=1)
    df_tot_area = remove_na_values(df_tot_area)

    df_tot_line = pd.concat(tmp_concat_line, axis=1)
    df_tot_line = df_tot_line.droplevel(0, axis=1)
    df_tot_line = remove_na_values(df_tot_line)

    if reorder_dispatch is not None:
        new_order = [col for col in reorder_dispatch if col in df_tot_area.columns] + [col for col in df_tot_area.columns if col not in reorder_dispatch]
        df_tot_area = df_tot_area[new_order]

    if select_time is None:
        temp = 'all'
    temp = f'{year}_{temp}'
    filename = f'{folder}/Dispatch_{scenario}_{temp}.png'
    dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors)


def stacked_bar_subplot(df, column_group, filename, dict_colors=None, figsize=(10, 6), year_ini=None,
                        order_scenarios=None, order_columns=None,       dict_scenarios=None, rotation=0, fonttick=14, legend=True,
                        format_y=lambda y, _: '{:.0f} GW'.format(y), cap=6, annotate=True):
    list_keys = list(df.columns)
    n_columns = int(len(list_keys))
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_group][0]).unique()
    n_rows = 1
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})

    # Ensure axes is iterable
    if n_rows == 1 and n_columns == 1:
        axes = [axes]

    handles, labels = None, None
    for k in range(n_rows * n_columns):
        column = k % n_columns
        ax = axes[column]

        try:
            key = list_keys[k]
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
                                f"{height:.0f}",  # Annotation text (formatted value)
                                ha="center", va="center",  # Center align the text
                                fontsize=10, color="black"  # Font size and color
                            )

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which=u'both', length=0)
            ax.set_xlabel('')

            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            if k > 0:
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

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
    plt.close(fig)


def make_stacked_bar_subplots(df, filename, dict_colors, selected_zone=None, column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None, dict_scenarios=None,
                              format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2):
    """
    Subplots with stacked bars. Can be used to explore the evolution of capacity over time and across scenarios.
    Args:
        df (pd.DataFrame): Dataframe with results
        filename (path): Path to save the figure
        dict_colors (dict): Dictionary with color arguments
        zone (str): Zone to select
        column_xaxis (str): Column for choosing the subplots
        column_stacked (str): Column name for choosing the column to stack values
        column_multiple_bars (str): Column for choosing the type of bars inside a given subplot.
        select_xaxis (str) Select a subset of subplots (for eg, a number of years)
        dict_grouping (dict) Dictionary for grouping variables and summing over a given group
        order_scenarios:
        dict_scenarios:
        format_y: Formatting y axis
        order_stacked (list): Reordering the variables that will be stacked
        cap (int): Under this cap, no annotation will be displayed

    Example:
        Stacked bar subplots for capacity (by fuel) evolution
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
        make_stacked_bar_subplots(epm_dict['pCapacityByFuel'], filename, dict_specs['colors'], zone='Liberia',
                   select_stacked=[2025, 2028, 2030], fuel_grouping=fuel_grouping, dict_scenarios=scenario_names,
                   order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                   format_y=lambda y, _: '{:.0f} MW'.format(y)
                   )

        Stacked bar subplots for reserve evolution
        filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
        make_stacked_bar_subplots(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], zone='Liberia',
                                  column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario',
                                  select_xaxis=[2025, 2028, 2030], dict_grouping=dict_grouping, dict_scenarios=scenario_names,
                                  order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                                  format_y=lambda y, _: '{:.0f} GWh'.format(y),
                                  order_stacked=['Hydro', 'Oil'], cap=2
                                   )
    """
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])

    if dict_grouping is not None:
        for key, grouping in dict_grouping.items():
            assert key in df.columns, f'Grouping parameter with key {key} is used but {key} is not in the columns.'
            df[key] = df[key].replace(grouping)  # case-specific, according to level of preciseness for dispatch plot

    df = (df.groupby([column_xaxis, column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())

    df = df.set_index([column_stacked, column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    stacked_bar_subplot(df, column_stacked, filename, dict_colors, format_y=format_y,
                        rotation=90, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                        order_columns=order_stacked, cap=cap)




if __name__ == '__main__':
    print(0)
