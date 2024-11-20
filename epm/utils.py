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

NAME_COLUMNS = {
    'pFuelDispatch': 'fuel',
    'pDispatch': 'attribute'
}


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
        if 'attribute' in i.columns:
            epm_dict[k] = epm_dict[k].astype({'attribute': 'str'})

    # Standardize names
    def standardize_names(key, mapping):
        if key in epm_dict.keys():
            epm_dict[key].replace(mapping, inplace=True)
            epm_dict[key].groupby(
                [i for i in epm_dict[key].columns if i != 'value']).sum().reset_index()

            new_fuels = [f for f in epm_dict[key]['fuel'].unique() if f not in mapping.values()]
            if new_fuels:
                raise ValueError(f'New fuels found in {key}: {new_fuels}. '
                                 f'Add fuels to the mapping in the /static folder and add in the colors.csv file.')
        else:
            print(f'{key} not found in epm_dict')

    standardize_names('pEnergyByFuel', dict_specs['fuel_mapping'])
    standardize_names('pCapacityByFuel', dict_specs['fuel_mapping'])
    standardize_names('pFuelDispatch', dict_specs['fuel_mapping'])
    standardize_names('pPlantFuelDispatch', dict_specs['tech_mapping'])

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

    plt.show()
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
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Demand {unit}',
                       title=f'Total demand - {selected_scenario} scenario',
                       filename=f'{folder}/TotalDemand_{plot_option}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
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
        line_plot(df_tot, 'year', 'value',
                       xlabel='Years',
                       ylabel=f'Generation {unit}',
                       title=f'Total generation - {selected_scenario} scenario',
                       filename=f'{folder}/TotalGeneration_{plot_option}.png')
    elif plot_option == 'bar':
        bar_plot(df_tot, 'year', 'value',
                      xlabel='Years',
                      ylabel=f'Generation {unit}',
                      title=f'Total generation - {selected_scenario} scenario',
                      filename=f'{folder}/TotalGeneration_{plot_option}.png')
    else:
        raise ValueError('Invalid plot_option argument. Choose between "line" and "bar"')


def subplot_pie(df, index, dict_colors, subplot_column, title='', figsize=(16, 4), percent_cap=6, filename=None):

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
    else:
        plt.tight_layout()
        plt.show()


def make_fuel_energy_mix_pie_plot(df, years, graphs_folder, dict_colors, BESS_included=False, Hydro_stor_included=False,
                                  figsize=(16, 4), percent_cap=6, selected_scenario=None):
    if not BESS_included:
        df = df[df['fuel'] != 'Battery Storage']
    if not Hydro_stor_included:
        df = df[df['fuel'] != 'Pumped-Hydro Storage']

    df = df.loc[df['year'].isin(years)].groupby(['year', 'fuel']).agg({'value': 'sum'}).reset_index()
    df['value'] = df['value'].apply(lambda x: 0 if x < 0 else x)

    title = f'Energy generation mix - {selected_scenario} scenario'
    temp = '_'.join([str(y) for y in years])
    filename = f'{graphs_folder}/EnergyGenerationMixPie_{temp}_{selected_scenario}.png'

    subplot_pie(df, 'fuel', dict_colors, 'year', title=title, figsize=figsize,
                     percent_cap=percent_cap, filename=filename)


def make_fuel_capacity_mix_pie_plot(df, years, graphs_folder, dict_colors, figsize=(16, 4), percent_cap=6,
                                    selected_scenario=None):

    df = df.loc[df['year'].isin(years)].groupby(['year', 'fuel']).agg({'value': 'sum'}).reset_index()
    df['value'] = df['value'].apply(lambda x: 0 if x < 0 else x)

    title = f'Energy capacity mix - {selected_scenario} scenario'
    temp = '_'.join([str(y) for y in years])
    filename = f'{graphs_folder}/EnergyCapacityMixPie_{temp}_{selected_scenario}.png'

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
    else:
        plt.tight_layout()
        plt.show()


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
    ax.set_ylabel('Generation (GWh)', fontweight='bold')
    ax.text(0, 1.2, f'Dispatch', fontsize=9, fontweight='bold', transform=ax.transAxes)

    # Add legend bottom center
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(df_area.columns), frameon=False)

    # Remove grid
    ax.grid(False)
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def make_fuel_dispatch_plot(pFuelDispatch, graph_folder, dict_colors, zone, year, scenario, column_stacked='fuel',
                            selected_scenario=None, fuel_grouping=None, select_time=None):
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
        if 'season' in select_time.keys():
            df = df.loc[df.season.isin(select_time['season'])]
        if 'day' in select_time.keys():
            df = df.loc[df.day.isin(select_time['day'])]

    df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
    df = df.droplevel(0, axis=1)

    df = df.where((df > 1e-6) | (df < -1e-6), np.nan)  # get rid of small values to avoid unneeded labels
    df = df.dropna(axis=1, how='all')

    filename = f'{graph_folder}/FuelDispatch_{selected_scenario}.png'
    dispatch_plot(df, filename, dict_colors)


def make_dispatch_plot_complete(dfs_area, dfs_line, graph_folder, dict_colors, zone, year, scenario,
                                selected_scenario=None, fuel_grouping=None, select_time=None):
    """
    Returns complete dispatch plot, with option to customize which data to include
    :param dfs_area: dict of DataFrame
        Dictionary with all the dataframe to be displayed as stacked areas
    :param dfs_line: dict of DataFrame
        Dictionary with all the dataframe to be displayed as lines
    :param graph_folder:
    :param dict_colors:
    :param zone:
    :param year:
    :param scenario:
    :param selected_scenario:
    :param fuel_grouping:
    :param select_time:
    :return:
    """
    tmp_concat_area = []
    for key in dfs_area:
        df = dfs_area[key]
        column_stacked = NAME_COLUMNS[key]
        df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['zone', 'year', 'scenario'])

        if column_stacked == 'fuel':
            if fuel_grouping is not None:
                df['fuel'] = df['fuel'].replace(
                    fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

        df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())

        if select_time is not None:
            if 'season' in select_time.keys():
                df = df.loc[df.season.isin(select_time['season'])]
            if 'day' in select_time.keys():
                df = df.loc[df.day.isin(select_time['day'])]

        df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]
        column_stacked = NAME_COLUMNS[key]
        df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
        df = df.drop(columns=['zone', 'year', 'scenario'])

        if column_stacked == 'fuel':
            if fuel_grouping is not None:
                df['fuel'] = df['fuel'].replace(
                    fuel_grouping)  # case-specific, according to level of preciseness for dispatch plot

        df = (df.groupby(['season', 'day', 't', column_stacked], observed=False).sum().reset_index())

        if select_time is not None:
            if 'season' in select_time.keys():
                df = df.loc[df.season.isin(select_time['season'])]
            if 'day' in select_time.keys():
                df = df.loc[df.day.isin(select_time['day'])]
        df = df.set_index(['season', 'day', 't', column_stacked]).unstack(column_stacked)
        tmp_concat_line.append(df)

    df_tot_area = pd.concat(tmp_concat_area, axis=1)
    df_tot_area = df_tot_area.droplevel(0, axis=1)

    df_tot_area = df_tot_area.where((df_tot_area > 1e-6) | (df_tot_area < -1e-6),
                                    np.nan)  # get rid of small values to avoid unneeded labels
    df_tot_area = df_tot_area.dropna(axis=1, how='all')

    df_tot_line = pd.concat(tmp_concat_line, axis=1)
    df_tot_line = df_tot_line.droplevel(0, axis=1)

    df_tot_line = df_tot_line.where((df_tot_line > 1e-6) | (df_tot_line < -1e-6),
                                    np.nan)  # get rid of small values to avoid unneeded labels
    df_tot_line = df_tot_line.dropna(axis=1, how='all')

    filename = f'{graph_folder}/Dispatch_{selected_scenario}.png'
    dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors)


def cluster_stackedbar_plot(df, group_column, colors=None, rotation=0, year_ini=None, order_scenarios=None,
                                filename=None, fonttick=14, ymin=0, legend=True, figtitle=None, ymax=None,
                                display_total=False, figsize=(12.8, 9.6)):
    # TODO: Adapt to EPM output
    # TODO: Add ymax and ymin

    list_keys = list(df.columns)
    if ymax is None:
        temp = df.copy()
        temp[temp < 0] = 0
        ymax = temp.groupby([i for i in temp.index.names if i != group_column]).sum().max().max() * 1.1

    n_columns = int(len(list_keys))
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != group_column][0]).unique()
    n_rows = 1
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, sharey='all',
                             gridspec_kw={'width_ratios': width_ratios})
    handles, labels = None, None
    for k in range(n_rows * n_columns):

        column = k % n_columns
        ax = axes[column]

        try:
            key = list_keys[k]
            df_temp = df[key].unstack(group_column)

            if key == year_ini:
                df_temp = df_temp.iloc[0, :]
                df_temp = df_temp.to_frame().T
                df_temp.index = ['Initial']
            else:
                if order_scenarios is not None:
                    df_temp = df_temp.loc[order_scenarios, :]

            df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=colors if colors is not None else None)

            if display_total:
                for i, (index, row) in enumerate(df_temp.iterrows()):
                    total = row.sum()
                    # Format the number as an integer without decimals
                    ax.annotate(f'{int(total)}€', (i, total), ha='center', va='bottom', fontsize=fonttick)
                    ax.plot(i, total, marker='d', color='black', markersize=5)

            ax.spines['left'].set_visible(False)
            ax.set_xlabel('')

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            # put tick label in bold
            ax.tick_params(axis='both', which='major', labelsize=fonttick)

            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()

        except IndexError:
            ax.axis('off')

    if figtitle is not None:
        fig.suptitle(figtitle, x=0.5, y=1.05, weight='bold', color='black', size=20)

    if legend:
        fig.legend(handles[::-1], labels[::-1], loc='center left', frameon=False, ncol=1,
                   bbox_to_anchor=(1, 0.5))

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    print(0)
