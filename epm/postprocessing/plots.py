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
from matplotlib.ticker import FuncFormatter

from .utils import NAME_COLUMNS, RENAME_COLUMNS

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


def make_auto_formatter(unit=""):
    def _format(y, _):
        # percentages handled separately
        if unit == "%":
            y = y * 100
            txt = f"{y:.2f}".rstrip('0').rstrip('.')
            return f"{txt}%"

        # other units: GW, MW, GWh, etc.
        # show fewer decimals cleanly
        if abs(y) >= 100:
            txt = f"{y:,.0f}"
        elif abs(y) >= 1:
            txt = f"{y:,.1f}"
        else:
            txt = f"{y:,.2f}"
        # txt = txt.rstrip('0').rstrip('.')
        return f"{txt} {unit}".strip()

    return _format


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


def make_demand_plot(pEnergyBalanceCountry, folder, years=None, plot_option='bar', selected_scenario=None, unit='MWh'):
    """
    Depreciated. Makes a plot of demand for all countries.
    
    Parameters:
    ----------
    pEnergyBalanceCountry: pd.DataFrame
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

    df_tot = pEnergyBalanceCountry.loc[pEnergyBalanceCountry['attribute'] == 'Demand: GWh']
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


def make_generation_plot(pEnergyFuel, folder, years=None, plot_option='bar', selected_scenario=None, unit='GWh',
                         BESS_included=True, Hydro_stor_included=True):
    """
    Makes a plot of demand for all countries.

    Parameters:
    ----------
    pEnergyBalanceCountry: pd.DataFrame
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
        pEnergyFuel = pEnergyFuel[pEnergyFuel['scenario'] == selected_scenario]

    if not BESS_included:
        pEnergyFuel = pEnergyFuel[pEnergyFuel['fuel'] != 'Battery Storage']

    if not Hydro_stor_included:
        pEnergyFuel = pEnergyFuel[pEnergyFuel['fuel'] != 'Pumped-Hydro Storage']

    df_tot = pEnergyFuel.groupby('year').agg({'value': 'sum'}).reset_index()

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


def stacked_area_plot(df, filename, dict_colors=None, column_xaxis='year', column_value='value', column_stacked='fuel',
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
    column_xaxis : str, default 'year'
        Column for x-axis.
    column_value : str, default 'value'
        Column for y-axis.
    column_stacked : str, default 'fuel'
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
        sorting_column = df.groupby(column_stacked)[sorting_column].first().sort_values().index

    # Plot stacked area for generation
    temp = df.groupby([column_xaxis, column_stacked])[column_value].sum().unstack(column_stacked)

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

        temp = df_2.groupby([column_xaxis])[column_value].sum()
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


def stacked_area_subplots(df, column_stacked, filename, dict_colors=None, order_scenarios=None, order_stacked=None,
                        dict_scenarios=None, rotation=0, fonttick=14, legend=True, format_y=lambda y, _: '{:.0f} GW'.format(y),
                        title=None, figsize=(10,6)):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_stacked : str
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
    order_stacked : list, optional
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
            df_temp = df[key].unstack(column_stacked)

            if order_stacked is not None:
                df_temp = df_temp[order_stacked]

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


def make_annotated_stacked_area_plot(df, filename, dict_colors=None, column_xaxis='year', column_value='value',
                                     column_stacked='fuel', annotate_column='generator'):
    
    df.sort_values(column_stacked, inplace=True)
    # complete year with 0 capacity when no data
    years = df[column_xaxis].unique()
    
    # For each group in the DataFrame, calculate year-over-year differences of the target column,
    # keep values greater than 1, format them with the group label, and merge into a dictionary 
    # (aggregating multiple group labels if they share the same year).
    annotate_dict = {}
    for n, g in df.groupby([annotate_column]):
        g.set_index(column_xaxis, inplace=True)
        g = g.loc[:, column_value]
        g = g.reindex(years, fill_value=0)
        g.sort_index(inplace=True)
        g = g.diff()
        g = g[g > 1].to_dict()
        g = {k: '{} - {:.0f}'.format(n[0], i) for k, i in g.items()}
        # if k in result.keys() add values to the existing dictionary
        for k, i in g.items():
            if k in annotate_dict.keys():
                annotate_dict[k] += '\n' + i
            else:
                annotate_dict[k] = i

    stacked_area_plot(df, filename, dict_colors, column_xaxis='year', column_value='value', column_stacked='fuel',
                      annotate=annotate_dict)


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

    if df_area is not None and not df_area.empty:
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
        'pDispatchFuel': epm_dict['pDispatchFuel'],
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
        filename = filename.split('.')[0] + f'_{temp}.pdf'

    dispatch_plot(df_tot_area, filename, df_line=df_tot_line, dict_colors=dict_colors, legend_loc=legend_loc, bottom=bottom,
                  figsize=figsize, ylabel=ylabel, title=title)


def stacked_bar_subplot(df, column_stacked, filename, df_errorbars=None, dict_colors=None, year_ini=None, order_scenarios=None,
                        order_stacked=None, dict_scenarios=None, rotation=0, fonttick=14, legend=True, format_y=lambda y, _: '{:.0f} GW'.format(y),
                        cap=6, annotate=True, show_total=False, title=None, figsize=(10,6), fontsize_label=10,
                        format_label="{:.1f}", hspace=0.4, cols_per_row=3, juxtaposed=False, bar_annotations=None,
                        annotation_pad=0.02, annotation_joiner='\n'):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot. Index may be multiple levels. First level corresponds to x axis, second level corresponds to stacked values.
        Columns of df correspond to subplots.
    column_stacked : str
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
    order_stacked : list, optional
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
    bar_annotations : dict, optional
        Nested dictionary keyed by subplot (column of df) and bar label mapping to an iterable of strings (e.g. power plants)
        to display above each bar. Default is None.
    annotation_pad : float, optional
        Relative padding (as a fraction of the maximum bar height) to place annotations above bars. Default is 0.02.
    annotation_joiner : str, optional
        String used to join items when the annotation is provided as an iterable. Default is a newline.
    Returns
    -------
    None
    """

    list_keys = list(df.columns)
    n_scenario = df.index.get_level_values([i for i in df.index.names if i != column_stacked][0]).unique()
    num_subplots = int(len(list_keys))
    n_columns = min(cols_per_row, num_subplots)  # Limit to 3 columns per row
    n_rows = int(np.ceil(num_subplots / n_columns))
    if year_ini is not None:
        width_ratios = [1] + [len(n_scenario)] * (n_columns - 1)
    else:
        width_ratios = [1] * n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(figsize[0], figsize[1]*n_rows), sharey='all',
                             gridspec_kw={'width_ratios': width_ratios, 'hspace': hspace})
    if n_rows * n_columns == 1:  # If only one subplot, `axes` is not an array
        axes = [axes]  # Convert to list to maintain indexing consistency
    else:
        axes = np.array(axes).flatten()  # Ensure it's always a 1D array

    # Add figure title 
    if len(list_keys) > 1:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    # should we use a stacked bar plot or not
    stacked = True
    if column_stacked is None:
        stacked = False

    handles, labels = None, None
    bar_annotations = bar_annotations or {}
    for k, key in enumerate(list_keys):
        ax = axes[k]

        try:
            df_temp = df[key].unstack(column_stacked) if column_stacked else df[key].to_frame()

            annotations_for_subplot = None
            if bar_annotations:
                if key in bar_annotations:
                    annotations_for_subplot = bar_annotations[key]
                elif len(list_keys) == 1 and bar_annotations:
                    annotations_for_subplot = next(iter(bar_annotations.values()))
                elif isinstance(key, tuple):
                    annotations_for_subplot = bar_annotations.get(tuple(key))

            if key == year_ini:
                df_temp = df_temp.iloc[0, :]
                df_temp = df_temp.to_frame().T
                df_temp.index = ['Initial']
            else:
                if dict_scenarios is not None:  # Renaming scenarios for plots
                    df_temp.index = df_temp.index.map(lambda x: dict_scenarios.get(x, x))
                    if annotations_for_subplot is not None:
                        annotations_for_subplot = {dict_scenarios.get(k, k): v for k, v in annotations_for_subplot.items()}
                if order_scenarios is not None:  # Reordering scenarios
                    df_temp = df_temp.loc[[c for c in order_scenarios if c in df_temp.index], :]
                if order_stacked is not None:
                    new_order = [c for c in order_stacked if c in df_temp.columns] + [c for c in df_temp.columns if c not in order_stacked]
                    df_temp = df_temp.loc[:,new_order]

            if not juxtaposed:
                df_temp.plot(ax=ax, kind='bar', stacked=stacked, linewidth=0,
                            color=dict_colors if dict_colors else None)
            else:  # stacked columns become one next to each other
                df_temp.T.plot(ax=ax, kind='bar', stacked=False, linewidth=0,
                            color=dict_colors if dict_colors else None)

            # Plot error bars if provided
            df_bar_totals = df_temp.sum(axis=1)

            if df_errorbars is not None:
                if not juxtaposed:
                    df_errorbars_temp = df_errorbars[key].unstack('error')
                    df_err_low = df_errorbars_temp['min'].reindex(df_temp.index)
                    df_err_high = df_errorbars_temp['max'].reindex(df_temp.index)

                    for i, idx in enumerate(df_temp.index):
                        x = i  # bar positions correspond to index in this order
                        height = df_bar_totals.loc[idx]
                        low = df_err_low.loc[idx] if pd.notna(df_err_low.loc[idx]) else height
                        high = df_err_high.loc[idx] if pd.notna(df_err_high.loc[idx]) else height
                        err_low = max(height - low, 0)
                        err_high = max(high - height, 0)
                        ax.errorbar(x, height, yerr=[[err_low], [err_high]], fmt='none',
                                    color='black', capsize=3, linewidth=1)

                else:
                    # New method
                    df_err_low = df_errorbars[key].unstack('error')['min']
                    df_err_high = df_errorbars[key].unstack('error')['max']

                    df_plot = df_temp.T  # rows: attribute, columns: scenario

                    # Build (scenario, attribute) -> x position from actual bar patches
                    bar_positions = {}
                    attr_list = list(df_plot.index)

                    # Loop through all containers and all bars inside
                    for container in ax.containers:
                        label = container.get_label()
                        if label == "_nolegend_":
                            continue

                        for i, bar in enumerate(container):
                            x = bar.get_x() + bar.get_width() / 2
                            if i < len(attr_list):
                                attr = attr_list[i]
                                bar_positions[(label, attr)] = x

                            # Try to infer which (scenario, attribute) this bar corresponds to
                            # for attr in df_plot.index:
                            #     expected_height = df_plot.loc[attr, label]
                            #     if pd.notna(expected_height) and np.isclose(expected_height, height):
                            #         bar_positions[(label, attr)] = x

                    # Now plot error bars for all expected combinations
                    for scenario in df_temp.index:
                        for attr in df_temp.columns:
                            height = df_temp.loc[scenario, attr]
                            height = 0 if pd.isna(height) else height
                            low = df_err_low.get((attr, scenario), np.nan)
                            high = df_err_high.get((attr, scenario), np.nan)

                            if pd.notna(low) and pd.notna(high):
                                err_low = max(height - low, 0)
                                err_high = max(high - height, 0)

                                # Real position if bar was drawn, else estimate
                                x = bar_positions.get((scenario, attr))
                                if x is None:
                                    i = list(df_temp.index).index(scenario)
                                    j = list(df_temp.columns).index(attr)
                                    x = i + j / (len(df_temp.columns) + 1)

                                ax.errorbar(x, height, yerr=[[err_low], [err_high]], fmt='none',
                                            color='black', capsize=3, linewidth=1)

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
                if isinstance(show_total, list):
                    df_total = df_temp.loc[:, show_total].sum(axis=1)
                else:
                    df_total = df_temp.sum(axis=1)

                x_positions = ax.get_xticks()
                if len(x_positions) < len(df_total):
                    x_positions = np.arange(len(df_total))
                else:
                    x_positions = x_positions[:len(df_total)]

                for (label, total), x in zip(df_total.items(), x_positions):
                    if pd.isna(total):
                        continue
                    ax.text(
                        x,
                        total * (1 + 0.02),
                        f"{total:,.0f}",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black',
                        fontweight='bold'
                    )
                ax.scatter(x_positions, df_total.values, color='black', s=20)

            if annotations_for_subplot and not juxtaposed:
                max_height = df_bar_totals.max() if not df_bar_totals.empty else 0
                pad_value = max_height * annotation_pad if max_height else annotation_pad
                xtick_positions = ax.get_xticks()
                for i, idx in enumerate(df_temp.index):
                    raw_text = annotations_for_subplot.get(idx)
                    if not raw_text:
                        continue
                    if isinstance(raw_text, str):
                        text = raw_text
                    else:
                        text = annotation_joiner.join([str(item) for item in raw_text if str(item)])
                    if not text:
                        continue
                    bar_height = df_bar_totals.loc[idx] if idx in df_bar_totals.index else None
                    if bar_height is None or pd.isna(bar_height):
                        continue
                    x_coord = xtick_positions[i] if i < len(xtick_positions) else i
                    y_coord = bar_height + pad_value
                    ax.text(x_coord, y_coord, text, ha='center', va='bottom', fontsize=fontsize_label, color='black')

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


def make_stacked_bar_subplots(df, filename, dict_colors, df_errorbars=None, selected_zone=None, selected_year=None, column_xaxis='year',
                              column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, dict_grouping=None, order_scenarios=None, dict_scenarios=None,
                              format_y=lambda y, _: '{:.0f} MW'.format(y), order_stacked=None, cap=2, annotate=True,
                              show_total=False, fonttick=12, rotation=0, title=None, fontsize_label=10,
                              format_label="{:.1f}", figsize=(10,6), hspace=0.4, cols_per_row=3, juxtaposed=False, year_ini=None,
                              column_annotation=None, annotation_pad=0.02, annotation_joiner='\n'):
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
    column_annotation : str, optional
        Column name containing categorical labels (e.g. plant names) to display above each bar.
    annotation_pad : float, optional
        Relative padding (fraction of bar height) for positioning annotations above bars.
    annotation_joiner : str, optional
        Separator used to join multiple annotation labels for a bar.

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
    make_stacked_bar_subplots(epm_dict['pCapacityFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
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
    if column_multiple_bars is None:
        print('column_multiple_bars cannot be None, but column_xaxis can. Automatically inverting.')
        column_multiple_bars = column_xaxis
        column_xaxis = None
    
    if selected_zone is not None:
        df = df[(df['zone'] == selected_zone)]
        df = df.drop(columns=['zone'])
        if df_errorbars is not None:
            df_errorbars = df_errorbars[(df_errorbars['zone'] == selected_zone)]
            df_errorbars = df_errorbars.drop(columns=['zone'])

    if selected_year is not None:
        df = df[(df['year'] == selected_year)]
        df = df.drop(columns=['year'])
        if df_errorbars is not None:
            df_errorbars = df_errorbars[(df_errorbars['year'] == selected_year)]
            df_errorbars = df_errorbars.drop(columns=['year'])

    if dict_grouping is not None:
        for key, grouping in dict_grouping.items():
            assert key in df.columns, f'Grouping parameter with key {key} is used but {key} is not in the columns.'
            df[key] = df[key].replace(grouping)  # case-specific, according to level of preciseness for dispatch plot

    bar_annotations = None
    if column_annotation is not None:
        if column_annotation not in df.columns:
            raise ValueError(f"column_annotation '{column_annotation}' not found in DataFrame columns.")

        annotation_df = df.copy()
        if column_value in annotation_df.columns:
            annotation_df = annotation_df[annotation_df[column_value] > 0]

        grouping_keys = []
        if column_xaxis is not None:
            grouping_keys.append(column_xaxis)
        if column_multiple_bars is not None:
            grouping_keys.append(column_multiple_bars)

        if not grouping_keys:
            grouping_keys = [column_annotation]

        annotation_df = annotation_df.dropna(subset=[column_annotation])

        def _collect_annotations(group):
            if column_annotation not in group.columns:
                return []
            if column_value in group.columns:
                totals = (group.groupby(column_annotation, observed=False)[column_value]
                          .sum()
                          .loc[lambda s: s > 0])
                if not totals.empty:
                    return totals.sort_values(ascending=False).index.tolist()
            values = group[column_annotation].tolist()
            unique_values = []
            for val in values:
                if val not in unique_values and val != "":
                    unique_values.append(val)
            return unique_values

        grouped_annotations = annotation_df.groupby(grouping_keys, observed=False).apply(_collect_annotations)

        if column_xaxis is not None:
            bar_annotations = {}
            for idx, plant_list in grouped_annotations.items():
                if not plant_list:
                    continue
                if isinstance(idx, tuple):
                    subplot_key = idx[0]
                    bar_key = idx[1] if len(idx) > 1 else None
                else:
                    subplot_key = idx
                    bar_key = None
                if len(grouping_keys) == 2:
                    subplot_key, bar_key = idx
                elif bar_key is None and column_multiple_bars is not None:
                    bar_key = idx
                if bar_key is None:
                    continue
                bar_annotations.setdefault(subplot_key, {})[bar_key] = plant_list
        else:
            bar_annotations = {'__single__': {}}
            for bar_key, plant_list in grouped_annotations.items():
                if not plant_list:
                    continue
                bar_annotations['__single__'][bar_key] = plant_list

    if column_xaxis is not None:
        if column_stacked is not None:
            df = (df.groupby([column_xaxis, column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_stacked, column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)
        else:
            df = (df.groupby([column_xaxis, column_multiple_bars], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)
        if df_errorbars is not None:
            if not juxtaposed:  # we sum over the stacked column
                df_errorbars = (df_errorbars.groupby([column_xaxis, 'error', column_multiple_bars], observed=False)[
                          column_value].sum().reset_index())
                df_errorbars = df_errorbars.set_index(['error', column_multiple_bars, column_xaxis]).squeeze().unstack(column_xaxis)
            else:  # we keep the stacked column
                df_errorbars = (df_errorbars.groupby([column_xaxis, 'error', column_stacked, column_multiple_bars], observed=False)[
                          column_value].sum().reset_index())
                df_errorbars = df_errorbars.set_index(['error', column_stacked, column_multiple_bars, column_xaxis]).squeeze().unstack(
                    column_xaxis)

    else:  # no subplots in this case
        if column_stacked is not None:
            df = (df.groupby([column_stacked, column_multiple_bars], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_stacked, column_multiple_bars])
        else:
            df = (df.groupby([column_multiple_bars], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_multiple_bars])
        if df_errorbars is not None:
            df_errorbars = (df_errorbars.groupby(['error', column_multiple_bars], observed=False)[column_value].sum().reset_index())
            df_errorbars = df_errorbars.set_index(['error', column_multiple_bars])

    if select_xaxis is not None:
        df = df.loc[:, [i for i in df.columns if i in select_xaxis]]

    if bar_annotations is not None:
        if column_xaxis is None:
            fallback = bar_annotations.get('__single__', {})
            if len(df.columns) == 1:
                bar_annotations = {df.columns[0]: fallback}
            else:
                bar_annotations = {col: fallback for col in df.columns if fallback}
        else:
            filtered_annotations = {}
            for key in df.columns:
                annotations_for_key = bar_annotations.get(key)
                if annotations_for_key:
                    filtered_annotations[key] = annotations_for_key
            bar_annotations = filtered_annotations if filtered_annotations else None

    if bar_annotations is not None and not bar_annotations:
        bar_annotations = None

    if not df.empty:  # handling the case where the subset is empty
        stacked_bar_subplot(df, column_stacked, filename, dict_colors=dict_colors, df_errorbars=df_errorbars, format_y=format_y,
                            rotation=rotation, order_scenarios=order_scenarios, dict_scenarios=dict_scenarios,
                            order_stacked=order_stacked, cap=cap, annotate=annotate, show_total=show_total, fonttick=fonttick, title=title, fontsize_label=fontsize_label,
                            format_label=format_label, figsize=figsize, hspace=hspace, cols_per_row=cols_per_row,
                            juxtaposed=juxtaposed, year_ini=year_ini, bar_annotations=bar_annotations,
                            annotation_pad=annotation_pad, annotation_joiner=annotation_joiner)


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


def simple_heatmap_plot(df, filename, unit="", title='', xcolumn='zone', ycolumn='year', valuecolumn='value'):
    """
    Create a heatmap from the given DataFrame and save it to a file.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'year', 'zone', and 'value
    - filename (str): Path to save the heatmap image.
    - fmt (str): Format for the annotations in the heatmap.
    - title (str): Title for the heatmap.
    
    Returns:
    - None
    """
    def make_formatter(unit):
        def _format(value):
            if pd.isna(value):
                return ""
            if unit == "%":
                value = value * 100
                txt = f"{value:.0f}"
                return f"{txt}%"
            if abs(value) >= 100:
                txt = f"{value:.0f}"
            elif abs(value) >= 1:
                txt = f"{value:.1f}"
            else:
                txt = f"{value:.2f}"
            txt = txt.rstrip('0').rstrip('.')
            return f"{txt} {unit}".strip()
        return _format

    fmt_func = make_formatter(unit)

    pivot_df = df.pivot(index=ycolumn, columns=xcolumn, values=valuecolumn)
    annot_df = pivot_df.map(fmt_func)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        pivot_df,
        cmap='cividis',
        annot=annot_df,
        fmt='',
        linewidths=0.5,
        linecolor='gray'
    )

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        from matplotlib.ticker import FuncFormatter
        formatter = FuncFormatter(lambda v, _: fmt_func(v))
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.yaxis.get_offset_text().set_visible(False)
        cbar.update_ticks()

    # Customization
    ax.set_ylabel("")  # Remove y-axis name
    ax.yaxis.set_label_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Move x-axis label (zone names) to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
    ax.set_title(title, pad=20)

    # Save to PDF
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def faceted_heatmap_plot(
        df,
        filename,
        unit="",
        title='',
        xcolumn='zone',
        ycolumn='year',
        valuecolumn='value',
        subplotcolumn='scenario',
        col_wrap=3,
        subplot_order=None,
        align_axes=False,
        xorder=None,
        yorder=None):
    """
    Create multiple heatmaps faceted by the values of `subplotcolumn`.

    Parameters:
    - df (DataFrame): Input data.
    - filename (str): Path to save the image.
    - unit (str): Unit string appended to annotations and colorbar.
    - title (str): Figure title.
    - xcolumn (str), ycolumn (str), valuecolumn (str): Columns to pivot.
    - subplotcolumn (str): Column used to create subplots.
    - col_wrap (int): Maximum number of subplot columns per row.
    - subplot_order (Sequence, optional): Explicit ordering for facet panels.
    - align_axes (bool): Whether to align all subplots on the same x/y labels.
    - xorder (Sequence, optional): Ordering applied to x-axis labels when aligning.
    - yorder (Sequence, optional): Ordering applied to y-axis labels when aligning.

    Returns:
    - None
    """
    if subplotcolumn not in df.columns:
        raise ValueError(f"Column '{subplotcolumn}' not found in DataFrame.")

    def sort_values(values, reference_series):
        dtype = reference_series.dtype

        if pd.api.types.is_categorical_dtype(dtype):
            categories = reference_series.cat.categories
            return [value for value in categories if value in values]

        if hasattr(dtype, "kind") and dtype.kind in "biufcM":
            return sorted(values)

        numeric_coerced = pd.to_numeric(pd.Series(values), errors='coerce')
        if not numeric_coerced.isna().any():
            return [v for _, v in sorted(zip(numeric_coerced.tolist(), values))]

        datetime_coerced = pd.to_datetime(pd.Series(values), errors='coerce')
        if not datetime_coerced.isna().any():
            return [v for _, v in sorted(zip(datetime_coerced.tolist(), values))]

        try:
            return sorted(values)
        except TypeError:
            return list(values)

    def make_formatter(unit):
        def _format(value):
            if pd.isna(value):
                return ""
            if unit == "%":
                value = value * 100
                txt = f"{value:.0f}"
                return f"{txt}%"
            if abs(value) >= 100:
                txt = f"{value:.0f}"
            elif abs(value) >= 1:
                txt = f"{value:.1f}"
            else:
                txt = f"{value:.2f}"
            txt = txt.rstrip('0').rstrip('.')
            return f"{txt} {unit}".strip()
        return _format

    fmt_func = make_formatter(unit)

    available_values = df[subplotcolumn].dropna().unique()

    if subplot_order is not None:
        missing = [value for value in subplot_order if value not in available_values]
        if missing:
            raise ValueError(f"Values {missing} in 'subplot_order' not found in '{subplotcolumn}'.")
        ordered_values = [value for value in subplot_order if value in available_values]
        leftover = [value for value in available_values if value not in ordered_values]
        if leftover:
            leftover_sorted = sort_values(list(leftover), df[subplotcolumn])
            ordered_values.extend(leftover_sorted)
    elif pd.api.types.is_categorical_dtype(df[subplotcolumn]):
        ordered_values = [value for value in df[subplotcolumn].cat.categories if value in available_values]
    else:
        ordered_values = pd.unique(available_values)
        if len(ordered_values) > 1:
            ordered_values = sort_values(list(ordered_values), df[subplotcolumn])

    if len(ordered_values) == 0:
        raise ValueError(f"No data available to facet by '{subplotcolumn}'.")

    numeric_values = df[valuecolumn].to_numpy(dtype=float)
    if np.isnan(numeric_values).all():
        raise ValueError(f"Column '{valuecolumn}' contains only NaN values.")

    vmin = np.nanmin(numeric_values)
    vmax = np.nanmax(numeric_values)

    if align_axes:
        unique_x = df[xcolumn].dropna().unique()
        unique_y = df[ycolumn].dropna().unique()

        if xorder is not None:
            unknown_x = [value for value in xorder if value not in unique_x]
            if unknown_x:
                raise ValueError(f"Values {unknown_x} in 'xorder' not present in '{xcolumn}'.")
            x_labels = [value for value in xorder if value in unique_x]
        else:
            x_labels = sort_values(list(unique_x), df[xcolumn]) if len(unique_x) > 1 else list(unique_x)

        if yorder is not None:
            unknown_y = [value for value in yorder if value not in unique_y]
            if unknown_y:
                raise ValueError(f"Values {unknown_y} in 'yorder' not present in '{ycolumn}'.")
            y_labels = [value for value in yorder if value in unique_y]
        else:
            y_labels = sort_values(list(unique_y), df[ycolumn]) if len(unique_y) > 1 else list(unique_y)
    else:
        x_labels = None
        y_labels = None

    if col_wrap is None or col_wrap <= 0:
        col_wrap = len(ordered_values)

    ncols = min(col_wrap, len(ordered_values))
    nrows = int(np.ceil(len(ordered_values) / ncols))

    figsize = (ncols * 4.5, nrows * 4.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    colorbar = None
    for idx, value in enumerate(ordered_values):
        ax = axes_flat[idx]
        subset = df[df[subplotcolumn] == value]
        pivot_df = subset.pivot(index=ycolumn, columns=xcolumn, values=valuecolumn)

        if align_axes:
            pivot_df = pivot_df.reindex(index=y_labels, columns=x_labels)

        annot_df = pivot_df.map(fmt_func)

        heatmap = sns.heatmap(
            pivot_df,
            cmap='cividis',
            annot=annot_df,
            fmt='',
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=(idx == 0)
        )

        if idx == 0:
            colorbar = heatmap.collections[0].colorbar
            if colorbar is not None:
                formatter = FuncFormatter(lambda v, _: fmt_func(v))
                colorbar.ax.yaxis.set_major_formatter(formatter)
                colorbar.ax.yaxis.get_offset_text().set_visible(False)
                colorbar.update_ticks()

        ax.set_ylabel("")
        ax.yaxis.set_label_position("left")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
        ax.set_title(str(value), pad=12)

    for idx in range(len(ordered_values), len(axes_flat)):
        axes_flat[idx].axis('off')

    if title:
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(filename)
    plt.close(fig)


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
    def rename_and_reoder(df, rename_index=None, rename_columns=None, order_index=None, order_stacked=None):
        if rename_index is not None:
            df.index = df.index.map(lambda x: rename_index.get(x, x))
        if rename_columns is not None:
            df.columns = df.columns.map(lambda x: rename_columns.get(x, x))
        if order_index is not None:
            df = df.loc[order_index, :]
        if order_stacked is not None:
            df = df.loc[:, order_stacked]
        return df
    
    summary = []

    if required_keys is None:
        required_keys = ['pCapacityFuel', 'pEnergyFuel', 'pEmissionsZone', 'pYearlyCostsZone']

    assert all(
        key in epm_results for key in required_keys), "Required keys for the summary are not included in epm_results"

    if fuel_capa_list is None:
        fuel_capa_list = ['Hydro', 'Solar', 'Wind']

    if fuel_gen_list is None:
        fuel_gen_list = ['Hydro', 'Oil']

    if summary_metrics_list is None:
        summary_metrics_list = ['Capex: $m']

    if 'pCapacityFuel' in required_keys:
        temp = epm_results['pCapacityFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pCapacityFuel'], values='value')
        temp = temp.loc[:, fuel_capa_list]
        temp = rename_and_reoder(temp, rename_columns=RENAME_COLUMNS)
        temp.columns = [f'{col} (MW)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEnergyFuel' in required_keys:
        temp = epm_results['pEnergyFuel'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.loc[:, fuel_gen_list]
        temp = temp.pivot_table(index=[rows_index], columns=NAME_COLUMNS['pEnergyFuel'], values='value')
        temp.columns = [f'{col} (GWh)' for col in temp.columns]
        temp = temp.round(0)
        summary.append(temp)

    if 'pEmissionsZone' in required_keys:
        temp = epm_results['pEmissionsZone'].copy()
        temp = temp[(temp['year'] == year)]
        if zone_list is not None:
            temp = temp[temp['zone'].isin(zone_list)]
        temp = temp.set_index(['scenario'])['value']
        temp = temp * 1e3
        temp.rename('ktCO2', inplace=True).to_frame()
        summary.append(temp)

    if False:
        if 'pEnergyBalanceCountry' in required_keys:
            temp = epm_results['pEnergyBalanceCountry'].copy()
            temp = temp[temp['attribute'] == 'Unmet demand: GWh']
            temp = temp.groupby(['scenario'])['value'].sum()

            t = epm_results['pEnergyBalanceCountry'].copy()
            t = t[t['attribute'] == 'Demand: GWh']
            t = t.groupby(['scenario'])['value'].sum()
            temp = (temp / t) * 1e3
            temp.rename('Unmet (‰)', inplace=True).to_frame()
            summary.append(temp)

            temp = epm_results['pEnergyBalanceCountry'].copy()
            if zone_list is not None:
                temp = temp[temp['zone'].isin(zone_list)]
            temp = temp[temp['attribute'] == 'Surplus generation: GWh']
            temp = temp.groupby(['scenario'])['value'].sum()

            t = epm_results['pEnergyBalanceCountry'].copy()
            if zone_list is not None:
                t = t[t['zone'].isin(zone_list)]
            t = t[t['attribute'] == 'Demand: GWh']
            t = t.groupby(['scenario'])['value'].sum()
            temp = (temp / t) * 1000

            temp.rename('Surplus (‰)', inplace=True).to_frame()
            summary.append(temp)

    if 'pYearlyCostsZone' in required_keys:
        temp = epm_results['pYearlyCostsZone'].copy()
        temp = temp[temp['attribute'] == 'Total Annual Cost by Zone: $m']
        temp = calculate_npv(temp, discount_rate)

        t = epm_results['pEnergyBalance'].copy()
        if zone_list is not None:
            t = t[t['zone'].isin(zone_list)]
        t = t[t['attribute'] == 'Demand: GWh']
        t = calculate_npv(t, discount_rate)

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
                           order_stacked=order_stacked, max_ticks=max_ticks, annotate=annotate, show_total=show_total,
                           fonttick=fonttick, title=title)


def multiple_lines_subplot(df, column_multiple_lines, filename, figsize=(10,6), dict_colors=None, order_index=None,
                            order_stacked=None, dict_scenarios=None, rotation=0, fonttick=14, legend=True,
                           format_y=lambda y, _: '{:.0f} GW'.format(y), annotation_format="{:.0f}",
                           max_ticks=10, annotate=True, show_total=False, title=None, ylim_bottom=None):
    """
    Create a stacked bar subplot from a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    column_stacked : str
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
    order_stacked : list, optional
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
            if order_stacked is not None:
                df_temp = df_temp[[c for c in order_stacked if c in df_temp.columns]]

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


def make_line_subplots(df, filename, column_xaxis, column_value, subplot_column,
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
    column_xaxis : str
        Name of the column for the x-axis.
    column_value : str
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
                ax.plot(data[column_xaxis], data[column_value], label=str(g), color=color)
        else:
            ax.plot(subset[column_xaxis], subset[column_value], color='steelblue')

        ax.set_title(str(key), fontsize=fonttick, fontweight='bold')
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, linestyle='--', alpha=0.5)

        if format_y:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

        if i % ncols == 0:
            ax.set_ylabel(ylabel if ylabel else column_value, fontsize=fonttick)

        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel if xlabel else column_xaxis, fontsize=fonttick)

        if group_column:
            ax.legend(frameon=False, fontsize=fonttick - 2)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=fonttick + 2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
