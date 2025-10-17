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
from matplotlib.ticker import MaxNLocator
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict

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


def _format_annotation_category(value):
    """
    Convert raw annotation categories into human-readable strings.

    Parameters
    ----------
    value : Any
        Original category label used when grouping the source dataframe.

    Returns
    -------
    str
        Clean text representation suitable for display in annotations.
        Tuple inputs are joined with ``" - "`` to improve readability.
    """
    if isinstance(value, tuple):
        return ' - '.join(str(v) for v in value)
    return str(value)


def build_axis_annotations(
    index_values,
    numeric_index,
    grouped,
    subset,
    column_xaxis,
    column_value,
    annotation_map=None,
    column_subplot=None,
    subplot_value=None,
    annotation_source=None,
    annotation_template="{category} - {value:.0f}",
    annotation_threshold=1.0,
    category_formatter=_format_annotation_category,
):
    """
    Combine manual and automatic annotations for charts sharing stacked-data logic.

    The helper merges user-specified labels from ``annotation_map`` and automatically
    generated messages highlighting sharp increases in ``annotation_source``. Automatic
    entries are triggered when the aggregated value rises by more than
    ``annotation_threshold`` between successive x-axis points. The resulting dictionaries
    can be reused by area, bar, or line plots that rely on the same stacked aggregation.

    Parameters
    ----------
    index_values : list
        Ordered x-axis values present in the plotted panel.
    numeric_index : bool
        Indicates whether x-axis labels are numeric (coordinates can be used directly).
    grouped : pandas.DataFrame
        Pivoted dataframe used to render the stacked layers; index must match ``index_values``.
    subset : pandas.DataFrame
        Filtered dataframe feeding the current panel, used to infer automatic annotations.
    column_xaxis : str
        Name of the column supplying x-axis values.
    column_value : str
        Name of the column providing numeric values to stack.
    annotation_map : dict, optional
        Manual annotations either keyed directly by x value or nested as
        ``subplot_value -> {x_value: text}``.
    column_subplot : str, optional
        Column name used for faceting; required when ``annotation_map`` stores nested keys.
    subplot_value : Any, optional
        Current facet identifier; used when extracting nested annotations.
    annotation_source : str, optional
        Column used to compute automatic annotations (e.g., plant name or fuel type).
    annotation_template : str, optional
        Template applied to automatic annotations. Must accept ``category`` and ``value``.
    annotation_threshold : float, optional
        Minimum increase (absolute value) that triggers an automatic annotation.
    category_formatter : callable, optional
        Function that converts raw ``annotation_source`` labels to readable text.

    Returns
    -------
    tuple(dict, dict)
        ``x_coord_map`` maps x values to matplotlib coordinates. ``resolved_annotations``
        stores the final annotation text per x value (empty entries are omitted).
    """
    x_coord_map = {val: (val if numeric_index else pos) for pos, val in enumerate(index_values)}

    resolved_annotations = {}
    if annotation_map:
        if column_subplot is not None and isinstance(annotation_map.get(subplot_value), dict):
            resolved_annotations.update(annotation_map.get(subplot_value, {}))
        elif column_subplot is None:
            resolved_annotations.update(annotation_map)

    resolved_annotations = {
        key: resolved_annotations[key]
        for key in resolved_annotations
        if key in x_coord_map
    }

    if annotation_source is not None and annotation_source in subset.columns:
        auto_annotations = {}
        structured_entries = defaultdict(lambda: defaultdict(list))
        fallback_entries = defaultdict(list)
        ordered_x = list(grouped.index)
        grouped_source = subset.groupby(annotation_source, observed=False)
        for category, group in grouped_source:
            series = (
                group.groupby(column_xaxis, observed=False)[column_value]
                .sum()
                .reindex(ordered_x, fill_value=0)
                .sort_index()
            )
            for x_val, delta in series.diff().dropna().items():
                if delta > annotation_threshold:
                    if isinstance(category, tuple) and len(category) >= 2:
                        group_key = category[0]
                        item_key = category[1] if len(category) == 2 else category[1:]
                        structured_entries[x_val][group_key].append((item_key, delta))
                    else:
                        text = annotation_template.format(
                            category=category_formatter(category),
                            value=delta
                        )
                        fallback_entries[x_val].append(text)

        for x_val, groups in structured_entries.items():
            lines = []
            for group_key, entries in groups.items():
                group_label = category_formatter(group_key)
                if group_label:
                    lines.append(group_label)
                for item_key, value in entries:
                    item_label = ''
                    if item_key is not None:
                        item_label = category_formatter(item_key)
                    if item_label:
                        lines.append(f"  • {item_label}: {value:.0f} MW")
                    else:
                        lines.append(f"  • {value:.0f} MW")
            if fallback_entries.get(x_val):
                lines.extend(fallback_entries.pop(x_val))
            auto_annotations[x_val] = "\n".join(lines).strip()

        for x_val, texts in fallback_entries.items():
            text = "\n".join(texts).strip()
            if not text:
                continue
            if x_val in auto_annotations and auto_annotations[x_val]:
                auto_annotations[x_val] = f"{auto_annotations[x_val]}\n{text}"
            else:
                auto_annotations[x_val] = text

        for key, text in auto_annotations.items():
            if key in resolved_annotations and resolved_annotations[key]:
                resolved_annotations[key] = f"{resolved_annotations[key]}\n{text}"
            else:
                resolved_annotations[key] = text

    return x_coord_map, resolved_annotations


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


def make_auto_yaxis_formatter(unit=""):
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

# Pie plots

def subplot_pie(df, index, dict_colors, column_subplot=None, title='', figsize=(16, 4), ax=None,
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
    column_subplot: str, optional
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

    
    if rename is not None:
        df[index] = df[index].replace(rename)
    if column_subplot is not None:
        # Group by the column for subplots
        groups = df.groupby(column_subplot)

        # Calculate the number of subplots
        num_subplots = len(groups)
        ncols = min(3, num_subplots)  # Limit to 3 columns per row
        nrows = int(np.ceil(num_subplots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1]*nrows))
        axes = np.array(axes).flatten()  # Ensure axes is iterable 1D array


        all_labels = set()  # Collect all labels for the combined legend
        for ax, (name, group) in zip(axes, groups):
            colors = [dict_colors[f] for f in group[index]]
            handles, labels = plot_pie_on_ax(ax, group, index, percent_cap, colors, title=f"{title} - {column_subplot}: {name}")
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

# Stacked area plots

def make_stacked_areaplot(
    df,
    filename=None,
    column_xaxis='year',
    column_value='value',
    column_stacked='fuel',
    column_subplot=None,
    select_subplots=None,
    subplot_order=None,
    subplot_labels=None,
    colors=None,
    filters=None,
    stack_order=None,
    stack_sort_by=None,
    select_x=None,
    rename_x=None,
    x_tick_interval=None,
    format_y=None,
    rotation=0,
    fonttick=12,
    title=None,
    x_label=None,
    y_label=None,
    legend_title=None,
    secondary_df=None,
    secondary_label='',
    secondary_color='brown',
    secondary_column_value=None,
    figsize=(12, 6),
    show_legend=True,
    annotation_map=None,
    annotation_source=None,
    annotation_threshold=1.0,
    annotation_template="{category} - {value:.0f}",
):
    """
    Render stacked area charts with optional subplots, annotations, and a secondary axis.
    Automatic annotations are rendered in a dedicated band above each subplot to avoid overlap.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``column_xaxis``, ``column_value``, and ``column_stacked`` columns.
    filename : str or None
        When provided, save the resulting figure at this path. Otherwise the plot is shown.
    column_xaxis : str, default 'year'
        Column name providing the x-axis values.
    column_value : str, default 'value'
        Column name providing stacked values.
    column_stacked : str, default 'fuel'
        Column name whose categories form the stacked layers.
    column_subplot : str, optional
        Column whose unique values generate subplot panels.
    select_subplots : iterable, optional
        Subset of subplot categories to render (order preserved).
    subplot_order : iterable, optional
        Custom ordering applied to subplot categories.
    subplot_labels : dict, optional
        Mapping from subplot category to display label.
    colors : dict, optional
        Mapping from stacked categories to colours; missing keys fall back to matplotlib defaults.
    filters : dict, optional
        Column filters applied before plotting. Accepts scalars or iterables.
    stack_order : iterable, optional
        Preferred ordering for stacked categories.
    stack_sort_by : str, optional
        Column used to sort stacked categories by their first observed value within each subplot.
    select_x : iterable, optional
        Restrict the x-axis to these values (order preserved).
    rename_x : dict, optional
        Mapping used to relabel x tick labels.
    x_tick_interval : int, optional
        For numeric x axes, retain only every n-th tick label.
    format_y : callable, optional
        Formatter applied to the left y-axis ticks.
    rotation : int, default 0
        Rotation for x tick labels.
    fonttick : int, default 12
        Base font size for ticks and subplot titles.
    title : str, optional
        Figure title (single panel) or suptitle (multiple panels).
    x_label : str, optional
        Custom label for the x-axis (defaults to ``column_xaxis``).
    y_label : str, optional
        Custom label for the y-axis (defaults to ``column_value``).
    legend_title : str, optional
        Title displayed above the legend. When omitted, the legend has no heading (consistent with stacked bar plots).
    secondary_df : pd.DataFrame, optional
        Auxiliary dataset rendered on a secondary y-axis (single panel only).
    secondary_label : str, optional
        Label for the secondary axis line.
    secondary_color : str, default 'brown'
        Colour used for the secondary axis line.
    secondary_column_value : str, optional
        Column used from ``secondary_df``; defaults to ``column_value``.
    figsize : tuple, default (12, 6)
        Figure size for a single panel. Multi-panel layouts scale the height automatically.
    show_legend : bool, default True
        Whether to render legend entries.
    annotation_map : dict, optional
        Pre-defined annotations keyed by x value or by subplot -> x value.
    annotation_source : str, optional
        Column name used to derive annotations from year-on-year increases.
    annotation_threshold : float, default 1.0
        Minimum increase required to add an automatic annotation.
    annotation_template : str, default "{category} - {value:.0f}"
        Template used when constructing automatic annotation text. Supports ``{category}`` and ``{value}``.
    """

    from matplotlib.ticker import FuncFormatter

    df = df.copy()
    filters = filters or {}
    rename_x = rename_x or {}
    subplot_labels = subplot_labels or {}
    annotation_map = annotation_map or {}
    if format_y is None:
        format_y = make_auto_yaxis_formatter("")

    for column, allowed in filters.items():
        if column not in df.columns:
            continue
        if isinstance(allowed, (list, tuple, set, np.ndarray)):
            df = df[df[column].isin(allowed)]
        else:
            df = df[df[column] == allowed]

    if df.empty:
        print("No data available to plot.")
        return

    secondary_column_value = secondary_column_value or column_value
    select_subplots = list(select_subplots) if select_subplots is not None else None

    if column_subplot is not None:
        subplot_values = list(dict.fromkeys(df[column_subplot].dropna().tolist()))
        if select_subplots is not None:
            subplot_values = [val for val in subplot_values if val in select_subplots]
        if subplot_order is not None:
            ordered = [val for val in subplot_order if val in subplot_values]
            subplot_values = ordered + [val for val in subplot_values if val not in ordered]
        if not subplot_values:
            print("No data available to plot.")
            return
    else:
        subplot_values = [None]

    num_panels = len(subplot_values)
    ncols = min(3, num_panels)
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0], figsize[1] * nrows),
        sharey=True if column_subplot is not None else False
    )
    axes = np.atleast_1d(axes).flatten()

    legend_handles = None
    legend_labels = None
    primary_handles = None
    primary_labels = None

    bottom_row_start = (nrows - 1) * ncols

    for idx, subplot_value in enumerate(subplot_values):
        ax = axes[idx]
        subset = df if column_subplot is None else df[df[column_subplot] == subplot_value]

        if subset.empty:
            ax.set_visible(False)
            continue

        grouped = (
            subset.groupby([column_xaxis, column_stacked], observed=False)[column_value]
            .sum()
            .unstack(column_stacked)
            .fillna(0)
            .sort_index()
        )

        if select_x is not None:
            ordered_x = [x for x in select_x if x in grouped.index]
            grouped = grouped.loc[ordered_x]

        if grouped.empty:
            ax.set_visible(False)
            continue

        column_sequence = list(grouped.columns)
        priority = []
        if stack_sort_by and stack_sort_by in subset.columns:
            sort_series = (
                subset.groupby(column_stacked, observed=False)[stack_sort_by]
                .first()
                .sort_values()
            )
            priority.extend(sort_series.index.tolist())
        if stack_order:
            priority.extend([col for col in stack_order if col not in priority])
        if priority:
            ordered_cols = [col for col in priority if col in column_sequence]
            ordered_cols += [col for col in column_sequence if col not in ordered_cols]
            grouped = grouped[ordered_cols]
            column_sequence = ordered_cols

        color_list = None
        if colors and all(col in colors for col in column_sequence):
            color_list = [colors[col] for col in column_sequence]

        grouped.plot.area(ax=ax, stacked=True, alpha=0.8, color=color_list)

        index_values = list(grouped.index)
        display_labels = [rename_x.get(val, val) for val in index_values]
        numeric_index = all(isinstance(val, (int, float, np.integer, np.floating)) for val in index_values)

        if numeric_index:
            ax.set_xticks(index_values)
            ax.set_xticklabels([str(lbl) for lbl in display_labels], rotation=rotation)
            if x_tick_interval:
                try:
                    base = index_values[0]
                    tick_candidates = [
                        val for val in index_values
                        if (val - base) % x_tick_interval == 0
                    ]
                    if tick_candidates:
                        ax.set_xticks(tick_candidates)
                        ax.set_xticklabels([str(rename_x.get(val, val)) for val in tick_candidates], rotation=rotation)
                except TypeError:
                    pass
        else:
            positions = np.arange(len(index_values))
            ax.set_xticks(positions)
            ax.set_xticklabels([str(lbl) for lbl in display_labels], rotation=rotation)

        format_ax(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelrotation=rotation)

        if column_subplot is not None:
            if idx < bottom_row_start:
                ax.set_xlabel('')
            else:
                label = '' if x_label is None else x_label
                ax.set_xlabel(label, fontsize=fonttick if label else fonttick)
        else:
            label = '' if x_label is None else x_label
            ax.set_xlabel(label, fontsize=fonttick if label else fonttick)

        if idx % ncols == 0:
            ylabel_text = '' if y_label is None else y_label
            ax.set_ylabel(ylabel_text, fontsize=fonttick if ylabel_text else fonttick)
            ax.yaxis.set_major_formatter(FuncFormatter(format_y))
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        if column_subplot is not None:
            subplot_title = subplot_labels.get(subplot_value, subplot_value)
            if subplot_title is not None:
                ax.set_title(str(subplot_title), fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

        handles, labels = ax.get_legend_handles_labels()
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.remove()
        labels = [str(label).replace('_', ' ') for label in labels]

        if column_subplot is None:
            primary_handles = list(handles)
            primary_labels = list(labels)
        else:
            if legend_handles is None:
                legend_handles = list(handles)
                legend_labels = list(labels)

        x_coord_map, resolved_annotations = build_axis_annotations(
            index_values=index_values,
            numeric_index=numeric_index,
            grouped=grouped,
            subset=subset,
            column_xaxis=column_xaxis,
            column_value=column_value,
            annotation_map=annotation_map,
            column_subplot=column_subplot,
            subplot_value=subplot_value,
            annotation_source=annotation_source,
            annotation_template=annotation_template,
            annotation_threshold=annotation_threshold,
            category_formatter=_format_annotation_category,
        )

        valid_annotations = {key: text for key, text in resolved_annotations.items() if text}
        if valid_annotations:
            divider = make_axes_locatable(ax)
            band_ax = divider.append_axes("top", size="18%", pad=0.25, sharex=ax)
            band_ax.set_ylim(0, 1)
            band_ax.set_facecolor('none')
            band_ax.set_yticks([])
            plt.setp(band_ax.get_xticklabels(), visible=False)
            for spine in band_ax.spines.values():
                spine.set_visible(False)
            for x_val, text in valid_annotations.items():
                if x_val not in grouped.index:
                    continue
                x_pos = x_coord_map[x_val]
                band_ax.text(
                    x_pos,
                    0.5,
                    text,
                    ha='center',
                    va='center',
                    fontsize=max(fonttick - 2, 8),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgrey', linewidth=0.8)
                )
            band_ax.tick_params(axis='x', which='both', length=0, labelbottom=False)

    # Hide any unused axes
    for idx in range(num_panels, len(axes)):
        fig.delaxes(axes[idx])

    tight_rect = None

    if column_subplot is not None and show_legend and legend_handles:
        legend_kwargs = dict(loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
        if legend_title:
            legend_kwargs['title'] = legend_title
        fig.legend(
            legend_handles[::-1],
            [label.replace('_', ' ') for label in legend_labels[::-1]],
            **legend_kwargs
        )
        tight_rect = [0, 0, 0.88, 1]

    if column_subplot is None:
        ax = axes[0]
        if title:
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=6, fontsize=fonttick + 1)

        secondary_handles = []
        secondary_labels = []

        if secondary_df is not None and not secondary_df.empty:
            sec = secondary_df.copy()
            for column, allowed in filters.items():
                if column not in sec.columns:
                    continue
                if isinstance(allowed, (list, tuple, set, np.ndarray)):
                    sec = sec[sec[column].isin(allowed)]
                else:
                    sec = sec[sec[column] == allowed]
            secondary_series = (
                sec.groupby(column_xaxis, observed=False)[secondary_column_value]
                .sum()
                .sort_index()
            )
            if select_x is not None:
                secondary_series = secondary_series.loc[[x for x in select_x if x in secondary_series.index]]

            if not secondary_series.empty:
                ax2 = ax.twinx()
                line, = ax2.plot(
                    secondary_series.index,
                    secondary_series.values,
                    color=secondary_color,
                    label=secondary_label or secondary_column_value
                )
                ax2.set_ylabel(secondary_label or secondary_column_value, color=secondary_color)
                format_ax(ax2, linewidth=False)
                secondary_handles.append(line)
                secondary_labels.append(line.get_label())

        if show_legend:
            combined_handles = []
            combined_labels = []
            if primary_handles:
                combined_handles.extend(primary_handles)
                combined_labels.extend([label.replace('_', ' ') for label in primary_labels])
            if secondary_handles:
                combined_handles.extend(secondary_handles)
                combined_labels.extend(secondary_labels)
            if combined_handles:
                legend_kwargs = dict(loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
                if legend_title:
                    legend_kwargs['title'] = legend_title
                fig.legend(
                    combined_handles,
                    combined_labels,
                    **legend_kwargs
                )
                tight_rect = [0, 0, 0.88, 1]
        else:
            legend_obj = ax.get_legend()
            if legend_obj is not None:
                legend_obj.remove()
    else:
        if title:
            fig.suptitle(title, fontsize=fonttick + 2)
            if tight_rect is None:
                tight_rect = [0, 0, 1, 0.95]
            else:
                tight_rect[3] = min(tight_rect[3], 0.95)

    if tight_rect:
        fig.tight_layout(rect=tight_rect)
    else:
        fig.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

# Disptach plots

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
    >>> dispatch_plot(df_area=df_area, df_line=df_line, dict_colors=dict_colors, filename='dispatch_plot.png')
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

    def remove_na_values(df):
        """Removes na values from a dataframe, to avoind unnecessary labels in plots."""
        df = df.where((df > 1e-6) | (df < -1e-6),
                                        np.nan)
        df = df.dropna(axis=1, how='all')
        return df
    
    def prepare_hourly_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=None, select_time=None):
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
        df = prepare_hourly_dataframe(df, zone='Liberia', year=2025, scenario='Baseline', column_stacked='fuel', fuel_grouping=None, select_time=select_time)
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

    tmp_concat_area = []
    for key in dfs_area:
        df = dfs_area[key]
        if stacked:  # we want to group data by a given column (eg, fuel for dispatch)
            column_stacked = NAME_COLUMNS[key]
        else:
            column_stacked = None
        df, temp = prepare_hourly_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_area.append(df)

    tmp_concat_line = []
    for key in dfs_line:
        df = dfs_line[key]
        if stacked:  # we want to group data by a given column (eg, fuel for dispatch)
            column_stacked = NAME_COLUMNS[key]
        else:
            column_stacked = None
        df, temp = prepare_hourly_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=fuel_grouping, select_time=select_time)
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

# Stacked bar plots

def make_stacked_barplot(df, filename, dict_colors, df_errorbars=None, column_xaxis='year',
                              column_stacked='fuel', column_multiple_bars='scenario',
                              column_value='value', select_xaxis=None, stacked_grouping=None, order_scenarios=None, dict_scenarios=None,
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
    stacked_grouping : dict, optional
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
    make_stacked_barplot(epm_dict['pCapacityFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
                              select_xaxis=[2025, 2028, 2030], stacked_grouping=fuel_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} MW'.format(y))

    Stacked bar subplots for reserve evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
    make_stacked_barplot(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], selected_zone='Liberia',
                              column_xaxis='year', column_stacked='fuel', column_multiple_bars='scenario',
                              select_xaxis=[2025, 2028, 2030], stacked_grouping=stacked_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} GWh'.format(y),
                              order_stacked=['Hydro', 'Oil'], cap=2)
    """
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

    if column_multiple_bars is None:
        print('column_multiple_bars cannot be None, but column_xaxis can. Automatically inverting.')
        column_multiple_bars = column_xaxis
        column_xaxis = None
    
    if stacked_grouping is not None:
        for key, grouping in stacked_grouping.items():
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

# Scatter plots

def make_scatter_plot(df, column_xaxis, column_yaxis, column_color, dict_colors,
                             ymax=None, xmax=None, title='', legend=None, filename=None,
                             size_scale=None, annotate_thresh=None, column_subplot=None,
                             figsize=None, share_axes=True):
    """
    Create scatter plots with optional subplots, coloring points by a categorical column.

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
    dict_colors : dict
        Dictionary mapping values in column_color to specific colors.
    ymax : float, optional
        Maximum y-axis value.
    xmax : float, optional
        Maximum x-axis value.
    title : str, optional
        Title of the plot (applies to the whole figure; subplots get suffixes).
    legend : str, optional
        Title for the legend. If None, legends are suppressed.
    filename : str, optional
        File name to save the plot. If None, the plot is displayed.
    size_scale : float, optional
        Scaling factor for point sizes based on `column_xaxis` values.
    annotate_thresh : float, optional
        Threshold for annotating points with generator names.
    column_subplot : str, optional
        Column name to split the data into subplots.
    figsize : tuple, optional
        Figure size. For subplots this is interpreted per subplot and scaled by layout.
    share_axes : bool, optional
        Whether subplots share x and y axes when `column_subplot` is provided.

    Returns
    -------
    None
        Displays or saves the scatter plot(s).
    """
    def scatter_plot_on_ax(ax, df, column_xaxis, column_yaxis, column_color, dict_colors,
                        ymax=None, xmax=None, title='', legend=None,
                        size_scale=None, annotate_thresh=None):
        """
        Plot category-coloured scatter points onto a supplied matplotlib Axes, returning legend handles for reuse.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes that receives the scatter plot.
        df : pd.DataFrame
            Data to plot. Must contain the columns referenced by the other arguments.
        column_xaxis : str
            Column name used for x values.
        column_yaxis : str
            Column name used for y values.
        column_color : str
            Column name whose categories determine the point colors.
        dict_colors : dict
            Maps each category in `column_color` to a matplotlib-compatible color.
        ymax : float, optional
            Upper y-axis limit. Lower bound is pinned to zero when provided.
        xmax : float, optional
            Upper x-axis limit. Lower bound is pinned to zero when provided.
        title : str, optional
            Axes title.
        legend : str, optional
            Legend title. When omitted the legend is removed from the axes.
        size_scale : float, optional
            Scale applied to `column_xaxis` values to compute marker areas.
        annotate_thresh : float, optional
            Minimum x value above which point annotations using the `generator` column are added.

        Returns
        -------
        tuple[list[matplotlib.lines.Line2D], list[str]]
            Legend handles and labels created while plotting (may be empty).
        """
        unique_values = df[column_color].unique()
        for val in unique_values:
            if val not in dict_colors:
                raise ValueError(f"No color specified for value '{val}' in {column_color}")

        color_map = {val: dict_colors[val] for val in unique_values}

        sizes = 50
        if size_scale is not None:
            sizes = df[column_xaxis] * size_scale

        handles = []
        labels = []

        for value, color in color_map.items():
            subset = df[df[column_color] == value]
            if subset.empty:
                continue

            if hasattr(sizes, "reindex"):
                subset_sizes = sizes.reindex(subset.index)
            else:
                subset_sizes = sizes

            ax.scatter(
                subset[column_xaxis],
                subset[column_yaxis],
                label=value,
                color=color,
                alpha=0.7,
                s=subset_sizes
            )

            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=8))
            labels.append(value)

            if annotate_thresh is not None and 'generator' in subset.columns:
                for i, txt in enumerate(subset['generator']):
                    if subset[column_xaxis].iloc[i] > annotate_thresh:
                        x_value = subset[column_xaxis].iloc[i]
                        y_value = subset[column_yaxis].iloc[i]
                        ax.annotate(
                            txt,
                            (x_value, y_value),
                            xytext=(5, 10),
                            textcoords='offset points',
                            fontsize=9,
                            color='black',
                            ha='left'
                        )

        if ymax is not None:
            ax.set_ylim(0, ymax)

        if xmax is not None:
            ax.set_xlim(0, xmax)

        ax.set_xlabel(column_xaxis)
        ax.set_ylabel(column_yaxis)
        ax.set_title(title)

        existing_legend = ax.get_legend()
        if legend is not None and handles:
            ax.legend(handles=handles, labels=labels, title=legend or column_color, frameon=False)
        elif existing_legend is not None:
            existing_legend.remove()

        ax.grid(True, linestyle='--', alpha=0.5)

        return handles, labels

    
    if column_subplot is not None:
        unique_values = df[column_subplot].unique()
        n_subplots = len(unique_values)
        if n_subplots == 0:
            return

        ncols = min(3, n_subplots)
        nrows = int(np.ceil(n_subplots / ncols))
        base_figsize = figsize or (12, 8)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(base_figsize[0] * ncols, base_figsize[1] * nrows),
            sharex=share_axes,
            sharey=share_axes
        )

        axes = np.array(axes).flatten()
        for idx, val in enumerate(unique_values):
            ax = axes[idx]
            subset_df = df[df[column_subplot] == val]
            scatter_plot_on_ax(
                ax,
                subset_df,
                column_xaxis,
                column_yaxis,
                column_color,
                dict_colors,
                ymax=ymax,
                xmax=xmax,
                title=f"{title} - {column_subplot}: {val}" if title else f"{column_subplot}: {val}",
                legend=legend,
                size_scale=size_scale,
                annotate_thresh=annotate_thresh
            )

        for idx in range(len(unique_values), len(axes)):
            fig.delaxes(axes[idx])

        fig.tight_layout()
    else:
        fig_size = figsize or (8, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        scatter_plot_on_ax(
            ax,
            df,
            column_xaxis,
            column_yaxis,
            column_color,
            dict_colors,
            ymax=ymax,
            xmax=xmax,
            title=title,
            legend=legend,
            size_scale=size_scale,
            annotate_thresh=annotate_thresh
        )

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# Heatmap plots

def heatmap_plot(
    df,
    filename=None,
    *,
    x_column='zone',
    y_column='year',
    value_column='value',
    subplot_column=None,
    title='',
    unit='',
    cmap='cividis',
    col_wrap=3,
    subplot_order=None,
    align_axes=False,
    x_order=None,
    y_order=None,
    figsize=None,
    filters=None,
    share_colorbar=True,
    vmin=None,
    vmax=None,
    fonttick=12,
):
    """
    Render one or many heatmaps with consistent annotation formatting and colour scaling.
    """
    from matplotlib.ticker import FuncFormatter

    if subplot_column is not None and subplot_column not in df.columns:
        raise ValueError(f"Column '{subplot_column}' not found in the DataFrame.")

    df = df.copy()
    filters = filters or {}

    for column, allowed in filters.items():
        if column not in df.columns:
            continue
        if isinstance(allowed, (list, tuple, set, np.ndarray)):
            df = df[df[column].isin(allowed)]
        else:
            df = df[df[column] == allowed]

    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    if df[value_column].isna().all():
        raise ValueError(f"Column '{value_column}' contains only NaN values.")

    def _make_formatter():
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

    fmt_func = _make_formatter()

    def _sort_values(values, reference_series):
        dtype = reference_series.dtype
        if pd.api.types.is_categorical_dtype(dtype):
            categories = reference_series.cat.categories
            return [val for val in categories if val in values]
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

    if subplot_column is not None:
        available = df[subplot_column].dropna().unique()
        if subplot_order is not None:
            missing = [val for val in subplot_order if val not in available]
            if missing:
                raise ValueError(f"Values {missing} in 'subplot_order' not present in '{subplot_column}'.")
            ordered_panels = [val for val in subplot_order if val in available]
            remainder = [val for val in available if val not in ordered_panels]
            if remainder:
                ordered_panels.extend(_sort_values(remainder, df[subplot_column]))
        elif pd.api.types.is_categorical_dtype(df[subplot_column]):
            ordered_panels = [val for val in df[subplot_column].cat.categories if val in available]
        else:
            ordered_panels = pd.unique(available)
            if len(ordered_panels) > 1:
                ordered_panels = _sort_values(list(ordered_panels), df[subplot_column])
        if not ordered_panels:
            raise ValueError(f"No data available to facet by '{subplot_column}'.")
    else:
        ordered_panels = [None]

    if align_axes:
        unique_x = df[x_column].dropna().unique()
        unique_y = df[y_column].dropna().unique()
        if x_order is not None:
            unknown_x = [val for val in x_order if val not in unique_x]
            if unknown_x:
                raise ValueError(f"Values {unknown_x} in 'x_order' not present in '{x_column}'.")
            x_labels = [val for val in x_order if val in unique_x]
        else:
            x_labels = _sort_values(list(unique_x), df[x_column]) if len(unique_x) > 1 else list(unique_x)
        if y_order is not None:
            unknown_y = [val for val in y_order if val not in unique_y]
            if unknown_y:
                raise ValueError(f"Values {unknown_y} in 'y_order' not present in '{y_column}'.")
            y_labels = [val for val in y_order if val in unique_y]
        else:
            y_labels = _sort_values(list(unique_y), df[y_column]) if len(unique_y) > 1 else list(unique_y)
    else:
        x_labels = None
        y_labels = None

    if vmin is None or vmax is None:
        numeric_values = df[value_column].to_numpy(dtype=float)
        if vmin is None:
            vmin = np.nanmin(numeric_values)
        if vmax is None:
            vmax = np.nanmax(numeric_values)

    num_panels = len(ordered_panels)
    if subplot_column is None:
        nrows, ncols = 1, 1
    else:
        if col_wrap is None or col_wrap <= 0:
            col_wrap = num_panels
        ncols = min(col_wrap, num_panels)
        nrows = int(np.ceil(num_panels / ncols))

    if figsize is None:
        if subplot_column is None:
            figsize = (10, 6)
        else:
            figsize = (ncols * 4.5, nrows * 4.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    colorbar = None
    for idx, panel_value in enumerate(ordered_panels):
        ax = axes_flat[idx]
        subset = df if subplot_column is None else df[df[subplot_column] == panel_value]
        if subset.empty:
            ax.axis('off')
            continue

        pivot_df = subset.pivot(index=y_column, columns=x_column, values=value_column)
        if align_axes:
            pivot_df = pivot_df.reindex(index=y_labels, columns=x_labels)

        annot_df = pivot_df.map(fmt_func)
        heatmap = sns.heatmap(
            pivot_df,
            cmap=cmap,
            annot=annot_df,
            fmt='',
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=share_colorbar and colorbar is None,
        )

        if colorbar is None and share_colorbar:
            colorbar = heatmap.collections[0].colorbar
            if colorbar is not None:
                colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_func(v)))
                colorbar.ax.yaxis.get_offset_text().set_visible(False)
                colorbar.update_ticks()

        ax.set_ylabel('')
        ax.yaxis.set_label_position("left")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')

        if subplot_column is not None:
            ax.set_title(str(panel_value), pad=12, fontsize=fonttick, fontweight='bold', color='dimgrey')
        elif title:
            ax.set_title(title, pad=20, fontsize=fonttick + 1, fontweight='bold', color='dimgrey')

    for idx in range(num_panels, len(axes_flat)):
        axes_flat[idx].axis('off')

    if subplot_column is not None and title:
        fig.suptitle(title, y=0.98, fontsize=fonttick + 2)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def heatmap_difference_plot(data, filename=None, percentage=False, baseline='Baseline'):
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

    heatmap_difference_plot(summary, filename, percentage=percentage, baseline=summary.index[0])

# Line plots

def make_line_plot(
    df,
    filename,
    column_xaxis,
    y_column,
    dict_colors=None,
    column_subplot=None,
    series_column=None,
    select_subplots=None,
    order_index=None,
    order_series=None,
    dict_scenarios=None,
    figsize=(10, 6),
    format_y=lambda y, _: '{:.0f} MW'.format(y),
    rotation=0,
    fonttick=12,
    legend=True,
    max_ticks=10,
    title=None,
    ymin=None,
    aggfunc=None,
    xlabel=None,
    ylabel=None
):
    """
    Build configurable line charts with optional subplots and multiple series per panel.

    Parameters
    ----------
    df : pd.DataFrame
        Source data containing at least ``column_xaxis`` and ``y_column`` plus any optional grouping columns.
    filename : str or None
        When provided the figure is saved at this path; otherwise the plot is shown interactively.
    column_xaxis : str
        Column containing the x-axis values (e.g. hour, year, timestep).
    y_column : str
        Column containing the numeric values to plot on the y-axis.
    dict_colors : dict, optional
        Maps each series identifier to a matplotlib-compatible color. Ignored when ``series_column`` is None.
    column_subplot : str, optional
        Column whose unique values create subplot panels.
    series_column : str, optional
        Column whose unique values create individual lines within each subplot.
    select_subplots : iterable, optional
        Subset of subplot categories to display, preserving the provided order.
    order_index : iterable, optional
        Explicit order to apply to the x-axis after aggregation.
    order_series : iterable, optional
        Explicit order and subset of series to plot when ``series_column`` is used.
    dict_scenarios : dict, optional
        Mapping applied to x-axis labels after aggregation (handy for renaming scenarios or timesteps).
    figsize : tuple(float, float), default (10, 6)
        Base figure size; automatically scaled when multiple subplots are generated.
    format_y : callable, optional
        Formatter applied to the shared y-axis ticks. Defaults to MW-formatted integers.
    rotation : int, default 0
        Rotation angle in degrees for x-axis tick labels.
    fonttick : int, default 12
        Base font size for tick labels and subplot titles.
    legend : bool, default True
        Show a consolidated legend when multiple series are present.
    max_ticks : int, default 10
        Maximum number of x tick labels per subplot. Labels are evenly sampled when exceeded.
    title : str, optional
        Figure title when only a single subplot is drawn.
    ymin : float, optional
        Lower bound forced on the y-axis across all panels.
    aggfunc : str or callable, optional
        Aggregation applied after grouping by subplot/series/x columns. If None, raw values are used.
    xlabel : str, optional
        Text for the shared x-axis label (defaults to ``column_xaxis``).
    ylabel : str, optional
        Text for the shared y-axis label (defaults to ``y_column``).
    selected_zone : str, optional
        Convenience filter applied when the DataFrame contains a ``zone`` column.
    selected_year : int, optional
        Convenience filter applied when the DataFrame contains a ``year`` column.
    annotation_format, annotate, show_total : optional
        Retained for backwards compatibility; currently unused.

    Example
    -------
    >>> line_plot_with_options(
    ...     df=dispatch_df,
    ...     filename=None,
    ...     column_xaxis='hour',
    ...     y_column='generation_mw',
    ...     column_subplot='scenario',
    ...     series_column='technology',
    ...     dict_colors={'Solar PV': '#FDB813', 'Wind': '#3A76D0'}
    ... )
    """
    from matplotlib.ticker import FuncFormatter

    df = df.copy()

    required_cols = {column_xaxis, y_column}
    if column_subplot is not None:
        required_cols.add(column_subplot)
    if series_column is not None:
        required_cols.add(series_column)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for plotting: {missing_cols}")

    if column_subplot is not None and select_subplots is not None:
        df = df[df[column_subplot].isin(select_subplots)]

    if df.empty:
        print("No data available to plot.")
        return

    columns_to_keep = [column_xaxis, y_column]
    if column_subplot is not None:
        columns_to_keep.append(column_subplot)
    if series_column is not None:
        columns_to_keep.append(series_column)
    df = df[columns_to_keep].dropna(subset=[column_xaxis, y_column])

    grouping_cols = []
    if column_subplot is not None:
        grouping_cols.append(column_subplot)
    if series_column is not None:
        grouping_cols.append(series_column)
    grouping_cols.append(column_xaxis)

    if aggfunc is not None:
        df = df.groupby(grouping_cols, observed=False)[y_column].agg(aggfunc).reset_index()

    if column_subplot is not None:
        unique_keys = list(dict.fromkeys(df[column_subplot].tolist()))
        subplot_keys = [key for key in unique_keys if select_subplots is None or key in select_subplots]
    else:
        subplot_keys = [None]

    if not subplot_keys:
        print("No data available to plot.")
        return

    num_subplots = len(subplot_keys)
    ncols = min(3, num_subplots)
    nrows = int(np.ceil(num_subplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        sharey=True
    )
    axes = np.array(axes).reshape(-1)

    all_handles, all_labels = [], []

    for idx, subplot_key in enumerate(subplot_keys):
        ax = axes[idx]
        if column_subplot is not None:
            subset = df[df[column_subplot] == subplot_key]
        else:
            subset = df

        if subset.empty:
            ax.set_visible(False)
            continue

        if series_column is not None:
            pivot = subset.pivot_table(
                index=column_xaxis,
                columns=series_column,
                values=y_column,
                aggfunc='mean'
            )
        else:
            pivot = subset.groupby(column_xaxis, observed=False)[y_column].mean().to_frame(y_column)

        if order_index is not None:
            pivot = pivot.reindex([val for val in order_index if val in pivot.index])

        if dict_scenarios is not None:
            pivot.index = pivot.index.map(lambda x: dict_scenarios.get(x, x))

        if series_column is not None and order_series is not None:
            pivot = pivot[[col for col in order_series if col in pivot.columns]]

        pivot = pivot.dropna(axis=1, how='all')
        pivot = pivot.dropna(axis=0, how='all')

        if pivot.empty:
            ax.set_visible(False)
            continue

        index_values = list(pivot.index)
        x_positions = list(range(len(index_values)))

        for col in pivot.columns:
            series = pivot[col]
            if series.isna().all():
                continue
            color = dict_colors.get(col) if dict_colors is not None else None
            label = str(col)
            line, = ax.plot(x_positions, series.values, marker='o', color=color, label=label)
            all_handles.append(line)
            all_labels.append(label)

        ax.set_xticks(x_positions)
        if index_values:
            if len(index_values) > max_ticks:
                positions = np.linspace(0, len(index_values) - 1, max_ticks, dtype=int)
            else:
                positions = np.arange(len(index_values))
            ax.set_xticks(positions)
            ax.set_xticklabels([index_values[i] for i in positions], rotation=rotation)

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)

        if len(subplot_keys) > 1:
            subplot_title = subplot_key
            if isinstance(subplot_title, tuple):
                subplot_title = '-'.join(str(v) for v in subplot_title)
            ax.set_title(str(subplot_title), fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)
        else:
            display_title = title if title is not None else subplot_key
            if display_title is not None:
                if isinstance(display_title, tuple):
                    display_title = '-'.join(str(v) for v in display_title)
                ax.set_title(str(display_title), fontweight='bold', color='dimgrey', pad=-1.6, fontsize=fonttick)

        if idx % ncols == 0:
            ylabel_text = ylabel if ylabel is not None else y_column
            ax.set_ylabel(ylabel_text, fontsize=fonttick)
            ax.yaxis.set_major_formatter(FuncFormatter(format_y))
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        if idx >= (nrows - 1) * ncols:
            xlabel_text = xlabel if xlabel is not None else column_xaxis
            ax.set_xlabel(xlabel_text, fontsize=fonttick)
        else:
            ax.set_xlabel('')

        ax.grid(True, linestyle='--', alpha=0.5)

        if ymin is not None:
            current_top = ax.get_ylim()[1]
            ax.set_ylim(bottom=ymin, top=current_top)

    for idx in range(num_subplots, len(axes)):
        fig.delaxes(axes[idx])

    applied_tight_layout = False
    if legend and all_handles:
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(all_handles, all_labels):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label.replace('_', ' '))
        if unique_handles:
            fig.legend(
                unique_handles,
                unique_labels,
                loc='center left',
                frameon=False,
                ncol=1,
                bbox_to_anchor=(1, 0.5)
            )
            fig.tight_layout(rect=[0, 0, 0.85, 1])
            applied_tight_layout = True

    if not applied_tight_layout:
        fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
