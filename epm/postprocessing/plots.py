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
from matplotlib.patches import Patch, FancyArrowPatch, Rectangle
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
from shapely.geometry import LineString, Point, LinearRing
import argparse
import shutil
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict, OrderedDict
import warnings

from .utils import NAME_COLUMNS, RENAME_COLUMNS

DEFAULT_FUEL_ORDER = None


def set_default_fuel_order(order):
    """Store the default stacking order for fuel plots."""
    global DEFAULT_FUEL_ORDER
    DEFAULT_FUEL_ORDER = list(order) if order else None


def safe_tight_layout(fig, rect=None, shrink_top=0.03):
    """
    Apply ``tight_layout`` but fall back gracefully when Matplotlib cannot accommodate decorations.

    When ``tight_layout`` raises a ``UserWarning`` (e.g., heavy legends/supertitles), we relax
    the top margin slightly via ``subplots_adjust`` instead of emitting a warning.
    """
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("error", category=UserWarning)
        try:
            if rect:
                fig.tight_layout(rect=rect)
            else:
                fig.tight_layout()
            return
        except UserWarning:
            pass

    if rect:
        left, bottom, right, top = rect
        fig.subplots_adjust(
            left=left,
            bottom=bottom,
            right=right,
            top=max(0.0, min(top, 1.0) - shrink_top),
        )
    else:
        fig.subplots_adjust(top=1 - shrink_top, right=0.96, left=0.08, bottom=0.08)

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
            group_totals = {
                group_key: sum(value for _, value in entries)
                for group_key, entries in groups.items()
            }
            for group_key, entries in sorted(groups.items(), key=lambda item: group_totals[item[0]], reverse=True):
                group_label = category_formatter(group_key)
                if group_label:
                    lines.append(group_label)
                for item_key, value in sorted(entries, key=lambda item: item[1], reverse=True):
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
    if num_panels == 1:
        fig_width, fig_height = figsize
    else:
        fig_width = figsize[0] * ncols
        fig_height = figsize[1] * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey=True if column_subplot is not None else False
    )
    axes = np.atleast_1d(axes).flatten()

    legend_handles = None
    legend_labels = None
    primary_handles = []
    primary_labels = []
    single_panel_title = title if column_subplot is None else None

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
                ax.set_title(str(subplot_title), fontweight='bold', color='dimgrey', pad=6, fontsize=fonttick)

        handles, labels = ax.get_legend_handles_labels()
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.remove()
        labels = [str(label).replace('_', ' ') for label in labels]

        if column_subplot is None:
            primary_handles.extend(handles)
            primary_labels.extend(labels)
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
            band_ax = divider.append_axes("top", size="20%", pad=0.6, sharex=ax)
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
    legend_anchor_x = 0.92

    if column_subplot is not None and show_legend and legend_handles:
        legend_kwargs = dict(loc='center left', frameon=False, bbox_to_anchor=(legend_anchor_x, 0.5))
        if legend_title:
            legend_kwargs['title'] = legend_title
        fig.legend(
            legend_handles[::-1],
            [label.replace('_', ' ') for label in legend_labels[::-1]],
            **legend_kwargs
        )
        tight_rect = [0, 0, legend_anchor_x, 1]

    if column_subplot is None:
        ax = axes[0]
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
                unique = OrderedDict()
                for handle, label in zip(combined_handles, combined_labels):
                    if label not in unique:
                        unique[label] = handle
                final_labels = list(unique.keys())
                final_handles = list(unique.values())
                legend_kwargs = dict(loc='center left', frameon=False, bbox_to_anchor=(legend_anchor_x, 0.5))
                if legend_title:
                    legend_kwargs['title'] = legend_title
                fig.legend(
                    final_handles,
                    final_labels,
                    **legend_kwargs
                )
                tight_rect = [0, 0, legend_anchor_x, 1]
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

    layout_rect = None
    if tight_rect:
        top_bound = min(tight_rect[3], 0.92)
        layout_rect = [tight_rect[0], tight_rect[1], min(tight_rect[2], 0.96), top_bound]
    else:
        layout_rect = [0, 0, 0.96, 0.92]
    safe_tight_layout(fig, rect=layout_rect, shrink_top=0.04)

    if column_subplot is None:
        if single_panel_title:
            fig.subplots_adjust(top=0.86)
            fig.suptitle(single_panel_title, fontsize=fonttick + 2, fontweight='bold', color='dimgrey')
    else:
        if title:
            fig.suptitle(title, fontsize=fonttick + 2, y=0.97)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

# Disptach plots

def dispatch_plot(df_area=None, filename=None, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom',
                  bottom=0, ylabel=None, title=None, order_stacked=None):
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
    order_stacked : list, optional
        Preferred stacking order for area series. Remaining columns follow afterwards.
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
        stack_order = order_stacked or DEFAULT_FUEL_ORDER
        if stack_order:
            ordered_cols = [c for c in stack_order if c in df_area.columns]
            ordered_cols += [c for c in df_area.columns if c not in ordered_cols]
            df_area = df_area.loc[:, ordered_cols]

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
        # Compute legend layout: 1 row up to 5 entries, 2 rows up to 10, else 3 rows.
        n_items = 0
        if df_area is not None:
            n_items = len(df_area.columns)
        elif df_line is not None:
            n_items = len(df_line.columns)
        rows = 1 if n_items <= 5 else (2 if n_items <= 10 else 3)
        ncol = max(1, int(np.ceil(n_items / rows))) if n_items else 1
        # Add more bottom padding when we need multiple rows.
        bottom_pad = 0.25 if rows == 1 else (0.32 if rows == 2 else 0.38)
        fig.subplots_adjust(bottom=bottom_pad)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol, frameon=False)

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


def dispatch_diff_plot(df_area=None, filename=None, dict_colors=None, figsize=(10, 6), legend_loc='bottom',
                       ylabel=None, title=None, order_stacked=None):
    """
    Generate a dispatch difference plot with separate stacked areas for positive and negative values.

    This plot shows the difference in dispatch between two scenarios, with positive differences
    (increases) stacked above zero and negative differences (decreases) stacked below zero.

    Parameters
    ----------
    df_area : pandas.DataFrame
        DataFrame containing difference data with fuels as columns and time as index.
    filename : str, optional
        Path to save the plot image. If not provided, the plot will be displayed.
    dict_colors : dict, optional
        Dictionary mapping column names to colors for the plot.
    figsize : tuple, default (10, 6)
        Size of the figure in inches.
    legend_loc : str, default 'bottom'
        Location of the legend. Options are 'bottom' or 'right'.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    order_stacked : list, optional
        Preferred stacking order for area series.
    """
    if df_area is None or df_area.empty:
        raise ValueError('No dataframe provided for the plot.')

    fig, ax = plt.subplots(figsize=figsize)

    # Apply stacking order
    stack_order = order_stacked or DEFAULT_FUEL_ORDER
    if stack_order:
        ordered_cols = [c for c in stack_order if c in df_area.columns]
        ordered_cols += [c for c in df_area.columns if c not in ordered_cols]
        df_area = df_area.loc[:, ordered_cols]

    # Create numeric x-axis for plotting
    x = np.arange(len(df_area))

    # Separate positive and negative contributions at each time step
    # For each timestep, stack positive values above zero and negative values below zero
    df_positive = df_area.clip(lower=0)
    df_negative = df_area.clip(upper=0)

    # Stack positive values
    pos_bottom = np.zeros(len(df_area))
    neg_bottom = np.zeros(len(df_area))

    plotted_labels = set()

    for col in df_area.columns:
        color = dict_colors.get(col, 'gray') if dict_colors else None

        # Plot positive part
        pos_values = df_positive[col].values
        if np.any(pos_values > 0):
            label = col if col not in plotted_labels else None
            ax.fill_between(x, pos_bottom, pos_bottom + pos_values,
                          color=color, alpha=0.8, label=label, linewidth=0)
            pos_bottom = pos_bottom + pos_values
            plotted_labels.add(col)

        # Plot negative part
        neg_values = df_negative[col].values
        if np.any(neg_values < 0):
            label = col if col not in plotted_labels else None
            ax.fill_between(x, neg_bottom + neg_values, neg_bottom,
                          color=color, alpha=0.8, label=label, linewidth=0)
            neg_bottom = neg_bottom + neg_values
            plotted_labels.add(col)

    # Add zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Set x-axis limits
    ax.set_xlim(0, len(df_area) - 1)

    pd_index = df_area.index
    format_dispatch_ax(ax, pd_index)

    # Add axis labels and title
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold')
    else:
        ax.set_ylabel('Generation Difference (MW)', fontsize=8.5)

    # Add legend
    if legend_loc == 'bottom':
        n_items = len(df_area.columns)
        rows = 1 if n_items <= 5 else (2 if n_items <= 10 else 3)
        ncol = max(1, int(np.ceil(n_items / rows))) if n_items else 1
        bottom_pad = 0.25 if rows == 1 else (0.32 if rows == 2 else 0.38)
        fig.subplots_adjust(bottom=bottom_pad)

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=ncol, frameon=False)
    elif legend_loc == 'right':
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, frameon=False)

    if title is not None:
        ax.text(
            y=ax.get_ylim()[1] * 1.1,
            x=sum(ax.get_xlim()) / 2,
            s=title,
            ha='center',
            fontsize=8.5
        )

    if filename is not None:
        fig.savefig(filename, bbox_inches=None, pad_inches=0.1, dpi=100)
        plt.close()
    else:
        plt.show()


def make_fuel_dispatchplot(dfs_area, dfs_line, dict_colors, zone, year, scenario, stacked=True,
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
        'pDispatchTechFuel': epm_dict['pDispatchTechFuel'],
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
    make_fuel_dispatchplot(dfs_to_plot_area, dfs_to_plot_line, folder_results / Path('images'), dict_specs['colors'],
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


def make_fuel_dispatch_diff_plot(dfs_area, dict_colors, zone, year, scenario,
                                 filename=None, fuel_grouping=None, select_time=None,
                                 legend_loc='bottom', figsize=(10, 6), ylabel=None, title=None):
    """
    Generates and saves a fuel dispatch difference plot.

    This function prepares data similar to make_fuel_dispatchplot but uses
    dispatch_diff_plot to handle mixed positive/negative values.

    Parameters
    ----------
    dfs_area : dict
        Dictionary containing dataframes for area plots (e.g., generation diff).
    dict_colors : dict
        Dictionary mapping fuel types to colors.
    zone : str
        Zone to filter data for.
    year : int
        Year to filter data for.
    scenario : str
        Scenario label for the diff data.
    filename : str, optional
        Path to save the figure.
    fuel_grouping : dict, optional
        Mapping to create aggregate fuel categories.
    select_time : dict, optional
        Time selection parameters for filtering the data.
    legend_loc : str, default 'bottom'
        Location of the legend.
    figsize : tuple, default (10, 6)
        Size of the figure.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    """

    def remove_na_values(df):
        """Removes na values from a dataframe."""
        df = df.where((df > 1e-6) | (df < -1e-6), np.nan)
        df = df.dropna(axis=1, how='all')
        return df

    def prepare_hourly_dataframe(df, zone, year, scenario, column_stacked, fuel_grouping=None, select_time=None):
        """Transforms a dataframe into format ready for plotting."""
        if 'zone' in df.columns:
            df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]
            df = df.drop(columns=['zone', 'year', 'scenario'])
        else:
            df = df[(df['year'] == year) & (df['scenario'] == scenario)]
            df = df.drop(columns=['year', 'scenario'])

        if column_stacked == 'fuel':
            if fuel_grouping is not None:
                df['fuel'] = df['fuel'].replace(fuel_grouping)

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
        if df is None or df.empty:
            continue
        column_stacked = NAME_COLUMNS.get(key)
        df, temp = prepare_hourly_dataframe(df, zone, year, scenario, column_stacked,
                                            fuel_grouping=fuel_grouping, select_time=select_time)
        tmp_concat_area.append(df)

    if len(tmp_concat_area) > 0:
        df_tot_area = pd.concat(tmp_concat_area, axis=1)
        df_tot_area = df_tot_area.droplevel(0, axis=1)
        df_tot_area = remove_na_values(df_tot_area)
    else:
        return

    if df_tot_area is None or df_tot_area.empty:
        return

    if select_time is None:
        temp = 'all'
    temp = f'{year}_{temp}'
    if filename is not None and isinstance(filename, str):
        filename = filename.split('.')[0] + f'_{temp}.pdf'

    dispatch_diff_plot(df_tot_area, filename, dict_colors=dict_colors, legend_loc=legend_loc,
                       figsize=figsize, ylabel=ylabel, title=title)


# Stacked bar plots

def make_stacked_barplot(df, filename, dict_colors, df_errorbars=None, overlay_df=None, legend_label=None, column_subplot='year',
                              column_stacked='fuel', column_xaxis='scenario',
                              column_value='value', select_subplot=None, stacked_grouping=None, order_scenarios=None, dict_scenarios=None,
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
    overlay_df : pd.DataFrame, optional
        Additional data sharing the same `column_subplot` (when provided), `column_xaxis`, and `column_value`
        used to overlay a line (with markers) on the bars.
    legend_label : str, optional
        Legend label associated with the overlay line. When provided the line is included in the primary legend.
    selected_zone : str
        Zone to select.
    column_subplot : str
        Column for choosing the subplots.
    column_stacked : str
        Column name for choosing the column to stack values.
    column_xaxis : str
        Column for choosing the type of bars inside a given subplot.
    column_value : str
        Column name for the values to be plotted.
    select_subplot : list, optional
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
        Reordering the variables that will be stacked. When ``column_stacked`` is
        ``'fuel'`` and this argument is omitted, the ordering declared in the fuel
        mapping file is applied automatically.
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
    make_stacked_barplot(epm_dict['pCapacityTechFuel'], filename, dict_specs['colors'], selected_zone='Liberia',
                              select_subplot=[2025, 2028, 2030], stacked_grouping=fuel_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} MW'.format(y))

    Stacked bar subplots for reserve evolution:
    filename = Path(RESULTS_FOLDER) / Path('images') / Path('ReserveEvolution.png')
    make_stacked_barplot(epm_dict['pReserveByPlant'], filename, dict_colors=dict_specs['colors'], selected_zone='Liberia',
                              column_subplot='year', column_stacked='fuel', column_xaxis='scenario',
                              select_subplot=[2025, 2028, 2030], stacked_grouping=stacked_grouping, dict_scenarios=scenario_names,
                              order_scenarios=['Baseline', 'High Hydro', 'High Demand', 'LowImport_LowThermal'],
                              format_y=lambda y, _: '{:.0f} GWh'.format(y),
                              order_stacked=['Hydro', 'Oil'], cap=2)
    """
    def stacked_bar_subplot(df, column_stacked, filename, df_errorbars=None, overlay_lookup=None, legend_label=None, dict_colors=None, year_ini=None, order_scenarios=None,
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
        df_errorbars : pandas.DataFrame, optional
            Error bar bounds to display for each bar. Expected columns include ``'error'`` with values
            ``'min'``/``'max'`` alongside the grouping keys.
        overlay_lookup : dict[str, pandas.Series], optional
            Mapping from subplot keys to 1D series indexed by ``column_xaxis`` that will be plotted as a
            line with markers over each bar, sharing the same y-axis.
        legend_label : str, optional
            Legend label for the overlay line when present.
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

        if order_stacked is None and column_stacked == 'fuel' and DEFAULT_FUEL_ORDER:
            order_stacked = DEFAULT_FUEL_ORDER

        handles, labels = None, None
        overlay_label_used = False
        bar_annotations = bar_annotations or {}
        for k, key in enumerate(list_keys):
            ax = axes[k]
            overlay_series = None

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

                if overlay_lookup is not None and not juxtaposed:
                    overlay_series = overlay_lookup.get(key)
                    if overlay_series is None and isinstance(key, tuple):
                        overlay_series = overlay_lookup.get(tuple(key))
                    if overlay_series is None and len(list_keys) == 1 and overlay_lookup:
                        overlay_series = next(iter(overlay_lookup.values()))
                    if overlay_series is not None:
                        overlay_series = overlay_series.copy()
                        if dict_scenarios is not None:
                            overlay_series.index = overlay_series.index.map(lambda x: dict_scenarios.get(x, x))
                        if order_scenarios is not None:
                            overlay_series = overlay_series.reindex(order_scenarios)
                        overlay_series = overlay_series.reindex(df_temp.index)

                if not juxtaposed:
                    df_temp.plot(ax=ax, kind='bar', stacked=stacked, linewidth=0,
                                color=dict_colors if dict_colors else None)
                else:  # stacked columns become one next to each other
                    df_temp.T.plot(ax=ax, kind='bar', stacked=False, linewidth=0,
                                color=dict_colors if dict_colors else None)

                # Plot error bars if provided
                df_bar_totals = df_temp.sum(axis=1)

                if overlay_series is not None and overlay_series.notna().any():
                    bar_centers = ax.get_xticks()
                    x_vals = []
                    y_vals = []
                    for idx_pos, idx_name in enumerate(df_temp.index):
                        if idx_name not in overlay_series.index:
                            continue
                        value = overlay_series.loc[idx_name]
                        if pd.isna(value):
                            continue
                        x_coord = bar_centers[idx_pos] if idx_pos < len(bar_centers) else idx_pos
                        x_vals.append(x_coord)
                        y_vals.append(value)
                    if x_vals:
                        use_label = legend_label if legend_label and not overlay_label_used else '_nolegend_'
                        ax.plot(x_vals, y_vals, color='tab:red', marker='o', linewidth=1.5, zorder=4, label=use_label)
                        if legend_label and not overlay_label_used:
                            overlay_label_used = True

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

    if column_xaxis is None:
        print('column_xaxis cannot be None, but column_subplot can. Automatically inverting.')
        column_xaxis = column_subplot
        column_subplot = None
    
    overlay_lookup = None
    overlay_source = None
    if overlay_df is not None:
        required_columns = {column_xaxis, column_value}
        if column_subplot is not None:
            required_columns.add(column_subplot)
        missing_columns = required_columns.difference(overlay_df.columns)
        if missing_columns:
            missing_fmt = ', '.join(sorted(missing_columns))
            raise ValueError(f"overlay_df is missing required columns: {missing_fmt}")
        overlay_source = overlay_df.copy()

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
        if column_subplot is not None:
            grouping_keys.append(column_subplot)
        if column_xaxis is not None:
            grouping_keys.append(column_xaxis)

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

        if column_subplot is not None:
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
                elif bar_key is None and column_xaxis is not None:
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

    if column_subplot is not None:
        if column_stacked is not None:
            df = (df.groupby([column_subplot, column_stacked, column_xaxis], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_stacked, column_xaxis, column_subplot]).squeeze().unstack(column_subplot)
        else:
            df = (df.groupby([column_subplot, column_xaxis], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_xaxis, column_subplot]).squeeze().unstack(column_subplot)
        if df_errorbars is not None:
            if not juxtaposed:  # we sum over the stacked column
                df_errorbars = (df_errorbars.groupby([column_subplot, 'error', column_xaxis], observed=False)[
                          column_value].sum().reset_index())
                df_errorbars = df_errorbars.set_index(['error', column_xaxis, column_subplot]).squeeze().unstack(column_subplot)
            else:  # we keep the stacked column
                df_errorbars = (df_errorbars.groupby([column_subplot, 'error', column_stacked, column_xaxis], observed=False)[
                          column_value].sum().reset_index())
                df_errorbars = df_errorbars.set_index(['error', column_stacked, column_xaxis, column_subplot]).squeeze().unstack(
                    column_subplot)
    else:  # no subplots in this case
        if column_stacked is not None:
            df = (df.groupby([column_stacked, column_xaxis], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_stacked, column_xaxis])
        else:
            df = (df.groupby([column_xaxis], observed=False)[column_value].sum().reset_index())
            df = df.set_index([column_xaxis])
        if df_errorbars is not None:
            df_errorbars = (df_errorbars.groupby(['error', column_xaxis], observed=False)[column_value].sum().reset_index())
            df_errorbars = df_errorbars.set_index(['error', column_xaxis])

    if overlay_source is not None:
        if column_subplot is not None:
            overlay_grouped = (overlay_source.groupby([column_subplot, column_xaxis], observed=False)[column_value]
                               .sum()
                               .reset_index())
            overlay_lookup = {
                subplot_key: subgroup.set_index(column_xaxis)[column_value]
                for subplot_key, subgroup in overlay_grouped.groupby(column_subplot, observed=False)
            }
        else:
            overlay_series = (overlay_source.groupby([column_xaxis], observed=False)[column_value]
                              .sum()
                              .rename(column_value))
            column_key = next(iter(df.columns), column_value)
            overlay_lookup = {column_key: overlay_series}

    if select_subplot is not None:
        df = df.loc[:, [i for i in df.columns if i in select_subplot]]
        if overlay_lookup is not None and column_subplot is not None:
            overlay_lookup = {k: v for k, v in overlay_lookup.items() if k in select_subplot}

    if overlay_lookup is not None and not overlay_lookup:
        overlay_lookup = None

    if bar_annotations is not None:
        if column_subplot is None:
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
        stacked_bar_subplot(df, column_stacked, filename, dict_colors=dict_colors, df_errorbars=df_errorbars, overlay_lookup=overlay_lookup, legend_label=legend_label, format_y=format_y,
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
        datetime_coerced = pd.to_datetime(pd.Series(values), errors='coerce', format='mixed')
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


def heatmap_difference_plot(
    data,
    filename=None,
    percentage=False,
    color_by_percentage=True,
    reference='baseline',
    reference_label='Reference'
):
    """
    Plots a heatmap showing differences from a reference scenario with color scales defined per column.

    Parameters
    ----------
    data : pd.DataFrame
        Table with metrics as rows and scenarios as columns.
    filename : str
        Path to save the plot.
    percentage : bool, optional
        Whether to show differences as percentages in the annotations.
    color_by_percentage : bool, optional
        If True (default), the colour scale is based on percentage differences relative to the reference.
        When False, absolute differences drive the colour scale.
    reference : str, optional
        Scenario name used as the reference (defaults to ``'baseline'``).
    reference_label : str, optional
        Display label for the reference column in the heatmap.
    """

    if reference not in data.columns:
        raise ValueError(f"Reference scenario '{reference}' not found in data columns.")

    # Ensure reference column is first
    column_order = [reference] + [col for col in data.columns if col != reference]
    data = data[column_order]

    # Calculate differences from the reference scenario
    baseline_values = data[reference]
    diff_absolute = data.subtract(baseline_values, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_percentage = diff_absolute.divide(baseline_values.replace({0: np.nan}), axis=0)
    diff_percentage = diff_percentage.replace([np.inf, -np.inf], np.nan)

    # Combine differences and baseline values for annotations
    annotations = data.map(lambda x: f"{x:,.0f}")  # Format baseline values
    # Format differences in percentage
    if percentage:
        def _fmt_pct(x):
            if pd.isna(x):
                return " (n/a)"
            return f" ({x:+,.0%})"
        diff_annotations = diff_percentage.applymap(_fmt_pct)
    else:
        diff_annotations = diff_absolute.map(lambda x: f" ({x:+,.0f})")
    # Drop difference annotation for the reference column since the delta is always zero
    diff_annotations[reference] = ""
    combined_annotations = annotations + diff_annotations  # Combine both

    # Normalize the color scale by column
    color_matrix = diff_percentage if color_by_percentage else diff_absolute
    color_matrix = color_matrix.fillna(0.0)
    diff_normalized = color_matrix.copy()
    for column in color_matrix.columns:
        col = color_matrix[column]
        col_min = col.min()
        col_max = col.max()
        if np.isclose(col_max, col_min):
            diff_normalized[column] = 0
        else:
            diff_normalized[column] = (col - col_min) / (col_max - col_min)

    # Rename reference column for display
    diff_norm_plot = diff_normalized.rename(columns={reference: reference_label})
    combined_annotations_plot = combined_annotations.rename(columns={reference: reference_label})

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap
    sns.heatmap(
        diff_norm_plot,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=combined_annotations_plot,  # Show baseline values and differences
        fmt="",  # Disable default formatting
        linewidths=0.5,
        ax=ax,
        cbar=False  # Remove color bar
    )

    # Highlight reference column with a light grey background
    ref_col_idx = list(diff_norm_plot.columns).index(reference_label)
    n_rows = diff_norm_plot.shape[0]
    for row_idx in range(n_rows):
        rect = Rectangle(
            (ref_col_idx, row_idx),
            1,
            1,
            facecolor='lightgrey',
            edgecolor='grey',
            linewidth=0.3,
            alpha=0.35,
            zorder=2,
            clip_on=False
        )
        ax.add_patch(rect)

    # Customize the axes
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    bold_markers = (
        'NPV of system cost',
        'Capacity - Total',
        'New capacity - Total',
        'Transmission capacity',
        'New transmission capacity',
        'Cumulative CAPEX - Total'
    )
    for label in ax.get_yticklabels():
        text = label.get_text()
        if any(marker in text for marker in bold_markers):
            label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        if label.get_text() == reference_label:
            label.set_fontweight('bold')
    ax.set_xlabel("")
    ax.set_ylabel("")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def make_heatmap_plot(
    epm_results,
    filename,
    percentage=False,
    scenario_order=None,
    year=None,
    zone_list=None,
    rename_columns=None,
    reference='baseline'
):
    """
    Build a scenario-comparison heatmap summarising core system metrics.

    The summary includes, in order:
        1. Discounted system costs (``pCostsSystem``), with the NPV column shown first.
        2. Total installed capacity in the final model year.
        3. New capacity additions by fuel in the final model year.
        4. Total available transmission capacity in the final year (no double counting).
        5. New transmission capacity added in the final model year.
        6. Cumulative CAPEX by component up to the final year.
        7. Yearly Investment costs per zone ($/MWh) for the final model year.

    Parameters
    ----------
    epm_results : dict
        Dictionary with post-processed EPM outputs (DataFrames).
    filename : str or Path
        Where to save the heatmap. If None, the figure is displayed.
    percentage : bool, optional
        When True, differences are displayed in percentage terms relative to the reference scenario.
    scenario_order : list-like, optional
        Desired ordering of scenarios.
    discount_rate, required_keys, fuel_capa_list, fuel_gen_list, summary_metrics_list, rows_index : deprecated
        Maintained only for backwards compatibility; they have no effect.
    zone_list : iterable, optional
        Optional subset of zones to include when aggregating system metrics.
    year : int, optional
        Year to use for annual metrics. Defaults to the maximum year found in each dataset.
    rename_columns : dict, optional
        Mapping applied to the final summary columns.
    reference : str, optional
        Scenario name to use as the reference column (defaults to ``'baseline'``).
    """

    TRADE_ATTRS = [
        "Import costs with internal zones: $m",
        "Import costs with external zones: $m",
        "Export revenues with internal zones: $m",
        "Export revenues with external zones: $m",
        "Trade shared benefits: $m",
        "Import costs with internal zones: $/MWh",
        "Import costs with external zones: $/MWh",
        "Export revenues with internal zones: $/MWh",
        "Export revenues with external zones: $/MWh",
        "Trade shared benefits: $/MWh"
    ]

    RESERVE_ATTRS = [
        "Unmet country spinning reserve costs: $m",
        "Unmet country planning reserve costs: $m",
        "Unmet country CO2 backstop cost: $m",
        "Unmet system planning reserve costs: $m",
        "Unmet system spinning reserve costs: $m",
        "Unmet country spinning reserve costs: $/MWh",
        "Unmet country planning reserve costs: $/MWh",
        "Unmet country CO2 backstop cost: $/MWh",
        "Unmet system planning reserve costs: $/MWh",
        "Unmet system spinning reserve costs: $/MWh"
    ]

    def _get_dataframe(key: str):
        if key not in epm_results:
            print(f"Warning: '{key}' is missing in epm_results; skipping its heatmap section.")
            return None
        df = epm_results[key]
        if not isinstance(df, pd.DataFrame):
            print(f"Warning: Expected '{key}' to be a pandas DataFrame, got {type(df)}; skipping.")
            return None
        return df.copy()

    def _merge_attribute_group(df: pd.DataFrame, attributes: list, group_name: str) -> pd.DataFrame:
        if 'attribute' not in df.columns:
            return df
        mask = df['attribute'].isin(attributes)
        if not mask.any():
            return df
        group_cols = [col for col in df.columns if col not in {'attribute', 'value'}]
        aggregated = (
            df.loc[mask]
            .groupby(group_cols, observed=False)['value']
            .sum()
            .reset_index()
        )
        suffix = ''
        for attr in attributes:
            if attr in df['attribute'].values and ':' in attr:
                suffix = attr.split(':', 1)[1].strip()
                break
        label = f"{group_name}: {suffix}" if suffix else group_name
        aggregated['attribute'] = label
        df_remaining = df.loc[~mask]
        df_merged = pd.concat([df_remaining, aggregated], ignore_index=True)
        return df_merged

    def _format_cost_columns(df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
        formatted = {}
        for column in df.columns:
            base = column.split(':')[0].strip()
            formatted[column] = f'{base} ({unit_label})'
        return df.rename(columns=formatted)

    def _filter_zone(df: pd.DataFrame) -> pd.DataFrame:
        if zone_list is None or 'zone' not in df.columns:
            return df
        return df[df['zone'].isin(zone_list)]

    frames = []

    # 1. Discounted system costs (pCostsSystem)
    costs_system = _get_dataframe('pCostsSystem')
    if costs_system is not None:
        costs_system = _merge_attribute_group(costs_system, TRADE_ATTRS, "Trade costs")
        costs_system = _merge_attribute_group(costs_system, RESERVE_ATTRS, "Reserve costs")
        costs_pivot_raw = costs_system.pivot_table(index='scenario', columns='attribute', values='value', aggfunc='sum')
        ordered_cost_columns = list(costs_pivot_raw.columns)
        npv_label = 'NPV of system cost: $m'
        if npv_label in ordered_cost_columns:
            ordered_cost_columns = [npv_label] + [col for col in ordered_cost_columns if col != npv_label]
        costs_pivot = costs_pivot_raw.reindex(columns=ordered_cost_columns)
        costs_pivot = _format_cost_columns(costs_pivot, 'M$')
        frames.append(costs_pivot)

    # Determine final year helper
    def _resolve_year(df: pd.DataFrame) -> int:
        if 'year' not in df.columns:
            raise KeyError("Expected a 'year' column when computing annual aggregates.")
        available_years = df['year'].dropna().unique()
        if year is not None:
            if year not in available_years:
                raise ValueError(f"Requested year {year} not available. Available years: {sorted(available_years)}")
            return int(year)
        return int(max(available_years))

    # 2. Total system capacity in the final year
    capacity_fuel_all = _get_dataframe('pCapacityTechFuel')
    if capacity_fuel_all is not None:
        capacity_year = _resolve_year(capacity_fuel_all)
        capacity_fuel = _filter_zone(capacity_fuel_all)
        capacity_summary = (
            capacity_fuel[capacity_fuel['year'] == capacity_year]
            .groupby(['scenario', 'fuel'], observed=False)['value']
            .sum()
            .groupby(level='scenario')
            .sum()
        )
        capacity_total_col = f'Capacity - Total {capacity_year} (MW)'
        capacity_summary = capacity_summary.to_frame(capacity_total_col)
        frames.append(capacity_summary)

    # 3. New capacity additions by fuel (aggregated across zones) in the final year
    new_capacity_all = _get_dataframe('pNewCapacityTechFuel')
    if new_capacity_all is not None:
        new_capacity_year = _resolve_year(new_capacity_all)
        new_capacity = _filter_zone(new_capacity_all)
        new_capacity_summary = (
            new_capacity[new_capacity['year'] == new_capacity_year]
            .groupby(['scenario', 'fuel'], observed=False)['value']
            .sum()
            .unstack('fuel')
            .fillna(0)
        )
        new_fuel_columns = list(new_capacity_summary.columns)
        renamed_new_capacity = {col: f'New capacity - {col} (MW)' for col in new_fuel_columns}
        new_capacity_summary = new_capacity_summary.rename(columns=renamed_new_capacity)
        new_capacity_total_col = f'New capacity - Total {new_capacity_year} (MW)'
        new_capacity_summary[new_capacity_total_col] = new_capacity_summary.sum(axis=1)
        ordered_new_capacity_cols = [new_capacity_total_col] + [renamed_new_capacity[col] for col in new_fuel_columns]
        new_capacity_summary = new_capacity_summary[ordered_new_capacity_cols]
        frames.append(new_capacity_summary)

    # 4. Transmission capacity (no double counting) in final year
    # optional argument for 1-zone model
    if 'pAnnualTransmissionCapacity' in epm_results.keys():
        transmission_all = _get_dataframe('pAnnualTransmissionCapacity')
        if transmission_all is not None:
            transmission_year = _resolve_year(transmission_all)
            transmission = transmission_all.copy()
            if zone_list is not None and {'zone', 'z2'}.issubset(transmission.columns):
                transmission = transmission[
                    transmission['zone'].isin(zone_list) | transmission['z2'].isin(zone_list)
                ]
            transmission_year_df = transmission[transmission['year'] == transmission_year].copy()
            if not transmission_year_df.empty:
                transmission_year_df['pair'] = transmission_year_df.apply(
                    lambda row: tuple(sorted((row['zone'], row['z2']))), axis=1
                )
                transmission_summary = (
                    transmission_year_df.groupby(['scenario', 'pair'], observed=False)['value']
                    .max()
                    .groupby('scenario')
                    .sum()
                    .to_frame(f'Transmission capacity {transmission_year} (MW)')
                )
            else:
                transmission_summary = pd.DataFrame(
                    columns=[f'Transmission capacity {transmission_year} (MW)']
                ).astype(float)
            frames.append(transmission_summary)

    if 'pNewTransmissionCapacity' in epm_results.keys():
        new_transmission_all = _get_dataframe('pNewTransmissionCapacity')
        if new_transmission_all is not None:
            new_transmission_year = _resolve_year(new_transmission_all)
            new_transmission = new_transmission_all.copy()
            if zone_list is not None and {'zone', 'z2'}.issubset(new_transmission.columns):
                new_transmission = new_transmission[
                    new_transmission['zone'].isin(zone_list) | new_transmission['z2'].isin(zone_list)
                ]
            new_transmission_year_df = new_transmission[new_transmission['year'] == new_transmission_year].copy()
            if not new_transmission_year_df.empty:
                if {'zone', 'z2'}.issubset(new_transmission_year_df.columns):
                    new_transmission_year_df['pair'] = new_transmission_year_df.apply(
                        lambda row: tuple(sorted((row['zone'], row['z2']))), axis=1
                    )
                    new_transmission_summary = (
                        new_transmission_year_df.groupby(['scenario', 'pair'], observed=False)['value']
                        .max()
                        .groupby('scenario')
                        .sum()
                        .to_frame(f'New transmission capacity {new_transmission_year} (MW)')
                    )
                else:
                    new_transmission_summary = (
                        new_transmission_year_df.groupby('scenario', observed=False)['value']
                        .sum()
                        .to_frame(f'New transmission capacity {new_transmission_year} (MW)')
                    )
            else:
                new_transmission_summary = pd.DataFrame(
                    columns=[f'New transmission capacity {new_transmission_year} (MW)']
                ).astype(float)
            frames.append(new_transmission_summary)

    # 6. Cumulative CAPEX by component up to final year
    capex_component_all = _get_dataframe('pCapexInvestmentComponent')
    if capex_component_all is not None:
        capex_year = _resolve_year(capex_component_all)
        capex_component = _filter_zone(capex_component_all)
        capex_cumulative = (
            capex_component[capex_component['year'] <= capex_year]
            .groupby(['scenario', 'attribute'], observed=False)['value']
            .sum()
            .unstack('attribute')
            .fillna(0)
            / 1e6  # convert USD to million USD
        )
        capex_columns = list(capex_cumulative.columns)
        capex_cumulative.columns = [f'CAPEX - {col} (M$)' for col in capex_columns]
        capex_total_col = f'Cumulative CAPEX - Total {capex_year} (M$)'
        capex_cumulative[capex_total_col] = capex_cumulative.sum(axis=1)
        capex_cumulative = capex_cumulative[[capex_total_col] + [col for col in capex_cumulative.columns if col != capex_total_col]]
        frames.append(capex_cumulative)

    # 7. Yearly generation cost per zone (last year, $/MWh)
    if 'pYearlyGenCostZonePerMWh' in epm_results.keys():
        yearly_cost_all = _get_dataframe('pYearlyGenCostZonePerMWh')
        if yearly_cost_all is not None:
            yearly_cost_year = _resolve_year(yearly_cost_all)
            yearly_cost = yearly_cost_all.copy()
            if zone_list is not None and 'zone' in yearly_cost.columns:
                yearly_cost = yearly_cost[yearly_cost['zone'].isin(zone_list)]
            yearly_cost = yearly_cost[yearly_cost['year'] == yearly_cost_year]
            if 'attribute' in yearly_cost.columns:
                # Target rows that already carry $/MWh information; otherwise keep everything.
                per_mwh_mask = yearly_cost['attribute'].str.contains('/MWh', case=False, na=False)
                if per_mwh_mask.any():
                    yearly_cost = yearly_cost[per_mwh_mask]
            if not yearly_cost.empty:
                yearly_cost_summary = (
                    yearly_cost.groupby(['scenario', 'zone'], observed=False)['value']
                    .sum()
                    .unstack('zone')
                    .fillna(0)
                )
                yearly_cost_summary.columns = [
                    f'Generation cost - {zone} ({yearly_cost_year}) $/MWh'
                    for zone in yearly_cost_summary.columns
                ]
                frames.append(yearly_cost_summary)

    if not frames:
        print("Warning: No data available to build the heatmap summary; skipping plot.")
        return

    # Align scenario index across all frames
    scenario_index = frames[0].index
    for frame in frames[1:]:
        scenario_index = scenario_index.union(frame.index)
    frames = [frame.reindex(scenario_index) for frame in frames]

    summary = pd.concat(frames, axis=1)
    summary = summary.fillna(0.0).astype(float)

    if rename_columns:
        summary = summary.rename(columns=rename_columns)

    if reference not in summary.index:
        raise ValueError(f"Reference scenario '{reference}' not present in the summary.")

    # Ensure reference scenario is first in row order
    summary = summary.loc[[reference] + [idx for idx in summary.index if idx != reference]]

    if scenario_order is not None:
        ordered = [scenario for scenario in scenario_order if scenario in summary.index]
        ordered += [scenario for scenario in summary.index if scenario not in ordered]
        ordered = [reference] + [idx for idx in ordered if idx != reference]
        summary = summary.loc[ordered]

    # Rotate so metrics are rows and scenarios are columns
    summary = summary.T
    summary = summary[[reference] + [col for col in summary.columns if col != reference]]

    heatmap_difference_plot(
        summary,
        filename,
        percentage=percentage,
        reference=reference,
        reference_label='Reference'
    )

# Line plots

def make_line_plot(
    df,
    filename,
    column_xaxis,
    y_column,
    preserve_x_spacing=False,
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
    preserve_x_spacing : bool, default False
        Plot x values at their numeric/datetime positions to preserve spacing; when False, values are treated as categories.
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
        use_numeric_spacing = preserve_x_spacing
        if use_numeric_spacing:
            try:
                numeric_index = pd.to_numeric(pivot.index, errors='coerce')
                if not np.isnan(numeric_index).any():
                    x_positions = numeric_index.to_numpy()
                else:
                    use_numeric_spacing = False
            except Exception:
                use_numeric_spacing = False

        for col in pivot.columns:
            series = pivot[col]
            if series.isna().all():
                continue
            color = dict_colors.get(col) if dict_colors is not None else None
            label = str(col)
            line, = ax.plot(x_positions, series.values, marker='o', color=color, label=label)
            all_handles.append(line)
            all_labels.append(label)

        if index_values:
            if len(index_values) > max_ticks:
                positions = np.linspace(0, len(index_values) - 1, max_ticks, dtype=int)
            else:
                positions = np.arange(len(index_values))
            ax.set_xticks([x_positions[i] for i in positions])
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
