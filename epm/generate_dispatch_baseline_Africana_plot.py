import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPATCH_ATTRIBUTE_CATEGORIES = {
    'Demand',
    'Exports',
    'Imports',
    'Storage Charge',
    'Unmet demand'
}

RENAME_COLUMNS = {
    'c': 'country',
    'c_0': 'country',
    'y': 'year',
    'v': 'value',
    's': 'scenario',
    'uni': 'attribute',
    'z': 'zone',
    'z_0': 'zone',
    'g': 'generator',
    'gen': 'generator',
    'f': 'fuel',
    'q': 'season',
    'd': 'day',
    't': 't',
    'sumhdr': 'attribute',
    'genCostCmp': 'attribute',
    'uni_1': 'tech',
    'uni_2': 'fuel',
    'y_3': 'year',
    'uni_0': 'attribute'
}


def read_plot_colors(resource_dir: Path) -> dict[str, str]:
    colors_file = resource_dir / 'colors.csv'
    colors = {}
    with colors_file.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            processing = row.get('Processing')
            color = row.get('Color')
            if processing is None or color is None:
                continue
            processing = processing.strip()
            color = color.strip()
            if not processing or not color or processing.startswith('#'):
                continue
            colors[processing] = color
    return colors


def load_dispatch_complete(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_COLUMNS)
    return df


def parse_techfuel_to_fuel(uni_value: str) -> str:
    if isinstance(uni_value, str) and '-' in uni_value:
        return uni_value.split('-', 1)[1]
    return uni_value


def prepare_dispatch_plot_data(df: pd.DataFrame, zone: str, year: int, scenario: str):
    df = df.copy()
    df = df[(df['zone'] == zone) & (df['year'] == year) & (df['scenario'] == scenario)]

    if df.empty:
        raise ValueError(f'No records found for zone={zone}, year={year}, scenario={scenario}')

    df_attributes = df[df['attribute'].isin(DISPATCH_ATTRIBUTE_CATEGORIES)].copy()
    df_demands = df_attributes[df_attributes['attribute'] == 'Demand']

    df_techfuel = df[~df['attribute'].isin(DISPATCH_ATTRIBUTE_CATEGORIES)].copy()
    df_techfuel['fuel'] = df_techfuel['attribute'].apply(parse_techfuel_to_fuel)

    if df_techfuel.empty:
        raise ValueError('No techfuel dispatch records found in pDispatchComplete.csv for the selected filter.')
    if df_demands.empty:
        raise ValueError('No demand dispatch records found in pDispatchComplete.csv for the selected filter.')

    df_area = (
        df_techfuel
        .groupby(['season', 'day', 't', 'fuel'], observed=False)['value']
        .sum()
        .reset_index()
        .pivot_table(index=['season', 'day', 't'], columns='fuel', values='value', fill_value=0)
    )
    df_line = (
        df_demands
        .groupby(['season', 'day', 't', 'attribute'], observed=False)['value']
        .sum()
        .reset_index()
        .pivot_table(index=['season', 'day', 't'], columns='attribute', values='value', fill_value=0)
    )

    df_area = df_area.sort_index()
    df_line = df_line.sort_index()

    return df_area, df_line


def format_dispatch_ax(ax, pd_index):
    n_rep_days = len(pd_index.get_level_values('day').unique())
    dispatch_seasons = pd_index.get_level_values('season').unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

    for d in range(total_days):
        x_d = 24 * d
        is_end_of_season = d % n_rep_days == 0
        linestyle = '-' if is_end_of_season else '--'
        ax.axvline(x=x_d, color='slategrey', linestyle=linestyle, linewidth=0.8)
        ax.text(
            x=x_d + 12,
            y=y_max * 0.99,
            s=f'd{(d % n_rep_days) + 1}',
            ha='center',
            fontsize=7
        )

    season_x_positions = [24 * n_rep_days * s + 12 * n_rep_days for s in range(len(dispatch_seasons))]
    ax.set_xticks(season_x_positions)
    ax.set_xticklabels(dispatch_seasons, fontsize=8)
    ax.set_xlim(left=0, right=24 * total_days)
    ax.set_xlabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)


def dispatch_plot(df_area, filename, dict_colors=None, df_line=None, figsize=(10, 6), legend_loc='bottom', bottom=0):
    fig, ax = plt.subplots(figsize=figsize)

    if df_area is not None and not df_area.empty:
        df_area.plot.area(ax=ax, stacked=True, color=dict_colors, linewidth=0)
        pd_index = df_area.index
    else:
        pd_index = df_line.index if df_line is not None else None

    if df_line is not None:
        if df_area is not None and df_area is not None:
            assert df_area.index.equals(df_line.index), 'Area and line dataframes must share the same index.'
        df_line.plot(ax=ax, color=dict_colors)
        pd_index = df_line.index

    if pd_index is None:
        raise ValueError('No data provided to plot.')

    format_dispatch_ax(ax, pd_index)
    ax.set_ylabel('Generation (MW)', fontsize=8.5)
    if bottom is not None:
        ax.set_ylim(bottom=bottom)

    n_items = len(df_area.columns) if df_area is not None else len(df_line.columns)
    rows = 1 if n_items <= 5 else (2 if n_items <= 10 else 3)
    ncol = max(1, int(np.ceil(n_items / rows))) if n_items else 1
    bottom_pad = 0.25 if rows == 1 else (0.32 if rows == 2 else 0.38)
    fig.subplots_adjust(bottom=bottom_pad)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol, frameon=False)

    fig.savefig(filename, bbox_inches=None, pad_inches=0.1, dpi=100)
    plt.close(fig)


def main():
    csv_path = r"C:\Users\wb590499\Documents\Projects\EPM\epm\output\simulations_run_20260603_184842\baseline\output_csv\pDispatchComplete.csv"
    output_pdf = r"C:\Users\wb590499\Documents\Projects\EPM\epm\output\simulations_run_20260603_184842\img\5_dispatch\Dispatch_baseline_Africana_all_seasons_2025_dry_wet1_2_3_4.pdf"

    out_dir = Path(output_pdf).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dispatch_complete(csv_path)
    df_area, df_line = prepare_dispatch_plot_data(df, zone='Africana', year=2025, scenario='baseline')

    resource_dir = Path(__file__).resolve().parent / 'resources' / 'postprocess'
    dict_colors = read_plot_colors(resource_dir)

    dispatch_plot(
        df_area=df_area,
        filename=output_pdf,
        dict_colors=dict_colors,
        df_line=df_line,
        legend_loc='bottom',
        bottom=0,
    )

    print(f'Wrote dispatch plot: {output_pdf}')


if __name__ == '__main__':
    main()
