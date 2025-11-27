import matplotlib

# Force non-interactive backend to avoid GUI requirements during batch runs (e.g., Snakemake)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_heatmap(df, tech, path=None, kind="daily", vmin=None, vmax=None):
    """
    Plot a capacity-factor heatmap.

    Inputs
    - df: pd.DataFrame or dict. For kind="daily", must contain columns season, day, hour and one column per zone (numeric).
      For kind="annual", pass either a DataFrame with CtryName and avg_CF or a dict keyed by tech -> that DataFrame.
    - tech: str label used in plot titles.
    - path: optional output filepath; if None the figure is shown.
    - kind: "daily" for season/day heatmap, "annual" for average CF by country.
    - vmin/vmax: optional colormap limits for the daily heatmap (shared across load/VRE plots).
    """
    if kind not in {"daily", "annual"}:
        raise ValueError('kind must be "daily" or "annual"')

    if kind == "daily":
        daily_df = (
            df.groupby(['season', 'day'], as_index=False)
            .mean(numeric_only=True)
            .drop(columns='hour')
        )
        daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)
        tmp = daily_df.sort_values(['season', 'day']).copy()

        # Keep only the numeric columns for the heatmap; strip helper columns after computing ticks
        heatmap_data = tmp.drop(columns=['season', 'day', 'season-day'], errors='ignore').copy()
        heatmap_data.index = tmp['season-day']

        season_labels = tmp['season'].values
        x_positions = []
        x_labels = []
        for i in range(1, len(season_labels)):
            if season_labels[i] != season_labels[i - 1]:
                x_positions.append(i)
                x_labels.append(season_labels[i])
        x_positions = [0] + x_positions
        x_labels = [season_labels[0]] + x_labels

        plt.figure(figsize=(12, 6))
        heatmap_kwargs = {'cmap': 'YlGnBu', 'xticklabels': False}
        if vmin is not None:
            heatmap_kwargs['vmin'] = vmin
        if vmax is not None:
            heatmap_kwargs['vmax'] = vmax
        sns.heatmap(heatmap_data.T, **heatmap_kwargs)
        plt.xticks(ticks=x_positions, labels=x_labels, rotation=0)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(f"Heatmap of Daily {tech} Capacity Factor")
    else:
        cf_stats = df if isinstance(df, pd.DataFrame) else df[tech]
        cf_lcoe_stats_pivot = cf_stats.set_index('CtryName')[['avg_CF']]
        plt.figure(figsize=(10, 6))
        sns.heatmap(cf_lcoe_stats_pivot, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Average Annual CF'})
        plt.title('Average Annual Capacity Factor by Country')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight', dpi=300 if kind == "annual" else None)
        plt.close()
    else:
        plt.show()


def make_boxplot(df, tech, path=None, value_label=None):
    """
    Boxplot of daily average capacity factors by season and zone.

    Inputs
    - df: pd.DataFrame with columns season, day, hour and one numeric column per zone.
    - tech: str label used in plot titles.
    - path: optional output filepath; if None the figure is shown.
    """
    df_energy = df.copy()
    daily_df = df_energy.groupby(['season', 'day'], as_index=False).mean(numeric_only=True).drop(columns='hour')
    daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)

    tmp = daily_df.sort_values(['season', 'day']).copy()

    value_cols = [
        col for col in tmp.columns
        if col not in ['zone', 'season', 'day', 'hour', 'season-day'] and pd.api.types.is_numeric_dtype(tmp[col])
    ]

    melted = tmp.melt(
        id_vars=['season', 'day'],
        value_vars=value_cols,
        var_name='zone',
        value_name='daily_mean'
    )

    # If season values look like months (1-12), relabel legend/title accordingly
    season_numeric = pd.to_numeric(melted['season'], errors='coerce')
    season_order_numeric = sorted(season_numeric.dropna().unique())
    season_ints = [int(s) for s in season_order_numeric if float(s).is_integer()]
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    is_month_data = set(season_ints) == set(month_names.keys())

    hue_col = "season"
    hue_order = None
    legend_title = "Season"
    title_dimension = "Season"
    if is_month_data:
        melted['season_label'] = season_numeric.map(lambda v: month_names.get(int(v)) if pd.notna(v) else None)
        hue_col = "season_label"
        hue_order = [month_names[int(s)] for s in season_order_numeric if int(s) in month_names]
        legend_title = "Month"
        title_dimension = "Month"

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted, x="zone", y="daily_mean", hue=hue_col, hue_order=hue_order)
    plt.xticks(rotation=45)

    plt.legend(
        title=legend_title,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )
    plt.ylabel("")
    plt.xlabel("")

    # Set ymin to 0
    plt.ylim(bottom=0)

    # Format x-ticks to show only the zone name (keep full labels to avoid truncation of slugs)
    xtick_labels = [str(col) for col in melted['zone'].unique()]
    plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=45)

    plt.title(f"Distribution of Daily {tech} Capacity Factor by {title_dimension} and Zone")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def approximate_monthly_cf(hourly_row):
    """
    Convert an 8760-hour capacity-factor profile to monthly averages.

    Inputs
    - hourly_row: sequence-like of length 8760 with hourly numeric values.
    """
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # hours/month
    monthly_cf = []
    cursor = 0
    for h in hours_per_month:
        monthly_cf.append(hourly_row[cursor:cursor+h].mean())
        cursor += h
    return monthly_cf

def make_monthly_heatmap(hourly_profiles, tech, filename=None):
    """
    Heatmap of monthly average capacity factors by country.

    Inputs
    - hourly_profiles: dict-like keyed by technology; each value is a DataFrame with column CtryName
      and 8760 hourly columns.
    - tech: technology key to select from hourly_profiles.
    - filename: optional output filepath; if None the figure is shown.
    """
    monthly_cf_dict = {}
    for _, row in hourly_profiles[tech].iterrows():
        country = row['CtryName']
        hourly_values = row.drop('CtryName').astype(float).values
        monthly_cf_dict[country] = approximate_monthly_cf(hourly_values)

    monthly_cf_df = pd.DataFrame.from_dict(monthly_cf_dict, orient='index',
                                           columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(monthly_cf_df, cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Monthly CF'})
    plt.title('Monthly Average Capacity Factor by Country')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()
