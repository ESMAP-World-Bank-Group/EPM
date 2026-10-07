"""
EPM Dashboard — Reusable Plotly chart builders.
All functions return a plotly.graph_objects.Figure.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FUEL_COLORS, FUEL_ORDER, COST_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_white"


def _ordered_fuels(fuels: list[str]) -> list[str]:
    """Return fuels in FUEL_ORDER, appending unknowns at the end."""
    ordered = [f for f in FUEL_ORDER if f in fuels]
    ordered += [f for f in fuels if f not in ordered]
    return ordered


def _fuel_color(fuel: str) -> str:
    return FUEL_COLORS.get(fuel, "#A9A9A9")


def empty_fig(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="gray"))
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      xaxis_visible=False, yaxis_visible=False)
    return fig


# ---------------------------------------------------------------------------
# KPI card (returns a dict for dbc.Card population, not a Figure)
# ---------------------------------------------------------------------------

def kpi_value(df_summary: pd.DataFrame, key: str, decimals: int = 0) -> str:
    """Extract a formatted scalar value from pSummary DataFrame."""
    if df_summary.empty:
        return "—"
    row = df_summary[df_summary["uni"] == key]
    if row.empty:
        return "—"
    val = row["value"].iloc[0]
    try:
        return f"{float(val):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# Stacked bar — capacity or energy by fuel
# ---------------------------------------------------------------------------

def stacked_bar_fuel(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "value",
    color_col: str = "f",
    facet_col: str | None = None,
    title: str = "",
    y_label: str = "",
    scenario_col: str | None = None,
) -> go.Figure:
    """
    Stacked bar chart coloured by fuel type.

    Parameters
    ----------
    df        : DataFrame with columns [x_col, color_col, y_col, ...]
    x_col     : column to use as x-axis (usually 'y' for year)
    facet_col : optional column to split into sub-plots (e.g. 'z' for zones)
    """
    if df.empty:
        return empty_fig()

    fuels = _ordered_fuels(df[color_col].unique().tolist())
    color_map = {f: _fuel_color(f) for f in fuels}

    # Group to avoid duplicates
    group_cols = [x_col, color_col]
    if facet_col:
        group_cols.append(facet_col)
    if scenario_col and scenario_col in df.columns:
        group_cols.append(scenario_col)
    df_agg = df.groupby(group_cols, as_index=False)[y_col].sum()

    kwargs = dict(
        data_frame=df_agg,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode="stack",
        color_discrete_map=color_map,
        category_orders={color_col: fuels},
        title=title,
        labels={y_col: y_label, x_col: "Year"},
        template=PLOTLY_TEMPLATE,
    )
    if facet_col:
        kwargs["facet_col"] = facet_col
        kwargs["facet_col_wrap"] = 3
    if scenario_col and scenario_col in df_agg.columns:
        kwargs["pattern_shape"] = scenario_col

    fig = px.bar(**kwargs)
    fig.update_layout(legend_title_text="Fuel", bargap=0.15)
    fig.update_xaxes(type="category")
    return fig


# ---------------------------------------------------------------------------
# Stacked area — energy mix over time
# ---------------------------------------------------------------------------

def area_fuel(
    df: pd.DataFrame,
    x_col: str = "y",
    y_col: str = "value",
    color_col: str = "f",
    zone: str | None = None,
    title: str = "",
    y_label: str = "GWh",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    if zone:
        df = df[df["z"] == zone]
    fuels = _ordered_fuels(df[color_col].unique().tolist())
    color_map = {f: _fuel_color(f) for f in fuels}
    df_agg = df.groupby([x_col, color_col], as_index=False)[y_col].sum()
    fig = px.area(
        df_agg, x=x_col, y=y_col, color=color_col,
        color_discrete_map=color_map,
        category_orders={color_col: fuels},
        title=title,
        labels={y_col: y_label, x_col: "Year"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(type="category")
    fig.update_layout(legend_title_text="Fuel")
    return fig


# ---------------------------------------------------------------------------
# Line chart — generic multi-series
# ---------------------------------------------------------------------------

def line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str = "",
    y_label: str = "",
    markers: bool = True,
) -> go.Figure:
    if df.empty:
        return empty_fig()
    fig = px.line(
        df, x=x_col, y=y_col, color=color_col,
        markers=markers, title=title,
        labels={y_col: y_label, x_col: "Year"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(type="category")
    return fig


# ---------------------------------------------------------------------------
# Emissions line
# ---------------------------------------------------------------------------

def emissions_line(
    df: pd.DataFrame,
    zones: list[str] | None = None,
    title: str = "CO₂ Emissions by Zone",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    if zones:
        df = df[df["z"].isin(zones)]
    color_col = "scenario" if "scenario" in df.columns else "z"
    df_agg = df.groupby(["z", "y"] + (["scenario"] if "scenario" in df.columns else []),
                         as_index=False)["value"].sum()
    return line_chart(df_agg, x_col="y", y_col="value",
                      color_col=color_col, title=title, y_label="Mt CO₂")


# ---------------------------------------------------------------------------
# Stacked bar — cost breakdown
# ---------------------------------------------------------------------------

def cost_bar(
    df: pd.DataFrame,
    zone: str | None = None,
    title: str = "System Cost Breakdown",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    if zone:
        df = df[df["z"] == zone]
    df_agg = df.groupby(["uni", "y"], as_index=False)["value"].sum()
    categories = df_agg["uni"].unique().tolist()
    color_map = {c: COST_COLORS.get(c, "#A9A9A9") for c in categories}
    fig = px.bar(
        df_agg, x="y", y="value", color="uni",
        barmode="stack",
        color_discrete_map=color_map,
        title=title,
        labels={"value": "$M", "y": "Year", "uni": "Cost component"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(type="category")
    fig.update_layout(legend_title_text="Component")
    return fig


# ---------------------------------------------------------------------------
# Pie / donut — energy mix for a single year
# ---------------------------------------------------------------------------

def pie_fuel(
    df: pd.DataFrame,
    year: int,
    zone: str | None = None,
    title: str = "",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    df_y = df[df["y"] == year].copy()
    if zone:
        df_y = df_y[df_y["z"] == zone]
    df_agg = df_y.groupby("f", as_index=False)["value"].sum()
    df_agg = df_agg[df_agg["value"] > 0]
    if df_agg.empty:
        return empty_fig(f"No data for year {year}")
    colors = [_fuel_color(f) for f in df_agg["f"]]
    fig = go.Figure(go.Pie(
        labels=df_agg["f"],
        values=df_agg["value"],
        marker_colors=colors,
        hole=0.35,
        textinfo="label+percent",
    ))
    fig.update_layout(title=title or f"Energy mix {year}", template=PLOTLY_TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# Sankey — trade flows
# ---------------------------------------------------------------------------

def sankey_trade(
    df: pd.DataFrame,
    year: int,
    title: str = "Electricity Trade",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    # Expect columns: z (from), z2 (to), y, value
    needed = {"z", "z2", "y", "value"}
    if not needed.issubset(df.columns):
        return empty_fig("Trade data missing required columns (z, z2, y, value)")
    df_y = df[(df["y"] == year) & (df["value"] > 0)].copy()
    if df_y.empty:
        return empty_fig(f"No trade data for year {year}")
    zones = sorted(set(df_y["z"].tolist()) | set(df_y["z2"].tolist()))
    idx   = {z: i for i, z in enumerate(zones)}
    fig = go.Figure(go.Sankey(
        node=dict(label=zones, pad=15, thickness=20),
        link=dict(
            source=[idx[z] for z in df_y["z"]],
            target=[idx[z] for z in df_y["z2"]],
            value=df_y["value"].tolist(),
        ),
    ))
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
    return fig


# ---------------------------------------------------------------------------
# RE share line
# ---------------------------------------------------------------------------

def re_share_line(
    df_re: pd.DataFrame,
    title: str = "Renewable Energy Share (%)",
) -> go.Figure:
    """df_re expected to have columns: z, y, re_share_pct (+ optional scenario)."""
    if df_re.empty:
        return empty_fig()
    color_col = "scenario" if "scenario" in df_re.columns else "z"
    return line_chart(df_re, x_col="y", y_col="re_share_pct",
                      color_col=color_col, title=title, y_label="RE share (%)")


# ---------------------------------------------------------------------------
# Heatmap — interchange matrix
# ---------------------------------------------------------------------------

def heatmap_interchange(
    df: pd.DataFrame,
    year: int,
    scenario: str | None = None,
    title: str = "Interchange Matrix (GWh)",
) -> go.Figure:
    if df.empty:
        return empty_fig()
    needed = {"z", "z2", "y", "value"}
    if not needed.issubset(df.columns):
        return empty_fig("Interchange data missing required columns")
    df_y = df[df["y"] == year].copy()
    if scenario and "scenario" in df.columns:
        df_y = df_y[df_y["scenario"] == scenario]
    pivot = df_y.pivot_table(index="z", columns="z2", values="value",
                              aggfunc="sum", fill_value=0)
    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        title=title,
        labels={"color": "GWh"},
        template=PLOTLY_TEMPLATE,
    )
    return fig
