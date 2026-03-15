"""Results — Zonal Comparison: grouped stacked bars by zone for a selected year."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import Input, Output, callback, dcc, html

import data_loader as dl


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        # ── Filter row 1 ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Indicator", className="form-label-sm"),
                dcc.Dropdown(id="zc-indicator", options=dl.INDICATOR_OPTIONS,
                             value="CapacityTechFuel", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Year", className="form-label-sm"),
                dcc.Dropdown(id="zc-year", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=1),
            dbc.Col([
                html.Label("Aggregation Level", className="form-label-sm"),
                dcc.RadioItems(
                    id="zc-spatial",
                    options=[{"label": " Country", "value": "c"},
                             {"label": " Zone",    "value": "z"}],
                    value="z", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Label("Scenarios", className="form-label-sm"),
                dcc.Dropdown(id="zc-scenarios", multi=True,
                             placeholder="All scenarios",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Reference Scenario", className="form-label-sm"),
                dcc.Dropdown(id="zc-ref-scenario", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("View", className="form-label-sm"),
                dcc.RadioItems(
                    id="zc-view",
                    options=[{"label": " Absolute",   "value": "Absolute"},
                             {"label": " Difference", "value": "Difference"},
                             {"label": " %",          "value": "Percentage"}],
                    value="Absolute", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=3),
        ], className="g-2")), className="mb-2 shadow-sm filter-card"),

        # ── Filter row 2 ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Zone / Country filter", className="form-label-sm"),
                dcc.Dropdown(id="zc-zones", multi=True, placeholder="All zones",
                             style={"fontSize": "0.85rem"}),
            ], md=4),
            dbc.Col([
                html.Label("Legend filter", className="form-label-sm"),
                dcc.Dropdown(id="zc-legend-filter", multi=True,
                             placeholder="All categories",
                             style={"fontSize": "0.85rem"}),
            ], md=4),
        ], className="g-2")), className="mb-3 shadow-sm filter-card"),

        # ── Chart ─────────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dcc.Graph(
            id="zc-chart",
            config={"displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"height": "600px"},
        )), className="shadow-sm border-0"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("zc-year",          "options"),
    Output("zc-year",          "value"),
    Output("zc-scenarios",     "options"),
    Output("zc-scenarios",     "value"),
    Output("zc-ref-scenario",  "options"),
    Output("zc-ref-scenario",  "value"),
    Output("zc-zones",         "options"),
    Output("zc-legend-filter", "options"),
    Input("filter-run",        "value"),
    Input("zc-spatial",        "value"),
    Input("zc-indicator",      "value"),
)
def init_zc_filters(run, spatial, indicator):
    if not run:
        return [], None, [], [], [], None, [], []
    years     = dl.get_merged_years(run)
    scenarios = dl.list_scenarios(run)
    units     = (dl.get_merged_zones(run) if spatial == "z"
                 else dl.get_merged_countries(run))
    y_opts = [{"label": str(y), "value": y} for y in years]
    s_opts = [{"label": s, "value": s} for s in scenarios]
    u_opts = [{"label": u, "value": u} for u in units]

    src, legend_col = dl.INDICATOR_SOURCE.get(indicator, ("techfuel", "techfuel"))
    df = (dl.load_techfuel(run) if src == "techfuel"
          else dl.load_costs_merged(run) if src == "costs"
          else dl.load_capex_merged(run))
    leg_opts = []
    if not df.empty and legend_col in df.columns:
        cats = sorted(df[legend_col].dropna().unique().tolist())
        leg_opts = [{"label": c, "value": c} for c in cats]

    default_ref = next((s for s in ["baseline", "Baseline"] if s in scenarios),
                       scenarios[0] if scenarios else None)
    return (y_opts, years[-1] if years else None,
            s_opts, scenarios,
            s_opts, default_ref,
            u_opts, leg_opts)


@callback(
    Output("zc-chart", "figure"),
    Input("zc-indicator",     "value"),
    Input("zc-year",          "value"),
    Input("zc-spatial",       "value"),
    Input("zc-scenarios",     "value"),
    Input("zc-ref-scenario",  "value"),
    Input("zc-view",          "value"),
    Input("zc-zones",         "value"),
    Input("zc-legend-filter", "value"),
    Input("filter-run",       "value"),
)
def update_zc_chart(indicator, year, spatial, scenarios, ref_scenario,
                    view, zones, legend_filter, run):
    empty = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    if not run or not indicator or not year:
        return empty

    src, legend_col = dl.INDICATOR_SOURCE[indicator]
    df = (dl.load_techfuel(run) if src == "techfuel"
          else dl.load_costs_merged(run) if src == "costs"
          else dl.load_capex_merged(run))

    if df.empty:
        return empty

    df = df[(df["attribute"] == indicator) & (df["y"] == year)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if scenarios:
        df = df[df["scenario"].isin(scenarios)]
    if zones and spatial in df.columns:
        df = df[df[spatial].isin(zones)]
    if legend_filter and legend_col in df.columns:
        df = df[df[legend_col].isin(legend_filter)]

    if spatial not in df.columns:
        return empty

    group_cols = ["scenario", spatial, legend_col]
    df_agg = df.groupby(group_cols, as_index=False)["value"].sum()
    if df_agg.empty:
        return empty

    # View mode
    if view == "Difference":
        df_ref = (df_agg[df_agg["scenario"] == ref_scenario]
                  [[spatial, legend_col, "value"]].copy()
                  .rename(columns={"value": "ref_value"}))
        df_agg = df_agg.merge(df_ref, on=[spatial, legend_col], how="left")
        df_agg["ref_value"] = df_agg["ref_value"].fillna(0)
        df_agg["value"] = df_agg["value"] - df_agg["ref_value"]
        df_agg = df_agg.drop(columns=["ref_value"])
    elif view == "Percentage":
        totals = df_agg.groupby(["scenario", spatial])["value"].transform("sum")
        df_agg["value"] = np.where(totals != 0, df_agg["value"] / totals * 100, 0)

    df_agg = df_agg[df_agg["value"].abs() > 1e-6]
    if df_agg.empty:
        return empty

    all_s = sorted(df_agg["scenario"].unique().tolist())
    sc_list = (["baseline"] if "baseline" in all_s else []) + \
              [s for s in all_s if s != "baseline"]
    if view == "Difference":
        df_agg = df_agg[df_agg["scenario"] != ref_scenario]
        sc_list = [s for s in sc_list if s != ref_scenario]

    zone_totals = df_agg.groupby(spatial)["value"].apply(lambda x: x.abs().sum())
    zones_list = zone_totals.sort_values(ascending=False).index.tolist()

    df_agg["x_label"] = df_agg[spatial] + " | " + df_agg["scenario"]
    x_order = [f"{z} | {s}" for z in zones_list for s in sc_list]
    x_order_filtered = [x for x in x_order if x in set(df_agg["x_label"].values)]

    if not x_order_filtered:
        return empty

    cats = df_agg[legend_col].unique().tolist()
    ordered_cats = [t for t in dl.TECH_ORDER if t in cats] + \
                   [t for t in cats if t not in dl.TECH_ORDER]

    fig = go.Figure()
    for cat in ordered_cats:
        sub = (df_agg[df_agg[legend_col] == cat]
               .set_index("x_label")["value"]
               .reindex(x_order_filtered))
        fig.add_trace(go.Bar(
            x=x_order_filtered, y=sub.values.tolist(),
            name=cat, marker_color=dl.TECH_COLORS.get(cat, "#aaaaaa"),
            hovertemplate=f"<b>{cat}</b><br>%{{x}}: %{{y:,.1f}}<extra></extra>",
        ))

    # Zone group separators + labels
    pos_map = {x: i for i, x in enumerate(x_order_filtered)}
    for i_z, z in enumerate(zones_list):
        z_pos = [pos_map[x] for x in x_order_filtered
                 if x.split(" | ")[0] == z]
        if not z_pos:
            continue
        if i_z > 0:
            fig.add_shape(type="line",
                          x0=z_pos[0] - 0.5, x1=z_pos[0] - 0.5,
                          y0=0, y1=1.0, xref="x", yref="paper",
                          line=dict(color="#aaaaaa", width=1.5, dash="dot"))
        fig.add_annotation(
            x=np.mean(z_pos), y=-0.18, xref="x", yref="paper",
            text=f"<b>{z}</b>", showarrow=False,
            font=dict(size=10, color="#1B2A4A"),
            xanchor="center", yanchor="top",
        )

    tick_labels = [x.split(" | ")[1] for x in x_order_filtered]
    fig.update_xaxes(ticktext=tick_labels, tickvals=x_order_filtered,
                     tickangle=-30, tickfont=dict(size=9))

    y_label = dl.INDICATOR_LABELS.get(indicator, indicator)
    if view == "Difference":
        y_label = f"Δ {y_label} vs {ref_scenario}"
    elif view == "Percentage":
        y_label = "Share (%)"

    fig.update_layout(
        barmode="relative" if view == "Difference" else "stack",
        title=dict(text=f"{dl.INDICATOR_LABELS.get(indicator, indicator)} — {int(year)}",
                   font=dict(size=13), x=0.01),
        margin=dict(l=60, r=20, t=45, b=120),
        legend=dict(orientation="v", x=1.01, y=1,
                    font=dict(size=10), title=dict(text="")),
        yaxis_title=y_label,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
        height=max(450, len(x_order_filtered) * 18 + 160),
        bargap=0.15,
    )
    return fig
