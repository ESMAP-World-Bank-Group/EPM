"""Results — Evolution: stacked bars by year × scenario, multiple indicators."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import Input, Output, callback, dcc, html, no_update

import data_loader as dl


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        # ── Filter row 1 ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("View", className="form-label-sm"),
                dcc.RadioItems(
                    id="evo-view",
                    options=[{"label": " Absolute",   "value": "Absolute"},
                             {"label": " Difference", "value": "Difference"},
                             {"label": " % Share",    "value": "Percentage"}],
                    value="Absolute", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=3),
            dbc.Col([
                html.Label("Indicator", className="form-label-sm"),
                dcc.Dropdown(id="evo-indicator", options=dl.INDICATOR_OPTIONS,
                             value="CapacityTechFuel", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Scenarios", className="form-label-sm"),
                dcc.Dropdown(id="evo-scenarios", multi=True,
                             placeholder="All scenarios",
                             style={"fontSize": "0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Reference Scenario", className="form-label-sm"),
                dcc.Dropdown(id="evo-ref-scenario", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=3),
        ], className="g-2")), className="mb-2 shadow-sm filter-card"),

        # ── Filter row 2 ──────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Aggregation Level", className="form-label-sm"),
                dcc.RadioItems(
                    id="evo-spatial",
                    options=[{"label": " Country", "value": "c"},
                             {"label": " Zone",    "value": "z"}],
                    value="z", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Label("Zone / Country", className="form-label-sm"),
                dcc.Dropdown(id="evo-zones", multi=True,
                             placeholder="All (aggregated)",
                             style={"fontSize": "0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Years", className="form-label-sm"),
                dcc.Dropdown(id="evo-years", multi=True,
                             placeholder="All years",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Legend filter", className="form-label-sm"),
                dcc.Dropdown(id="evo-legend-filter", multi=True,
                             placeholder="All categories",
                             style={"fontSize": "0.85rem"}),
            ], md=3),
        ], className="g-2")), className="mb-2 shadow-sm filter-card"),

        # ── Filter row 3 (line overlay) ───────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Line Overlay", className="form-label-sm"),
                dcc.Dropdown(id="evo-line-indicator",
                             options=dl.LINE_INDICATOR_OPTIONS,
                             value="", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=4),
        ], className="g-2")), className="mb-3 shadow-sm filter-card"),

        # ── Chart ─────────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dcc.Graph(
            id="evo-chart",
            config={"displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"height": "580px"},
        )), className="shadow-sm border-0"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("evo-ref-scenario",  "options"),
    Output("evo-ref-scenario",  "value"),
    Output("evo-scenarios",     "options"),
    Output("evo-scenarios",     "value"),
    Output("evo-years",         "options"),
    Output("evo-zones",         "options"),
    Output("evo-legend-filter", "options"),
    Input("filter-run",         "value"),
    Input("evo-spatial",        "value"),
    Input("evo-indicator",      "value"),
)
def init_evo_dropdowns(run, spatial, indicator):
    if not run:
        return [], None, [], [], [], [], []
    scenarios = dl.list_scenarios(run)
    years     = dl.get_merged_years(run)
    units     = (dl.get_merged_zones(run) if spatial == "z"
                 else dl.get_merged_countries(run))
    s_opts = [{"label": s, "value": s} for s in scenarios]
    y_opts = [{"label": str(y), "value": y} for y in years]
    u_opts = [{"label": u, "value": u} for u in units]

    src, legend_col = dl.INDICATOR_SOURCE.get(indicator, ("techfuel", "techfuel"))
    if src == "techfuel":
        df = dl.load_techfuel(run)
    elif src == "costs":
        df = dl.load_costs_merged(run)
    else:
        df = dl.load_capex_merged(run)

    leg_opts = []
    if not df.empty and legend_col in df.columns:
        cats = sorted(df[legend_col].dropna().unique().tolist())
        leg_opts = [{"label": c, "value": c} for c in cats]

    default_ref = next((s for s in ["baseline", "Baseline"] if s in scenarios),
                       scenarios[0] if scenarios else None)
    return (s_opts, default_ref,
            s_opts, scenarios,
            y_opts, u_opts, leg_opts)


@callback(
    Output("evo-chart", "figure"),
    Input("evo-indicator",      "value"),
    Input("evo-line-indicator", "value"),
    Input("evo-scenarios",      "value"),
    Input("evo-ref-scenario",   "value"),
    Input("evo-view",           "value"),
    Input("evo-spatial",        "value"),
    Input("evo-zones",          "value"),
    Input("evo-years",          "value"),
    Input("evo-legend-filter",  "value"),
    Input("filter-run",         "value"),
)
def update_evo_chart(indicator, line_ind, scenarios, ref_scenario,
                     view, spatial, zones, years_filter, legend_filter, run):
    empty = go.Figure()
    empty.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        annotations=[dict(text="Select a run and indicator to display.",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color="#aaa"))],
    )
    if not run or not indicator:
        return empty

    src, legend_col = dl.INDICATOR_SOURCE[indicator]
    if src == "techfuel":
        df = dl.load_techfuel(run)
    elif src == "costs":
        df = dl.load_costs_merged(run)
    else:
        df = dl.load_capex_merged(run)

    if df.empty:
        if df.attrs.get("no_merged_data"):
            empty.update_layout(annotations=[dict(
                text="No merged output files found. Please re-run the model.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="#e07b39"),
            )])
        return empty

    df = df[df["attribute"] == indicator].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if scenarios:
        df = df[df["scenario"].isin(scenarios)]
    if years_filter:
        df = df[df["y"].isin(years_filter)]
    if zones and spatial in df.columns:
        df = df[df[spatial].isin(zones)]
    if legend_filter and legend_col in df.columns:
        df = df[df[legend_col].isin(legend_filter)]

    group_cols = ["scenario", "y", legend_col]
    df_agg = df.groupby(group_cols, as_index=False)["value"].sum()
    if df_agg.empty:
        return empty

    # View mode
    if view == "Difference":
        df_ref = (df_agg[df_agg["scenario"] == ref_scenario]
                  [["y", legend_col, "value"]].copy()
                  .rename(columns={"value": "ref_value"}))
        df_agg = df_agg.merge(df_ref, on=["y", legend_col], how="left")
        df_agg["ref_value"] = df_agg["ref_value"].fillna(0)
        df_agg["value"] = df_agg["value"] - df_agg["ref_value"]
        df_agg = df_agg.drop(columns=["ref_value"])
        df_agg = df_agg[df_agg["scenario"] != ref_scenario]
    elif view == "Percentage":
        totals = df_agg.groupby(["scenario", "y"])["value"].transform("sum")
        df_agg["value"] = np.where(totals != 0, df_agg["value"] / totals * 100, 0)

    df_agg = df_agg[df_agg["value"].abs() > 1e-6]
    if df_agg.empty:
        return empty

    years_list = sorted(df_agg["y"].unique().tolist())
    all_s = sorted(df_agg["scenario"].unique().tolist())
    sc_list = (["baseline"] if "baseline" in all_s else []) + \
              [s for s in all_s if s != "baseline"]

    df_agg["x_label"] = df_agg["y"].astype(int).astype(str) + " | " + df_agg["scenario"]
    x_order = [f"{int(y)} | {s}" for y in years_list for s in sc_list]
    x_order_filtered = [x for x in x_order if x in set(df_agg["x_label"].values)]

    cats = df_agg[legend_col].unique().tolist()
    ordered_cats = [t for t in dl.TECH_ORDER if t in cats] + \
                   [t for t in cats if t not in dl.TECH_ORDER]

    bar_mode = "relative" if view == "Difference" else "stack"
    fig = go.Figure()
    for cat in ordered_cats:
        cat_df = df_agg[df_agg[legend_col] == cat].copy()
        if cat_df.empty:
            continue
        color = dl.TECH_COLORS.get(cat, "#aaaaaa")
        fig.add_trace(go.Bar(
            x=cat_df["x_label"], y=cat_df["value"],
            name=cat, marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br><b>{cat}</b>: %{{y:,.1f}}<extra></extra>",
            legendgroup=cat,
        ))
    fig.update_layout(barmode=bar_mode)
    fig.update_xaxes(categoryorder="array", categoryarray=x_order_filtered)

    # Year group separators + labels
    pos_map = {x: i for i, x in enumerate(x_order_filtered)}
    for i_y, y in enumerate(years_list):
        y_pos = [pos_map[x] for x in x_order_filtered if x.startswith(f"{int(y)} | ")]
        if not y_pos:
            continue
        if i_y > 0:
            fig.add_shape(type="line",
                          x0=y_pos[0] - 0.5, x1=y_pos[0] - 0.5, y0=0, y1=1.0,
                          xref="x", yref="paper",
                          line=dict(color="#aaaaaa", width=1.5, dash="dot"))
        fig.add_annotation(x=np.mean(y_pos), y=1.04, xref="x", yref="paper",
                           text=f"<b>{int(y)}</b>", showarrow=False,
                           font=dict(size=12, color="#1B2A4A"), xanchor="center")

    tick_labels = [x.split(" | ")[1] for x in x_order_filtered]
    fig.update_xaxes(ticktext=tick_labels, tickvals=x_order_filtered,
                     tickangle=-30, tickfont=dict(size=10))

    y_label = dl.INDICATOR_LABELS.get(indicator, indicator)
    if view == "Difference":
        y_label = f"Δ {y_label} vs {ref_scenario}"
    elif view == "Percentage":
        y_label = "Share (%)"

    # Total dot
    bar_totals = (df_agg.groupby("x_label")["value"].sum()
                  .reindex(x_order_filtered).dropna())
    if not bar_totals.empty:
        fig.add_trace(go.Scatter(
            x=bar_totals.index.tolist(), y=bar_totals.values.tolist(),
            mode="markers",
            marker=dict(color="#1B2A4A", size=9, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            name="Total",
            hovertemplate="<b>Total</b> %{x}: %{y:,.0f}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        margin=dict(l=10, r=20, t=55, b=60),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
        yaxis_title=y_label,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11), hovermode="closest", bargap=0.2,
    )

    # Line overlay
    if line_ind:
        yz = dl.load_yearly_zone(run)
        if not yz.empty:
            yz_sub = yz[yz["attribute"] == line_ind].copy()
            if zones and spatial in yz_sub.columns:
                yz_sub = yz_sub[yz_sub[spatial].isin(zones)]
            if scenarios:
                yz_sub = yz_sub[yz_sub["scenario"].isin(scenarios)]
            if years_filter:
                yz_sub = yz_sub[yz_sub["y"].isin(years_filter)]
            yz_line = yz_sub.groupby(["scenario", "y"], as_index=False)["value"].sum()
            for scen in sc_list:
                s_line = yz_line[yz_line["scenario"] == scen].sort_values("y")
                s_line = s_line[s_line["y"].isin(years_list)].copy()
                s_line["x_label"] = s_line["y"].astype(int).astype(str) + " | " + scen
                s_line = s_line[s_line["x_label"].isin(x_order_filtered)]
                if not s_line.empty:
                    fig.add_trace(go.Scatter(
                        x=s_line["x_label"], y=s_line["value"],
                        name=f"{dl.INDICATOR_LABELS.get(line_ind, line_ind)} ({scen})",
                        yaxis="y2", mode="lines+markers",
                        line=dict(width=2, dash="dash"), marker=dict(size=6),
                    ))
            fig.update_layout(yaxis2=dict(
                title=dl.INDICATOR_LABELS.get(line_ind, line_ind),
                overlaying="y", side="right", showgrid=False,
            ))

    return fig
