"""Results — Power Plants: plant ranking bar + LCOE vs utilization scatter."""

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

import data_loader as dl


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Scenario", className="form-label-sm"),
                dcc.Dropdown(id="pp-scenario", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Year", className="form-label-sm"),
                dcc.Dropdown(id="pp-year", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Spatial Resolution", className="form-label-sm"),
                dcc.RadioItems(
                    id="pp-spatial",
                    options=[{"label": " Country", "value": "c"},
                             {"label": " Zone",    "value": "z"}],
                    value="z", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Label("Zone / Country", className="form-label-sm"),
                dcc.Dropdown(id="pp-zone", multi=True, placeholder="All zones",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Plant Indicator", className="form-label-sm"),
                dcc.Dropdown(id="pp-indicator",
                             options=dl.PLANT_INDICATOR_OPTIONS,
                             value="CapacityPlant", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Top N plants", className="form-label-sm"),
                dcc.Slider(id="pp-topn", min=10, max=50, step=5, value=25,
                           marks={10: "10", 25: "25", 50: "50"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], md=2),
        ], className="g-2")), className="mb-3 shadow-sm filter-card"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("Plant Ranking")),
                dbc.CardBody(dcc.Graph(id="pp-bar",
                                       config={"displayModeBar": False},
                                       style={"height": "540px"})),
            ], className="shadow-sm border-0"), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("LCOE vs Utilization Factor")),
                dbc.CardBody(dcc.Graph(id="pp-scatter",
                                       config={"displayModeBar": True,
                                               "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                                       style={"height": "540px"})),
            ], className="shadow-sm border-0"), md=6),
        ], className="g-3"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("pp-scenario", "options"),
    Output("pp-scenario", "value"),
    Output("pp-year",     "options"),
    Output("pp-year",     "value"),
    Input("filter-run",   "value"),
)
def init_pp_filters(run):
    if not run:
        return [], None, [], None
    scenarios = dl.list_scenarios(run)
    years     = dl.get_merged_years(run)
    s_opts = [{"label": s, "value": s} for s in scenarios]
    y_opts = [{"label": str(y), "value": y} for y in years]
    default_s = next((s for s in ["baseline", "Baseline"] if s in scenarios),
                     scenarios[0] if scenarios else None)
    return s_opts, default_s, y_opts, (years[-1] if years else None)


@callback(
    Output("pp-zone",   "options"),
    Output("pp-zone",   "value"),
    Input("pp-spatial", "value"),
    Input("filter-run", "value"),
    State("pp-zone",    "value"),
)
def update_pp_zones(spatial, run, current_value):
    if not run:
        return [], None
    units = (dl.get_merged_zones(run) if spatial == "z"
             else dl.get_merged_countries(run))
    opts = [{"label": u, "value": u} for u in units]
    if ctx.triggered_id == "pp-spatial":
        return opts, None
    if not current_value:
        return opts, no_update
    valid = set(units)
    kept = ([v for v in current_value if v in valid]
            if isinstance(current_value, list)
            else (current_value if current_value in valid else None))
    return opts, kept


@callback(
    Output("pp-bar",     "figure"),
    Output("pp-scatter", "figure"),
    Input("pp-scenario", "value"),
    Input("pp-year",     "value"),
    Input("pp-spatial",  "value"),
    Input("pp-zone",     "value"),
    Input("pp-indicator","value"),
    Input("pp-topn",     "value"),
    Input("filter-run",  "value"),
)
def update_pp_charts(scenario, year, spatial, zone, indicator, topn, run):
    empty = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    if not run or not scenario or not year:
        return empty, empty

    df = dl.load_plants_merged(run)
    if df.empty:
        msg = "No plant data found (pPlantMerged.csv). Please re-run the model."
        empty.add_annotation(text=msg, xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=12, color="#e07b39"))
        return empty, empty

    spatial_col = "z" if spatial == "z" else "c"
    if spatial_col not in df.columns:
        spatial_col = "z"
    df_sel = df[(df["scenario"] == scenario) & (df["y"] == year)].copy()
    if zone:
        zones_list = zone if isinstance(zone, list) else [zone]
        if spatial_col in df_sel.columns:
            df_sel = df_sel[df_sel[spatial_col].isin(zones_list)]
    df_sel["value"] = pd.to_numeric(df_sel["value"], errors="coerce")
    df_sel = df_sel.dropna(subset=["value"])

    # Plant ranking bar
    ind_df = df_sel[df_sel["attribute"] == indicator].copy()
    ind_df = ind_df[ind_df["value"] > 0]
    if ind_df.empty:
        bar_fig = empty
    else:
        ind_df = ind_df.nlargest(topn, "value").sort_values("value", ascending=True)
        g_col  = "g" if "g" in ind_df.columns else "uni"
        tf_col = "techfuel" if "techfuel" in ind_df.columns else None
        label  = next((o["label"] for o in dl.PLANT_INDICATOR_OPTIONS
                       if o["value"] == indicator), indicator)
        bar_kwargs = dict(x="value", y=g_col, orientation="h",
                          labels={"value": label, g_col: "Plant"},
                          template="plotly_white")
        if tf_col:
            bar_kwargs["color"] = tf_col
            bar_kwargs["color_discrete_map"] = dl.TECH_COLORS
            bar_kwargs["labels"]["techfuel"] = "Technology"
        bar_fig = px.bar(ind_df, **bar_kwargs)
        bar_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=20),
            legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9)),
            xaxis_title=label, yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white", font=dict(size=10),
        )
        bar_fig.update_yaxes(tickfont=dict(size=9))

    # LCOE vs utilization scatter
    lcoe_attr = next((a for a in ["PlantAnnualLCOE", "LCOEPlant", "pPlantAnnualLCOE"]
                      if a in df_sel["attribute"].values), None)
    util_attr = next((a for a in ["UtilizationPlant", "UtilizationFactor"]
                      if a in df_sel["attribute"].values), None)
    capa_attr = "CapacityPlant"
    g_col     = "g" if "g" in df_sel.columns else "uni"
    tf_col    = "techfuel" if "techfuel" in df_sel.columns else None
    merge_keys = [k for k in [g_col, tf_col] if k and k in df_sel.columns]

    scatter_fig = empty
    if lcoe_attr and util_attr:
        lcoe_df = df_sel[df_sel["attribute"] == lcoe_attr].rename(columns={"value": "lcoe"})
        util_df = df_sel[df_sel["attribute"] == util_attr].rename(columns={"value": "util"})
        capa_df = df_sel[df_sel["attribute"] == capa_attr].rename(columns={"value": "capa"})
        if not lcoe_df.empty and not util_df.empty:
            scatter = lcoe_df[merge_keys + ["lcoe"]].merge(
                util_df[merge_keys + ["util"]], on=merge_keys, how="inner")
            if not capa_df.empty:
                scatter = scatter.merge(capa_df[merge_keys + ["capa"]],
                                        on=merge_keys, how="left")
            else:
                scatter["capa"] = 10
            scatter = scatter.dropna(subset=["lcoe", "util"])
            if not scatter.empty:
                sc_kwargs = dict(
                    x="util", y="lcoe", size="capa", size_max=40,
                    hover_name=g_col,
                    labels={"util": "Utilization Factor",
                            "lcoe": "LCOE (USD/MWh)"},
                    template="plotly_white",
                )
                if tf_col and tf_col in scatter.columns:
                    sc_kwargs["color"] = tf_col
                    sc_kwargs["color_discrete_map"] = dl.TECH_COLORS
                    sc_kwargs["labels"]["techfuel"] = "Technology"
                scatter_fig = px.scatter(scatter, **sc_kwargs)
                scatter_fig.update_layout(
                    margin=dict(l=10, r=20, t=10, b=20),
                    legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
                    xaxis_title="Utilization Factor", yaxis_title="LCOE (USD/MWh)",
                    plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
                )

    return bar_fig, scatter_fig
