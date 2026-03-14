"""Input Editor — Demand (forecast wide-format, demand profile, efficiency factor)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import data_loader as dl


def _grid(grid_id: str, height: str = "380px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        style={"height": height},
        className="ag-theme-alpine",
    )


def _col_defs_wide(df: pd.DataFrame, id_cols: list) -> list:
    """Column defs for wide-format tables (id_cols = read-only, rest = editable)."""
    defs = []
    for c in df.columns:
        editable = c not in id_cols
        defs.append({
            "field": str(c),
            "editable": editable,
            "type": "numericColumn" if editable else None,
            "cellStyle": {} if editable
                         else {"backgroundColor": "#f8f9fa", "color": "#6c757d"},
        })
    return defs


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Demand Inputs", className="mb-1"),
        html.P("Edit demand forecast (GWh/MW by zone and year), load profiles and "
               "energy efficiency factors.", className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="demand-project", options=options,
                             value=default, clearable=False),
            ], width=3),
        ], className="mb-3"),

        dbc.Tabs([

            # ── Demand Forecast ───────────────────────────────────────────
            dbc.Tab(label="Demand Forecast", tab_id="tab-forecast", children=[
                dbc.Row(className="mt-3", children=[
                    # Left: editable grid
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-demand-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-demand-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Wide format: rows = zones, columns = years. "
                               "Edit cells directly.",
                               className="text-muted small mb-1"),
                        _grid("demand-grid"),
                    ], width=6),

                    # Right: chart with zone + type filter
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="demand-chart-zone",
                                             options=[], value=None,
                                             placeholder="All zones",
                                             multi=False, className="small"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Type", className="small"),
                                dcc.Dropdown(id="demand-chart-type",
                                             options=[], value=None,
                                             placeholder="All types",
                                             className="small"),
                            ], width=6),
                        ], className="mb-2"),
                        dcc.Graph(id="demand-preview-chart",
                                  config={"displayModeBar": False}),
                    ], width=6),
                ]),
            ]),

            # ── Demand Profile ────────────────────────────────────────────
            dbc.Tab(label="Demand Profile (read-only)", tab_id="tab-profile", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        html.P("Hourly load profile shape. Read-only — upload a new CSV "
                               "to replace.", className="text-muted small mb-2"),
                        _grid("profile-grid", height="460px"),
                    ], width=12),
                ]),
            ]),

            # ── Energy Efficiency Factor ──────────────────────────────────
            dbc.Tab(label="Efficiency Factor", tab_id="tab-eff", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-eff-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-eff-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Energy efficiency reduction factor per zone per year (fraction).",
                               className="text-muted small mb-1"),
                        _grid("eff-grid"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Efficiency reduction over time", className="text-muted"),
                        dcc.Graph(id="eff-chart", config={"displayModeBar": False}),
                    ], width=6),
                ]),
            ]),
        ]),
    ])


# ── Load all grids when project changes ──────────────────────────────────────

@callback(
    Output("demand-grid",       "rowData"),
    Output("demand-grid",       "columnDefs"),
    Output("profile-grid",      "rowData"),
    Output("profile-grid",      "columnDefs"),
    Output("eff-grid",          "rowData"),
    Output("eff-grid",          "columnDefs"),
    Output("demand-chart-zone", "options"),
    Output("demand-chart-zone", "value"),
    Output("demand-chart-type", "options"),
    Output("demand-chart-type", "value"),
    Output("eff-chart",         "figure"),
    Input("demand-project",     "value"),
)
def load(folder):
    empty = ([], [])
    empty_fig = px.line(title="No data", template="plotly_white")

    if not folder:
        return *empty, *empty, *empty, [], None, [], None, empty_fig

    df_dem     = dl.load_input(folder, "demand_forecast")
    df_profile = dl.load_input(folder, "demand_profile")
    df_eff     = dl.load_input(folder, "efficiency")

    # ── Demand forecast (wide format: z, type, 2025, 2030, ...) ──────────
    if not df_dem.empty:
        id_cols_dem = [c for c in ["z", "type"] if c in df_dem.columns]
        dem_rows    = df_dem.to_dict("records")
        dem_cols    = _col_defs_wide(df_dem, id_cols_dem)
        zone_opts   = [{"label": z, "value": z}
                       for z in sorted(df_dem["z"].unique())] if "z" in df_dem.columns else []
        type_opts   = [{"label": t, "value": t}
                       for t in sorted(df_dem["type"].unique())] if "type" in df_dem.columns else []
        default_zone = zone_opts[0]["value"] if zone_opts else None
        default_type = type_opts[0]["value"] if type_opts else None
    else:
        dem_rows = []; dem_cols = []
        zone_opts = []; type_opts = []
        default_zone = None; default_type = None

    # ── Demand profile (read-only) ────────────────────────────────────────
    if not df_profile.empty:
        prof_cols = [{"field": str(c), "editable": False,
                      "cellStyle": {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
                     for c in df_profile.columns]
        prof_rows = df_profile.head(500).to_dict("records")  # cap at 500 rows for perf
    else:
        prof_rows = []; prof_cols = []

    # ── Efficiency factor ─────────────────────────────────────────────────
    if not df_eff.empty:
        id_cols_eff = [c for c in ["z", "c"] if c in df_eff.columns]
        eff_rows = df_eff.to_dict("records")
        eff_cols = _col_defs_wide(df_eff, id_cols_eff)

        # Efficiency chart — melt if wide
        year_cols = [c for c in df_eff.columns
                     if c not in id_cols_eff and str(c).isdigit()]
        if year_cols and id_cols_eff:
            df_melt = df_eff.melt(id_vars=id_cols_eff, value_vars=year_cols,
                                  var_name="year", value_name="factor")
            fig_eff = px.line(df_melt, x="year", y="factor",
                              color=id_cols_eff[0],
                              markers=True,
                              title="Energy Efficiency Factor",
                              labels={"factor": "Reduction factor", "year": "Year"},
                              template="plotly_white")
        elif "value" in df_eff.columns:
            fig_eff = px.line(df_eff, x="y" if "y" in df_eff.columns else df_eff.columns[1],
                              y="value",
                              color="z" if "z" in df_eff.columns else None,
                              markers=True, title="Energy Efficiency Factor",
                              template="plotly_white")
        else:
            fig_eff = empty_fig
    else:
        eff_rows = []; eff_cols = []; fig_eff = empty_fig

    return (
        dem_rows, dem_cols,
        prof_rows, prof_cols,
        eff_rows, eff_cols,
        zone_opts, default_zone,
        type_opts, default_type,
        fig_eff,
    )


# ── Update demand preview chart when zone / type filter changes ───────────────

@callback(
    Output("demand-preview-chart", "figure"),
    Input("demand-chart-zone",     "value"),
    Input("demand-chart-type",     "value"),
    Input("demand-grid",           "rowData"),
    prevent_initial_call=False,
)
def update_demand_chart(zone, dtype, rows):
    empty_fig = px.line(title="Select a zone to preview",
                        template="plotly_white")
    if not rows:
        return empty_fig

    df = pd.DataFrame(rows)
    id_cols = [c for c in ["z", "type"] if c in df.columns]
    year_cols = [c for c in df.columns if c not in id_cols]

    # Filter
    if zone and "z" in df.columns:
        df = df[df["z"] == zone]
    if dtype and "type" in df.columns:
        df = df[df["type"] == dtype]

    if df.empty:
        return empty_fig

    # Melt wide → long for plotting
    color_col = [c for c in id_cols if c != "z" and c != "type"]
    melt_id = id_cols
    df_melt = df.melt(id_vars=melt_id, value_vars=year_cols,
                      var_name="year", value_name="value")
    df_melt["value"] = pd.to_numeric(df_melt["value"], errors="coerce")
    df_melt = df_melt.dropna(subset=["value"])

    color = "type" if "type" in df_melt.columns and dtype is None else \
            ("z" if "z" in df_melt.columns and zone is None else None)

    title = f"Demand Forecast{' — ' + zone if zone else ''}{' (' + dtype + ')' if dtype else ''}"

    fig = px.line(df_melt, x="year", y="value", color=color,
                  markers=True, title=title,
                  labels={"value": "GWh / MW", "year": "Year"},
                  template="plotly_white")
    fig.update_xaxes(type="category")
    return fig


# ── Save callbacks ────────────────────────────────────────────────────────────

@callback(
    Output("save-demand-msg", "children"),
    Input("save-demand-btn",  "n_clicks"),
    State("demand-grid",      "rowData"),
    State("demand-project",   "value"),
    prevent_initial_call=True,
)
def save_demand(n, rows, folder):
    if not rows or not folder:
        return dbc.Badge("Nothing to save", color="warning")
    ok = dl.save_input(folder, "demand_forecast", pd.DataFrame(rows))
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")


@callback(
    Output("save-eff-msg", "children"),
    Input("save-eff-btn",  "n_clicks"),
    State("eff-grid",      "rowData"),
    State("demand-project","value"),
    prevent_initial_call=True,
)
def save_eff(n, rows, folder):
    if not rows or not folder:
        return dbc.Badge("Nothing to save", color="warning")
    ok = dl.save_input(folder, "efficiency", pd.DataFrame(rows))
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")
