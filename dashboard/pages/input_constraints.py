"""Input Editor — Constraints (carbon price, emissions caps only)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import data_loader as dl


def _grid(grid_id: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        style={"height": "360px"},
        className="ag-theme-alpine",
    )


def _col_defs(df: pd.DataFrame) -> list:
    return [
        {"field": c,
         "editable": df[c].dtype in ("float64", "int64") or c == "value",
         "cellStyle": {} if (df[c].dtype in ("float64", "int64") or c == "value")
                     else {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
        for c in df.columns
    ]


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Policy Constraints", className="mb-1"),
        html.P("Edit carbon price trajectories and CO₂ emissions caps.",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="const-project", options=options,
                             value=default, clearable=False),
            ], width=3),
        ], className="mb-3"),

        dbc.Tabs([

            # ── Carbon Price ──────────────────────────────────────────────
            dbc.Tab(label="Carbon Price", tab_id="tab-cp", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-cp-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-cp-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Carbon price trajectory ($/tCO₂). "
                               "Only used when carbon price is enabled in Settings.",
                               className="text-muted small"),
                        _grid("cp-grid"),
                    ], width=5),
                    dbc.Col([
                        html.H6("Carbon Price trajectory", className="text-muted"),
                        dcc.Graph(id="cp-chart", config={"displayModeBar": False}),
                    ], width=7),
                ]),
            ]),

            # ── Emissions Caps ────────────────────────────────────────────
            dbc.Tab(label="Emissions Caps", tab_id="tab-em", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save System Cap", id="save-em-sys-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-em-sys-msg"), width="auto"),
                        ], className="mb-2"),
                        html.H6("System-wide CO₂ cap (Mt)"),
                        html.P("Applied when 'Apply system CO2 constraints' is ON in Settings.",
                               className="text-muted small"),
                        _grid("em-sys-grid"),
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save Country Cap", id="save-em-cnt-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-em-cnt-msg"), width="auto"),
                        ], className="mb-2"),
                        html.H6("Country-level CO₂ cap (Mt)"),
                        html.P("Applied when 'Apply country CO2 constraint' is ON in Settings.",
                               className="text-muted small"),
                        _grid("em-cnt-grid"),
                    ], width=6),
                ]),
            ]),

            # ── Max Fuel Limit ────────────────────────────────────────────
            dbc.Tab(label="Fuel Limits", tab_id="tab-fuel-lim", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-fuel-lim-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-fuel-lim-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Maximum annual fuel consumption limit by zone and fuel (GWh or mmBTU).",
                               className="text-muted small"),
                        _grid("fuel-lim-grid"),
                    ], width=12),
                ]),
            ]),
        ]),
    ])


@callback(
    Output("cp-grid",       "rowData"),  Output("cp-grid",       "columnDefs"),
    Output("em-sys-grid",   "rowData"),  Output("em-sys-grid",   "columnDefs"),
    Output("em-cnt-grid",   "rowData"),  Output("em-cnt-grid",   "columnDefs"),
    Output("fuel-lim-grid", "rowData"),  Output("fuel-lim-grid", "columnDefs"),
    Output("cp-chart",      "figure"),
    Input("const-project",  "value"),
)
def load(folder):
    empty = ([], [])
    empty_fig = px.line(title="No data", template="plotly_white")
    if not folder:
        return *empty, *empty, *empty, *empty, empty_fig

    df_cp       = dl.load_input(folder, "carbon_price")
    df_em_s     = dl.load_input(folder, "emissions_total")
    df_em_c     = dl.load_input(folder, "emissions_country")
    df_fuel_lim = dl.load_input(folder, "max_fuel")

    def rc(df):
        if df.empty:
            return [], []
        return df.to_dict("records"), _col_defs(df)

    fig = empty_fig
    if not df_cp.empty and "y" in df_cp.columns and "value" in df_cp.columns:
        fig = px.line(df_cp, x="y", y="value", markers=True,
                      title="Carbon Price ($/tCO₂)",
                      labels={"value": "$/tCO₂", "y": "Year"},
                      template="plotly_white")
        fig.update_xaxes(type="category")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

    return (*rc(df_cp), *rc(df_em_s), *rc(df_em_c), *rc(df_fuel_lim), fig)


@callback(Output("save-cp-msg",  "children"), Input("save-cp-btn",  "n_clicks"),
          State("cp-grid",       "rowData"),  State("const-project", "value"),
          prevent_initial_call=True)
def save_cp(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "carbon_price", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-em-sys-msg", "children"), Input("save-em-sys-btn", "n_clicks"),
          State("em-sys-grid",      "rowData"), State("const-project",    "value"),
          prevent_initial_call=True)
def save_em_sys(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "emissions_total", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-em-cnt-msg", "children"), Input("save-em-cnt-btn", "n_clicks"),
          State("em-cnt-grid",      "rowData"), State("const-project",    "value"),
          prevent_initial_call=True)
def save_em_cnt(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "emissions_country", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-fuel-lim-msg", "children"), Input("save-fuel-lim-btn", "n_clicks"),
          State("fuel-lim-grid",      "rowData"), State("const-project",       "value"),
          prevent_initial_call=True)
def save_fuel_lim(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "max_fuel", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")
