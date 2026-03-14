"""Input Editor — Supply (generator data, fuel prices, CAPEX trajectories)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import data_loader as dl


def _col_defs(df: pd.DataFrame, read_only_cols: list) -> list:
    defs = []
    for c in df.columns:
        editable = c not in read_only_cols
        defs.append({
            "field": c, "editable": editable,
            "sortable": True, "filter": True, "resizable": True,
            "cellStyle": {} if editable else {"backgroundColor": "#f8f9fa", "color": "#6c757d"},
        })
    return defs


def _grid(grid_id: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[],
        columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 100},
        dashGridOptions={"animateRows": True},
        style={"height": "420px"},
        className="ag-theme-alpine",
    )


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Supply Inputs", className="mb-1"),
        html.P("Edit generator data, fuel prices and CAPEX trajectories. "
               "Changes are saved to CSV when you click Save.",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="supply-project", options=options,
                             value=default, clearable=False),
            ], width=3),
        ], className="mb-3"),

        dbc.Tabs(id="supply-tabs", active_tab="tab-gen", children=[

            dbc.Tab(label="Generator Data", tab_id="tab-gen", children=[
                dbc.Row(className="mt-3 mb-2", children=[
                    dbc.Col(dbc.Button("Save Generator Data", id="save-gen-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-gen-msg"), width="auto"),
                ]),
                _grid("gen-grid"),
            ]),

            dbc.Tab(label="Fuel Prices", tab_id="tab-fuel", children=[
                dbc.Row(className="mt-3 mb-2", children=[
                    dbc.Col(dbc.Button("Save Fuel Prices", id="save-fuel-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-fuel-msg"), width="auto"),
                ]),
                _grid("fuel-grid"),
            ]),

            dbc.Tab(label="CAPEX Trajectories", tab_id="tab-capex", children=[
                dbc.Row(className="mt-3 mb-2", children=[
                    dbc.Col(dbc.Button("Save CAPEX", id="save-capex-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-capex-msg"), width="auto"),
                ]),
                _grid("capex-grid"),
            ]),
        ]),
    ])


@callback(
    Output("gen-grid",   "rowData"),
    Output("gen-grid",   "columnDefs"),
    Output("fuel-grid",  "rowData"),
    Output("fuel-grid",  "columnDefs"),
    Output("capex-grid", "rowData"),
    Output("capex-grid", "columnDefs"),
    Input("supply-project", "value"),
)
def load_grids(folder):
    empty = ([], [])
    if not folder:
        return *empty, *empty, *empty

    df_gen   = dl.load_input(folder, "gen_data")
    df_fuel  = dl.load_input(folder, "fuel_price")
    df_capex = dl.load_input(folder, "capex")

    def rows_cols(df, ro_cols):
        if df.empty:
            return [], []
        return df.to_dict("records"), _col_defs(df, ro_cols)

    return (
        *rows_cols(df_gen,   ["z", "g"]),
        *rows_cols(df_fuel,  ["z", "f"]),
        *rows_cols(df_capex, ["f"]),
    )


@callback(
    Output("save-gen-msg",  "children"),
    Input("save-gen-btn",   "n_clicks"),
    State("gen-grid",       "rowData"),
    State("supply-project", "value"),
    prevent_initial_call=True,
)
def save_gen(n, rows, folder):
    if not rows or not folder:
        return dbc.Badge("Nothing to save", color="warning")
    ok = dl.save_input(folder, "gen_data", pd.DataFrame(rows))
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")


@callback(
    Output("save-fuel-msg", "children"),
    Input("save-fuel-btn",  "n_clicks"),
    State("fuel-grid",      "rowData"),
    State("supply-project", "value"),
    prevent_initial_call=True,
)
def save_fuel(n, rows, folder):
    if not rows or not folder:
        return dbc.Badge("Nothing to save", color="warning")
    ok = dl.save_input(folder, "fuel_price", pd.DataFrame(rows))
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")


@callback(
    Output("save-capex-msg", "children"),
    Input("save-capex-btn",  "n_clicks"),
    State("capex-grid",      "rowData"),
    State("supply-project",  "value"),
    prevent_initial_call=True,
)
def save_capex(n, rows, folder):
    if not rows or not folder:
        return dbc.Badge("Nothing to save", color="warning")
    ok = dl.save_input(folder, "capex", pd.DataFrame(rows))
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")
