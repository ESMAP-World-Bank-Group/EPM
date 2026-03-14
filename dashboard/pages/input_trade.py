"""Input Editor — Trade & Transmission."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import data_loader as dl


def _grid(grid_id: str, height: str = "360px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        style={"height": height},
        className="ag-theme-alpine",
    )


def _col_defs(df: pd.DataFrame, read_only: list) -> list:
    return [
        {"field": c, "editable": c not in read_only,
         "cellStyle": {} if c not in read_only
                     else {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
        for c in df.columns
    ]


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Trade & Transmission", className="mb-1"),
        html.P("Edit interconnection capacities, candidate new lines and trade prices.",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="trade-project", options=options,
                             value=default, clearable=False),
            ], width=3),
        ], className="mb-3"),

        dbc.Tabs([
            # ── Transfer Limits ──────────────────────────────────────────
            dbc.Tab(label="Transfer Limits", tab_id="tab-tl", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-tl-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-tl-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Existing interconnection capacity between zones (MW).",
                               className="text-muted small"),
                        _grid("tl-grid"),
                    ], width=7),
                    dbc.Col([
                        html.H6("Capacity by corridor", className="text-muted"),
                        dcc.Graph(id="tl-chart"),
                    ], width=5),
                ]),
            ]),

            # ── New Transmission ─────────────────────────────────────────
            dbc.Tab(label="New Transmission", tab_id="tab-nt", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-nt-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-nt-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Candidate new transmission lines — set capacity to 0 to exclude.",
                               className="text-muted small"),
                        _grid("nt-grid"),
                    ], width=12),
                ]),
            ]),

            # ── Trade Prices ─────────────────────────────────────────────
            dbc.Tab(label="Trade Prices", tab_id="tab-tp", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-tp-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-tp-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Import/export prices with external zones ($/MWh).",
                               className="text-muted small"),
                        _grid("tp-grid"),
                    ], width=7),
                    dbc.Col([
                        html.H6("Price by zone", className="text-muted"),
                        dcc.Graph(id="tp-chart"),
                    ], width=5),
                ]),
            ]),

            # ── External Transfer Limits ──────────────────────────────────
            dbc.Tab(label="External Transfer Limits", tab_id="tab-etl", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-etl-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-etl-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Max capacity for imports/exports with external zones (MW).",
                               className="text-muted small"),
                        _grid("etl-grid"),
                    ], width=12),
                ]),
            ]),
        ]),
    ])


@callback(
    Output("tl-grid",    "rowData"),  Output("tl-grid",  "columnDefs"),
    Output("nt-grid",    "rowData"),  Output("nt-grid",  "columnDefs"),
    Output("tp-grid",    "rowData"),  Output("tp-grid",  "columnDefs"),
    Output("etl-grid",   "rowData"),  Output("etl-grid", "columnDefs"),
    Output("tl-chart",   "figure"),
    Output("tp-chart",   "figure"),
    Input("trade-project", "value"),
)
def load(folder):
    empty = ([], [])
    empty_fig = px.bar(title="No data", template="plotly_white")
    if not folder:
        return *empty, *empty, *empty, *empty, empty_fig, empty_fig

    df_tl  = dl.load_input(folder, "transfer_limit")
    df_nt  = dl.load_input(folder, "new_transmission")
    df_tp  = dl.load_input(folder, "trade_price")
    df_etl = dl.load_input(folder, "ext_transfer")

    def rc(df, ro):
        if df.empty:
            return [], []
        return df.to_dict("records"), _col_defs(df, ro)

    # Transfer limit chart — bar by corridor
    fig_tl = empty_fig
    if not df_tl.empty:
        id_cols = [c for c in df_tl.columns if c not in ("value",) and
                   not str(c).isdigit()]
        year_cols = [c for c in df_tl.columns if str(c).isdigit() or
                     (isinstance(c, int))]
        if "value" in df_tl.columns and "z" in df_tl.columns:
            corridor = df_tl["z"].astype(str)
            if "z2" in df_tl.columns:
                corridor = df_tl["z"] + " → " + df_tl["z2"]
            df_plot = df_tl.copy()
            df_plot["corridor"] = corridor
            y_col = "y" if "y" in df_tl.columns else df_tl.columns[-1]
            fig_tl = px.bar(df_plot, x="corridor", y="value",
                            color="y" if "y" in df_tl.columns else None,
                            title="Transfer Limits (MW)",
                            labels={"value": "MW", "corridor": ""},
                            template="plotly_white", barmode="group")
            fig_tl.update_xaxes(tickangle=30)

    # Trade price chart
    fig_tp = empty_fig
    if not df_tp.empty and "value" in df_tp.columns:
        color_col = "z" if "z" in df_tp.columns else None
        x_col = "y" if "y" in df_tp.columns else df_tp.columns[0]
        fig_tp = px.line(df_tp, x=x_col, y="value", color=color_col,
                         markers=True,
                         title="Trade Prices ($/MWh)",
                         labels={"value": "$/MWh"},
                         template="plotly_white")
        fig_tp.update_xaxes(type="category")

    return (
        *rc(df_tl,  ["z", "z2"]),
        *rc(df_nt,  ["z", "z2"]),
        *rc(df_tp,  ["z", "zext"]),
        *rc(df_etl, ["z", "zext"]),
        fig_tl, fig_tp,
    )


# Save callbacks
@callback(Output("save-tl-msg",  "children"), Input("save-tl-btn",  "n_clicks"),
          State("tl-grid",  "rowData"), State("trade-project", "value"),
          prevent_initial_call=True)
def save_tl(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "transfer_limit", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-nt-msg",  "children"), Input("save-nt-btn",  "n_clicks"),
          State("nt-grid",  "rowData"), State("trade-project", "value"),
          prevent_initial_call=True)
def save_nt(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "new_transmission", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-tp-msg",  "children"), Input("save-tp-btn",  "n_clicks"),
          State("tp-grid",  "rowData"), State("trade-project", "value"),
          prevent_initial_call=True)
def save_tp(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "trade_price", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-etl-msg", "children"), Input("save-etl-btn", "n_clicks"),
          State("etl-grid", "rowData"), State("trade-project", "value"),
          prevent_initial_call=True)
def save_etl(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "ext_transfer", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")
