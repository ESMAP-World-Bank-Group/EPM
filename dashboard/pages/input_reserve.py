"""Input Editor — Reserve margins & spinning reserve requirements."""

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
        style={"height": "340px"},
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
        html.H4("Reserve Requirements", className="mb-1"),
        html.P("Edit planning reserve margins and spinning reserve requirements.",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="reserve-project", options=options,
                             value=default, clearable=False),
            ], width=3),
        ], className="mb-3"),

        dbc.Tabs([
            # ── Planning Reserve Margin ───────────────────────────────────
            dbc.Tab(label="Planning Reserve Margin", tab_id="tab-prm", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-prm2-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-prm2-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Required planning reserve margin as a fraction (e.g. 0.15 = 15%). "
                               "Applied per zone per year.",
                               className="text-muted small"),
                        _grid("prm2-grid"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Reserve margin by zone", className="text-muted"),
                        dcc.Graph(id="prm-chart"),
                    ], width=6),
                ]),
            ]),

            # ── Spinning Reserve — Country ────────────────────────────────
            dbc.Tab(label="Spinning Reserve (Country)", tab_id="tab-src", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-src-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-src-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Country-level spinning reserve requirement (MW or fraction).",
                               className="text-muted small"),
                        _grid("src-grid"),
                    ], width=12),
                ]),
            ]),

            # ── Spinning Reserve — System ─────────────────────────────────
            dbc.Tab(label="Spinning Reserve (System)", tab_id="tab-srs", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-srs-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-srs-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("System-wide spinning reserve requirement (MW or fraction).",
                               className="text-muted small"),
                        _grid("srs-grid"),
                    ], width=12),
                ]),
            ]),
        ]),
    ])


@callback(
    Output("prm2-grid", "rowData"),  Output("prm2-grid", "columnDefs"),
    Output("src-grid",  "rowData"),  Output("src-grid",  "columnDefs"),
    Output("srs-grid",  "rowData"),  Output("srs-grid",  "columnDefs"),
    Output("prm-chart", "figure"),
    Input("reserve-project", "value"),
)
def load(folder):
    empty = ([], [])
    empty_fig = px.bar(title="No data", template="plotly_white")
    if not folder:
        return *empty, *empty, *empty, empty_fig

    df_prm = dl.load_input(folder, "planning_reserve")
    df_src = dl.load_input(folder, "spinning_country")
    df_srs = dl.load_input(folder, "spinning_system")

    def rc(df, ro):
        if df.empty:
            return [], []
        return df.to_dict("records"), _col_defs(df, ro)

    # Planning reserve chart
    fig = empty_fig
    if not df_prm.empty and "value" in df_prm.columns:
        color_col = "z" if "z" in df_prm.columns else None
        x_col     = "y" if "y" in df_prm.columns else df_prm.columns[0]
        df_plot = df_prm.copy()
        df_plot["value_pct"] = df_plot["value"] * 100
        fig = px.line(df_plot, x=x_col, y="value_pct", color=color_col,
                      markers=True,
                      title="Planning Reserve Margin (%)",
                      labels={"value_pct": "%", x_col: "Year"},
                      template="plotly_white")
        fig.update_xaxes(type="category")

    return (
        *rc(df_prm, ["z"]),
        *rc(df_src, ["c"]),
        *rc(df_srs, []),
        fig,
    )


@callback(Output("save-prm2-msg", "children"), Input("save-prm2-btn", "n_clicks"),
          State("prm2-grid", "rowData"), State("reserve-project", "value"),
          prevent_initial_call=True)
def save_prm(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "planning_reserve", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-src-msg", "children"), Input("save-src-btn", "n_clicks"),
          State("src-grid", "rowData"), State("reserve-project", "value"),
          prevent_initial_call=True)
def save_src(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "spinning_country", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")


@callback(Output("save-srs-msg", "children"), Input("save-srs-btn", "n_clicks"),
          State("srs-grid", "rowData"), State("reserve-project", "value"),
          prevent_initial_call=True)
def save_srs(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") if dl.save_input(folder, "spinning_system", pd.DataFrame(rows)) else dbc.Badge("Failed", color="danger")
