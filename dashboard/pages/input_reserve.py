"""Input Editor — Reserve margins & spinning reserve requirements."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT


def _grid(grid_id: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id, rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        style={"height": "340px"}, className="ag-theme-alpine",
    )


def _col_defs(df: pd.DataFrame, read_only: list) -> list:
    return [{"field": c, "editable": c not in read_only,
             "cellStyle": {} if c not in read_only
                         else {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
            for c in df.columns]


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)
    return html.Div([
        dbc.Row([
            dbc.Col(html.H4("Reserve Requirements", className="mb-0"), width="auto"),
        ], className="mb-1 align-items-center"),
        html.P("Edit planning reserve margins and spinning reserve requirements.",
               className="text-muted mb-3"),
        html.Div(
            dcc.Dropdown(id="reserve-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),
        dbc.Tabs([
            dbc.Tab(label="Planning Reserve Margin", tab_id="tab-prm", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("r-prm"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-prm2-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-prm2-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Required planning reserve margin as a fraction (0.15 = 15%).",
                               className="text-muted small"),
                        _grid("prm2-grid"),
                        html.Div(make_open_folder_btn("rsv-prm-open"), className="mt-1 mb-2"),
                    ], width=8),
                ]),
            ]),
            dbc.Tab(label="Spinning Reserve (Country)", tab_id="tab-src", children=[
                make_variant_bar("r-src"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-src-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-src-msg"), width="auto"),
                ]),
                html.P("Country-level spinning reserve requirement. Tip: typically set to cover the largest single generating unit in the zone.",
                       className="text-muted small"),
                _grid("src-grid"),
                html.Div(make_open_folder_btn("rsv-src-open"), className="mt-1 mb-2"),
            ]),
            dbc.Tab(label="Spinning Reserve (System)", tab_id="tab-srs", children=[
                make_variant_bar("r-srs"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-srs-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-srs-msg"), width="auto"),
                ]),
                html.P("System-wide spinning reserve requirement. Tip: typically set to cover the largest single generating unit in the system.",
                       className="text-muted small"),
                _grid("srs-grid"),
                html.Div(make_open_folder_btn("rsv-srs-open"), className="mt-1 mb-2"),
            ]),
        ]),
    ])


@callback(
    Output("prm2-grid",      "rowData"),  Output("prm2-grid", "columnDefs"),
    Output("r-prm-variant",  "options"),
    Output("src-grid",       "rowData"),  Output("src-grid",  "columnDefs"),
    Output("r-src-variant",  "options"),
    Output("srs-grid",       "rowData"),  Output("srs-grid",  "columnDefs"),
    Output("r-srs-variant",  "options"),
    Input("reserve-project", "value"),
    Input("r-prm-variant",   "value"),
    Input("r-src-variant",   "value"),
    Input("r-srs-variant",   "value"),
)
def load(folder, prm_var, src_var, srs_var):
    empty = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    if not folder:
        return (*empty, base_opts, *empty, base_opts, *empty, base_opts)

    df_prm = dl.load_variant(folder, "planning_reserve",  prm_var)
    df_src = dl.load_variant(folder, "spinning_country",  src_var)
    df_srs = dl.load_variant(folder, "spinning_system",   srs_var)

    def rc(df, ro): return (df.to_dict("records"), _col_defs(df, ro)) if not df.empty else ([], [])

    return (
        *rc(df_prm, ["z"]),  variant_options(folder, "planning_reserve"),
        *rc(df_src, ["c"]),  variant_options(folder, "spinning_country"),
        *rc(df_srs, []),     variant_options(folder, "spinning_system"),
    )


@callback(Output("save-prm2-msg", "children"), Input("save-prm2-btn", "n_clicks"),
          State("prm2-grid", "rowData"), State("reserve-project", "value"),
          State("r-prm-variant", "value"), prevent_initial_call=True)
def save_prm(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "planning_reserve", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-src-msg", "children"), Input("save-src-btn", "n_clicks"),
          State("src-grid", "rowData"), State("reserve-project", "value"),
          State("r-src-variant", "value"), prevent_initial_call=True)
def save_src(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "spinning_country", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-srs-msg", "children"), Input("save-srs-btn", "n_clicks"),
          State("srs-grid", "rowData"), State("reserve-project", "value"),
          State("r-srs-variant", "value"), prevent_initial_call=True)
def save_srs(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "spinning_system", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("r-prm-dup-msg", "children"),
          Output("r-prm-variant", "options", allow_duplicate=True),
          Output("r-prm-variant", "value", allow_duplicate=True),
          Input("r-prm-dup-btn", "n_clicks"),
          State("r-prm-variant", "value"), State("r-prm-dup-name", "value"),
          State("reserve-project", "value"), prevent_initial_call=True)
def dup_prm(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "planning_reserve", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "planning_reserve"), name
    return "Name exists or error", no_update, no_update


@callback(Output("r-src-dup-msg", "children"),
          Output("r-src-variant", "options", allow_duplicate=True),
          Output("r-src-variant", "value", allow_duplicate=True),
          Input("r-src-dup-btn", "n_clicks"),
          State("r-src-variant", "value"), State("r-src-dup-name", "value"),
          State("reserve-project", "value"), prevent_initial_call=True)
def dup_src(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "spinning_country", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "spinning_country"), name
    return "Name exists or error", no_update, no_update


@callback(Output("r-srs-dup-msg", "children"),
          Output("r-srs-variant", "options", allow_duplicate=True),
          Output("r-srs-variant", "value", allow_duplicate=True),
          Input("r-srs-dup-btn", "n_clicks"),
          State("r-srs-variant", "value"), State("r-srs-dup-name", "value"),
          State("reserve-project", "value"), prevent_initial_call=True)
def dup_srs(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "spinning_system", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "spinning_system"), name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("rsv-prm-open", "n_clicks"),
          State("reserve-project", "value"),
          State("r-prm-variant", "value"),
          prevent_initial_call=True)
def open_prm_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "planning_reserve", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("rsv-src-open", "n_clicks"),
          State("reserve-project", "value"),
          State("r-src-variant", "value"),
          prevent_initial_call=True)
def open_src_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "spinning_country", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("rsv-srs-open", "n_clicks"),
          State("reserve-project", "value"),
          State("r-srs-variant", "value"),
          prevent_initial_call=True)
def open_srs_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "spinning_system", variant)
