"""Input Editor — Constraints (carbon price, emissions caps, fuel limits)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT


def _grid(grid_id: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id, rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        dashGridOptions={"rowSelection": "multiple"},
        style={"height": "360px"}, className="ag-theme-alpine",
    )


def _icon_btns(add_id, del_id):
    return [
        dbc.Button(html.I(className="bi bi-plus-lg"), id=add_id, color="link",
                   className="text-secondary p-0 me-1",
                   style={"fontSize": "0.78rem"}, title="Add row"),
        dbc.Button(html.I(className="bi bi-trash"), id=del_id, color="link",
                   className="text-danger p-0",
                   style={"fontSize": "0.78rem"}, title="Delete selected"),
    ]


def _col_defs(df: pd.DataFrame) -> list:
    return [{"field": c, "editable": True, "cellStyle": {}} for c in df.columns]


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)
    return html.Div([
        dbc.Row([
            dbc.Col(html.H4("Policy Constraints", className="mb-0"), width="auto"),
            dbc.Col(
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Reload"],
                           id="con-reload-btn", color="outline-secondary", size="sm"),
                width="auto", className="ms-auto",
            ),
        ], className="mb-1 align-items-center justify-content-between"),
        html.P("Edit carbon price trajectories and CO₂ emissions caps.",
               className="text-muted mb-3"),
        html.Div(
            dcc.Dropdown(id="const-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),
        dbc.Tabs([
            # ── Carbon Price ──────────────────────────────────────────────
            dbc.Tab(label="Carbon Price", tab_id="tab-cp", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("c-cp"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-cp-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-cp-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("Carbon price trajectory ($/tCO₂).",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col(_icon_btns("add-cp-btn", "del-cp-btn"),
                                    width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("cp-grid"),
                        html.Div(make_open_folder_btn("con-cp-open"), className="mt-1 mb-2"),
                    ], width=5),
                    dbc.Col([
                        dcc.Graph(id="cp-chart", config={"displayModeBar": False}),
                    ], width=7),
                ]),
            ]),
            # ── Emissions Caps ────────────────────────────────────────────
            dbc.Tab(label="Emissions Caps", tab_id="tab-em", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("c-ems"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save System Cap", id="save-em-sys-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-em-sys-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.H6("System-wide CO₂ cap (Mt)", className="mb-0"), width="auto"),
                            dbc.Col(_icon_btns("add-em-sys-btn", "del-em-sys-btn"),
                                    width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("em-sys-grid"),
                        html.Div(make_open_folder_btn("con-ems-open"), className="mt-1 mb-2"),
                    ], width=6),
                    dbc.Col([
                        make_variant_bar("c-emc"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save Country Cap", id="save-em-cnt-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-em-cnt-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.H6("Country-level CO₂ cap (Mt)", className="mb-0"), width="auto"),
                            dbc.Col(_icon_btns("add-em-cnt-btn", "del-em-cnt-btn"),
                                    width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("em-cnt-grid"),
                        html.Div(make_open_folder_btn("con-emc-open"), className="mt-1 mb-2"),
                    ], width=6),
                ]),
            ]),
            # ── Fuel Limits ───────────────────────────────────────────────
            dbc.Tab(label="Fuel Limits", tab_id="tab-fuel-lim", children=[
                make_variant_bar("c-fl"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-fuel-lim-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-fuel-lim-msg"), width="auto"),
                ]),
                dbc.Row([
                    dbc.Col(html.P("Maximum annual fuel consumption by zone and fuel.",
                                   className="text-muted small mb-1"), width="auto"),
                    dbc.Col(_icon_btns("add-fuel-lim-btn", "del-fuel-lim-btn"),
                            width="auto", className="ms-auto d-flex align-items-center"),
                ], className="align-items-center mb-1"),
                _grid("fuel-lim-grid"),
                html.Div(make_open_folder_btn("con-fl-open"), className="mt-1 mb-2"),
            ]),
        ]),
    ])


@callback(
    Output("cp-grid",       "rowData"),  Output("cp-grid",       "columnDefs"),
    Output("c-cp-variant",  "options"),
    Output("em-sys-grid",   "rowData"),  Output("em-sys-grid",   "columnDefs"),
    Output("c-ems-variant", "options"),
    Output("em-cnt-grid",   "rowData"),  Output("em-cnt-grid",   "columnDefs"),
    Output("c-emc-variant", "options"),
    Output("fuel-lim-grid", "rowData"),  Output("fuel-lim-grid", "columnDefs"),
    Output("c-fl-variant",  "options"),
    Output("cp-chart",      "figure"),
    Input("const-project",  "value"),
    Input("c-cp-variant",   "value"),
    Input("c-ems-variant",  "value"),
    Input("c-emc-variant",  "value"),
    Input("c-fl-variant",   "value"),
    Input("con-reload-btn", "n_clicks"),
)
def load(folder, cp_var, ems_var, emc_var, fl_var, _reload=None):
    if _reload:
        dl.clear_input_cache()
    empty = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    empty_fig = px.line(title="No data", template="plotly_white")
    if not folder:
        return (*empty, base_opts, *empty, base_opts, *empty, base_opts,
                *empty, base_opts, empty_fig)

    df_cp       = dl.load_variant(folder, "carbon_price",      cp_var)
    df_em_s     = dl.load_variant(folder, "emissions_total",   ems_var)
    df_em_c     = dl.load_variant(folder, "emissions_country", emc_var)
    df_fuel_lim = dl.load_variant(folder, "max_fuel",          fl_var)

    def rc(df): return (df.to_dict("records"), _col_defs(df)) if not df.empty else ([], [])

    fig = empty_fig
    if not df_cp.empty and "y" in df_cp.columns and "value" in df_cp.columns:
        df_plot = df_cp.copy()
        df_plot["value"] = pd.to_numeric(df_plot["value"], errors="coerce")
        df_plot = df_plot.dropna(subset=["value"])
        if not df_plot.empty:
            fig = px.line(df_plot, x="y", y="value", markers=True,
                          title="Carbon Price ($/tCO₂)",
                          labels={"value": "$/tCO₂", "y": "Year"},
                          template="plotly_white")
            fig.update_xaxes(type="category")
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        else:
            fig = px.line(title="No carbon price values set", template="plotly_white")

    return (
        *rc(df_cp),       variant_options(folder, "carbon_price"),
        *rc(df_em_s),     variant_options(folder, "emissions_total"),
        *rc(df_em_c),     variant_options(folder, "emissions_country"),
        *rc(df_fuel_lim), variant_options(folder, "max_fuel"),
        fig,
    )


@callback(Output("save-cp-msg", "children"), Input("save-cp-btn", "n_clicks"),
          State("cp-grid", "rowData"), State("const-project", "value"),
          State("c-cp-variant", "value"), prevent_initial_call=True)
def save_cp(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "carbon_price", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-em-sys-msg", "children"), Input("save-em-sys-btn", "n_clicks"),
          State("em-sys-grid", "rowData"), State("const-project", "value"),
          State("c-ems-variant", "value"), prevent_initial_call=True)
def save_em_sys(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "emissions_total", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-em-cnt-msg", "children"), Input("save-em-cnt-btn", "n_clicks"),
          State("em-cnt-grid", "rowData"), State("const-project", "value"),
          State("c-emc-variant", "value"), prevent_initial_call=True)
def save_em_cnt(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "emissions_country", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-fuel-lim-msg", "children"), Input("save-fuel-lim-btn", "n_clicks"),
          State("fuel-lim-grid", "rowData"), State("const-project", "value"),
          State("c-fl-variant", "value"), prevent_initial_call=True)
def save_fuel_lim(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "max_fuel", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("c-cp-dup-msg", "children"),
          Output("c-cp-variant", "options", allow_duplicate=True),
          Output("c-cp-variant", "value", allow_duplicate=True),
          Input("c-cp-dup-btn", "n_clicks"),
          State("c-cp-variant", "value"), State("c-cp-dup-name", "value"),
          State("const-project", "value"), prevent_initial_call=True)
def dup_cp(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "carbon_price", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "carbon_price"), name
    return "Name exists or error", no_update, no_update


@callback(Output("c-ems-dup-msg", "children"),
          Output("c-ems-variant", "options", allow_duplicate=True),
          Output("c-ems-variant", "value", allow_duplicate=True),
          Input("c-ems-dup-btn", "n_clicks"),
          State("c-ems-variant", "value"), State("c-ems-dup-name", "value"),
          State("const-project", "value"), prevent_initial_call=True)
def dup_ems(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "emissions_total", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "emissions_total"), name
    return "Name exists or error", no_update, no_update


@callback(Output("c-emc-dup-msg", "children"),
          Output("c-emc-variant", "options", allow_duplicate=True),
          Output("c-emc-variant", "value", allow_duplicate=True),
          Input("c-emc-dup-btn", "n_clicks"),
          State("c-emc-variant", "value"), State("c-emc-dup-name", "value"),
          State("const-project", "value"), prevent_initial_call=True)
def dup_emc(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "emissions_country", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "emissions_country"), name
    return "Name exists or error", no_update, no_update


@callback(Output("c-fl-dup-msg", "children"),
          Output("c-fl-variant", "options", allow_duplicate=True),
          Output("c-fl-variant", "value", allow_duplicate=True),
          Input("c-fl-dup-btn", "n_clicks"),
          State("c-fl-variant", "value"), State("c-fl-dup-name", "value"),
          State("const-project", "value"), prevent_initial_call=True)
def dup_fl(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "max_fuel", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "max_fuel"), name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("con-cp-open", "n_clicks"),
          State("const-project", "value"),
          State("c-cp-variant", "value"),
          prevent_initial_call=True)
def open_cp_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "carbon_price", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("con-ems-open", "n_clicks"),
          State("const-project", "value"),
          State("c-ems-variant", "value"),
          prevent_initial_call=True)
def open_ems_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "emissions_total", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("con-emc-open", "n_clicks"),
          State("const-project", "value"),
          State("c-emc-variant", "value"),
          prevent_initial_call=True)
def open_emc_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "emissions_country", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("con-fl-open", "n_clicks"),
          State("const-project", "value"),
          State("c-fl-variant", "value"),
          prevent_initial_call=True)
def open_fl_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "max_fuel", variant)


# ---------------------------------------------------------------------------
# Add / Delete row callbacks
# ---------------------------------------------------------------------------

def _empty_row(rows):
    return {k: "" for k in rows[0].keys()} if rows else {}

def _delete_selected(rows, selected):
    if not selected: return rows
    sel = {tuple(sorted(r.items())) for r in selected}
    return [r for r in rows if tuple(sorted(r.items())) not in sel]

for _grid_id, _add_id, _del_id in [
    ("cp-grid",       "add-cp-btn",       "del-cp-btn"),
    ("em-sys-grid",   "add-em-sys-btn",   "del-em-sys-btn"),
    ("em-cnt-grid",   "add-em-cnt-btn",   "del-em-cnt-btn"),
    ("fuel-lim-grid", "add-fuel-lim-btn", "del-fuel-lim-btn"),
]:
    @callback(Output(_grid_id, "rowData", allow_duplicate=True),
              Input(_add_id, "n_clicks"),
              State(_grid_id, "rowData"),
              prevent_initial_call=True)
    def _add(n, rows, _gid=_grid_id):
        rows = rows or []
        return rows + [_empty_row(rows)]

    @callback(Output(_grid_id, "rowData", allow_duplicate=True),
              Input(_del_id, "n_clicks"),
              State(_grid_id, "rowData"),
              State(_grid_id, "selectedRows"),
              prevent_initial_call=True)
    def _del(n, rows, selected, _gid=_grid_id):
        return _delete_selected(rows or [], selected or [])
