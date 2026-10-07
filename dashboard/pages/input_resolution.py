"""Input Editor — Spatial & Temporal Resolution (zcmap, y, pHours)."""

import json

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from components.page_nav import make_page_nav
from config import INPUT_ROOT


def _grid(grid_id: str, height: str = "340px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id, rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        dashGridOptions={"rowSelection": "multiple"},
        style={"height": height}, className="ag-theme-alpine",
    )


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)
    return html.Div([
        make_page_nav("input-resolution"),
        dbc.Row([
            dbc.Col(html.H4("Resolution", className="mb-0"), width="auto"),
            dbc.Col(
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Reload"],
                           id="res-reload-btn", color="outline-secondary", size="sm"),
                width="auto", className="ms-auto",
            ),
        ], className="mb-1 align-items-center justify-content-between"),
        html.P("Define the spatial (zones/countries) and temporal (years, periods) resolution.",
               className="text-muted mb-3"),

        html.Div(
            dcc.Dropdown(id="res-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),

        dbc.Tabs([
            # ── Spatial: Zone–Country Map ──────────────────────────────────
            dbc.Tab(label="Zones (zcmap)", tab_id="tab-zcmap", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-zcmap-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-zcmap-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("Zone → country mapping. Edit the 'c' column to reassign zones.",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col([
                                dbc.Button(html.I(className="bi bi-plus-lg"), id="add-zcmap-btn",
                                           color="link", className="text-secondary p-0 me-1",
                                           style={"fontSize": "0.78rem"}, title="Add row"),
                                dbc.Button(html.I(className="bi bi-trash"), id="del-zcmap-btn",
                                           color="link", className="text-danger p-0",
                                           style={"fontSize": "0.78rem"}, title="Delete selected"),
                            ], width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("zcmap-grid", "380px"),
                        html.Div(make_open_folder_btn("res-zcmap-open"), className="mt-1 mb-2"),
                    ], width=4),
                    dbc.Col([
                        dcc.Graph(id="zcmap-chart", config={"displayModeBar": False},
                                  style={"height": "430px"}),
                    ], width=8),
                ]),
            ]),

            # ── Temporal: Planning Years ───────────────────────────────────
            dbc.Tab(label="Planning Years (y)", tab_id="tab-years", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("res-y"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-years-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-years-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("List of modelled planning years.",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col([
                                dbc.Button(html.I(className="bi bi-plus-lg"), id="add-years-btn",
                                           color="link", className="text-secondary p-0 me-1",
                                           style={"fontSize": "0.78rem"}, title="Add row"),
                                dbc.Button(html.I(className="bi bi-trash"), id="del-years-btn",
                                           color="link", className="text-danger p-0",
                                           style={"fontSize": "0.78rem"}, title="Delete selected"),
                            ], width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("years-grid", "340px"),
                        html.Div(make_open_folder_btn("res-years-open"), className="mt-1 mb-2"),
                    ], width=5),
                ]),
            ]),

            # ── Temporal: Representative Periods ──────────────────────────
            dbc.Tab(label="Time Periods (pHours)", tab_id="tab-phours", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("res-ph"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-phours-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-phours-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("Representative periods (q × d) and their hourly weights.",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col([
                                dbc.Button(html.I(className="bi bi-plus-lg"), id="add-phours-btn",
                                           color="link", className="text-secondary p-0 me-1",
                                           style={"fontSize": "0.78rem"}, title="Add row"),
                                dbc.Button(html.I(className="bi bi-trash"), id="del-phours-btn",
                                           color="link", className="text-danger p-0",
                                           style={"fontSize": "0.78rem"}, title="Delete selected"),
                            ], width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("phours-grid", "380px"),
                        html.Div(make_open_folder_btn("res-ph-open"), className="mt-1 mb-2"),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id="phours-chart", config={"displayModeBar": False},
                                  style={"height": "430px"}),
                    ], width=6),
                ]),
            ]),
        ]),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_geojson(folder):
    """Try to load linestring_countries.geojson from the input folder. Returns dict or None."""
    path = INPUT_ROOT / folder / "linestring_countries.geojson"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _zone_network_figure(gj, df_zc):
    """
    Simple zone map: one diamond marker per zone with name label.
    Country borders shown for geographic context.
    """
    if gj is None:
        fig = go.Figure()
        fig.update_layout(title="linestring_countries.geojson not found in this project folder",
                          template="plotly_white")
        return fig

    features = gj.get("features", [])

    # Collect unique zones with centroid coordinates
    # First pass: primary z zones (have lat/lon)
    zone_info = {}
    for feat in features:
        props = feat["properties"]
        z = props.get("z", "")
        if z and (z not in zone_info or zone_info[z]["lat"] is None):
            zone_info[z] = {
                "lat": props.get("country_ini_lat"),
                "lon": props.get("country_ini_lon"),
            }
    # Second pass: add z_other zones not yet seen
    for feat in features:
        props = feat["properties"]
        z2 = props.get("z_other", "")
        if z2 and z2 not in zone_info:
            zone_info[z2] = {"lat": None, "lon": None}

    # Merge from zcmap if available
    if df_zc is not None and not df_zc.empty and "z" in df_zc.columns:
        pass  # lat/lon already from geojson

    if not zone_info:
        fig = go.Figure()
        fig.update_layout(title="No zone data", template="plotly_white")
        return fig

    zdf = pd.DataFrame.from_dict(zone_info, orient="index").reset_index()
    zdf.columns = ["z", "lat", "lon"]
    zdf = zdf.dropna(subset=["lat", "lon"])

    if zdf.empty:
        fig = go.Figure()
        fig.update_layout(title="No zone coordinates found", template="plotly_white")
        return fig

    fig = go.Figure()

    # Diamond markers with zone name labels
    fig.add_trace(go.Scattergeo(
        lat=zdf["lat"].tolist(),
        lon=zdf["lon"].tolist(),
        mode="markers+text",
        marker={
            "symbol": "diamond",
            "size": 10,
            "color": "#2c5f8a",
            "line": {"width": 1.5, "color": "white"},
        },
        text=zdf["z"].tolist(),
        textposition="top right",
        textfont={"size": 11, "color": "#1a1a2e", "family": "Arial"},
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # Map extent
    lat_min = zdf["lat"].min() - 4
    lat_max = zdf["lat"].max() + 4
    lon_min = zdf["lon"].min() - 4
    lon_max = zdf["lon"].max() + 4

    fig.update_geos(
        scope="world",
        showland=True,       landcolor="#f2f2f2",
        showocean=True,      oceancolor="#ddeeff",
        showcoastlines=True, coastlinecolor="#aaaaaa",
        showcountries=True,  countrycolor="#888888",
        countrywidth=1,
        showframe=False,
        projection_type="natural earth",
        lataxis_range=[lat_min, lat_max],
        lonaxis_range=[lon_min, lon_max],
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    return fig


def _phours_heatmap(df_ph):
    """
    Heatmap: x = quarter (q), y = day type (d),
    colour = total weight (sum of t1..t24) = hours/year represented.
    """
    empty_fig = px.bar(title="No data", template="plotly_white")
    if df_ph is None or df_ph.empty:
        return empty_fig

    t_cols = [c for c in df_ph.columns if c not in ("q", "d")]
    if not t_cols:
        return empty_fig

    df = df_ph.copy()
    df["total_hours"] = df[t_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    if "q" not in df.columns or "d" not in df.columns:
        # Fallback bar chart if structure differs
        df["period"] = df.iloc[:, 0].astype(str) + "-" + df.iloc[:, 1].astype(str)
        return px.bar(df, x="period", y="total_hours",
                      title="Hours per Representative Period",
                      labels={"total_hours": "Hours/year", "period": "Period"},
                      template="plotly_white", text_auto=True)

    # Build pivot for heatmap
    pivot = df.pivot_table(index="d", columns="q", values="total_hours", aggfunc="sum")
    pivot = pivot.fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.round(0).astype(int),
        texttemplate="%{text}h",
        colorbar={"title": "Hours/year"},
        hovertemplate="Quarter: %{x}<br>Day: %{y}<br>Hours: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Hours per Representative Period (q × d)",
        xaxis_title="Quarter",
        yaxis_title="Day type",
        template="plotly_white",
        yaxis={"autorange": "reversed"},
    )
    return fig


# ---------------------------------------------------------------------------
# Main load callback
# ---------------------------------------------------------------------------

@callback(
    Output("zcmap-grid",     "rowData"),    Output("zcmap-grid",     "columnDefs"),
    Output("years-grid",     "rowData"),    Output("years-grid",     "columnDefs"),
    Output("res-y-variant",  "options"),
    Output("phours-grid",    "rowData"),    Output("phours-grid",    "columnDefs"),
    Output("res-ph-variant", "options"),
    Output("zcmap-chart",    "figure"),
    Output("phours-chart",   "figure"),
    Input("res-project",     "value"),
    Input("res-y-variant",   "value"),
    Input("res-ph-variant",  "value"),
    Input("res-reload-btn",  "n_clicks"),
)
def load(folder, y_var, ph_var, _reload=None):
    if _reload:
        dl.clear_input_cache()
    empty = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    empty_fig = px.bar(title="No data", template="plotly_white")

    if not folder:
        return (*empty, *empty, base_opts, *empty, base_opts,
                empty_fig, empty_fig)

    df_zc = dl.load_variant(folder, "zcmap_input",  None)
    df_y  = dl.load_variant(folder, "years",        y_var)
    df_ph = dl.load_variant(folder, "phours",       ph_var)

    # zcmap grid
    if not df_zc.empty:
        zc_cols = [{"field": c, "editable": True, "cellStyle": {}}
                   for c in df_zc.columns]
        zc_rows = df_zc.to_dict("records")
    else:
        zc_cols, zc_rows = [], []

    # Zone network map
    gj = _load_geojson(folder)
    fig_zc = _zone_network_figure(gj, df_zc if not df_zc.empty else None)

    # years grid
    if not df_y.empty:
        y_cols = [{"field": c, "editable": True} for c in df_y.columns]
        y_rows = df_y.to_dict("records")
    else:
        y_cols, y_rows = [], []

    # pHours grid
    if not df_ph.empty:
        ph_cols = [{"field": c, "editable": True, "cellStyle": {}}
                   for c in df_ph.columns]
        ph_rows = df_ph.to_dict("records")
    else:
        ph_cols, ph_rows = [], []

    fig_ph = _phours_heatmap(df_ph if not df_ph.empty else None)

    return (
        zc_rows,  zc_cols,
        y_rows,   y_cols,   variant_options(folder, "years"),
        ph_rows,  ph_cols,  variant_options(folder, "phours"),
        fig_zc, fig_ph,
    )


# ---------------------------------------------------------------------------
# Save callbacks
# ---------------------------------------------------------------------------

@callback(Output("save-zcmap-msg", "children"), Input("save-zcmap-btn", "n_clicks"),
          State("zcmap-grid", "rowData"), State("res-project", "value"),
          prevent_initial_call=True)
def save_zcmap(n, rows, folder):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "zcmap_input", None, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-years-msg", "children"), Input("save-years-btn", "n_clicks"),
          State("years-grid", "rowData"), State("res-project", "value"),
          State("res-y-variant", "value"), prevent_initial_call=True)
def save_years(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "years", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-phours-msg", "children"), Input("save-phours-btn", "n_clicks"),
          State("phours-grid", "rowData"), State("res-project", "value"),
          State("res-ph-variant", "value"), prevent_initial_call=True)
def save_phours(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "phours", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


# ---------------------------------------------------------------------------
# Duplicate callbacks
# ---------------------------------------------------------------------------

@callback(Output("res-y-dup-msg", "children"),
          Output("res-y-variant", "options", allow_duplicate=True),
          Output("res-y-variant", "value",   allow_duplicate=True),
          Input("res-y-dup-btn", "n_clicks"),
          State("res-y-variant", "value"), State("res-y-dup-name", "value"),
          State("res-project", "value"), prevent_initial_call=True)
def dup_years(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "years", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "years"), name
    return "Name exists or error", no_update, no_update


@callback(Output("res-ph-dup-msg", "children"),
          Output("res-ph-variant", "options", allow_duplicate=True),
          Output("res-ph-variant", "value",   allow_duplicate=True),
          Input("res-ph-dup-btn", "n_clicks"),
          State("res-ph-variant", "value"), State("res-ph-dup-name", "value"),
          State("res-project", "value"), prevent_initial_call=True)
def dup_phours(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "phours", v, name)
    if ok:
        return "Created ✓", variant_options(folder, "phours"), name
    return "Name exists or error", no_update, no_update


# ---------------------------------------------------------------------------
# Add / Delete row callbacks
# ---------------------------------------------------------------------------

def _empty_row(rows):
    """Return a new blank row matching the columns of existing rows."""
    if not rows:
        return {}
    return {k: "" for k in rows[0].keys()}


def _delete_selected(rows, selected):
    """Remove selected rows from rows list."""
    if not selected:
        return rows
    selected_tuples = {tuple(sorted(r.items())) for r in selected}
    return [r for r in rows if tuple(sorted(r.items())) not in selected_tuples]


@callback(Output("zcmap-grid", "rowData", allow_duplicate=True),
          Input("add-zcmap-btn", "n_clicks"),
          State("zcmap-grid", "rowData"),
          prevent_initial_call=True)
def add_zcmap_row(n, rows):
    rows = rows or []
    return rows + [_empty_row(rows)]


@callback(Output("zcmap-grid", "rowData", allow_duplicate=True),
          Input("del-zcmap-btn", "n_clicks"),
          State("zcmap-grid", "rowData"),
          State("zcmap-grid", "selectedRows"),
          prevent_initial_call=True)
def del_zcmap_row(n, rows, selected):
    return _delete_selected(rows or [], selected or [])


@callback(Output("years-grid", "rowData", allow_duplicate=True),
          Input("add-years-btn", "n_clicks"),
          State("years-grid", "rowData"),
          prevent_initial_call=True)
def add_years_row(n, rows):
    rows = rows or []
    return rows + [_empty_row(rows) or {"y": ""}]


@callback(Output("years-grid", "rowData", allow_duplicate=True),
          Input("del-years-btn", "n_clicks"),
          State("years-grid", "rowData"),
          State("years-grid", "selectedRows"),
          prevent_initial_call=True)
def del_years_row(n, rows, selected):
    return _delete_selected(rows or [], selected or [])


@callback(Output("phours-grid", "rowData", allow_duplicate=True),
          Input("add-phours-btn", "n_clicks"),
          State("phours-grid", "rowData"),
          prevent_initial_call=True)
def add_phours_row(n, rows):
    rows = rows or []
    return rows + [_empty_row(rows)]


@callback(Output("phours-grid", "rowData", allow_duplicate=True),
          Input("del-phours-btn", "n_clicks"),
          State("phours-grid", "rowData"),
          State("phours-grid", "selectedRows"),
          prevent_initial_call=True)
def del_phours_row(n, rows, selected):
    return _delete_selected(rows or [], selected or [])


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("res-zcmap-open", "n_clicks"),
          State("res-project", "value"),
          prevent_initial_call=True)
def open_zcmap_csv(n, folder):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "zcmap_input", None)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("res-years-open", "n_clicks"),
          State("res-project", "value"),
          State("res-y-variant", "value"),
          prevent_initial_call=True)
def open_years_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "years", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("res-ph-open", "n_clicks"),
          State("res-project", "value"),
          State("res-ph-variant", "value"),
          prevent_initial_call=True)
def open_phours_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "phours", variant)
