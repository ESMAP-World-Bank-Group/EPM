"""Input Editor — Trade & Transmission."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback, no_update
import pandas as pd
import plotly.express as px
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT


def _grid(grid_id: str, height: str = "360px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id, rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        dashGridOptions={"rowSelection": "multiple"},
        style={"height": height}, className="ag-theme-alpine",
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
            dbc.Col(html.H4("Trade & Transmission", className="mb-0"), width="auto"),
            dbc.Col(
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Reload"],
                           id="tra-reload-btn", color="outline-secondary", size="sm"),
                width="auto", className="ms-auto",
            ),
        ], className="mb-1 align-items-center justify-content-between"),
        html.P("Edit interconnection capacities, candidate new lines and trade prices.",
               className="text-muted mb-3"),
        html.Div(
            dcc.Dropdown(id="trade-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),
        dbc.Tabs([
            dbc.Tab(label="Transfer Limits", tab_id="tab-tl", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("t-tl"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-tl-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-tl-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("Existing interconnection capacity between zones (MW).",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col(_icon_btns("add-tl-btn", "del-tl-btn"),
                                    width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("tl-grid"),
                        html.Div(make_open_folder_btn("trade-tl-open"), className="mt-1 mb-2"),
                    ], width=5),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Year", className="small fw-semibold"),
                                dcc.Dropdown(id="tl-year-filter", options=[], value=None,
                                             clearable=False, className="small",
                                             style={"minWidth": "90px"}),
                            ], width="auto"),
                            dbc.Col([
                                dbc.Label("Quarter", className="small fw-semibold"),
                                dcc.Dropdown(id="tl-quarter-filter",
                                             options=[{"label": f"Q{i}", "value": f"Q{i}"} for i in range(1, 5)],
                                             value="Q1", clearable=False, className="small",
                                             style={"minWidth": "80px"}),
                            ], width="auto"),
                        ], className="mb-2 align-items-end g-2"),
                        dcc.Graph(id="tl-chart", config={"displayModeBar": False},
                                  style={"height": "460px"}),
                    ], width=7),
                ]),
            ]),
            dbc.Tab(label="New Transmission", tab_id="tab-nt", children=[
                make_variant_bar("t-nt"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-nt-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-nt-msg"), width="auto"),
                ]),
                dbc.Row([
                    dbc.Col(html.P("Candidate new transmission lines.",
                                   className="text-muted small mb-1"), width="auto"),
                    dbc.Col(_icon_btns("add-nt-btn", "del-nt-btn"),
                            width="auto", className="ms-auto d-flex align-items-center"),
                ], className="align-items-center mb-1"),
                _grid("nt-grid"),
                html.Div(make_open_folder_btn("trade-nt-open"), className="mt-1 mb-2"),
            ]),
            dbc.Tab(label="Trade Prices", tab_id="tab-tp", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("t-tp"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-tp-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-tp-msg"), width="auto"),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(html.P("Import/export prices with external zones ($/MWh).",
                                           className="text-muted small mb-1"), width="auto"),
                            dbc.Col(_icon_btns("add-tp-btn", "del-tp-btn"),
                                    width="auto", className="ms-auto d-flex align-items-center"),
                        ], className="align-items-center mb-1"),
                        _grid("tp-grid"),
                        html.Div(make_open_folder_btn("trade-tp-open"), className="mt-1 mb-2"),
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("External Zone", className="small fw-semibold"),
                                dcc.Dropdown(id="tp-zext-filter", options=[], value=None,
                                             placeholder="Select zone…", clearable=True,
                                             className="small"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Year", className="small fw-semibold"),
                                dcc.Dropdown(id="tp-year-filter", options=[], value=None,
                                             placeholder="Select year…", clearable=True,
                                             className="small"),
                            ], width=6),
                        ], className="mb-2 g-2"),
                        dcc.Graph(id="tp-chart", config={"displayModeBar": False},
                                  style={"height": "400px"}),
                    ], width=6),
                ]),
            ]),
            dbc.Tab(label="External Transfer Limits", tab_id="tab-etl", children=[
                make_variant_bar("t-etl"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-etl-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-etl-msg"), width="auto"),
                ]),
                dbc.Row([
                    dbc.Col(html.P("Max capacity for imports/exports with external zones (MW).",
                                   className="text-muted small mb-1"), width="auto"),
                    dbc.Col(_icon_btns("add-etl-btn", "del-etl-btn"),
                            width="auto", className="ms-auto d-flex align-items-center"),
                ], className="align-items-center mb-1"),
                _grid("etl-grid"),
                html.Div(make_open_folder_btn("trade-etl-open"), className="mt-1 mb-2"),
            ]),
        ]),
    ])


@callback(
    Output("tl-grid",       "rowData"),  Output("tl-grid",  "columnDefs"),
    Output("t-tl-variant",  "options"),
    Output("nt-grid",       "rowData"),  Output("nt-grid",  "columnDefs"),
    Output("t-nt-variant",  "options"),
    Output("tp-grid",       "rowData"),  Output("tp-grid",  "columnDefs"),
    Output("t-tp-variant",  "options"),
    Output("etl-grid",      "rowData"),  Output("etl-grid", "columnDefs"),
    Output("t-etl-variant", "options"),
    Output("tl-chart",      "figure", allow_duplicate=True),
    Output("tp-chart",      "figure"),
    Output("tl-year-filter", "options"),
    Output("tp-zext-filter", "options"),
    Output("tp-year-filter", "options"),
    Input("trade-project",  "value"),
    Input("t-tl-variant",   "value"),
    Input("t-nt-variant",   "value"),
    Input("t-tp-variant",   "value"),
    Input("t-etl-variant",  "value"),
    Input("tra-reload-btn", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)
def load(folder, tl_var, nt_var, tp_var, etl_var, _reload=None):
    if _reload:
        dl.clear_input_cache()
    empty = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    empty_fig = px.bar(title="No data", template="plotly_white")
    if not folder:
        return (*empty, base_opts, *empty, base_opts, *empty, base_opts,
                *empty, base_opts, empty_fig, empty_fig, [], [], [])

    df_tl  = dl.load_variant(folder, "transfer_limit",   tl_var)
    df_nt  = dl.load_variant(folder, "new_transmission", nt_var)
    df_tp  = dl.load_variant(folder, "trade_price",      tp_var)
    df_etl = dl.load_variant(folder, "ext_transfer",     etl_var)

    # Rename unnamed columns to standard names (trade price CSV often has blank headers)
    if not df_tp.empty:
        col_renames = {}
        unnamed_cols = [c for c in df_tp.columns if str(c).startswith("Unnamed:")]
        std_names = ["zext", "q", "d", "y"]
        for i, col in enumerate(unnamed_cols[:4]):
            col_renames[col] = std_names[i]
        if col_renames:
            df_tp = df_tp.rename(columns=col_renames)

    def rc(df, ro): return (df.to_dict("records"), _col_defs(df, ro)) if not df.empty else ([], [])

    # Year filter options for TL
    yr_cols_tl = [c for c in df_tl.columns if str(c).isdigit()] if not df_tl.empty else []
    tl_year_opts = [{"label": str(y), "value": str(y)} for y in sorted(yr_cols_tl)]

    # tl-chart is handled by the reactive update_tl_map callback; return empty here
    fig_tl = empty_fig

    fig_tp = px.line(title="Select an external zone", template="plotly_white")

    tp_zext_opts = [{"label": z, "value": z} for z in sorted(df_tp["zext"].unique())] \
                   if not df_tp.empty and "zext" in df_tp.columns else []
    tp_year_opts = [{"label": str(y), "value": str(y)} for y in sorted(df_tp["y"].unique())] \
                   if not df_tp.empty and "y" in df_tp.columns else []

    return (
        *rc(df_tl,  ["From", "To", "z", "z2"]),   variant_options(folder, "transfer_limit"),
        *rc(df_nt,  ["z", "z2"]),                  variant_options(folder, "new_transmission"),
        *rc(df_tp,  ["z", "zext", "q", "d", "y"]), variant_options(folder, "trade_price"),
        *rc(df_etl, ["z", "zext"]),                variant_options(folder, "ext_transfer"),
        fig_tl, fig_tp, tl_year_opts, tp_zext_opts, tp_year_opts,
    )


@callback(
    Output("tl-chart",         "figure"),
    Output("tl-year-filter",   "value"),
    Input("tl-grid",           "rowData"),
    Input("tl-year-filter",    "value"),
    Input("tl-quarter-filter", "value"),
    Input("trade-project",     "value"),
)
def update_tl_map(rows, year, quarter, folder):
    import json
    import plotly.graph_objects as go

    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data", template="plotly_white")

    if not rows:
        return empty_fig, year

    df = pd.DataFrame(rows)

    # Auto-select first year if none selected
    yr_cols = [c for c in df.columns if str(c).isdigit()]
    if not yr_cols:
        return empty_fig, year
    if not year or year not in yr_cols:
        year = yr_cols[0]

    # Filter by quarter
    if quarter and "q" in df.columns:
        df_q = df[df["q"] == quarter]
    else:
        df_q = df

    if df_q.empty or year not in df_q.columns:
        return empty_fig, year

    df_q = df_q.copy()
    df_q["capacity"] = pd.to_numeric(df_q[year], errors="coerce").fillna(0)

    # Load geojson for zone centroids
    zone_coords = {}
    if folder:
        gj_path = INPUT_ROOT / folder / "linestring_countries.geojson"
        if gj_path.exists():
            try:
                with open(gj_path, encoding="utf-8") as f:
                    gj = json.load(f)
                for feat in gj["features"]:
                    props = feat["properties"]
                    z = props.get("z", "")
                    if z and z not in zone_coords:
                        zone_coords[z] = {
                            "lat": props.get("country_ini_lat"),
                            "lon": props.get("country_ini_lon"),
                        }
            except Exception:
                pass

    # Determine From/To column names
    from_col = "From" if "From" in df_q.columns else (df_q.columns[0] if len(df_q.columns) > 0 else None)
    to_col   = "To"   if "To"   in df_q.columns else (df_q.columns[1] if len(df_q.columns) > 1 else None)

    if not zone_coords or not from_col or not to_col:
        # Fallback bar chart
        df_q["corridor"] = df_q[from_col].astype(str) + " \u2192 " + df_q[to_col].astype(str)
        fig = px.bar(df_q, x="corridor", y="capacity",
                     title=f"Transfer Limits {year} {quarter or ''} (MW)",
                     labels={"capacity": "MW", "corridor": ""},
                     template="plotly_white")
        fig.update_xaxes(tickangle=30)
        return fig, year

    fig = go.Figure()

    # Collect label positions separately so they can be drawn last (on top)
    labeled_pairs = set()
    label_lats, label_lons, label_texts = [], [], []

    # Draw connection lines first
    for _, row in df_q.iterrows():
        frm = str(row[from_col])
        to  = str(row[to_col])
        cap = row["capacity"]
        if frm not in zone_coords or to not in zone_coords:
            continue
        lat0, lon0 = zone_coords[frm]["lat"], zone_coords[frm]["lon"]
        lat1, lon1 = zone_coords[to]["lat"], zone_coords[to]["lon"]
        if any(v is None for v in [lat0, lon0, lat1, lon1]):
            continue
        mid_lat = (lat0 + lat1) / 2
        mid_lon = (lon0 + lon1) / 2
        width = max(1.5, min(8, cap / 150)) if cap > 0 else 1
        line_color = "#4682b4" if cap > 0 else "#cccccc"
        fig.add_trace(go.Scattergeo(
            lat=[lat0, lat1], lon=[lon0, lon1],
            mode="lines",
            line={"width": width, "color": line_color},
            hovertemplate=f"{frm} \u2192 {to}: {cap:.0f} MW<extra></extra>",
            showlegend=False,
        ))
        # Collect label (one per undirected pair, offset above midpoint)
        pair_key = frozenset([frm, to])
        if cap > 0 and pair_key not in labeled_pairs:
            labeled_pairs.add(pair_key)
            label_lats.append(mid_lat + 0.9)
            label_lons.append(mid_lon)
            label_texts.append(f"<b>{cap:.0f}</b>")

    # Draw zone points
    zones_shown = set()
    for _, row in df_q.iterrows():
        for z in [str(row[from_col]), str(row[to_col])]:
            if z in zone_coords and z not in zones_shown:
                zones_shown.add(z)

    zone_lats  = [zone_coords[z]["lat"]  for z in zones_shown if zone_coords[z]["lat"]  is not None]
    zone_lons  = [zone_coords[z]["lon"]  for z in zones_shown if zone_coords[z]["lon"]  is not None]
    zone_names = [z for z in zones_shown if zone_coords[z]["lat"] is not None]

    if zone_lats:
        fig.add_trace(go.Scattergeo(
            lat=zone_lats, lon=zone_lons,
            mode="markers+text",
            marker={"size": 8, "color": "#444", "line": {"width": 1, "color": "white"}},
            text=zone_names, textposition="top right",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    # Draw capacity labels last so they appear above all lines and markers
    if label_lats:
        fig.add_trace(go.Scattergeo(
            lat=label_lats, lon=label_lons,
            mode="text",
            text=label_texts,
            textfont={"size": 10, "color": "#1a3a5c"},
            hoverinfo="skip",
            showlegend=False,
        ))

    if zone_lats:
        lat_min = min(zone_lats) - 3
        lat_max = max(zone_lats) + 3
        lon_min = min(zone_lons) - 3
        lon_max = max(zone_lons) + 3
    else:
        lat_min, lat_max, lon_min, lon_max = -40, 40, -20, 60

    fig.update_geos(
        showland=True, landcolor="#f0f0f0",
        showocean=True, oceancolor="#d6eaf8",
        showcoastlines=True, coastlinecolor="#aaaaaa",
        showcountries=True, countrycolor="#cccccc",
        showframe=False,
        projection_type="natural earth",
        lataxis_range=[lat_min, lat_max],
        lonaxis_range=[lon_min, lon_max],
    )
    fig.update_layout(
        title=f"Transfer Limits {year} {quarter or ''} (MW) \u2014 line width \u221d capacity",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    return fig, year


@callback(
    Output("tp-chart",        "figure", allow_duplicate=True),
    Output("tp-zext-filter",  "options", allow_duplicate=True),
    Output("tp-year-filter",  "options", allow_duplicate=True),
    Input("tp-zext-filter",   "value"),
    Input("tp-year-filter",   "value"),
    Input("tp-grid",          "rowData"),
    prevent_initial_call=True,
)
def update_tp_chart(zext, year, rows):
    import plotly.graph_objects as go
    empty_fig = px.line(title="Select an external zone", template="plotly_white")
    if not rows:
        return empty_fig, no_update, no_update

    df = pd.DataFrame(rows)

    # Rename unnamed columns if needed
    col_renames = {}
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    for i, col in enumerate(unnamed[:4]):
        col_renames[col] = ["zext", "q", "d", "y"][i]
    if col_renames:
        df = df.rename(columns=col_renames)

    # Recompute dropdown options from full grid data
    zext_opts = [{"label": z, "value": z} for z in sorted(df["zext"].unique())] \
                if "zext" in df.columns else []
    year_opts = [{"label": str(y), "value": str(y)} for y in sorted(df["y"].unique())] \
                if "y" in df.columns else []

    if not zext or "zext" not in df.columns:
        return empty_fig, zext_opts, year_opts

    df = df[df["zext"] == zext]
    if year and "y" in df.columns:
        df = df[df["y"].astype(str) == str(year)]
    if df.empty:
        return px.line(title="No data for selection", template="plotly_white"), zext_opts, year_opts

    # Find t-columns
    t_cols = sorted(
        [c for c in df.columns if str(c).startswith("t") and len(str(c)) > 1
         and str(c)[1:].lstrip("0").isdigit()],
        key=lambda x: int(str(x)[1:].lstrip("0") or "0"),
    )
    if not t_cols:
        return px.line(title="No hourly columns", template="plotly_white"), zext_opts, year_opts

    quarters  = sorted(df["q"].unique()) if "q" in df.columns else []
    day_types = sorted(df["d"].unique()) if "d" in df.columns else []
    if not quarters or not day_types:
        return px.line(title="No q/d columns", template="plotly_white"), zext_opts, year_opts

    # Build x-axis labels: Q1/h01 … Q4/h24
    x_labels = [f"{q}/h{str(t)[1:].zfill(2)}" for q in quarters for t in t_cols]

    palette = ["#0d6efd", "#dc3545", "#198754", "#fd7e14",
               "#6f42c1", "#20c997", "#e83e8c", "#ffc107"]
    color_map = {d: palette[i % len(palette)] for i, d in enumerate(day_types)}

    fig = go.Figure()
    for d in day_types:
        y_vals = []
        for q in quarters:
            sub = df[(df["q"] == q) & (df["d"] == d)]
            if sub.empty:
                y_vals.extend([None] * len(t_cols))
            else:
                row_vals = [pd.to_numeric(sub.iloc[0].get(c, None), errors="coerce")
                            for c in t_cols]
                y_vals.extend(row_vals)
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_vals,
            mode="lines",
            line={"color": color_map.get(d, "#888"), "width": 1.8},
            name=str(d),
        ))

    for i in range(1, len(quarters)):
        fig.add_vline(x=i * len(t_cols) - 0.5, line_dash="dot",
                      line_color="#aaaaaa", opacity=0.6)

    tick_vals = [i * len(t_cols) for i in range(len(quarters))]
    year_label = f" ({year})" if year else ""
    fig.update_layout(
        title=f"Trade Prices — {zext}{year_label} ($/MWh)",
        xaxis={
            "title": "Quarter × Hour of day",
            "tickmode": "array",
            "tickvals": [x_labels[v] for v in tick_vals],
            "ticktext": list(quarters),
        },
        yaxis={"title": "$/MWh"},
        template="plotly_white",
        legend={"title": "Day type"},
    )
    return fig, zext_opts, year_opts


@callback(Output("save-tl-msg", "children"), Input("save-tl-btn", "n_clicks"),
          State("tl-grid", "rowData"), State("trade-project", "value"),
          State("t-tl-variant", "value"), prevent_initial_call=True)
def save_tl(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved \u2713", color="success") \
        if dl.save_variant(folder, "transfer_limit", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-nt-msg", "children"), Input("save-nt-btn", "n_clicks"),
          State("nt-grid", "rowData"), State("trade-project", "value"),
          State("t-nt-variant", "value"), prevent_initial_call=True)
def save_nt(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved \u2713", color="success") \
        if dl.save_variant(folder, "new_transmission", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-tp-msg", "children"), Input("save-tp-btn", "n_clicks"),
          State("tp-grid", "rowData"), State("trade-project", "value"),
          State("t-tp-variant", "value"), prevent_initial_call=True)
def save_tp(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved \u2713", color="success") \
        if dl.save_variant(folder, "trade_price", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-etl-msg", "children"), Input("save-etl-btn", "n_clicks"),
          State("etl-grid", "rowData"), State("trade-project", "value"),
          State("t-etl-variant", "value"), prevent_initial_call=True)
def save_etl(n, rows, folder, v):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved \u2713", color="success") \
        if dl.save_variant(folder, "ext_transfer", v, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("t-tl-dup-msg", "children"),
          Output("t-tl-variant", "options", allow_duplicate=True),
          Output("t-tl-variant", "value", allow_duplicate=True),
          Input("t-tl-dup-btn", "n_clicks"),
          State("t-tl-variant", "value"), State("t-tl-dup-name", "value"),
          State("trade-project", "value"), prevent_initial_call=True)
def dup_tl(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "transfer_limit", v, name)
    if ok:
        return "Created \u2713", variant_options(folder, "transfer_limit"), name
    return "Name exists or error", no_update, no_update


@callback(Output("t-nt-dup-msg", "children"),
          Output("t-nt-variant", "options", allow_duplicate=True),
          Output("t-nt-variant", "value", allow_duplicate=True),
          Input("t-nt-dup-btn", "n_clicks"),
          State("t-nt-variant", "value"), State("t-nt-dup-name", "value"),
          State("trade-project", "value"), prevent_initial_call=True)
def dup_nt(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "new_transmission", v, name)
    if ok:
        return "Created \u2713", variant_options(folder, "new_transmission"), name
    return "Name exists or error", no_update, no_update


@callback(Output("t-tp-dup-msg", "children"),
          Output("t-tp-variant", "options", allow_duplicate=True),
          Output("t-tp-variant", "value", allow_duplicate=True),
          Input("t-tp-dup-btn", "n_clicks"),
          State("t-tp-variant", "value"), State("t-tp-dup-name", "value"),
          State("trade-project", "value"), prevent_initial_call=True)
def dup_tp(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "trade_price", v, name)
    if ok:
        return "Created \u2713", variant_options(folder, "trade_price"), name
    return "Name exists or error", no_update, no_update


@callback(Output("t-etl-dup-msg", "children"),
          Output("t-etl-variant", "options", allow_duplicate=True),
          Output("t-etl-variant", "value", allow_duplicate=True),
          Input("t-etl-dup-btn", "n_clicks"),
          State("t-etl-variant", "value"), State("t-etl-dup-name", "value"),
          State("trade-project", "value"), prevent_initial_call=True)
def dup_etl(n, v, name, folder):
    from dash import no_update
    if not name or not folder: return "Enter a name", no_update, no_update
    name = name.strip()
    ok = dl.duplicate_variant(folder, "ext_transfer", v, name)
    if ok:
        return "Created \u2713", variant_options(folder, "ext_transfer"), name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("trade-tl-open", "n_clicks"),
          State("trade-project", "value"),
          State("t-tl-variant", "value"),
          prevent_initial_call=True)
def open_tl_csv(n, folder, variant):
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "transfer_limit", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("trade-nt-open", "n_clicks"),
          State("trade-project", "value"),
          State("t-nt-variant", "value"),
          prevent_initial_call=True)
def open_nt_csv(n, folder, variant):
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "new_transmission", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("trade-tp-open", "n_clicks"),
          State("trade-project", "value"),
          State("t-tp-variant", "value"),
          prevent_initial_call=True)
def open_tp_csv(n, folder, variant):
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "trade_price", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("trade-etl-open", "n_clicks"),
          State("trade-project", "value"),
          State("t-etl-variant", "value"),
          prevent_initial_call=True)
def open_etl_csv(n, folder, variant):
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "ext_transfer", variant)


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
    ("tl-grid",  "add-tl-btn",  "del-tl-btn"),
    ("nt-grid",  "add-nt-btn",  "del-nt-btn"),
    ("tp-grid",  "add-tp-btn",  "del-tp-btn"),
    ("etl-grid", "add-etl-btn", "del-etl-btn"),
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
