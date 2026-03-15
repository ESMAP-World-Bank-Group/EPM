"""Input Editor — Demand (forecast, demand profile, efficiency factor)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT


def _normalise_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the type column to 'type' regardless of what it's called in the CSV (e.g. 'pe')."""
    if df.empty or "type" in df.columns:
        return df
    # Type column = non-z, non-year column whose values contain "peak"/"energy"
    for col in df.columns:
        if str(col).isdigit():
            continue
        if str(col).lower() == "z":
            continue
        return df.rename(columns={col: "type"})
    return df


def _grid(grid_id: str, height: str = "380px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id, rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 90, "sortable": True,
                       "filter": True, "resizable": True},
        style={"height": height}, className="ag-theme-alpine",
    )


def _col_defs_wide(df: pd.DataFrame, id_cols: list) -> list:
    defs = []
    for c in df.columns:
        d = {"field": str(c), "editable": c not in id_cols,
             "cellStyle": {} if c not in id_cols
                          else {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
        if c not in id_cols:
            d["type"] = "numericColumn"
        defs.append(d)
    return defs


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)
    return html.Div([
        dbc.Row([
            dbc.Col(html.H4("Demand Inputs", className="mb-0"), width="auto"),
            dbc.Col(make_open_folder_btn("demand-open-btn"), width="auto",
                    className="d-flex align-items-center"),
        ], className="mb-1 align-items-center"),
        html.P("Edit demand forecast, load profiles and energy efficiency factors.",
               className="text-muted mb-3"),
        html.Div(
            dcc.Dropdown(id="demand-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),
        dbc.Tabs([
            # ── Demand Forecast ───────────────────────────────────────────
            dbc.Tab(label="Demand Forecast", tab_id="tab-forecast", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("d-dem"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-demand-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-demand-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Wide format: rows = zones × types, columns = years.",
                               className="text-muted small mb-1"),
                        _grid("demand-grid"),
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="demand-chart-zone", options=[], value=None,
                                             placeholder="All zones", className="small"),
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
                        html.P("Hourly load profile shape (normalised, 0–1). Read-only.",
                               className="text-muted small mb-2"),
                        _grid("profile-grid", height="400px"),
                    ], width=5),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="profile-zone-filter", options=[], value=None,
                                             placeholder="Select zone…", className="small"),
                            ], width=6),
                        ], className="mb-2"),
                        dcc.Graph(id="profile-chart", config={"displayModeBar": False},
                                  style={"height": "420px"}),
                    ], width=7),
                ]),
            ]),
            # ── Energy Efficiency Factor ──────────────────────────────────
            dbc.Tab(label="Efficiency Factor", tab_id="tab-eff", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        make_variant_bar("d-eff"),
                        dbc.Row([
                            dbc.Col(dbc.Button("Save", id="save-eff-btn",
                                               color="success", size="sm"), width="auto"),
                            dbc.Col(html.Div(id="save-eff-msg"), width="auto"),
                        ], className="mb-2"),
                        html.P("Energy efficiency reduction factor per zone/year.",
                               className="text-muted small mb-1"),
                        _grid("eff-grid"),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id="eff-chart", config={"displayModeBar": False}),
                    ], width=6),
                ]),
            ]),
        ]),
    ])


@callback(
    Output("demand-grid",          "rowData"),  Output("demand-grid",       "columnDefs"),
    Output("d-dem-variant",        "options"),
    Output("profile-grid",         "rowData"),  Output("profile-grid",      "columnDefs"),
    Output("eff-grid",             "rowData"),  Output("eff-grid",          "columnDefs"),
    Output("d-eff-variant",        "options"),
    Output("demand-chart-zone",    "options"),  Output("demand-chart-zone", "value"),
    Output("eff-chart",            "figure"),
    Output("profile-zone-filter",  "options"),
    Input("demand-project",        "value"),
    Input("d-dem-variant",         "value"),
    Input("d-eff-variant",         "value"),
)
def load(folder, dem_var, eff_var):
    empty = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    empty_fig = px.line(title="No data", template="plotly_white")
    if not folder:
        return (*empty, base_opts, *empty, *empty, base_opts,
                [], None, empty_fig, [])

    df_dem     = _normalise_demand_df(dl.load_variant(folder, "demand_forecast", dem_var))
    df_profile = dl.load_variant(folder, "demand_profile",  None)
    df_eff     = dl.load_variant(folder, "efficiency",      eff_var)

    # Demand forecast
    if not df_dem.empty:
        id_cols = [c for c in ["z", "type"] if c in df_dem.columns]
        dem_rows = df_dem.to_dict("records")
        dem_cols = _col_defs_wide(df_dem, id_cols)
        zone_opts = [{"label": z, "value": z} for z in sorted(df_dem["z"].dropna().astype(str).unique())] \
                    if "z" in df_dem.columns else []
        def_zone = zone_opts[0]["value"] if zone_opts else None
    else:
        dem_rows = []; dem_cols = []; zone_opts = []
        def_zone = None

    # Demand profile (read-only)
    if not df_profile.empty:
        prof_cols = [{"field": str(c), "editable": False,
                      "cellStyle": {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
                     for c in df_profile.columns]
        prof_rows = df_profile.head(500).to_dict("records")
        prof_zone_opts = [{"label": z, "value": z}
                          for z in sorted(df_profile["z"].unique())] \
                         if "z" in df_profile.columns else []
    else:
        prof_rows = []; prof_cols = []; prof_zone_opts = []

    # Efficiency factor
    if not df_eff.empty:
        id_cols_eff = [c for c in ["z", "c"] if c in df_eff.columns]
        eff_rows = df_eff.to_dict("records")
        eff_cols = _col_defs_wide(df_eff, id_cols_eff)
        yr_cols = [c for c in df_eff.columns if c not in id_cols_eff and str(c).isdigit()]
        if yr_cols and id_cols_eff:
            df_melt = df_eff.melt(id_vars=id_cols_eff, value_vars=yr_cols,
                                  var_name="year", value_name="factor")
            fig_eff = px.line(df_melt, x="year", y="factor", color=id_cols_eff[0],
                              markers=True, title="Energy Efficiency Factor",
                              labels={"factor": "Reduction factor"},
                              template="plotly_white")
        elif "value" in df_eff.columns:
            fig_eff = px.line(df_eff,
                              x="y" if "y" in df_eff.columns else df_eff.columns[1],
                              y="value", markers=True,
                              title="Energy Efficiency Factor",
                              template="plotly_white")
        else:
            fig_eff = empty_fig
    else:
        eff_rows = []; eff_cols = []; fig_eff = empty_fig

    return (
        dem_rows, dem_cols, variant_options(folder, "demand_forecast"),
        prof_rows, prof_cols,
        eff_rows, eff_cols, variant_options(folder, "efficiency"),
        zone_opts, def_zone,
        fig_eff,
        prof_zone_opts,
    )


@callback(
    Output("demand-preview-chart", "figure"),
    Input("demand-chart-zone",  "value"),
    Input("demand-project",     "value"),
    Input("d-dem-variant",      "value"),
)
def update_demand_chart(zone, folder, variant):
    import plotly.graph_objects as go
    empty_fig = px.line(title="No data", template="plotly_white")
    if not folder:
        return empty_fig
    df = _normalise_demand_df(dl.load_variant(folder, "demand_forecast", variant))
    if df.empty:
        return empty_fig

    # Year columns: digit-only names (int from disk, str from grid — handle both)
    yr_cols = sorted([c for c in df.columns if str(c).isdigit()], key=lambda x: int(str(x)))
    if not yr_cols:
        return empty_fig

    if zone and "z" in df.columns:
        df = df[df["z"].astype(str) == str(zone)]
    if df.empty:
        return empty_fig

    title = f"Demand Forecast{' — ' + zone if zone else ' — All zones'}"

    if "type" not in df.columns:
        # No type column: just sum everything
        df_melt = df[yr_cols].apply(pd.to_numeric, errors="coerce")
        totals = df_melt.sum()
        fig = go.Figure(go.Scatter(x=list(yr_cols), y=totals.values,
                                   mode="lines+markers", name="Demand"))
        fig.update_layout(title=title, xaxis_title="Year",
                          template="plotly_white")
        return fig

    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    types_in_data = sorted(df["type"].dropna().unique())

    # Map type→axis: "peak" → right y2, others → left y
    type_colors = {"energy": "#0d6efd", "peak": "#dc3545"}
    type_labels = {"energy": "Energy (GWh)", "peak": "Peak (MW)"}
    default_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig = go.Figure()
    right_types = [t for t in types_in_data if "peak" in t]
    left_types  = [t for t in types_in_data if "peak" not in t]
    has_right   = bool(right_types)

    # Pre-compute per-type totals so we can set axis ranges
    type_totals = {}
    for t in types_in_data:
        df_t = df[df["type"] == t][yr_cols].apply(pd.to_numeric, errors="coerce")
        type_totals[t] = df_t.sum()   # Series index=year

    for i, t in enumerate(left_types + right_types):
        totals   = type_totals[t]
        years    = [str(c) for c in totals.index]
        vals     = pd.to_numeric(totals.values, errors="coerce")
        color    = type_colors.get(t, default_colors[i % len(default_colors)])
        label    = type_labels.get(t, t.title())
        yaxis    = "y2" if t in right_types else "y"
        is_right = t in right_types
        fig.add_trace(go.Scatter(
            x=years, y=vals,
            mode="lines+markers", name=label,
            yaxis=yaxis,
            line={"color": color, "width": 2.5},
            marker={"symbol": "diamond" if is_right else "circle",
                    "size": 7, "color": color},
        ))

    left_color  = type_colors.get(left_types[0],  "#0d6efd") if left_types  else "#333"
    right_color = type_colors.get(right_types[0], "#dc3545") if right_types else "#dc3545"

    # Set left axis range: 0 → max (fills full chart height)
    left_max = max(
        pd.to_numeric(type_totals[t].values, errors="coerce").max()
        for t in left_types
    ) if left_types else 1
    left_range = [0, left_max * 1.15]

    layout = dict(
        title=title,
        xaxis={"title": None},          # no "Year" label needed
        yaxis={
            "title": {"text": type_labels.get(left_types[0], "Value") if left_types else "Value",
                      "font": {"color": left_color}},
            "tickfont": {"color": left_color},
            "range": left_range,
        },
        template="plotly_white",
        legend={"orientation": "h", "y": -0.12},
    )
    if has_right:
        # Right axis range: 0 → max_peak * 2.5 so curve occupies lower ~40%
        # of chart — visually distinct from the energy curve which fills full height
        right_max = max(
            pd.to_numeric(type_totals[t].values, errors="coerce").max()
            for t in right_types
        )
        layout["yaxis2"] = {
            "title": {"text": type_labels.get(right_types[0], "Peak"),
                      "font": {"color": right_color}},
            "tickfont": {"color": right_color},
            "overlaying": "y", "side": "right",
            "showgrid": False,
            "range": [0, right_max * 2.5],
        }
    fig.update_layout(**layout)
    return fig


@callback(
    Output("profile-chart", "figure"),
    Input("profile-zone-filter", "value"),
    Input("profile-grid",        "rowData"),
)
def update_profile_chart(zone, rows):
    import plotly.graph_objects as go
    if not rows:
        return px.line(title="No profile data", template="plotly_white")
    df = pd.DataFrame(rows)
    if zone and "z" in df.columns:
        df = df[df["z"] == zone]
    if df.empty:
        return px.line(title="No data for selected zone", template="plotly_white")

    # Find t-columns (t1..t24 or t01..t24)
    t_cols = [c for c in df.columns if str(c).startswith("t") and str(c)[1:].isdigit()]
    if not t_cols:
        t_cols = [c for c in df.columns if str(c).startswith("t") and str(c)[1:].lstrip("0").isdigit() and len(str(c)) > 1]
    t_cols = sorted(t_cols, key=lambda x: int(str(x)[1:].lstrip("0") or "0"))
    if not t_cols:
        return px.line(title="No time columns found", template="plotly_white")

    quarters  = sorted(df["q"].unique()) if "q" in df.columns else []
    day_types = sorted(df["d"].unique()) if "d" in df.columns else []
    if not quarters or not day_types:
        return px.line(title="No q/d columns", template="plotly_white")

    # Build x-axis labels: Q1/h01 … Q4/h24
    x_labels = []
    for q in quarters:
        for t in t_cols:
            h = str(t)[1:].zfill(2)
            x_labels.append(f"{q}/h{h}")

    palette = ["#0d6efd", "#dc3545", "#198754", "#fd7e14",
               "#6f42c1", "#20c997", "#e83e8c", "#ffc107"]
    color_map = {d: palette[i % len(palette)] for i, d in enumerate(day_types)}

    fig = go.Figure()
    for d in day_types:
        y_vals = []
        for q in quarters:
            sub = df[(df["q"] == q) & (df["d"] == d)] if "q" in df.columns and "d" in df.columns else df
            if sub.empty:
                y_vals.extend([None] * len(t_cols))
            else:
                row_vals = [pd.to_numeric(sub.iloc[0].get(c, None), errors="coerce") for c in t_cols]
                y_vals.extend(row_vals)
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_vals,
            mode="lines",
            line={"color": color_map.get(d, "#888"), "width": 1.8},
            name=str(d),
        ))

    # Quarter boundary vertical lines
    for i in range(1, len(quarters)):
        fig.add_vline(x=i * len(t_cols) - 0.5, line_dash="dot",
                      line_color="#aaaaaa", opacity=0.6)

    # Quarter tick marks at start of each quarter
    tick_vals = [i * len(t_cols) for i in range(len(quarters))]
    tick_text = list(quarters)

    zone_label = f" — {zone}" if zone else ""
    fig.update_layout(
        title=f"Demand Profile{zone_label}",
        xaxis={
            "title": "Quarter × Hour of day",
            "tickmode": "array",
            "tickvals": [x_labels[v] for v in tick_vals],
            "ticktext": tick_text,
        },
        yaxis={"title": "Normalised load (0–1)"},
        template="plotly_white",
        legend={"title": "Day type"},
    )
    return fig


@callback(Output("save-demand-msg", "children"), Input("save-demand-btn", "n_clicks"),
          State("demand-grid", "rowData"), State("demand-project", "value"),
          State("d-dem-variant", "value"), prevent_initial_call=True)
def save_demand(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "demand_forecast", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-eff-msg", "children"), Input("save-eff-btn", "n_clicks"),
          State("eff-grid", "rowData"), State("demand-project", "value"),
          State("d-eff-variant", "value"), prevent_initial_call=True)
def save_eff(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "efficiency", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("d-dem-dup-msg", "children"),
          Output("d-dem-variant", "options", allow_duplicate=True),
          Output("d-dem-variant", "value", allow_duplicate=True),
          Input("d-dem-dup-btn", "n_clicks"),
          State("d-dem-variant", "value"), State("d-dem-dup-name", "value"),
          State("demand-project", "value"), prevent_initial_call=True)
def dup_dem(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "demand_forecast", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "demand_forecast"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("d-eff-dup-msg", "children"),
          Output("d-eff-variant", "options", allow_duplicate=True),
          Output("d-eff-variant", "value", allow_duplicate=True),
          Input("d-eff-dup-btn", "n_clicks"),
          State("d-eff-variant", "value"), State("d-eff-dup-name", "value"),
          State("demand-project", "value"), prevent_initial_call=True)
def dup_eff(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "efficiency", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "efficiency"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("demand-open-btn", "n_clicks"),
          State("demand-project", "value"), prevent_initial_call=True)
def open_demand_folder(n, folder):
    from dash import no_update
    if not folder: return no_update
    return str(INPUT_ROOT / folder)
