"""Input Editor — Supply (generators, fuel prices, CAPEX, availability, storage, VRE)."""

import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT, RESOURCES
import pandas as _pd_colors


def _load_tech_colors() -> dict:
    """Read colors.csv fresh each call so UI reflects any user edits."""
    path = RESOURCES / "colors.csv"
    if not path.exists():
        return {}
    try:
        df = _pd_colors.read_csv(path, comment="#")
        df.columns = [c.strip() for c in df.columns]
        return {str(row["Processing"]).strip(): str(row["Color"]).strip()
                for _, row in df.iterrows()
                if str(row.get("Processing", "")).strip()
                and not str(row.get("Processing", "")).startswith("#")}
    except Exception:
        return {}


def _tech_color(tech: str, color_map: dict) -> str:
    """Case-insensitive substring match: 'Solar_PV' → matches 'Solar' → orange."""
    t_lower = tech.lower()
    for key, color in color_map.items():
        if key.lower() in t_lower or t_lower in key.lower():
            return color
    return "#A9A9A9"  # grey fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid(grid_id: str, height: str = "420px") -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        rowData=[], columnDefs=[],
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True,
                       "filter": True, "resizable": True},
        dashGridOptions={"animateRows": True},
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


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        dbc.Row([
            dbc.Col(html.H4("Supply Inputs", className="mb-0"), width="auto"),
        ], className="mb-1 align-items-center"),
        html.P("Edit generator data, fuel prices, CAPEX, availability and storage.",
               className="text-muted mb-3"),

        html.Div(
            dcc.Dropdown(id="supply-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),

        dbc.Tabs(id="supply-tabs", active_tab="tab-gen", children=[

            # ── Generator Data ────────────────────────────────────────────
            dbc.Tab(label="Generator Data", tab_id="tab-gen", children=[
                make_variant_bar("s-gen"),
                dbc.Row(className="mb-2 align-items-center", children=[
                    dbc.Col(dbc.Button("Save", id="save-gen-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-gen-msg"), width="auto"),
                ]),
                html.P("Existing and candidate generator plants. "
                       "Status: 1=existing, 2=candidate, 3=must-build.",
                       className="text-muted small"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col([
                        _grid("gen-grid"),
                        html.Div(make_open_folder_btn("sup-gen-open"), className="mt-1 mb-2"),
                    ], width=7),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="gen-zone-filter", options=[], value=None,
                                             placeholder="All zones", className="small"),
                            ], width=10),
                        ], className="mb-3"),
                        dcc.Graph(id="gen-pie-chart", config={"displayModeBar": False},
                                  style={"height": "340px"}),
                    ], width=5),
                ]),
                dcc.Graph(id="gen-chart", config={"displayModeBar": False},
                          style={"height": "380px"}),
            ]),

            # ── Storage ───────────────────────────────────────────────────
            dbc.Tab(label="Storage", tab_id="tab-storage", children=[
                make_variant_bar("s-sto"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-storage-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-storage-msg"), width="auto"),
                ]),
                html.P("Battery and pumped-hydro storage parameters.",
                       className="text-muted small"),
                _grid("storage-grid"),
                html.Div(make_open_folder_btn("sup-sto-open"), className="mt-1 mb-2"),
            ]),

            # ── Availability ──────────────────────────────────────────────
            dbc.Tab(label="Availability", tab_id="tab-avail", children=[
                make_variant_bar("s-avail"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-avail-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-avail-msg"), width="auto"),
                ]),
                html.P("Plant availability factor by season/quarter (0–1).",
                       className="text-muted small"),
                _grid("avail-grid", height="420px"),
                html.Div(make_open_folder_btn("sup-avail-open"), className="mt-1 mb-2"),
            ]),

            # ── Fuel Prices ───────────────────────────────────────────────
            dbc.Tab(label="Fuel Prices", tab_id="tab-fuel", children=[
                make_variant_bar("s-fuel"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-fuel-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-fuel-msg"), width="auto"),
                ]),
                dbc.Row([
                    dbc.Col([
                        _grid("fuel-grid", height="380px"),
                        html.Div(make_open_folder_btn("sup-fuel-open"), className="mt-1 mb-2"),
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="fuel-zone-filter", options=[], value=None,
                                             placeholder="All zones", className="small"),
                            ], width=6),
                        ], className="mb-2"),
                        dcc.Graph(id="fuel-chart", config={"displayModeBar": False}),
                    ], width=6),
                ]),
            ]),

            # ── CAPEX Trajectories ────────────────────────────────────────
            dbc.Tab(label="CAPEX Trajectories", tab_id="tab-capex", children=[
                make_variant_bar("s-capex"),
                dbc.Row(className="mb-2", children=[
                    dbc.Col(dbc.Button("Save", id="save-capex-btn",
                                       color="success", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="save-capex-msg"), width="auto"),
                ]),
                dbc.Row([
                    dbc.Col([
                        _grid("capex-grid", height="380px"),
                        html.Div(make_open_folder_btn("sup-capex-open"), className="mt-1 mb-2"),
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="capex-zone-filter", options=[], value=None,
                                             placeholder="All zones", className="small"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Technology", className="small"),
                                dcc.Dropdown(id="capex-tech-filter", options=[], value=None,
                                             placeholder="All techs", className="small"),
                            ], width=6),
                        ], className="mb-2 g-2"),
                        dcc.Graph(id="capex-chart", config={"displayModeBar": False}),
                    ], width=6),
                ]),
            ]),

            # ── VRE Profiles (view-only) ──────────────────────────────────
            dbc.Tab(label="VRE Profiles (view)", tab_id="tab-vre", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col([
                        html.P("VRE generation profiles — read-only (first 500 rows shown).",
                               className="text-muted small mb-2"),
                        _grid("vre-grid", height="380px"),
                    ], width=5),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Zone", className="small"),
                                dcc.Dropdown(id="vre-zone-filter", options=[], value=None,
                                             placeholder="Select zone…", className="small"),
                            ], width=5),
                            dbc.Col([
                                dbc.Label("Technology", className="small"),
                                dcc.Dropdown(id="vre-tech-filter", options=[], value=None,
                                             placeholder="Select tech…", className="small"),
                            ], width=5),
                        ], className="mb-2 g-2"),
                        dcc.Graph(id="vre-chart", config={"displayModeBar": False},
                                  style={"height": "400px"}),
                    ], width=7),
                ]),
            ]),
        ]),
    ])


# ---------------------------------------------------------------------------
# Load all grids
# ---------------------------------------------------------------------------

@callback(
    Output("gen-grid",          "rowData"),    Output("gen-grid",     "columnDefs"),
    Output("s-gen-variant",     "options"),
    Output("storage-grid",      "rowData"),    Output("storage-grid", "columnDefs"),
    Output("s-sto-variant",     "options"),
    Output("avail-grid",        "rowData"),    Output("avail-grid",   "columnDefs"),
    Output("s-avail-variant",   "options"),
    Output("fuel-grid",         "rowData"),    Output("fuel-grid",    "columnDefs"),
    Output("s-fuel-variant",    "options"),
    Output("fuel-chart",        "figure"),
    Output("capex-grid",        "rowData"),    Output("capex-grid",   "columnDefs"),
    Output("s-capex-variant",   "options"),
    Output("capex-chart",       "figure"),
    Output("vre-grid",          "rowData"),    Output("vre-grid",     "columnDefs"),
    Output("vre-zone-filter",   "options"),
    Output("vre-tech-filter",   "options"),
    Output("capex-zone-filter", "options"),
    Output("capex-tech-filter", "options"),
    Output("gen-zone-filter",   "options"),
    Output("fuel-zone-filter",  "options"),
    Input("supply-project",     "value"),
    Input("s-gen-variant",      "value"),
    Input("s-sto-variant",      "value"),
    Input("s-avail-variant",    "value"),
    Input("s-fuel-variant",     "value"),
    Input("s-capex-variant",    "value"),
)
def load_all(folder, gen_var, sto_var, avail_var, fuel_var, capex_var):
    empty     = ([], [])
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    empty_fig = px.line(title="No data", template="plotly_white")

    if not folder:
        return (*empty, base_opts, *empty, base_opts,
                *empty, base_opts,
                *empty, base_opts, empty_fig,
                *empty, base_opts, empty_fig,
                *empty, [], [],
                [], [], [],
                [])

    def rc(df, ro): return (df.to_dict("records"), _col_defs(df, ro)) if not df.empty else ([], [])
    def opts(key):  return variant_options(folder, key)

    df_gen     = dl.load_variant(folder, "gen_data",      gen_var)
    df_storage = dl.load_variant(folder, "storage_data",  sto_var)
    df_avail   = dl.load_variant(folder, "availability",  avail_var)
    df_fuel    = dl.load_variant(folder, "fuel_price",    fuel_var)
    # Rename unnamed columns (fuel price CSV has blank headers for first cols)
    if not df_fuel.empty:
        col_renames = {}
        unnamed = [c for c in df_fuel.columns if str(c).startswith("Unnamed:")]
        std_names = ["z", "f"]
        for i, col in enumerate(unnamed[:2]):
            col_renames[col] = std_names[i]
        if col_renames:
            df_fuel = df_fuel.rename(columns=col_renames)
        # Drop trailing unnamed columns
        df_fuel = df_fuel[[c for c in df_fuel.columns if not str(c).startswith("Unnamed:")]]
    df_capex   = dl.load_variant(folder, "capex",         capex_var)
    if df_capex.empty:
        df_capex = dl.load_variant(folder, "capex_default", None)

    # Try loading VRE profile
    try:
        from config import INPUT_ROOT
        vre_path = INPUT_ROOT / folder / "supply" / "pVREProfile.csv"
        df_vre = pd.read_csv(vre_path) if vre_path.exists() else pd.DataFrame()
    except Exception:
        df_vre = pd.DataFrame()

    # Fuel price chart (wide format: cols = years, rows = zone×fuel)
    fig_fuel = empty_fig
    if not df_fuel.empty:
        id_cols_fuel = [c for c in ["z", "f"] if c in df_fuel.columns]
        yr_cols_fuel = [c for c in df_fuel.columns
                        if c not in id_cols_fuel and str(c).isdigit()]
        if yr_cols_fuel and id_cols_fuel:
            df_fuel_melt = df_fuel.melt(id_vars=id_cols_fuel, value_vars=yr_cols_fuel,
                                        var_name="year", value_name="price")
            df_fuel_melt["price"] = pd.to_numeric(df_fuel_melt["price"], errors="coerce")
            df_fuel_melt = df_fuel_melt.dropna(subset=["price"])
            color_col = "f" if "f" in df_fuel_melt.columns else None
            fig_fuel = px.line(df_fuel_melt, x="year", y="price", color=color_col,
                               markers=True, title="Fuel Prices ($/mmBTU)",
                               labels={"price": "$/mmBTU", "year": "Year"},
                               template="plotly_white")
            fig_fuel.update_xaxes(type="category")
        elif "value" in df_fuel.columns:
            color = "f" if "f" in df_fuel.columns else None
            x     = "y" if "y" in df_fuel.columns else df_fuel.columns[0]
            fig_fuel = px.line(df_fuel, x=x, y="value", color=color,
                               markers=True, title="Fuel Prices ($/mmBTU)",
                               labels={"value": "$/mmBTU"}, template="plotly_white")
            fig_fuel.update_xaxes(type="category")

    # CAPEX chart
    fig_capex = empty_fig
    if not df_capex.empty:
        id_cols = [c for c in df_capex.columns if not str(c).isdigit()]
        yr_cols = [c for c in df_capex.columns if str(c).isdigit()]
        if yr_cols and id_cols:
            df_melt = df_capex.melt(id_vars=id_cols, value_vars=yr_cols,
                                    var_name="year", value_name="capex")
            df_melt["capex"] = pd.to_numeric(df_melt["capex"], errors="coerce")
            fig_capex = px.line(df_melt, x="year", y="capex", color=id_cols[0],
                                markers=True, title="CAPEX Trajectories ($/kW)",
                                labels={"capex": "$/kW"}, template="plotly_white")

    # VRE (read-only)
    vre_rows = df_vre.head(500).to_dict("records") if not df_vre.empty else []
    vre_cols = [{"field": str(c), "editable": False,
                 "cellStyle": {"backgroundColor": "#f8f9fa", "color": "#6c757d"}}
                for c in df_vre.columns] if not df_vre.empty else []

    vre_zone_col = "zone" if "zone" in df_vre.columns else ("z" if "z" in df_vre.columns else None)
    vre_zones = [{"label": z, "value": z} for z in sorted(df_vre[vre_zone_col].unique())] \
                if not df_vre.empty and vre_zone_col else []
    vre_techs = [{"label": t, "value": t} for t in sorted(df_vre["tech"].unique())] \
                if not df_vre.empty and "tech" in df_vre.columns else []

    capex_zones = [{"label": z, "value": z} for z in sorted(df_capex["z"].unique())] \
                  if not df_capex.empty and "z" in df_capex.columns else []
    capex_techs = [{"label": t, "value": t} for t in sorted(df_capex["tech"].unique())] \
                  if not df_capex.empty and "tech" in df_capex.columns else []

    gen_zones = [{"label": z, "value": z} for z in sorted(df_gen["z"].unique())] \
                if not df_gen.empty and "z" in df_gen.columns else []

    fuel_zones = [{"label": z, "value": z} for z in sorted(df_fuel["z"].unique())] \
                 if not df_fuel.empty and "z" in df_fuel.columns else []

    return (
        *rc(df_gen,     ["z", "g"]),   opts("gen_data"),
        *rc(df_storage, ["g", "z"]),   opts("storage_data"),
        *rc(df_avail,   ["g"]),        opts("availability"),
        *rc(df_fuel,    ["z", "f"]),   opts("fuel_price"),   fig_fuel,
        *rc(df_capex,   ["f"]),        opts("capex"),        fig_capex,
        vre_rows, vre_cols, vre_zones, vre_techs,
        capex_zones, capex_techs,
        gen_zones,
        fuel_zones,
    )


@callback(
    Output("vre-chart", "figure"),
    Input("vre-zone-filter", "value"),
    Input("vre-tech-filter", "value"),
    Input("vre-grid",        "rowData"),
)
def update_vre_chart(zone, tech, rows):
    if not rows:
        return px.line(title="No VRE data", template="plotly_white")
    df = pd.DataFrame(rows)
    zone_col = "zone" if "zone" in df.columns else ("z" if "z" in df.columns else None)
    if zone and zone_col:
        df = df[df[zone_col] == zone]
    if tech and "tech" in df.columns:
        df = df[df["tech"] == tech]
    if df.empty:
        return px.line(title="No data for selection", template="plotly_white")

    # Find t-columns (t1..t24 or t01..t24)
    t_cols = sorted(
        [c for c in df.columns if str(c).startswith("t") and len(str(c)) > 1
         and str(c)[1:].lstrip("0").isdigit()],
        key=lambda x: int(str(x)[1:].lstrip("0") or "0"),
    )
    if not t_cols:
        return px.line(title="No time columns", template="plotly_white")

    quarters  = sorted(df["q"].unique()) if "q" in df.columns else []
    day_types = sorted(df["d"].unique()) if "d" in df.columns else []
    if not quarters or not day_types:
        return px.line(title="No q/d columns", template="plotly_white")

    # Build x-axis labels: Q1/h01 … Q4/h24
    x_labels = [f"{q}/h{str(t)[1:].zfill(2)}" for q in quarters for t in t_cols]

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
                row_vals = [pd.to_numeric(sub.iloc[0].get(c, None), errors="coerce")
                            for c in t_cols]
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

    tick_vals = [i * len(t_cols) for i in range(len(quarters))]
    label = f"{tech or 'VRE'}{' — ' + zone if zone else ''}"
    fig.update_layout(
        title=f"VRE Generation Profile: {label}",
        xaxis={
            "title": "Quarter × Hour of day",
            "tickmode": "array",
            "tickvals": [x_labels[v] for v in tick_vals],
            "ticktext": list(quarters),
        },
        yaxis={"title": "Capacity Factor (0–1)", "range": [0, 1.05]},
        template="plotly_white",
        legend={"title": "Day type"},
    )
    return fig


@callback(
    Output("capex-chart",       "figure", allow_duplicate=True),
    Input("capex-zone-filter",  "value"),
    Input("capex-tech-filter",  "value"),
    Input("capex-grid",         "rowData"),
    prevent_initial_call=True,
)
def update_capex_chart(zone, tech, rows):
    empty_fig = px.line(title="No CAPEX data", template="plotly_white")
    if not rows:
        return empty_fig
    df = pd.DataFrame(rows)
    if zone and "z" in df.columns:
        df = df[df["z"] == zone]
    if tech and "tech" in df.columns:
        df = df[df["tech"] == tech]
    if df.empty:
        return empty_fig
    id_cols = [c for c in df.columns if not str(c).isdigit()]
    yr_cols = [c for c in df.columns if str(c).isdigit()]
    if not yr_cols:
        return empty_fig
    color_col = "tech" if "tech" in df.columns and not tech else \
                ("z"   if "z"    in df.columns and not zone  else id_cols[0] if id_cols else None)
    df_melt = df.melt(id_vars=id_cols, value_vars=yr_cols,
                      var_name="year", value_name="capex")
    df_melt["capex"] = pd.to_numeric(df_melt["capex"], errors="coerce")
    df_melt = df_melt.dropna(subset=["capex"])
    if df_melt.empty:
        return empty_fig
    title = "CAPEX Trajectories ($/kW)"
    if zone: title += f" — {zone}"
    if tech: title += f" — {tech}"
    fig = px.line(df_melt, x="year", y="capex", color=color_col,
                  markers=True, title=title,
                  labels={"capex": "$/kW", "year": "Year"},
                  template="plotly_white")
    fig.update_xaxes(type="category")
    return fig


@callback(
    Output("fuel-chart",      "figure", allow_duplicate=True),
    Input("fuel-zone-filter", "value"),
    Input("fuel-grid",        "rowData"),
    prevent_initial_call=True,
)
def update_fuel_chart(zone, rows):
    empty_fig = px.line(title="No fuel data", template="plotly_white")
    if not rows:
        return empty_fig
    df = pd.DataFrame(rows)
    # Rename unnamed cols if needed
    col_renames = {c: ["z", "f"][i] for i, c in enumerate(
        [c for c in df.columns if str(c).startswith("Unnamed:")][:2])}
    if col_renames:
        df = df.rename(columns=col_renames)
    if zone and "z" in df.columns:
        df = df[df["z"] == zone]
    if df.empty:
        return empty_fig
    id_cols = [c for c in ["z", "f"] if c in df.columns]
    yr_cols = [c for c in df.columns if c not in id_cols and str(c).isdigit()]
    if not yr_cols:
        return empty_fig
    df_melt = df.melt(id_vars=id_cols, value_vars=yr_cols,
                      var_name="year", value_name="price")
    df_melt["price"] = pd.to_numeric(df_melt["price"], errors="coerce")
    df_melt = df_melt.dropna(subset=["price"])
    color_col = "f" if "f" in df_melt.columns else None
    title = f"Fuel Prices ($/mmBTU){' — ' + zone if zone else ''}"
    fig = px.line(df_melt, x="year", y="price", color=color_col,
                  markers=True, title=title,
                  labels={"price": "$/mmBTU", "year": "Year"},
                  template="plotly_white")
    fig.update_xaxes(type="category")
    return fig


@callback(
    Output("gen-chart",     "figure"),
    Output("gen-pie-chart", "figure"),
    Input("gen-zone-filter",  "value"),
    Input("gen-grid",         "rowData"),
)
def update_gen_chart(zone, rows):
    empty_fig = px.bar(title="No generator data", template="plotly_white")
    empty_pie = px.pie(title="No data", template="plotly_white")
    if not rows:
        return empty_fig, empty_pie
    df = pd.DataFrame(rows)
    if zone and "z" in df.columns:
        df = df[df["z"] == zone]
    if df.empty:
        return empty_fig, empty_pie

    status_map = {1: "Existing", 2: "Committed", 3: "Candidate",
                  "1": "Existing", "2": "Committed", "3": "Candidate"}
    status_colors = {"Existing": "steelblue", "Committed": "mediumseagreen", "Candidate": "gold"}

    if "Status" in df.columns and "tech" in df.columns and "Capacity" in df.columns:
        df = df.copy()
        df["Status_label"] = df["Status"].map(status_map).fillna("Other")
        df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce").fillna(0)
        df_grp = df.groupby(["tech", "Status_label"])["Capacity"].sum().reset_index()
        # Sort techs by total capacity descending (largest bar at top)
        tech_order = (df_grp.groupby("tech")["Capacity"].sum()
                      .sort_values(ascending=True).index.tolist())
        df_grp["tech"] = pd.Categorical(df_grp["tech"], categories=tech_order, ordered=True)
        df_grp = df_grp.sort_values("tech")
        title = f"Capacity by Tech{' — ' + zone if zone else ''}"
        fig = px.bar(df_grp, x="Capacity", y="tech", color="Status_label",
                     orientation="h", barmode="stack",
                     color_discrete_map=status_colors,
                     title=title,
                     labels={"Capacity": "MW", "tech": "", "Status_label": "Status"},
                     template="plotly_white")
        fig.update_layout(
            legend_title_text="Status",
            title_font_size=13,
            legend={"font": {"size": 11}},
            yaxis={"tickfont": {"size": 11}, "automargin": True},
            xaxis={"title": "MW", "tickfont": {"size": 11}},
            margin={"t": 40, "b": 10, "l": 10},
        )

        # Stacked bar: Existing + Committed mix, one bar each, colored by fuel
        fuel_col = next((c for c in ["fuel", "f", "Fuel"] if c in df.columns), None)
        mix_title = f"Capacity Mix{' — ' + zone if zone else ''}"
        cmap = _load_tech_colors()

        df_mix = df[df["Status_label"].isin(["Existing", "Committed"])].copy()
        if not df_mix.empty and fuel_col:
            # tech→fuel (dominant fuel per tech)
            tech_fuel = (df_mix.groupby("tech")[fuel_col]
                         .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "")
                         .to_dict())
            # Build long-form dataframe for px.bar
            grp = (df_mix.groupby(["Status_label", "tech"])["Capacity"]
                   .sum().reset_index())
            grp = grp[grp["Capacity"] > 0]
            # Order fuels: baseload thermal at bottom → VRE → Battery at top
            FUEL_STACK_ORDER = [
                "Coal", "Uranium", "Gas", "LNG", "HFO", "LFO", "Diesel",
                "Geothermal", "Biomass", "Water", "CSP", "Wind", "Solar",
                "Import", "StorageEnergy", "Battery",
            ]
            def _fuel_rank(tech):
                fuel = tech_fuel.get(tech, "")
                try:
                    return FUEL_STACK_ORDER.index(fuel)
                except ValueError:
                    return len(FUEL_STACK_ORDER)  # unknowns go above everything

            all_techs = grp["tech"].unique().tolist()
            # Within same fuel, sort by total capacity descending (largest first = lower)
            cap_rank = grp.groupby("tech")["Capacity"].sum().to_dict()
            tech_order = sorted(all_techs,
                                key=lambda t: (_fuel_rank(t), -cap_rank.get(t, 0)))
            grp["tech"] = pd.Categorical(grp["tech"], categories=tech_order, ordered=True)
            grp = grp.sort_values("tech")
            color_map = {t: cmap.get(tech_fuel.get(t, ""), "#A9A9A9") for t in tech_order}

            fig_pie = px.bar(
                grp, x="Status_label", y="Capacity", color="tech",
                text="tech", barmode="stack",
                color_discrete_map=color_map,
                title=mix_title,
                labels={"Capacity": "MW", "Status_label": "", "tech": "Tech"},
                template="plotly_white",
                category_orders={"Status_label": ["Existing", "Committed"]},
            )
            fig_pie.update_traces(
                marker_line_color="#888888",
                marker_line_width=1.5,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=9, color="white"),
                cliponaxis=False,
            )
            fig_pie.update_layout(
                title_font_size=12,
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(title="MW", tickfont=dict(size=10)),
                showlegend=False,
                margin=dict(t=30, b=40, l=5, r=5),
            )
        else:
            fig_pie = empty_pie
    else:
        fig = px.bar(title="Missing tech/Status/Capacity columns", template="plotly_white")
        fig_pie = empty_pie

    return fig, fig_pie


# ---------------------------------------------------------------------------
# Save callbacks
# ---------------------------------------------------------------------------

@callback(Output("save-gen-msg", "children"), Input("save-gen-btn", "n_clicks"),
          State("gen-grid", "rowData"), State("supply-project", "value"),
          State("s-gen-variant", "value"), prevent_initial_call=True)
def save_gen(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "gen_data", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-storage-msg", "children"), Input("save-storage-btn", "n_clicks"),
          State("storage-grid", "rowData"), State("supply-project", "value"),
          State("s-sto-variant", "value"), prevent_initial_call=True)
def save_storage(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "storage_data", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-avail-msg", "children"), Input("save-avail-btn", "n_clicks"),
          State("avail-grid", "rowData"), State("supply-project", "value"),
          State("s-avail-variant", "value"), prevent_initial_call=True)
def save_avail(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "availability", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-fuel-msg", "children"), Input("save-fuel-btn", "n_clicks"),
          State("fuel-grid", "rowData"), State("supply-project", "value"),
          State("s-fuel-variant", "value"), prevent_initial_call=True)
def save_fuel(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "fuel_price", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


@callback(Output("save-capex-msg", "children"), Input("save-capex-btn", "n_clicks"),
          State("capex-grid", "rowData"), State("supply-project", "value"),
          State("s-capex-variant", "value"), prevent_initial_call=True)
def save_capex(n, rows, folder, variant):
    if not rows or not folder: return dbc.Badge("Nothing", color="warning")
    return dbc.Badge("Saved ✓", color="success") \
        if dl.save_variant(folder, "capex", variant, pd.DataFrame(rows)) \
        else dbc.Badge("Failed", color="danger")


# ---------------------------------------------------------------------------
# Duplicate callbacks
# ---------------------------------------------------------------------------

@callback(Output("s-gen-dup-msg", "children"),
          Output("s-gen-variant", "options", allow_duplicate=True),
          Output("s-gen-variant", "value", allow_duplicate=True),
          Input("s-gen-dup-btn", "n_clicks"),
          State("s-gen-variant", "value"), State("s-gen-dup-name", "value"),
          State("supply-project", "value"), prevent_initial_call=True)
def dup_gen(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "gen_data", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "gen_data"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("s-sto-dup-msg", "children"),
          Output("s-sto-variant", "options", allow_duplicate=True),
          Output("s-sto-variant", "value", allow_duplicate=True),
          Input("s-sto-dup-btn", "n_clicks"),
          State("s-sto-variant", "value"), State("s-sto-dup-name", "value"),
          State("supply-project", "value"), prevent_initial_call=True)
def dup_sto(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "storage_data", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "storage_data"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("s-avail-dup-msg", "children"),
          Output("s-avail-variant", "options", allow_duplicate=True),
          Output("s-avail-variant", "value", allow_duplicate=True),
          Input("s-avail-dup-btn", "n_clicks"),
          State("s-avail-variant", "value"), State("s-avail-dup-name", "value"),
          State("supply-project", "value"), prevent_initial_call=True)
def dup_avail(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "availability", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "availability"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("s-fuel-dup-msg", "children"),
          Output("s-fuel-variant", "options", allow_duplicate=True),
          Output("s-fuel-variant", "value", allow_duplicate=True),
          Input("s-fuel-dup-btn", "n_clicks"),
          State("s-fuel-variant", "value"), State("s-fuel-dup-name", "value"),
          State("supply-project", "value"), prevent_initial_call=True)
def dup_fuel(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "fuel_price", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "fuel_price"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("s-capex-dup-msg", "children"),
          Output("s-capex-variant", "options", allow_duplicate=True),
          Output("s-capex-variant", "value", allow_duplicate=True),
          Input("s-capex-dup-btn", "n_clicks"),
          State("s-capex-variant", "value"), State("s-capex-dup-name", "value"),
          State("supply-project", "value"), prevent_initial_call=True)
def dup_capex(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "capex", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "capex"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("sup-gen-open", "n_clicks"),
          State("supply-project", "value"),
          State("s-gen-variant", "value"),
          prevent_initial_call=True)
def open_gen_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "gen_data", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("sup-sto-open", "n_clicks"),
          State("supply-project", "value"),
          State("s-sto-variant", "value"),
          prevent_initial_call=True)
def open_storage_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "storage_data", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("sup-avail-open", "n_clicks"),
          State("supply-project", "value"),
          State("s-avail-variant", "value"),
          prevent_initial_call=True)
def open_avail_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "availability", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("sup-fuel-open", "n_clicks"),
          State("supply-project", "value"),
          State("s-fuel-variant", "value"),
          prevent_initial_call=True)
def open_fuel_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "fuel_price", variant)


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("sup-capex-open", "n_clicks"),
          State("supply-project", "value"),
          State("s-capex-variant", "value"),
          prevent_initial_call=True)
def open_capex_csv(n, folder, variant):
    from dash import no_update
    if not n or not folder: return no_update
    return dl.resolve_variant_path(folder, "capex", variant)
