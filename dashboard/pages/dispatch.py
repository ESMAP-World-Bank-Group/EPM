"""Results — Dispatch: hourly stacked-area dispatch with price overlay."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

import data_loader as dl


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Scenario", className="form-label-sm"),
                dcc.Dropdown(id="dp-scenario", multi=True,
                             placeholder="Select scenario(s)",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Spatial Resolution", className="form-label-sm"),
                dcc.RadioItems(
                    id="dp-spatial",
                    options=[{"label": " Country", "value": "c"},
                             {"label": " Zone",    "value": "z"}],
                    value="z", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                ),
            ], md=2),
            dbc.Col([
                html.Label("Zone / Country", className="form-label-sm"),
                dcc.Dropdown(id="dp-zone", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Year", className="form-label-sm"),
                dcc.Dropdown(id="dp-year", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=1),
            dbc.Col([
                html.Label("View", className="form-label-sm"),
                dcc.RadioItems(
                    id="dp-view",
                    options=[{"label": " Single Day",  "value": "single"},
                             {"label": " Full Year",   "value": "full"},
                             {"label": " Difference",  "value": "diff"}],
                    value="single", inline=True, className="mt-1",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "10px", "fontSize": "0.85rem"},
                ),
            ], md=3),
            dbc.Col([
                html.Label("Quarter", className="form-label-sm"),
                dcc.Dropdown(id="dp-quarter", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=1),
            dbc.Col([
                html.Label("Day Type", className="form-label-sm"),
                dcc.Dropdown(id="dp-day", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=1),
        ], className="g-2")), className="mb-2 shadow-sm filter-card"),

        dbc.Row([
            dbc.Col(dbc.Button("Load Dispatch Data", id="dp-load-btn",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(html.Span(id="dp-status", className="text-muted",
                              style={"fontSize": "0.82rem", "lineHeight": "31px"}),
                    width="auto"),
        ], className="mb-3 align-items-center"),

        dbc.Card(dbc.CardBody(
            dcc.Graph(id="dp-chart",
                      config={"displayModeBar": True,
                              "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                      style={"width": "100%"}),
        ), className="shadow-sm border-0 mb-3"),

        dbc.Card([
            dbc.CardHeader(html.B("Hourly Electricity Price (USD/MWh)")),
            dbc.CardBody(dcc.Graph(id="dp-price-chart",
                                   config={"displayModeBar": False},
                                   style={"height": "200px"})),
        ], className="shadow-sm border-0"),

        dcc.Store(id="dp-data-store"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Filter init
# ---------------------------------------------------------------------------

@callback(
    Output("dp-scenario", "options"),
    Output("dp-scenario", "value"),
    Output("dp-year",     "options"),
    Output("dp-year",     "value"),
    Input("filter-run",   "value"),
)
def init_dp_scenario_year(run):
    if not run:
        return [], [], [], None
    scenarios = dl.list_scenarios(run)
    years     = dl.get_merged_years(run)
    s_opts = [{"label": s, "value": s} for s in scenarios]
    y_opts = [{"label": str(y), "value": y} for y in years]
    default_s = next((s for s in ["baseline", "Baseline"] if s in scenarios),
                     [scenarios[0]] if scenarios else [])
    if isinstance(default_s, str):
        default_s = [default_s]
    return s_opts, default_s, y_opts, (years[0] if years else None)


@callback(
    Output("dp-zone",    "options"),
    Output("dp-zone",    "value"),
    Input("dp-spatial",  "value"),
    Input("filter-run",  "value"),
    State("dp-zone",     "value"),
)
def update_dp_zones(spatial, run, current_value):
    if not run:
        return [], None
    units = (dl.get_merged_zones(run) if spatial == "z"
             else dl.get_merged_countries(run))
    opts = [{"label": u, "value": u} for u in units]
    if ctx.triggered_id == "dp-spatial":
        return opts, (units[0] if units else None)
    if current_value and current_value in set(units):
        return opts, no_update
    return opts, (units[0] if units else None)


# ---------------------------------------------------------------------------
# Data load on button click
# ---------------------------------------------------------------------------

@callback(
    Output("dp-data-store", "data"),
    Output("dp-status",     "children"),
    Output("dp-quarter",    "options"),
    Output("dp-quarter",    "value"),
    Output("dp-day",        "options"),
    Output("dp-day",        "value"),
    Input("dp-load-btn",    "n_clicks"),
    State("dp-scenario",    "value"),
    State("dp-year",        "value"),
    State("dp-spatial",     "value"),
    State("dp-zone",        "value"),
    State("filter-run",     "value"),
    prevent_initial_call=True,
)
def load_dp_data(n, scenarios, year, spatial, zone, run):
    empty = []
    if not scenarios or not year or not zone or not run:
        return None, "Select filters first.", empty, None, empty, None
    active = scenarios if isinstance(scenarios, list) else [scenarios]
    df = dl.load_dispatch_merged(run, active)
    if df.empty:
        msg = ("No dispatch data found. "
               "Make sure pDispatchComplete.csv exists in output_csv/.")
        return None, msg, empty, None, empty, None

    spatial_col = "z" if spatial == "z" else "c"
    if spatial_col not in df.columns:
        spatial_col = "z"
    df = df[(df[spatial_col] == zone) & (df["y"] == year)].copy()
    if df.empty:
        return None, "No data for this selection.", empty, None, empty, None

    quarters = sorted(df["q"].dropna().unique())
    days     = sorted(df["d"].dropna().unique())
    q_opts = [{"label": q, "value": q} for q in quarters]
    d_opts = [{"label": d, "value": d} for d in days]
    data_json = df[["scenario", "q", "d", "t", "uni", "value"]].to_json(orient="split")
    status = f"Loaded — {zone} / {', '.join(active)} / {int(year)}"
    return data_json, status, q_opts, (quarters[0] if quarters else None), \
           d_opts, (days[0] if days else None)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _parse(df):
    df = df.copy()
    for col, src in [("t_num", "t"), ("q_num", "q"), ("d_num", "d")]:
        df[col] = df[src].astype(str).str.extract(r"(\d+)")[0].astype(float)
    return df.dropna(subset=["t_num", "q_num", "d_num"]).astype(
        {"t_num": int, "q_num": int, "d_num": int})


def _time_index(df):
    ti = (df[["q_num", "d_num", "t_num", "q", "d"]]
          .drop_duplicates()
          .sort_values(["q_num", "d_num", "t_num"])
          .reset_index(drop=True))
    ti["x"] = ti.index
    return ti


def _add_row(fig, df, ti, row, legend_shown):
    df = df.merge(ti[["q_num", "d_num", "t_num", "x"]],
                  on=["q_num", "d_num", "t_num"])
    x_arr = ti["x"].values
    gen_df    = df[df["uni"] != "Demand"]
    demand_df = df[df["uni"] == "Demand"]
    techs   = gen_df["uni"].unique().tolist()
    ordered = [t for t in dl.TECH_ORDER if t in techs] + \
              [t for t in techs if t not in dl.TECH_ORDER]
    for tech in ordered:
        vals = (gen_df[gen_df["uni"] == tech]
                .groupby("x")["value"].sum()
                .reindex(x_arr).fillna(0).values)
        color = dl.TECH_COLORS.get(tech, "#aaaaaa")
        for sg, arr, check in (
            ("pos", np.maximum(vals, 0), lambda a: a.max() > 1e-6),
            ("neg", np.minimum(vals, 0), lambda a: a.min() < -1e-6),
        ):
            if not check(arr):
                continue
            sl = tech not in legend_shown
            legend_shown.add(tech)
            fig.add_trace(go.Scatter(
                x=x_arr, y=arr, name=tech,
                stackgroup=f"{sg}_{row}", mode="none",
                fillcolor=color, legendgroup=tech, showlegend=sl,
                hovertemplate=f"<b>{tech}</b><br>%{{y:.1f}} MW<extra></extra>",
            ), row=row, col=1)
    if not demand_df.empty:
        dem = demand_df.groupby("x")["value"].sum().reindex(x_arr).fillna(0)
        sl = "Demand" not in legend_shown
        legend_shown.add("Demand")
        fig.add_trace(go.Scatter(
            x=dem.index, y=dem.values, name="Demand", mode="lines",
            line=dict(color="#e74c3c", width=1.5),
            legendgroup="Demand", showlegend=sl,
        ), row=row, col=1)


def _separators(fig, ti, day_weights=None):
    qd = (ti.groupby(["q_num", "d_num", "q", "d"])["x"]
          .agg(x_min="min", x_max="max").reset_index())
    first_q  = qd["q_num"].min()
    n_total  = len(qd)
    pct_def  = 100.0 / n_total if n_total else 0
    for _, r in qd.iterrows():
        first_d = r["d_num"] == qd[qd["q_num"] == r["q_num"]]["d_num"].min()
        if not (r["q_num"] == first_q and first_d):
            is_q = first_d
            fig.add_shape(type="line",
                          x0=r["x_min"] - 0.5, x1=r["x_min"] - 0.5,
                          y0=0, y1=1.0, xref="x", yref="paper",
                          line=dict(color="#bbbbbb" if is_q else "#e0e0e0",
                                    width=1.0 if is_q else 0.5,
                                    dash="solid" if is_q else "dot"))
        mid = (r["x_min"] + r["x_max"]) / 2
        pct = day_weights.get((r["q"], r["d"]), pct_def) if day_weights else pct_def
        fig.add_annotation(x=mid, y=1.005, xref="x", yref="paper",
                           text=f"<span style='font-size:7px;color:#ccc'>{pct:.1f}%</span>",
                           showarrow=False, xanchor="center", yanchor="bottom")
    for q_num in sorted(qd["q_num"].unique()):
        qr = qd[qd["q_num"] == q_num]
        mid = (qr["x_min"].min() + qr["x_max"].max()) / 2
        fig.add_annotation(x=mid, y=-0.05, xref="x", yref="paper",
                           text=f"<b>{qr.iloc[0]['q']}</b>",
                           showarrow=False, font=dict(size=10, color="#555"),
                           xanchor="center", yanchor="top")
    tick_vals = [r["x"] for _, r in ti.iterrows() if r["t_num"] % 6 == 1]
    tick_text = [str(r["t_num"]) for _, r in ti.iterrows() if r["t_num"] % 6 == 1]
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, tickfont=dict(size=8))


def _build_chart(df, zone, year, view, quarter=None, day=None, day_weights=None):
    all_s = sorted(df["scenario"].unique())
    sc_list = (["baseline"] if "baseline" in all_s else []) + \
              [s for s in all_s if s != "baseline"]

    if view == "diff":
        if len(sc_list) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Select ≥ 2 scenarios for Difference view.",
                               xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=14))
            fig.update_layout(height=440, paper_bgcolor="white", plot_bgcolor="white")
            return fig
        ref, cmp = sc_list[0], sc_list[1]
        ref_df = _parse(df[df["scenario"] == ref])
        cmp_df = _parse(df[df["scenario"] == cmp])
        ti = _time_index(ref_df)
        ref_agg = ref_df.groupby(["q_num", "d_num", "t_num", "uni"])["value"].sum().reset_index()
        cmp_agg = cmp_df.groupby(["q_num", "d_num", "t_num", "uni"])["value"].sum().reset_index()
        delta = cmp_agg.merge(ref_agg, on=["q_num", "d_num", "t_num", "uni"],
                              how="outer", suffixes=("", "_ref"))
        delta["value"] = delta["value"].fillna(0) - delta["value_ref"].fillna(0)
        delta = delta.drop(columns=["value_ref"])
        delta = delta.merge(
            ti[["q_num", "d_num", "t_num", "q", "d"]].drop_duplicates(),
            on=["q_num", "d_num", "t_num"])
        fig = make_subplots(rows=1, cols=1)
        _add_row(fig, delta, ti, 1, set())
        _separators(fig, ti, day_weights=day_weights)
        fig.update_layout(
            title=dict(text=f"{zone} — Δ Dispatch ({cmp} − {ref}) | {int(year)}",
                       font=dict(size=12), x=0.01),
            margin=dict(l=10, r=20, t=50, b=60),
            xaxis_title="Hours", yaxis_title="MW",
            legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified", height=480,
        )
        return fig

    n = len(sc_list)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=max(0.04, 0.12 / n),
                        subplot_titles=sc_list[:])
    s0 = _parse(df[df["scenario"] == sc_list[0]])
    if view == "single":
        s0 = s0[(s0["q"] == quarter) & (s0["d"] == day)]
    ti = _time_index(s0)
    legend_shown = set()
    for i, sc in enumerate(sc_list, 1):
        sdf = _parse(df[df["scenario"] == sc])
        if view == "single":
            sdf = sdf[(sdf["q"] == quarter) & (sdf["d"] == day)]
        _add_row(fig, sdf, ti, i, legend_shown)
    if view == "full":
        _separators(fig, ti, day_weights=day_weights)
    else:
        fig.update_xaxes(tickmode="linear", dtick=2)
    height = max(440, 420 * n)
    title_sfx = f" | {quarter} | {day}" if view == "single" else ""
    fig.update_layout(
        title=dict(text=f"{zone} — Dispatch | {int(year)}{title_sfx}",
                   font=dict(size=12), x=0.01),
        margin=dict(l=10, r=20, t=50, b=60),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=height,
    )
    for i in range(1, n + 1):
        ax = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(**{ax: dict(title="MW")})
    return fig


# ---------------------------------------------------------------------------
# Chart callback
# ---------------------------------------------------------------------------

@callback(
    Output("dp-chart",       "figure"),
    Output("dp-price-chart", "figure"),
    Input("dp-data-store",   "data"),
    Input("dp-quarter",      "value"),
    Input("dp-day",          "value"),
    Input("dp-view",         "value"),
    State("dp-year",         "value"),
    State("dp-zone",         "value"),
    State("filter-run",      "value"),
)
def update_dp_charts(data_json, quarter, day, view, year, zone, run):
    empty = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10), height=440,
    )
    if not data_json:
        return empty, empty
    if view == "single" and (not quarter or not day):
        return empty, empty

    df = pd.read_json(data_json, orient="split")
    day_weights = dl.load_phours_merged(run) if run else {}
    dispatch_fig = _build_chart(df, zone, year, view, quarter, day,
                                day_weights=day_weights)

    # Price chart
    price_fig = go.Figure()
    if run:
        scenarios = sorted(df["scenario"].unique())
        price_df  = dl.load_hourly_price_merged(run, scenarios)
        if not price_df.empty:
            colors = ["#2c6fad", "#e07b39", "#27ae60", "#8e44ad"]
            sc_list = (["baseline"] if "baseline" in scenarios else []) + \
                      [s for s in scenarios if s != "baseline"]
            for i, sc in enumerate(sc_list):
                filt = (price_df["scenario"] == sc) & (price_df["y"] == year)
                if "z" in price_df.columns:
                    filt &= (price_df["z"] == zone)
                if view == "single":
                    filt &= (price_df["q"] == quarter) & (price_df["d"] == day)
                p = price_df[filt].copy()
                if p.empty:
                    continue
                p["t_num"] = p["t"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                p["q_num"] = p["q"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                p["d_num"] = p["d"].astype(str).str.extract(r"(\d+)")[0].astype(float)
                p = p.dropna(subset=["t_num", "q_num", "d_num"]).sort_values(
                    ["q_num", "d_num", "t_num"])
                p["x"] = range(len(p))
                x_vals = p["t_num"].values if view == "single" else p["x"].values
                price_fig.add_trace(go.Scatter(
                    x=x_vals, y=p["value"].values,
                    name=sc, mode="lines",
                    fill="tozeroy" if i == 0 else "none",
                    fillcolor="rgba(44, 111, 173, 0.12)",
                    line=dict(color=colors[i % len(colors)], width=1.5),
                ))
    price_fig.update_layout(
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis_title="Hour(s)", yaxis_title="USD/MWh",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    return dispatch_fig, price_fig
