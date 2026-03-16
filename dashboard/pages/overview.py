"""Results — Overview: KPI cards, interconnection map, capacity mix, dispatch."""

import math

import dash_bootstrap_components as dbc
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

import data_loader as dl


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _arrowhead_geo(src_lat, src_lon, dst_lat, dst_lon, hw_deg=0.45):
    cos_lat = math.cos(math.radians((src_lat + dst_lat) / 2))
    dlat = dst_lat - src_lat
    dlon = (dst_lon - src_lon) * cos_lat
    length = math.sqrt(dlat ** 2 + dlon ** 2)
    if length < 1e-6:
        return [], []
    ulat, ulon = dlat / length, dlon / length
    perp_lat, perp_lon = -ulon, ulat
    tip_lat  = src_lat + 0.56 * (dst_lat - src_lat)
    tip_lon  = src_lon + 0.56 * (dst_lon - src_lon)
    base_lat = src_lat + 0.44 * (dst_lat - src_lat)
    base_lon = src_lon + 0.44 * (dst_lon - src_lon)
    c1_lat = base_lat + perp_lat * hw_deg
    c1_lon = base_lon + perp_lon * hw_deg / cos_lat
    c2_lat = base_lat - perp_lat * hw_deg
    c2_lon = base_lon - perp_lon * hw_deg / cos_lat
    return [tip_lat, c1_lat, c2_lat, tip_lat], [tip_lon, c1_lon, c2_lon, tip_lon]


def _parse_time(df):
    for col, src in [("q_num", "q"), ("d_num", "d"), ("t_num", "t")]:
        df[col] = pd.to_numeric(
            df[src].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    df = df.dropna(subset=["q_num", "d_num", "t_num"])
    df[["q_num", "d_num", "t_num"]] = df[["q_num", "d_num", "t_num"]].astype(int)
    return df


def _dispatch_annual_fig(dispatch_df, zone, scenario, year,
                         price_df=None, day_weights=None):
    df = dispatch_df[
        (dispatch_df["z"] == zone) &
        (dispatch_df["scenario"] == scenario) &
        (dispatch_df["y"] == year)
    ].copy()
    empty = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    if df.empty:
        return empty

    df = _parse_time(df)
    slots = (
        df[["q_num", "d_num", "t_num", "q", "d"]]
        .drop_duplicates()
        .sort_values(["q_num", "d_num", "t_num"])
        .reset_index(drop=True)
    )
    slots["x"] = slots.index
    df = df.merge(slots, on=["q_num", "d_num", "t_num", "q", "d"], how="left")

    pos_sum = df[df["value"] > 0].groupby("x")["value"].sum()
    neg_sum = df[df["value"] < 0].groupby("x")["value"].sum()
    y1_max = pos_sum.max() if not pos_sum.empty else 1000
    y1_min = neg_sum.min() if not neg_sum.empty else 0

    fig = go.Figure()
    uni_col = "uni" if "uni" in df.columns else "techfuel"
    tech_list = [t for t in dl.TECH_ORDER if t in df[uni_col].unique()]
    other = [t for t in df[uni_col].unique() if t not in dl.TECH_ORDER and t != "Demand"]
    shown = set()
    for tech in other + tech_list:
        t_df = df[df[uni_col] == tech].sort_values("x")
        if t_df.empty:
            continue
        xi, vals = t_df["x"].values, t_df["value"].values
        color = dl.TECH_COLORS.get(tech, "#aaaaaa")
        show = tech not in shown
        shown.add(tech)
        pos, neg = np.maximum(vals, 0), np.minimum(vals, 0)
        if pos.sum() > 0:
            fig.add_trace(go.Scatter(
                x=xi, y=pos, name=tech, mode="none",
                fill="tonexty", stackgroup="pos",
                fillcolor=color, line=dict(width=0),
                showlegend=show, legendgroup=tech,
                hovertemplate=f"<b>{tech}</b>: %{{y:,.0f}} MW<extra></extra>",
            ))
        if neg.sum() < 0:
            fig.add_trace(go.Scatter(
                x=xi, y=neg, name=tech, mode="none",
                fill="tonexty", stackgroup="neg",
                fillcolor=color, line=dict(width=0),
                showlegend=False, legendgroup=tech,
            ))

    dem = df[df[uni_col] == "Demand"].sort_values("x")
    if not dem.empty:
        fig.add_trace(go.Scatter(
            x=dem["x"], y=dem["value"], name="Demand", mode="lines",
            line=dict(color="#e74c3c", width=1.5),
        ))

    has_price, y2 = False, {}
    if price_df is not None and not price_df.empty:
        p = price_df[
            (price_df["z"] == zone) &
            (price_df["scenario"] == scenario) &
            (price_df["y"] == year)
        ].copy()
        if not p.empty:
            p = _parse_time(p)
            p = p.merge(slots[["q_num", "d_num", "t_num", "x"]],
                        on=["q_num", "d_num", "t_num"], how="inner").sort_values("x")
            fig.add_trace(go.Scatter(
                x=p["x"], y=p["value"], name="Marginal Cost",
                mode="lines", yaxis="y2",
                line=dict(color="#2c3e50", width=1),
                hovertemplate="<b>Marginal Cost</b>: %{y:.1f} USD/MWh<extra></extra>",
            ))
            has_price = True
            p_max_val = p["value"].max() * 1.1
            y1_span = y1_max - y1_min
            if y1_span > 0 and y1_min < 0:
                zero_frac = -y1_min / y1_span
                y2_min = zero_frac * p_max_val / (zero_frac - 1)
            else:
                y2_min = 0
            y2 = dict(title="USD/MWh", overlaying="y", side="right",
                      showgrid=False, zeroline=False,
                      range=[y2_min, p_max_val])

    qd = (slots.groupby(["q_num", "d_num", "q", "d"])["x"]
          .agg(x_min="min", x_max="max").reset_index())
    first_q = qd["q_num"].min()
    n_total = len(qd)
    pct_default = 100.0 / n_total if n_total else 0
    for _, r in qd.iterrows():
        first_d_in_q = r["d_num"] == qd[qd["q_num"] == r["q_num"]]["d_num"].min()
        very_first = r["q_num"] == first_q and first_d_in_q
        if not very_first:
            is_q = first_d_in_q
            fig.add_shape(type="line",
                          x0=r["x_min"] - 0.5, x1=r["x_min"] - 0.5,
                          y0=0, y1=1, xref="x", yref="paper",
                          line=dict(color="#bbbbbb" if is_q else "#e0e0e0",
                                    width=1.0 if is_q else 0.5,
                                    dash="solid" if is_q else "dot"))
        mid = (r["x_min"] + r["x_max"]) / 2
        pct = day_weights.get((r["q"], r["d"]), pct_default) if day_weights else pct_default
        fig.add_annotation(x=mid, y=1.005, xref="x", yref="paper",
                           text=f"<span style='font-size:7px;color:#cccccc'>{pct:.1f}%</span>",
                           showarrow=False, xanchor="center", yanchor="bottom")
    for q_num in sorted(qd["q_num"].unique()):
        qr = qd[qd["q_num"] == q_num]
        mid = (qr["x_min"].min() + qr["x_max"].max()) / 2
        fig.add_annotation(x=mid, y=1.03, xref="x", yref="paper",
                           text=f"<span style='font-size:8px;color:#888'>Q{int(q_num)}</span>",
                           showarrow=False, xanchor="center", yanchor="bottom")

    fig.update_layout(
        margin=dict(l=10, r=50 if has_price else 10, t=25, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="v", x=1.08 if has_price else 1.01, y=1,
                    font=dict(size=9)),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="MW", zeroline=True, zerolinecolor="#cccccc",
                   range=[y1_min * 1.05 if y1_min < 0 else None, y1_max * 1.05]),
        yaxis2=y2, font=dict(size=10), hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------

def _kpi_card(title, value_id, icon, color, sub_id=None, sub_label=None):
    value_col = [
        html.P(title, className="text-muted mb-0",
               style={"fontSize": "0.78rem", "fontWeight": 600}),
        html.H4(id=value_id, className="mb-0 fw-bold"),
    ]
    if sub_id:
        value_col.append(
            html.P([html.Span(sub_label + " ", style={"fontWeight": 500}),
                    html.Span(id=sub_id)],
                   className="text-muted mb-0", style={"fontSize": "0.78rem"})
        )
    return dbc.Card([
        dbc.CardBody(dbc.Row([
            dbc.Col(html.I(className=f"bi {icon}",
                           style={"fontSize": "2rem", "color": color}), width=3),
            dbc.Col(value_col, width=9),
        ], align="center")),
    ], className="shadow-sm border-0 h-100")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        # ── Page-level filters ────────────────────────────────────────────
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Scenario", className="form-label-sm"),
                dcc.Dropdown(id="ov-scenario", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Year", className="form-label-sm"),
                dcc.Dropdown(id="ov-year", clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=2),
        ], className="g-2")), className="mb-3 shadow-sm filter-card"),

        # ── KPI strip ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_kpi_card("Generation Capacity", "ov-kpi-gen-capa",
                              "bi-lightning-charge", "#2d9e4f"), md=3),
            dbc.Col(_kpi_card("Total Demand", "ov-kpi-demand-energy",
                              "bi-graph-up", "#f77f00",
                              sub_id="ov-kpi-demand-peak", sub_label="Peak:"), md=3),
            dbc.Col(_kpi_card("Transmission Capacity", "ov-kpi-tr-capa",
                              "bi-diagram-3", "#2c6fad"), md=3),
            dbc.Col(_kpi_card("Trade Volume", "ov-kpi-trade",
                              "bi-arrow-left-right", "#7b4f9e"), md=3),
        ], className="mb-3 g-3"),

        # ── Map + Capacity bar ────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("Interconnections — Net Flow & Utilization")),
                dbc.CardBody(dcc.Graph(id="ov-map",
                                       config={"displayModeBar": False},
                                       style={"height": "400px"}),
                             style={"padding": "4px"}),
            ], className="shadow-sm border-0 h-100"), md=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B("Capacity Mix — Selected Year")),
                dbc.CardBody(dcc.Graph(id="ov-capacity-bar",
                                       config={"displayModeBar": False},
                                       style={"height": "390px"})),
            ], className="shadow-sm border-0 h-100"), md=6),
        ], className="mb-3 g-3"),

        # ── Dispatch on map click ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.B(id="ov-dispatch-title",
                                      children="Annual Dispatch — click a zone on the map")),
                dbc.CardBody(dcc.Graph(id="ov-dispatch",
                                       config={"displayModeBar": False},
                                       style={"height": "380px"})),
            ], className="shadow-sm border-0"), md=12),
        ], className="mb-3 g-3"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("ov-scenario", "options"),
    Output("ov-scenario", "value"),
    Output("ov-year",     "options"),
    Output("ov-year",     "value"),
    Input("filter-run",   "value"),
    State("ov-scenario",  "value"),
    State("ov-year",      "value"),
)
def init_ov_filters(run, cur_scenario, cur_year):
    if not run:
        return [], None, [], None
    scenarios = dl.list_scenarios(run)
    years     = dl.get_merged_years(run)
    s_opts = [{"label": s, "value": s} for s in scenarios]
    y_opts = [{"label": str(y), "value": y} for y in years]
    default_s = next((s for s in ["baseline", "Baseline"] if s in scenarios),
                     scenarios[0] if scenarios else None)
    s_val = cur_scenario if cur_scenario in scenarios else default_s
    y_val = cur_year if cur_year in years else (years[-1] if years else None)
    return s_opts, s_val, y_opts, y_val


@callback(
    Output("ov-kpi-gen-capa",      "children"),
    Output("ov-kpi-demand-energy", "children"),
    Output("ov-kpi-demand-peak",   "children"),
    Output("ov-kpi-tr-capa",       "children"),
    Output("ov-kpi-trade",         "children"),
    Input("filter-run",   "value"),
    Input("ov-scenario",  "value"),
    Input("ov-year",      "value"),
)
def update_ov_kpis(run, scenario, year):
    dash = "—"
    if not run or not scenario or not year:
        return dash, dash, dash, dash, dash

    # Generation capacity
    tf = dl.load_techfuel(run)
    gen_capa = dash
    if not tf.empty:
        sub = tf[(tf["scenario"] == scenario) &
                 (tf["attribute"] == "CapacityTechFuel") &
                 (tf["y"] == year)]
        if not sub.empty:
            gen_capa = f"{sub['value'].sum() / 1000:.1f} GW"

    # Demand energy + peak
    yz = dl.load_yearly_zone(run)
    demand_energy = demand_peak = dash
    if not yz.empty:
        sub_y = yz[(yz["scenario"] == scenario) & (yz["y"] == year)]
        dem_e = sub_y[sub_y["attribute"] == "DemandEnergyZone"]["value"].sum()
        dem_p = sub_y[sub_y["attribute"] == "DemandPeakZone"]["value"].sum()
        demand_energy = f"{dem_e / 1000:.1f} TWh" if dem_e > 0 else dash
        demand_peak   = f"{dem_p / 1000:.1f} GW"  if dem_p > 0 else dash

    # Transmission capacity
    tr = dl.load_transmission_merged(run)
    tr_capa = dash
    if not tr.empty:
        sub_tr = tr[(tr["scenario"] == scenario) &
                    (tr["attribute"] == "TransmissionCapacity") &
                    (tr["y"] == year)].dropna(subset=["value", "z2"])
        if not sub_tr.empty:
            sub_tr = sub_tr.copy()
            sub_tr["key"] = sub_tr.apply(
                lambda r: tuple(sorted([r["z"], r["z2"]])), axis=1)
            tr_capa = f"{sub_tr.groupby('key')['value'].max().sum():,.0f} MW"

    # Trade volume
    trade = dash
    if not tr.empty:
        sub_ic = tr[(tr["scenario"] == scenario) &
                    (tr["attribute"] == "Interchange") &
                    (tr["y"] == year)].dropna(subset=["value"])
        if not sub_ic.empty:
            trade = f"{sub_ic['value'].sum() / 2:,.0f} GWh"

    return gen_capa, demand_energy, demand_peak, tr_capa, trade


@callback(
    Output("ov-map",          "figure"),
    Output("ov-capacity-bar", "figure"),
    Input("filter-run",   "value"),
    Input("ov-scenario",  "value"),
    Input("ov-year",      "value"),
)
def update_ov_map_and_capacity(run, scenario, year):
    empty_fig = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if not run or not scenario or not year:
        return empty_fig, empty_fig

    zone_coords = dl.load_zone_coords(run)
    tf = dl.load_techfuel(run)
    if tf.empty:
        return empty_fig, empty_fig

    sub = tf[(tf["scenario"] == scenario) &
             (tf["attribute"] == "CapacityTechFuel") &
             (tf["y"] == year)]

    # ── Capacity bar (always shown) ────────────────────────────────────────
    techfuel_col = "techfuel" if "techfuel" in sub.columns else (
        "uni" if "uni" in sub.columns else None)
    if techfuel_col and not sub.empty:
        pivot = sub.groupby(["z", techfuel_col])["value"].sum().reset_index()
        tech_order = [t for t in dl.TECH_ORDER if t in pivot[techfuel_col].unique()]
        other = [t for t in pivot[techfuel_col].unique() if t not in tech_order]
        zone_order = (pivot.groupby("z")["value"].sum()
                      .sort_values(ascending=False).index.tolist())
        bar_fig = px.bar(
            pivot, x="z", y="value", color=techfuel_col,
            color_discrete_map=dl.TECH_COLORS,
            category_orders={techfuel_col: other + tech_order, "z": zone_order},
            labels={"value": "Capacity (MW)", "z": "Zone", techfuel_col: "Technology"},
            template="plotly_white",
        )
        bar_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10)),
            xaxis_tickangle=-30, yaxis_title="Capacity (MW)",
            plot_bgcolor="white", paper_bgcolor="white",
        )
    else:
        bar_fig = empty_fig

    # ── Map (only if zone coords available) ───────────────────────────────
    if not zone_coords:
        map_fig = go.Figure().update_layout(
            paper_bgcolor="white",
            annotations=[dict(
                text="No zone coordinate file found (linestring_countries.geojson).",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="#aaa"),
            )],
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return map_fig, bar_fig

    tr = dl.load_transmission_merged(run)
    map_fig = go.Figure()

    if not tr.empty:
        tr_y = tr[(tr["scenario"] == scenario) & (tr["y"] == year)]

        def _get(attr, z1, z2):
            v = tr_y[(tr_y["attribute"] == attr) &
                     (tr_y["z"] == z1) & (tr_y["z2"] == z2)]["value"]
            return float(v.iloc[0]) if not v.empty else 0.0

        pairs_raw = tr_y[tr_y["attribute"] == "Interchange"][["z", "z2"]].drop_duplicates()
        seen, corridors = set(), []
        for _, row in pairs_raw.iterrows():
            z1, z2 = row["z"], row["z2"]
            if z1 not in zone_coords or z2 not in zone_coords:
                continue
            key = tuple(sorted([z1, z2]))
            if key in seen:
                continue
            seen.add(key)
            corridors.append(key)

        if corridors:
            vols = [abs(_get("Interchange", z1, z2)) + abs(_get("Interchange", z2, z1))
                    for z1, z2 in corridors]
            max_vol = max(vols) if vols else 1.0

            for (z1, z2), vol in zip(corridors, vols):
                lat1, lon1 = zone_coords[z1]
                lat2, lon2 = zone_coords[z2]
                u1 = _get("InterconUtilization", z1, z2)
                u2 = _get("InterconUtilization", z2, z1)
                util = (u1 + u2) / 2 if (u1 + u2) > 0 else max(u1, u2)
                cap  = max(_get("TransmissionCapacity", z1, z2),
                           _get("TransmissionCapacity", z2, z1))
                flow_fwd = abs(_get("Interchange", z1, z2))
                flow_bwd = abs(_get("Interchange", z2, z1))
                if flow_fwd >= flow_bwd:
                    src_lat, src_lon, dst_lat, dst_lon = lat1, lon1, lat2, lon2
                    src_z, dst_z = z1, z2
                else:
                    src_lat, src_lon, dst_lat, dst_lon = lat2, lon2, lat1, lon1
                    src_z, dst_z = z2, z1
                color = pc.sample_colorscale("YlOrRd", [min(max(util, 0), 1)])[0]
                width = 1.5 + 5.0 * (vol / max_vol)
                hover = (f"<b>{z1} ↔ {z2}</b><br>"
                         f"Net flow: {src_z} → {dst_z}<br>"
                         f"Volume: {vol:,.0f} GWh<br>"
                         f"Capacity: {cap:,.0f} MW<br>"
                         f"Utilization: {util*100:.1f}%<extra></extra>")
                n = 20
                lats_l = [lat1 + i/n * (lat2-lat1) for i in range(n+1)]
                lons_l = [lon1 + i/n * (lon2-lon1) for i in range(n+1)]
                map_fig.add_trace(go.Scattergeo(
                    lat=lats_l, lon=lons_l, mode="lines",
                    line=dict(color=color, width=width),
                    hovertemplate=hover, showlegend=False,
                ))
                arr_lats, arr_lons = _arrowhead_geo(
                    src_lat, src_lon, dst_lat, dst_lon, hw_deg=0.45)
                if arr_lats:
                    map_fig.add_trace(go.Scattergeo(
                        lat=arr_lats, lon=arr_lons, mode="lines",
                        fill="toself", fillcolor=color,
                        line=dict(color=color, width=0.5),
                        hoverinfo="skip", showlegend=False,
                    ))

    # Zone dots
    yz = dl.load_yearly_zone(run)
    yz_y = yz[(yz["scenario"] == scenario) & (yz["y"] == year)] \
           if not yz.empty else pd.DataFrame()
    tr_ic = (tr[(tr["scenario"] == scenario) & (tr["y"] == year) &
                (tr["attribute"] == "Interchange")].dropna(subset=["value", "z2"])
             if not tr.empty else pd.DataFrame())

    def _zone_val(df_sub, attr):
        if df_sub.empty:
            return {}
        s = df_sub[df_sub["attribute"] == attr] if "attribute" in df_sub.columns else df_sub
        return s.groupby("z")["value"].sum().to_dict()

    tf_sc_y = tf[(tf["scenario"] == scenario) & (tf["y"] == year)]
    capa_by_z   = _zone_val(tf_sc_y[tf_sc_y["attribute"] == "CapacityTechFuel"],
                             "CapacityTechFuel")
    gen_by_z    = _zone_val(tf_sc_y[tf_sc_y.get("attribute", pd.Series()).eq("EnergyTechFuelComplete")]
                            if "attribute" in tf_sc_y.columns else pd.DataFrame(),
                             "EnergyTechFuelComplete")
    demand_by_z = _zone_val(yz_y[yz_y["attribute"] == "DemandEnergyZone"]
                            if not yz_y.empty else pd.DataFrame(), "DemandEnergyZone")
    price_by_z  = (yz_y[yz_y["attribute"] == "GenCostsPerMWh"].groupby("z")["value"].mean().to_dict()
                   if not yz_y.empty else {})
    exports_by_z = tr_ic.groupby("z")["value"].sum().to_dict()  if not tr_ic.empty else {}
    imports_by_z = tr_ic.groupby("z2")["value"].sum().to_dict() if not tr_ic.empty else {}

    z_in_data = (set(tr[(tr["scenario"] == scenario)]["z"].dropna().unique()) |
                 set(tr[(tr["scenario"] == scenario)]["z2"].dropna().unique())
                 if not tr.empty else set())

    zone_lats, zone_lons, zone_names, zone_hovers = [], [], [], []
    for z, (lat, lon) in zone_coords.items():
        if z_in_data and z not in z_in_data:
            continue
        capa  = capa_by_z.get(z, 0)
        dem   = demand_by_z.get(z, 0)
        gen   = gen_by_z.get(z, 0)
        exp_  = exports_by_z.get(z, 0)
        imp_  = imports_by_z.get(z, 0)
        price = price_by_z.get(z, 0)
        zone_lats.append(lat); zone_lons.append(lon); zone_names.append(z)
        zone_hovers.append(
            f"<b>{z}</b><br>"
            f"Capacity: {capa/1000:.1f} GW<br>"
            f"Demand: {dem/1000:.1f} TWh<br>"
            f"Generation: {gen/1000:.1f} TWh<br>"
            f"Exports: {exp_:,.0f} GWh<br>"
            f"Imports: {imp_:,.0f} GWh<br>"
            f"Price: {price:.1f} USD/MWh<extra></extra>"
        )

    if zone_lats:
        prices_list = [price_by_z.get(z, 0) for z in zone_names]
        p_min = min(prices_list) if prices_list else 0
        p_max = max(prices_list) if prices_list else 1
        map_fig.add_trace(go.Scattergeo(
            lat=zone_lats, lon=zone_lons,
            mode="markers+text",
            text=zone_names, textposition="top right",
            textfont=dict(size=9, color="#555555"),
            marker=dict(
                size=10, color=prices_list,
                colorscale="Blues", cmin=p_min, cmax=p_max,
                showscale=True,
                colorbar=dict(title=dict(text="Price<br>($/MWh)", side="right"),
                              thickness=12, len=0.40, x=0.01, xanchor="left",
                              y=0.15, yanchor="bottom", tickfont=dict(size=8)),
                line=dict(color="white", width=1),
            ),
            hovertemplate=zone_hovers, showlegend=False,
        ))

    map_fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(
            colorscale="YlOrRd", cmin=0, cmax=100, color=[50], showscale=True,
            colorbar=dict(title=dict(text="Utilization (%)", side="right"),
                          thickness=12, len=0.55, x=1.0,
                          tickvals=[0, 25, 50, 75, 100],
                          ticktext=["0%", "25%", "50%", "75%", "100%"]),
        ),
        showlegend=False, hoverinfo="skip",
    ))

    all_lats = [lat for lat, lon in zone_coords.values()]
    all_lons = [lon for lat, lon in zone_coords.values()]
    lat_pad = max(3, (max(all_lats) - min(all_lats)) * 0.15)
    lon_pad = max(3, (max(all_lons) - min(all_lons)) * 0.15)
    map_fig.update_geos(
        showcoastlines=True, coastlinecolor="#cccccc",
        showcountries=True,  countrycolor="#dddddd",
        showland=True,       landcolor="#f5f5f5",
        showframe=False,     projection_type="mercator",
        lataxis_range=[min(all_lats) - lat_pad, max(all_lats) + lat_pad],
        lonaxis_range=[min(all_lons) - lon_pad, max(all_lons) + lon_pad],
    )
    map_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white", geo_bgcolor="#e8f4fd",
    )
    return map_fig, bar_fig


@callback(
    Output("ov-dispatch",       "figure"),
    Output("ov-dispatch-title", "children"),
    Input("ov-map",             "clickData"),
    State("filter-run",         "value"),
    State("ov-scenario",        "value"),
    State("ov-year",            "value"),
)
def update_ov_dispatch(click, run, scenario, year):
    empty = go.Figure().update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    default_title = "Annual Dispatch — click a zone on the map"
    if not click or not run or not scenario or not year:
        return empty, default_title

    point = click["points"][0]
    zone = point.get("text")
    if not zone:
        return empty, default_title

    zone_coords = dl.load_zone_coords(run)
    if zone not in zone_coords:
        return empty, default_title

    dispatch_df = dl.load_dispatch_merged(run, [scenario])
    if dispatch_df.empty:
        return empty, f"Annual Dispatch — {zone} (no dispatch data)"

    price_df    = dl.load_hourly_price_merged(run, [scenario])
    day_weights = dl.load_phours_merged(run)
    fig = _dispatch_annual_fig(
        dispatch_df, zone, scenario, year,
        price_df=price_df if not price_df.empty else None,
        day_weights=day_weights if day_weights else None,
    )
    return fig, f"Annual Dispatch — {zone}  |  {scenario}  |  {int(year)}"
