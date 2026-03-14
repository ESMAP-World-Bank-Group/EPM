"""Results — Overview page: KPI cards + summary charts."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import (
    stacked_bar_fuel, pie_fuel, line_chart, re_share_line, empty_fig, kpi_value
)


def _kpi_card(title: str, value: str, unit: str, icon: str, color: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className=f"bi {icon} fs-3 text-{color}"),
                    html.Div([
                        html.P(title, className="text-muted small mb-0"),
                        html.H4(value, className="mb-0 fw-bold"),
                        html.Small(unit, className="text-muted"),
                    ], className="ms-3"),
                ], className="d-flex align-items-center"),
            ])
        ], className="shadow-sm border-0 h-100"),
        width=3, className="mb-3",
    )


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and at least one scenario using the filters above.",
                         color="info")

    sc0    = scenarios[0]
    kpis   = dl.get_kpis(run, sc0)
    df_cap = dl.get_capacity(run, scenarios)
    df_nrg = dl.get_energy(run, scenarios)
    df_re  = dl.get_re_share(run, sc0, zones or None)

    # Filter by zone if selected
    if zones:
        df_cap = df_cap[df_cap["z"].isin(zones)]
        df_nrg = df_nrg[df_nrg["z"].isin(zones)]

    # Filter by year range
    if years and len(years) == 2:
        df_cap = df_cap[df_cap["y"].between(years[0], years[1])]
        df_nrg = df_nrg[df_nrg["y"].between(years[0], years[1])]

    # KPI values
    npv  = kpis.get("npv")
    npv_str = f"{npv/1000:,.1f} B$" if npv else "—"
    cap  = kpis.get("total_capacity")
    cap_str = f"{cap/1000:,.1f} GW" if cap else "—"
    em   = kpis.get("total_emissions")
    em_str = f"{em:,.0f} Mt" if em else "—"
    inv  = kpis.get("total_investment")
    inv_str = f"{inv/1000:,.1f} B$" if inv else "—"

    # Last year in data for pie
    last_year = int(df_nrg["y"].max()) if not df_nrg.empty else 2030

    return html.Div([
        html.H4(f"Overview — {sc0}", className="mb-3"),

        # KPI row
        dbc.Row([
            _kpi_card("System NPV",       npv_str, "discounted", "bi-currency-dollar", "primary"),
            _kpi_card("Capacity Added",   cap_str, "total new",  "bi-lightning",        "success"),
            _kpi_card("Total Emissions",  em_str,  "CO₂",        "bi-cloud-haze2",      "danger"),
            _kpi_card("Total Investment", inv_str, "undiscounted","bi-bank",             "warning"),
        ]),

        html.Hr(),

        # Charts row
        dbc.Row([
            dbc.Col([
                html.H6("Installed Capacity by Fuel", className="text-muted"),
                dcc.Graph(
                    figure=stacked_bar_fuel(
                        df_cap, x_col="y", y_label="MW",
                        title="Capacity by Fuel (all zones)",
                    ),
                    config={"displayModeBar": True},
                ),
            ], width=6),

            dbc.Col([
                html.H6(f"Energy Mix — {last_year}", className="text-muted"),
                dcc.Graph(
                    figure=pie_fuel(df_nrg, year=last_year,
                                    title=f"Generation Mix {last_year}"),
                    config={"displayModeBar": True},
                ),
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.H6("Renewable Energy Share (%)", className="text-muted"),
                dcc.Graph(
                    figure=re_share_line(df_re),
                    config={"displayModeBar": True},
                ),
            ], width=12),
        ], className="mt-3"),
    ])
