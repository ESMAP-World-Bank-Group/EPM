"""Results — Energy mix page."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import area_fuel, pie_fuel, re_share_line, stacked_bar_fuel


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and scenario.", color="info")

    df_nrg = dl.get_energy(run, scenarios)
    sc0    = scenarios[0]

    if zones:
        df_nrg = df_nrg[df_nrg["z"].isin(zones)]
    if years and len(years) == 2:
        df_nrg = df_nrg[df_nrg["y"].between(years[0], years[1])]

    df_re  = dl.get_re_share(run, sc0, zones or None)
    last_y = int(df_nrg["y"].max()) if not df_nrg.empty else 2030

    return html.Div([
        html.H4("Energy Mix", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="Generation over Time", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=stacked_bar_fuel(
                            df_nrg, x_col="y", y_label="GWh",
                            title="Total Generation by Fuel",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label="Area Chart", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=area_fuel(
                            df_nrg, title="Generation Mix (Area)",
                            y_label="GWh",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label=f"Snapshot — {last_y}", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=pie_fuel(df_nrg, year=last_y,
                                        title=f"Energy Mix {last_y}"),
                        config={"displayModeBar": True}
                    ), width=6),
                    dbc.Col(dcc.Graph(
                        figure=re_share_line(df_re,
                                             title="Renewable Share (%) by Zone"),
                        config={"displayModeBar": True}
                    ), width=6),
                ]),
            ]),

            dbc.Tab(label="By Zone", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=stacked_bar_fuel(
                            df_nrg, x_col="y", y_label="GWh",
                            facet_col="z",
                            title="Generation by Fuel per Zone",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),
        ]),
    ])
