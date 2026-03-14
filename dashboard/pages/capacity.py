"""Results — Capacity expansion page."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import stacked_bar_fuel, line_chart, empty_fig


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and scenario.", color="info")

    df_cap = dl.get_capacity(run, scenarios)
    df_new = dl.get_new_capacity(run, scenarios)

    if zones:
        df_cap = df_cap[df_cap["z"].isin(zones)]
        df_new = df_new[df_new["z"].isin(zones)]
    if years and len(years) == 2:
        df_cap = df_cap[df_cap["y"].between(years[0], years[1])]
        df_new = df_new[df_new["y"].between(years[0], years[1])]

    # Total capacity over time per scenario
    if not df_cap.empty and "scenario" in df_cap.columns:
        df_total = df_cap.groupby(["y", "scenario"], as_index=False)["value"].sum()
        fig_total = line_chart(df_total, x_col="y", y_col="value",
                               color_col="scenario",
                               title="Total Installed Capacity by Scenario",
                               y_label="MW")
    else:
        fig_total = empty_fig()

    return html.Div([
        html.H4("Capacity Expansion", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="Total Capacity", children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(
                        figure=stacked_bar_fuel(
                            df_cap, x_col="y", y_label="MW",
                            title="Installed Capacity by Fuel",
                        ), config={"displayModeBar": True}
                    ), width=8),
                    dbc.Col(dcc.Graph(
                        figure=fig_total,
                        config={"displayModeBar": True}
                    ), width=4),
                ], className="mt-3"),
            ]),

            dbc.Tab(label="By Zone", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=stacked_bar_fuel(
                            df_cap, x_col="y", y_label="MW",
                            facet_col="z",
                            title="Capacity by Fuel per Zone",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label="New Investments", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=stacked_bar_fuel(
                            df_new, x_col="y", y_label="MW",
                            title="New Capacity Added by Fuel",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),
        ]),
    ])
