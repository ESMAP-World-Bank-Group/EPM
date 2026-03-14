"""Results — Emissions page."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import emissions_line, line_chart, stacked_bar_fuel, empty_fig


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and scenario.", color="info")

    df_em  = dl.get_emissions(run, scenarios)
    df_int = dl.get_emissions_intensity(run, scenarios)

    if zones:
        df_em  = df_em[df_em["z"].isin(zones)]
        df_int = df_int[df_int["z"].isin(zones)]
    if years and len(years) == 2:
        df_em  = df_em[df_em["y"].between(years[0], years[1])]
        df_int = df_int[df_int["y"].between(years[0], years[1])]

    color_col = "scenario" if "scenario" in df_em.columns else "z"
    color_int = "scenario" if "scenario" in df_int.columns else "z"

    fig_em  = emissions_line(df_em,  zones=zones or None)
    fig_int = (line_chart(df_int, x_col="y", y_col="value",
                          color_col=color_int,
                          title="Emissions Intensity by Zone",
                          y_label="gCO₂/kWh")
               if not df_int.empty else empty_fig())

    # Bar by zone for last year
    last_y = int(df_em["y"].max()) if not df_em.empty else 2030
    df_bar = df_em[df_em["y"] == last_y].groupby(["z", "scenario"] if "scenario" in df_em.columns else ["z"],
                                                   as_index=False)["value"].sum()

    return html.Div([
        html.H4("CO₂ Emissions", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="Emissions over Time", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(figure=fig_em,
                                      config={"displayModeBar": True}), width=12),
                ]),
            ]),

            dbc.Tab(label="Emissions Intensity", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(figure=fig_int,
                                      config={"displayModeBar": True}), width=12),
                ]),
            ]),

            dbc.Tab(label=f"By Zone — {last_y}", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=line_chart(
                            df_bar, x_col="z", y_col="value",
                            color_col="scenario" if "scenario" in df_bar.columns else "z",
                            title=f"Emissions by Zone ({last_y})",
                            y_label="Mt CO₂",
                            markers=False,
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),
        ]),
    ])
