"""Results — Trade page."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import sankey_trade, heatmap_interchange, line_chart, empty_fig


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and scenario.", color="info")

    sc0        = scenarios[0]
    df_trade   = dl.get_interchange(run, scenarios)

    last_y = None
    if not df_trade.empty and "y" in df_trade.columns:
        if years and len(years) == 2:
            df_trade = df_trade[df_trade["y"].between(years[0], years[1])]
        last_y = int(df_trade["y"].max()) if not df_trade.empty else None

    # Line chart of net trade per zone over time
    if not df_trade.empty and {"z", "y", "value"}.issubset(df_trade.columns):
        color_col = "scenario" if "scenario" in df_trade.columns else "z"
        df_line = df_trade.groupby(
            ["z", "y"] + (["scenario"] if "scenario" in df_trade.columns else []),
            as_index=False
        )["value"].sum()
        fig_line = line_chart(df_line, x_col="y", y_col="value",
                              color_col=color_col,
                              title="Net Interchange by Zone",
                              y_label="GWh")
    else:
        fig_line = empty_fig()

    return html.Div([
        html.H4("Electricity Trade & Interchange", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="Flow Diagram (Sankey)", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=sankey_trade(
                            df_trade[df_trade.get("scenario", sc0) == sc0]
                            if "scenario" in df_trade.columns else df_trade,
                            year=last_y or 2030,
                            title=f"Trade Flows — {last_y}",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label="Heatmap", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=heatmap_interchange(
                            df_trade, year=last_y or 2030, scenario=sc0,
                            title=f"Interchange Matrix — {last_y}",
                        ), config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label="Over Time", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(figure=fig_line,
                                      config={"displayModeBar": True}), width=12),
                ]),
            ]),
        ]),
    ])
