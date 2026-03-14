"""Results — Costs page."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from components.charts import cost_bar, line_chart, empty_fig


def layout(run, scenarios, zones, years):
    if not run or not scenarios:
        return dbc.Alert("Select a run and scenario.", color="info")

    df_cost = dl.get_cost_summary(run, scenarios)

    if zones:
        df_cost = df_cost[df_cost["z"].isin(zones)]
    if years and len(years) == 2:
        df_cost = df_cost[df_cost["y"].between(years[0], years[1])]

    # NPV total per scenario
    if not df_cost.empty:
        df_npv = df_cost.groupby(
            ["scenario"] if "scenario" in df_cost.columns else ["uni"],
            as_index=False
        )["value"].sum()
        fig_npv = line_chart(
            df_npv,
            x_col="scenario" if "scenario" in df_npv.columns else "uni",
            y_col="value",
            color_col="scenario" if "scenario" in df_npv.columns else "uni",
            title="Total System Cost by Scenario ($M)",
            y_label="$M",
            markers=True,
        )
    else:
        fig_npv = empty_fig()

    return html.Div([
        html.H4("System Costs", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="Cost Breakdown over Time", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(
                        figure=cost_bar(df_cost,
                                        title="System Cost Components by Year"),
                        config={"displayModeBar": True}
                    ), width=12),
                ]),
            ]),

            dbc.Tab(label="Scenario Comparison", children=[
                dbc.Row(className="mt-3", children=[
                    dbc.Col(dcc.Graph(figure=fig_npv,
                                      config={"displayModeBar": True}), width=12),
                ]),
            ]),
        ]),
    ])
