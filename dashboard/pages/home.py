"""Home page — project selector."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, ctx
import data_loader as dl


def layout(*args):
    folders = dl.list_input_folders()

    folder_cards = []
    for f in folders:
        folder_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f, className="card-title"),
                        html.Small(f"epm/input/{f}", className="text-muted d-block mb-2"),
                        dbc.Button("Open →", id={"type": "open-project", "index": f},
                                   color="primary", size="sm"),
                    ])
                ], className="h-100 shadow-sm"),
                width=3, className="mb-3",
            )
        )

    return html.Div([
        html.H3("EPM — Electricity Planning Model", className="mb-1"),
        html.P("Select a project to start editing inputs or viewing results.",
               className="text-muted mb-4"),

        dbc.Row([
            dbc.Col(html.H5("Projects"), width=12),
            *folder_cards,
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col(html.H5("Create New Project"), width=12),
            dbc.Col([
                dbc.Label("Clone from existing project"),
                dcc.Dropdown(
                    id="clone-source",
                    options=[{"label": f, "value": f} for f in folders],
                    placeholder="Select a source folder…",
                    className="mb-2",
                ),
                dbc.Input(id="clone-name",
                          placeholder="New project name (e.g. data_namibia)",
                          className="mb-2"),
                dbc.Button("Create Project", id="clone-btn", color="success"),
                html.Div(id="clone-result", className="mt-2"),
            ], width=4),
        ]),
    ])


@callback(
    Output("store-active-project", "data"),
    Output("url",                  "pathname"),
    Input({"type": "open-project", "index": ALL}, "n_clicks"),
    State({"type": "open-project", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def open_project(n_clicks_list, ids):
    if not any(n for n in n_clicks_list if n):
        return no_update, no_update
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        project = triggered["index"]
        return project, "/input-settings"
    return no_update, no_update


@callback(
    Output("clone-result", "children"),
    Input("clone-btn",     "n_clicks"),
    State("clone-source",  "value"),
    State("clone-name",    "value"),
    prevent_initial_call=True,
)
def clone_project(n, source, name):
    if not source or not name:
        return dbc.Alert("Please select a source and enter a name.", color="warning")
    name = name.strip()
    if not name.startswith("data_"):
        name = "data_" + name
    ok = dl.clone_input_folder(source, name)
    if ok:
        return dbc.Alert(f"Project '{name}' created successfully.", color="success")
    return dbc.Alert("Failed — folder already exists or source not found.", color="danger")
