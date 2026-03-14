"""Run Manager — Scenario builder."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ALL
import pandas as pd
import data_loader as dl
from config import INPUT_ROOT


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Scenario Builder", className="mb-1"),
        html.P("Define which scenarios to run. Each scenario maps to a data folder "
               "(and optionally overrides specific input files).",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Base project"),
                dcc.Dropdown(id="sc-base-project", options=options,
                             value=default, clearable=False),
            ], width=3),
            dbc.Col([
                dbc.Label("\u00a0"),
                dbc.Button("+ Add Scenario", id="add-scenario-btn",
                           color="outline-primary", size="sm"),
            ], width=2),
        ], className="mb-4 align-items-end"),

        html.Div(id="scenarios-container", children=[
            _scenario_row(0, default),
        ]),

        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Button("Save scenarios.csv", id="save-scenarios-btn",
                               color="success"), width="auto"),
            dbc.Col(html.Div(id="save-scenarios-msg"), width="auto"),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Link(
                dbc.Button("Next → Launch Run", color="primary"),
                href="/run-launch"
            ), width="auto"),
        ]),
    ])


def _scenario_row(idx: int, default_folder: str | None) -> dbc.Card:
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    return dbc.Card(
        dbc.CardBody(dbc.Row([
            dbc.Col([
                dbc.Label("Scenario name"),
                dbc.Input(id={"type": "sc-name", "index": idx},
                          value="baseline" if idx == 0 else f"scenario_{idx}",
                          placeholder="e.g. high_carbon"),
            ], width=3),
            dbc.Col([
                dbc.Label("Input folder"),
                dcc.Dropdown(id={"type": "sc-folder", "index": idx},
                             options=options, value=default_folder, clearable=False),
            ], width=3),
            dbc.Col([
                dbc.Label("Description (optional)"),
                dbc.Input(id={"type": "sc-desc", "index": idx},
                          placeholder="e.g. Carbon price $50/t from 2030"),
            ], width=5),
            dbc.Col([
                dbc.Label("\u00a0"),
                dbc.Button("✕", id={"type": "sc-remove", "index": idx},
                           color="outline-danger", size="sm"),
            ], width=1),
        ])),
        className="mb-2 shadow-sm border-0",
        id={"type": "sc-card", "index": idx},
    )


@callback(
    Output("scenarios-container", "children"),
    Input("add-scenario-btn",     "n_clicks"),
    State("scenarios-container",  "children"),
    State("sc-base-project",      "value"),
    prevent_initial_call=True,
)
def add_scenario(n, current_rows, base):
    n_existing = len(current_rows)
    current_rows.append(_scenario_row(n_existing, base))
    return current_rows


@callback(
    Output("save-scenarios-msg", "children"),
    Input("save-scenarios-btn",  "n_clicks"),
    State({"type": "sc-name",   "index": ALL}, "value"),
    State({"type": "sc-folder", "index": ALL}, "value"),
    State({"type": "sc-desc",   "index": ALL}, "value"),
    State("sc-base-project",    "value"),
    prevent_initial_call=True,
)
def save_scenarios(n, names, folders, descs, base_folder):
    if not names or not base_folder:
        return dbc.Badge("Nothing to save", color="warning")
    rows = []
    for name, folder, desc in zip(names, folders, descs or [""]*len(names)):
        if name and folder:
            rows.append({"scenario": name, "folder_input": folder,
                         "description": desc or ""})
    if not rows:
        return dbc.Badge("No valid scenarios", color="warning")
    df = pd.DataFrame(rows)
    path = INPUT_ROOT / base_folder / "scenarios_dashboard.csv"
    try:
        df.to_csv(path, index=False)
        return dbc.Badge(f"Saved to {path.name} ✓", color="success")
    except Exception as e:
        return dbc.Badge(f"Error: {e}", color="danger")
