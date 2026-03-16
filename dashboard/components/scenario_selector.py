"""
Shared scenario selector component for input editor pages.

Usage in a page:
    from components.scenario_selector import make_selector, scenario_info_bar

    # In layout():
    make_selector("supply", folders, default_folder)

    # The selector exposes two IDs per page_prefix:
    #   "{prefix}-project"  — folder dropdown
    #   "{prefix}-scenario" — scenario dropdown
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import data_loader as dl


def make_selector(prefix: str, folders: list[str], default_folder: str | None) -> dbc.Card:
    """
    Returns a Card with project + scenario dropdowns.
    IDs: {prefix}-project, {prefix}-scenario
    """
    options = [{"label": f, "value": f} for f in folders]

    scenarios = dl.get_input_scenario_names(default_folder) if default_folder else []
    sc_options = [{"label": s, "value": s} for s in scenarios]
    default_sc = scenarios[0] if scenarios else None

    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project / Data folder", className="fw-semibold small"),
                    dcc.Dropdown(
                        id=f"{prefix}-project",
                        options=options,
                        value=default_folder,
                        clearable=False,
                        className="small",
                    ),
                ], width=3),

                dbc.Col([
                    dbc.Label("Scenario", className="fw-semibold small"),
                    dcc.Dropdown(
                        id=f"{prefix}-scenario",
                        options=sc_options,
                        value=default_sc,
                        clearable=True,
                        placeholder="Default (no override)",
                        className="small",
                    ),
                ], width=3),

                dbc.Col(
                    html.Div(id=f"{prefix}-scenario-info"),
                    width=6,
                    className="d-flex align-items-end pb-1",
                ),
            ], align="end"),
        ),
        className="mb-3 border-0 shadow-sm bg-light",
    )


def register_scenario_update_callback(prefix: str):
    """
    Register a callback that updates the scenario dropdown options
    when the project changes, and shows info about overrides.
    Call once per page prefix at module level.
    """
    @callback(
        Output(f"{prefix}-scenario",      "options"),
        Output(f"{prefix}-scenario",      "value"),
        Output(f"{prefix}-scenario-info", "children"),
        Input(f"{prefix}-project",        "value"),
        Input(f"{prefix}-scenario",       "value"),
    )
    def _update(folder, scenario):
        if not folder:
            return [], None, ""

        scenarios = dl.get_input_scenario_names(folder)
        options   = [{"label": s, "value": s} for s in scenarios]

        # Keep selected scenario if still valid, else reset to first
        if scenario not in scenarios:
            scenario = scenarios[0] if scenarios else None

        # Build info badge
        info = _build_info(folder, scenario)
        return options, scenario, info

    return _update


def _build_info(folder: str, scenario: str | None) -> html.Div:
    """Build the info badges shown next to the dropdowns."""
    if not folder:
        return html.Div()

    # Check if scenarios.csv even exists for this folder
    sc_data = dl.load_input_scenarios(folder)
    if not sc_data:
        return html.Div([
            dbc.Badge("No scenarios.csv", color="secondary", className="me-1"),
            html.Small("Add a scenarios.csv to this folder to enable scenario switching.",
                       className="text-muted ms-1"),
        ])

    if not scenario:
        return html.Div([
            dbc.Badge("Default inputs", color="secondary", className="me-1"),
            html.Small("No scenario override active — editing base folder files.",
                       className="text-muted ms-1"),
        ])

    # Count overrides for this scenario
    sc_data = dl.load_input_scenarios(folder)
    overrides = {k: v for k, v in sc_data.get(scenario, {}).items() if v}
    n = len(overrides)

    badges = [
        dbc.Badge(scenario, color="primary", className="me-1 fs-6"),
        dbc.Badge(f"{n} override{'s' if n != 1 else ''}",
                  color="warning" if n > 0 else "secondary",
                  className="me-2"),
    ]

    if overrides:
        override_list = html.Ul([
            html.Li(html.Small(f"{param} → {path}", className="font-monospace"),
                    className="text-muted")
            for param, path in list(overrides.items())[:5]
        ], className="mb-0 ps-3 mt-1")
        badges.append(override_list)

    return html.Div(badges)
