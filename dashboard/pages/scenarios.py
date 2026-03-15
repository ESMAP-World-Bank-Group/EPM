"""Scenario Builder — combine input variants into named scenarios."""

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
import data_loader as dl
from config import INPUT_CSV


# ---------------------------------------------------------------------------
# Parameters shown in the builder, grouped by section
# ---------------------------------------------------------------------------

PARAM_GROUPS = [
    ("Settings", [
        ("settings_input", "Model Settings"),
    ]),
    ("Load", [
        ("demand_forecast", "Demand Forecast"),
        ("efficiency",      "Energy Efficiency Factor"),
    ]),
    ("Supply", [
        ("gen_data",     "Generator Data"),
        ("fuel_price",   "Fuel Prices"),
        ("capex",        "CAPEX Trajectories"),
        ("availability", "Availability"),
        ("storage_data", "Storage"),
    ]),
    ("Trade", [
        ("transfer_limit",   "Transfer Limits"),
        ("new_transmission", "New Transmission"),
        ("trade_price",      "Trade Prices"),
        ("ext_transfer",     "External Transfer Limits"),
    ]),
    ("Constraints", [
        ("carbon_price",      "Carbon Price"),
        ("emissions_total",   "Emissions Cap (System)"),
        ("emissions_country", "Emissions Cap (Country)"),
        ("max_fuel",          "Max Fuel Limits"),
    ]),
    ("Reserve", [
        ("planning_reserve", "Planning Reserve Margin"),
        ("spinning_country", "Spinning Reserve (Country)"),
        ("spinning_system",  "Spinning Reserve (System)"),
    ]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_names(folder):
    if not folder:
        return ["Baseline"]
    sc_data = dl.load_input_scenarios(folder)
    names = list(sc_data.keys())
    if "Baseline" not in names:
        names = ["Baseline"] + names
    return names


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)
    sc_names = _scenario_names(default)
    sc_opts = [{"label": s, "value": s} for s in sc_names]

    return html.Div([
        html.H4("Scenario Builder", className="mb-1"),
        html.P(
            "Define which input variant each scenario uses. "
            "Columns = scenarios · Rows = parameters · Cells = which variant to use.",
            className="text-muted mb-3",
        ),

        # Hidden project store
        html.Div(
            dcc.Dropdown(
                id="scb-project",
                options=[{"label": f, "value": f} for f in folders],
                value=default, clearable=False,
            ),
            style={"display": "none"},
        ),

        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                dbc.Label("Scenarios", className="fw-semibold small"),
                dcc.Dropdown(
                    id="scb-scenario-select",
                    options=sc_opts,
                    value=sc_names[0] if sc_names else None,
                    clearable=True,
                    placeholder="All scenarios",
                    className="small",
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("New scenario name", className="fw-semibold small"),
                dbc.Input(id="scb-new-name", placeholder="e.g. HighDemand", size="sm"),
            ], width=3),
            dbc.Col([
                dbc.Label("\u00a0", className="d-block"),
                dbc.Button("+ Add Scenario", id="scb-add-btn",
                           color="outline-primary", size="sm"),
            ], width=2),
            dbc.Col([
                dbc.Label("\u00a0", className="d-block"),
                dbc.Button("Save scenarios.csv", id="scb-save-btn",
                           color="success", size="sm"),
                html.Span(id="scb-save-msg", className="ms-2 small text-success"),
            ], width=4),
        ], align="end")), className="mb-3 border-0 shadow-sm bg-light"),

        html.Div(id="scb-add-msg", className="small text-info mb-2"),
        html.Div(id="scb-table"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("scb-table",           "children"),
    Output("scb-add-msg",         "children"),
    Output("scb-scenario-select", "options"),
    Output("scb-scenario-select", "value"),
    Input("scb-project",          "value"),
    Input("scb-add-btn",          "n_clicks"),
    State("scb-new-name",         "value"),
    State("scb-scenario-select",  "value"),
)
def build_table(folder, add_clicks, new_name, current_selection):
    msg = ""
    selected = current_selection

    # Handle Add Scenario
    if ctx.triggered_id == "scb-add-btn" and add_clicks:
        if folder and new_name and new_name.strip():
            name = new_name.strip()
            ok = dl.add_scenario_column(folder, name)
            if ok:
                msg = f"Added '{name}' ✓"
                selected = name          # auto-select the new scenario
            else:
                msg = f"'{name}' already exists or error."
        else:
            msg = "Enter a project and scenario name first."

    if not folder:
        return dbc.Alert("Select a project.", color="info"), msg, [], None

    sc_data = dl.load_input_scenarios(folder)
    scenario_names = list(sc_data.keys())
    if "Baseline" not in scenario_names:
        scenario_names = ["Baseline"] + scenario_names
    sc_opts = [{"label": s, "value": s} for s in scenario_names]

    # If selected scenario no longer valid, fall back
    if selected not in scenario_names:
        selected = scenario_names[0] if scenario_names else None

    if not scenario_names:
        return dbc.Alert(
            [html.Strong("No scenarios defined. "),
             "Add a scenario name above and click '+ Add Scenario'."],
            color="info",
        ), msg, sc_opts, None

    # Determine which scenarios to show (all, or just the selected one)
    show_names = scenario_names  # always show all columns

    # Header row
    header = dbc.Row(
        [dbc.Col(html.Strong("Parameter"), width=3)] +
        [dbc.Col(
            dbc.Badge(
                sc,
                color="primary" if sc != selected else "success",
                className="w-100 text-wrap",
            ),
            width=True,
         )
         for sc in show_names],
        className="border-bottom pb-2 mb-1 fw-bold",
    )

    rows = [header]
    for group_label, params in PARAM_GROUPS:
        rows.append(
            dbc.Row(dbc.Col(
                html.Small(group_label.upper(),
                           className="text-muted fw-semibold"),
                width=12,
            ), className="mt-3 mb-1")
        )

        for key, label in params:
            variants_dict = dl.list_variants(folder, key)
            var_opts = [{"label": v, "value": v} for v in variants_dict]
            spec = INPUT_CSV.get(key, ("", "x.csv"))
            gams_name = Path(spec[1]).stem

            cells = [
                dbc.Col(
                    html.Small(label, className="text-muted"),
                    width=3,
                    className="d-flex align-items-center",
                )
            ]
            for sc in show_names:
                override_path = sc_data.get(sc, {}).get(gams_name)
                current = dl.override_path_to_variant(folder, key, override_path)
                if current not in variants_dict:
                    current = "Baseline"
                cells.append(dbc.Col(
                    dcc.Dropdown(
                        id={"type": "scb-cell", "key": key, "sc": sc},
                        options=var_opts,
                        value=current,
                        clearable=False,
                        className="small",
                    ),
                    width=True,
                ))
            rows.append(
                dbc.Row(cells, className="py-1 border-bottom align-items-center")
            )

    return html.Div(rows), msg, sc_opts, selected


@callback(
    Output("scb-save-msg", "children"),
    Input("scb-save-btn",  "n_clicks"),
    State("scb-project",   "value"),
    State({"type": "scb-cell", "key": ALL, "sc": ALL}, "value"),
    State({"type": "scb-cell", "key": ALL, "sc": ALL}, "id"),
    prevent_initial_call=True,
)
def save_scenarios(n, folder, cell_values, cell_ids):
    if not folder:
        return "No project selected."
    sc_dict: dict[str, dict[str, str]] = {}
    for val, cid in zip(cell_values, cell_ids):
        sc = cid["sc"]
        key = cid["key"]
        sc_dict.setdefault(sc, {})[key] = val or "Baseline"
    ok = dl.save_input_scenarios(folder, sc_dict)
    return "Saved ✓" if ok else "Failed — check file permissions."
