"""Run Configuration — build the epm.py command to run."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ctx, no_update
import data_loader as dl
from config import INPUT_ROOT


from pathlib import Path as _Path

def _badge(label, color="secondary"):
    return dbc.Badge(label, color=color, className="me-1")


def _env_info_banner():
    """Small info strip showing conda env + working directory used for the run."""
    epm_dir = (_Path(__file__).parent.parent.parent / "epm").resolve()
    return dbc.Alert([
        html.Span("Conda env: ", className="fw-semibold"),
        html.Code("gams_env"),
        html.Span("   |   Working dir: ", className="fw-semibold ms-3"),
        html.Code(str(epm_dir)),
    ], color="light", className="py-1 px-2 small mb-0 border")


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)

    # Load available scenarios for the default project
    sc_names = _get_scenario_names(default)

    return html.Div([
        html.H4("Run Configuration", className="mb-1"),
        html.P("Configure epm.py run options, then proceed to Launch.",
               className="text-muted mb-3"),

        # Hidden project store
        html.Div(
            dcc.Dropdown(
                id="rc-project",
                options=[{"label": f, "value": f} for f in folders],
                value=default, clearable=False,
            ),
            style={"display": "none"},
        ),

        dbc.Row([
            # ── Left column: main options ──────────────────────────────
            dbc.Col([

                # Scenarios
                dbc.Card([
                    dbc.CardHeader(html.Strong("Scenarios")),
                    dbc.CardBody([
                        html.P("Select which scenarios to run (defined in the Scenario Builder). "
                               "Leave empty to run all.",
                               className="text-muted small mb-2"),
                        dcc.Dropdown(
                            id="rc-scenarios",
                            options=[{"label": s, "value": s} for s in sc_names],
                            value=[],
                            multi=True,
                            placeholder="All scenarios",
                            className="small",
                        ),
                        html.Div(id="rc-scenarios-info", className="small text-muted mt-1"),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

                # Model type
                dbc.Card([
                    dbc.CardHeader(html.Strong("Model Type")),
                    dbc.CardBody([
                        html.P("MIP: full integer (default, slower). "
                               "RMIP: relaxed/continuous (faster, less precise).",
                               className="text-muted small mb-2"),
                        dbc.RadioItems(
                            id="rc-modeltype",
                            options=[
                                {"label": "MIP  (default)",  "value": "MIP"},
                                {"label": "RMIP (relaxed)",  "value": "RMIP"},
                            ],
                            value="MIP",
                            inline=True,
                        ),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

                # Parallelism
                dbc.Card([
                    dbc.CardHeader(html.Strong("Parallelism")),
                    dbc.CardBody([
                        html.P("Number of CPU cores to use (one per scenario in parallel).",
                               className="text-muted small mb-2"),
                        dbc.Row([
                            dbc.Col(dbc.Label("CPUs:"), width="auto"),
                            dbc.Col(
                                dcc.Slider(id="rc-cpu", min=1, max=16, step=1, value=1,
                                           marks={i: str(i) for i in [1,2,4,8,16]},
                                           tooltip={"placement": "bottom"}),
                                width=8,
                            ),
                        ], align="center"),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

                # Analysis mode
                dbc.Card([
                    dbc.CardHeader(html.Strong("Analysis Mode")),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="rc-analysis",
                            options=[
                                {"label": "Standard",         "value": "standard"},
                                {"label": "Sensitivity",      "value": "sensitivity"},
                                {"label": "Monte Carlo",      "value": "montecarlo"},
                            ],
                            value="standard",
                            inline=True,
                        ),
                        html.Div(id="rc-analysis-extra", className="mt-2"),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

            ], width=6),

            # ── Right column: output & advanced ───────────────────────
            dbc.Col([

                # Output options
                dbc.Card([
                    dbc.CardHeader(html.Strong("Output Options")),
                    dbc.CardBody([
                        dbc.Checklist(
                            id="rc-output-opts",
                            options=[
                                {"label": "Reduced output (fewer CSV files)",
                                 "value": "reduced_output"},
                                {"label": "Zip output folder after run",
                                 "value": "output_zip"},
                                {"label": "Reduced yearly CSV definitions (Tableau)",
                                 "value": "reduce_definition_csv"},
                            ],
                            value=[],
                            switch=True,
                        ),
                        html.Hr(className="my-2"),
                        dbc.Label("Simulation label (optional)", className="small"),
                        dbc.Input(id="rc-label", placeholder="e.g. v1_baseline_run",
                                  size="sm", className="mb-1"),
                        html.Small("Defaults to a timestamp-based name if blank.",
                                   className="text-muted"),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

                # Advanced
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Button("Advanced options", id="rc-advanced-toggle",
                                   color="link", size="sm", className="p-0 text-decoration-none fw-semibold"),
                    ),
                    dbc.Collapse(
                        dbc.CardBody([
                            dbc.Checklist(
                                id="rc-advanced-opts",
                                options=[
                                    {"label": "Simple mode (relax DiscreteCap, useful for testing)",
                                     "value": "simple"},
                                    {"label": "Debug mode (verbose GAMS log)",
                                     "value": "debug"},
                                    {"label": "Trace mode (GAMS trace file)",
                                     "value": "trace"},
                                ],
                                value=[],
                                switch=True,
                            ),
                        ]),
                        id="rc-advanced-collapse",
                        is_open=False,
                    ),
                ], className="mb-3 shadow-sm border-0"),

                # Generated command
                dbc.Card([
                    dbc.CardHeader(html.Strong("Generated Command")),
                    dbc.CardBody([
                        _env_info_banner(),
                        html.Pre(id="rc-command",
                                 className="bg-dark text-white p-2 rounded small mt-2",
                                 style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
                        dbc.Button("Copy", id="rc-copy-btn", size="sm",
                                   color="outline-secondary", className="mt-1"),
                        html.Span(id="rc-copy-msg", className="small text-success ms-2"),
                    ]),
                ], className="mb-3 shadow-sm border-0"),

            ], width=6),
        ]),

        html.Hr(),
        dbc.Row([
            dbc.Col(
                dcc.Link(dbc.Button("Next → Launch Run", color="primary"),
                         href="/run-launch"),
                width="auto",
            ),
        ]),
    ])


def _get_scenario_names(folder):
    """Return list of scenario names from scenarios.csv for a project."""
    if not folder:
        return []
    try:
        sc_data = dl.load_input_scenarios(folder)
        return list(sc_data.keys())
    except Exception:
        return []


# ── Toggle advanced collapse ──────────────────────────────────────────────
@callback(
    Output("rc-advanced-collapse", "is_open"),
    Input("rc-advanced-toggle",    "n_clicks"),
    State("rc-advanced-collapse",  "is_open"),
    prevent_initial_call=True,
)
def toggle_advanced(n, is_open):
    return not is_open


# ── Reload scenario list when project changes ─────────────────────────────
@callback(
    Output("rc-scenarios",     "options"),
    Output("rc-scenarios",     "value"),
    Output("rc-scenarios-info","children"),
    Input("rc-project",        "value"),
)
def update_scenarios(folder):
    names = _get_scenario_names(folder)
    opts  = [{"label": s, "value": s} for s in names]
    if not names:
        info = html.Small("No scenarios.csv found — define scenarios in Scenario Builder first.",
                          className="text-warning")
    else:
        info = html.Small(f"{len(names)} scenario(s) available.", className="text-muted")
    return opts, [], info


# ── Show/hide analysis mode extras ───────────────────────────────────────
@callback(
    Output("rc-analysis-extra", "children"),
    Input("rc-analysis", "value"),
)
def analysis_extra(mode):
    if mode == "sensitivity":
        return dbc.Alert(
            ["Sensitivity analysis reads ", html.Code("sensitivity.csv"),
             " in the project folder. Each row is a parameter with ±% perturbations."],
            color="info", className="small mb-0 py-2",
        )
    if mode == "montecarlo":
        return dbc.Row([
            dbc.Col(dbc.Label("Samples", className="small"), width="auto"),
            dbc.Col(dbc.Input(id="rc-mc-samples", type="number", value=10,
                              min=2, max=1000, size="sm", style={"width": "90px"}),
                    width="auto"),
            dbc.Col(html.Small("Reads uncertainties.csv from project folder.",
                               className="text-muted"), width="auto"),
        ], align="center", className="g-2")
    return ""


# ── Build command string ──────────────────────────────────────────────────
@callback(
    Output("rc-command", "children"),
    Input("rc-project",      "value"),
    Input("rc-scenarios",    "value"),
    Input("rc-modeltype",    "value"),
    Input("rc-cpu",          "value"),
    Input("rc-analysis",     "value"),
    Input("rc-output-opts",  "value"),
    Input("rc-label",        "value"),
    Input("rc-advanced-opts","value"),
)
def build_command(folder, sc_selected, modeltype, cpu, analysis, output_opts, label, advanced):
    if not folder:
        return "# Select a project first"

    parts = ["python epm.py"]
    parts.append(f"--folder_input {folder}")

    if modeltype and modeltype != "MIP":
        parts.append(f"--modeltype {modeltype}")

    if cpu and cpu > 1:
        parts.append(f"--cpu {cpu}")

    # Scenarios
    parts.append("--scenarios scenarios.csv")
    if sc_selected:
        parts.append("--selected_scenarios " + " ".join(sc_selected))

    # Analysis mode
    if analysis == "sensitivity":
        parts.append("--sensitivity")
    elif analysis == "montecarlo":
        parts.append("--montecarlo")

    # Output options
    output_opts = output_opts or []
    for opt in output_opts:
        parts.append(f"--{opt}")

    # Label
    if label and label.strip():
        parts.append(f"--simulation_label {label.strip()}")

    # Advanced
    advanced = advanced or []
    if "simple" in advanced:
        parts.append("--simple")
    if "debug" in advanced:
        parts.append("--debug")
    if "trace" in advanced:
        parts.append("--trace")

    return " \\\n  ".join(parts)


# ── Persist configuration to session store ───────────────────────────────────
@callback(
    Output("run-config-store", "data"),
    Input("rc-project",       "value"),
    Input("rc-scenarios",     "value"),
    Input("rc-modeltype",     "value"),
    Input("rc-cpu",           "value"),
    Input("rc-analysis",      "value"),
    Input("rc-output-opts",   "value"),
    Input("rc-label",         "value"),
    Input("rc-advanced-opts", "value"),
)
def save_config(folder, scenarios, modeltype, cpu, analysis, output_opts, label, advanced):
    return {
        "folder":      folder,
        "scenarios":   scenarios or [],
        "modeltype":   modeltype or "MIP",
        "cpu":         cpu or 1,
        "analysis":    analysis or "standard",
        "output_opts": output_opts or [],
        "label":       (label or "").strip(),
        "advanced":    advanced or [],
    }
