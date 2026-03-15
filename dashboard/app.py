"""
EPM Dashboard — Main entry point
Run with:  python dashboard/app.py   (from repo root)
           OR cd dashboard && python app.py
"""

import sys
from pathlib import Path

# Ensure dashboard/ is on sys.path so local imports work
sys.path.insert(0, str(Path(__file__).parent))

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

import data_loader as dl
from config import APP_TITLE, THEME_NAME

# ---------------------------------------------------------------------------
# Import page layout functions
# ---------------------------------------------------------------------------
from pages import (
    home,
    overview,
    capacity,
    energy_mix,
    emissions,
    costs,
    trade,
    input_manager,
    input_settings,
    input_resolution,
    input_supply,
    input_demand,
    input_trade,
    input_reserve,
    input_constraints,
    scenarios,
    run_config,
    run_launch,
    run_history,
)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
THEME = getattr(dbc.themes, THEME_NAME)

app = Dash(
    __name__,
    external_stylesheets=[THEME, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,   # pages load dynamically
    title=APP_TITLE,
)
server = app.server   # expose for gunicorn if needed

# ---------------------------------------------------------------------------
# Navigation sidebar
# ---------------------------------------------------------------------------

NAV_ITEMS = [
    ("PROJECT",  [
        ("bi bi-house-fill",          "Home",              "home"),
    ]),
    ("INPUTS",   [
        ("bi bi-folder2-open",        "Manage Inputs",     "input-manager"),
        ("bi bi-gear-fill",           "Settings",          "input-settings"),
        ("bi bi-grid-3x3",            "Resolution",        "input-resolution"),
        ("bi bi-graph-up",            "Demand",            "input-demand"),
        ("bi bi-lightning-charge",    "Supply",            "input-supply"),
        ("bi bi-arrow-left-right",    "Trade",             "input-trade"),
        ("bi bi-shield-check",        "Reserve",           "input-reserve"),
        ("bi bi-sliders",             "Constraints",       "input-constraints"),
        ("bi bi-diagram-3",           "Scenario Builder",  "scenarios"),
    ]),
    ("RUN",      [
        ("bi bi-list-check",          "Run Config",        "run-config"),
        ("bi bi-play-circle-fill",    "Launch",            "run-launch"),
        ("bi bi-clock-history",       "History",           "run-history"),
    ]),
    ("RESULTS",  [
        ("bi bi-speedometer2",        "Overview",          "overview"),
        ("bi bi-bar-chart-fill",      "Capacity",          "capacity"),
        ("bi bi-pie-chart-fill",      "Energy Mix",        "energy-mix"),
        ("bi bi-cloud-haze2",         "Emissions",         "emissions"),
        ("bi bi-currency-dollar",     "Costs",             "costs"),
        ("bi bi-arrow-left-right",    "Trade",             "trade"),
    ]),
]


def make_nav_link(icon: str, label: str, page_id: str) -> html.Li:
    return html.Li(
        dcc.Link(
            [html.I(className=f"{icon} me-2"), label],
            href=f"/{page_id}",
            className="nav-link text-white px-3 py-2",
            id=f"nav-{page_id}",
        ),
        className="nav-item",
    )


def build_sidebar() -> dbc.Col:
    nav_children = []
    for section_title, items in NAV_ITEMS:
        nav_children.append(
            html.Small(section_title,
                       className="text-uppercase text-muted px-3 mt-3 mb-1 d-block",
                       style={"fontSize": "0.7rem", "letterSpacing": "0.08em"})
        )
        nav_children += [make_nav_link(ic, lbl, pid) for ic, lbl, pid in items]

    return dbc.Col(
        [
            # Logo / brand
            html.Div([
                html.Img(src="/assets/logo.png", height="40px",
                         className="me-2",
                         style={"display": "none"},   # hidden until logo added
                         id="sidebar-logo"),
                html.Span("EPM", className="fw-bold fs-3 text-white"),
                html.Br(),
                html.Small("World Bank Group",
                           className="text-muted", style={"fontSize": "0.7rem"}),
            ], className="px-3 py-3 border-bottom border-secondary"),

            # Active project indicator
            html.Div([
                html.Small("Project:", className="text-muted",
                           style={"fontSize": "0.65rem", "letterSpacing": "0.05em"}),
                html.Div(id="sidebar-project-label",
                         className="text-white fw-semibold",
                         style={"fontSize": "0.78rem", "wordBreak": "break-all"}),
            ], className="px-3 py-2 border-bottom border-secondary"),

            # Navigation
            html.Ul(nav_children, className="nav flex-column mt-2 mb-auto"),
        ],
        width=2,
        className="d-flex flex-column bg-dark min-vh-100 py-0",
        style={"position": "sticky", "top": 0, "overflowY": "auto"},
    )


# ---------------------------------------------------------------------------
# Global filter bar (shared across Result pages)
# ---------------------------------------------------------------------------

def build_filter_bar() -> dbc.Card:
    runs = dl.list_runs()
    run_options = [{"label": r, "value": r} for r in runs]
    default_run = runs[0] if runs else None

    # Pre-populate scenarios for default run
    scenarios = dl.list_scenarios(default_run) if default_run else []
    scenario_options = [{"label": s, "value": s} for s in scenarios]

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Simulation run", className="fw-semibold small"),
                    dcc.Dropdown(id="filter-run", options=run_options,
                                 value=default_run, clearable=False,
                                 className="small"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Scenarios", className="fw-semibold small"),
                    dcc.Dropdown(id="filter-scenarios", options=scenario_options,
                                 value=scenarios[:1], multi=True,
                                 className="small"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Zones", className="fw-semibold small"),
                    dcc.Dropdown(id="filter-zones", options=[], value=[],
                                 multi=True, placeholder="All zones",
                                 className="small"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Years", className="fw-semibold small"),
                    dcc.RangeSlider(id="filter-years", min=2020, max=2050,
                                    step=1, value=[2020, 2050],
                                    marks={y: str(y) for y in range(2020, 2055, 5)},
                                    tooltip={"placement": "bottom"}),
                ], width=3),
            ], align="center"),
        ]),
        className="mb-3 border-0 shadow-sm",
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),

    # Store shared state across callbacks
    dcc.Store(id="store-active-project", storage_type="session"),
    dcc.Store(id="store-active-job",     storage_type="session"),
    dcc.Store(id="run-config-store",     storage_type="session"),
    dcc.Store(id="open-file-store",      storage_type="memory"),
    html.Div(id="open-file-dummy",       style={"display": "none"}),

    dbc.Row([
        # Left sidebar
        build_sidebar(),

        # Main content
        dbc.Col([
            # Filter bar — always in DOM, hidden on non-result pages
            html.Div(build_filter_bar(), id="filter-bar-container",
                     style={"display": "none"}),

            # Page content
            html.Div(id="page-content", className="p-3"),
        ], width=10),
    ], className="g-0"),
], style={"minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# 0. Refresh available simulation runs on every navigation
@app.callback(
    Output("filter-run", "options"),
    Output("filter-run", "value"),
    Input("url", "pathname"),
    State("filter-run", "value"),
)
def refresh_runs(pathname, current_run):
    runs    = dl.list_runs()
    options = [{"label": r, "value": r} for r in runs]
    value   = current_run if current_run in runs else (runs[0] if runs else None)
    return options, value


# 1. Update scenarios when run changes
@app.callback(
    Output("filter-scenarios", "options"),
    Output("filter-scenarios", "value"),
    Input("filter-run", "value"),
)
def update_scenarios(run):
    if not run:
        return [], []
    scenarios = dl.list_scenarios(run)
    options   = [{"label": s, "value": s} for s in scenarios]
    return options, scenarios[:1]


# 2. Update zone options when run + scenarios change
@app.callback(
    Output("filter-zones", "options"),
    Input("filter-run",       "value"),
    Input("filter-scenarios", "value"),
)
def update_zones(run, scenarios):
    if not run or not scenarios:
        return []
    zones = dl.get_zones(run, scenarios[0])
    return [{"label": z, "value": z} for z in zones]


# 3. Update year slider when run + scenario change
@app.callback(
    Output("filter-years", "min"),
    Output("filter-years", "max"),
    Output("filter-years", "marks"),
    Output("filter-years", "value"),
    Input("filter-run",       "value"),
    Input("filter-scenarios", "value"),
)
def update_years(run, scenarios):
    if not run or not scenarios:
        return 2020, 2050, {}, [2020, 2050]
    years = dl.get_years(run, scenarios[0])
    if not years:
        return 2020, 2050, {}, [2020, 2050]
    marks = {y: str(y) for y in years}
    return years[0], years[-1], marks, [years[0], years[-1]]


# 4. Show active project in sidebar
@app.callback(
    Output("sidebar-project-label", "children"),
    Input("store-active-project", "data"),
)
def update_sidebar_project(project):
    if not project:
        return html.Small("— none selected —", className="text-muted fst-italic")
    return project


# Route URL → page content + show/hide filter bar
@app.callback(
    Output("page-content",         "children"),
    Output("filter-bar-container", "style"),
    Input("url",                   "pathname"),
    State("filter-run",            "value"),
    State("filter-scenarios",      "value"),
    State("filter-zones",          "value"),
    State("filter-years",          "value"),
    State("store-active-project",  "data"),
)
def render_page(pathname, run, sc_filter, zones, years, active_project):
    path = (pathname or "/").lstrip("/") or "home"

    RESULT_PAGES = {"overview", "capacity", "energy-mix",
                    "emissions", "costs", "trade"}
    filter_style = {"display": "block"} if path in RESULT_PAGES else {"display": "none"}

    pages = {
        "home":              home.layout,
        "overview":          lambda: overview.layout(run, sc_filter, zones, years),
        "capacity":          lambda: capacity.layout(run, sc_filter, zones, years),
        "energy-mix":        lambda: energy_mix.layout(run, sc_filter, zones, years),
        "emissions":         lambda: emissions.layout(run, sc_filter, zones, years),
        "costs":             lambda: costs.layout(run, sc_filter, zones, years),
        "trade":             lambda: trade.layout(run, sc_filter, zones, years),
        "input-manager":     lambda: input_manager.layout(active_project),
        "input-settings":    lambda: input_settings.layout(active_project),
        "input-resolution":  lambda: input_resolution.layout(active_project),
        "input-demand":      lambda: input_demand.layout(active_project),
        "input-supply":      lambda: input_supply.layout(active_project),
        "input-trade":       lambda: input_trade.layout(active_project),
        "input-reserve":     lambda: input_reserve.layout(active_project),
        "input-constraints": lambda: input_constraints.layout(active_project),
        "scenarios":         lambda: scenarios.layout(active_project),
        "run-config":        lambda: run_config.layout(active_project),
        "run-launch":        run_launch.layout,
        "run-history":       run_history.layout,
    }

    render_fn = pages.get(path, home.layout)
    try:
        content = render_fn() if callable(render_fn) else render_fn
    except Exception as e:
        content = dbc.Alert(f"Error loading page: {e}", color="danger")

    return content, filter_style


# Open file/folder in OS default application (local use only)
@app.callback(
    Output("open-file-dummy", "children"),
    Input("open-file-store",  "data"),
    prevent_initial_call=True,
)
def open_file_os(path):
    if not path:
        return ""
    import os, sys, subprocess
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("RENDER") is None   # False on Render, True locally
    app.run(debug=debug, host="0.0.0.0", port=port)
