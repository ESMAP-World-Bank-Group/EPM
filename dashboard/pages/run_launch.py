"""Run Manager — Launch model run, show status in dashboard."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import data_loader as dl
from backend.job_manager import launch_run, stop_job, get_latest_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_summary(cfg: dict) -> html.Div:
    """Read-only table of run_config settings."""
    if not cfg:
        return dbc.Alert(
            ["No configuration found. Go to ",
             dcc.Link("Run Config", href="/run-config"), " first."],
            color="warning", className="mb-0 small",
        )

    folder    = cfg.get("folder", "—")
    scenarios = cfg.get("scenarios") or []
    modeltype = cfg.get("modeltype", "MIP")
    cpu       = cfg.get("cpu", 1)
    analysis  = cfg.get("analysis", "standard")
    out_opts  = cfg.get("output_opts") or []
    label     = cfg.get("label", "")
    advanced  = cfg.get("advanced") or []

    def _badge(text, color="secondary"):
        return dbc.Badge(text, color=color, className="me-1")

    sc_display = (
        ", ".join(scenarios)
        if scenarios
        else html.Em("all scenarios", className="text-muted")
    )

    rows = [
        ("Folder",     html.Code(folder)),
        ("Scenarios",  sc_display),
        ("Model type", _badge(modeltype, "primary")),
        ("CPUs",       str(cpu)),
        ("Analysis",   _badge(analysis, "info")),
    ]
    if out_opts:
        rows.append(("Output opts", html.Span([_badge(o) for o in out_opts])))
    if label:
        rows.append(("Label", html.Code(label)))
    if advanced:
        rows.append(("Advanced", html.Span([_badge(a, "warning") for a in advanced])))

    table_rows = [
        html.Tr([
            html.Td(k, className="text-muted pe-3 small"),
            html.Td(v, className="small"),
        ])
        for k, v in rows
    ]
    return html.Div([
        html.Table(html.Tbody(table_rows), className="mb-1"),
        html.Small([
            "To change settings go to ",
            dcc.Link("Run Config", href="/run-config"), ".",
        ], className="text-muted"),
    ])


def _status_panel(job: dict | None) -> html.Div:
    """Render the status area based on job state."""
    if job is None:
        return html.P("No active run.", className="text-muted small mb-0")

    status = job.get("status", "running")
    name   = job.get("run_name", "")
    t0     = job.get("started_at", "")
    t1     = job.get("ended_at")

    # Duration string
    dur = ""
    if t0 and t1:
        from datetime import datetime as _dt
        try:
            secs = int((_dt.fromisoformat(t1) - _dt.fromisoformat(t0)).total_seconds())
            dur  = f"  ·  Duration: {secs // 60}m {secs % 60}s"
        except Exception:
            pass

    if status == "running":
        return dbc.Alert([
            html.Div([
                html.Span(
                    className="spinner-border spinner-border-sm me-2",
                    style={"verticalAlign": "middle"},
                ),
                html.Strong("Run launched — "),
                html.Span(name),
            ], className="mb-1"),
            html.Div([
                html.I(className="bi bi-terminal me-1"),
                "Follow progress in the CMD window that just opened.",
            ], className="small mb-1"),
            html.Div([
                html.I(className="bi bi-arrow-right-circle me-1"),
                'When you see "EPM workflow completed" in the CMD, you can close it and go to ',
                dcc.Link("Results →", href="/overview"),
                ".",
            ], className="small"),
        ], color="warning", className="mb-0")

    if status == "completed":
        return dbc.Alert([
            html.Span("✓  Run completed!", className="fw-semibold me-2"),
            html.Span(f"{name}{dur}", className="small"),
            html.Div([
                dcc.Link(
                    dbc.Button("View Results →", color="success", size="sm",
                               className="mt-2"),
                    href="/overview",
                ),
            ]),
        ], color="success", className="mb-0 py-2")

    if status in ("failed", "stopped"):
        label = "Failed" if status == "failed" else "Stopped"
        rc    = job.get("returncode")
        return dbc.Alert([
            html.Span(f"✗  {label}.", className="fw-semibold me-2"),
            html.Span(
                f"{name}  ·  Exit code: {rc}{dur}" if rc is not None else name,
                className="small",
            ),
        ], color="danger", className="mb-0 py-2")

    return html.P(f"Status: {status}", className="text-muted small mb-0")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    latest     = get_latest_job()
    active_job = latest["job_id"] if latest and latest["status"] == "running" else None

    return html.Div([
        html.H4("Launch Model Run", className="mb-1"),
        html.P("Review configuration, then launch.", className="text-muted mb-3"),

        # ── Config summary ────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H6("Run Configuration", className="mb-2 fw-semibold"),
            html.Div(id="launch-config-summary"),
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Run label (optional)", className="small"),
                    dbc.Input(id="launch-name",
                              placeholder="e.g. EAPP March 2026",
                              size="sm"),
                    html.Small("Overrides the label from Run Config if set.",
                               className="text-muted"),
                ], width=5),
                dbc.Col([
                    dbc.Label("\u00a0", className="small"),
                    dbc.Button("Launch Run", id="launch-btn", color="success",
                               className="d-block w-100"),
                ], width=3),
            ], className="align-items-end g-2 mt-1"),
        ]), className="mb-3 shadow-sm border-0"),

        # ── Command preview ──────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.H6("Command to be run", className="mb-1 fw-semibold"),
            html.Pre(id="launch-command-preview",
                     className="bg-dark text-white p-2 rounded small mb-0",
                     style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"}),
        ]), className="mb-3 shadow-sm border-0"),

        # ── Status ───────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody([
            html.Div(
                id="launch-status-panel",
                children=_status_panel(latest if latest else None),
            ),
            html.Div([
                dbc.Button("■  Stop", id="stop-btn", color="outline-danger",
                           size="sm", className="mt-2"),
            ]),
        ]), className="mb-3 shadow-sm border-0"),

        dcc.Store(id="active-job-id", data=active_job),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("launch-config-summary",   "children"),
    Output("launch-command-preview",  "children"),
    Input("run-config-store", "data"),
)
def show_config(cfg):
    cfg = cfg or {}
    summary = _config_summary(cfg)

    # Build the command string the same way job_manager does
    folder    = cfg.get("folder", "")
    modeltype = cfg.get("modeltype", "MIP")
    cpu       = cfg.get("cpu", 1)
    scenarios = cfg.get("scenarios") or []
    analysis  = cfg.get("analysis", "standard")
    out_opts  = cfg.get("output_opts") or []
    label     = cfg.get("label", "")
    advanced  = cfg.get("advanced") or []

    if not folder:
        cmd_str = "# No folder selected"
    else:
        parts = [f"conda activate gams_env && python -u epm.py --folder_input {folder}"]
        if modeltype and modeltype != "MIP":
            parts.append(f"--modeltype {modeltype}")
        if cpu and cpu > 1:
            parts.append(f"--cpu {cpu}")
        parts.append("--scenarios scenarios.csv")
        if scenarios:
            parts.append("--selected_scenarios " + " ".join(scenarios))
        if analysis == "sensitivity":
            parts.append("--sensitivity")
        elif analysis == "montecarlo":
            parts.append("--montecarlo")
        for opt in out_opts:
            parts.append(f"--{opt}")
        if label:
            parts.append(f"--simulation_label {label}")
        if "simple" in advanced:
            parts.append("--simple")
        if "debug" in advanced:
            parts.append("--debug")
        if "trace" in advanced:
            parts.append("--trace")
        cmd_str = " \\\n  ".join(parts)

    return summary, cmd_str


@callback(
    Output("active-job-id",      "data"),
    Output("launch-status-panel","children"),
    Input("launch-btn",          "n_clicks"),
    State("run-config-store",    "data"),
    State("launch-name",         "value"),
    prevent_initial_call=True,
)
def start_run(n, cfg, override_name):
    cfg = cfg or {}
    folder = cfg.get("folder")
    if not folder:
        return None, dbc.Alert(
            ["No folder selected. Go to ",
             dcc.Link("Run Config", href="/run-config"), "."],
            color="warning",
        )

    label = (
        override_name.strip()
        if override_name and override_name.strip()
        else cfg.get("label", "")
    )

    job_id = launch_run(
        folder_input=folder,
        cpu=cfg.get("cpu", 1),
        run_name=label or "",
        scenarios=cfg.get("scenarios") or [],
        modeltype=cfg.get("modeltype", "MIP"),
        analysis=cfg.get("analysis", "standard"),
        output_opts=cfg.get("output_opts") or [],
        label=label,
        advanced=cfg.get("advanced") or [],
    )

    from backend.job_manager import get_job as _gj
    return job_id, _status_panel(_gj(job_id))


@callback(
    Output("launch-status-panel", "children", allow_duplicate=True),
    Input("stop-btn",             "n_clicks"),
    State("active-job-id",        "data"),
    prevent_initial_call=True,
)
def stop(n, job_id):
    if job_id:
        stop_job(job_id)
    return _status_panel(None)
