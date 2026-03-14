"""Run Manager — Launch model run with live log streaming."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import data_loader as dl
from backend.job_manager import launch_run, stop_job, read_log, get_latest_job


def layout(*args):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]

    latest = get_latest_job()
    active_job = latest["job_id"] if latest and latest["status"] == "running" else None

    return html.Div([
        html.H4("Launch Model Run", className="mb-1"),
        html.P("Configure and launch epm.py. The log streams live below.",
               className="text-muted mb-3"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Input folder"),
                    dcc.Dropdown(id="launch-folder", options=options,
                                 value=options[0]["value"] if options else None,
                                 clearable=False),
                ], width=3),
                dbc.Col([
                    dbc.Label("CPU cores"),
                    dcc.Dropdown(id="launch-cpu",
                                 options=[{"label": str(i), "value": i} for i in [1,2,4,8]],
                                 value=2, clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("Run label (optional)"),
                    dbc.Input(id="launch-name", placeholder="e.g. EAPP March 2026"),
                ], width=4),
                dbc.Col([
                    dbc.Label("\u00a0"),
                    dbc.Button("▶ Run Model", id="launch-btn", color="success",
                               className="d-block w-100"),
                ], width=3),
            ], className="align-items-end"),
        ]), className="mb-3 shadow-sm border-0"),

        dbc.Row([
            dbc.Col([
                html.Div(id="launch-status", className="mb-2"),
                dbc.Button("■ Stop", id="stop-btn", color="danger",
                           size="sm", className="me-2"),
                dcc.Link(dbc.Button("View Results →", color="primary", size="sm"),
                         href="/overview"),
            ]),
        ], className="mb-2"),

        dbc.Card(
            dbc.CardBody(
                html.Pre(id="log-output",
                         style={
                             "height": "420px",
                             "overflowY": "auto",
                             "backgroundColor": "#1e1e1e",
                             "color": "#d4d4d4",
                             "fontSize": "0.78rem",
                             "fontFamily": "monospace",
                             "padding": "12px",
                             "margin": 0,
                             "whiteSpace": "pre-wrap",
                             "wordBreak": "break-all",
                         })
            ),
            className="border-0 shadow-sm",
        ),

        # Interval for polling log
        dcc.Interval(id="log-interval", interval=1500, n_intervals=0,
                     disabled=active_job is None),
        dcc.Store(id="active-job-id", data=active_job),
    ])


@callback(
    Output("active-job-id",  "data"),
    Output("log-interval",   "disabled"),
    Output("launch-status",  "children"),
    Input("launch-btn",      "n_clicks"),
    State("launch-folder",   "value"),
    State("launch-cpu",      "value"),
    State("launch-name",     "value"),
    prevent_initial_call=True,
)
def start_run(n, folder, cpu, name):
    if not folder:
        return None, True, dbc.Alert("Select an input folder.", color="warning")
    job_id = launch_run(folder_input=folder, cpu=cpu or 2, run_name=name or "")
    badge = dbc.Badge("● RUNNING", color="warning", className="fs-6")
    return job_id, False, badge


@callback(
    Output("log-output",    "children"),
    Output("log-interval",  "disabled", allow_duplicate=True),
    Output("launch-status", "children", allow_duplicate=True),
    Input("log-interval",   "n_intervals"),
    State("active-job-id",  "data"),
    prevent_initial_call=True,
)
def update_log(n, job_id):
    if not job_id:
        return "No active run.", True, ""

    from backend.job_manager import get_job
    job   = get_job(job_id)
    lines = read_log(job_id, last_n=300)
    log_text = "\n".join(lines) if lines else "Waiting for output…"

    if job and job["status"] == "running":
        status = dbc.Badge("● RUNNING", color="warning", className="fs-6")
        return log_text, False, status
    elif job and job["status"] == "completed":
        status = dbc.Badge("✓ COMPLETED", color="success", className="fs-6")
        return log_text, True, status
    elif job and job["status"] in ("failed", "stopped"):
        status = dbc.Badge(f"✗ {job['status'].upper()}", color="danger", className="fs-6")
        return log_text, True, status

    return log_text, False, ""


@callback(
    Output("launch-status", "children", allow_duplicate=True),
    Input("stop-btn",       "n_clicks"),
    State("active-job-id",  "data"),
    prevent_initial_call=True,
)
def stop(n, job_id):
    if job_id:
        stop_job(job_id)
        return dbc.Badge("■ STOPPED", color="secondary", className="fs-6")
    return ""
