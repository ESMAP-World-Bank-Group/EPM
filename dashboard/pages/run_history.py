"""Run Manager — History of past runs."""

import shutil
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update, ctx, ALL
import data_loader as dl
from backend.job_manager import get_all_jobs
from config import OUTPUT_ROOT


def layout(*args):
    runs      = dl.list_runs()
    past_jobs = get_all_jobs()

    # Completed simulation output folders
    run_rows = []
    for run in runs:
        scenarios = dl.list_scenarios(run)
        run_rows.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col(html.Strong(run, className="small"), width=4),
                    dbc.Col(
                        html.Small(
                            f"{len(scenarios)} scenario(s): {', '.join(scenarios[:3])}"
                            + (" …" if len(scenarios) > 3 else ""),
                            className="text-muted",
                        ),
                        width=4,
                    ),
                    dbc.Col([
                        dcc.Link(
                            dbc.Button("View Results", color="outline-primary", size="sm",
                                       className="me-1"),
                            href="/overview",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-folder2-open")],
                            id={"type": "hist-open-btn", "run": run},
                            size="sm", color="outline-secondary", className="me-1",
                            title="Open output folder",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-trash3")],
                            id={"type": "hist-del-btn", "run": run},
                            size="sm", color="outline-danger",
                            title="Delete this run folder",
                        ),
                    ], width=4, className="d-flex align-items-center"),
                ], align="center"),
            ])
        )

    # In-session job history
    job_rows = []
    for job in past_jobs:
        color = {"running": "warning", "completed": "success",
                 "failed": "danger", "stopped": "secondary"}.get(job["status"], "light")
        job_rows.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col(html.Strong(job.get("run_name", job["job_id"])), width=4),
                    dbc.Col(f"Folder: {job['folder']}", width=3),
                    dbc.Col(f"Started: {job['started_at'][:16]}", width=3),
                    dbc.Col(dbc.Badge(job["status"].upper(), color=color), width=2),
                ], align="center"),
            ])
        )

    return html.Div([
        html.H4("Run History", className="mb-3"),

        # Delete confirmation modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Delete run folder?")),
            dbc.ModalBody(html.Div(id="hist-del-confirm-body")),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="hist-del-cancel", color="secondary",
                           className="me-2"),
                dbc.Button("Delete", id="hist-del-confirm", color="danger"),
            ]),
        ], id="hist-del-modal", is_open=False),

        # Store for pending delete
        dcc.Store(id="hist-pending-del", data=None),

        html.H6("Output folders (completed model runs)", className="text-muted"),
        html.Div(id="hist-run-list",
                 children=dbc.ListGroup(
                     run_rows or [dbc.ListGroupItem("No completed runs found.")],
                     className="mb-4",
                 )),

        html.H6("Session job log", className="text-muted"),
        dbc.ListGroup(job_rows or [dbc.ListGroupItem("No jobs launched in this session.")]),
    ])


# ── Open output folder ────────────────────────────────────────────────────────
@callback(
    Output("open-file-store", "data", allow_duplicate=True),
    Input({"type": "hist-open-btn", "run": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_run_folder(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in n_clicks_list if n):
        return no_update
    run = triggered["run"]
    return str(OUTPUT_ROOT / run)


# ── Delete: open modal ────────────────────────────────────────────────────────
@callback(
    Output("hist-del-modal",        "is_open"),
    Output("hist-del-confirm-body", "children"),
    Output("hist-pending-del",      "data"),
    Input({"type": "hist-del-btn", "run": ALL}, "n_clicks"),
    Input("hist-del-cancel",  "n_clicks"),
    Input("hist-del-confirm", "n_clicks"),
    State("hist-pending-del", "data"),
    prevent_initial_call=True,
)
def toggle_del_modal(del_clicks, cancel, confirm, pending):
    triggered = ctx.triggered_id

    if triggered == "hist-del-cancel":
        return False, no_update, None

    if triggered == "hist-del-confirm":
        return False, no_update, None

    # A trash button was clicked
    if isinstance(triggered, dict) and triggered.get("type") == "hist-del-btn":
        if not any(n for n in del_clicks if n):
            return no_update, no_update, no_update
        run = triggered["run"]
        body = [
            html.P(["Are you sure you want to permanently delete:"]),
            html.Code(run),
            html.P("This cannot be undone.", className="text-danger small mt-2"),
        ]
        return True, body, run

    return no_update, no_update, no_update


# ── Delete: execute ───────────────────────────────────────────────────────────
@callback(
    Output("hist-run-list",    "children"),
    Output("hist-del-modal",   "is_open", allow_duplicate=True),
    Input("hist-del-confirm",  "n_clicks"),
    State("hist-pending-del",  "data"),
    prevent_initial_call=True,
)
def do_delete(n, run):
    if not n or not run:
        return no_update, no_update
    folder = OUTPUT_ROOT / run
    try:
        if folder.exists():
            shutil.rmtree(folder)
    except Exception:
        pass

    # Rebuild run list
    runs = dl.list_runs()
    run_rows = []
    for r in runs:
        scenarios = dl.list_scenarios(r)
        run_rows.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col(html.Strong(r, className="small"), width=4),
                    dbc.Col(
                        html.Small(
                            f"{len(scenarios)} scenario(s): {', '.join(scenarios[:3])}"
                            + (" …" if len(scenarios) > 3 else ""),
                            className="text-muted",
                        ),
                        width=4,
                    ),
                    dbc.Col([
                        dcc.Link(
                            dbc.Button("View Results", color="outline-primary", size="sm",
                                       className="me-1"),
                            href="/overview",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-folder2-open")],
                            id={"type": "hist-open-btn", "run": r},
                            size="sm", color="outline-secondary", className="me-1",
                            title="Open output folder",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-trash3")],
                            id={"type": "hist-del-btn", "run": r},
                            size="sm", color="outline-danger",
                            title="Delete this run folder",
                        ),
                    ], width=4, className="d-flex align-items-center"),
                ], align="center"),
            ])
        )

    return (
        dbc.ListGroup(run_rows or [dbc.ListGroupItem("No completed runs found.")],
                      className="mb-4"),
        False,
    )
