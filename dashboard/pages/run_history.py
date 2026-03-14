"""Run Manager — History of past runs."""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl
from backend.job_manager import get_all_jobs


def layout(*args):
    runs      = dl.list_runs()
    past_jobs = get_all_jobs()

    # Completed simulation output folders
    run_rows = []
    for run in runs:
        scenarios = dl.list_scenarios(run)
        kpis = {}
        if scenarios:
            kpis = dl.get_kpis(run, scenarios[0])
        npv_str = f"${kpis.get('npv', 0)/1000:,.1f} B" if kpis.get("npv") else "—"
        run_rows.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col(html.Strong(run), width=4),
                    dbc.Col(f"{len(scenarios)} scenario(s): {', '.join(scenarios[:3])}", width=4),
                    dbc.Col(f"NPV: {npv_str}", width=2),
                    dbc.Col(dcc.Link(
                        dbc.Button("View Results", color="outline-primary", size="sm"),
                        href="/overview"
                    ), width=2),
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

        html.H6("Output folders (completed model runs)", className="text-muted"),
        dbc.ListGroup(run_rows or [dbc.ListGroupItem("No completed runs found.")],
                      className="mb-4"),

        html.H6("Session job log", className="text-muted"),
        dbc.ListGroup(job_rows or [dbc.ListGroupItem("No jobs launched in this session.")]),
    ])
