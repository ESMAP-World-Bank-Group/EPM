"""
Reusable variant selector component for input editor pages.

Each parameter can have multiple named file variants (e.g. Baseline, high, low).
Variants follow the naming: {stem}_{name}.csv in the same subfolder as the base file.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import data_loader as dl


def make_project_selector(prefix: str, folders: list, default: str | None) -> dbc.Card:
    """Project / data-folder dropdown card."""
    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project / Data folder", className="fw-semibold small"),
                    dcc.Dropdown(
                        id=f"{prefix}-project",
                        options=[{"label": f, "value": f} for f in folders],
                        value=default,
                        clearable=False,
                        className="small",
                    ),
                ], width=3),
                dbc.Col(
                    html.Small(
                        "Select a variant per tab. Use the Scenario Builder to "
                        "combine variants into named scenarios.",
                        className="text-muted",
                    ),
                    width=9,
                    className="d-flex align-items-center",
                ),
            ], align="center"),
        ),
        className="mb-3 border-0 shadow-sm bg-light",
    )


def make_variant_bar(prefix: str) -> html.Div:
    """
    Compact inline variant selector row.

    IDs exposed:
        {prefix}-variant   — Dropdown (selected variant name)
        {prefix}-dup-name  — Input (new variant name for copy)
        {prefix}-dup-btn   — Button (trigger duplicate)
        {prefix}-dup-msg   — Small text (feedback)
    """
    return html.Div(
        dbc.Row([
            dbc.Col(
                html.Small("Variant:", className="text-muted"),
                width="auto", className="d-flex align-items-center",
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=f"{prefix}-variant",
                    options=[{"label": "Baseline", "value": "Baseline"}],
                    value="Baseline",
                    clearable=False,
                    style={"minWidth": "150px"},
                    className="small",
                ),
                width="auto",
            ),
            dbc.Col(
                dbc.Input(
                    id=f"{prefix}-dup-name",
                    placeholder="Copy as…",
                    size="sm",
                    style={"width": "130px"},
                ),
                width="auto",
            ),
            dbc.Col(
                dbc.Button(
                    "⧉", id=f"{prefix}-dup-btn",
                    size="sm", color="outline-secondary",
                    title="Duplicate current variant with new name",
                ),
                width="auto",
            ),
            dbc.Col(
                html.Small(id=f"{prefix}-dup-msg", className="text-success"),
                width="auto",
            ),
        ], align="center", className="g-2"),
        className="mb-2 mt-1",
    )


def make_open_folder_btn(btn_id: str) -> dbc.Button:
    """Small discreet button to open the input folder in the OS file explorer."""
    return dbc.Button(
        [html.I(className="bi bi-folder2-open me-1"), "Open folder"],
        id=btn_id,
        size="sm",
        color="outline-secondary",
        title="Open input folder in file explorer",
        style={"fontSize": "0.75rem"},
    )


def variant_options(folder: str, key: str) -> list:
    """Return Dropdown options list for a parameter's detected variants."""
    if not folder:
        return [{"label": "Baseline", "value": "Baseline"}]
    return [{"label": name, "value": name}
            for name in dl.list_variants(folder, key)]
