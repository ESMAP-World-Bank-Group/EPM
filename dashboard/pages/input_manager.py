"""Input Overview — section cards + unified project file browser."""

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

import data_loader as dl
from config import INPUT_ROOT

# ---------------------------------------------------------------------------
# Sections — order mirrors the modelling workflow
# ---------------------------------------------------------------------------

SECTIONS = [
    {
        "key":      "settings",
        "icon":     "bi-gear-fill",
        "color":    "#2c6fad",
        "title":    "Settings",
        "desc":     "Solver options, model years, discount rate, CO₂ price, fuel prices.",
        "page":     "input-settings",
        "csv_files": [
            ("",  "pSettings.csv"),
            ("",  "y.csv"),
            ("",  "pHours.csv"),
            ("",  "zcmap.csv"),
            ("",  "scenarios.csv"),
        ],
    },
    {
        "key":      "resolution",
        "icon":     "bi-grid-3x3",
        "color":    "#7b4f9e",
        "title":    "Resolution",
        "desc":     "Spatial zones, seasons, representative days and hours.",
        "page":     "input-resolution",
        "csv_files": [
            ("resolution", "pRepresentativeDays.csv"),
            ("resolution", "pRepresentativeWeeks.csv"),
            ("resolution", "pSeasons.csv"),
        ],
    },
    {
        "key":      "demand",
        "icon":     "bi-graph-up",
        "color":    "#f77f00",
        "title":    "Demand",
        "desc":     "Demand forecasts and load profiles by zone.",
        "page":     "input-demand",
        "csv_files": [
            ("load", "pDemandForecast.csv"),
            ("load", "pDemandProfile.csv"),
            ("load", "pDemandData.csv"),
            ("load", "pEnergyEfficiencyFactor.csv"),
        ],
    },
    {
        "key":      "supply",
        "icon":     "bi-lightning-charge",
        "color":    "#2d9e4f",
        "title":    "Supply",
        "desc":     "Generator data, capacity, costs and availability factors.",
        "page":     "input-supply",
        "csv_files": [
            ("supply", "pGenDataInput.csv"),
            ("supply", "pGenDataInputDefault.csv"),
            ("supply", "pStorageDataInput.csv"),
            ("supply", "pFuelPrice.csv"),
            ("supply", "pCapexTrajectoriesCustom.csv"),
            ("supply", "pCapexTrajectoriesDefault.csv"),
            ("supply", "pAvailabilityCustom.csv"),
        ],
    },
    {
        "key":      "trade",
        "icon":     "bi-arrow-left-right",
        "color":    "#16a085",
        "title":    "Trade & Transmission",
        "desc":     "Interconnection capacities, trade limits and costs.",
        "page":     "input-trade",
        "csv_files": [
            ("transmission", "pTransferLimit.csv"),
            ("transmission", "pNewTransmission.csv"),
            ("transmission", "pTradePrice.csv"),
            ("transmission", "pExtTransferLimit.csv"),
        ],
    },
    {
        "key":      "reserve",
        "icon":     "bi-shield-check",
        "color":    "#e07b39",
        "title":    "Reserve",
        "desc":     "Planning and spinning reserve requirements.",
        "page":     "input-reserve",
        "csv_files": [
            ("reserve", "pPlanningReserveMargin.csv"),
            ("reserve", "pSpinningReserveReqCountry.csv"),
            ("reserve", "pSpinningReserveReqSystem.csv"),
        ],
    },
    {
        "key":      "constraints",
        "icon":     "bi-sliders",
        "color":    "#d62728",
        "title":    "Constraints & Policy",
        "desc":     "RE targets, emission caps, carbon budgets.",
        "page":     "input-constraints",
        "csv_files": [
            ("constraint", "pEmissionsTotal.csv"),
            ("constraint", "pEmissionsCountry.csv"),
            ("constraint", "pCarbonPrice.csv"),
            ("constraint", "pMaxFuellimit.csv"),
            ("constraint", "pMaxGenerationByFuel.csv"),
        ],
    },
    {
        "key":      "scenarios",
        "icon":     "bi-diagram-3",
        "color":    "#8e44ad",
        "title":    "Scenario Builder",
        "desc":     "Combine parameter variants into named model scenarios.",
        "page":     "scenarios",
        "csv_files": [
            ("", "scenarios.csv"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_exists(folder: str | None, subfolder: str, filename: str) -> bool:
    if not folder:
        return False
    base = INPUT_ROOT / folder
    path = base / subfolder / filename if subfolder else base / filename
    return path.exists()


def _count_existing(folder: str | None, section: dict) -> tuple[int, int]:
    """Return (n_existing, n_total) for a section's CSV files."""
    total   = len(section["csv_files"])
    existing = sum(
        1 for sf, fn in section["csv_files"]
        if _file_exists(folder, sf, fn)
    )
    return existing, total


# ---------------------------------------------------------------------------
# Card (clean — no file list inside)
# ---------------------------------------------------------------------------

def _section_card(section: dict, folder: str | None) -> dbc.Col:
    color             = section["color"]
    n_ok, n_total     = _count_existing(folder, section)

    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                # Header
                html.Div([
                    html.I(className=f"bi {section['icon']} me-2",
                           style={"fontSize": "1.35rem", "color": color}),
                    html.Span(section["title"], className="fw-semibold",
                              style={"fontSize": "0.95rem", "color": "#1B2A4A"}),
                ], className="d-flex align-items-center mb-2"),

                # Description
                html.P(section["desc"], className="text-muted mb-3",
                       style={"fontSize": "0.80rem", "minHeight": "38px"}),

                # Edit button
                dbc.Button(
                    [html.I(className="bi bi-pencil-square me-1"), "Edit"],
                    href=f"/{section['page']}",
                    size="sm", color="outline-primary",
                    className="w-100",
                    disabled=not folder,
                ),
            ]),
            html.Div(style={
                "height": "4px",
                "backgroundColor": color,
                "borderRadius": "0 0 4px 4px",
            }),
        ], className="shadow-sm border-0 h-100"),
        md=3, className="mb-3",
    )


# ---------------------------------------------------------------------------
# File browser — grouped by section, shown below the cards
# ---------------------------------------------------------------------------

def _dot(color: str, dashed: bool = False) -> html.Span:
    """Small colored dot. Dashed outline style for variant files."""
    if dashed:
        return html.Span(style={
            "display":      "inline-block",
            "width":        "6px",
            "height":       "6px",
            "borderRadius": "50%",
            "border":       f"1.5px solid {color}",
            "flexShrink":   "0",
            "marginRight":  "7px",
            "marginTop":    "1px",
        })
    return html.Span(style={
        "display":         "inline-block",
        "width":           "6px",
        "height":          "6px",
        "borderRadius":    "50%",
        "backgroundColor": color,
        "flexShrink":      "0",
        "marginRight":     "7px",
        "marginTop":       "1px",
    })


def _file_row(fname: str, exists: bool, variant: bool = False) -> html.Div:
    if variant:
        dot   = _dot("#6f42c1", dashed=True)   # purple outline = variant
        color = "#6f42c1"
        style = {"fontSize": "0.74rem", "fontFamily": "monospace",
                 "color": color, "fontStyle": "italic"}
    else:
        dot   = _dot("#28a745" if exists else "#ced4da")
        style = {"fontSize": "0.74rem", "fontFamily": "monospace",
                 "color": "#495057" if exists else "#adb5bd"}
    return html.Div([dot, html.Span(fname, style=style)],
                    style={"display": "flex", "alignItems": "center",
                           "padding": "2px 0"})


def _file_browser(folder: str | None) -> dbc.Card:
    cols = []
    for section in SECTIONS:
        color = section["color"]

        # Known files
        known_names = {fn for _, fn in section["csv_files"]}
        file_rows = [
            _file_row(fn, _file_exists(folder, sf, fn))
            for sf, fn in section["csv_files"]
        ]

        # Variant files — CSVs found in the subfolder(s) not in the predefined list
        if folder:
            scanned_subs = {sf for sf, _ in section["csv_files"]}
            for sf in scanned_subs:
                base = INPUT_ROOT / folder
                d    = base / sf if sf else base
                if d.exists():
                    extras = sorted(
                        p.name for p in d.glob("*.csv")
                        if p.name not in known_names
                    )
                    if extras:
                        file_rows.append(html.Div(
                            html.Small("variants", style={
                                "fontSize":      "0.65rem",
                                "color":         "#6f42c1",
                                "letterSpacing": "0.04em",
                                "textTransform": "uppercase",
                                "paddingTop":    "4px",
                                "display":       "block",
                            })
                        ))
                        file_rows += [_file_row(fn, True, variant=True) for fn in extras]

        group = html.Div([
            # Section header
            html.Div([
                html.I(className=f"bi {section['icon']} me-1",
                       style={"color": color, "fontSize": "0.8rem"}),
                html.Span(section["title"],
                          className="fw-semibold text-uppercase",
                          style={"fontSize": "0.7rem", "letterSpacing": "0.05em",
                                 "color": "#1B2A4A"}),
            ], style={
                "display":      "flex",
                "alignItems":   "center",
                "borderLeft":   f"3px solid {color}",
                "paddingLeft":  "8px",
                "marginBottom": "4px",
            }),
            html.Div(file_rows, style={"paddingLeft": "11px"}),
        ], style={"marginBottom": "18px"})

        cols.append(dbc.Col(group, md=3))

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className="bi bi-folder2-open me-2",
                       style={"color": "#6c757d"}),
                html.Span("Project files",
                          className="fw-semibold",
                          style={"fontSize": "0.88rem", "color": "#1B2A4A"}),
                html.Small(
                    f"  —  {folder}" if folder else "  —  no project selected",
                    className="text-muted ms-2",
                    style={"fontSize": "0.75rem"},
                ),
            ], className="mb-3"),
            dbc.Row(cols, className="g-2"),
        ]),
        className="shadow-sm border-0",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(active_project=None):
    folders = dl.list_input_folders()
    current = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Input Manager", className="mb-1"),
        html.P("Select a section to view and edit input data.",
               className="text-muted mb-2", style={"fontSize": "0.85rem"}),
        html.Div(id="im-no-project-warning"),

        # Section cards
        html.Div(
            id="im-section-cards",
            children=dbc.Row(
                [_section_card(s, current) for s in SECTIONS],
                className="g-3",
            ),
        ),

        # File browser
        html.Div(id="im-file-browser",
                 children=_file_browser(current),
                 className="mt-4"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("im-section-cards",      "children"),
    Output("im-file-browser",       "children"),
    Output("im-no-project-warning", "children"),
    Input("store-active-project", "data"),
)
def refresh(folder):
    cards   = dbc.Row([_section_card(s, folder) for s in SECTIONS], className="g-3")
    browser = _file_browser(folder)
    warning = dbc.Alert(
        [html.I(className="bi bi-exclamation-triangle-fill me-2"),
         "No project selected — please select a project from the dropdown above before editing."],
        color="warning", className="py-2 mb-3", style={"fontSize": "0.85rem"},
    ) if not folder else None
    return cards, browser, warning
