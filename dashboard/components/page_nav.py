"""Small prev/next navigation bar for input editor pages."""

import dash_bootstrap_components as dbc
from dash import html

# Page order (sidebar order)
_PAGES = [
    ("input-settings",    "Settings"),
    ("input-resolution",  "Resolution"),
    ("input-demand",      "Demand"),
    ("input-supply",      "Supply"),
    ("input-trade",       "Trade"),
    ("input-reserve",     "Reserve"),
    ("input-constraints", "Constraints"),
]

_HREF = {pid: f"/{pid}" for pid, _ in _PAGES}
_LABEL = {pid: label for pid, label in _PAGES}
_IDX = {pid: i for i, (pid, _) in enumerate(_PAGES)}


def make_page_nav(current: str) -> html.Div:
    """Return a subtle nav bar with ← Input Manager and Next → buttons."""
    idx = _IDX.get(current, 0)
    next_pid, next_label = (_PAGES[idx + 1] if idx + 1 < len(_PAGES) else (None, None))

    back_btn = dbc.Button(
        [html.I(className="bi bi-grid-3x3-gap me-1"), "Input Manager"],
        href="/input-manager",
        size="sm",
        color="link",
        className="text-muted px-0",
        style={"fontSize": "0.78rem", "textDecoration": "none"},
    )

    next_btn = dbc.Button(
        [next_label, html.I(className="bi bi-chevron-right ms-1")],
        href=f"/{next_pid}",
        size="sm",
        color="link",
        className="text-muted px-0",
        style={"fontSize": "0.78rem", "textDecoration": "none"},
    ) if next_pid else html.Span()

    return html.Div(
        dbc.Row([
            dbc.Col(back_btn, width="auto"),
            dbc.Col(next_btn, width="auto", className="ms-auto"),
        ], align="center"),
        style={
            "borderBottom": "1px solid #e9ecef",
            "paddingBottom": "6px",
            "marginBottom": "12px",
        },
    )
