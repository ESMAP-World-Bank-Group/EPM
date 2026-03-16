"""Results — Results Table: filterable DataTable with CSV export."""

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html

import data_loader as dl


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout(*args):
    return dbc.Container([
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("Scenario", className="form-label-sm"),
                dcc.Dropdown(id="rt-scenario", multi=True, placeholder="All",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Zone", className="form-label-sm"),
                dcc.Dropdown(id="rt-zone", multi=True, placeholder="All",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                html.Label("Attribute", className="form-label-sm"),
                dcc.Dropdown(id="rt-attribute", multi=True, placeholder="All",
                             style={"fontSize": "0.85rem"}),
            ], md=3),
            dbc.Col([
                html.Label("Year", className="form-label-sm"),
                dcc.Dropdown(id="rt-year", multi=True, placeholder="All",
                             style={"fontSize": "0.85rem"}),
            ], md=2),
            dbc.Col([
                dbc.Label("\u00a0", className="form-label-sm"),
                dbc.Button("Export CSV", id="rt-export-btn",
                           color="secondary", size="sm", className="d-block"),
                dcc.Download(id="rt-download"),
            ], md=1),
        ], className="g-2")), className="mb-3 shadow-sm filter-card"),

        html.Div(id="rt-row-count", className="text-muted mb-2",
                 style={"fontSize": "0.82rem"}),

        dbc.Card(dbc.CardBody(
            dash_table.DataTable(
                id="rt-table",
                columns=[], data=[],
                page_size=25, page_action="native",
                sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1B2A4A", "color": "white",
                    "fontWeight": "bold", "fontSize": "12px",
                    "border": "1px solid #dee2e6",
                },
                style_cell={
                    "fontSize": "11px", "padding": "6px 10px",
                    "border": "1px solid #dee2e6", "textAlign": "left",
                    "minWidth": "60px", "maxWidth": "220px",
                    "overflow": "hidden", "textOverflow": "ellipsis",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                    {"if": {"filter_query": "{value} > 0", "column_id": "value"},
                     "color": "#155724"},
                    {"if": {"filter_query": "{value} < 0", "column_id": "value"},
                     "color": "#721c24"},
                ],
                tooltip_delay=0, tooltip_duration=None,
            )
        ), className="shadow-sm border-0"),
    ], fluid=True, className="py-3 px-2")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("rt-scenario",  "options"),
    Output("rt-zone",      "options"),
    Output("rt-attribute", "options"),
    Output("rt-year",      "options"),
    Input("filter-run",    "value"),
)
def init_rt_filters(run):
    if not run:
        return [], [], [], []
    scenarios = dl.list_scenarios(run)
    zones     = dl.get_merged_zones(run)
    years     = dl.get_merged_years(run)

    attrs = set()
    for df in [dl.load_techfuel(run), dl.load_costs_merged(run),
               dl.load_yearly_zone(run)]:
        if not df.empty and "attribute" in df.columns:
            attrs.update(df["attribute"].dropna().unique().tolist())

    return (
        [{"label": s, "value": s} for s in sorted(scenarios)],
        [{"label": z, "value": z} for z in sorted(zones)],
        [{"label": a, "value": a} for a in sorted(attrs)],
        [{"label": str(y), "value": y} for y in sorted(years)],
    )


@callback(
    Output("rt-table",     "columns"),
    Output("rt-table",     "data"),
    Output("rt-row-count", "children"),
    Input("rt-scenario",   "value"),
    Input("rt-zone",       "value"),
    Input("rt-attribute",  "value"),
    Input("rt-year",       "value"),
    Input("filter-run",    "value"),
)
def update_rt_table(scenarios, zones, attributes, years, run):
    if not run:
        return [], [], "Select a run."

    dfs = []
    for df in [dl.load_techfuel(run), dl.load_costs_merged(run),
               dl.load_capex_merged(run), dl.load_yearly_zone(run)]:
        if not df.empty:
            dfs.append(df)

    if not dfs:
        msg = ("No merged output files found. "
               "Please re-run the model to generate pTechFuelMerged.csv, etc.")
        return [], [], msg

    df_all = pd.concat(dfs, ignore_index=True)

    if scenarios:
        df_all = df_all[df_all["scenario"].isin(scenarios)]
    if zones and "z" in df_all.columns:
        df_all = df_all[df_all["z"].isin(zones)]
    if attributes and "attribute" in df_all.columns:
        df_all = df_all[df_all["attribute"].isin(attributes)]
    if years and "y" in df_all.columns:
        df_all = df_all[df_all["y"].isin(years)]

    if "value" in df_all.columns:
        df_all["value"] = pd.to_numeric(df_all["value"], errors="coerce").round(4)

    keep = [c for c in ["scenario", "c", "z", "attribute", "y",
                         "techfuel", "uni", "tech", "f", "g", "value"]
            if c in df_all.columns]
    df_out = df_all[keep].drop_duplicates().reset_index(drop=True)

    MAX_ROWS = 5000
    if len(df_out) > MAX_ROWS:
        df_out = df_out.head(MAX_ROWS)
        count_msg = f"Showing first {MAX_ROWS:,} rows — apply filters to narrow down."
    else:
        count_msg = f"{len(df_out):,} rows"

    columns = [{"name": c.upper(), "id": c,
                "type": "numeric" if c == "value" else "text"}
               for c in df_out.columns]
    return columns, df_out.to_dict("records"), count_msg


@callback(
    Output("rt-download",    "data"),
    Input("rt-export-btn",   "n_clicks"),
    State("rt-table",        "data"),
    State("rt-table",        "columns"),
    prevent_initial_call=True,
)
def export_rt_csv(n_clicks, data, columns):
    import dash
    if not data:
        return dash.no_update
    col_names = [c["id"] for c in columns]
    df = pd.DataFrame(data, columns=col_names)
    return dcc.send_data_frame(df.to_csv, "epm_results_export.csv", index=False)
