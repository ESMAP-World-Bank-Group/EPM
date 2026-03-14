"""Input Editor — Settings (pSettings.csv) with grouped UI."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ALL
import pandas as pd
import data_loader as dl
from config import INPUT_ROOT


# Settings that are boolean (0/1) — shown as toggle switches
BOOLEAN_ABBREVS = {
    "fEnableCapacityExpansion", "fDispatchMode", "fEnableInternalExchange",
    "fRemoveInternalTransferLimit", "fAllowTransferExpansion",
    "fEnableExternalExchange", "fEnableCarbonPrice", "fEnableEnergyEfficiency",
    "fEnableCSP", "fEnableStorage", "fEnableEconomicRetirement",
    "fUseSimplifiedDemand", "fCountIntercoForReserves",
    "fApplyPlanningReserveConstraint", "fApplyCountrySpinReserveConstraint",
    "fApplySystemSpinReserveConstraint", "fEnableCapexTrajectoryH2",
    "fEnableH2Production", "fApplyCountryCo2Constraint",
    "fApplySystemCo2Constraint", "fApplyFuelConstraint",
    "fApplyGenerationPhaseout", "fApplyCapitalConstraint",
    "fApplyMinGenShareAllHours", "fApplyRampConstraint",
    "fApplyMinGenCommitment", "fApplyMUDT", "fApplyStartupCost",
    "EPM_FILL_HYDRO_AVAILABILITY", "EPM_FILL_HYDRO_CAPEX",
    "EPM_FILL_ROR_FROM_AVAILABILITY",
}


def _build_settings_ui(df: pd.DataFrame) -> list:
    """Build grouped settings UI from pSettings.csv DataFrame."""
    if df.empty:
        return [html.P("pSettings.csv not found.", className="text-muted")]

    ui = []
    section_rows = []
    current_section = "General"

    for _, row in df.iterrows():
        param = str(row.get("Parameter", "") or "").strip()
        abbr  = str(row.get("Abbreviation", "") or "").strip()
        val   = row.get("Value", "")

        # Section header (no abbreviation, non-empty parameter)
        if param and not abbr:
            if section_rows:
                ui.append(_section_card(current_section, section_rows))
                section_rows = []
            current_section = param
            continue

        # Skip fully blank rows
        if not abbr:
            continue

        section_rows.append((param, abbr, val))

    if section_rows:
        ui.append(_section_card(current_section, section_rows))

    return ui


def _section_card(title: str, rows: list) -> dbc.Card:
    items = []
    for param, abbr, val in rows:
        is_bool = abbr in BOOLEAN_ABBREVS
        try:
            num_val = float(val) if str(val).strip() not in ("", "nan") else None
        except (ValueError, TypeError):
            num_val = None

        if is_bool:
            checked = bool(num_val) if num_val is not None else False
            control = dbc.Switch(
                id={"type": "setting-switch", "index": abbr},
                value=checked,
                className="ms-2",
            )
        else:
            control = dbc.Input(
                id={"type": "setting-input", "index": abbr},
                type="number",
                value=num_val if num_val is not None else "",
                size="sm",
                style={"width": "120px"},
                debounce=True,
            )

        items.append(
            dbc.Row([
                dbc.Col(html.Small(param, className="text-muted"), width=7),
                dbc.Col(html.Code(abbr, style={"fontSize": "0.72rem"}), width=3),
                dbc.Col(control, width=2, className="d-flex align-items-center justify-content-end"),
            ], className="py-1 border-bottom align-items-center")
        )

    return dbc.Card([
        dbc.CardHeader(html.Strong(title, className="text-uppercase small",
                                    style={"letterSpacing": "0.05em"})),
        dbc.CardBody(items, className="py-1 px-3"),
    ], className="mb-3 shadow-sm border-0")


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        html.H4("Model Settings", className="mb-1"),
        html.P("Configure model switches and parameters. Toggles = on/off features. "
               "Numbers = numeric parameters.",
               className="text-muted mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Project"),
                dcc.Dropdown(id="settings-project", options=options,
                             value=default, clearable=False),
            ], width=3),
            dbc.Col([
                dbc.Label("\u00a0"),
                dbc.Button("Save Settings", id="save-settings-btn",
                           color="success", className="d-block"),
            ], width=2),
            dbc.Col(html.Div(id="save-settings-msg",
                             className="d-flex align-items-end pb-1"), width=3),
        ], className="mb-4 align-items-end"),

        html.Div(id="settings-ui"),
    ])


@callback(
    Output("settings-ui", "children"),
    Input("settings-project", "value"),
)
def load_settings(folder):
    if not folder:
        return html.Div()
    df = dl.load_input(folder, "settings_input")
    return _build_settings_ui(df)


@callback(
    Output("save-settings-msg", "children"),
    Input("save-settings-btn",  "n_clicks"),
    State("settings-project",   "value"),
    State({"type": "setting-switch", "index": ALL}, "value"),
    State({"type": "setting-switch", "index": ALL}, "id"),
    State({"type": "setting-input",  "index": ALL}, "value"),
    State({"type": "setting-input",  "index": ALL}, "id"),
    prevent_initial_call=True,
)
def save_settings(n, folder, switch_vals, switch_ids, input_vals, input_ids):
    if not folder:
        return dbc.Badge("No project selected", color="warning")

    df = dl.load_input(folder, "settings_input")
    if df.empty:
        return dbc.Badge("Could not load settings", color="danger")

    # Build abbr → new_value map
    updates = {}
    for sid, sval in zip(switch_ids, switch_vals):
        updates[sid["index"]] = 1 if sval else 0
    for iid, ival in zip(input_ids, input_vals):
        if ival is not None and str(ival).strip() != "":
            updates[iid["index"]] = ival

    # Apply updates to DataFrame
    df_new = df.copy()
    for i, row in df_new.iterrows():
        abbr = str(row.get("Abbreviation", "") or "").strip()
        if abbr in updates:
            df_new.at[i, "Value"] = updates[abbr]

    ok = dl.save_input(folder, "settings_input", df_new)
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")


