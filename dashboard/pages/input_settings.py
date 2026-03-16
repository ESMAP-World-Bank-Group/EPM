"""Input Editor — Settings (pSettings.csv) with tabbed UI and variant selector."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, ALL
import pandas as pd
import data_loader as dl
from components.variant_selector import make_variant_bar, variant_options, make_open_folder_btn
from config import INPUT_ROOT

# ---------------------------------------------------------------------------
# Section → Tab mapping
# ---------------------------------------------------------------------------
SECTION_TO_TAB = {
    "PARAMETERS":       "Parameters",
    "INTERCONNECTION":  "Interconnection",
    "INTERNAL":         "Interconnection",
    "EXTERNAL":         "Interconnection",
    "OPTIONAL FEATURES":"Features",
    "PLANNING RESERVES":"Reserves",
    "SPINNING RESERVES":"Reserves",
    "H2":               "H2",
    "POLICY":           "Policy",
    "PLANTS":           "Plants",
    "INPUT TREATMENTS": "Treatment",
}
TAB_ORDER = ["Core", "Parameters", "Interconnection", "Features",
             "Reserves", "Policy", "Plants", "H2", "Treatment"]

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


def _clean(val) -> str:
    """Return empty string for NaN/None, else stripped string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _parse_settings(df: pd.DataFrame) -> dict[str, list]:
    """
    Parse pSettings.csv into {tab_name: [(param, abbr, value), ...]}.
    Skips blank rows and section header rows.
    """
    tabs: dict[str, list] = {t: [] for t in TAB_ORDER}
    current_tab = "Core"

    for _, row in df.iterrows():
        param = _clean(row.get("Parameter",    ""))
        abbr  = _clean(row.get("Abbreviation", ""))
        val   = row.get("Value", "")

        # Skip fully blank rows
        if not param and not abbr:
            continue
        # Skip rows where param is blank (abbr-only rows shouldn't exist)
        if not param:
            continue

        # Section header — non-empty param, empty abbr
        if param and not abbr:
            current_tab = SECTION_TO_TAB.get(param.upper(), current_tab)
            continue

        # Skip rows without a valid abbreviation
        if not abbr:
            continue

        # Clean value
        clean_val = "" if (val is None or (isinstance(val, float) and pd.isna(val))
                           or str(val).strip() in ("nan", "NaN", "")) else val

        tabs[current_tab].append((param, abbr, clean_val))

    return tabs


def _setting_row(param: str, abbr: str, val) -> dbc.Row:
    """Build one setting row with label, abbreviation code, and control."""
    is_bool = abbr in BOOLEAN_ABBREVS
    try:
        num_val = float(val) if str(val).strip() not in ("", "nan", "NaN") else None
    except (ValueError, TypeError):
        num_val = None

    if is_bool:
        checked = bool(num_val) if num_val is not None else False
        control = dbc.Switch(
            id={"type": "setting-switch", "index": abbr},
            value=checked,
        )
    else:
        control = dbc.Input(
            id={"type": "setting-input", "index": abbr},
            type="number",
            value=num_val if num_val is not None else "",
            size="sm",
            style={"width": "130px"},
            debounce=True,
        )

    return dbc.Row([
        dbc.Col(html.Small(param, className="text-muted"), width=6),
        dbc.Col(
            html.Code(abbr, style={"fontSize": "0.70rem", "color": "#6c757d"}),
            width=4,
        ),
        dbc.Col(control, width=2,
                className="d-flex align-items-center justify-content-end"),
    ], className="py-1 border-bottom align-items-center")


def _tab_content(rows: list) -> html.Div:
    if not rows:
        return html.P("No settings in this category.", className="text-muted mt-3")
    return html.Div([_setting_row(p, a, v) for p, a, v in rows],
                    className="mt-2")


def layout(active_project=None):
    folders = dl.list_input_folders()
    default = active_project or (folders[0] if folders else None)

    return html.Div([
        dbc.Row([
            dbc.Col(html.H4("Model Settings", className="mb-0"), width="auto"),
            dbc.Col(make_open_folder_btn("settings-open-btn"), width="auto",
                    className="d-flex align-items-center"),
            dbc.Col(
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Reload"],
                           id="stg-reload-btn", color="outline-secondary", size="sm"),
                width="auto", className="ms-auto",
            ),
        ], className="mb-1 align-items-center justify-content-between"),
        html.P("Configure model switches (toggles) and numeric parameters.",
               className="text-muted mb-3"),

        html.Div(
            dcc.Dropdown(id="settings-project",
                         options=[{"label": f, "value": f} for f in folders],
                         value=default, clearable=False),
            style={"display": "none"},
        ),

        make_variant_bar("stg"),

        dbc.Row([
            dbc.Col(dbc.Button("Save Settings", id="save-settings-btn",
                               color="success"), width="auto"),
            dbc.Col(html.Div(id="save-settings-msg"), width="auto"),
        ], className="mb-3 align-items-center"),

        html.Div(id="settings-tabs-content"),
    ])


@callback(
    Output("settings-tabs-content", "children"),
    Output("stg-variant",          "options"),
    Input("settings-project",      "value"),
    Input("stg-variant",           "value"),
    Input("stg-reload-btn",        "n_clicks"),
)
def load_settings(folder, variant, _reload=None):
    if _reload:
        dl.clear_input_cache()
    base_opts = [{"label": "Baseline", "value": "Baseline"}]
    if not folder:
        return html.Div(), base_opts

    df = dl.load_variant(folder, "settings_input", variant)
    if df.empty:
        return dbc.Alert("pSettings.csv not found in this project.", color="warning"), base_opts

    tabs_data = _parse_settings(df)
    tab_items = []
    for tab_name in TAB_ORDER:
        rows = tabs_data.get(tab_name, [])
        if not rows:
            continue
        tab_items.append(
            dbc.Tab(label=tab_name, tab_id=f"stab-{tab_name.lower()}",
                    children=_tab_content(rows))
        )

    if not tab_items:
        return dbc.Alert("No settings found.", color="warning"), base_opts

    return dbc.Tabs(tab_items, active_tab="stab-core"), variant_options(folder, "settings_input")


@callback(
    Output("save-settings-msg", "children"),
    Input("save-settings-btn",  "n_clicks"),
    State("settings-project",   "value"),
    State("stg-variant",        "value"),
    State({"type": "setting-switch", "index": ALL}, "value"),
    State({"type": "setting-switch", "index": ALL}, "id"),
    State({"type": "setting-input",  "index": ALL}, "value"),
    State({"type": "setting-input",  "index": ALL}, "id"),
    prevent_initial_call=True,
)
def save_settings(n, folder, variant, switch_vals, switch_ids, input_vals, input_ids):
    if not folder:
        return dbc.Badge("No project selected", color="warning")

    df = dl.load_variant(folder, "settings_input", variant)
    if df.empty:
        return dbc.Badge("Could not load settings", color="danger")

    updates = {}
    for sid, sval in zip(switch_ids, switch_vals):
        updates[sid["index"]] = 1 if sval else 0
    for iid, ival in zip(input_ids, input_vals):
        if ival is not None and str(ival).strip() != "":
            updates[iid["index"]] = ival

    df_new = df.copy()
    for i, row in df_new.iterrows():
        abbr = _clean(row.get("Abbreviation", ""))
        if abbr in updates:
            df_new.at[i, "Value"] = updates[abbr]

    ok = dl.save_variant(folder, "settings_input", variant, df_new)
    return dbc.Badge("Saved ✓", color="success") if ok else dbc.Badge("Failed", color="danger")


@callback(Output("stg-dup-msg", "children"),
          Output("stg-variant", "options", allow_duplicate=True),
          Output("stg-variant", "value", allow_duplicate=True),
          Input("stg-dup-btn", "n_clicks"),
          State("stg-variant", "value"), State("stg-dup-name", "value"),
          State("settings-project", "value"), prevent_initial_call=True)
def dup_settings(n, variant, new_name, folder):
    from dash import no_update
    if not new_name or not folder: return "Enter a name", no_update, no_update
    new_name = new_name.strip()
    ok = dl.duplicate_variant(folder, "settings_input", variant, new_name)
    if ok:
        return "Created ✓", variant_options(folder, "settings_input"), new_name
    return "Name exists or error", no_update, no_update


@callback(Output("open-file-store", "data", allow_duplicate=True),
          Input("settings-open-btn", "n_clicks"),
          State("settings-project", "value"), prevent_initial_call=True)
def open_settings_folder(n, folder):
    from dash import no_update
    if not n or not folder: return no_update
    return str(INPUT_ROOT / folder / "pSettings.csv")
