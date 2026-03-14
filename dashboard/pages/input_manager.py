"""Input Manager — project selector shown at top of all input pages."""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import data_loader as dl


def layout(active_project=None):
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]

    return html.Div([
        html.H4("Input Manager", className="mb-3"),
        html.P("Select a project to edit its input data, or create a new one from a template.",
               className="text-muted"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Active project"),
                dcc.Dropdown(id="active-project-dropdown", options=options,
                             value=active_project or (folders[0] if folders else None),
                             clearable=False),
            ], width=4),
            dbc.Col([
                dbc.Label("Clone from"),
                dcc.Dropdown(id="clone-src-mgr", options=options,
                             placeholder="Source folder…"),
            ], width=3),
            dbc.Col([
                dbc.Label("New name"),
                dbc.Input(id="clone-name-mgr", placeholder="data_newproject"),
            ], width=3),
            dbc.Col([
                dbc.Label("\u00a0"),
                dbc.Button("Clone", id="clone-btn-mgr", color="success",
                           className="d-block"),
            ], width=2),
        ], className="mb-3 align-items-end"),

        html.Div(id="clone-result-mgr"),
        html.Hr(),

        html.Div(id="project-file-tree"),
    ])


@callback(
    Output("clone-result-mgr",    "children"),
    Output("active-project-dropdown", "options"),
    Input("clone-btn-mgr",        "n_clicks"),
    State("clone-src-mgr",        "value"),
    State("clone-name-mgr",       "value"),
    prevent_initial_call=True,
)
def clone(n, source, name):
    if not source or not name:
        return dbc.Alert("Source and name required.", color="warning"), []
    name = name.strip()
    if not name.startswith("data_"):
        name = "data_" + name
    ok = dl.clone_input_folder(source, name)
    folders = dl.list_input_folders()
    options = [{"label": f, "value": f} for f in folders]
    if ok:
        return dbc.Alert(f"'{name}' created.", color="success"), options
    return dbc.Alert("Failed — already exists or source not found.", color="danger"), options


@callback(
    Output("project-file-tree", "children"),
    Input("active-project-dropdown", "value"),
)
def show_file_tree(folder):
    if not folder:
        return html.Div()
    files = dl.list_input_files(folder)
    items = []
    for subfolder, fnames in sorted(files.items()):
        items.append(html.Li([
            html.Strong(subfolder),
            html.Ul([html.Li(f, className="text-muted small") for f in sorted(fnames)])
        ]))
    return html.Div([
        html.H6(f"Files in {folder}"),
        html.Ul(items),
    ])
