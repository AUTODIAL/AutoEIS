"""Page 3 – Run Fitting & Bayesian Inference."""

import dash
import numpy as np
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

import task_manager
from utils import nyquist_figure, store_decode

dash.register_page(__name__, path="/fitting", name="Fitting")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3(
            [html.I(className="fa fa-flask me-2 text-primary"), "Step 3 — Run Fitting & Bayesian Inference"],
            className="mb-3",
        ),

        # Config summary
        dbc.Card(
            dbc.CardBody(html.Div(id="fit-config-summary")),
            className="mb-3",
        ),

        # Run / cancel buttons
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-play me-2"), "Run Analysis"],
                        id="btn-run",
                        color="success",
                        size="lg",
                        n_clicks=0,
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-stop me-2"), "Cancel"],
                        id="btn-cancel",
                        color="danger",
                        size="lg",
                        outline=True,
                        disabled=True,
                    )
                ),
            ],
            className="mb-4 g-2",
        ),

        # Progress area
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id="progress-stage", className="fw-semibold text-muted"),
                                md=9,
                            ),
                            dbc.Col(
                                html.Div(id="progress-pct", className="text-end text-muted"),
                                md=3,
                            ),
                        ]
                    ),
                    dbc.Progress(id="progress-bar", value=0, striped=True, animated=False, className="mt-2"),
                    html.Div(id="progress-alert", className="mt-3"),
                ]
            ),
            id="progress-card",
            className="mb-3",
        ),

        # Candidate circuits table (populated as search runs)
        dbc.Collapse(
            [
                html.H5("Candidate Circuits Found", className="fw-semibold mt-3 mb-2"),
                html.Div(id="circuits-table"),
            ],
            id="circuits-collapse",
            is_open=False,
        ),

        # Live Nyquist preview (best circuit fit after inference)
        dbc.Collapse(
            [
                html.H5("Fit Preview", className="fw-semibold mt-3 mb-2"),
                dcc.Graph(id="fit-preview-graph"),
            ],
            id="fit-preview-collapse",
            is_open=False,
        ),

        html.Hr(),

        # Navigation
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-arrow-left me-2"), "Back"],
                        href="/mode",
                        color="secondary",
                        outline=True,
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-chart-bar me-2"), "View Results"],
                        id="btn-to-results",
                        href="/results",
                        color="primary",
                        className="float-end",
                        disabled=True,
                    )
                ),
            ]
        ),

        # Polling interval — always running (lightweight: returns fast when no task)
        dcc.Interval(id="poll-interval", interval=1500, disabled=False, n_intervals=0),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("fit-config-summary", "children"),
    Input("store-data", "data"),
    Input("store-mode", "data"),
)
def show_summary(data, mode):
    items = []
    if data:
        freq, Z = store_decode(data)
        items += [
            dbc.Badge(f"Data: {len(freq)} points", color="primary", className="me-2"),
            dbc.Badge(
                f"Freq: {freq.min():.2e}–{freq.max():.2e} Hz",
                color="secondary",
                className="me-2",
            ),
        ]
    if mode:
        m = mode.get("mode", "auto")
        items.append(dbc.Badge(f"Mode: {m}", color="info", className="me-2"))
        if m == "auto":
            p = mode.get("params", {})
            items += [
                dbc.Badge(f"iters={p.get('iters', '?')}", color="light", text_color="dark", className="me-1"),
                dbc.Badge(f"complexity={p.get('complexity', '?')}", color="light", text_color="dark", className="me-1"),
                dbc.Badge(f"pop={p.get('population_size', '?')}", color="light", text_color="dark", className="me-1"),
            ]
        elif m == "expert":
            items.append(dbc.Badge(mode.get("circuit_str", "?"), color="light", text_color="dark"))
        elif m == "quick":
            for c in mode.get("circuits", []):
                items.append(dbc.Badge(c, color="light", text_color="dark", className="me-1"))

    if not items:
        return html.P("No data or mode configured yet. Go back to complete previous steps.", className="text-muted")
    return html.Div(items)


@callback(
    Output("store-task-id", "data"),
    Output("btn-run", "disabled"),
    Output("btn-cancel", "disabled"),
    Input("btn-run", "n_clicks"),
    State("store-data", "data"),
    State("store-mode", "data"),
    prevent_initial_call=True,
)
def start_run(n_clicks, data, mode):
    if not n_clicks or data is None or mode is None:
        return dash.no_update, dash.no_update, dash.no_update

    tid = task_manager.create_task()
    if tid is None:
        tid = task_manager.create_error_task(
            f"Server is busy — the maximum number of concurrent analyses "
            f"({task_manager.MAX_CONCURRENT}) is already running. "
            "Please try again in a few minutes."
        )
        return tid, False, True  # run btn stays enabled, cancel disabled

    freq, Z = store_decode(data)
    task_manager.run_analysis(tid, freq, Z, mode)
    return tid, True, False


@callback(
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("btn-cancel", "n_clicks"),
    State("store-task-id", "data"),
    prevent_initial_call=True,
)
def cancel_run(n_clicks, tid):
    if tid:
        task_manager._update(tid, status="error", error="Cancelled by user.")
    # Keep interval alive so poll_progress picks up "error" status and updates the UI.
    # poll_progress will disable the interval itself when it sees status=="error".
    return dash.no_update


@callback(
    Output("progress-bar", "value"),
    Output("progress-bar", "animated"),
    Output("progress-bar", "color"),
    Output("progress-stage", "children"),
    Output("progress-pct", "children"),
    Output("progress-alert", "children"),
    Output("circuits-table", "children"),
    Output("circuits-collapse", "is_open"),
    Output("fit-preview-graph", "figure"),
    Output("fit-preview-collapse", "is_open"),
    Output("btn-to-results", "disabled"),
    Output("store-results", "data"),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("btn-run", "disabled", allow_duplicate=True),
    Output("btn-cancel", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("store-task-id", "data"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def poll_progress(n, tid, data):
    no_change = (
        dash.no_update, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        dash.no_update, dash.no_update,
    )
    if not tid:
        return no_change

    state = task_manager.get_task(tid)
    if not state:
        return no_change

    progress = state.get("progress", 0)
    stage = state.get("stage", "")
    status = state.get("status", "pending")
    circuits_data = state.get("circuits")
    results_data = state.get("results")

    # Progress bar appearance
    if status == "done":
        bar_color = "success"
        animated = False
    elif status == "error":
        bar_color = "danger"
        animated = False
    else:
        bar_color = "info"
        animated = True

    # Alert
    alert = dash.no_update
    if status == "error":
        alert = dbc.Alert(
            [html.Strong("Error: "), html.Pre(state.get("error", "Unknown error"), style={"whiteSpace": "pre-wrap"})],
            color="danger",
            dismissable=True,
        )

    # Circuits table
    circuits_table = dash.no_update
    circuits_open = dash.no_update
    if circuits_data:
        rows = [
            {"Circuit": r["circuitstring"], "N params": len(r.get("Parameters", {}))}
            for r in circuits_data
        ]
        circuits_table = _make_circuits_table(rows)
        circuits_open = True

    # Fit preview
    preview_fig = dash.no_update
    preview_open = dash.no_update
    store_results = dash.no_update
    results_btn_disabled = dash.no_update

    if results_data and data:
        store_results = results_data
        results_btn_disabled = False
        # Show first converged result as preview
        freq_d, Z_d = store_decode(data)
        for r in results_data:
            if r and r.get("converged"):
                try:
                    import autoeis as ae
                    circuit_fn = ae.utils.generate_circuit_fn(r["circuit"])
                    variables = r["variables"]
                    samples = r["samples"]
                    p_med = [float(np.median(samples[v])) for v in variables]
                    Z_fit = circuit_fn(freq_d, p_med)
                    preview_fig = nyquist_figure(
                        freq_d, Z_d, Z_fit=Z_fit, title=f"Best fit: {r['circuit']}"
                    )
                    preview_open = True
                except Exception:
                    pass
                break

    # Stop polling when done or error
    stop_poll = status in ("done", "error")

    return (
        progress,
        animated,
        bar_color,
        stage,
        f"{progress}%",
        alert,
        circuits_table,
        circuits_open,
        preview_fig,
        preview_open,
        results_btn_disabled,
        store_results,
        stop_poll,
        status not in ("done", "error"),
        status in ("done", "error"),
    )


def _make_circuits_table(rows):
    from dash import dash_table as dt
    return dt.DataTable(
        data=rows,
        columns=[
            {"name": "Circuit string", "id": "Circuit"},
            {"name": "# Parameters", "id": "N params"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "fontFamily": "monospace", "padding": "4px 8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        page_size=15,
    )
