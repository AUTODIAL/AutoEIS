"""Page 1 – Data Upload & Preprocessing."""

import json

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html
from dash import dash_table as dt
import dash_bootstrap_components as dbc

from utils import (
    bode_figure,
    guess_columns,
    nyquist_compare_figure,
    nyquist_figure,
    parse_uploaded_file,
    store_decode,
    store_encode,
)

dash.register_page(__name__, path="/", name="Data")

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _upload_card():
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Upload(
                    id="upload-eis",
                    children=html.Div(
                        [
                            html.I(className="fa fa-upload me-2"),
                            "Drag & drop or ",
                            html.A("select a file", className="text-primary fw-bold"),
                            "  (CSV / TXT / XLSX)",
                        ]
                    ),
                    style={
                        "width": "100%",
                        "height": "70px",
                        "lineHeight": "70px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "8px",
                        "borderColor": "#6c757d",
                        "textAlign": "center",
                        "cursor": "pointer",
                    },
                    multiple=False,
                ),
                html.Div(id="col-mapping-row", className="mt-3"),
            ]
        ),
        className="mb-3",
    )


def _col_mapping_row(columns):
    opts = [{"label": c, "value": c} for c in columns]
    guess = guess_columns(columns)
    return dbc.Row(
        [
            # Frequency
            dbc.Col(
                [
                    dbc.Label("Frequency", className="fw-semibold"),
                    dcc.Dropdown(id="col-freq", options=opts, value=guess["freq"], clearable=False),
                ],
                md=2,
            ),
            # Column 2: Re(Z) or Ewe_mod
            dbc.Col(
                [
                    dbc.Label(id="label-col2", children="Re(Z)", className="fw-semibold"),
                    dcc.Dropdown(id="col-zreal", options=opts, value=guess["zreal"], clearable=False),
                ],
                md=2,
            ),
            # Column 3: Im(Z) or I_mod
            dbc.Col(
                [
                    dbc.Label(id="label-col3", children="Im(Z)", className="fw-semibold"),
                    dcc.Dropdown(id="col-zimag", options=opts, value=guess["zimag"], clearable=False),
                ],
                md=2,
            ),
            # Column 4: Phase — only visible in magphase mode
            html.Div(
                dbc.Col(
                    [
                        dbc.Label(id="label-col4", children="Phase", className="fw-semibold"),
                        dcc.Dropdown(id="col-phase", options=opts, value=None, clearable=False),
                    ],
                    md=12,
                ),
                id="col4-container",
                style={"display": "none", "width": "16.6%", "paddingRight": "8px"},
            ),
            # Input format + phase unit
            dbc.Col(
                [
                    dbc.Label("Input format", className="fw-semibold"),
                    dbc.RadioItems(
                        id="format-select",
                        options=[
                            {"label": "Re(Z) / Im(Z)", "value": "reimag"},
                            {"label": "Ewe/I/Phase", "value": "magphase"},
                        ],
                        value="reimag",
                        inline=True,
                        className="small",
                    ),
                    # Phase unit: visible in magphase mode
                    html.Div(
                        dbc.Select(
                            id="phase-unit",
                            options=[
                                {"label": "Radians", "value": "rad"},
                                {"label": "Degrees (°)", "value": "deg"},
                            ],
                            value="rad",
                            size="sm",
                        ),
                        id="phase-unit-container",
                        style={"display": "none"},
                        className="mt-1",
                    ),
                ],
                md=2,
            ),
            # Options
            dbc.Col(
                [
                    dbc.Label("Options", className="fw-semibold"),
                    dbc.Checklist(
                        id="check-negate",
                        options=[{"label": "Negate phase", "value": "negate"}],
                        value=[],
                        switch=True,
                    ),
                ],
                md=1,
            ),
            dbc.Col(
                dbc.Button("Load →", id="btn-load", color="primary", className="w-100 mt-4"),
                md=1,
            ),
        ],
        className="g-2 align-items-end",
    )


def _preprocess_card():
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Button(
                    [html.I(className="fa fa-sliders me-2"), "Data Preprocessing"],
                    id="pp-toggle",
                    color="link",
                    className="text-dark fw-bold p-0",
                )
            ),
            dbc.Collapse(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # KK Validation
                                dbc.Col(
                                    [
                                        html.H6(
                                            [html.I(className="fa fa-check-circle me-1 text-success"), "KK Validation"],
                                            className="fw-semibold",
                                        ),
                                        dbc.Checklist(
                                            id="check-kk",
                                            options=[{"label": "Enable Lin-KK", "value": "enable"}],
                                            value=["enable"],
                                            switch=True,
                                        ),
                                        dbc.Label("Tolerance threshold:", className="mt-2 small"),
                                        dcc.Slider(
                                            id="slider-kk",
                                            min=0.01, max=0.5, step=0.01, value=0.05,
                                            marks={0.01: "0.01", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                    md=4,
                                ),
                                # Frequency Range
                                dbc.Col(
                                    [
                                        html.H6(
                                            [html.I(className="fa fa-cut me-1 text-warning"), "Frequency Range Crop"],
                                            className="fw-semibold",
                                        ),
                                        dbc.Checklist(
                                            id="check-freq-range",
                                            options=[{"label": "Enable freq crop", "value": "enable"}],
                                            value=[],
                                            switch=True,
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col([
                                                    dbc.Label("Min (Hz)", className="small"),
                                                    dbc.Input(id="input-freq-min", type="number", min=0, step=0.001, size="sm", placeholder="auto"),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.Label("Max (Hz)", className="small"),
                                                    dbc.Input(id="input-freq-max", type="number", min=0, step=1, size="sm", placeholder="auto"),
                                                ], md=6),
                                            ],
                                            className="g-1 mt-2",
                                        ),
                                        html.Div(id="freq-range-label", className="text-muted small mt-1"),
                                    ],
                                    md=4,
                                ),
                                # Heuristic Filtering
                                dbc.Col(
                                    [
                                        html.H6(
                                            [html.I(className="fa fa-magic me-1 text-info"), "Heuristic Filtering"],
                                            className="fw-semibold",
                                        ),
                                        dbc.Checklist(
                                            id="check-heuristic",
                                            options=[{"label": "Enable (H1 + H2)", "value": "enable"}],
                                            value=["enable"],
                                            switch=True,
                                        ),
                                        html.Small(
                                            [
                                                html.Strong("H1:"), " discard invalid high-freq points (min |Im(Z)|)",
                                                html.Br(),
                                                html.Strong("H2:"), " remove +Im(Z) at high freq",
                                            ],
                                            className="text-muted d-block mt-1",
                                        ),
                                        dbc.Label("High-freq threshold (Hz):", className="mt-2 small"),
                                        dbc.Input(
                                            id="input-hf-threshold",
                                            type="number",
                                            value=1000,
                                            min=1,
                                            step=1,
                                            size="sm",
                                        ),
                                    ],
                                    md=4,
                                ),
                            ]
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            [html.I(className="fa fa-play me-2"), "Apply Preprocessing"],
                                            id="btn-preprocess",
                                            color="success",
                                            className="me-2",
                                        ),
                                        dbc.Button(
                                            [html.I(className="fa fa-undo me-2"), "Reset"],
                                            id="btn-reset-pp",
                                            color="secondary",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        # Before / After comparison
                        dbc.Collapse(
                            [
                                html.Hr(),
                                html.H6("Before vs. After", className="fw-semibold mt-2"),
                                dcc.Graph(id="compare-graph", style={"height": "400px"}),
                                html.Div(id="pp-stats", className="text-muted small"),
                            ],
                            id="compare-collapse",
                            is_open=False,
                        ),
                        # Lin-KK residuals
                        dbc.Collapse(
                            [
                                html.Hr(),
                                html.H6(
                                    [html.I(className="fa fa-check-circle me-1 text-success"),
                                     "Lin-KK Validation Residuals"],
                                    className="fw-semibold mt-2",
                                ),
                                html.Small(
                                    "Relative residuals from Kramers-Kronig validation. "
                                    "Points outside ±5% (or your chosen tolerance) were removed.",
                                    className="text-muted d-block mb-2",
                                ),
                                dcc.Graph(id="kk-residual-plot", style={"height": "300px"}),
                            ],
                            id="kk-residual-collapse",
                            is_open=False,
                        ),
                    ]
                ),
                id="pp-collapse",
                is_open=True,
            ),
        ],
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3(
            [html.I(className="fa fa-database me-2 text-primary"), "Step 1 — Load & Preprocess Data"],
            className="mb-3",
        ),
        _upload_card(),
        html.Div(id="upload-alert"),

        # Main content (hidden until data is loaded)
        html.Div(
            id="data-main",
            style={"display": "none"},
            children=[
                dbc.Row(
                    [
                        # Left: plot
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(label="Nyquist", tab_id="nyquist"),
                                                dbc.Tab(label="Bode", tab_id="bode"),
                                            ],
                                            id="plot-tabs",
                                            active_tab="nyquist",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="eis-plot",
                                                config={"displayModeBar": True, "modeBarButtonsToAdd": ["select2d", "lasso2d"]},
                                                style={"height": "460px"},
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        [html.I(className="fa fa-trash me-1"), "Remove Selected"],
                                                        id="btn-remove-pts",
                                                        color="danger",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        [html.I(className="fa fa-undo me-1"), "Undo Last"],
                                                        id="btn-undo",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        [html.I(className="fa fa-refresh me-1"), "Reset All"],
                                                        id="btn-reset-all",
                                                        color="warning",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="mt-2",
                                                size="sm",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            md=7,
                        ),
                        # Right: table
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Data Table"),
                                    dbc.CardBody(
                                        [
                                            dt.DataTable(
                                                id="data-table",
                                                columns=[
                                                    {"name": "#", "id": "idx", "type": "numeric"},
                                                    {"name": "Freq [Hz]", "id": "freq", "type": "numeric", "format": dt.Format.Format(precision=4, scheme=dt.Format.Scheme.exponent)},
                                                    {"name": "Re(Z) [Ω]", "id": "z_real", "type": "numeric", "format": dt.Format.Format(precision=4, scheme=dt.Format.Scheme.exponent)},
                                                    {"name": "−Im(Z) [Ω]", "id": "z_imag_neg", "type": "numeric", "format": dt.Format.Format(precision=4, scheme=dt.Format.Scheme.exponent)},
                                                ],
                                                row_selectable="multi",
                                                selected_rows=[],
                                                style_table={"height": "400px", "overflowY": "auto"},
                                                page_action="none",
                                                style_cell={"fontSize": "12px", "padding": "4px 8px", "fontFamily": "monospace"},
                                                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                                                style_data_conditional=[
                                                    {
                                                        "if": {"state": "selected"},
                                                        "backgroundColor": "#cfe2ff",
                                                    }
                                                ],
                                                filter_action="native",
                                                sort_action="native",
                                            ),
                                            dbc.Button(
                                                [html.I(className="fa fa-trash me-1"), "Remove Selected Rows"],
                                                id="btn-remove-rows",
                                                color="danger",
                                                size="sm",
                                                className="mt-2",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            md=5,
                        ),
                    ],
                    className="mb-3 g-3",
                ),

                _preprocess_card(),

                # Empirical Rs info card
                html.Div(id="rs-display", className="mb-2"),

                # Summary badge
                html.Div(id="data-summary", className="mb-3"),

                # Navigation
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fa fa-arrow-right me-2"), "Next: Choose Analysis Mode"],
                            href="/mode",
                            color="primary",
                            className="float-end",
                        ),
                    )
                ),
            ],
        ),

        # One-shot interval fires once after page mount to restore display from app-level stores
        dcc.Interval(id="data-page-init", interval=150, max_intervals=1, n_intervals=0),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("data-main", "style"),
    Input("store-raw", "data"),
    Input("data-page-init", "n_intervals"),
)
def toggle_main(raw, _):
    return {"display": "block"} if raw else {"display": "none"}


@callback(
    Output("col-mapping-row", "children", allow_duplicate=True),
    Input("data-page-init", "n_intervals"),
    State("store-df-cols", "data"),
    prevent_initial_call=True,
)
def restore_col_mapping(_, df_cols_json):
    if not df_cols_json:
        return dash.no_update
    meta = json.loads(df_cols_json)
    return _col_mapping_row(meta["cols"])


@callback(
    Output("col-mapping-row", "children"),
    Output("store-df-cols", "data"),
    Output("upload-alert", "children"),
    Input("upload-eis", "contents"),
    State("upload-eis", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update
    df, err = parse_uploaded_file(contents, filename)
    if err:
        alert = dbc.Alert(err, color="danger", dismissable=True)
        return dash.no_update, dash.no_update, alert
    cols = df.columns.tolist()
    return _col_mapping_row(cols), json.dumps({"cols": cols, "data": df.to_json()}), dash.no_update


@callback(
    Output("store-raw", "data"),
    Output("store-deleted", "data"),
    Output("upload-alert", "children", allow_duplicate=True),
    Input("btn-load", "n_clicks"),
    State("store-df-cols", "data"),
    State("col-freq", "value"),
    State("col-zreal", "value"),
    State("col-zimag", "value"),
    State("col-phase", "value"),
    State("check-negate", "value"),
    State("format-select", "value"),
    State("phase-unit", "value"),
    prevent_initial_call=True,
)
def on_load(n_clicks, df_cols_json, col_freq, col_ewe, col_i, col_phase, negate, fmt, phase_unit):
    if not n_clicks or not df_cols_json:
        return dash.no_update, dash.no_update, dash.no_update

    meta = json.loads(df_cols_json)
    df = pd.read_json(meta["data"])

    if fmt == "magphase":
        required = [col_freq, col_ewe, col_i, col_phase]
    else:
        required = [col_freq, col_ewe, col_i]

    missing = [c for c in required if c and c not in df.columns]
    none_cols = [name for c, name in zip(required, ["freq", "col2", "col3", "phase"]) if not c]
    if missing or none_cols:
        msg = (f"Columns not found: {missing}" if missing else "") + (
            f"  Please select: {none_cols}" if none_cols else ""
        )
        return dash.no_update, dash.no_update, dbc.Alert(msg.strip(), color="danger", dismissable=True)

    freq = df[col_freq].values.astype(float)

    if fmt == "magphase":
        # Exact formula from OT2_functions.impedance_convert:
        #   Zmodulus = Ewe_mod / I_mod
        #   Zreal = cos(phase) * Zmodulus
        #   Zimag = sin(phase) * Zmodulus
        ewe_mod = df[col_ewe].values.astype(float)
        i_mod   = df[col_i].values.astype(float)
        phase   = df[col_phase].values.astype(float)
        if "negate" in (negate or []):
            phase = -phase
        if (phase_unit or "rad") == "deg":
            phase = phase * (np.pi / 180.0)
        zmod   = ewe_mod / i_mod
        z_real = np.cos(phase) * zmod
        z_imag = np.sin(phase) * zmod
    else:
        z_real = df[col_ewe].values.astype(float)
        z_imag = df[col_i].values.astype(float)
        if "negate" in (negate or []):
            z_imag = -z_imag

    Z = z_real + 1j * z_imag

    # Drop rows where freq or Z is NaN/inf (same as OT2_functions NaN handling)
    valid = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    n_dropped = int((~valid).sum())
    freq, Z = freq[valid], Z[valid]

    if len(freq) == 0:
        return dash.no_update, dash.no_update, dbc.Alert(
            "All data points are NaN/inf after parsing. Check column selection.", color="danger", dismissable=True
        )

    raw = store_encode(freq, Z)
    alert = (
        dbc.Alert(f"Loaded {len(freq)} points ({n_dropped} NaN/inf rows dropped).", color="info", dismissable=True, duration=4000)
        if n_dropped else None
    )
    return raw, [], alert


# ---------------------------------------------------------------------------
# Update column labels and phase-unit visibility when format changes
# ---------------------------------------------------------------------------

@callback(
    Output("label-col2", "children"),
    Output("label-col3", "children"),
    Output("col4-container", "style"),
    Output("phase-unit-container", "style"),
    Input("format-select", "value"),
    prevent_initial_call=True,
)
def update_col_labels(fmt):
    if fmt == "magphase":
        return (
            "Ewe_mod (|V|)",
            "I_mod (|I|)",
            {"display": "block", "width": "16.6%", "paddingRight": "8px"},
            {"display": "block"},
        )
    return "Re(Z)", "Im(Z)", {"display": "none", "width": "16.6%", "paddingRight": "8px"}, {"display": "none"}


# ---------------------------------------------------------------------------
# Sync freq-range slider labels and positions from raw data
# ---------------------------------------------------------------------------

@callback(
    Output("input-freq-min", "value"),
    Output("input-freq-max", "value"),
    Output("freq-range-label", "children"),
    Input("store-raw", "data"),
    Input("data-page-init", "n_intervals"),
    prevent_initial_call=True,
)
def sync_freq_inputs(raw, _):
    if raw is None:
        return None, None, ""
    freq, _ = store_decode(raw)
    freq_pos = freq[freq > 0]
    if len(freq_pos) == 0:
        return None, None, ""
    fmin = float(freq_pos.min())
    fmax = float(freq_pos.max())
    label = f"Data range: {fmin:.3g} – {fmax:.3g} Hz"
    return fmin, fmax, label


# ---------------------------------------------------------------------------
# Toggle preprocessing collapse
# ---------------------------------------------------------------------------

@callback(
    Output("pp-collapse", "is_open"),
    Input("pp-toggle", "n_clicks"),
    State("pp-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_pp(n, is_open):
    return not is_open


# ---------------------------------------------------------------------------
# Build pending-delete set from plot click / lasso-select
# ---------------------------------------------------------------------------

@callback(
    Output("store-pending-delete", "data"),
    Input("eis-plot", "clickData"),
    Input("eis-plot", "selectedData"),
    State("store-pending-delete", "data"),
    State("store-deleted", "data"),
    prevent_initial_call=True,
)
def update_pending_from_plot(click_data, selected_data, pending, deleted):
    pending = list(pending or [])
    deleted_set = set(deleted or [])
    trigger_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else None

    if trigger_prop == "eis-plot.clickData":
        if click_data and click_data.get("points"):
            pt = click_data["points"][0]
            if "customdata" in pt:
                idx = int(pt["customdata"])
                if idx not in deleted_set:
                    if idx in pending:
                        pending.remove(idx)  # toggle off
                    else:
                        pending.append(idx)  # toggle on

    elif trigger_prop == "eis-plot.selectedData":
        if selected_data and selected_data.get("points"):
            for pt in selected_data["points"]:
                if "customdata" in pt:
                    idx = int(pt["customdata"])
                    if idx not in deleted_set and idx not in pending:
                        pending.append(idx)

    return pending


# ---------------------------------------------------------------------------
# Manage deleted indices (pending flush / table select / undo / reset)
# ---------------------------------------------------------------------------

@callback(
    Output("store-deleted", "data", allow_duplicate=True),
    Output("store-pending-delete", "data", allow_duplicate=True),
    Input("btn-remove-pts", "n_clicks"),
    Input("btn-remove-rows", "n_clicks"),
    Input("btn-undo", "n_clicks"),
    Input("btn-reset-all", "n_clicks"),
    State("store-pending-delete", "data"),
    State("data-table", "selected_rows"),
    State("data-table", "data"),
    State("store-deleted", "data"),
    prevent_initial_call=True,
)
def manage_deletions(
    n_remove_pts, n_remove_rows, n_undo, n_reset,
    pending, selected_table_rows, table_data, deleted
):
    deleted = list(deleted or [])
    pending = list(pending or [])
    trigger = ctx.triggered_id

    if trigger == "btn-reset-all":
        return [], []

    if trigger == "btn-undo" and deleted:
        return deleted[:-1], pending

    if trigger == "btn-remove-pts":
        for i in pending:
            if i not in deleted:
                deleted.append(i)
        return deleted, []  # clear pending after committing

    if trigger == "btn-remove-rows" and selected_table_rows and table_data:
        for row_num in selected_table_rows:
            orig_idx = table_data[row_num]["idx"]
            if orig_idx not in deleted:
                deleted.append(orig_idx)

    return deleted, pending


# ---------------------------------------------------------------------------
# Render main EIS plot
# ---------------------------------------------------------------------------

@callback(
    Output("eis-plot", "figure"),
    Input("plot-tabs", "active_tab"),
    Input("store-raw", "data"),
    Input("store-deleted", "data"),
    Input("store-pending-delete", "data"),
    Input("data-page-init", "n_intervals"),
    prevent_initial_call=True,
)
def render_plot(tab, raw, deleted, pending, _):
    if raw is None:
        return {}
    freq, Z = store_decode(raw)
    if tab == "bode":
        return bode_figure(freq, Z, deleted=deleted, pending=pending)
    return nyquist_figure(freq, Z, deleted=deleted, pending=pending)


# ---------------------------------------------------------------------------
# Render data table (reflects deletions)
# ---------------------------------------------------------------------------

@callback(
    Output("data-table", "data"),
    Output("data-table", "selected_rows"),
    Input("store-raw", "data"),
    Input("store-deleted", "data"),
    Input("data-page-init", "n_intervals"),
    prevent_initial_call=True,
)
def render_table(raw, deleted, _):
    if raw is None:
        return [], []
    freq, Z = store_decode(raw)
    deleted_set = set(deleted or [])
    rows = [
        {
            "idx": i,
            "freq": float(freq[i]),
            "z_real": float(Z.real[i]),
            "z_imag_neg": float(-Z.imag[i]),
        }
        for i in range(len(freq))
        if i not in deleted_set
    ]
    return rows, []


# ---------------------------------------------------------------------------
# Apply preprocessing → update main store + show comparison
# ---------------------------------------------------------------------------

def _apply_eis_heuristics(freq, Z, high_freq_threshold=1e3):
    """Port of OT2_functions.apply_eis_heuristics."""
    freq = np.asarray(freq)
    Z = np.asarray(Z)
    idx_sort = np.argsort(freq)[::-1]
    freq, Z = freq[idx_sort], Z[idx_sort]

    # H1: at very high freq Im(Z)→0; keep from the min-|Im| point downward
    high_freq_mask = freq > high_freq_threshold
    if np.any(high_freq_mask):
        hf_indices = np.where(high_freq_mask)[0]
        idx_local = np.argmin(np.abs(Z.imag[high_freq_mask]))
        idx_global = hf_indices[idx_local]
        freq, Z = freq[idx_global:], Z[idx_global:]

    # H2: remove +Im(Z) points at high freq
    high_freq_mask = freq > high_freq_threshold
    if np.any(high_freq_mask):
        remove = high_freq_mask & (Z.imag > 0)
        freq, Z = freq[~remove], Z[~remove]

    return freq, Z


def _empirical_rs(freq, Z, n_highfreq=30):
    """Return (Rs, freq_at_Rs) using the min-|Im(Z)| heuristic."""
    idx_sorted = np.argsort(freq)[::-1]
    idx_high = idx_sorted[: min(n_highfreq, len(freq))]
    Z_high = Z[idx_high]
    freq_high = freq[idx_high]
    idx_min = np.argmin(np.abs(Z_high.imag))
    return float(Z_high.real[idx_min]), float(freq_high[idx_min])


@callback(
    Output("store-data", "data"),
    Output("compare-graph", "figure"),
    Output("compare-collapse", "is_open"),
    Output("pp-stats", "children"),
    Output("kk-residual-plot", "figure"),
    Output("kk-residual-collapse", "is_open"),
    Input("btn-preprocess", "n_clicks"),
    Input("btn-reset-pp", "n_clicks"),
    State("store-raw", "data"),
    State("store-deleted", "data"),
    State("check-kk", "value"),
    State("slider-kk", "value"),
    State("check-freq-range", "value"),
    State("input-freq-min", "value"),
    State("input-freq-max", "value"),
    State("check-heuristic", "value"),
    State("input-hf-threshold", "value"),
    prevent_initial_call=True,
)
def apply_preprocess(n_apply, n_reset, raw, deleted, kk_on, kk_tol,
                     freq_range_on, freq_min, freq_max, heuristic_on, hf_threshold):
    import plotly.graph_objects as go
    from autoeis.utils import preprocess_impedance_data

    trigger = ctx.triggered_id

    if raw is None:
        return dash.no_update, {}, False, "", {}, False

    freq_raw, Z_raw = store_decode(raw)

    # Apply manual deletions
    mask = np.ones(len(freq_raw), dtype=bool)
    for i in (deleted or []):
        if i < len(mask):
            mask[i] = False
    freq = freq_raw[mask]
    Z = Z_raw[mask]

    if trigger == "btn-reset-pp":
        encoded = store_encode(freq, Z)
        encoded["freq_raw"] = freq_raw.tolist()
        encoded["z_real_raw"] = Z_raw.real.tolist()
        encoded["z_imag_raw"] = Z_raw.imag.tolist()
        return encoded, {}, False, "Preprocessing reset.", {}, False

    # Step 1: Frequency range crop (direct Hz values)
    if "enable" in (freq_range_on or []):
        lo = float(freq_min) if freq_min is not None else freq.min()
        hi = float(freq_max) if freq_max is not None else freq.max()
        m = (freq >= lo) & (freq <= hi)
        freq, Z = freq[m], Z[m]

    freq_before, Z_before = freq.copy(), Z.copy()

    # Step 2: Heuristic filtering (H1 + H2)
    if "enable" in (heuristic_on or []):
        threshold = float(hf_threshold or 1e3)
        freq, Z = _apply_eis_heuristics(freq, Z, high_freq_threshold=threshold)

    # Step 3: KK validation — capture residuals when enabled
    kk_fig = {}
    kk_open = False
    if "enable" in (kk_on or []):
        try:
            hf_thresh = float("inf") if "enable" in (heuristic_on or []) else float(hf_threshold or 1e3)
            freq, Z, aux = preprocess_impedance_data(
                freq, Z,
                tol_linKK=float(kk_tol),
                high_freq_threshold=hf_thresh,
                return_aux=True,
            )
            # Build KK residual plot (Plotly)
            try:
                res_real = np.array(aux.res.real) * 100
                res_imag = np.array(aux.res.imag) * 100
                kk_freq = freq  # residuals correspond to retained points
                tol_pct = float(kk_tol) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=kk_freq, y=res_real, mode="markers+lines",
                    name="Re residual [%]",
                    marker=dict(color="#1f77b4", size=5),
                    line=dict(width=1),
                ))
                fig.add_trace(go.Scatter(
                    x=kk_freq, y=res_imag, mode="markers+lines",
                    name="Im residual [%]",
                    marker=dict(color="#d62728", size=5),
                    line=dict(width=1),
                ))
                fig.add_hline(y=tol_pct, line_dash="dash", line_color="gray",
                              annotation_text=f"+{tol_pct:.0f}%", annotation_position="top right")
                fig.add_hline(y=-tol_pct, line_dash="dash", line_color="gray",
                              annotation_text=f"−{tol_pct:.0f}%", annotation_position="bottom right")
                fig.add_hline(y=0, line_color="black", opacity=0.3, line_width=1)
                fig.update_xaxes(type="log", title_text="Frequency [Hz]")
                fig.update_yaxes(title_text="Relative residual [%]")
                fig.update_layout(
                    title=dict(text="Lin-KK Validation Residuals", x=0.5),
                    hovermode="x unified",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=50),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
                )
                kk_fig = fig
                kk_open = True
            except Exception:
                pass
        except Exception as exc:
            return dash.no_update, {}, False, dbc.Alert(str(exc), color="danger"), {}, False

    n_before = len(freq_before)
    n_after = len(freq)
    pct_kept = 100 * n_after / max(n_before, 1)

    # Empirical Rs on cleaned data
    rs_text = ""
    if len(freq) > 0:
        try:
            rs, f_rs = _empirical_rs(freq, Z)
            rs_text = f"  |  ⚡ Empirical Rs ≈ {rs:.4g} Ω  (at {f_rs:.3g} Hz)"
        except Exception:
            pass

    encoded = store_encode(freq, Z)
    encoded["freq_raw"] = freq_raw.tolist()
    encoded["z_real_raw"] = Z_raw.real.tolist()
    encoded["z_imag_raw"] = Z_raw.imag.tolist()

    compare_fig = nyquist_compare_figure(freq_before, Z_before, freq, Z)
    stats_msg = (
        f"Before: {n_before} points → After: {n_after} points "
        f"({pct_kept:.0f}% retained){rs_text}"
    )

    return encoded, compare_fig, True, stats_msg, kk_fig, kk_open


# ---------------------------------------------------------------------------
# Empirical Rs display (updates on raw load and after preprocessing)
# ---------------------------------------------------------------------------

@callback(
    Output("rs-display", "children"),
    Input("store-raw", "data"),
    Input("store-data", "data"),
    Input("data-page-init", "n_intervals"),
    prevent_initial_call=True,
)
def show_rs(raw, processed, _):
    source = processed if processed else raw
    if source is None:
        return ""
    try:
        freq, Z = store_decode(source)
        rs, f_rs = _empirical_rs(freq, Z)
        return dbc.Alert(
            [
                html.I(className="fa fa-bolt me-2 text-warning"),
                html.Strong("Empirical Rs ≈ "),
                f"{rs:.4g} Ω",
                html.Span(f"  (at {f_rs:.3g} Hz, high-freq min |Im(Z)| point)", className="text-muted ms-2 small"),
            ],
            color="light",
            className="py-2 mb-0",
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Data summary badge
# ---------------------------------------------------------------------------

@callback(
    Output("data-summary", "children"),
    Input("store-data", "data"),
    Input("store-raw", "data"),
    Input("data-page-init", "n_intervals"),
    prevent_initial_call=True,
)
def update_summary(processed, raw, _):
    source = processed if processed else raw
    if source is None:
        return ""
    freq, Z = store_decode(source)
    if len(freq) == 0:
        return dbc.Badge("0 points (empty data)", color="danger")
    badges = [
        dbc.Badge(f"{len(freq)} points", color="primary", className="me-1"),
        dbc.Badge(
            f"f: {freq.min():.2e} – {freq.max():.2e} Hz",
            color="secondary",
            className="me-1",
        ),
        dbc.Badge(
            "Preprocessed ✓" if processed and "freq_raw" in processed else "Raw data",
            color="success" if processed and "freq_raw" in processed else "warning",
        ),
    ]
    return html.Div(badges, className="mt-1")
