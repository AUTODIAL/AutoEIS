"""Page 4 – Results: ranking, parameter posteriors, plots, export."""

import base64
import io
import json

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash import dash_table as dt
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import bode_figure, mpl_fig_to_src, nyquist_figure, residual_figure, store_decode
from eis_posterior_quality import assess_posterior_quality

dash.register_page(__name__, path="/results", name="Results")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3(
            [html.I(className="fa fa-chart-bar me-2 text-primary"), "Step 4 — Results & Export"],
            className="mb-3",
        ),

        html.Div(id="results-alert"),

        # Ranking table
        dbc.Card(
            [
                dbc.CardHeader(html.H5("ECM Ranking (converged circuits)", className="mb-0")),
                dbc.CardBody(html.Div(id="ranking-table")),
            ],
            className="mb-3",
        ),

        # Detailed view for selected circuit
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.H5("Circuit Detail", className="mb-0"), md=6),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="circuit-selector",
                                    placeholder="Select a circuit…",
                                    clearable=False,
                                    style={"fontSize": "13px"},
                                ),
                                md=6,
                            ),
                        ],
                        align="center",
                    )
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Circuit diagram
                                dbc.Col(
                                    [
                                        html.H6("Circuit Diagram", className="fw-semibold"),
                                        html.Div(id="detail-diagram"),
                                    ],
                                    md=4,
                                ),
                                # Parameter table with plausibility
                                dbc.Col(
                                    [
                                        html.H6("Parameter Posterior Summary", className="fw-semibold"),
                                        html.Div(id="detail-params"),
                                    ],
                                    md=8,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Tabs(
                                            [
                                                dcc.Tab(
                                                    dcc.Graph(id="detail-nyquist", style={"height": "420px"}),
                                                    label="Nyquist",
                                                ),
                                                dcc.Tab(
                                                    dcc.Graph(id="detail-bode", style={"height": "420px"}),
                                                    label="Bode",
                                                ),
                                                dcc.Tab(
                                                    dcc.Graph(id="detail-residual", style={"height": "320px"}),
                                                    label="Residuals",
                                                ),
                                                dcc.Tab(
                                                    dcc.Graph(id="detail-posterior-pred", style={"height": "420px"}),
                                                    label="Posterior Prediction",
                                                ),
                                                dcc.Tab(
                                                    html.Div(id="detail-posterior"),
                                                    label="Parameter Posteriors",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),
                    ]
                ),
            ],
            className="mb-3",
        ),

        # Export section
        dbc.Card(
            [
                dbc.CardHeader(html.H5("Export", className="mb-0")),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        [html.I(className="fa fa-file-csv me-2"), "Export Parameters CSV"],
                                        id="btn-export-csv",
                                        color="success",
                                        outline=True,
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        [html.I(className="fa fa-file-code me-2"), "Export Results JSON"],
                                        id="btn-export-json",
                                        color="info",
                                        outline=True,
                                    ),
                                    md=3,
                                ),
                            ],
                            className="g-2",
                        ),
                        dcc.Download(id="download-csv"),
                        dcc.Download(id="download-json"),
                    ]
                ),
            ],
            className="mb-3",
        ),

        dbc.Button(
            [html.I(className="fa fa-arrow-left me-2"), "Back to Fitting"],
            href="/fitting",
            color="secondary",
            outline=True,
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Status styling for plausibility
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "NORMAL":       "✓",
    "LOGNORMAL":    "✓",
    "UNCERTAIN":    "~",
    "MULTIMODAL":   "✗",
    "BOUNDARY_HIT": "✗",
    "UNINFORMATIVE":"✗",
    "UNIFORM":      "✗",
}


# ---------------------------------------------------------------------------
# Helper: compute fit quality (RMSE) using median posterior parameters
# ---------------------------------------------------------------------------

def _compute_fit(result: dict, freq: np.ndarray, Z: np.ndarray):
    import autoeis as ae
    variables = result["variables"]
    samples = result["samples"]
    p_med = [float(np.median(samples[v])) for v in variables]
    circuit_fn = ae.utils.generate_circuit_fn(result["circuit"])
    Z_fit = circuit_fn(freq, p_med)
    rmse = float(np.sqrt(np.mean(np.abs(Z - Z_fit) ** 2)))
    return Z_fit, rmse, p_med


# ---------------------------------------------------------------------------
# Helper: parameter summary table with plausibility
# ---------------------------------------------------------------------------

def _param_summary_table(result: dict, quality: dict | None = None):
    variables = result["variables"]
    samples = result["samples"]
    per_param = (quality or {}).get("per_parameter", {})

    rows = []
    for v in variables:
        s = np.array(samples[v])
        pq = per_param.get(v, {})
        status = pq.get("status", "")
        passes = pq.get("passes", None)

        if passes is True:
            plaus = "PASS"
        elif passes is False:
            plaus = "FAIL"
        else:
            plaus = ""

        rows.append(
            {
                "Parameter": v,
                "Plausible": plaus,
                "Distribution": status,
                "Median": f"{np.median(s):.4e}",
                "Mean": f"{np.mean(s):.4e}",
                "Std": f"{np.std(s):.4e}",
                "95% CI": f"[{np.percentile(s, 2.5):.3e}, {np.percentile(s, 97.5):.3e}]",
            }
        )

    style_cond = [
        {
            "if": {"filter_query": '{Plausible} = "PASS"', "column_id": "Plausible"},
            "color": "#198754",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": '{Plausible} = "FAIL"', "column_id": "Plausible"},
            "color": "#dc3545",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": '{Plausible} = "FAIL"', "column_id": "Distribution"},
            "color": "#dc3545",
        },
    ]

    legend = html.Div(
        [
            html.Small(html.Strong("Plausibility legend:"), className="text-muted"),
            html.Ul(
                [
                    html.Li([
                        html.Span("PASS", style={"color": "#198754", "fontWeight": "bold"}),
                        html.Span(" — posterior well-constrained: NORMAL (symmetric bell curve), "
                                  "LOGNORMAL (right-skewed, typical for R/C), "
                                  "UNCERTAIN (wide but unimodal)"),
                    ], className="small text-muted"),
                    html.Li([
                        html.Span("FAIL", style={"color": "#dc3545", "fontWeight": "bold"}),
                        html.Span(" — MULTIMODAL: multiple peaks — parameter is non-identifiable"),
                    ], className="small text-muted"),
                    html.Li([
                        html.Span("FAIL", style={"color": "#dc3545", "fontWeight": "bold"}),
                        html.Span(" — BOUNDARY_HIT: chain hit the prior boundary (value at its limit)"),
                    ], className="small text-muted"),
                    html.Li([
                        html.Span("FAIL", style={"color": "#dc3545", "fontWeight": "bold"}),
                        html.Span(" — UNINFORMATIVE / UNIFORM: posterior is flat — data carries no information about this parameter"),
                    ], className="small text-muted"),
                ],
                className="mb-1 ps-3",
            ),
        ],
        className="mb-2",
    )

    table = dt.DataTable(
        data=rows,
        columns=[{"name": k, "id": k} for k in
                 ["Parameter", "Plausible", "Distribution", "Median", "Mean", "Std", "95% CI"]],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "fontFamily": "monospace", "padding": "4px 8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=style_cond,
    )
    return html.Div([legend, table])


# ---------------------------------------------------------------------------
# Helper: posterior prediction Nyquist (grey envelope + measured data)
# ---------------------------------------------------------------------------

def _posterior_prediction_figure(result: dict) -> go.Figure:
    import autoeis as ae

    variables = result["variables"]
    samples = result["samples"]
    freq = np.array(result["freq"])
    Z_meas = np.array(result["z_real"]) + 1j * np.array(result["z_imag"])

    circuit_fn = ae.utils.generate_circuit_fn(result["circuit"])

    n_total = len(samples[variables[0]])
    n_draw = min(150, n_total)
    indices = np.linspace(0, n_total - 1, n_draw, dtype=int)

    x_pred, y_pred = [], []
    for i in indices:
        params = [float(samples[v][i]) for v in variables]
        try:
            Z_sim = circuit_fn(freq, params)
            x_pred.extend(list(Z_sim.real) + [None])
            y_pred.extend(list(-Z_sim.imag) + [None])
        except Exception:
            pass

    fig = go.Figure()

    if x_pred:
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="lines",
                name=f"Posterior samples ({n_draw})",
                line=dict(color="lightgray", width=1),
                opacity=0.6,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=Z_meas.real,
            y=-Z_meas.imag,
            mode="markers",
            name="Measured",
            marker=dict(color="#1f77b4", size=8, opacity=0.9),
        )
    )

    fig.update_layout(
        title=dict(text=f"Posterior Prediction — {result['circuit']}", x=0.5),
        xaxis_title="Re(Z) [Ω]",
        yaxis_title="−Im(Z) [Ω]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode="closest",
        height=420,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: per-parameter posterior histograms with plausibility colouring
# ---------------------------------------------------------------------------

def _posterior_plots(result: dict, quality: dict | None = None):
    variables = result["variables"]
    samples = result["samples"]
    per_param = (quality or {}).get("per_parameter", {})

    n = len(variables)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=variables,
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    for idx, v in enumerate(variables):
        row = idx // cols + 1
        col = idx % cols + 1
        s = np.array(samples[v])

        pq = per_param.get(v, {})
        status = pq.get("status", "NORMAL")
        passes = pq.get("passes", True)
        color = "#198754" if passes else "#dc3545"
        if status == "UNCERTAIN":
            color = "#fd7e14"

        log_scale = bool(np.std(s) / (np.median(s) + 1e-30) > 2)
        x = np.log10(s + 1e-30) if log_scale else s
        xlabel = f"log₁₀({v})" if log_scale else v

        fig.add_trace(
            go.Histogram(
                x=x, nbinsx=40, name=xlabel,
                showlegend=False,
                marker_color=color, opacity=0.75,
            ),
            row=row, col=col,
        )

        if status and status != "NORMAL":
            fig.add_annotation(
                text=f"{_STATUS_ICON.get(status, '?')} {status}",
                xref=f"x{idx + 1 if idx > 0 else ''} domain",
                yref=f"y{idx + 1 if idx > 0 else ''} domain",
                x=0.98, y=0.95,
                showarrow=False,
                font=dict(size=10, color=color),
                align="right",
                row=row, col=col,
            )

    fig.update_layout(
        height=max(260 * rows, 300),
        margin=dict(t=50, b=30, l=40, r=20),
    )
    return dcc.Graph(figure=fig)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("ranking-table", "children"),
    Output("circuit-selector", "options"),
    Output("circuit-selector", "value"),
    Output("results-alert", "children"),
    Input("store-results", "data"),
    State("store-data", "data"),
)
def populate_ranking(results, data):
    if not results:
        msg = dbc.Alert(
            "No results yet. Complete Step 3 first.",
            color="warning",
        )
        return msg, [], None, None

    if not data:
        return dbc.Alert("Data store is empty. Please restart from Step 1.", color="danger"), [], None, None

    freq, Z = store_decode(data)

    rows = []
    for i, r in enumerate(results):
        if r is None or not r.get("converged"):
            rows.append(
                {
                    "Rank": i + 1,
                    "Circuit": (r or {}).get("circuit", "N/A"),
                    "Converged": "✗",
                    "RMSE [Ω]": "—",
                    "# Params": "—",
                    "Bad Params": "—",
                }
            )
            continue
        try:
            _, rmse, _ = _compute_fit(r, freq, Z)
        except Exception:
            rmse = float("nan")

        # Count implausible parameters
        n_bad = 0
        try:
            q = assess_posterior_quality(
                r["samples"],
                r["variables"],
                num_divergences=r.get("num_divergences", 0),
                num_samples=len(r["samples"].get(r["variables"][0], [])),
                verbose=False,
            )
            n_bad = len(q.get("failed_params", []))
        except Exception:
            pass

        rows.append(
            {
                "Rank": i + 1,
                "Circuit": r["circuit"],
                "Converged": "✓",
                "RMSE [Ω]": f"{rmse:.4e}",
                "# Params": len(r["variables"]),
                "Bad Params": n_bad,
            }
        )

    def sort_key(row):
        try:
            return float(row["RMSE [Ω]"])
        except Exception:
            return float("inf")

    rows.sort(key=sort_key)
    for i, r in enumerate(rows):
        r["Rank"] = i + 1

    table = dt.DataTable(
        data=rows,
        columns=[{"name": k, "id": k} for k in
                 ["Rank", "Circuit", "Converged", "RMSE [Ω]", "# Params", "Bad Params"]],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "fontFamily": "monospace", "padding": "4px 10px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"filter_query": '{Converged} = "✓"', "column_id": "Converged"}, "color": "#198754"},
            {"if": {"filter_query": '{Converged} = "✗"', "column_id": "Converged"}, "color": "#dc3545"},
            {"if": {"row_index": 0, "filter_query": '{Converged} = "✓"'}, "backgroundColor": "#d1e7dd"},
            {"if": {"filter_query": "{Bad Params} > 0", "column_id": "Bad Params"},
             "color": "#dc3545", "fontWeight": "bold"},
        ],
        tooltip_header={
            "Bad Params": "Number of parameters with FAIL posterior quality "
                          "(MULTIMODAL / BOUNDARY_HIT / UNINFORMATIVE / UNIFORM)"
        },
        tooltip_delay=0,
        tooltip_duration=None,
    )

    opts = [
        {"label": f"[{r['Rank']}] {r['Circuit']}", "value": r["Circuit"]}
        for r in rows
        if r.get("Converged") == "✓"
    ]
    default = opts[0]["value"] if opts else None
    return table, opts, default, None


@callback(
    Output("detail-diagram", "children"),
    Output("detail-params", "children"),
    Output("detail-nyquist", "figure"),
    Output("detail-bode", "figure"),
    Output("detail-residual", "figure"),
    Output("detail-posterior-pred", "figure"),
    Output("detail-posterior", "children"),
    Input("circuit-selector", "value"),
    State("store-results", "data"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def show_detail(circuit_str, results, data):
    _empty_figs = ({}, {}, {}, {})
    _empty_children = (html.Div(), html.Div(), html.Div())
    empty = _empty_children[:2] + _empty_figs + _empty_children[2:]
    if not circuit_str or not results or not data:
        return empty

    freq, Z = store_decode(data)

    result = next((r for r in results if r and r.get("circuit") == circuit_str), None)
    if result is None or not result.get("converged"):
        return (dbc.Alert("No converged result for this circuit.", color="warning"),
                html.Div(), {}, {}, {}, {}, html.Div())

    # ── Posterior quality assessment ─────────────────────────────────────
    quality = None
    try:
        quality = assess_posterior_quality(
            result["samples"],
            result["variables"],
            num_divergences=result.get("num_divergences", 0),
            num_samples=len(result["samples"].get(result["variables"][0], [])),
            verbose=False,
        )
    except Exception:
        pass

    # ── Circuit diagram ───────────────────────────────────────────────────
    diagram = html.Div()
    try:
        from autoeis.visualization import draw_circuit
        fig_mpl = draw_circuit(circuit_str)
        if fig_mpl is not None:
            src = mpl_fig_to_src(fig_mpl)
            diagram = html.Img(src=src, style={"maxWidth": "100%"})
        else:
            raise ValueError("draw_circuit returned None")
    except Exception:
        diagram = html.Div([
            html.Small("Circuit diagram (requires lcapy + pdflatex):", className="text-muted"),
            html.Br(),
            html.Code(circuit_str, className="text-primary fs-6"),
        ])

    # ── Quality badge (overall) ───────────────────────────────────────────
    if quality is not None:
        overall = quality.get("overall_pass", True)
        div_ratio = quality.get("div_ratio", 0)
        failed = quality.get("failed_params", [])
        badge_color = "success" if overall else "danger"
        badge_text = (
            f"✓ PASS — divergence {div_ratio:.1%}, all params acceptable"
            if overall
            else f"✗ FAIL — {', '.join(failed) if failed else 'divergence too high'}"
        )
        quality_badge = dbc.Alert(badge_text, color=badge_color, className="py-1 small mb-2")
    else:
        quality_badge = html.Div()

    # ── Fit ───────────────────────────────────────────────────────────────
    try:
        Z_fit, rmse, p_med = _compute_fit(result, freq, Z)
    except Exception as exc:
        return diagram, dbc.Alert(str(exc), color="danger"), {}, {}, {}, {}, html.Div()

    # ── Parameter summary with plausibility ──────────────────────────────
    param_table = html.Div([
        quality_badge,
        _param_summary_table(result, quality),
    ])

    # ── Plots (Plotly) ────────────────────────────────────────────────────
    nq_fig = nyquist_figure(freq, Z, Z_fit=Z_fit, title=f"Nyquist — {circuit_str}")
    bd_fig = bode_figure(freq, Z, Z_fit=Z_fit, freq_fit=freq, title=f"Bode — {circuit_str}")
    res_fig = residual_figure(freq, Z, Z_fit)

    try:
        pred_fig = _posterior_prediction_figure(result)
    except Exception as exc:
        pred_fig = go.Figure().add_annotation(text=f"Error: {exc}", showarrow=False)

    posterior = _posterior_plots(result, quality)

    return diagram, param_table, nq_fig, bd_fig, res_fig, pred_fig, posterior


# ---------------------------------------------------------------------------
# Export callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("download-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    State("store-results", "data"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def export_csv(n, results, data):
    if not results or not data:
        return dash.no_update
    freq, Z = store_decode(data)
    rows = []
    for r in results:
        if not r or not r.get("converged"):
            continue
        variables = r["variables"]
        samples = r["samples"]
        row = {"circuit": r["circuit"]}
        for v in variables:
            s = np.array(samples[v])
            row[f"{v}_median"] = np.median(s)
            row[f"{v}_mean"] = np.mean(s)
            row[f"{v}_std"] = np.std(s)
        try:
            _, rmse, _ = _compute_fit(r, freq, Z)
            row["rmse"] = rmse
        except Exception:
            row["rmse"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "autoeis_results.csv", index=False)


@callback(
    Output("download-json", "data"),
    Input("btn-export-json", "n_clicks"),
    State("store-results", "data"),
    prevent_initial_call=True,
)
def export_json(n, results):
    if not results:
        return dash.no_update
    content = json.dumps(results, indent=2)
    return dict(content=content, filename="autoeis_results.json")
