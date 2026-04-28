"""Page 2 – Choose Analysis Mode (Quick / Auto / Expert)."""

import dash
from dash import Input, Output, State, callback, ctx, dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/mode", name="Mode")

# ---------------------------------------------------------------------------
# Common ECM list for Quick mode
# ---------------------------------------------------------------------------

QUICK_CIRCUITS = [
    {
        "id": "R1-[R2,P1]",
        "name": "Randles (CPE)",
        "formula": "R₁ − (R₂ ‖ CPE₁)",
        "desc": "Classic Randles cell with CPE double layer. Solution resistance + parallel charge-transfer / CPE.",
    },
    {
        "id": "R1-[R2,P1]-[R3,P2]",
        "name": "Two Time Constants",
        "formula": "R₁ − (R₂ ‖ CPE₁) − (R₃ ‖ CPE₂)",
        "desc": "Two RC-like loops — e.g. surface film + charge transfer.",
    },
    {
        "id": "R1-[R2-P1,R3]",
        "name": "CPE Diffusion",
        "formula": "R₁ − ((R₂ − CPE₁) ‖ R₃)",
        "desc": "Two parallel paths: CPE-resistance branch vs. pure resistance. Models porous or diffusion-limited electrodes.",
    },
    {
        "id": "R1-[R2-P2,P1]",
        "name": "Randles-Warburg",
        "formula": "R₁ − ((R₂ − CPE₂) ‖ CPE₁)",
        "desc": "Randles with diffusion arm: (Rct + Warburg-CPE) in parallel with double-layer CPE.",
    },
    {
        "id": "R1-[R2,P1]-[R3,P2]-[R4,P3]",
        "name": "Three Time Constants",
        "formula": "R₁ − (R₂ ‖ CPE₁) − (R₃ ‖ CPE₂) − (R₄ ‖ CPE₃)",
        "desc": "Three relaxation processes — e.g. grain boundary, surface film, and charge transfer.",
    },
    {
        "id": "R1-P1",
        "name": "Series CPE",
        "formula": "R₁ − CPE₁",
        "desc": "Electrolyte resistance in series with a CPE. Minimal model for supercapacitors / EDLC.",
    },
]


_CARD_STYLE_DEFAULT = {
    "cursor": "pointer", "height": "100%",
    "border": "2px solid #dee2e6", "borderRadius": "0.375rem",
}
_CARD_STYLE_SELECTED = {
    "cursor": "pointer", "height": "100%",
    "border": "3px solid #198754", "borderRadius": "0.375rem",
    "backgroundColor": "#d1e7dd",
}


def _quick_card(circuit):
    return html.Div(
        dbc.Card(
            [
                dbc.CardHeader(html.Strong(circuit["name"])),
                dbc.CardBody(
                    [
                        html.Code(circuit["id"], className="text-primary d-block mb-1"),
                        html.Span(circuit["formula"], className="text-muted small d-block mb-1"),
                        html.P(circuit["desc"], className="small mb-0"),
                    ]
                ),
            ],
            className="h-100 border-0 bg-transparent",
        ),
        id={"type": "quick-card", "index": circuit["id"]},
        n_clicks=0,
        style=_CARD_STYLE_DEFAULT,
    )


def _auto_controls():
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    "AutoEIS will use a genetic/evolutionary algorithm to search for "
                    "candidate equivalent circuit models. Tune the search hyperparameters below.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Iterations"),
                                dbc.Input(id="auto-iters", type="number", value=100, min=1, max=500, step=1),
                                dbc.FormText("Number of independent runs"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Complexity"),
                                dbc.Input(id="auto-complexity", type="number", value=12, min=2, max=20, step=1),
                                dbc.FormText("Max circuit elements"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Population size"),
                                dbc.Input(id="auto-pop", type="number", value=100, min=5, max=500, step=1),
                                dbc.FormText("ECMs per generation"),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Generations"),
                                dbc.Input(id="auto-gens", type="number", value=30, min=5, max=200, step=1),
                                dbc.FormText("Evolutionary steps"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Tolerance"),
                                dbc.Select(
                                    id="auto-tol",
                                    options=[
                                        {"label": "5×10⁻² (very loose)", "value": "5e-2"},
                                        {"label": "1×10⁻² (default)", "value": "1e-2"},
                                        {"label": "5×10⁻³", "value": "5e-3"},
                                        {"label": "1×10⁻³ (strict)", "value": "1e-3"},
                                        {"label": "1×10⁻⁴ (very strict)", "value": "1e-4"},
                                    ],
                                    value="1e-2",
                                ),
                                dbc.FormText("Convergence threshold"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Circuit elements"),
                                dbc.Checklist(
                                    id="auto-terminals",
                                    options=[
                                        {"label": "R (resistor)", "value": "R"},
                                        {"label": "C (capacitor)", "value": "C"},
                                        {"label": "L (inductor)", "value": "L"},
                                        {"label": "P (CPE)", "value": "P"},
                                    ],
                                    value=["R", "L", "P"],
                                    inline=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Hr(),
                html.H6("Bayesian Inference Settings", className="fw-semibold"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Warmup samples"),
                                dbc.Input(id="bi-warmup", type="number", value=2500, min=100, step=100),
                                dbc.FormText("Default: 2500"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Posterior samples"),
                                dbc.Input(id="bi-samples", type="number", value=1000, min=100, step=100),
                                dbc.FormText("Default: 1000"),
                            ],
                            md=4,
                        ),
                    ]
                ),
            ]
        )
    )


def _expert_controls():
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    "Enter a circuit string using AutoEIS CDC notation. "
                    "Use R, C, L, P for elements; − for series; [ , ] for parallel.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Circuit string:", className="fw-semibold"),
                                dbc.Input(
                                    id="expert-circuit",
                                    placeholder="e.g.  R1-[R2,P1]",
                                    debounce=True,
                                ),
                                dbc.FormText("Examples: R1  |  R1-C1  |  R1-[R2,P1]  |  R1-[R2,P1]-[R3,P2]"),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fa fa-eye me-1"), "Preview Circuit"],
                                id="btn-preview",
                                color="info",
                                className="mt-4",
                            ),
                            md=4,
                        ),
                    ]
                ),
                # Circuit diagram preview
                dbc.Collapse(
                    [
                        html.Hr(),
                        html.H6("Circuit Diagram", className="fw-semibold"),
                        html.Div(id="circuit-diagram"),
                    ],
                    id="diagram-collapse",
                    is_open=False,
                ),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        html.H3(
            [html.I(className="fa fa-cogs me-2 text-primary"), "Step 2 — Choose Analysis Mode"],
            className="mb-3",
        ),

        # Mode selector tabs
        dbc.Tabs(
            [
                dbc.Tab(label="⚡ Quick Mode", tab_id="quick"),
                dbc.Tab(label="🤖 Auto Mode", tab_id="auto"),
                dbc.Tab(label="🔬 Expert Mode", tab_id="expert"),
            ],
            id="mode-tabs",
            active_tab="auto",
            className="mb-3",
        ),

        # Quick mode content
        dbc.Collapse(
            [
                html.P(
                    "Select one or more common equivalent circuit models to fit directly.",
                    className="text-muted",
                ),
                dbc.Row(
                    [dbc.Col(_quick_card(c), md=4, className="mb-3") for c in QUICK_CIRCUITS],
                ),
                html.Div(id="quick-selected-label", className="text-muted small mb-2"),
                dcc.Store(id="store-quick-selected", data=[]),
            ],
            id="collapse-quick",
            is_open=False,
        ),

        # Auto mode content
        dbc.Collapse(_auto_controls(), id="collapse-auto", is_open=True),

        # Expert mode content
        dbc.Collapse(_expert_controls(), id="collapse-expert", is_open=False),

        html.Hr(),

        # Navigation
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-arrow-left me-2"), "Back"],
                        href="/",
                        color="secondary",
                        outline=True,
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fa fa-play me-2"), "Next: Run Fitting"],
                        id="btn-next-mode",
                        href="/fitting",
                        color="primary",
                        className="float-end",
                    )
                ),
            ]
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("collapse-quick", "is_open"),
    Output("collapse-auto", "is_open"),
    Output("collapse-expert", "is_open"),
    Input("mode-tabs", "active_tab"),
)
def switch_mode(tab):
    return tab == "quick", tab == "auto", tab == "expert"


@callback(
    Output("store-quick-selected", "data"),
    Output("quick-selected-label", "children"),
    Output({"type": "quick-card", "index": dash.ALL}, "style"),
    Input({"type": "quick-card", "index": dash.ALL}, "n_clicks"),
    State("store-quick-selected", "data"),
    prevent_initial_call=True,
)
def toggle_quick_selection(n_clicks_list, selected):
    triggered = ctx.triggered_id
    # Guard: only process real clicks (triggered id must be a dict with an index)
    if not triggered or not isinstance(triggered, dict) or "index" not in triggered:
        return dash.no_update, dash.no_update, dash.no_update
    # Guard: ensure the triggering card actually has a click (not a re-render artifact)
    triggered_nc = next(
        (nc for c, nc in zip(QUICK_CIRCUITS, n_clicks_list or []) if c["id"] == triggered["index"]),
        None,
    )
    if not triggered_nc:
        return dash.no_update, dash.no_update, dash.no_update

    selected = list(selected or [])
    cid = triggered["index"]
    if cid in selected:
        selected.remove(cid)
    else:
        selected.append(cid)

    if selected:
        label = dbc.Alert(
            [html.I(className="fa fa-check-circle me-2"), f"Selected: {', '.join(selected)}"],
            color="success", className="py-2 mb-0",
        )
    else:
        label = html.Span("No circuits selected.", className="text-muted")

    styles = [
        _CARD_STYLE_SELECTED if c["id"] in selected else _CARD_STYLE_DEFAULT
        for c in QUICK_CIRCUITS
    ]
    return selected, label, styles


@callback(
    Output("circuit-diagram", "children"),
    Output("diagram-collapse", "is_open"),
    Input("btn-preview", "n_clicks"),
    State("expert-circuit", "value"),
    prevent_initial_call=True,
)
def preview_circuit(n, circuit_str):
    if not circuit_str or not circuit_str.strip():
        return dbc.Alert("Please enter a circuit string first.", color="warning"), False
    try:
        from autoeis.visualization import draw_circuit
        from utils import mpl_fig_to_src
        fig = draw_circuit(circuit_str.strip())
        if fig is None:
            raise ValueError("draw_circuit returned None (lcapy may not be installed).")
        src = mpl_fig_to_src(fig)
        return html.Img(src=src, style={"maxWidth": "100%"}), True
    except Exception as exc:
        return dbc.Alert(f"Could not render: {exc}", color="warning"), True


# Auto-save mode config to shared store whenever any setting changes
@callback(
    Output("store-mode", "data"),
    Input("mode-tabs", "active_tab"),
    Input("store-quick-selected", "data"),
    Input("expert-circuit", "value"),
    Input("auto-iters", "value"),
    Input("auto-complexity", "value"),
    Input("auto-pop", "value"),
    Input("auto-gens", "value"),
    Input("auto-tol", "value"),
    Input("auto-terminals", "value"),
    Input("bi-warmup", "value"),
    Input("bi-samples", "value"),
)
def save_mode_config(
    active_tab,
    quick_selected,
    expert_circuit,
    iters, complexity, pop, gens, tol, terminals,
    bi_warmup, bi_samples,
):
    params = {
        "iters": int(iters or 100),
        "complexity": int(complexity or 12),
        "population_size": int(pop or 100),
        "generations": int(gens or 30),
        "tol": float(tol or 1e-2),
        "terminals": "".join(terminals or ["R", "L", "P"]),
        "num_warmup": int(bi_warmup or 2500),
        "num_samples": int(bi_samples or 1000),
    }

    if active_tab == "quick":
        return {"mode": "quick", "circuits": quick_selected or [], "params": params}
    elif active_tab == "expert":
        return {"mode": "expert", "circuit_str": (expert_circuit or "").strip(), "params": params}
    return {"mode": "auto", "params": params}
