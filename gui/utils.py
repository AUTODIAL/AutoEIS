"""Helper utilities for AutoEIS GUI: file parsing and Plotly figure builders."""

import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_uploaded_file(contents: str, filename: str) -> tuple[pd.DataFrame | None, str | None]:
    """Decode a Dash Upload component payload into a DataFrame."""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.lower().endswith(".txt"):
            # Try common separators in order
            for sep in ["\t", ",", ";", r"\s+"]:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=sep, engine="python")
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue
            else:
                return None, "Could not parse .txt file — try saving as CSV."
        else:
            return None, f"Unsupported format: {filename}. Use CSV, TXT, or XLSX."
        return df, None
    except Exception as exc:
        return None, str(exc)


def guess_columns(columns: list[str]) -> dict[str, str | None]:
    """Heuristically guess which column maps to freq / Zreal / Zimag."""
    mapping = {"freq": None, "zreal": None, "zimag": None}
    for col in columns:
        c = col.lower().replace(" ", "").replace("(", "").replace(")", "").replace("/", "")
        if mapping["freq"] is None and any(k in c for k in ["freq", "hz", "f"]):
            mapping["freq"] = col
        elif mapping["zreal"] is None and any(k in c for k in ["re", "real", "zre", "z'"]):
            mapping["zreal"] = col
        elif mapping["zimag"] is None and any(k in c for k in ["im", "imag", "zim", 'z"', "zi"]):
            mapping["zimag"] = col
    return mapping


# ---------------------------------------------------------------------------
# Store serialisation helpers
# ---------------------------------------------------------------------------

def store_encode(freq: np.ndarray, Z: np.ndarray) -> dict:
    return {
        "freq": freq.tolist(),
        "z_real": Z.real.tolist(),
        "z_imag": Z.imag.tolist(),
    }


def store_decode(d: dict) -> tuple[np.ndarray, np.ndarray]:
    # Force float64: JSON null → NaN (not None, which would create object dtype)
    freq = np.array(d["freq"], dtype=float)
    Z = np.array(d["z_real"], dtype=float) + 1j * np.array(d["z_imag"], dtype=float)
    return freq, Z


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def _nyquist_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        xaxis_title="Re(Z) [Ω]",
        yaxis_title="−Im(Z) [Ω]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode="closest",
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=460,
    )
    return fig


def nyquist_figure(
    freq: np.ndarray,
    Z: np.ndarray,
    deleted: list[int] | None = None,
    pending: list[int] | None = None,
    Z_fit: np.ndarray | None = None,
    title: str = "Nyquist Plot",
) -> go.Figure:
    """Interactive Nyquist scatter; pending removal shown orange, deleted shown faded ×."""
    deleted_set = set(deleted or [])
    pending_set = set(pending or [])
    n = len(Z)
    active = [i for i in range(n) if i not in deleted_set and i not in pending_set]
    pending_list = [i for i in range(n) if i in pending_set and i not in deleted_set]
    removed = [i for i in range(n) if i in deleted_set]

    fig = go.Figure()

    if active:
        fig.add_trace(
            go.Scatter(
                x=Z.real[active],
                y=-Z.imag[active],
                mode="markers",
                name="Data",
                marker=dict(color="#1f77b4", size=8, symbol="circle"),
                customdata=active,
                text=[
                    f"idx {i}<br>f={freq[i]:.3e} Hz<br>"
                    f"Re={Z.real[i]:.4g} Ω<br>Im={Z.imag[i]:.4g} Ω"
                    for i in active
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if pending_list:
        fig.add_trace(
            go.Scatter(
                x=Z.real[pending_list],
                y=-Z.imag[pending_list],
                mode="markers",
                name="Pending",
                marker=dict(color="#ff7f0e", size=10, symbol="circle",
                            line=dict(color="white", width=1.5)),
                customdata=pending_list,
                text=[f"idx {i} [pending removal — click Remove Selected]" for i in pending_list],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if removed:
        fig.add_trace(
            go.Scatter(
                x=Z.real[removed],
                y=-Z.imag[removed],
                mode="markers",
                name="Removed",
                marker=dict(color="lightgray", size=8, symbol="x", opacity=0.5),
                customdata=removed,
                text=[f"idx {i} [removed]" for i in removed],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if Z_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=Z_fit.real,
                y=-Z_fit.imag,
                mode="lines",
                name="Fit",
                line=dict(color="#ff7f0e", width=2),
            )
        )

    fig.update_layout(title=dict(text=title, x=0.5))
    return _nyquist_layout(fig)


def nyquist_compare_figure(
    freq_raw: np.ndarray,
    Z_raw: np.ndarray,
    freq_clean: np.ndarray,
    Z_clean: np.ndarray,
) -> go.Figure:
    """Overlay before/after on the same Nyquist axes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Z_raw.real, y=-Z_raw.imag,
            mode="markers", name="Before",
            marker=dict(color="lightblue", size=7, opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Z_clean.real, y=-Z_clean.imag,
            mode="markers", name="After",
            marker=dict(color="#2ca02c", size=7),
        )
    )
    fig.update_layout(title=dict(text="Before vs. After Preprocessing", x=0.5))
    return _nyquist_layout(fig)


def bode_figure(
    freq: np.ndarray,
    Z: np.ndarray,
    deleted: list[int] | None = None,
    pending: list[int] | None = None,
    Z_fit: np.ndarray | None = None,
    freq_fit: np.ndarray | None = None,
    title: str = "Bode Plot",
) -> go.Figure:
    """Two-row Bode plot: |Z| magnitude (top) and phase (bottom)."""
    deleted_set = set(deleted or [])
    pending_set = set(pending or [])
    n = len(Z)
    active = [i for i in range(n) if i not in deleted_set and i not in pending_set]
    pending_list = [i for i in range(n) if i in pending_set and i not in deleted_set]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("|Z| [Ω]", "−Phase [°]"),
    )

    if active:
        freq_a = freq[active]
        Z_a = Z[active]
        fig.add_trace(
            go.Scatter(
                x=freq_a, y=np.abs(Z_a),
                mode="markers", name="|Z|",
                marker=dict(color="#1f77b4", size=6),
                customdata=active,
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=freq_a, y=-np.angle(Z_a, deg=True),
                mode="markers", name="Phase",
                marker=dict(color="#d62728", size=6),
                customdata=active,
                showlegend=True,
            ),
            row=2, col=1,
        )

    if pending_list:
        freq_p = freq[pending_list]
        Z_p = Z[pending_list]
        fig.add_trace(
            go.Scatter(
                x=freq_p, y=np.abs(Z_p),
                mode="markers", name="Pending",
                marker=dict(color="#ff7f0e", size=8, line=dict(color="white", width=1)),
                customdata=pending_list,
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=freq_p, y=-np.angle(Z_p, deg=True),
                mode="markers", name="Pending (phase)",
                marker=dict(color="#ff7f0e", size=8, line=dict(color="white", width=1)),
                customdata=pending_list,
                showlegend=False,
            ),
            row=2, col=1,
        )

    if Z_fit is not None and freq_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=freq_fit, y=np.abs(Z_fit),
                mode="lines", name="|Z| fit",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=freq_fit, y=-np.angle(Z_fit, deg=True),
                mode="lines", name="Phase fit",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=2, col=1,
        )

    fig.update_xaxes(type="log", title_text="Frequency [Hz]", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(title=dict(text=title, x=0.5), height=480, margin=dict(l=50, r=20, t=60, b=50))
    return fig


def residual_figure(freq: np.ndarray, Z_meas: np.ndarray, Z_fit: np.ndarray) -> go.Figure:
    """Plot relative residuals (Re and Im) vs frequency."""
    res_real = (Z_meas.real - Z_fit.real) / np.abs(Z_meas)
    res_imag = (Z_meas.imag - Z_fit.imag) / np.abs(Z_meas)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=res_real * 100, mode="markers+lines", name="Re residual [%]"))
    fig.add_trace(go.Scatter(x=freq, y=res_imag * 100, mode="markers+lines", name="Im residual [%]"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_xaxes(type="log", title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="Relative residual [%]")
    fig.update_layout(
        title=dict(text="Fit Residuals", x=0.5),
        hovermode="x unified",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def mpl_fig_to_src(mpl_fig) -> str:
    """Convert a matplotlib Figure to a base64 PNG src string for <img>."""
    import io as _io
    buf = _io.BytesIO()
    mpl_fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    import matplotlib.pyplot as plt
    plt.close(mpl_fig)
    return f"data:image/png;base64,{encoded}"
