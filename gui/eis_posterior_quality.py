"""
Posterior distribution quality assessment for AutoEIS ECM selection.

Implements a three-layer check to classify each circuit parameter's posterior as:
    NORMAL        — approximately Gaussian (ideal)
    LOGNORMAL     — log-space Gaussian; physically natural for positive params (pass with flag)
    UNCERTAIN     — borderline shape; not clearly failing (pass with warning)
    MULTIMODAL    — two or more distinct peaks (fail)
    BOUNDARY_HIT  — mode or mass pile-up near sample boundary (fail)
    UNINFORMATIVE — posterior too wide to constrain the parameter (fail)
    UNIFORM       — flat distribution; prior not updated (fail)

Usage
-----
    from eis_posterior_quality import assess_posterior_quality

    samples   = BI_results[0].mcmc.get_samples()
    variables = ae.parser.get_parameter_labels(test_circuit)

    quality = assess_posterior_quality(
        samples,
        variables,
        num_divergences=BI_results[0].num_divergences,
        num_samples=num_samples,
        verbose=True,
    )

    if not quality['overall_pass']:
        print("Posterior invalid:", quality['summary'])
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde as sp_kde


# ---------------------------------------------------------------------------
# Tunable defaults (override via keyword arguments)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    div_threshold        = 0.20,   # max acceptable divergence ratio
    rel_width_max        = 10.0,   # (q97.5-q2.5)/|median| → uninformative above this
    skew_lognormal       = 1.5,    # |skewness| threshold to trigger log-normal check
    bc_threshold         = 0.60,   # bimodality coefficient (Pearson); 0.555 is the theoretical cut-off
    peak_prominence_frac = 0.25,   # secondary KDE peak must exceed this fraction of main peak
    # --- Uniform detection ---
    flatness_threshold   = 0.40,   # KDE min/max ratio above this → uniform-like
    # --- Boundary / wall-hit detection ---
    # Uses KDE edge density (fraction of peak density at the sample extremes).
    # High edge density means the posterior is still rising at the boundary (truncated).
    # For a long-tailed but valid distribution, KDE decays to near-zero at the sample
    # extremes, so edge density is low even if the mode sits close to the boundary.
    edge_density_threshold = 0.25,  # KDE at sample edge / peak > this → wall hit
    edge_n_pts             = 20,    # number of KDE grid points averaged at each edge (out of 512)
)


def _bimodality_coefficient(skewness: float, kurt: float, n: int) -> float:
    """Pearson bimodality coefficient with finite-sample correction."""
    if n > 3:
        g1 = skewness * np.sqrt(n * (n - 1)) / (n - 2)
        # Corrected excess kurtosis (Fisher)
        g2 = kurt * (n - 1) * (n + 1) / ((n - 2) * (n - 3))
    else:
        g1, g2 = skewness, kurt
    denom = g2 + 3
    return (g1 ** 2 + 1) / denom if denom != 0 else 1.0


def _kde_analysis(s: np.ndarray, peak_prominence_frac: float, edge_n_pts: int = 20):
    """
    Compute KDE on samples and return a dict of KDE-derived statistics.

    Returned keys
    -------------
    kde_vals          : np.ndarray or None
    x_grid            : np.ndarray or None
    n_peaks           : int
    kde_flatness      : float  — min(KDE) / max(KDE) over the grid.
                                 Near 1 → uniform-like; near 0 → peaked.
    left_edge_density : float  — mean KDE in first ``edge_n_pts`` grid points / max(KDE).
                                 High → posterior is still rising at the left boundary (wall hit).
                                 Low  → posterior naturally decays at the left extreme.
    right_edge_density: float  — same for the right boundary.
    """
    s_min, s_max = s.min(), s.max()
    s_range = s_max - s_min

    _null = dict(
        kde_vals=None, x_grid=None, n_peaks=1,
        kde_flatness=0.0, left_edge_density=0.0, right_edge_density=0.0,
    )

    if s_range < 1e-14:
        return _null  # degenerate: all samples identical

    try:
        kde_obj  = sp_kde(s, bw_method="silverman")
        x_grid   = np.linspace(s_min, s_max, 512)
        kde_vals = kde_obj(x_grid)
    except Exception:
        return _null

    peak_max = kde_vals.max()
    if peak_max < 1e-30:
        return _null

    # Peak count
    prominence_min = peak_prominence_frac * peak_max
    peak_idxs, _   = find_peaks(kde_vals, prominence=prominence_min)
    n_peaks        = max(1, len(peak_idxs))

    # Flatness: ratio of minimum to maximum KDE density.
    # A true uniform distribution will have min/max close to 1 across the grid.
    # A peaked (normal/lognormal) distribution will have min/max close to 0.
    kde_flatness = float(kde_vals.min() / peak_max)

    # Edge density: average KDE in the outermost ``edge_n_pts`` grid points,
    # normalised by the peak density.
    # For a distribution that is TRUNCATED at the boundary, density is still
    # substantial at the sample extremes → edge density is high.
    # For a distribution with genuine long tails, the KDE decays to near-zero
    # at the sample extremes even if the mode is close to the boundary.
    n_pts            = min(edge_n_pts, len(kde_vals) // 4)
    left_edge_density  = float(kde_vals[:n_pts].mean()  / peak_max)
    right_edge_density = float(kde_vals[-n_pts:].mean() / peak_max)

    return dict(
        kde_vals=kde_vals, x_grid=x_grid, n_peaks=n_peaks,
        kde_flatness=kde_flatness,
        left_edge_density=left_edge_density,
        right_edge_density=right_edge_density,
    )


def _classify_single(s: np.ndarray, cfg: dict) -> dict:
    """
    Assess one parameter's posterior sample array.

    Returns a dict with keys: status, passes, reason, and various diagnostics.
    """
    n      = len(s)
    s_min  = float(s.min())
    s_max  = float(s.max())
    s_range= s_max - s_min

    # ── Descriptive statistics ────────────────────────────────────────────
    median   = float(np.median(s))
    q025     = float(np.percentile(s, 2.5))
    q975     = float(np.percentile(s, 97.5))
    skewness = float(stats.skew(s))
    kurt     = float(stats.kurtosis(s))   # excess kurtosis; Normal = 0
    bc       = _bimodality_coefficient(skewness, kurt, n)

    rel_width = (q975 - q025) / abs(median) if abs(median) > 1e-14 else np.inf

    # ── KDE analysis ─────────────────────────────────────────────────────
    kde_info = _kde_analysis(
        s,
        peak_prominence_frac=cfg["peak_prominence_frac"],
        edge_n_pts=cfg["edge_n_pts"],
    )
    kde_vals           = kde_info["kde_vals"]
    n_peaks            = kde_info["n_peaks"]
    kde_flatness       = kde_info["kde_flatness"]
    left_edge_density  = kde_info["left_edge_density"]
    right_edge_density = kde_info["right_edge_density"]

    # ── Layer 1: Boundary / wall-hit detection ────────────────────────────
    # We use KDE edge density instead of mode_pos to avoid false positives on
    # long-tailed (e.g. lognormal) distributions whose natural mode sits close
    # to the sample minimum but whose density DOES decay at the sample extreme.
    #
    # True wall-hit:  posterior is truncated by the prior boundary → the KDE
    #                 is still high (and rising) at the sample edge.
    # Long-tail valid: posterior naturally decays to near-zero at the sample
    #                  extreme even when the mode is close to the boundary.
    #
    # Degenerate case (zero range) is unconditionally flagged.
    if s_range < 1e-14:
        wall_hit = True
    else:
        edt = cfg["edge_density_threshold"]
        wall_hit = left_edge_density > edt or right_edge_density > edt

    # ── Layer 2: Multimodality ────────────────────────────────────────────
    is_multimodal = n_peaks > 1

    # ── Layer 3: Informativeness / shape ─────────────────────────────────
    is_too_wide = rel_width > cfg["rel_width_max"]

    # Uniform detection: KDE flatness ratio (min/max) is close to 1, meaning
    # density is roughly constant across the support.
    # This is more direct than kurtosis-based checks and robust to small samples
    # because it uses the shape of the estimated density rather than raw moments.
    # Fall back to a kurtosis + width check when KDE is unavailable.
    if kde_vals is not None:
        is_flat = kde_flatness > cfg["flatness_threshold"]
    else:
        is_flat = kurt < -0.8 and rel_width > 3.0

    # Log-normal check: only meaningful when all samples are strictly positive
    is_lognormal = False
    log_skew_val = None
    if s_min > 1e-14 and abs(skewness) > cfg["skew_lognormal"]:
        log_s        = np.log(s)
        log_skew_val = float(abs(stats.skew(log_s)))
        # Log-transform must reduce |skewness| by at least 50 %
        if log_skew_val < 0.5 * abs(skewness):
            is_lognormal = True

    # ── Classification (ordered by severity) ────────────────────────────
    if is_multimodal:
        status = "MULTIMODAL"
        passes = False
        reason = f"KDE shows {n_peaks} significant peaks"

    elif wall_hit:
        status = "BOUNDARY_HIT"
        passes = False
        reason = (
            f"left_edge_density={left_edge_density:.2f}, "
            f"right_edge_density={right_edge_density:.2f} "
            f"(threshold={cfg['edge_density_threshold']})"
        )

    elif is_too_wide:
        status = "UNINFORMATIVE"
        passes = False
        reason = f"rel_width={rel_width:.1f} > threshold {cfg['rel_width_max']}"

    elif is_flat:
        status = "UNIFORM"
        passes = False
        reason = (
            f"kde_flatness={kde_flatness:.2f} > threshold {cfg['flatness_threshold']}"
            if kde_vals is not None
            else f"kurt={kurt:.2f}, rel_width={rel_width:.1f}"
        )

    elif is_lognormal:
        # Unimodal + constrained in log-space; physically common for EIS params
        status = "LOGNORMAL"
        passes = True
        reason = (
            f"skew reduced from {skewness:.2f} to {log_skew_val:.2f} in log-space"
        )

    elif abs(skewness) < cfg["skew_lognormal"] and abs(kurt) < 3.0:
        status = "NORMAL"
        passes = True
        reason = ""

    else:
        # Borderline — not clearly failing under lenient small-sample thresholds
        status = "UNCERTAIN"
        passes = True
        reason = f"skew={skewness:.2f}, kurt={kurt:.2f}, bc={bc:.2f}"

    return {
        "status"    : status,
        "passes"    : passes,
        "reason"    : reason,
        "median"    : median,
        "q2.5"      : q025,
        "q97.5"     : q975,
        "rel_width" : rel_width,
        "skewness"  : skewness,
        "kurtosis"  : kurt,
        "bc"        : bc,
        "n_peaks"            : n_peaks,
        "wall_hit"           : wall_hit,
        "lognormal"          : is_lognormal,
        "kde_flatness"       : kde_flatness,
        "left_edge_density"  : left_edge_density,
        "right_edge_density" : right_edge_density,
    }


def assess_posterior_quality(
    samples,
    variables,
    num_divergences: int = 0,
    num_samples: int = None,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Assess posterior distribution quality for all circuit parameters.

    Parameters
    ----------
    samples : dict
        Raw MCMC sample dict from ``BI_results[0].mcmc.get_samples()``.
        Keys are parameter names; values are 1-D arrays of MCMC draws.
    variables : list[str]
        Circuit parameter names from ``ae.parser.get_parameter_labels(circuit)``.
        Only these keys are evaluated (noise params such as ``sigma.*`` are ignored).
    num_divergences : int
        Number of MCMC divergences, e.g. ``BI_results[0].num_divergences``.
    num_samples : int, optional
        Total MCMC sample count.  Auto-detected from ``samples`` when omitted.
    verbose : bool
        Print a per-parameter diagnostic table (default True).
    **kwargs
        Override any default threshold.  Valid keys:

        div_threshold          (default 0.20)
        rel_width_max          (default 10.0)
        skew_lognormal         (default 1.5)
        bc_threshold           (default 0.60)
        peak_prominence_frac   (default 0.25)
        flatness_threshold     (default 0.40)  — KDE min/max ratio; above → uniform
        edge_density_threshold (default 0.25)  — KDE edge/peak ratio; above → wall hit
        edge_n_pts             (default 20)    — KDE grid points averaged at each edge

    Returns
    -------
    dict with keys:
        overall_pass   : bool   — True only when divergence check AND all params pass
        divergence_ok  : bool
        div_ratio      : float
        per_parameter  : dict   — keyed by parameter name → diagnostic sub-dict
        failed_params  : list[str]
        failure_reasons: dict   — keyed by failed parameter name → reason string
        summary        : str    — human-readable one-liner
    """
    cfg = {**_DEFAULTS, **kwargs}

    # ── Divergence check ─────────────────────────────────────────────────
    if num_samples is None:
        first = next((v for v in variables if v in samples), None)
        num_samples = len(samples[first]) if first else 1

    div_ratio    = num_divergences / max(num_samples, 1)
    divergence_ok = div_ratio <= cfg["div_threshold"]

    # ── Per-parameter assessment ─────────────────────────────────────────
    per_param      = {}
    failed_params  = []
    failure_reasons= {}

    for var in variables:
        if var not in samples:
            continue

        s      = np.asarray(samples[var], dtype=float)
        result = _classify_single(s, cfg)
        per_param[var] = result

        if not result["passes"]:
            failed_params.append(var)
            failure_reasons[var] = f"[{result['status']}] {result['reason']}"

    # ── Verbose table ────────────────────────────────────────────────────
    if verbose:
        _PASS = "\033[32m✓\033[0m"
        _FAIL = "\033[31m✗\033[0m"
        print("\n  Posterior quality assessment")
        print(f"  {'Param':<12}  {'Status':<15}  {'median':>10}  "
              f"{'rel_w':>6}  {'skew':>6}  {'kurt':>6}  {'peaks':>5}  {'wall':>5}")
        print("  " + "-" * 76)
        for var, r in per_param.items():
            icon = _PASS if r["passes"] else _FAIL
            print(
                f"  {icon} {var:<12}  {r['status']:<15}  "
                f"{r['median']:>10.3e}  {r['rel_width']:>6.1f}  "
                f"{r['skewness']:>+6.2f}  {r['kurtosis']:>+6.2f}  "
                f"{r['n_peaks']:>5d}  {str(r['wall_hit']):>5}"
            )
            if r["reason"]:
                print(f"              └─ {r['reason']}")

        div_label = "ok" if divergence_ok else f"FAIL (>{cfg['div_threshold']:.0%})"
        print(f"\n  Divergence ratio : {div_ratio:.1%}  [{div_label}]")

    # ── Overall verdict ──────────────────────────────────────────────────
    overall_pass = divergence_ok and (len(failed_params) == 0)

    if overall_pass:
        summary = (
            f"PASS — divergence {div_ratio:.1%}, "
            f"all {len(per_param)} posteriors acceptable."
        )
    else:
        parts = []
        if not divergence_ok:
            parts.append(f"divergence {div_ratio:.1%} > {cfg['div_threshold']:.0%}")
        if failed_params:
            parts.append(f"failed params: {failed_params}")
        summary = "FAIL — " + "; ".join(parts)

    if verbose:
        print(f"\n  {summary}\n")

    return {
        "overall_pass"   : overall_pass,
        "divergence_ok"  : divergence_ok,
        "div_ratio"      : div_ratio,
        "per_parameter"  : per_param,
        "failed_params"  : failed_params,
        "failure_reasons": failure_reasons,
        "summary"        : summary,
    }
