"""
Posterior distribution diagnostics for Bayesian inference results.

.. currentmodule:: autoeis.diagnostics

.. autosummary::
   :toctree: generated/

    posterior_gaussianity_score
    assess_posterior_sample
    assess_posterior_quality
    plot_posterior_quality

"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, probplot

log = logging.getLogger(__name__)

__all__ = [
    "PosteriorQuality",
    "posterior_gaussianity_score",
    "assess_posterior_sample",
    "assess_posterior_quality",
    "assess_inference_results",
    "add_posterior_quality",
    "plot_posterior_quality",
]

_AUXILIARY_SAMPLE_NAMES = {"lp__", "log_likelihood"}


@dataclass(frozen=True)
class PosteriorQuality:
    """Quality assessment for one posterior parameter distribution."""

    parameter: str
    score: float
    r2: float
    skewness: float
    modes: int
    shape: str
    status: str
    threshold: float
    peak_locations: tuple[float, ...] = ()
    peak_densities: tuple[float, ...] = ()

    @property
    def good(self) -> bool:
        """Whether the posterior distribution passes the score threshold."""
        return self.status == "Good"

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "parameter": self.parameter,
            "score": self.score,
            "r2": self.r2,
            "skewness": self.skewness,
            "modes": self.modes,
            "shape": self.shape,
            "status": self.status,
            "good": self.good,
            "threshold": self.threshold,
            "peak_locations": self.peak_locations,
            "peak_densities": self.peak_densities,
        }


def _as_sample_array(sample: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert a posterior sample to a clean flat finite array."""
    sample = np.asarray(sample, dtype=float).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size < 3:
        raise ValueError("Posterior sample must contain at least three finite values.")
    return sample


def _skewness(sample: np.ndarray) -> float:
    """Calculate skewness using central moments."""
    centered = sample - np.mean(sample)
    m2 = np.mean(centered**2)
    if m2 <= 0:
        return 0.0
    m3 = np.mean(centered**3)
    return float(m3 / (m2**1.5))


def _qq_r2(sample: np.ndarray) -> float:
    """Return R^2 of the normal Q-Q plot linear fit."""
    (theoretical, ordered), _ = probplot(sample, dist="norm")
    if np.allclose(ordered, ordered[0]):
        return 1.0

    slope, intercept = np.polyfit(theoretical, ordered, deg=1)
    fitted = slope * theoretical + intercept
    ss_res = np.sum((ordered - fitted) ** 2)
    ss_tot = np.sum((ordered - np.mean(ordered)) ** 2)
    if ss_tot <= 0:
        return 1.0
    return float(np.clip(1 - ss_res / ss_tot, 0, 1))


def _detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_height_fraction: float = 0.4,
    threshold_scale: float = 6e-5,
    min_distance_fraction: float = 0.4,
) -> list[int]:
    """Detect KDE peaks by consensus thresholding and non-maximum suppression."""
    if x.size < 3 or y.size < 3 or not np.any(y):
        return []

    thresholds = [
        threshold_scale * (float(np.max(y)) - float(np.min(y))),
        threshold_scale * float(np.std(y)),
        threshold_scale * float(np.subtract(*np.percentile(y, [75, 25]))),
    ]

    candidates_by_method = []
    for threshold in thresholds:
        peaks = []
        for idx in range(1, len(y) - 1):
            is_peak = y[idx] > y[idx - 1] and y[idx] > y[idx + 1]
            is_prominent = y[idx] - min(y[idx - 1], y[idx + 1]) > threshold
            if is_peak and is_prominent:
                peaks.append(idx)
        candidates_by_method.append(set(peaks))

    consensus = []
    for idx in sorted(set.union(*candidates_by_method) if candidates_by_method else set()):
        votes = sum(idx in candidates for candidates in candidates_by_method)
        if votes >= 2:
            consensus.append(idx)

    height_threshold = min_height_fraction * float(np.max(y))
    candidates = [idx for idx in consensus if y[idx] >= height_threshold]
    candidates.sort(key=lambda idx: y[idx], reverse=True)

    dx = float(np.median(np.diff(np.sort(x))))
    if dx <= 0:
        return sorted(candidates)

    bandwidth = 1.06 * float(np.std(x)) * len(x) ** (-1 / 5)
    min_distance = max(1, int(round((min_distance_fraction * bandwidth) / dx)))

    kept = []
    for idx in candidates:
        if all(abs(idx - other) >= min_distance for other in kept):
            kept.append(idx)

    return sorted(kept)


def _estimate_modes(
    sample: np.ndarray, n_grid: int
) -> tuple[int, np.ndarray, np.ndarray, list[int]]:
    """Estimate posterior modes from a KDE curve."""
    if np.allclose(sample, sample[0]):
        x = np.array([sample[0], sample[0]])
        y = np.array([1.0, 1.0])
        return 1, x, y, [0]

    x = np.linspace(float(np.min(sample)), float(np.max(sample)), n_grid)
    try:
        y = gaussian_kde(sample)(x)
    except np.linalg.LinAlgError:
        log.debug("KDE failed for posterior sample; treating distribution as invalid.")
        return 0, x, np.zeros_like(x), []

    peaks = _detect_peaks(x, y)
    return len(peaks), x, y, peaks


def posterior_gaussianity_score(r2: float, skewness: float, modes: int) -> float:
    """Compute the PART I Gaussianity score from Q-Q fit, skewness, and modes."""
    if not np.isfinite(r2) or not np.isfinite(skewness) or modes < 1:
        return 0.0
    score = r2 * 100
    score -= abs(skewness) * 5
    score -= modes * 9
    return float(max(0, round(score, 2)))


def _classify_distribution(skewness: float, modes: int, skew_threshold: float) -> str:
    """Classify a posterior shape from mode count and skewness."""
    if not np.isfinite(skewness) or modes < 1:
        return "Invalid"
    if modes > 1:
        return "Multi-Peak"
    if abs(skewness) >= 2:
        return "Touch-Bound"
    if skewness <= -skew_threshold:
        return "Left-Skewed"
    if skewness >= skew_threshold:
        return "Right-Skewed"
    return "Normal"


def assess_posterior_sample(
    sample: Sequence[float] | np.ndarray,
    parameter: str = "parameter",
    *,
    threshold: float = 79,
    skew_threshold: float = 0.5,
    n_grid: int = 300,
) -> PosteriorQuality:
    """Assess one posterior parameter distribution.

    Parameters
    ----------
    sample
        Posterior samples for one parameter.
    parameter
        Parameter name used in returned diagnostics and plots.
    threshold
        Minimum Gaussianity score required for ``status == "Good"``.
    skew_threshold
        Absolute skewness threshold used to label skewed distributions.
    n_grid
        Number of KDE grid points used for mode detection.

    Returns
    -------
    PosteriorQuality
        Diagnostic metrics and classification for the sample.
    """
    sample = _as_sample_array(sample)

    r2 = _qq_r2(sample)
    skew = _skewness(sample)
    modes, x, y, peaks = _estimate_modes(sample, n_grid=n_grid)
    score = posterior_gaussianity_score(r2, skew, modes)
    shape = _classify_distribution(skew, modes, skew_threshold)
    status = "Good" if score >= threshold else "Bad"

    return PosteriorQuality(
        parameter=parameter,
        score=score,
        r2=round(r2, 6),
        skewness=round(skew, 6),
        modes=modes,
        shape=shape,
        status=status,
        threshold=threshold,
        peak_locations=tuple(float(x[idx]) for idx in peaks),
        peak_densities=tuple(float(y[idx]) for idx in peaks),
    )


def _extract_samples(obj: Any) -> Mapping[str, np.ndarray]:
    """Extract sample mapping from MCMC, InferenceResult, or mapping-like input."""
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "samples"):
        return obj.samples
    if hasattr(obj, "get_samples"):
        return obj.get_samples()
    if hasattr(obj, "mcmc") and hasattr(obj.mcmc, "get_samples"):
        return obj.mcmc.get_samples()
    raise TypeError(
        "Expected an InferenceResult, a numpyro MCMC object, or a mapping of samples."
    )


def assess_posterior_quality(
    samples: Any,
    *,
    threshold: float = 79,
    var_names: Sequence[str] | None = None,
    skip_auxiliary: bool = True,
    save_plots: bool = False,
    save_dir: str | Path | None = None,
    n_grid: int = 300,
) -> pd.DataFrame:
    """Assess all posterior parameter distributions in an inference result.

    Parameters
    ----------
    samples
        An AutoEIS ``InferenceResult``, NumPyro ``MCMC`` object, or mapping from
        parameter names to posterior sample arrays.
    threshold
        Minimum Gaussianity score required for a parameter to pass.
    var_names
        Optional subset of parameter names to assess.
    skip_auxiliary
        Skip sampler bookkeeping variables such as ``lp__`` and ``log_likelihood``.
    save_plots
        If True, save one KDE/histogram diagnostic plot per parameter.
    save_dir
        Directory for saved plots. Defaults to ``posterior_quality`` when
        ``save_plots=True``.
    n_grid
        Number of KDE grid points used for mode detection.

    Returns
    -------
    pandas.DataFrame
        One row per parameter with score, shape, and pass/fail status.
    """
    sample_mapping = _extract_samples(samples)
    requested = set(var_names) if var_names is not None else None
    plot_dir = Path(save_dir or "posterior_quality") if save_plots else None

    results = []
    for name, sample in sample_mapping.items():
        if skip_auxiliary and name in _AUXILIARY_SAMPLE_NAMES:
            continue
        if requested is not None and name not in requested:
            continue

        result = assess_posterior_sample(
            sample,
            str(name),
            threshold=threshold,
            n_grid=n_grid,
        )
        results.append(result.to_dict())

        if plot_dir is not None:
            fig, _ = plot_posterior_quality(sample, result=result, save_dir=plot_dir)
            plt.close(fig)

    columns = [
        "parameter",
        "score",
        "r2",
        "skewness",
        "modes",
        "shape",
        "status",
        "good",
        "threshold",
        "peak_locations",
        "peak_densities",
    ]
    return pd.DataFrame(results, columns=columns)


def assess_inference_results(
    results: Sequence[Any],
    *,
    threshold: float = 79,
    save_plots: bool = False,
    save_dir: str | Path | None = None,
    n_grid: int = 300,
) -> pd.DataFrame:
    """Summarize posterior quality for a sequence of inference results."""
    rows = []
    for idx, result in enumerate(results):
        circuit = getattr(result, "circuit", None)
        converged = bool(getattr(result, "converged", True))
        if converged:
            result_save_dir = None
            if save_plots:
                result_save_dir = Path(save_dir or "posterior_quality") / f"inference_{idx}"
            quality = assess_posterior_quality(
                result,
                threshold=threshold,
                save_plots=save_plots,
                save_dir=result_save_dir,
                n_grid=n_grid,
            )
            bad = quality.loc[~quality["good"], "parameter"].to_list()
            posterior_score = float(quality["score"].sum())
        else:
            quality = pd.DataFrame()
            bad = ["Not Converged"]
            posterior_score = 0.0

        rows.append(
            {
                "circuit": circuit,
                "InferenceResult": result,
                "Posterior Score": posterior_score,
                "Bad Parameters": ", ".join(bad) if bad else "None",
                "n_bad_parameters": len(bad),
                "posterior_quality": quality,
            }
        )

    return pd.DataFrame(rows)


def add_posterior_quality(
    circuits: pd.DataFrame,
    results: Sequence[Any] | None = None,
    *,
    freq: np.ndarray | None = None,
    Z: np.ndarray | None = None,
    threshold: float = 79,
    save_plots: bool = False,
    save_dir: str | Path | None = None,
    compute_metrics: bool = False,
    sort: bool = True,
    n_grid: int = 300,
) -> pd.DataFrame:
    """Add posterior quality scores to a circuits dataframe.

    Parameters
    ----------
    circuits
        Circuits dataframe, usually the dataframe passed to Bayesian inference.
    results
        Optional sequence of AutoEIS ``InferenceResult`` objects. If omitted,
        ``circuits`` must already contain an ``InferenceResult`` column.
    freq, Z
        Frequency and impedance arrays. Required only when ``compute_metrics=True``.
    threshold
        Minimum per-parameter Gaussianity score required to pass.
    save_plots
        If True, save one KDE/histogram diagnostic plot per parameter.
    save_dir
        Directory for saved plots.
    compute_metrics
        If True, also call :func:`autoeis.core.compute_fitness_metrics`.
    sort
        Sort circuits without bad parameters first, then by posterior score.
    n_grid
        Number of KDE grid points used for mode detection.

    Returns
    -------
    pandas.DataFrame
        Copy of ``circuits`` with posterior quality columns added.
    """
    circuits = circuits.copy()
    if results is None:
        if "InferenceResult" not in circuits:
            raise ValueError("Provide results or a circuits dataframe with InferenceResult.")
        results = circuits["InferenceResult"].to_list()
    elif len(results) != len(circuits):
        raise ValueError("results must have the same length as circuits.")
    else:
        circuits["InferenceResult"] = list(results)

    quality = assess_inference_results(
        results,
        threshold=threshold,
        save_plots=save_plots,
        save_dir=save_dir,
        n_grid=n_grid,
    )
    for column in ["Posterior Score", "Bad Parameters", "n_bad_parameters"]:
        circuits[column] = quality[column].to_list()
    circuits["Posterior Quality"] = quality["posterior_quality"].to_list()

    if compute_metrics:
        if freq is None or Z is None:
            raise ValueError("freq and Z are required when compute_metrics=True.")
        from .core import compute_fitness_metrics

        circuits = compute_fitness_metrics(circuits, freq, Z)

    if sort:
        has_bad = circuits["Bad Parameters"].str.strip().str.lower() != "none"
        circuits = circuits.assign(_has_bad_parameters=has_bad)
        circuits.sort_values(
            by=["_has_bad_parameters", "Posterior Score"],
            ascending=[True, False],
            inplace=True,
            ignore_index=True,
        )
        circuits.drop(columns=["_has_bad_parameters"], inplace=True)

    circuits["Posterior Score Rank"] = np.arange(1, len(circuits) + 1)
    return circuits


def plot_posterior_quality(
    sample: Sequence[float] | np.ndarray,
    *,
    result: PosteriorQuality | None = None,
    parameter: str | None = None,
    threshold: float = 79,
    n_grid: int = 300,
    save_dir: str | Path | None = None,
    ax: Sequence[plt.Axes] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot posterior KDE, detected peaks, and histogram diagnostics."""
    sample = _as_sample_array(sample)
    if result is None:
        result = assess_posterior_sample(
            sample,
            parameter=parameter or "parameter",
            threshold=threshold,
            n_grid=n_grid,
        )

    modes, x, y, peaks = _estimate_modes(sample, n_grid=n_grid)
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    else:
        axes = np.asarray(ax)
        fig = axes[0].figure

    axes[0].plot(x, y, label="KDE Distribution")
    if peaks:
        axes[0].scatter(
            x[peaks],
            y[peaks],
            color="red",
            marker="x",
            s=90,
            label=f"Peaks: {modes}",
        )

    summary = (
        f"Score: {result.score:.2f}\n"
        f"R2: {result.r2:.4f}\n"
        f"Skewness: {result.skewness:.4f}"
    )
    axes[0].text(
        0.98,
        0.95,
        summary,
        transform=axes[0].transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.5"},
    )
    axes[0].set_title(f"{result.parameter}: {result.shape} | {result.status}")
    axes[0].set_xlabel("Parameter value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(sample, bins=30, alpha=0.75, color="tab:blue", edgecolor="black")
    axes[1].set_title("Posterior sample histogram")
    axes[1].set_xlabel("Parameter value")
    axes[1].set_ylabel("Count")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in result.parameter)
        fig.savefig(save_dir / f"{safe_name}-{result.score:.2f}.png", dpi=150)

    return fig, axes
