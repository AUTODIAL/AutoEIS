import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot, gaussian_kde
from sklearn.linear_model import LinearRegression


# ---------- Peak-finding helpers ----------
def suppress_close_peaks(idx_sorted, window):
    kept = []
    for i in idx_sorted:
        if all(abs(i - j) >= window for j in kept):
            kept.append(i)
    return sorted(kept)


def detect_peaks(x, y, alpha=6e-5, height_ratio=0.5, tol=1e-3, debug=False):
    """Return (indices, coords) of peaks that satisfy height + consensus."""
    def apply_method(method):
        if method == "range":
            thr = alpha * (y.max() - y.min())
        elif method == "std":
            thr = alpha * y.std()
        elif method == "iqr":
            q1, q3 = np.percentile(y, [25, 75])
            thr = alpha * (q3 - q1)
        else:
            raise ValueError
        peaks = [
            (i, x[i], y[i])
            for i in range(1, len(y) - 1)
            if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] - min(y[i - 1], y[i + 1]) > thr
        ]
        return peaks

    methods = ["range", "std", "iqr"]
    peak_sets = [set(apply_method(m)) for m in methods]

    # ── consensus voting (≥2 methods) ─────────────────────────
    consensus = {
        (idx, px, py)
        for idx, px, py in peak_sets[0]
        if sum(any(abs(px - px2) < tol for _, px2, _ in s) for s in peak_sets) >= 2
    }

    # ── height filter ─────────────────────────────────────────
    y_max = y.max()
    height_thr = height_ratio * y_max
    cand = [(idx, px, py) for idx, px, py in consensus if py >= height_thr]
    if debug:
        for _, px, py in cand:
            print(f"  peak@{px:.4g}  height={py:.4g}  thr={height_thr:.4g}")

    if not cand:
        return [], []

    # ── non-max suppression ───────────────────────────────────
    dx = np.median(np.diff(np.sort(x)))
    bw = 1.06 * y.std() * len(x) ** (-1 / 5)
    window_pts = round((0.4 * bw) / dx)  # Same formula originally used
    idx_sorted = [idx for idx, *_ in sorted(cand, key=lambda p: p[2], reverse=True)]
    idx_keep = suppress_close_peaks(idx_sorted, window_pts)

    peaks_coord = [(float(x[i]), float(y[i])) for i in idx_keep]
    return idx_keep, peaks_coord


def skewness(sample):
    m2 = np.mean((sample - sample.mean()) ** 2)
    m3 = np.mean((sample - sample.mean()) ** 3)
    return 0.0 if m2 == 0 else round(m3 / m2 ** 1.5, 4)


# ---------- main ----------
def evaluate_posterior_distribution(
    mcmc,
    threshold=79,
    save_plot=False,
    save_dir="figures",
    height_ratio=0.5,
    debug=False,
):
    def score_one(sample):
        (t, q), _ = probplot(sample, dist="norm")
        r2 = LinearRegression().fit(t[:, None], q[:, None]).score(t[:, None], q[:, None])
        sk = skewness(sample)

        x = np.linspace(sample.min(), sample.max(), 600)  # finer grid
        y = gaussian_kde(sample)(x)
        idx, coords = detect_peaks(x, y, height_ratio=height_ratio, debug=debug)
        modes = len(idx)
        score = r2 * 100 - abs(sk) * 5 - modes * 9
        return max(0, round(score, 2)), r2, sk, modes, x, y, coords

    def save_plot_func(label, sample, x, y, peaks, skew, score, r2, cat1, cat2):
        os.makedirs(os.path.join(save_dir, cat1, cat2), exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, y, label="KDE")
        if peaks:
            px, py = zip(*peaks)
            plt.scatter(px, py, color="red", s=80, marker="x")
        plt.title(f"{label} KDE | Score:{score} | R²:{r2:.3f} | Skew:{skew:.3f}")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.hist(sample, bins=30, edgecolor="black")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, cat1, cat2, f"{label}-{score}.png"))
        plt.close()

    total_score, bad_params = 0, []

    for name, chain in mcmc.get_samples().items():
        if name in ("lp__", "log_likelihood"):
            continue

        sample = np.asarray(chain)
        score, r2, sk, modes, x, y, peaks = score_one(sample)
        total_score += score
        if score < threshold:
            bad_params.append(name)

        if save_plot:
            if math.isnan(sk):
                cat1 = "Invalid"
            elif abs(sk) >= 2:
                cat1 = "Touch-Bound"
            elif modes > 1:
                cat1 = "Multi-Peak"
            elif sk < -0.5:
                cat1 = "Right-Skewed"
            elif sk > 0.5:
                cat1 = "Left-Skewed"
            else:
                cat1 = "Normal"
            cat2 = "Good" if score >= threshold else "Bad"
            save_plot_func(name, sample, x, y, peaks, sk, score, r2, cat1, cat2)
    

    return total_score, bad_params
