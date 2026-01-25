"""
Correlation Analysis
====================

Measures linear and non-linear association between signals.

Methods:
    - Pearson: Linear correlation
    - Spearman: Rank correlation (monotonic relationships)
    - Kendall: Concordance (robust to outliers)

Rolling correlation enables detection of relationship breakdown.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """Output from pairwise correlation analysis"""

    pearson: float
    spearman: float
    kendall: float
    pearson_pvalue: float
    spearman_pvalue: float

    rolling_pearson: Optional[np.ndarray] = None
    rolling_window: int = 0

    correlation_mean: float = 0.0
    correlation_std: float = 0.0
    correlation_min: float = 0.0
    correlation_max: float = 0.0

    strength: str = "none"          # none | weak | moderate | strong
    stability: str = "stable"       # stable | unstable | breakdown


@dataclass
class CorrelationMatrixResult:
    """Output from multi-signal correlation analysis"""

    pearson_matrix: np.ndarray
    spearman_matrix: np.ndarray

    mean_correlation: float
    median_correlation: float
    correlation_dispersion: float

    n_strong_pairs: int
    n_weak_pairs: int

    eigenvalues: np.ndarray
    variance_explained_1: float
    effective_dimension: float


def compute(
    x: np.ndarray,
    y: np.ndarray,
    rolling_window: Optional[int] = None
) -> CorrelationResult:
    """Compute correlation between two signals."""
    from scipy import stats

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 3:
        return CorrelationResult(
            pearson=0.0, spearman=0.0, kendall=0.0,
            pearson_pvalue=1.0, spearman_pvalue=1.0
        )

    valid = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[valid], y[valid]

    if len(x_c) < 3:
        return CorrelationResult(
            pearson=0.0, spearman=0.0, kendall=0.0,
            pearson_pvalue=1.0, spearman_pvalue=1.0
        )

    pearson_r, pearson_p = stats.pearsonr(x_c, y_c)
    spearman_r, spearman_p = stats.spearmanr(x_c, y_c)
    kendall_r, _ = stats.kendalltau(x_c, y_c)

    rolling_pearson = None
    corr_mean, corr_std = pearson_r, 0.0
    corr_min, corr_max = pearson_r, pearson_r

    if rolling_window and rolling_window < n:
        rolling_pearson = np.full(n, np.nan)
        for i in range(rolling_window, n):
            xw, yw = x[i-rolling_window:i], y[i-rolling_window:i]
            vw = ~(np.isnan(xw) | np.isnan(yw))
            if np.sum(vw) >= 3:
                r, _ = stats.pearsonr(xw[vw], yw[vw])
                rolling_pearson[i] = r

        vr = ~np.isnan(rolling_pearson)
        if np.any(vr):
            corr_mean = float(np.nanmean(rolling_pearson))
            corr_std = float(np.nanstd(rolling_pearson))
            corr_min = float(np.nanmin(rolling_pearson))
            corr_max = float(np.nanmax(rolling_pearson))

    abs_r = abs(pearson_r)
    strength = "none" if abs_r < 0.2 else "weak" if abs_r < 0.5 else "moderate" if abs_r < 0.8 else "strong"

    if rolling_window and corr_std > 0:
        stability = "breakdown" if corr_min * corr_max < 0 else "unstable" if corr_std > 0.3 else "stable"
    else:
        stability = "stable"

    return CorrelationResult(
        pearson=float(pearson_r), spearman=float(spearman_r), kendall=float(kendall_r),
        pearson_pvalue=float(pearson_p), spearman_pvalue=float(spearman_p),
        rolling_pearson=rolling_pearson, rolling_window=rolling_window or 0,
        correlation_mean=corr_mean, correlation_std=corr_std,
        correlation_min=corr_min, correlation_max=corr_max,
        strength=strength, stability=stability
    )


def compute_matrix(signals: np.ndarray) -> CorrelationMatrixResult:
    """Compute correlation matrix for multiple signals (n_signals, n_obs)."""
    from scipy import stats

    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_signals = signals.shape[0]

    pearson_matrix = np.corrcoef(signals)

    spearman_matrix = np.eye(n_signals)
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            r, _ = stats.spearmanr(signals[i], signals[j])
            spearman_matrix[i, j] = spearman_matrix[j, i] = r

    mask = ~np.eye(n_signals, dtype=bool)
    off_diag = pearson_matrix[mask]

    mean_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    median_corr = float(np.median(off_diag)) if len(off_diag) > 0 else 0.0
    dispersion = float(np.std(off_diag)) if len(off_diag) > 0 else 0.0

    n_strong = int(np.sum(np.abs(off_diag) > 0.7)) // 2
    n_weak = int(np.sum(np.abs(off_diag) < 0.3)) // 2

    eigenvalues = np.sort(np.linalg.eigvalsh(pearson_matrix))[::-1]
    total_var = np.sum(eigenvalues)
    var_exp_1 = eigenvalues[0] / total_var if total_var > 0 else 0.0

    eig_pos = eigenvalues[eigenvalues > 0]
    p = eig_pos / np.sum(eig_pos) if len(eig_pos) > 0 else np.array([1.0])
    eff_dim = 1.0 / np.sum(p**2)

    return CorrelationMatrixResult(
        pearson_matrix=pearson_matrix, spearman_matrix=spearman_matrix,
        mean_correlation=mean_corr, median_correlation=median_corr,
        correlation_dispersion=dispersion, n_strong_pairs=n_strong, n_weak_pairs=n_weak,
        eigenvalues=eigenvalues, variance_explained_1=float(var_exp_1),
        effective_dimension=float(eff_dim)
    )
