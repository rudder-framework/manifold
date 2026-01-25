"""
Lead/Lag Detection
==================

Identifies temporal relationships between signals.

Methods:
    - Cross-correlation at multiple lags
    - Optimal lag finding
    - Lead/lag stability over time

Unlike Granger causality, this directly measures correlation at different offsets.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class LeadLagResult:
    """Output from lead/lag analysis"""

    # Optimal lag (positive = x leads, negative = y leads)
    optimal_lag: int
    optimal_correlation: float

    # Cross-correlation function
    lags: np.ndarray
    cross_correlations: np.ndarray

    # Contemporaneous correlation (lag 0)
    contemporaneous_corr: float

    # Lead/lag classification
    relationship: str  # 'x_leads' | 'y_leads' | 'contemporaneous' | 'ambiguous'
    lead_magnitude: int  # Absolute lag

    # Confidence
    confidence: float  # How much better is optimal vs contemporaneous

    # Stability (if rolling)
    lag_stable: bool = True
    lag_std: float = 0.0


@dataclass
class LeadLagMatrixResult:
    """Output from multi-signal lead/lag analysis"""

    # Optimal lag matrix [i,j] = optimal lag for i vs j
    lag_matrix: np.ndarray

    # Correlation at optimal lag
    correlation_at_optimal: np.ndarray

    # Leader scores (how often each signal leads)
    leader_scores: np.ndarray

    # Top leaders
    top_leaders: List[int]


def compute(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 20,
    rolling_window: Optional[int] = None
) -> LeadLagResult:
    """
    Detect lead/lag relationship between two signals.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag to test in either direction
        rolling_window: Window for stability analysis

    Returns:
        LeadLagResult
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Remove means
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Cross-correlation
    lags = np.arange(-max_lag, max_lag + 1)
    cross_corrs = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            cross_corrs[i] = np.corrcoef(x, y)[0, 1]
        elif lag > 0:
            # x leads y (x at time t correlated with y at time t+lag)
            if n - lag > 2:
                cross_corrs[i] = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            # y leads x (y at time t correlated with x at time t+|lag|)
            abs_lag = abs(lag)
            if n - abs_lag > 2:
                cross_corrs[i] = np.corrcoef(x[abs_lag:], y[:-abs_lag])[0, 1]

    # Handle NaN
    cross_corrs = np.nan_to_num(cross_corrs, nan=0.0)

    # Find optimal lag
    optimal_idx = np.argmax(np.abs(cross_corrs))
    optimal_lag = int(lags[optimal_idx])
    optimal_corr = float(cross_corrs[optimal_idx])

    # Contemporaneous
    contemp_idx = np.where(lags == 0)[0][0]
    contemp_corr = float(cross_corrs[contemp_idx])

    # Classify relationship
    if optimal_lag > 1:
        relationship = "x_leads"
    elif optimal_lag < -1:
        relationship = "y_leads"
    elif abs(optimal_corr) > abs(contemp_corr) + 0.1:
        relationship = "ambiguous"
    else:
        relationship = "contemporaneous"

    # Confidence: improvement over contemporaneous
    confidence = abs(optimal_corr) - abs(contemp_corr)
    confidence = max(0.0, min(1.0, confidence * 5))  # Scale to 0-1

    # Rolling stability analysis
    lag_stable = True
    lag_std = 0.0

    if rolling_window and rolling_window < n - max_lag:
        rolling_lags = []
        for start in range(0, n - rolling_window, rolling_window // 2):
            end = start + rolling_window
            if end > n:
                break

            x_win = x[start:end]
            y_win = y[start:end]

            best_lag, best_corr = 0, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    r = np.corrcoef(x_win, y_win)[0, 1]
                elif lag > 0 and end - lag > start + 2:
                    r = np.corrcoef(x_win[:-lag], y_win[lag:])[0, 1]
                elif lag < 0 and end - abs(lag) > start + 2:
                    r = np.corrcoef(x_win[abs(lag):], y_win[:-abs(lag)])[0, 1]
                else:
                    continue

                if abs(r) > abs(best_corr):
                    best_lag, best_corr = lag, r

            rolling_lags.append(best_lag)

        if len(rolling_lags) > 1:
            lag_std = float(np.std(rolling_lags))
            lag_stable = lag_std < 3  # Stable if std < 3 periods

    return LeadLagResult(
        optimal_lag=optimal_lag,
        optimal_correlation=optimal_corr,
        lags=lags,
        cross_correlations=cross_corrs,
        contemporaneous_corr=contemp_corr,
        relationship=relationship,
        lead_magnitude=abs(optimal_lag),
        confidence=float(confidence),
        lag_stable=lag_stable,
        lag_std=lag_std
    )


def compute_matrix(
    signals: np.ndarray,
    max_lag: int = 20
) -> LeadLagMatrixResult:
    """
    Compute lead/lag relationships for multiple signals.

    Args:
        signals: 2D array (n_signals, n_observations)
        max_lag: Maximum lag to test

    Returns:
        LeadLagMatrixResult
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]

    lag_matrix = np.zeros((n_signals, n_signals), dtype=int)
    corr_at_optimal = np.zeros((n_signals, n_signals))

    for i in range(n_signals):
        for j in range(n_signals):
            if i == j:
                corr_at_optimal[i, j] = 1.0
                continue

            result = compute(signals[i], signals[j], max_lag)
            lag_matrix[i, j] = result.optimal_lag
            corr_at_optimal[i, j] = result.optimal_correlation

    # Leader scores (count how often each signal leads others)
    leader_scores = np.zeros(n_signals)
    for i in range(n_signals):
        for j in range(n_signals):
            if i != j and lag_matrix[i, j] > 1:
                leader_scores[i] += 1

    leader_scores = leader_scores / (n_signals - 1) if n_signals > 1 else leader_scores

    # Top leaders
    top_leaders = list(np.argsort(leader_scores)[::-1][:3])

    return LeadLagMatrixResult(
        lag_matrix=lag_matrix,
        correlation_at_optimal=corr_at_optimal,
        leader_scores=leader_scores,
        top_leaders=top_leaders
    )
