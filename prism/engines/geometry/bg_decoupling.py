"""
Decoupling Detection
====================

Detects when historical relationships between signals break down.

This is the ALERT system of behavioral geometry:
    - Correlations that flip sign
    - Stable relationships that suddenly weaken
    - Cluster memberships that change
    - Network structure that fragments

Decoupling often precedes or accompanies regime changes.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DecouplingType(Enum):
    """Types of relationship breakdown"""
    NONE = "none"
    WEAKENING = "weakening"           # Correlation dropping
    SIGN_FLIP = "sign_flip"           # Positive -> negative or vice versa
    VOLATILITY_DIVERGENCE = "vol_divergence"  # Volatility ratio changed
    LEAD_LAG_SHIFT = "lead_lag_shift"  # Leadership changed
    STRUCTURAL = "structural"          # Multiple changes at once


@dataclass
class DecouplingResult:
    """Output from pairwise decoupling detection"""

    # Detection
    decoupling_detected: bool
    decoupling_type: DecouplingType

    # Correlation breakdown
    historical_correlation: float     # Long-term average
    recent_correlation: float         # Short-term average
    correlation_change: float         # Recent - historical
    correlation_zscore: float         # How unusual is recent?

    # Sign flip detection
    sign_flipped: bool
    sign_flip_date_idx: Optional[int] = None

    # Volatility ratio
    historical_vol_ratio: float = 1.0  # sigma_x / sigma_y historical
    recent_vol_ratio: float = 1.0      # sigma_x / sigma_y recent
    vol_ratio_change: float = 0.0

    # Timing
    decoupling_start_idx: Optional[int] = None
    decoupling_duration: int = 0

    # Severity
    severity: str = "none"            # none | mild | moderate | severe


@dataclass
class DecouplingMatrixResult:
    """Output from multi-signal decoupling analysis"""

    # Decoupling flags [i,j] = True if decoupled
    decoupling_matrix: np.ndarray

    # Type matrix
    type_matrix: np.ndarray           # Stores DecouplingType.value

    # Severity matrix
    severity_matrix: np.ndarray       # 0-3 scale

    # Summary
    n_decoupled_pairs: int
    n_severe: int

    # Most affected signals
    most_decoupled: List[int]         # Signals with most broken relationships

    # Alerts
    alerts: List[str]


def compute(
    x: np.ndarray,
    y: np.ndarray,
    historical_window: int = 100,
    recent_window: int = 20,
    correlation_threshold: float = 0.3,
    zscore_threshold: float = 2.0
) -> DecouplingResult:
    """
    Detect decoupling between two signals.

    Args:
        x, y: Signals (same length)
        historical_window: Window for historical baseline
        recent_window: Window for recent comparison
        correlation_threshold: Minimum change to flag
        zscore_threshold: Z-score threshold for unusual

    Returns:
        DecouplingResult
    """
    from scipy import stats

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < historical_window + recent_window:
        return DecouplingResult(
            decoupling_detected=False,
            decoupling_type=DecouplingType.NONE,
            historical_correlation=0.0,
            recent_correlation=0.0,
            correlation_change=0.0,
            correlation_zscore=0.0,
            sign_flipped=False,
            severity="none"
        )

    # Split into historical and recent
    x_hist = x[:historical_window]
    y_hist = y[:historical_window]
    x_recent = x[-recent_window:]
    y_recent = y[-recent_window:]

    # Correlations
    hist_corr, _ = stats.pearsonr(x_hist, y_hist)
    recent_corr, _ = stats.pearsonr(x_recent, y_recent)

    # Rolling correlation for z-score calculation
    rolling_corrs = []
    for i in range(recent_window, n - recent_window + 1, recent_window // 2):
        r, _ = stats.pearsonr(x[i-recent_window:i], y[i-recent_window:i])
        rolling_corrs.append(r)

    if len(rolling_corrs) > 2:
        corr_mean = np.mean(rolling_corrs[:-1])  # Exclude most recent
        corr_std = np.std(rolling_corrs[:-1])
        corr_zscore = (recent_corr - corr_mean) / corr_std if corr_std > 0 else 0.0
    else:
        corr_zscore = 0.0

    corr_change = recent_corr - hist_corr

    # Sign flip detection
    sign_flipped = (hist_corr * recent_corr < 0) and abs(hist_corr) > 0.2 and abs(recent_corr) > 0.2

    # Find sign flip location
    sign_flip_idx = None
    if sign_flipped:
        for i in range(historical_window, n - recent_window):
            window_corr, _ = stats.pearsonr(x[i:i+recent_window], y[i:i+recent_window])
            if hist_corr * window_corr < 0:
                sign_flip_idx = i
                break

    # Volatility ratio analysis
    hist_vol_x = np.std(x_hist)
    hist_vol_y = np.std(y_hist)
    recent_vol_x = np.std(x_recent)
    recent_vol_y = np.std(y_recent)

    hist_vol_ratio = hist_vol_x / hist_vol_y if hist_vol_y > 0 else 1.0
    recent_vol_ratio = recent_vol_x / recent_vol_y if recent_vol_y > 0 else 1.0
    vol_ratio_change = recent_vol_ratio / hist_vol_ratio if hist_vol_ratio > 0 else 1.0

    # Determine decoupling type
    decoupling_type = DecouplingType.NONE
    decoupling_detected = False

    if sign_flipped:
        decoupling_type = DecouplingType.SIGN_FLIP
        decoupling_detected = True
    elif abs(corr_change) > correlation_threshold and abs(corr_zscore) > zscore_threshold:
        decoupling_type = DecouplingType.WEAKENING
        decoupling_detected = True
    elif abs(np.log(vol_ratio_change)) > 0.5:  # 65% change in ratio
        decoupling_type = DecouplingType.VOLATILITY_DIVERGENCE
        decoupling_detected = True

    # Multiple simultaneous changes = structural
    n_changes = sum([
        abs(corr_change) > correlation_threshold,
        sign_flipped,
        abs(np.log(vol_ratio_change)) > 0.5
    ])
    if n_changes >= 2:
        decoupling_type = DecouplingType.STRUCTURAL

    # Severity
    if not decoupling_detected:
        severity = "none"
    elif sign_flipped or n_changes >= 2:
        severity = "severe"
    elif abs(corr_zscore) > 3:
        severity = "moderate"
    else:
        severity = "mild"

    # Decoupling timing
    decoupling_start = sign_flip_idx if sign_flip_idx else (n - recent_window if decoupling_detected else None)
    decoupling_duration = n - decoupling_start if decoupling_start else 0

    return DecouplingResult(
        decoupling_detected=decoupling_detected,
        decoupling_type=decoupling_type,
        historical_correlation=float(hist_corr),
        recent_correlation=float(recent_corr),
        correlation_change=float(corr_change),
        correlation_zscore=float(corr_zscore),
        sign_flipped=sign_flipped,
        sign_flip_date_idx=sign_flip_idx,
        historical_vol_ratio=float(hist_vol_ratio),
        recent_vol_ratio=float(recent_vol_ratio),
        vol_ratio_change=float(vol_ratio_change),
        decoupling_start_idx=decoupling_start,
        decoupling_duration=decoupling_duration,
        severity=severity
    )


def compute_matrix(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    historical_window: int = 100,
    recent_window: int = 20
) -> DecouplingMatrixResult:
    """
    Detect decoupling across multiple signals.

    Args:
        signals: 2D array (n_signals, n_observations)
        signal_ids: Optional signal identifiers
        historical_window: Window for historical baseline
        recent_window: Window for recent comparison

    Returns:
        DecouplingMatrixResult
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]

    decoupling_matrix = np.zeros((n_signals, n_signals), dtype=bool)
    type_matrix = np.empty((n_signals, n_signals), dtype=object)
    severity_matrix = np.zeros((n_signals, n_signals))

    type_matrix.fill(DecouplingType.NONE.value)

    alerts = []

    for i in range(n_signals):
        for j in range(i+1, n_signals):
            result = compute(
                signals[i], signals[j],
                historical_window, recent_window
            )

            decoupling_matrix[i, j] = decoupling_matrix[j, i] = result.decoupling_detected
            type_matrix[i, j] = type_matrix[j, i] = result.decoupling_type.value

            sev_val = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}[result.severity]
            severity_matrix[i, j] = severity_matrix[j, i] = sev_val

            # Generate alerts
            if result.severity == "severe":
                sig_i = signal_ids[i] if signal_ids else f"signal_{i}"
                sig_j = signal_ids[j] if signal_ids else f"signal_{j}"
                alerts.append(
                    f"SEVERE: {sig_i} <-> {sig_j} decoupled ({result.decoupling_type.value})"
                )

    n_decoupled = int(np.sum(decoupling_matrix)) // 2
    n_severe = int(np.sum(severity_matrix >= 3)) // 2

    # Most decoupled signals
    decoupled_counts = np.sum(decoupling_matrix, axis=1)
    most_decoupled = list(np.argsort(decoupled_counts)[::-1][:3])

    return DecouplingMatrixResult(
        decoupling_matrix=decoupling_matrix,
        type_matrix=type_matrix,
        severity_matrix=severity_matrix,
        n_decoupled_pairs=n_decoupled,
        n_severe=n_severe,
        most_decoupled=most_decoupled,
        alerts=alerts
    )
