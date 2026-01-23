"""
Dirac Impulse Detection
=======================

Detects impulse (δ-like) discontinuities characterized by:
    - Sharp spike above threshold
    - Decay back toward baseline
    - Transient effect (not permanent)

Examples:
    - News shocks
    - Error spikes
    - Anomalies

The Dirac impulse is the derivative of the Heaviside step.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Dict, List


def compute(
    series: np.ndarray,
    threshold_sigma: float = 3.0,
    decay_window: int = 5
) -> Dict[str, float]:
    """
    Detect impulse (Dirac-like) discontinuities.

    Args:
        series: 1D numpy array of observations
        threshold_sigma: Z-score threshold for spike detection
        decay_window: Window to look for decay

    Returns:
        dict with:
            - detected: Boolean - any impulses found?
            - count: Number of impulses
            - max_magnitude: Largest impulse (σ units)
            - mean_magnitude: Average impulse size
            - mean_half_life: Average decay rate
            - up_ratio: Fraction of positive impulses
            - locations: Indices of impulses (list)
    """
    n = len(series)

    if n < 10:
        return _empty_result()

    # Rolling window for local statistics
    window = min(20, n // 5)
    if window < 3:
        return _empty_result()

    # Detrend using rolling mean
    trend = uniform_filter1d(series.astype(float), size=window, mode='nearest')
    detrended = series - trend

    # Rolling std
    rolling_std = np.zeros(n)
    for i in range(window, n):
        rolling_std[i] = np.std(detrended[i-window:i])
    rolling_std[:window] = rolling_std[window] if window < n else 1.0
    rolling_std[rolling_std < 1e-10] = 1.0

    # Z-scores
    z_scores = detrended / rolling_std

    # Find spikes
    spike_mask = np.abs(z_scores) > threshold_sigma
    spike_indices = np.where(spike_mask)[0]

    if len(spike_indices) == 0:
        return _empty_result()

    # Cluster nearby spikes
    impulses = []
    current_cluster = [spike_indices[0]]

    for idx in spike_indices[1:]:
        if idx - current_cluster[-1] <= decay_window:
            current_cluster.append(idx)
        else:
            peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
            impulses.append(peak_idx)
            current_cluster = [idx]

    # Don't forget last cluster
    peak_idx = current_cluster[np.argmax(np.abs(z_scores[current_cluster]))]
    impulses.append(peak_idx)

    # Compute metrics
    magnitudes = np.abs(z_scores[impulses])
    directions = np.sign(z_scores[impulses])

    # Estimate half-lives
    half_lives = []
    for imp_idx in impulses:
        peak_val = np.abs(detrended[imp_idx])
        half_val = peak_val / 2

        for k in range(1, min(decay_window * 2, n - imp_idx)):
            if np.abs(detrended[imp_idx + k]) < half_val:
                half_lives.append(k)
                break
        else:
            half_lives.append(decay_window)

    return {
        'detected': True,
        'count': len(impulses),
        'max_magnitude': float(np.max(magnitudes)),
        'mean_magnitude': float(np.mean(magnitudes)),
        'mean_half_life': float(np.mean(half_lives)) if half_lives else float(decay_window),
        'up_ratio': float(np.mean(directions > 0)),
        'locations': impulses
    }


def _empty_result() -> Dict[str, float]:
    """Return empty result for no detections."""
    return {
        'detected': False,
        'count': 0,
        'max_magnitude': 0.0,
        'mean_magnitude': 0.0,
        'mean_half_life': 0.0,
        'up_ratio': 0.5,
        'locations': []
    }
