"""
Heaviside Step Detection
========================

Detects step (H-like) discontinuities characterized by:
    - Permanent level shift
    - New stable level after change
    - Non-transient effect

Examples:
    - Regime changes
    - Structural breaks
    - Policy changes

The Heaviside step is the integral of the Dirac impulse.
"""

import numpy as np
from typing import Dict, List


def compute(
    series: np.ndarray,
    threshold_sigma: float = 2.0,
    min_stable_periods: int = 5
) -> Dict[str, float]:
    """
    Detect step (Heaviside-like) discontinuities.

    Args:
        series: 1D numpy array of observations
        threshold_sigma: Threshold for jump detection (σ units)
        min_stable_periods: Minimum periods for post-jump stability

    Returns:
        dict with:
            - detected: Boolean - any steps found?
            - count: Number of steps
            - max_magnitude: Largest step (σ units)
            - mean_magnitude: Average step size
            - up_ratio: Fraction of positive steps
            - locations: Indices of steps (list)
    """
    n = len(series)

    if n < min_stable_periods * 3:
        return _empty_result()

    # Compute differences
    diff = np.diff(series)

    # Rolling std for threshold
    window = min(20, n // 5)
    rolling_std = np.std(series[:window]) if window > 0 else 1.0
    if rolling_std < 1e-10:
        rolling_std = 1.0

    threshold = threshold_sigma * rolling_std

    # Find large jumps
    jump_mask = np.abs(diff) > threshold
    jump_indices = np.where(jump_mask)[0]

    if len(jump_indices) == 0:
        return _empty_result()

    # Verify each jump is a true step (level persists)
    confirmed_steps = []

    for idx in jump_indices:
        # Check if level is stable after jump
        post_start = idx + 1
        post_end = min(idx + 1 + min_stable_periods, n)

        if post_end - post_start < min_stable_periods:
            continue

        post_segment = series[post_start:post_end]
        post_std = np.std(post_segment)

        # If post-jump segment is relatively stable, it's a step
        if post_std < rolling_std * 1.5:
            confirmed_steps.append(idx)

    if len(confirmed_steps) == 0:
        return _empty_result()

    # Compute metrics
    step_magnitudes = np.abs(diff[confirmed_steps]) / rolling_std
    step_directions = np.sign(diff[confirmed_steps])

    return {
        'detected': True,
        'count': len(confirmed_steps),
        'max_magnitude': float(np.max(step_magnitudes)),
        'mean_magnitude': float(np.mean(step_magnitudes)),
        'up_ratio': float(np.mean(step_directions > 0)),
        'locations': confirmed_steps
    }


def _empty_result() -> Dict[str, float]:
    """Return empty result for no detections."""
    return {
        'detected': False,
        'count': 0,
        'max_magnitude': 0.0,
        'mean_magnitude': 0.0,
        'up_ratio': 0.5,
        'locations': []
    }
