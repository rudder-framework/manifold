"""
Derivative Analysis
===================

Computes first, second, and third derivatives (velocity, acceleration, jerk)
for trajectory characterization.

Key metrics:
    - d1: Velocity (rate of change)
    - d2: Acceleration (change in velocity)
    - d3: Jerk (smoothness indicator)
    - momentum_strength: |mean velocity| / volatility

Sign changes in derivatives are EARLY WARNING signals
of direction reversals.
"""

import numpy as np
from typing import Dict


def compute(series: np.ndarray) -> Dict[str, float]:
    """
    Compute derivative statistics.

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - d1_mean: Mean velocity
            - d1_std: Velocity variability
            - d1_max: Maximum speed
            - d2_mean: Mean acceleration
            - d2_std: Acceleration variability
            - d2_max: Maximum acceleration
            - d3_mean: Mean jerk
            - d3_std: Jerk variability (smoothness inverse)
            - momentum_strength: |d1_mean| / d1_std
            - acceleration_regime: 'accelerating' | 'decelerating' | 'neutral'
            - smoothness: 1 / (1 + jerk_std)
            - d1_sign_changes: Velocity reversals
            - d2_sign_changes: Acceleration reversals
            - sign_change_rate: Sign changes per observation
    """
    n = len(series)

    if n < 4:
        return _empty_result()

    # Compute derivatives
    d1 = np.diff(series)  # Velocity
    d2 = np.diff(d1) if len(d1) > 1 else np.array([0])  # Acceleration
    d3 = np.diff(d2) if len(d2) > 1 else np.array([0])  # Jerk

    # First derivative (velocity) statistics
    d1_mean = float(np.mean(d1))
    d1_std = float(np.std(d1)) if len(d1) > 1 else 0.0
    d1_max = float(np.max(np.abs(d1)))

    # Momentum strength: |mean| / std (trending vs oscillating)
    momentum_strength = abs(d1_mean) / (d1_std + 1e-10)

    # Second derivative (acceleration) statistics
    d2_mean = float(np.mean(d2)) if len(d2) > 0 else 0.0
    d2_std = float(np.std(d2)) if len(d2) > 1 else 0.0
    d2_max = float(np.max(np.abs(d2))) if len(d2) > 0 else 0.0

    # Acceleration regime
    if d2_mean > d2_std * 0.5:
        accel_regime = 'accelerating'
    elif d2_mean < -d2_std * 0.5:
        accel_regime = 'decelerating'
    else:
        accel_regime = 'neutral'

    # Third derivative (jerk) statistics
    d3_mean = float(np.mean(d3)) if len(d3) > 0 else 0.0
    d3_std = float(np.std(d3)) if len(d3) > 1 else 0.0

    # Smoothness: inverse of jerk variance
    smoothness = 1.0 / (1.0 + d3_std)

    # Sign changes (early warning signals)
    d1_sign_changes = _count_sign_changes(d1)
    d2_sign_changes = _count_sign_changes(d2)
    sign_change_rate = d1_sign_changes / n

    return {
        'd1_mean': d1_mean,
        'd1_std': d1_std,
        'd1_max': d1_max,
        'd2_mean': d2_mean,
        'd2_std': d2_std,
        'd2_max': d2_max,
        'd3_mean': d3_mean,
        'd3_std': d3_std,
        'momentum_strength': float(momentum_strength),
        'acceleration_regime': accel_regime,
        'smoothness': float(smoothness),
        'd1_sign_changes': d1_sign_changes,
        'd2_sign_changes': d2_sign_changes,
        'sign_change_rate': float(sign_change_rate)
    }


def _count_sign_changes(arr: np.ndarray) -> int:
    """Count number of sign changes in array."""
    if len(arr) < 2:
        return 0
    signs = np.sign(arr)
    signs = signs[signs != 0]  # Remove zeros
    if len(signs) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def _empty_result() -> Dict[str, float]:
    """Return empty result for insufficient data."""
    return {
        'd1_mean': 0.0,
        'd1_std': 0.0,
        'd1_max': 0.0,
        'd2_mean': 0.0,
        'd2_std': 0.0,
        'd2_max': 0.0,
        'd3_mean': 0.0,
        'd3_std': 0.0,
        'momentum_strength': 0.0,
        'acceleration_regime': 'neutral',
        'smoothness': 1.0,
        'd1_sign_changes': 0,
        'd2_sign_changes': 0,
        'sign_change_rate': 0.0
    }
