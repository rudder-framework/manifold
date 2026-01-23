"""
Structural Discontinuity Analysis
=================================

Analyzes the overall pattern of discontinuities:
    - Interval between discontinuities
    - Regularity of occurrences
    - Acceleration (are they getting more frequent?)

This provides meta-information about the discontinuity
process itself.
"""

import numpy as np
from typing import Dict, List
from .dirac import compute as compute_dirac
from .heaviside import compute as compute_heaviside


def compute(series: np.ndarray) -> Dict[str, float]:
    """
    Analyze structural discontinuity patterns.

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - dirac: Dirac detection results
            - heaviside: Heaviside detection results
            - total_count: Combined discontinuity count
            - mean_interval: Average time between discontinuities
            - interval_cv: Coefficient of variation (regularity)
            - dominant_period: Characteristic period if regular
            - is_accelerating: Are discontinuities getting more frequent?
            - any_detected: Boolean - any discontinuities found?
    """
    # Detect both types
    dirac_result = compute_dirac(series)
    heaviside_result = compute_heaviside(series)

    # Combine locations
    all_locations = sorted(
        dirac_result['locations'] + heaviside_result['locations']
    )

    total_count = len(all_locations)
    any_detected = dirac_result['detected'] or heaviside_result['detected']

    if total_count < 2:
        return {
            'dirac': dirac_result,
            'heaviside': heaviside_result,
            'total_count': total_count,
            'mean_interval': 0.0,
            'interval_cv': 0.0,
            'dominant_period': 0.0,
            'is_accelerating': False,
            'any_detected': any_detected
        }

    # Analyze intervals
    intervals = np.diff(all_locations)
    mean_interval = float(np.mean(intervals))
    interval_cv = float(np.std(intervals) / mean_interval) if mean_interval > 0 else 0.0

    # Check if accelerating (intervals getting shorter)
    if len(intervals) >= 3:
        first_half = np.mean(intervals[:len(intervals)//2])
        second_half = np.mean(intervals[len(intervals)//2:])
        is_accelerating = second_half < first_half * 0.8
    else:
        is_accelerating = False

    # Dominant period (if regular)
    if interval_cv < 0.5:  # Relatively regular
        dominant_period = mean_interval
    else:
        dominant_period = 0.0

    return {
        'dirac': dirac_result,
        'heaviside': heaviside_result,
        'total_count': total_count,
        'mean_interval': mean_interval,
        'interval_cv': interval_cv,
        'dominant_period': dominant_period,
        'is_accelerating': is_accelerating,
        'any_detected': any_detected
    }
