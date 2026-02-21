"""
Rate of Change Engine.

Delegates to pmtvs rate_of_change primitive.
"""

import warnings

import numpy as np
from typing import Dict
from manifold.core._pmtvs import rate_of_change


def compute(y: np.ndarray, I: np.ndarray = None) -> Dict[str, float]:
    """
    Compute rate of change metrics.

    Args:
        y: Signal values
        I: Index values (optional, assumes uniform spacing dt=1 if None)

    Returns:
        dict with mean_rate, max_rate, min_rate, rate_std, abs_max_rate
    """
    result = {
        'mean_rate': np.nan,
        'max_rate': np.nan,
        'min_rate': np.nan,
        'rate_std': np.nan,
        'abs_max_rate': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 3:
        return result

    try:
        roc = rate_of_change(y)  # returns array of differences

        result = {
            'mean_rate': float(np.mean(roc)),
            'max_rate': float(np.max(roc)),
            'min_rate': float(np.min(roc)),
            'rate_std': float(np.std(roc)),
            'abs_max_rate': float(np.max(np.abs(roc))),
        }
    except Exception as e:
        warnings.warn(f"rate_of_change.compute: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    return result
