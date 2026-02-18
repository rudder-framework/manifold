"""
Rate of Change Engine.

Delegates to pmtvs rate_of_change primitive.
"""

import warnings

import numpy as np
from typing import Dict
from manifold.primitives.individual.trend_features import rate_of_change


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
        r = rate_of_change(y)

        result = {
            'mean_rate': r.get('mean_roc', np.nan),
            'max_rate': r.get('max_roc', np.nan),
            'min_rate': np.nan,
            'rate_std': r.get('std_roc', np.nan),
            'abs_max_rate': r.get('max_roc', np.nan),
        }
    except Exception as e:
        warnings.warn(f"rate_of_change.compute: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    return result
