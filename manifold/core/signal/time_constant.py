"""
Time Constant Engine.

Delegates to pmtvs time_constant primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.domain import time_constant


def compute(y: np.ndarray, I: np.ndarray = None) -> Dict[str, float]:
    """
    Estimate exponential time constant (tau in index units).

    Args:
        y: Signal values
        I: Index values (optional, unused — pmtvs uses 0..n-1)

    Returns:
        dict with time_constant, equilibrium_value, fit_r2, is_decay
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    result = {
        'time_constant': np.nan,
        'equilibrium_value': np.nan,
        'fit_r2': np.nan,
        'is_decay': None,
    }

    if len(y) < 10:
        return result

    r = time_constant(y)

    result['time_constant'] = r.get('tau', np.nan)
    result['fit_r2'] = r.get('r_squared', np.nan)

    # Determine decay/rise from signal endpoints
    n = len(y)
    y_start = float(np.mean(y[:max(1, n // 10)]))
    y_end = float(np.mean(y[-max(1, n // 10):]))
    result['is_decay'] = y_start > y_end
    result['equilibrium_value'] = y_end

    return result
