"""
ADF Stat Engine.

Augmented Dickey-Fuller test for stationarity.
Imports from primitives/individual/stationarity.py (canonical).
"""

import numpy as np
from typing import Dict
from prism.primitives.individual.stationarity import stationarity_test


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute Augmented Dickey-Fuller test statistic.

    Args:
        y: Signal values (1D array)

    Returns:
        dict with:
            adf_stat: Test statistic (more negative = more stationary)
            adf_pvalue: p-value (< 0.05 typically means stationary)

    Notes:
        H0 = unit root (non-stationary)
        p < 0.05 → reject H0 → stationary
    """
    result = {
        'adf_stat': np.nan,
        'adf_pvalue': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 20:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        # Constant signal is trivially stationary
        result['adf_stat'] = -np.inf
        result['adf_pvalue'] = 0.0
        return result

    try:
        stat, pvalue = stationarity_test(y, test='adf')
        result['adf_stat'] = stat
        result['adf_pvalue'] = pvalue
    except Exception:
        pass

    return result
