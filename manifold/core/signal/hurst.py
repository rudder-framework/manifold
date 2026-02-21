"""
Hurst Exponent Engine.

Imports from primitives/individual/fractal.py (canonical).
Primitives handle min_samples via config - no redundant checks here.
"""

import numpy as np
from typing import Dict
from pmtvs import hurst_exponent
from manifold.core._pmtvs import dfa
from manifold.core._compat import hurst_r2


def compute(y: np.ndarray, method: str = 'rs') -> Dict[str, float]:
    """
    Compute Hurst exponent of signal.

    Args:
        y: Signal values
        method: 'rs' (rescaled range) or 'dfa' (detrended fluctuation)

    Returns:
        dict with 'hurst' and 'hurst_r2' keys
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    return {
        'hurst': hurst_exponent(y, method=method),
        'hurst_r2': hurst_r2(y),
    }


def compute_dfa(y: np.ndarray) -> Dict[str, float]:
    """Compute DFA exponent."""
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    return {'dfa': dfa(y)}
