"""
Rolling Crest Factor Engine.

Crest factor over sliding window - detects impulsive events.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from ..signal.basic_stats import compute_crest_factor


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling crest factor.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_crest_factor' array
    """
    params = params or {}
    window = params.get('window', 100)
    stride = params.get('stride', 1)

    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_crest_factor': result}

    for i in range(0, n - window + 1, stride):
        chunk = y[i:i + window]
        result[i + window - 1] = compute_crest_factor(chunk)['crest_factor']

    return {'rolling_crest_factor': result}
