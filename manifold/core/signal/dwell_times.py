"""
Dwell Times Engine.

Delegates to pmtvs dwell_times primitive.
"""

import numpy as np
from typing import Dict, Any
from manifold.primitives.individual.discrete import dwell_times


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute dwell time statistics.

    Parameters
    ----------
    y : np.ndarray
        Input signal (discrete or continuous).
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Dwell time statistics.
    """
    y = np.asarray(y).flatten()

    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")

    r = dwell_times(y)

    mean_dwell = r.get('mean_dwell', np.nan)
    max_dwell = r.get('max_dwell', np.nan)
    min_dwell = r.get('min_dwell', np.nan)

    return {
        'dwell_mean': mean_dwell,
        'dwell_std': np.nan,
        'dwell_max': max_dwell,
        'dwell_min': min_dwell,
        'dwell_cv': np.nan,
        'n_dwells': r.get('n_states', 0),
    }
