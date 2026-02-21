"""
Dwell Times Engine.

Delegates to pmtvs dwell_times primitive.
"""

import numpy as np
from typing import Dict, Any
# TODO: needs pmtvs export — dwell_times


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

    # Inline dwell time computation (dwell_times not yet in pmtvs)
    y_clean = y[~np.isnan(y)]
    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y_clean)}")

    # Compute run lengths (consecutive same values)
    diffs = np.diff(y_clean)
    change_points = np.where(diffs != 0)[0] + 1
    runs = np.diff(np.concatenate([[0], change_points, [len(y_clean)]]))

    return {
        'dwell_mean': float(np.mean(runs)),
        'dwell_std': float(np.std(runs)),
        'dwell_max': float(np.max(runs)),
        'dwell_min': float(np.min(runs)),
        'dwell_cv': float(np.std(runs) / np.mean(runs)) if np.mean(runs) > 0 else np.nan,
        'n_dwells': len(runs),
    }
