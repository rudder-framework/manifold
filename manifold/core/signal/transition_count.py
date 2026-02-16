"""
Transition Count Engine
=======================

Counts total state transitions within a window.
The simplest scalar measure of discrete signal activity.

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    transition_count - Total number of state changes in the window

Physics:
    - High transition_count = chattering / oscillating between states
    - Low transition_count = stable in one state
    - transition_count trending up = system destabilizing
    - transition_count trending down = system settling
    - This is the discrete equivalent of zero-crossing rate for continuous signals.

Relationship to other engines:
    transition_matrix → full NxN state-to-state counts
    transition_count  → scalar: sum of off-diagonal (total state changes)
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 2


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Count state transitions in a window.

    Parameters
    ----------
    y : np.ndarray
        Input signal (discrete or continuous).
    n_bins : int
        Number of bins for continuous signals. Ignored if signal
        has fewer than n_bins unique values (already discrete).

    Returns
    -------
    dict
        {'transition_count': int}
    """
    y = np.asarray(y).flatten()

    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")

    # Remove NaNs but preserve order
    mask = ~np.isnan(y)
    y_clean = y[mask]

    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} non-NaN samples, got {len(y_clean)}")

    # Discretize if continuous
    unique_vals = np.unique(y_clean)
    if len(unique_vals) > n_bins:
        bins = np.linspace(np.nanmin(y_clean), np.nanmax(y_clean), n_bins + 1)
        levels = np.digitize(y_clean, bins, right=True)
    else:
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label[v] for v in y_clean])

    # Count transitions: number of times consecutive samples differ
    count = int(np.sum(levels[1:] != levels[:-1]))

    return {
        'transition_count': count,
    }
