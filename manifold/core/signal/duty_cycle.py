"""
Duty Cycle Engine
=================

Computes state occupancy statistics for discrete signals.
Returns fixed-width summary statistics, NOT per-state columns
(which would be variable-size and break parquet schema).

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    duty_dominant   - Fraction of window in the most occupied state
    duty_secondary  - Fraction of window in the second most occupied state
    duty_ratio      - dominant / secondary (how lopsided the occupancy is)
    duty_balance    - Normalized entropy of occupancy (1.0 = uniform, 0.0 = one state)
    duty_range      - Max duty - min duty (spread of state occupancy fractions)

Physics:
    - duty_dominant → 1.0: signal stuck in one state
    - duty_balance → 1.0: signal visits all states equally
    - duty_ratio spiking: one state dominating over the next
    - duty_range increasing: state occupancy becoming more uneven
    - This is the discrete equivalent of the mean for continuous signals —
      where does the signal live most of the time?

Relationship to other engines:
    level_histogram → shape statistics of the level distribution
    level_count     → number of distinct states + dominant fraction
    duty_cycle      → occupancy fractions summarized as balance/spread metrics
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute duty cycle (state occupancy) statistics.

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
        Duty cycle summary statistics.
    """
    y = np.asarray(y).flatten()
    y_clean = y[~np.isnan(y)]

    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y_clean)}")

    # Discretize if continuous
    unique_vals = np.unique(y_clean)
    if len(unique_vals) > n_bins:
        bins = np.linspace(np.nanmin(y_clean), np.nanmax(y_clean), n_bins + 1)
        levels = np.digitize(y_clean, bins, right=True)
    else:
        val_to_label = {v: i for i, v in enumerate(sorted(unique_vals))}
        levels = np.array([val_to_label[v] for v in y_clean])

    # Compute fractions per state
    _, counts = np.unique(levels, return_counts=True)
    total = len(y_clean)
    fractions = counts / total
    k = len(fractions)

    # Sort descending for dominant/secondary
    sorted_fracs = np.sort(fractions)[::-1]

    dominant = float(sorted_fracs[0])
    secondary = float(sorted_fracs[1]) if k >= 2 else 0.0
    ratio = dominant / secondary if secondary > 0 else float('inf')

    # Balance: normalized entropy of occupancy distribution
    # 1.0 = all states equally occupied, 0.0 = single state
    if k > 1:
        entropy = -np.sum(fractions * np.log2(fractions + 1e-12))
        max_entropy = np.log2(k)
        balance = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        balance = 0.0

    # Range: max fraction - min fraction
    duty_range = float(sorted_fracs[0] - sorted_fracs[-1])

    return {
        'duty_dominant': dominant,
        'duty_secondary': secondary,
        'duty_ratio': float(ratio),
        'duty_balance': balance,
        'duty_range': duty_range,
    }
