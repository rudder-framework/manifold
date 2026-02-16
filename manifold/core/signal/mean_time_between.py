"""
Mean Time Between Engine
========================

Computes per-state mean consecutive run lengths, then summarizes
across states into fixed-width statistics.

Returns summary statistics, NOT per-state columns
(which would be variable-size and break parquet schema).

Designed for DISCRETE, BINARY, STEP, EVENT signal types.

Outputs:
    mtb_mean  - Mean of per-state mean run lengths
    mtb_std   - Std of per-state mean run lengths
    mtb_max   - Longest per-state mean run length (stickiest state)
    mtb_min   - Shortest per-state mean run length (most transient state)
    mtb_cv    - Coefficient of variation (mtb_std / mtb_mean)

Physics:
    - mtb_mean increasing → system dwelling longer (slowing down)
    - mtb_mean decreasing → system cycling faster (speeding up)
    - mtb_cv high → some states much stickier than others
    - mtb_cv low → all states have similar dwell times
    - mtb_max / mtb_min diverging → asymmetric state dynamics
    - This is the discrete equivalent of autocorrelation time —
      how long does the system stay in each state?

Relationship to other engines:
    dwell_times       → statistics across ALL runs (pooled across states)
    mean_time_between → per-state mean run lengths, then summarized across states

The distinction matters: dwell_times treats all runs equally regardless of
which state they belong to. mean_time_between first aggregates per-state,
then computes cross-state statistics. A system with two states where state A
dwells for 100 samples and state B dwells for 2 samples will have the same
dwell_mean (~51) but very different mtb statistics (mtb_max=100, mtb_min=2,
mtb_cv high).
"""

import numpy as np
from typing import Dict, Any


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute mean time between transitions per state, then summarize.

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
        Mean-time-between summary statistics.
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

    # Find run boundaries
    changes = np.where(levels[1:] != levels[:-1])[0] + 1
    boundaries = np.concatenate([[0], changes, [len(levels)]])

    # Collect run lengths per state
    run_lengths_per_state: Dict[int, list] = {}
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        state = int(levels[start])
        length = end - start
        if state not in run_lengths_per_state:
            run_lengths_per_state[state] = []
        run_lengths_per_state[state].append(length)

    # Compute per-state mean run length
    state_means = np.array([
        float(np.mean(lengths))
        for lengths in run_lengths_per_state.values()
    ])

    if len(state_means) == 0:
        return {
            'mtb_mean': np.nan,
            'mtb_std': np.nan,
            'mtb_max': np.nan,
            'mtb_min': np.nan,
            'mtb_cv': np.nan,
        }

    mean_val = float(np.mean(state_means))
    std_val = float(np.std(state_means))

    return {
        'mtb_mean': mean_val,
        'mtb_std': std_val,
        'mtb_max': float(np.max(state_means)),
        'mtb_min': float(np.min(state_means)),
        'mtb_cv': std_val / mean_val if mean_val > 0 else np.nan,
    }
