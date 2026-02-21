"""
DMD (Dynamic Mode Decomposition) Engine.

Delegates to pmtvs dmd_analysis primitive.
"""

import numpy as np
from typing import Dict
# TODO: needs pmtvs export — dmd_analysis
from manifold.core._pmtvs import dynamic_mode_decomposition, dmd_frequencies, dmd_growth_rates


def compute(y: np.ndarray, rank: int = None, dt: float = 1.0) -> Dict[str, float]:
    """
    Compute DMD of signal.

    Args:
        y: Signal values
        rank: Maximum rank for truncation (default: auto)
        dt: Index step between consecutive samples (default: 1.0)

    Returns:
        dict with dmd_dominant_freq, dmd_growth_rate, dmd_is_stable, dmd_n_modes
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 10:
        return {
            'dmd_dominant_freq': np.nan,
            'dmd_growth_rate': np.nan,
            'dmd_is_stable': np.nan,
            'dmd_n_modes': np.nan,
        }

    try:
        # Build Hankel matrices for DMD
        half = n // 2
        X = np.column_stack([y[i:i + half] for i in range(n - half)])
        eigenvalues, modes = dynamic_mode_decomposition(X[:, :-1], X[:, 1:])
        freqs = dmd_frequencies(eigenvalues, dt=dt)
        growths = dmd_growth_rates(eigenvalues, dt=dt)

        dominant_idx = np.argmax(np.abs(modes[0])) if len(modes) > 0 else 0
        return {
            'dmd_dominant_freq': float(freqs[dominant_idx]) if len(freqs) > 0 else np.nan,
            'dmd_growth_rate': float(growths[dominant_idx]) if len(growths) > 0 else np.nan,
            'dmd_is_stable': float(np.all(growths <= 0)) if len(growths) > 0 else np.nan,
            'dmd_n_modes': float(len(eigenvalues)),
        }
    except Exception:
        return {
            'dmd_dominant_freq': np.nan,
            'dmd_growth_rate': np.nan,
            'dmd_is_stable': np.nan,
            'dmd_n_modes': np.nan,
        }
