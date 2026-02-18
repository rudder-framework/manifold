"""
DMD (Dynamic Mode Decomposition) Engine.

Delegates to pmtvs dmd_analysis primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.dmd_features import dmd_analysis


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
    return dmd_analysis(y, rank=rank, dt=dt)
