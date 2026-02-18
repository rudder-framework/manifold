"""
Hilbert Stability Engine.

Delegates to pmtvs hilbert_stability primitive.
"""

import numpy as np
from manifold.primitives.individual.stability import hilbert_stability


def compute(y: np.ndarray, fs: float = 1.0) -> dict:
    """
    Compute Hilbert-derived stability metrics.

    Args:
        y: Signal values
        fs: Sampling frequency (default 1.0)

    Returns:
        dict with 11 stability metrics
    """
    return hilbert_stability(y)
