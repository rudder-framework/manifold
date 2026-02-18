"""
Phase Coherence Engine.

Delegates to pmtvs phase_coherence primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.trend_features import phase_coherence as _phase_coherence


def compute(y: np.ndarray, n_segments: int = 4) -> Dict[str, float]:
    """
    Measure phase coherence.

    Args:
        y: Signal values
        n_segments: Number of segments to compare

    Returns:
        dict with phase_coherence, coherence_std, coherence_trend
    """
    return _phase_coherence(y)
