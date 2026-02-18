"""
Fundamental Frequency Engine.

Delegates to pmtvs fundamental_frequency primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.spectral_features import fundamental_frequency


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Detect fundamental frequency.

    Args:
        y: Signal values

    Returns:
        dict with fundamental_freq, fundamental_power, fundamental_ratio, fundamental_confidence
    """
    return fundamental_frequency(y)
