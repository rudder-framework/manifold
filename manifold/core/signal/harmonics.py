"""
Harmonics Engine.

Delegates to pmtvs harmonic_analysis primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.spectral_features import harmonic_analysis


def compute(y: np.ndarray, sample_rate: float = 1.0, fundamental: float = None) -> Dict[str, float]:
    """
    Compute harmonic analysis of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz
        fundamental: Known fundamental frequency. If None, auto-detected.

    Returns:
        dict with fundamental_freq, fundamental_amplitude, harmonic_2x, harmonic_3x, thd
    """
    return harmonic_analysis(y, fs=sample_rate, fundamental=fundamental)
