"""
Spectral Engine.

Delegates to pmtvs spectral primitives.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.spectral import (
    dominant_frequency,
    spectral_centroid,
    spectral_bandwidth,
    spectral_entropy,
)
from manifold.primitives.individual.spectral_features import spectral_slope


def compute(y: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral properties of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz (default: 1.0)

    Returns:
        dict with spectral_slope, dominant_freq, spectral_entropy,
        spectral_centroid, spectral_bandwidth
    """
    result = {
        'spectral_slope': np.nan,
        'dominant_freq': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 4:
        return result

    if np.std(y) < 1e-10:
        result['spectral_entropy'] = 0.0
        result['spectral_centroid'] = 0.0
        result['spectral_bandwidth'] = 0.0
        result['spectral_slope'] = 0.0
        result['dominant_freq'] = 0.0
        return result

    result['dominant_freq'] = dominant_frequency(y, fs=sample_rate)
    result['spectral_entropy'] = spectral_entropy(y, fs=sample_rate, normalize=True)
    result['spectral_centroid'] = spectral_centroid(y, fs=sample_rate)
    result['spectral_bandwidth'] = spectral_bandwidth(y, fs=sample_rate)
    result['spectral_slope'] = spectral_slope(y, fs=sample_rate)

    return result
