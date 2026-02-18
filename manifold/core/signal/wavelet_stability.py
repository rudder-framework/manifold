"""
Wavelet Stability Engine.

Delegates to pmtvs wavelet_stability primitive.
"""

import numpy as np
from manifold.primitives.individual.stability import wavelet_stability


def compute(y: np.ndarray, fs: float = 1.0, n_scales: int = 16) -> dict:
    """
    Compute wavelet-derived stability metrics.

    Args:
        y: Signal values
        fs: Sampling frequency
        n_scales: Number of wavelet scales to analyze

    Returns:
        dict with 10 wavelet stability metrics
    """
    r = wavelet_stability(y)

    # Map unprefixed pmtvs keys to wavelet_-prefixed keys expected by stages
    return {
        'wavelet_energy_low': r.get('energy_low', np.nan),
        'wavelet_energy_mid': r.get('energy_mid', np.nan),
        'wavelet_energy_high': r.get('energy_high', np.nan),
        'wavelet_energy_ratio': r.get('energy_ratio', np.nan),
        'wavelet_entropy': r.get('entropy', np.nan),
        'wavelet_concentration': r.get('concentration', np.nan),
        'wavelet_dominant_scale': r.get('dominant_scale', np.nan),
        'wavelet_energy_drift': r.get('energy_drift', np.nan),
        'wavelet_temporal_std': r.get('temporal_std', np.nan),
        'wavelet_intermittency': r.get('intermittency', np.nan),
    }
