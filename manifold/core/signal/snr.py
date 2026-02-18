"""
SNR (Signal-to-Noise Ratio) Engine.

Delegates to pmtvs snr primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.spectral_features import snr as _snr


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute signal-to-noise ratio.

    Args:
        y: Signal values

    Returns:
        dict with snr_db, snr_linear, signal_power, noise_power
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    result = {
        'snr_db': np.nan,
        'snr_linear': np.nan,
        'signal_power': np.nan,
        'noise_power': np.nan,
    }

    if len(y) < 8:
        return result

    r = _snr(y)
    result['snr_db'] = r.get('snr_db', np.nan)
    result['snr_linear'] = r.get('snr', np.nan)

    return result
