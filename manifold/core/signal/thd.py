"""
THD (Total Harmonic Distortion) Engine.

Delegates to pmtvs thd primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.spectral_features import thd as _thd


def compute(y: np.ndarray, n_harmonics: int = 5) -> Dict[str, float]:
    """
    Compute total harmonic distortion.

    Args:
        y: Signal values
        n_harmonics: Number of harmonics to measure (default 5)

    Returns:
        dict with thd_percent, thd_db, thd_fundamental_power, thd_harmonic_power, n_harmonics_found
    """
    r = _thd(y, n_harmonics=n_harmonics)

    return {
        'thd_percent': r.get('thd_percent', np.nan),
        'thd_db': r.get('thd_db', np.nan),
        'thd_fundamental_power': r.get('fundamental_power', np.nan),
        'thd_harmonic_power': r.get('harmonic_power', np.nan),
        'n_harmonics_found': r.get('n_harmonics_found', 0),
    }
