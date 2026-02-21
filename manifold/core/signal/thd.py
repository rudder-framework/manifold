"""
THD (Total Harmonic Distortion) Engine.

Delegates to pmtvs thd primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._compat import total_harmonic_distortion


def compute(y: np.ndarray, n_harmonics: int = 5) -> Dict[str, float]:
    """
    Compute total harmonic distortion.

    Args:
        y: Signal values
        n_harmonics: Number of harmonics to measure (default 5)

    Returns:
        dict with thd_percent, thd_db, thd_fundamental_power, thd_harmonic_power, n_harmonics_found
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 8:
        return {
            'thd_percent': np.nan,
            'thd_db': np.nan,
            'thd_fundamental_power': np.nan,
            'thd_harmonic_power': np.nan,
            'n_harmonics_found': 0,
        }

    try:
        thd_val = total_harmonic_distortion(y)
        thd_pct = float(thd_val) * 100 if not np.isnan(thd_val) else np.nan
        thd_db = 20 * np.log10(float(thd_val)) if thd_val > 0 else np.nan

        return {
            'thd_percent': thd_pct,
            'thd_db': thd_db,
            'thd_fundamental_power': np.nan,
            'thd_harmonic_power': np.nan,
            'n_harmonics_found': n_harmonics,
        }
    except Exception:
        return {
            'thd_percent': np.nan,
            'thd_db': np.nan,
            'thd_fundamental_power': np.nan,
            'thd_harmonic_power': np.nan,
            'n_harmonics_found': 0,
        }
