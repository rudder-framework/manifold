"""
Fundamental Frequency Engine.

Delegates to pmtvs fundamental_frequency primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._pmtvs import dominant_frequency, psd
# TODO: needs pmtvs export — fundamental_frequency (full analysis)


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Detect fundamental frequency.

    Args:
        y: Signal values

    Returns:
        dict with fundamental_freq, fundamental_power, fundamental_ratio, fundamental_confidence
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 8:
        return {
            'fundamental_freq': np.nan,
            'fundamental_power': np.nan,
            'fundamental_ratio': np.nan,
            'fundamental_confidence': np.nan,
        }

    try:
        freq = dominant_frequency(y)
        freqs, power = psd(y)
        total = float(np.sum(power))
        if total > 0 and not np.isnan(freq):
            # Find power at dominant frequency
            idx = np.argmin(np.abs(freqs - freq))
            fund_power = float(power[idx])
            ratio = fund_power / total
        else:
            fund_power = np.nan
            ratio = np.nan

        return {
            'fundamental_freq': float(freq),
            'fundamental_power': fund_power,
            'fundamental_ratio': ratio,
            'fundamental_confidence': np.nan,
        }
    except Exception:
        return {
            'fundamental_freq': np.nan,
            'fundamental_power': np.nan,
            'fundamental_ratio': np.nan,
            'fundamental_confidence': np.nan,
        }
