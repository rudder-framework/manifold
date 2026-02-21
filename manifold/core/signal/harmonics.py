"""
Harmonics Engine.

Delegates to pmtvs harmonic_analysis primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._pmtvs import dominant_frequency, psd
from manifold.core._compat import harmonic_ratio, total_harmonic_distortion
# TODO: needs pmtvs export — harmonic_analysis (full analysis)


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
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 8:
        return {
            'fundamental_freq': np.nan,
            'fundamental_amplitude': np.nan,
            'harmonic_2x': np.nan,
            'harmonic_3x': np.nan,
            'thd': np.nan,
        }

    try:
        if fundamental is None:
            fundamental = dominant_frequency(y, fs=sample_rate)

        freqs, power = psd(y, fs=sample_rate)
        thd_val = total_harmonic_distortion(y)

        # Find amplitudes at fundamental and harmonics
        if not np.isnan(fundamental) and fundamental > 0:
            idx_f = np.argmin(np.abs(freqs - fundamental))
            idx_2 = np.argmin(np.abs(freqs - 2 * fundamental))
            idx_3 = np.argmin(np.abs(freqs - 3 * fundamental))
            fund_amp = float(np.sqrt(power[idx_f]))
            h2 = float(np.sqrt(power[idx_2]))
            h3 = float(np.sqrt(power[idx_3]))
        else:
            fund_amp = np.nan
            h2 = np.nan
            h3 = np.nan

        return {
            'fundamental_freq': float(fundamental),
            'fundamental_amplitude': fund_amp,
            'harmonic_2x': h2,
            'harmonic_3x': h3,
            'thd': float(thd_val),
        }
    except Exception:
        return {
            'fundamental_freq': np.nan,
            'fundamental_amplitude': np.nan,
            'harmonic_2x': np.nan,
            'harmonic_3x': np.nan,
            'thd': np.nan,
        }
