"""
SNR (Signal-to-Noise Ratio) Engine.

Delegates to pmtvs snr primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._pmtvs import psd
# TODO: needs pmtvs export — snr


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

    try:
        # Simple SNR: signal power = variance of smoothed, noise = variance of residual
        freqs, power = psd(y)
        total_power = float(np.sum(power))
        # Top 10% of spectral power = signal, rest = noise
        sorted_power = np.sort(power)[::-1]
        n_signal = max(1, len(sorted_power) // 10)
        signal_power = float(np.sum(sorted_power[:n_signal]))
        noise_power = max(total_power - signal_power, 1e-12)
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else np.nan

        result['snr_db'] = snr_db
        result['snr_linear'] = snr_linear
        result['signal_power'] = signal_power
        result['noise_power'] = noise_power
    except Exception:
        pass

    return result
