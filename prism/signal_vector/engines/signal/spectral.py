"""
Spectral Engines
=================

Frequency domain analysis per window.

Engines:
- spectral: Dominant frequency, spectral centroid, spectral spread
- spectral_entropy: Flatness of power distribution
- band_power: Power in frequency bands
- frequency_bands: Detailed band decomposition
"""

import numpy as np
from typing import Dict, Any, List


def compute_spectral(values: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral features.
    
    Returns:
    - dominant_freq: Frequency with maximum power
    - spectral_centroid: Center of mass of spectrum
    - spectral_spread: Spread around centroid
    - spectral_rolloff: Frequency below which 85% power lies
    """
    if len(values) < 4:
        return {
            'dominant_freq': np.nan,
            'spectral_centroid': np.nan,
            'spectral_spread': np.nan,
            'spectral_rolloff': np.nan,
        }
    
    # FFT
    n = len(values)
    fft = np.fft.rfft(values - np.mean(values))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    
    # Normalize power
    total_power = np.sum(power)
    if total_power < 1e-10:
        return {
            'dominant_freq': 0.0,
            'spectral_centroid': 0.0,
            'spectral_spread': 0.0,
            'spectral_rolloff': 0.0,
        }
    
    power_norm = power / total_power
    
    # Dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = freqs[dominant_idx]
    
    # Spectral centroid (weighted mean frequency)
    spectral_centroid = np.sum(freqs * power_norm)
    
    # Spectral spread (weighted std)
    spectral_spread = np.sqrt(np.sum(power_norm * (freqs - spectral_centroid) ** 2))
    
    # Spectral rolloff (85% cumulative power)
    cumsum = np.cumsum(power_norm)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    
    return {
        'dominant_freq': float(dominant_freq),
        'spectral_centroid': float(spectral_centroid),
        'spectral_spread': float(spectral_spread),
        'spectral_rolloff': float(spectral_rolloff),
    }


def compute_spectral_entropy(values: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral entropy (flatness of spectrum).
    
    High entropy: Broadband/white noise
    Low entropy: Narrowband/periodic
    """
    if len(values) < 4:
        return {'spectral_entropy': np.nan}
    
    # FFT
    fft = np.fft.rfft(values - np.mean(values))
    power = np.abs(fft) ** 2
    
    # Normalize to probability distribution
    total = np.sum(power)
    if total < 1e-10:
        return {'spectral_entropy': 0.0}
    
    p = power / total
    p = p[p > 1e-10]  # Remove zeros for log
    
    # Shannon entropy
    entropy = -np.sum(p * np.log2(p))
    
    # Normalize by max possible entropy
    max_entropy = np.log2(len(power))
    if max_entropy > 0:
        entropy = entropy / max_entropy
    
    return {'spectral_entropy': float(entropy)}


def compute_band_power(
    values: np.ndarray,
    bands: List[float] = None,
    sample_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute power in frequency bands.
    
    Default bands (normalized to Nyquist):
    [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    """
    if bands is None:
        bands = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    
    if len(values) < 4:
        return {f'band_{i}': np.nan for i in range(len(bands))}
    
    # FFT
    n = len(values)
    fft = np.fft.rfft(values - np.mean(values))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    
    nyquist = sample_rate / 2
    total_power = np.sum(power)
    
    if total_power < 1e-10:
        return {f'band_{i}': 0.0 for i in range(len(bands))}
    
    results = {}
    prev_freq = 0.0
    
    for i, band_frac in enumerate(bands):
        band_freq = band_frac * nyquist
        mask = (freqs >= prev_freq) & (freqs < band_freq)
        band_power = np.sum(power[mask]) / total_power
        results[f'band_{i}'] = float(band_power)
        prev_freq = band_freq
    
    # Final band: from last cutoff to Nyquist
    mask = freqs >= prev_freq
    results[f'band_{len(bands)}'] = float(np.sum(power[mask]) / total_power)
    
    return results


def compute_frequency_bands(
    values: np.ndarray,
    sample_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute detailed frequency band features.
    
    Returns power ratios in standard bands.
    """
    return compute_band_power(values, sample_rate=sample_rate)


# Engine registry
ENGINES = {
    'spectral': compute_spectral,
    'spectral_entropy': compute_spectral_entropy,
    'band_power': compute_band_power,
    'frequency_bands': compute_frequency_bands,
}
