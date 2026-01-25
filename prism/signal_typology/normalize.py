"""
Signal Typology Normalization
=============================

Converts raw engine metrics to normalized 0-1 axis scores.

Each axis has a dedicated normalization function that maps
engine outputs to a 0-1 range with consistent semantic meaning:
    0 = low end of axis (forgetful, predictable, stable, etc.)
    1 = high end of axis (persistent, entropic, clustered, etc.)

Principle: Data = math. Labels = rendering.
- Store normalized 0-1 scores
- Classification computed at query/display time
- Threshold changes don't require recomputation
"""

import numpy as np
from typing import Dict, Any


def normalize_memory(metrics: Dict[str, Any]) -> float:
    """
    Memory axis: Forgetful (0) → Persistent (1)

    Based on Hurst exponent which is naturally 0-1:
        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        H > 0.5: Persistent (trending)
    """
    h = metrics.get('hurst_exponent', 0.5)
    return float(np.clip(h, 0, 1))


def normalize_information(metrics: Dict[str, Any]) -> float:
    """
    Information axis: Predictable (0) → Entropic (1)

    Combines entropy measures:
        - Permutation entropy: ordinal pattern randomness (0-1)
        - Sample entropy: signal regularity (0 = regular, high = complex)
    """
    pe = metrics.get('permutation_entropy', 0.5)
    se = metrics.get('sample_entropy', 1.0)

    # Permutation entropy is typically 0-1
    pe_norm = float(np.clip(pe, 0, 1))

    # Sample entropy: cap at 3 and scale to 0-1
    se_norm = float(np.clip(se / 3.0, 0, 1))

    # Weighted combination
    return 0.6 * pe_norm + 0.4 * se_norm


def normalize_frequency(metrics: Dict[str, Any]) -> float:
    """
    Frequency axis: Aperiodic (0) → Periodic (1)

    Inverse of spectral entropy:
        - High spectral entropy = uniform frequency distribution = aperiodic
        - Low spectral entropy = concentrated frequency = periodic

    spectral_entropy is Shannon entropy, ranges 0 to ~5 (log of freq bins)
    We scale by max expected value ~5
    """
    se = metrics.get('spectral_entropy', 2.5)  # Default to middle
    # Shannon entropy ranges 0 to ~5, normalize to 0-1
    se_norm = float(np.clip(se / 5.0, 0, 1))
    # Invert: high entropy = aperiodic (0), low entropy = periodic (1)
    return 1.0 - se_norm


def normalize_volatility(metrics: Dict[str, Any]) -> float:
    """
    Volatility axis: Stable (0) → Clustered (1)

    Based on GARCH persistence (alpha + beta):
        - Low persistence: volatility shocks die quickly
        - High persistence: volatility clusters
    """
    persistence = metrics.get('garch_persistence', 0.5)
    return float(np.clip(persistence, 0, 1))


def normalize_wavelet(metrics: Dict[str, Any]) -> float:
    """
    Wavelet axis: Single-scale (0) → Multi-scale (1)

    Based on wavelet entropy (energy distribution across scales):
        - Low entropy: energy concentrated in few scales
        - High entropy: energy spread across many scales

    scale_entropy is Shannon entropy of 6 scales, max ≈ log(6) ≈ 1.79
    """
    we = metrics.get('wavelet_entropy', 0.9)  # Default to middle
    # scale_entropy ranges 0 to ~1.79, normalize to 0-1
    we_norm = we / 1.8
    return float(np.clip(we_norm, 0, 1))


def normalize_derivatives(metrics: Dict[str, Any]) -> float:
    """
    Derivatives axis: Smooth (0) → Spiky (1)

    Based on derivative kurtosis:
        - Normal distribution: kurtosis = 3
        - Heavy tails (spikes): kurtosis > 3

    Use log scale since kurtosis can be very high (>1000):
        - k=3 → 0 (smooth)
        - k=10 → 0.33
        - k=100 → 0.67
        - k=1000 → 1.0 (spiky)
    """
    k = metrics.get('derivative_kurtosis', 3.0)
    if k <= 3:
        return 0.0
    # Log scale: log10(k-2) maps 3→0.0, 10→0.33, 100→0.67, 1000→1.0
    score = np.log10(k - 2) / 3.0  # divide by 3 since log10(1000)=3
    return float(np.clip(score, 0, 1))


def normalize_recurrence(metrics: Dict[str, Any]) -> float:
    """
    Recurrence axis: Wandering (0) → Returning (1)

    Combines RQA metrics:
        - Recurrence rate: % of recurrent points
        - Determinism: % of points in diagonal lines
    """
    rr = metrics.get('recurrence_rate', 0.0)
    det = metrics.get('determinism', 0.0)

    rr_norm = float(np.clip(rr, 0, 1))
    det_norm = float(np.clip(det, 0, 1))

    return 0.5 * rr_norm + 0.5 * det_norm


def normalize_discontinuity(metrics: Dict[str, Any]) -> float:
    """
    Discontinuity axis: Continuous (0) → Step-like (1)

    Based on level shift detection:
        - Normalized by signal length
        - More shifts = more step-like
    """
    shifts = metrics.get('level_shift_count', 0)
    n = metrics.get('n_samples', 1000)

    # Expect ~1 shift per 500 samples as "high"
    expected_max = n / 500
    score = shifts / max(expected_max, 1)

    return float(np.clip(score, 0, 1))


def normalize_momentum(metrics: Dict[str, Any]) -> float:
    """
    Momentum axis: Reverting (0) → Trending (1)

    Reinterpretation of Hurst exponent:
        H < 0.5: Mean-reverting
        H = 0.5: Random
        H > 0.5: Trending

    Note: Same source as memory but different semantic framing.
    """
    h = metrics.get('hurst_exponent', 0.5)
    return float(np.clip(h, 0, 1))


# All normalization functions by axis name
NORMALIZERS = {
    'memory': normalize_memory,
    'information': normalize_information,
    'frequency': normalize_frequency,
    'volatility': normalize_volatility,
    'wavelet': normalize_wavelet,
    'derivatives': normalize_derivatives,
    'recurrence': normalize_recurrence,
    'discontinuity': normalize_discontinuity,
    'momentum': normalize_momentum,
}

AXIS_NAMES = list(NORMALIZERS.keys())


def normalize_all(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize all axes from metrics.

    Args:
        metrics: Dict with raw engine outputs

    Returns:
        Dict with normalized 0-1 scores for each axis
    """
    profile = {}
    for axis, normalizer in NORMALIZERS.items():
        try:
            profile[axis] = normalizer(metrics)
        except Exception:
            profile[axis] = 0.5  # Default to indeterminate on error
    return profile


def metrics_to_profile(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert metrics dict to profile dict.

    Preserves identification fields (signal_id, timestamp, n_samples)
    and adds normalized axis scores.

    Args:
        metrics: Full metrics dict with engine outputs

    Returns:
        Profile dict with identification + normalized scores
    """
    profile = {
        'signal_id': metrics.get('signal_id', 'unknown'),
        'timestamp': metrics.get('timestamp'),
        'n_samples': metrics.get('n_samples', 0),
    }

    # Add normalized axes
    profile.update(normalize_all(metrics))

    return profile
