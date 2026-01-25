"""
Signal Typology Orchestrator
============================

One of four ORTHON analytical frameworks: What IS this signal?

Produces two outputs:
    1. signal_typology_metrics.parquet - Raw engine measurements
    2. signal_typology_profile.parquet - Normalized 0-1 axis scores

The Nine Axes:
    1. Memory        - Forgetful (0) → Persistent (1)
    2. Information   - Predictable (0) → Entropic (1)
    3. Frequency     - Aperiodic (0) → Periodic (1)
    4. Volatility    - Stable (0) → Clustered (1)
    5. Wavelet       - Single-scale (0) → Multi-scale (1)
    6. Derivatives   - Smooth (0) → Spiky (1)
    7. Recurrence    - Wandering (0) → Returning (1)
    8. Discontinuity - Continuous (0) → Step-like (1)
    9. Momentum      - Reverting (0) → Trending (1)

Principle: Data = math. Labels = rendering.
- Store normalized 0-1 scores
- Classification computed at query/display time
- Threshold changes don't require recomputation

Architecture:
    observations.parquet
        ↓ (engines)
    signal_typology_metrics.parquet  (raw measurements)
        ↓ (normalize.py)
    signal_typology_profile.parquet  (0-1 scores)
        ↓ (classify.py at runtime)
    UI displays labels
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .normalize import normalize_all, metrics_to_profile, AXIS_NAMES
from .classify import classify_profile, summarize_profile


def compute_metrics(signal: np.ndarray, signal_id: str = "unknown") -> Dict[str, Any]:
    """
    Compute all raw metrics from a signal.

    This calls individual engines and assembles the metrics dict.

    Args:
        signal: 1D numpy array of signal values
        signal_id: Identifier for the signal

    Returns:
        Dict with all raw metrics (32+ columns)
    """
    signal = np.asarray(signal, dtype=float).flatten()
    n_samples = len(signal)

    metrics = {
        'signal_id': signal_id,
        'timestamp': datetime.now(),
        'n_samples': n_samples,
    }

    if n_samples < 30:
        # Return empty metrics for insufficient data
        return _empty_metrics(signal_id, n_samples)

    # === MEMORY ENGINES ===
    try:
        from ..engines.memory.hurst_rs import compute as hurst_rs
        h_result = hurst_rs(signal)
        # hurst_rs returns dict with 'hurst_exponent' key
        if isinstance(h_result, dict):
            metrics['hurst_exponent'] = h_result.get('hurst_exponent', h_result.get('H', 0.5))
        else:
            metrics['hurst_exponent'] = getattr(h_result, 'hurst_exponent', getattr(h_result, 'H', 0.5))
    except Exception:
        metrics['hurst_exponent'] = 0.5

    try:
        from ..engines.memory.acf_decay import compute as acf_decay
        acf_result = acf_decay(signal)
        if isinstance(acf_result, dict):
            metrics['acf_decay_rate'] = acf_result.get('decay_rate', 0.0)
            metrics['acf_lag1'] = acf_result.get('lag1', 0.0)
        else:
            metrics['acf_decay_rate'] = getattr(acf_result, 'decay_rate', 0.0)
            metrics['acf_lag1'] = getattr(acf_result, 'lag1', 0.0)
    except Exception:
        metrics['acf_decay_rate'] = 0.0
        metrics['acf_lag1'] = 0.0

    # === INFORMATION ENGINES ===
    try:
        from ..engines.information.permutation_entropy import compute as perm_entropy
        pe_result = perm_entropy(signal)
        metrics['permutation_entropy'] = pe_result.get('entropy', 0.5) if isinstance(pe_result, dict) else getattr(pe_result, 'entropy', 0.5)
    except Exception:
        metrics['permutation_entropy'] = 0.5

    try:
        from ..engines.information.sample_entropy import compute as samp_entropy
        se_result = samp_entropy(signal)
        metrics['sample_entropy'] = se_result.get('entropy', 1.0) if isinstance(se_result, dict) else getattr(se_result, 'entropy', 1.0)
    except Exception:
        metrics['sample_entropy'] = 1.0

    # === FREQUENCY ENGINES ===
    try:
        # Use windowed/spectral which has spectral_entropy
        from ..engines.windowed.spectral import compute_spectral
        spec_result = compute_spectral(signal)
        if isinstance(spec_result, dict):
            metrics['spectral_entropy'] = spec_result.get('spectral_entropy', 0.5)
            metrics['dominant_frequency'] = spec_result.get('dominant_freq', spec_result.get('dominant_frequency', 0.0))
            metrics['fft_peak_power'] = spec_result.get('total_power', spec_result.get('peak_power', 0.0))
            metrics['spectral_centroid'] = spec_result.get('spectral_centroid', 0.0)
            metrics['spectral_flatness'] = spec_result.get('spectral_flatness', 0.0)
        else:
            metrics['spectral_entropy'] = getattr(spec_result, 'spectral_entropy', 0.5)
            metrics['dominant_frequency'] = getattr(spec_result, 'dominant_freq', 0.0)
            metrics['fft_peak_power'] = getattr(spec_result, 'total_power', 0.0)
            metrics['spectral_centroid'] = getattr(spec_result, 'spectral_centroid', 0.0)
            metrics['spectral_flatness'] = getattr(spec_result, 'spectral_flatness', 0.0)
    except Exception:
        metrics['spectral_entropy'] = 0.5
        metrics['dominant_frequency'] = 0.0
        metrics['fft_peak_power'] = 0.0
        metrics['spectral_centroid'] = 0.0
        metrics['spectral_flatness'] = 0.0

    # === VOLATILITY ENGINES ===
    try:
        from ..engines.volatility.garch import compute as garch
        garch_result = garch(signal)
        if isinstance(garch_result, dict):
            metrics['garch_alpha'] = garch_result.get('alpha', 0.0)
            metrics['garch_beta'] = garch_result.get('beta', 0.0)
            metrics['garch_persistence'] = garch_result.get('persistence', 0.5)
        else:
            metrics['garch_alpha'] = getattr(garch_result, 'alpha', 0.0)
            metrics['garch_beta'] = getattr(garch_result, 'beta', 0.0)
            metrics['garch_persistence'] = getattr(garch_result, 'persistence', 0.5)
    except Exception:
        metrics['garch_alpha'] = 0.0
        metrics['garch_beta'] = 0.0
        metrics['garch_persistence'] = 0.5

    # Rolling variance ratio (via engine)
    try:
        from ..engines.typology.rolling_volatility import compute as rolling_vol
        rv_result = rolling_vol(signal)
        metrics['rolling_std_ratio'] = rv_result.get('rolling_std_ratio', 1.0)
    except Exception:
        metrics['rolling_std_ratio'] = 1.0

    # === WAVELET ENGINES ===
    try:
        from ..engines.frequency.wavelet import compute as wavelet
        wav_result = wavelet(signal)
        if isinstance(wav_result, dict):
            # wavelet returns 'scale_entropy' not 'entropy', and energy_by_scale array
            energy_by_scale = wav_result.get('energy_by_scale', [0.33, 0.33, 0.33])
            if isinstance(energy_by_scale, (list, np.ndarray)) and len(energy_by_scale) >= 3:
                metrics['wavelet_energy_low'] = float(energy_by_scale[0])
                metrics['wavelet_energy_mid'] = float(energy_by_scale[1])
                metrics['wavelet_energy_high'] = float(sum(energy_by_scale[2:]))
            else:
                metrics['wavelet_energy_low'] = wav_result.get('energy_low', 0.33)
                metrics['wavelet_energy_mid'] = wav_result.get('energy_mid', 0.33)
                metrics['wavelet_energy_high'] = wav_result.get('energy_high', 0.33)
            metrics['wavelet_entropy'] = wav_result.get('scale_entropy', wav_result.get('entropy', 0.5))
        else:
            energy_by_scale = getattr(wav_result, 'energy_by_scale', [0.33, 0.33, 0.33])
            if isinstance(energy_by_scale, (list, np.ndarray)) and len(energy_by_scale) >= 3:
                metrics['wavelet_energy_low'] = float(energy_by_scale[0])
                metrics['wavelet_energy_mid'] = float(energy_by_scale[1])
                metrics['wavelet_energy_high'] = float(sum(energy_by_scale[2:]))
            else:
                metrics['wavelet_energy_low'] = 0.33
                metrics['wavelet_energy_mid'] = 0.33
                metrics['wavelet_energy_high'] = 0.33
            metrics['wavelet_entropy'] = getattr(wav_result, 'scale_entropy', getattr(wav_result, 'entropy', 0.5))
    except Exception:
        metrics['wavelet_energy_low'] = 0.33
        metrics['wavelet_energy_mid'] = 0.33
        metrics['wavelet_energy_high'] = 0.33
        metrics['wavelet_entropy'] = 0.5

    # === DERIVATIVE STATS (via engine) ===
    try:
        from ..engines.typology.derivative_stats import compute as derivative_stats
        deriv_result = derivative_stats(signal)
        metrics['derivative_mean'] = deriv_result.get('derivative_mean', 0.0)
        metrics['derivative_std'] = deriv_result.get('derivative_std', 0.0)
        metrics['derivative_kurtosis'] = deriv_result.get('derivative_kurtosis', 3.0)
        metrics['zero_crossing_rate'] = deriv_result.get('zero_crossing_rate', 0.0)
    except Exception:
        metrics['derivative_mean'] = 0.0
        metrics['derivative_std'] = 0.0
        metrics['derivative_kurtosis'] = 3.0
        metrics['zero_crossing_rate'] = 0.0

    # === RECURRENCE (RQA) ===
    try:
        from ..engines.recurrence.rqa import compute as rqa
        rqa_result = rqa(signal)
        if isinstance(rqa_result, dict):
            metrics['recurrence_rate'] = rqa_result.get('recurrence_rate', 0.0)
            metrics['determinism'] = rqa_result.get('determinism', 0.0)
            metrics['laminarity'] = rqa_result.get('laminarity', 0.0)
            metrics['trapping_time'] = rqa_result.get('trapping_time', 0.0)
        else:
            metrics['recurrence_rate'] = getattr(rqa_result, 'recurrence_rate', 0.0)
            metrics['determinism'] = getattr(rqa_result, 'determinism', 0.0)
            metrics['laminarity'] = getattr(rqa_result, 'laminarity', 0.0)
            metrics['trapping_time'] = getattr(rqa_result, 'trapping_time', 0.0)
    except Exception:
        metrics['recurrence_rate'] = 0.0
        metrics['determinism'] = 0.0
        metrics['laminarity'] = 0.0
        metrics['trapping_time'] = 0.0

    # === DISCONTINUITY (CUSUM / Level Shifts via engine) ===
    try:
        from ..engines.typology.cusum import compute as cusum_engine
        cusum_result = cusum_engine(signal)
        metrics['cusum_max'] = cusum_result.get('cusum_max', 0.0)
        metrics['cusum_crossings'] = cusum_result.get('cusum_crossings', 0)
        metrics['level_shift_count'] = cusum_result.get('level_shift_count', 0)
        metrics['level_shift_magnitude_mean'] = cusum_result.get('level_shift_magnitude_mean', 0.0)
    except Exception:
        metrics['cusum_max'] = 0.0
        metrics['cusum_crossings'] = 0
        metrics['level_shift_count'] = 0
        metrics['level_shift_magnitude_mean'] = 0.0

    # === RUNS TEST (Momentum via engine) ===
    try:
        from ..engines.momentum.runs_test import compute as runs_test_engine
        runs_result = runs_test_engine(signal, mode='static')
        metrics['runs_test_z'] = runs_result.get('z_score', 0.0)
        metrics['runs_ratio'] = runs_result.get('runs_ratio', 1.0)
    except Exception:
        metrics['runs_test_z'] = 0.0
        metrics['runs_ratio'] = 1.0

    return metrics


def _empty_metrics(signal_id: str, n_samples: int) -> Dict[str, Any]:
    """Return empty metrics dict for insufficient data."""
    return {
        'signal_id': signal_id,
        'timestamp': datetime.now(),
        'n_samples': n_samples,
        'hurst_exponent': np.nan,
        'acf_decay_rate': np.nan,
        'acf_lag1': np.nan,
        'permutation_entropy': np.nan,
        'sample_entropy': np.nan,
        'spectral_entropy': np.nan,
        'dominant_frequency': np.nan,
        'fft_peak_power': np.nan,
        'garch_alpha': np.nan,
        'garch_beta': np.nan,
        'garch_persistence': np.nan,
        'rolling_std_ratio': np.nan,
        'wavelet_energy_low': np.nan,
        'wavelet_energy_mid': np.nan,
        'wavelet_energy_high': np.nan,
        'wavelet_entropy': np.nan,
        'derivative_mean': np.nan,
        'derivative_std': np.nan,
        'derivative_kurtosis': np.nan,
        'zero_crossing_rate': np.nan,
        'recurrence_rate': np.nan,
        'determinism': np.nan,
        'laminarity': np.nan,
        'trapping_time': np.nan,
        'cusum_max': np.nan,
        'cusum_crossings': np.nan,
        'level_shift_count': np.nan,
        'level_shift_magnitude_mean': np.nan,
        'runs_test_z': np.nan,
        'runs_ratio': np.nan,
    }


def run_signal_typology(
    signals: Dict[str, np.ndarray],
    entity_id: str = "unknown",
    timestamp: Optional[datetime] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run Signal Typology on multiple signals.

    Args:
        signals: Dict of {signal_id: np.array}
        entity_id: Entity identifier
        timestamp: Optional timestamp for all signals

    Returns:
        Tuple of (metrics_rows, profile_rows)
    """
    timestamp = timestamp or datetime.now()

    metrics_rows = []
    profile_rows = []

    for signal_id, signal in signals.items():
        # Compute raw metrics
        metrics = compute_metrics(signal, signal_id)
        metrics['entity_id'] = entity_id
        metrics['timestamp'] = timestamp
        metrics_rows.append(metrics)

        # Normalize to profile
        profile = metrics_to_profile(metrics)
        profile['entity_id'] = entity_id
        profile['timestamp'] = timestamp
        profile_rows.append(profile)

    return metrics_rows, profile_rows


def analyze_single(
    signal: np.ndarray,
    signal_id: str = "signal",
    entity_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze a single signal and return full results.

    Args:
        signal: 1D numpy array
        signal_id: Signal identifier
        entity_id: Entity identifier

    Returns:
        Dict with metrics, profile, and classification
    """
    metrics = compute_metrics(signal, signal_id)
    metrics['entity_id'] = entity_id

    profile = metrics_to_profile(metrics)
    profile['entity_id'] = entity_id

    classification = classify_profile(profile)
    summary = summarize_profile(profile)

    return {
        'metrics': metrics,
        'profile': profile,
        'classification': classification,
        'summary': summary,
    }


def get_fingerprint(profile: Dict[str, float]) -> np.ndarray:
    """
    Convert profile to a fingerprint vector.

    Args:
        profile: Dict with axis scores

    Returns:
        numpy array of scores in canonical order
    """
    return np.array([profile.get(ax, 0.5) for ax in AXIS_NAMES])


def fingerprint_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, 3 = maximally different for 9 axes)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_regime_change(
    previous_profile: Dict[str, float],
    current_profile: Dict[str, float],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect regime change between two profiles.

    Args:
        previous_profile: Profile from previous window
        current_profile: Profile from current window
        threshold: Change threshold for flagging

    Returns:
        Dict with change detection results
    """
    fp_prev = get_fingerprint(previous_profile)
    fp_curr = get_fingerprint(current_profile)

    distance = fingerprint_distance(fp_prev, fp_curr)

    changes = {}
    moving_axes = []
    stable_axes = []

    for ax in AXIS_NAMES:
        prev_val = previous_profile.get(ax, 0.5)
        curr_val = current_profile.get(ax, 0.5)
        delta = curr_val - prev_val

        changes[ax] = {
            'previous': prev_val,
            'current': curr_val,
            'delta': delta,
            'abs_delta': abs(delta),
        }

        if abs(delta) >= threshold:
            moving_axes.append(ax)
        else:
            stable_axes.append(ax)

    return {
        'regime_changed': distance >= threshold,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'moving_axes': moving_axes,
        'stable_axes': stable_axes,
    }


# Backwards compatibility
AXIS_NAMES_OLD = ['memory', 'periodicity', 'volatility', 'discontinuity', 'impulsivity', 'complexity']
