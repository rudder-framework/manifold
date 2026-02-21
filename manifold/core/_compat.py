"""Functions not yet in published pmtvs 0.1.4 — inlined until pmtvs-core ships."""

import numpy as np

# graph_laplacian: dev pmtvs has it at top level, published 0.1.4 has laplacian_matrix
try:
    from pmtvs import graph_laplacian  # noqa: F401  — dev 0.3.x
except ImportError:
    from pmtvs.matrix.graph import laplacian_matrix as graph_laplacian  # noqa: F401  — published 0.1.4


# ---------------------------------------------------------------------------
# wavelet_stability  (pmtvs_dynamics.stability — added after 0.1.4)
# ---------------------------------------------------------------------------
def wavelet_stability(y: np.ndarray) -> dict:
    """Analyze wavelet energy distribution and stability."""
    nan_result = {
        "energy_low": np.nan, "energy_mid": np.nan, "energy_high": np.nan,
        "energy_ratio": np.nan, "entropy": np.nan, "concentration": np.nan,
        "dominant_scale": np.nan, "energy_drift": np.nan,
        "temporal_std": np.nan, "intermittency": np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return nan_result

    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2
    n_bins = len(power)
    if n_bins < 3:
        return nan_result

    b1 = n_bins // 3
    b2 = 2 * n_bins // 3
    energy_low_val = float(np.sum(power[:b1]))
    energy_mid_val = float(np.sum(power[b1:b2]))
    energy_high_val = float(np.sum(power[b2:]))

    total_energy = energy_low_val + energy_mid_val + energy_high_val
    if total_energy < 1e-12:
        return nan_result

    e_low = float(energy_low_val / total_energy)
    e_mid = float(energy_mid_val / total_energy)
    e_high = float(energy_high_val / total_energy)
    e_ratio = float(energy_low_val / energy_high_val) if energy_high_val > 1e-12 else np.nan

    energies = np.array([e_low, e_mid, e_high])
    nonzero = energies[energies > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    concentration = float(np.max(energies) / np.sum(energies))
    dominant_scale = int(np.argmax(energies))

    window_size = max(len(y) // 10, 2)
    n_windows = len(y) - window_size + 1
    if n_windows < 2:
        return {
            "energy_low": e_low, "energy_mid": e_mid, "energy_high": e_high,
            "energy_ratio": e_ratio, "entropy": entropy,
            "concentration": concentration, "dominant_scale": dominant_scale,
            "energy_drift": 0.0, "temporal_std": 0.0, "intermittency": 0.0,
        }

    rolling_energy = np.array(
        [np.sum(y[i : i + window_size] ** 2) for i in range(n_windows)]
    )
    t = np.arange(n_windows, dtype=np.float64)
    coeffs = np.polyfit(t, rolling_energy, 1)
    energy_drift = float(coeffs[0])
    temporal_std = float(np.std(rolling_energy))
    re_mean = float(np.mean(rolling_energy))
    intermittency = (
        float(np.mean(((rolling_energy - re_mean) / temporal_std) ** 4) - 3.0)
        if temporal_std > 1e-12
        else 0.0
    )

    return {
        "energy_low": e_low, "energy_mid": e_mid, "energy_high": e_high,
        "energy_ratio": e_ratio, "entropy": entropy,
        "concentration": concentration, "dominant_scale": dominant_scale,
        "energy_drift": energy_drift, "temporal_std": temporal_std,
        "intermittency": intermittency,
    }


# ---------------------------------------------------------------------------
# hilbert_stability  (pmtvs_dynamics.stability — added after 0.1.4)
# ---------------------------------------------------------------------------
def hilbert_stability(y: np.ndarray) -> dict:
    """Analyze instantaneous frequency and amplitude stability via Hilbert transform."""
    from scipy.signal import hilbert

    nan_result = {
        "inst_freq_mean": np.nan, "inst_freq_std": np.nan,
        "inst_freq_stability": np.nan, "inst_freq_kurtosis": np.nan,
        "inst_freq_skewness": np.nan, "inst_freq_range": np.nan,
        "inst_freq_drift": np.nan, "inst_amp_cv": np.nan,
        "inst_amp_trend": np.nan, "phase_coherence": np.nan,
        "am_fm_ratio": np.nan,
    }

    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return nan_result

    analytic = hilbert(y)
    inst_amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi)

    if len(inst_freq) < 3:
        return nan_result

    freq_mean = float(np.mean(inst_freq))
    freq_std = float(np.std(inst_freq))

    if abs(freq_mean) > 1e-12:
        freq_stability = float(np.clip(1.0 - (freq_std / abs(freq_mean)), 0.0, 1.0))
    else:
        freq_stability = 0.0

    if freq_std > 1e-12:
        freq_kurtosis = float(
            np.mean(((inst_freq - freq_mean) / freq_std) ** 4) - 3.0
        )
        freq_skewness = float(
            np.mean(((inst_freq - freq_mean) / freq_std) ** 3)
        )
    else:
        freq_kurtosis = 0.0
        freq_skewness = 0.0

    freq_range = float(np.max(inst_freq) - np.min(inst_freq))
    t = np.arange(len(inst_freq), dtype=np.float64)
    coeffs = np.polyfit(t, inst_freq, 1)
    freq_drift = float(coeffs[0])

    amp_mean = float(np.mean(inst_amp))
    amp_std = float(np.std(inst_amp))
    amp_cv = float(amp_std / amp_mean) if amp_mean > 1e-12 else np.nan

    t_amp = np.arange(len(inst_amp), dtype=np.float64)
    amp_coeffs = np.polyfit(t_amp, inst_amp, 1)
    amp_trend = float(amp_coeffs[0])

    phase_diff = np.diff(phase)
    phase_coherence = float(np.mean(np.cos(phase_diff)))
    am_fm = float(amp_std / freq_std) if freq_std > 1e-12 else np.nan

    return {
        "inst_freq_mean": freq_mean, "inst_freq_std": freq_std,
        "inst_freq_stability": freq_stability, "inst_freq_kurtosis": freq_kurtosis,
        "inst_freq_skewness": freq_skewness, "inst_freq_range": freq_range,
        "inst_freq_drift": freq_drift, "inst_amp_cv": amp_cv,
        "inst_amp_trend": amp_trend, "phase_coherence": phase_coherence,
        "am_fm_ratio": am_fm,
    }


# ---------------------------------------------------------------------------
# local_outlier_factor  (pmtvs_dynamics.domain — added after 0.1.4)
# ---------------------------------------------------------------------------
def local_outlier_factor(y: np.ndarray, n_neighbors: int = 20) -> dict:
    """Compute Local Outlier Factor scores for a 1D signal."""
    nan_result = {"mean_lof": np.nan, "max_lof": np.nan, "outlier_fraction": np.nan}
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < n_neighbors + 1:
        return nan_result

    n = len(y)
    lof_scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        all_dists = np.abs(y[i] - y)
        neighbor_indices = np.argsort(all_dists)[1 : n_neighbors + 1]

        reach_dists = np.empty(len(neighbor_indices), dtype=np.float64)
        for j_idx, j in enumerate(neighbor_indices):
            j_dists = np.abs(y[j] - y)
            j_dists_sorted = np.sort(j_dists)
            k_dist_j = j_dists_sorted[min(n_neighbors, n - 1)]
            reach_dists[j_idx] = max(k_dist_j, abs(y[i] - y[j]))

        mean_reach = np.mean(reach_dists)
        if mean_reach < 1e-15:
            lof_scores[i] = 1.0
            continue

        lrd_i = 1.0 / mean_reach

        neighbor_lrds = np.empty(len(neighbor_indices), dtype=np.float64)
        for j_idx, j in enumerate(neighbor_indices):
            j_all_dists = np.abs(y[j] - y)
            j_neighbor_indices = np.argsort(j_all_dists)[1 : n_neighbors + 1]
            j_reach_dists = np.empty(len(j_neighbor_indices), dtype=np.float64)
            for k_idx, k in enumerate(j_neighbor_indices):
                k_dists = np.sort(np.abs(y[k] - y))
                k_dist_k = k_dists[min(n_neighbors, n - 1)]
                j_reach_dists[k_idx] = max(k_dist_k, abs(y[j] - y[k]))
            mean_reach_j = np.mean(j_reach_dists)
            neighbor_lrds[j_idx] = 1.0 / max(mean_reach_j, 1e-15)

        lof_scores[i] = float(np.mean(neighbor_lrds) / lrd_i) if lrd_i > 0 else 1.0

    return {
        "mean_lof": float(np.mean(lof_scores)),
        "max_lof": float(np.max(lof_scores)),
        "outlier_fraction": float(np.sum(lof_scores > 1.5) / len(lof_scores)),
    }


# ---------------------------------------------------------------------------
# time_constant  (pmtvs_dynamics.domain — added after 0.1.4)
# ---------------------------------------------------------------------------
def time_constant(y: np.ndarray) -> dict:
    """Estimate exponential decay time constant via log-linear regression."""
    nan_result = {"tau": np.nan, "r_squared": np.nan}
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]

    if len(y) < 5:
        return nan_result

    diffs = np.diff(y)
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) < 2:
        return nan_result

    n_positive = np.sum(nonzero_diffs > 0)
    n_negative = np.sum(nonzero_diffs < 0)
    dominant_fraction = max(n_positive, n_negative) / len(nonzero_diffs)
    if dominant_fraction < 0.7:
        return nan_result

    y_min = np.min(y)
    epsilon = 1e-10
    y_shifted = y - y_min + epsilon
    t = np.arange(len(y), dtype=np.float64)
    log_y = np.log(y_shifted)

    valid = np.isfinite(log_y)
    if np.sum(valid) < 3:
        return nan_result

    t_valid = t[valid]
    log_y_valid = log_y[valid]
    coeffs = np.polyfit(t_valid, log_y_valid, 1)
    slope = coeffs[0]

    if abs(slope) < 1e-15:
        return nan_result

    tau = abs(-1.0 / slope)
    fitted = np.polyval(coeffs, t_valid)
    ss_res = np.sum((log_y_valid - fitted) ** 2)
    ss_tot = np.sum((log_y_valid - np.mean(log_y_valid)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    return {"tau": float(tau), "r_squared": r_squared}


# ---------------------------------------------------------------------------
# harmonic_ratio / total_harmonic_distortion  (pmtvs_spectral — added after 0.1.4)
# ---------------------------------------------------------------------------
def _psd(signal, fs=1.0):
    """Simple PSD via scipy — avoids circular dependency on pmtvs.individual.spectral."""
    from scipy.signal import periodogram

    freqs, power = periodogram(signal, fs=fs)
    return freqs, power


def harmonic_ratio(signal: np.ndarray, fs: float = 1.0, n_harmonics: int = 5) -> float:
    """Compute harmonic-to-noise ratio."""
    freqs, power = _psd(signal, fs=fs)
    if len(power) < 3:
        return np.nan
    f0_idx = np.argmax(power[1:]) + 1
    f0 = freqs[f0_idx]
    if f0 == 0:
        return np.nan
    harmonic_power = 0.0
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for h in range(1, n_harmonics + 1):
        harmonic_freq = h * f0
        if harmonic_freq > freqs[-1]:
            break
        h_idx = int(round(harmonic_freq / freq_resolution))
        if h_idx < len(power):
            harmonic_power += power[h_idx]
    total_power = np.sum(power)
    noise_power = total_power - harmonic_power
    if noise_power <= 0:
        return np.inf
    return float(harmonic_power / noise_power)


def total_harmonic_distortion(
    signal: np.ndarray, fs: float = 1.0, n_harmonics: int = 10
) -> float:
    """Compute total harmonic distortion (THD)."""
    freqs, power = _psd(signal, fs=fs)
    if len(power) < 3:
        return np.nan
    f0_idx = np.argmax(power[1:]) + 1
    f0 = freqs[f0_idx]
    fundamental_power = power[f0_idx]
    if f0 == 0 or fundamental_power == 0:
        return np.nan
    harmonic_power_sum = 0.0
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for h in range(2, n_harmonics + 1):
        harmonic_freq = h * f0
        if harmonic_freq > freqs[-1]:
            break
        h_idx = int(round(harmonic_freq / freq_resolution))
        if h_idx < len(power):
            harmonic_power_sum += power[h_idx]
    return float(np.sqrt(harmonic_power_sum) / np.sqrt(fundamental_power))


# ---------------------------------------------------------------------------
# hurst_r2  (pmtvs_fractal.core — added after 0.1.4)
# ---------------------------------------------------------------------------
_RS_MIN_K = 10
_RS_MAX_K_RATIO = 0.25
_RS_MAX_K_CAP = 500


def hurst_r2(signal: np.ndarray) -> float:
    """Compute R-squared of Hurst exponent fit (rescaled-range method)."""
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]
    n = len(signal)

    if n < _RS_MIN_K:
        return np.nan

    max_k = min(int(n * _RS_MAX_K_RATIO), _RS_MAX_K_CAP)
    log_k = []
    log_rs = []

    for k in range(_RS_MIN_K, max_k):
        n_subseries = n // k
        rs_sum = 0.0
        for i in range(n_subseries):
            subseries = signal[i * k : (i + 1) * k]
            Y = np.cumsum(subseries - np.mean(subseries))
            R = np.max(Y) - np.min(Y)
            S = np.std(subseries, ddof=1)
            if S > 0:
                rs_sum += R / S
        if n_subseries > 0:
            rs_avg = rs_sum / n_subseries
            if rs_avg > 0:
                log_k.append(np.log(k))
                log_rs.append(np.log(rs_avg))

    if len(log_k) < 3:
        return np.nan

    slope, intercept = np.polyfit(log_k, log_rs, 1)
    predicted = slope * np.array(log_k) + intercept
    ss_res = np.sum((np.array(log_rs) - predicted) ** 2)
    ss_tot = np.sum((np.array(log_rs) - np.mean(log_rs)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# spectral_flatness  (pmtvs_spectral — added after 0.1.4)
# ---------------------------------------------------------------------------
def spectral_flatness(signal: np.ndarray, fs: float = 1.0) -> float:
    """Compute spectral flatness (Wiener entropy). 0=tonal, 1=noise-like."""
    freqs, power = _psd(signal, fs=fs)
    power = power[power > 0]
    if len(power) == 0:
        return np.nan
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    if arithmetic_mean == 0:
        return np.nan
    return float(geometric_mean / arithmetic_mean)


# ---------------------------------------------------------------------------
# kurtosis  (pmtvs_statistics — added after 0.1.4)
# ---------------------------------------------------------------------------
def kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """Compute excess kurtosis (Fisher). Fallback for pmtvs < 0.3."""
    from scipy.stats import kurtosis as scipy_kurtosis
    return float(scipy_kurtosis(signal, fisher=fisher))
