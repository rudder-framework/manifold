"""
Lyapunov Exponent Estimation
============================

The largest Lyapunov exponent (λ) measures the rate of
separation of infinitesimally close trajectories.

    - λ < 0: Stable attractor (converging)
    - λ ≈ 0: Edge of chaos (critical)
    - λ > 0: Chaotic (diverging, sensitive dependence)

This is a simplified implementation using nearest neighbors.
For production, consider nolds or similar packages.

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Wolf et al. (1985) "Determining Lyapunov exponents from a time series"
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
    embedding_dim: int = 3,
    delay: int = 1,
    max_vectors: int = 300,
) -> Dict[str, Any]:
    """
    Estimate largest Lyapunov exponent.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode
        embedding_dim: Phase space embedding dimension
        delay: Time delay for embedding
        max_vectors: Maximum vectors for analysis

    Returns:
        mode='static': {'lyapunov_exponent': float, 'confidence': float}
        mode='windowed': {'lyapunov_exponent': array, 'confidence': array, 't': array, ...}
        mode='point': {'lyapunov_exponent': float, 'confidence': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series, embedding_dim, delay, max_vectors)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size, embedding_dim, delay, max_vectors)
    elif mode == 'point':
        return _compute_point(series, t, window_size, embedding_dim, delay, max_vectors)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
    max_vectors: int = 300,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent on entire signal."""
    n = len(series)
    n_vectors = n - (embedding_dim - 1) * delay

    if n_vectors < 50:
        return {'lyapunov_exponent': 0.0, 'confidence': 0.0}

    # Create embedded vectors
    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            embedded[i, j] = series[i + j * delay]

    # Subsample for efficiency
    if n_vectors > max_vectors:
        indices = np.linspace(0, n_vectors - 1, max_vectors, dtype=int)
        embedded = embedded[indices]
        n_vectors = max_vectors

    distances = cdist(embedded, embedded, 'euclidean')

    # Track divergence from nearest neighbors
    divergences = []

    for i in range(n_vectors - 10):
        dist_row = distances[i].copy()
        # Exclude self and temporal neighbors
        dist_row[max(0, i - 3):min(n_vectors, i + 4)] = np.inf

        nearest_idx = np.argmin(dist_row)
        initial_dist = dist_row[nearest_idx]

        if initial_dist < 1e-10:
            continue

        # Track divergence over time
        for k in range(1, min(10, n_vectors - max(i, nearest_idx) - 1)):
            if i + k < n_vectors and nearest_idx + k < n_vectors:
                later_dist = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                if later_dist > 1e-10 and initial_dist > 1e-10:
                    divergences.append((k, np.log(later_dist / initial_dist)))

    if len(divergences) < 10:
        return {'lyapunov_exponent': 0.0, 'confidence': 0.0}

    # Linear regression on divergence vs time
    times = np.array([d[0] for d in divergences])
    log_divs = np.array([d[1] for d in divergences])

    slope, _, r_value, _, _ = stats.linregress(times, log_divs)

    return {
        'lyapunov_exponent': float(slope),
        'confidence': float(r_value ** 2)
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
    embedding_dim: int = 3,
    delay: int = 1,
    max_vectors: int = 300,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'lyapunov_exponent': np.array([]),
            'confidence': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    lyap_values = []
    conf_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window, embedding_dim, delay, max_vectors)

        t_values.append(start + window_size // 2)
        lyap_values.append(result['lyapunov_exponent'])
        conf_values.append(result['confidence'])

    return {
        'lyapunov_exponent': np.array(lyap_values),
        'confidence': np.array(conf_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
    embedding_dim: int = 3,
    delay: int = 1,
    max_vectors: int = 300,
) -> Dict[str, Any]:
    """Estimate Lyapunov exponent at specific time t."""
    if t is None:
        raise ValueError("t is required for point mode")

    n = len(series)

    # Center window on t
    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = series[start:end]

    n_vectors = len(window) - (embedding_dim - 1) * delay
    if n_vectors < 50:
        return {
            'lyapunov_exponent': 0.0,
            'confidence': 0.0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window, embedding_dim, delay, max_vectors)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
