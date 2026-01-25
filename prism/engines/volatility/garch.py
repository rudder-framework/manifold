"""
GARCH(1,1) Volatility Model
===========================

Generalized Autoregressive Conditional Heteroskedasticity model.

σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Key parameters:
    - α (alpha): ARCH effect (shock impact)
    - β (beta): GARCH effect (persistence)
    - α + β: Total persistence (< 1 for stationarity)

Persistence classification:
    - α + β < 0.85: Dissipating (shocks fade quickly)
    - 0.85 ≤ α + β < 0.99: Persistent (shocks linger)
    - α + β ≥ 0.99: Integrated (shocks permanent)

Supports three computation modes:
    - static: Entire signal → single value
    - windowed: Rolling windows → time series
    - point: At time t → single value

References:
    Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(
    series: np.ndarray,
    mode: str = 'static',
    t: Optional[int] = None,
    window_size: int = 200,
    step_size: int = 20,
) -> Dict[str, Any]:
    """
    Estimate GARCH(1,1) parameters using method of moments.

    For production, use the arch package for ML estimation.
    This implementation provides a robust approximation.

    Args:
        series: 1D numpy array of observations
        mode: 'static', 'windowed', or 'point'
        t: Time index for point mode
        window_size: Window size for windowed/point modes
        step_size: Step between windows for windowed mode

    Returns:
        mode='static': {'alpha': float, 'beta': float, 'persistence': float, ...}
        mode='windowed': {'alpha': array, 'beta': array, 't': array, ...}
        mode='point': {'alpha': float, 'beta': float, 't': int, ...}
    """
    series = np.asarray(series).flatten()

    if mode == 'static':
        return _compute_static(series)
    elif mode == 'windowed':
        return _compute_windowed(series, window_size, step_size)
    elif mode == 'point':
        return _compute_point(series, t, window_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")


def _compute_static(series: np.ndarray) -> Dict[str, Any]:
    """Estimate GARCH(1,1) on entire signal."""
    # Compute returns
    returns = np.diff(series)

    if len(returns) < 20:
        return {
            'alpha': 0.1,
            'beta': 0.8,
            'omega': 0.001,
            'persistence': 0.9,
            'unconditional_variance': float(np.var(returns)) if len(returns) > 0 else 0.0
        }

    # Squared returns
    r2 = returns ** 2

    # Sample statistics
    mean_r2 = np.mean(r2)
    var_r2 = np.var(r2)

    if var_r2 < 1e-10:
        return {
            'alpha': 0.1,
            'beta': 0.8,
            'omega': 0.001,
            'persistence': 0.9,
            'unconditional_variance': float(mean_r2)
        }

    # ACF at lag 1 of squared returns
    if len(r2) > 1:
        acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1]
        if np.isnan(acf1):
            acf1 = 0.5
    else:
        acf1 = 0.5

    # Method of moments estimates
    # Persistence ≈ ACF(1) of squared returns
    persistence = np.clip(acf1, 0, 0.999)

    # Typical alpha/beta split
    alpha = np.clip(persistence * 0.15, 0.01, 0.3)
    beta = np.clip(persistence - alpha, 0, 0.99)

    # Omega from unconditional variance
    unconditional_var = mean_r2
    omega = unconditional_var * (1 - alpha - beta) if (1 - alpha - beta) > 0 else 0.001

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'omega': float(omega),
        'persistence': float(alpha + beta),
        'unconditional_variance': float(unconditional_var)
    }


def _compute_windowed(
    series: np.ndarray,
    window_size: int,
    step_size: int,
) -> Dict[str, Any]:
    """Estimate GARCH(1,1) over rolling windows."""
    n = len(series)

    if n < window_size:
        return {
            'alpha': np.array([]),
            'beta': np.array([]),
            'omega': np.array([]),
            'persistence': np.array([]),
            'unconditional_variance': np.array([]),
            't': np.array([]),
            'window_size': window_size,
            'step_size': step_size,
        }

    t_values = []
    alpha_values = []
    beta_values = []
    omega_values = []
    persistence_values = []
    uncond_var_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        result = _compute_static(window)

        t_values.append(start + window_size // 2)
        alpha_values.append(result['alpha'])
        beta_values.append(result['beta'])
        omega_values.append(result['omega'])
        persistence_values.append(result['persistence'])
        uncond_var_values.append(result['unconditional_variance'])

    return {
        'alpha': np.array(alpha_values),
        'beta': np.array(beta_values),
        'omega': np.array(omega_values),
        'persistence': np.array(persistence_values),
        'unconditional_variance': np.array(uncond_var_values),
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }


def _compute_point(
    series: np.ndarray,
    t: int,
    window_size: int,
) -> Dict[str, Any]:
    """Estimate GARCH(1,1) at specific time t."""
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

    if len(window) < 21:  # Need at least 20 returns
        return {
            'alpha': 0.1,
            'beta': 0.8,
            'omega': 0.001,
            'persistence': 0.9,
            'unconditional_variance': 0.0,
            't': t,
            'window_start': start,
            'window_end': end,
        }

    result = _compute_static(window)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
