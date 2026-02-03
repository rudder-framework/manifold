"""
Trend Engines
==============

Trend detection and characterization.

Engines:
- rate_of_change: Linear slope
- trend_r2: Goodness of linear fit
- detrend_std: Residual variation after detrending
- cusum: Cumulative sum for change detection
"""

import numpy as np
from typing import Dict, Any


def compute_rate_of_change(values: np.ndarray) -> Dict[str, float]:
    """
    Compute linear rate of change (slope).
    
    Fits y = mx + b, returns m.
    """
    if len(values) < 2:
        return {'rate_of_change': np.nan}
    
    n = len(values)
    x = np.arange(n)
    
    # Linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    
    num = np.sum((x - x_mean) * (values - y_mean))
    denom = np.sum((x - x_mean) ** 2)
    
    if denom < 1e-10:
        return {'rate_of_change': 0.0}
    
    slope = num / denom
    
    return {'rate_of_change': float(slope)}


def compute_trend_r2(values: np.ndarray) -> Dict[str, float]:
    """
    Compute R² of linear trend fit.
    
    R² = 1: Perfect linear trend
    R² = 0: No linear trend
    """
    if len(values) < 2:
        return {'trend_r2': np.nan}
    
    n = len(values)
    x = np.arange(n)
    
    # Linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    
    num = np.sum((x - x_mean) * (values - y_mean))
    denom = np.sum((x - x_mean) ** 2)
    
    if denom < 1e-10:
        return {'trend_r2': 0.0}
    
    slope = num / denom
    intercept = y_mean - slope * x_mean
    
    # Predicted values
    y_pred = slope * x + intercept
    
    # R²
    ss_res = np.sum((values - y_pred) ** 2)
    ss_tot = np.sum((values - y_mean) ** 2)
    
    if ss_tot < 1e-10:
        return {'trend_r2': 1.0}  # Constant signal
    
    r2 = 1 - (ss_res / ss_tot)
    r2 = np.clip(r2, 0.0, 1.0)
    
    return {'trend_r2': float(r2)}


def compute_detrend_std(values: np.ndarray) -> Dict[str, float]:
    """
    Compute standard deviation after linear detrending.
    
    Low: Signal is well-explained by linear trend
    High: Significant variation beyond linear trend
    """
    if len(values) < 2:
        return {'detrend_std': np.nan}
    
    n = len(values)
    x = np.arange(n)
    
    # Linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    
    num = np.sum((x - x_mean) * (values - y_mean))
    denom = np.sum((x - x_mean) ** 2)
    
    if denom < 1e-10:
        slope = 0.0
    else:
        slope = num / denom
    
    intercept = y_mean - slope * x_mean
    
    # Detrended residuals
    residuals = values - (slope * x + intercept)
    detrend_std = np.std(residuals, ddof=1)
    
    return {'detrend_std': float(detrend_std)}


def compute_cusum(values: np.ndarray) -> Dict[str, float]:
    """
    Compute CUSUM (cumulative sum) statistics.
    
    Returns:
    - cusum_range: Range of cumulative deviations
    - cusum_max: Maximum cumulative deviation
    - cusum_min: Minimum cumulative deviation
    """
    if len(values) < 2:
        return {
            'cusum_range': np.nan,
            'cusum_max': np.nan,
            'cusum_min': np.nan,
        }
    
    # CUSUM of deviations from mean
    mean = np.mean(values)
    cusum = np.cumsum(values - mean)
    
    cusum_max = np.max(cusum)
    cusum_min = np.min(cusum)
    cusum_range = cusum_max - cusum_min
    
    return {
        'cusum_range': float(cusum_range),
        'cusum_max': float(cusum_max),
        'cusum_min': float(cusum_min),
    }


# Engine registry
ENGINES = {
    'rate_of_change': compute_rate_of_change,
    'trend_r2': compute_trend_r2,
    'detrend_std': compute_detrend_std,
    'cusum': compute_cusum,
}
