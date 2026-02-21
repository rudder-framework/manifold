"""
Trend Engine.

Delegates to pmtvs trend_decomposition, mann_kendall_test, rate_of_change primitives.
"""

import warnings

import numpy as np
from typing import Dict

from manifold.core._pmtvs import trend as _trend, mann_kendall_test, rate_of_change as _rate_of_change


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute trend properties.

    Args:
        y: Signal values

    Returns:
        dict with trend_slope, trend_r2, detrend_std, cusum_range
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 3:
        return {
            'trend_slope': np.nan,
            'trend_r2': np.nan,
            'detrend_std': np.nan,
            'cusum_range': np.nan,
        }

    slope, r2 = _trend(y)

    # Compute detrend_std and cusum_range inline (was in trend_decomposition)
    x = np.arange(len(y), dtype=float)
    detrended = y - (slope * x + (np.mean(y) - slope * np.mean(x)))
    detrend_std = float(np.std(detrended))

    centered = y - np.mean(y)
    cusum = np.cumsum(centered)
    cusum_range = float(np.max(cusum) - np.min(cusum))

    return {
        'trend_slope': float(slope),
        'trend_r2': float(r2),
        'detrend_std': detrend_std,
        'cusum_range': cusum_range,
    }


def compute_mann_kendall(y: np.ndarray) -> Dict[str, float]:
    """
    Non-parametric trend test.

    Returns:
        dict with mk_stat, mk_pvalue, mk_slope (Theil-Sen)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 10:
        return {
            'mk_stat': np.nan,
            'mk_pvalue': np.nan,
            'mk_slope': np.nan,
        }

    try:
        stat, pvalue = mann_kendall_test(y)
        # Theil-Sen slope computed inline (pmtvs returns stat, pvalue only)
        n = len(y)
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if j != i:
                    slopes.append((y[j] - y[i]) / (j - i))
        slope = float(np.median(slopes)) if slopes else np.nan
        return {
            'mk_stat': float(stat),
            'mk_pvalue': float(pvalue),
            'mk_slope': slope,
        }
    except ValueError:
        return {
            'mk_stat': np.nan,
            'mk_pvalue': np.nan,
            'mk_slope': np.nan,
        }
    except Exception as e:
        warnings.warn(f"trend.compute_mann_kendall: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        return {
            'mk_stat': np.nan,
            'mk_pvalue': np.nan,
            'mk_slope': np.nan,
        }


def compute_rate_of_change(y: np.ndarray) -> Dict[str, float]:
    """
    Compute rate of change statistics.

    Returns:
        dict with mean_roc, std_roc, max_roc (all normalized)
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 3:
        return {
            'mean_roc': np.nan,
            'std_roc': np.nan,
            'max_roc': np.nan,
        }

    # pmtvs rate_of_change uses raw diff; we normalize by signal std
    std_y = np.std(y)
    if std_y > 0:
        y_norm = y / std_y
    else:
        y_norm = y

    roc = _rate_of_change(y_norm)  # returns array

    return {
        'mean_roc': float(np.mean(roc)),
        'std_roc': float(np.std(roc)),
        'max_roc': float(np.max(np.abs(roc))),
    }
