"""
Trend Engine.

Delegates to pmtvs trend_decomposition, mann_kendall_test, rate_of_change primitives.
"""

import warnings

import numpy as np
from typing import Dict

from manifold.primitives.individual.trend_features import (
    trend_decomposition,
    rate_of_change as _rate_of_change,
)
from manifold.primitives.individual.stationarity import mann_kendall_test


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

    r = trend_decomposition(y)

    return {
        'trend_slope': r.get('trend_slope', np.nan),
        'trend_r2': r.get('trend_r2', np.nan),
        'detrend_std': r.get('detrend_std', np.nan),
        'cusum_range': r.get('cusum_max', np.nan),
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
        stat, pvalue, slope = mann_kendall_test(y)
        return {
            'mk_stat': float(stat),
            'mk_pvalue': float(pvalue),
            'mk_slope': float(slope),
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

    r = _rate_of_change(y_norm)

    return {
        'mean_roc': r.get('mean_roc', np.nan),
        'std_roc': r.get('std_roc', np.nan),
        'max_roc': r.get('max_roc', np.nan),
    }
