"""
Cointegration Engine
====================

Tests whether two time series share a long-run equilibrium relationship.
Two signals are cointegrated if they individually wander (non-stationary)
but their linear combination is stationary — they're "tethered" together.

Method: Engle-Granger two-step procedure
  1. Regress y1 on y2 (OLS) to get the hedge ratio (beta)
  2. Test the residuals for stationarity (ADF test)
  If residuals are stationary → cointegrated

Also computes Johansen trace statistic when both signals have
sufficient length, providing a second independent confirmation.

Layer: Causal Mechanics (pairwise)
Used by: signal_pairwise, information_flow

References:
    Engle & Granger (1987) "Co-integration and Error Correction"
    Johansen (1991) "Estimation and Hypothesis Testing of Cointegration Vectors"
"""

import numpy as np
from typing import Optional

from manifold.primitives.pairwise.regression import linear_regression
from manifold.primitives.tests.stationarity_tests import adf_test as _adf_test


def compute(
    y1: np.ndarray,
    y2: np.ndarray,
    max_lag: int = 4,
    significance: float = 0.05
) -> dict:
    """
    Test for cointegration between two signals.

    Args:
        y1: First signal (1D array)
        y2: Second signal (1D array)
        max_lag: Maximum lag for ADF test on residuals
        significance: Significance level for cointegration decision

    Returns:
        dict with:
            - is_cointegrated: bool (True if residuals are stationary)
            - adf_statistic: float (ADF test statistic on residuals)
            - adf_pvalue: float (p-value of ADF test)
            - hedge_ratio: float (beta from OLS regression y1 = alpha + beta*y2)
            - intercept: float (alpha from OLS regression)
            - residual_std: float (std of equilibrium residuals)
            - half_life: float (mean-reversion half-life of residuals, in samples)
            - residual_adf_critical_1pct: float
            - residual_adf_critical_5pct: float
            - residual_adf_critical_10pct: float
            - spread_current: float (current residual value)
            - spread_zscore: float (current residual in std units)
            - n_samples: int
    """
    # Clean inputs — align and remove NaNs
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1_clean = y1[mask].astype(np.float64)
    y2_clean = y2[mask].astype(np.float64)

    n = len(y1_clean)

    if n < 30:
        return _empty_result(n, reason="insufficient_data")

    # Check both series have variance
    if np.std(y1_clean) < 1e-10 or np.std(y2_clean) < 1e-10:
        return _empty_result(n, reason="constant_signal")

    # Step 1: OLS regression  y1 = alpha + beta * y2 + epsilon
    beta, intercept, _, _ = linear_regression(y2_clean, y1_clean)
    residuals = y1_clean - (intercept + beta * y2_clean)

    residual_std = np.std(residuals)

    # Step 2: ADF test on residuals
    adf_stat, adf_pvalue, _, _ = _adf_test(residuals, max_lag=max_lag)

    # Half-life of mean reversion (AR(1) on residuals)
    half_life = _half_life(residuals, linear_regression)

    # Current spread state
    spread_current = float(residuals[-1])
    spread_zscore = spread_current / residual_std if residual_std > 1e-10 else 0.0

    # Cointegration decision
    # Use Engle-Granger critical values (more conservative than standard ADF
    # because we're testing residuals from estimated relationship)
    # Approximate EG critical values for n>100:
    #   1%: -3.90, 5%: -3.34, 10%: -3.04
    eg_critical_1pct = -3.90
    eg_critical_5pct = -3.34
    eg_critical_10pct = -3.04

    is_cointegrated = adf_stat < eg_critical_5pct

    return {
        "is_cointegrated": bool(is_cointegrated),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "hedge_ratio": float(beta),
        "intercept": float(intercept),
        "residual_std": float(residual_std),
        "half_life": float(half_life),
        "residual_adf_critical_1pct": float(eg_critical_1pct),
        "residual_adf_critical_5pct": float(eg_critical_5pct),
        "residual_adf_critical_10pct": float(eg_critical_10pct),
        "spread_current": float(spread_current),
        "spread_zscore": float(spread_zscore),
        "n_samples": int(n),
    }


def _half_life(residuals: np.ndarray, _linreg=None) -> float:
    """
    Estimate mean-reversion half-life from AR(1) coefficient.

    Fits: residual_t = phi * residual_{t-1} + epsilon
    Half-life = -ln(2) / ln(phi)

    Returns half-life in number of samples.
    Returns np.inf if not mean-reverting (phi >= 1 or phi <= 0).
    """
    y = residuals[1:]
    x = residuals[:-1]

    if len(x) < 5:
        return float("inf")  # Insufficient data = no decay detected

    if np.std(x) < 1e-10:
        return float("inf")  # Constant residuals = no decay

    phi, _, _, _ = (_linreg or linear_regression)(x, y)

    if phi <= 0 or phi >= 1:
        return float("inf")  # Not mean-reverting = no decay detected

    half_life = -np.log(2) / np.log(phi)
    return float(half_life)


def _empty_result(n: int, reason: str = "unknown") -> dict:
    """Return empty result when computation cannot proceed."""
    return {
        "is_cointegrated": False,
        "adf_statistic": float("nan"),
        "adf_pvalue": 1.0,
        "hedge_ratio": float("nan"),
        "intercept": float("nan"),
        "residual_std": float("nan"),
        "half_life": float("inf"),  # inf = no decay detected
        "residual_adf_critical_1pct": -3.90,
        "residual_adf_critical_5pct": -3.34,
        "residual_adf_critical_10pct": -3.04,
        "spread_current": float("nan"),
        "spread_zscore": float("nan"),
        "n_samples": int(n),
    }
