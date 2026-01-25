"""
Granger Causality
=================

Tests whether one signal helps predict another.

"X Granger-causes Y" means:
    Past values of X improve predictions of Y
    beyond what past values of Y alone provide.

NOT actual causality - just predictive precedence.
But useful for identifying lead/lag relationships.

Academic references:
    - Granger (1969) "Investigating Causal Relations by Econometric Models"
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GrangerResult:
    """Output from pairwise Granger causality test"""

    # X -> Y direction
    x_causes_y: bool
    x_causes_y_fstat: float
    x_causes_y_pvalue: float

    # Y -> X direction
    y_causes_x: bool
    y_causes_x_fstat: float
    y_causes_x_pvalue: float

    # Interpretation
    relationship: str  # 'x_leads' | 'y_leads' | 'bidirectional' | 'independent'

    # Optimal lag
    optimal_lag: int

    # Effect strength
    effect_strength: float  # Pseudo R^2 improvement


@dataclass
class GrangerMatrixResult:
    """Output from multi-signal Granger analysis"""

    # Causality matrices (value = -log10(pvalue) if significant, else 0)
    causality_matrix: np.ndarray    # [i,j] = i causes j

    # Direction matrix
    # +1 = row causes col, -1 = col causes row, 0 = no relationship
    direction_matrix: np.ndarray

    # Summary
    n_causal_pairs: int
    n_bidirectional: int

    # Most influential (causes many)
    top_causers: List[int]

    # Most influenced (caused by many)
    top_effects: List[int]


def compute(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
    significance: float = 0.05
) -> GrangerResult:
    """
    Test Granger causality between two signals.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag to test
        significance: P-value threshold

    Returns:
        GrangerResult
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < max_lag + 10:
        return GrangerResult(
            x_causes_y=False, x_causes_y_fstat=0.0, x_causes_y_pvalue=1.0,
            y_causes_x=False, y_causes_x_fstat=0.0, y_causes_x_pvalue=1.0,
            relationship="independent", optimal_lag=1, effect_strength=0.0
        )

    # Test X -> Y
    x_to_y_f, x_to_y_p, x_to_y_lag, x_to_y_strength = _granger_test(x, y, max_lag)

    # Test Y -> X
    y_to_x_f, y_to_x_p, y_to_x_lag, y_to_x_strength = _granger_test(y, x, max_lag)

    x_causes_y = x_to_y_p < significance
    y_causes_x = y_to_x_p < significance

    # Interpret relationship
    if x_causes_y and y_causes_x:
        relationship = "bidirectional"
        optimal_lag = min(x_to_y_lag, y_to_x_lag)
    elif x_causes_y:
        relationship = "x_leads"
        optimal_lag = x_to_y_lag
    elif y_causes_x:
        relationship = "y_leads"
        optimal_lag = y_to_x_lag
    else:
        relationship = "independent"
        optimal_lag = 1

    effect_strength = max(x_to_y_strength, y_to_x_strength)

    return GrangerResult(
        x_causes_y=x_causes_y,
        x_causes_y_fstat=float(x_to_y_f),
        x_causes_y_pvalue=float(x_to_y_p),
        y_causes_x=y_causes_x,
        y_causes_x_fstat=float(y_to_x_f),
        y_causes_x_pvalue=float(y_to_x_p),
        relationship=relationship,
        optimal_lag=optimal_lag,
        effect_strength=float(effect_strength)
    )


def _granger_test(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[float, float, int, float]:
    """
    Test if X Granger-causes Y.
    Returns (F-stat, p-value, optimal_lag, effect_strength).
    """
    from scipy import stats

    n = len(y)
    best_f, best_p, best_lag, best_strength = 0.0, 1.0, 1, 0.0

    for lag in range(1, max_lag + 1):
        if n - lag < lag + 5:
            continue

        # Build matrices
        Y = y[lag:]

        # Restricted model: Y ~ Y_lags
        X_restricted = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])

        # Unrestricted model: Y ~ Y_lags + X_lags
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag-i-1:n-i-1] for i in range(lag)]
        ])

        # Add constant
        ones = np.ones((len(Y), 1))
        X_r = np.hstack([ones, X_restricted])
        X_u = np.hstack([ones, X_unrestricted])

        try:
            # Fit models
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]

            # Residuals
            resid_r = Y - X_r @ beta_r
            resid_u = Y - X_u @ beta_u

            # RSS
            rss_r = np.sum(resid_r**2)
            rss_u = np.sum(resid_u**2)

            # F-test
            df1 = lag  # Additional parameters
            df2 = len(Y) - X_u.shape[1]

            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                # Effect strength (R^2 improvement)
                r2_r = 1 - rss_r / np.var(Y) / len(Y)
                r2_u = 1 - rss_u / np.var(Y) / len(Y)
                strength = r2_u - r2_r

                if p_value < best_p:
                    best_f, best_p, best_lag, best_strength = f_stat, p_value, lag, strength
        except:
            continue

    return best_f, best_p, best_lag, best_strength


def compute_matrix(
    signals: np.ndarray,
    max_lag: int = 5,
    significance: float = 0.05
) -> GrangerMatrixResult:
    """
    Compute Granger causality matrix for multiple signals.

    Args:
        signals: 2D array (n_signals, n_observations)
        max_lag: Maximum lag to test
        significance: P-value threshold

    Returns:
        GrangerMatrixResult
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]

    causality_matrix = np.zeros((n_signals, n_signals))
    direction_matrix = np.zeros((n_signals, n_signals))

    n_causal = 0
    n_bidir = 0

    for i in range(n_signals):
        for j in range(n_signals):
            if i == j:
                continue

            result = compute(signals[i], signals[j], max_lag, significance)

            # i causes j
            if result.x_causes_y:
                causality_matrix[i, j] = -np.log10(result.x_causes_y_pvalue + 1e-10)
                n_causal += 1

            # Direction
            if result.relationship == "x_leads":
                direction_matrix[i, j] = 1
            elif result.relationship == "y_leads":
                direction_matrix[i, j] = -1

    # Count bidirectional
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            if causality_matrix[i, j] > 0 and causality_matrix[j, i] > 0:
                n_bidir += 1

    # Top causers (row sums)
    causal_out = np.sum(causality_matrix > 0, axis=1)
    top_causers = list(np.argsort(causal_out)[::-1][:3])

    # Top effects (column sums)
    causal_in = np.sum(causality_matrix > 0, axis=0)
    top_effects = list(np.argsort(causal_in)[::-1][:3])

    return GrangerMatrixResult(
        causality_matrix=causality_matrix,
        direction_matrix=direction_matrix,
        n_causal_pairs=n_causal // 2,  # Approximate unique pairs
        n_bidirectional=n_bidir,
        top_causers=top_causers,
        top_effects=top_effects
    )
