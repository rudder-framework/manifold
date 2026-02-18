"""
Attractor Reconstruction Engine.

Discovers hidden dynamical structure using Takens embedding.

Computes:
- Optimal embedding dimension (false nearest neighbors)
- Optimal delay (mutual information)
- Correlation dimension (attractor complexity)
- Recurrence quantification analysis (RQA)

Delegates all math to pmtvs primitives.
"""

import warnings

import numpy as np
from typing import Dict, Any, Optional

from manifold.primitives.embedding import (
    time_delay_embedding,
    optimal_delay,
    optimal_dimension,
)
from manifold.primitives.dynamical.dimension import (
    correlation_dimension as _correlation_dimension,
)
from manifold.primitives.dynamical.rqa import (
    recurrence_matrix as _recurrence_matrix,
    recurrence_rate as _recurrence_rate,
    determinism as _determinism,
    laminarity as _laminarity,
    trapping_time as _trapping_time,
    entropy_rqa as _entropy_rqa,
)


def compute(
    y: np.ndarray,
    min_samples: int = 100,
    max_dim: int = 10,
    max_delay: int = 100,
) -> Dict[str, Any]:
    """
    Reconstruct attractor and compute properties.

    Args:
        y: Signal values
        min_samples: Minimum samples required
        max_dim: Maximum embedding dimension to test
        max_delay: Maximum delay to test

    Returns:
        dict with embedding_dim, embedding_tau, correlation_dim, etc.
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < min_samples:
        return _empty_result()

    try:
        # Find optimal embedding parameters
        tau = optimal_delay(y, max_lag=min(max_delay, n // 10))
        dim = optimal_dimension(y, tau, max_dim=max_dim)

        # Embed signal
        embedded = time_delay_embedding(y, dimension=dim, delay=tau)

        if len(embedded) < 50:
            return _empty_result()

        # Compute correlation dimension via primitives
        corr_dim, log_r, log_C = _correlation_dimension(
            y, dimension=dim, delay=tau,
        )

        # Estimate R² from the log-log fit
        if log_r is not None and len(log_r) >= 5:
            from manifold.primitives.pairwise.regression import linear_regression
            _, _, corr_dim_r2, _ = linear_regression(log_r, log_C)
        else:
            corr_dim_r2 = np.nan

        return {
            'embedding_dim': dim,
            'embedding_tau': tau,
            'correlation_dim': float(corr_dim) if np.isfinite(corr_dim) else None,
            'correlation_dim_r2': float(corr_dim_r2) if np.isfinite(corr_dim_r2) else None,
            'n_embedded': len(embedded),
        }

    except (ValueError, np.linalg.LinAlgError):
        return _empty_result()
    except Exception as e:
        warnings.warn(f"attractor.compute: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        return _empty_result()


def _empty_result() -> Dict[str, Any]:
    """Return empty result."""
    return {
        'embedding_dim': None,
        'embedding_tau': None,
        'correlation_dim': None,
        'correlation_dim_r2': None,
        'n_embedded': 0,
    }


def compute_recurrence_matrix(
    embedded: np.ndarray,
    threshold: float = None,
    threshold_pct: float = 10.0,
) -> np.ndarray:
    """
    Compute recurrence matrix via primitives.

    Args:
        embedded: Embedded trajectory
        threshold: Fixed threshold distance
        threshold_pct: Percentile of distances for threshold

    Returns:
        Boolean recurrence matrix
    """
    # primitives recurrence_matrix takes raw signal + embedding params,
    # but we already have embedded data. Pass through appropriately.
    # Use dimension=1 and delay=1 since data is already embedded.
    return _recurrence_matrix(
        embedded[:, 0] if embedded.ndim == 2 else embedded,
        dimension=embedded.shape[1] if embedded.ndim == 2 else 1,
        delay=1,
        threshold=threshold,
        threshold_percentile=threshold_pct,
    )


def compute_rqa_from_matrix(R: np.ndarray) -> Dict[str, float]:
    """
    Compute RQA metrics from recurrence matrix via primitives.

    Returns:
        dict with recurrence_rate, determinism, laminarity, etc.
    """
    return {
        'recurrence_rate': float(_recurrence_rate(R)),
        'determinism': float(_determinism(R)),
        'laminarity': float(_laminarity(R)),
        'rqa_entropy': float(_entropy_rqa(R)),
        'trapping_time': float(_trapping_time(R)),
    }
