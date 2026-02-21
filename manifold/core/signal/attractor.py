"""
Attractor Engine.

Delegates to pmtvs embedding and dimension primitives.
Returns numbers only - Prime classifies attractor types.
"""

import warnings

import numpy as np
from typing import Dict, Union

from pmtvs import optimal_delay, optimal_dimension, time_delay_embedding
from manifold.core._pmtvs import correlation_dimension


def compute(y: np.ndarray, embedding_dim: int = None, delay: int = None) -> Dict[str, Union[float, int]]:
    """
    Compute attractor properties of signal.

    Args:
        y: Signal values
        embedding_dim: Embedding dimension (auto-detected if None)
        delay: Time delay (auto-detected if None)

    Returns:
        dict with embedding_dim, correlation_dim, delay
    """
    result = {
        'embedding_dim': np.nan,
        'correlation_dim': np.nan,
        'delay': np.nan,
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 100:
        return result

    if np.std(y) < 1e-10:
        result['correlation_dim'] = 0.0
        return result

    try:
        if delay is None:
            delay = optimal_delay(y, max_lag=min(100, n // 10), method='autocorr')

        if embedding_dim is None:
            embedding_dim = optimal_dimension(y, delay=delay, max_dim=10)

        trajectory = time_delay_embedding(y, dim=embedding_dim, tau=delay)
        corr_dim = correlation_dimension(trajectory)

        result = {
            'embedding_dim': int(embedding_dim),
            'correlation_dim': float(corr_dim) if not np.isnan(corr_dim) else np.nan,
            'delay': int(delay),
        }

    except (ValueError, ImportError):
        pass
    except Exception as e:
        warnings.warn(f"attractor.compute: {type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    return result
