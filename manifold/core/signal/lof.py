"""
LOF Engine.

Delegates to pmtvs local_outlier_factor primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._compat import local_outlier_factor


def compute(y: np.ndarray, n_neighbors: int = 20, embedding_dim: int = 3, delay: int = 1) -> Dict[str, float]:
    """
    Compute Local Outlier Factor scores.

    Args:
        y: Signal values
        n_neighbors: Number of neighbors for LOF
        embedding_dim: Embedding dimension (unused — pmtvs uses 1D LOF)
        delay: Time delay for embedding (unused)

    Returns:
        dict with lof_max, lof_mean, lof_std, outlier_fraction, n_outliers
    """
    r = local_outlier_factor(y, n_neighbors=n_neighbors)

    return {
        'lof_max': r.get('max_lof', np.nan),
        'lof_mean': r.get('mean_lof', np.nan),
        'lof_std': np.nan,
        'outlier_fraction': r.get('outlier_fraction', np.nan),
        'n_outliers': 0,
    }
