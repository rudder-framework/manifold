"""
Recurrence Quantification Analysis (RQA) Engines.

Delegates to pmtvs recurrence and dimension primitives.
"""

import numpy as np
from typing import Dict, Any
from pmtvs import optimal_dimension, time_delay_embedding
from manifold.core._pmtvs import recurrence_matrix, recurrence_rate as _recurrence_rate, determinism as _determinism, correlation_dimension as _correlation_dimension


def compute_recurrence_matrix(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1) -> np.ndarray:
    """Compute recurrence matrix from time series."""
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < embed_dim * delay + 1:
        return np.array([[]])

    trajectory = time_delay_embedding(x, dim=embed_dim, tau=delay)
    return recurrence_matrix(trajectory, threshold=threshold)


def compute_recurrence_rate(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1) -> Dict[str, float]:
    """Compute recurrence rate (RR)."""
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < embed_dim * delay + 1:
        return {'recurrence_rate': np.nan}

    R = compute_recurrence_matrix(x, threshold, embed_dim, delay)
    if R.size == 0:
        return {'recurrence_rate': np.nan}

    return {'recurrence_rate': float(_recurrence_rate(R))}


def compute_determinism(x: np.ndarray, threshold: float = None, embed_dim: int = 3, delay: int = 1, min_line: int = 2) -> Dict[str, float]:
    """Compute determinism (DET)."""
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < embed_dim * delay + 1:
        return {'determinism': np.nan}

    R = compute_recurrence_matrix(x, threshold, embed_dim, delay)
    if R.size == 0:
        return {'determinism': np.nan}

    return {'determinism': float(_determinism(R, min_line_length=min_line))}


def compute_correlation_dimension(x: np.ndarray, embed_dims: list = None, delay: int = 1) -> Dict[str, float]:
    """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 10:
        return {'correlation_dimension': np.nan}

    embed_dim = max(embed_dims) if embed_dims else 5
    trajectory = time_delay_embedding(x, dim=embed_dim, tau=delay)
    return {'correlation_dimension': float(_correlation_dimension(trajectory))}


def compute_embedding_dim(x: np.ndarray, max_dim: int = 10, delay: int = 1, threshold: float = 0.1) -> Dict[str, float]:
    """Estimate optimal embedding dimension."""
    x = np.asarray(x).flatten()
    x = x[~np.isnan(x)]

    if len(x) < 10:
        return {'embedding_dim': np.nan}

    return {'embedding_dim': float(optimal_dimension(x, tau=delay, max_dim=max_dim))}


def compute(x: np.ndarray, metric: str = 'all') -> Dict[str, float]:
    """Compute RQA metrics."""
    if metric == 'recurrence_rate':
        return compute_recurrence_rate(x)
    elif metric == 'determinism':
        return compute_determinism(x)
    elif metric == 'correlation_dimension':
        return compute_correlation_dimension(x)
    elif metric == 'embedding_dim':
        return compute_embedding_dim(x)
    else:
        result = {}
        result.update(compute_recurrence_rate(x))
        result.update(compute_determinism(x))
        result.update(compute_correlation_dimension(x))
        result.update(compute_embedding_dim(x))
        return result
