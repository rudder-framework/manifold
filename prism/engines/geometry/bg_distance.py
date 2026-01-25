"""
Distance Metrics
================

Measures dissimilarity between signals.

Methods:
    - Euclidean: Standard L2 distance (on returns or levels)
    - Cosine: Angular distance (direction, ignoring magnitude)
    - DTW: Dynamic Time Warping (allows time stretching)
    - Correlation distance: 1 - |correlation|

Distance matrices enable clustering and MDS visualization.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class DistanceResult:
    """Output from pairwise distance analysis"""

    euclidean: float
    cosine: float
    correlation_distance: float     # 1 - |r|

    # Normalized versions
    euclidean_normalized: float     # Divided by joint std

    # DTW (if computed)
    dtw: Optional[float] = None
    dtw_path_length: int = 0


@dataclass
class DistanceMatrixResult:
    """Output from multi-signal distance analysis"""

    euclidean_matrix: np.ndarray
    cosine_matrix: np.ndarray
    correlation_distance_matrix: np.ndarray

    # Summary statistics
    mean_distance: float
    median_distance: float
    distance_dispersion: float

    # Diameter (max pairwise distance)
    diameter: float
    diameter_pair: tuple


def compute(
    x: np.ndarray,
    y: np.ndarray,
    use_returns: bool = True,
    compute_dtw: bool = False
) -> DistanceResult:
    """
    Compute distance between two signals.

    Args:
        x, y: Signals
        use_returns: If True, compute on returns instead of levels
        compute_dtw: If True, compute DTW (expensive)

    Returns:
        DistanceResult
    """
    from scipy import stats
    from scipy.spatial.distance import cosine as cosine_dist

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Optionally convert to returns
    if use_returns and n > 1:
        x_vals = np.diff(x) / (np.abs(x[:-1]) + 1e-10)
        y_vals = np.diff(y) / (np.abs(y[:-1]) + 1e-10)
    else:
        x_vals = x
        y_vals = y

    # Remove NaN
    valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_c, y_c = x_vals[valid], y_vals[valid]

    if len(x_c) < 2:
        return DistanceResult(
            euclidean=0.0, cosine=0.0, correlation_distance=1.0,
            euclidean_normalized=0.0
        )

    # Euclidean
    euclidean = float(np.sqrt(np.sum((x_c - y_c)**2)))

    # Normalized Euclidean
    joint_std = np.std(np.concatenate([x_c, y_c]))
    euclidean_norm = euclidean / (joint_std * np.sqrt(len(x_c))) if joint_std > 0 else 0.0

    # Cosine distance
    norm_x = np.linalg.norm(x_c)
    norm_y = np.linalg.norm(y_c)
    if norm_x > 0 and norm_y > 0:
        cosine = float(cosine_dist(x_c, y_c))
    else:
        cosine = 1.0

    # Correlation distance
    r, _ = stats.pearsonr(x_c, y_c)
    corr_dist = 1.0 - abs(r)

    # DTW (optional, expensive)
    dtw_dist = None
    dtw_path_length = 0

    if compute_dtw:
        dtw_dist, dtw_path_length = _compute_dtw(x_c, y_c)

    return DistanceResult(
        euclidean=euclidean,
        cosine=cosine,
        correlation_distance=float(corr_dist),
        euclidean_normalized=float(euclidean_norm),
        dtw=dtw_dist,
        dtw_path_length=dtw_path_length
    )


def _compute_dtw(x: np.ndarray, y: np.ndarray) -> tuple:
    """Simple DTW implementation."""
    n, m = len(x), len(y)

    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )

    # Backtrack to find path length
    i, j = n, m
    path_length = 0
    while i > 0 and j > 0:
        path_length += 1
        costs = [
            (dtw_matrix[i-1, j], (i-1, j)),
            (dtw_matrix[i, j-1], (i, j-1)),
            (dtw_matrix[i-1, j-1], (i-1, j-1))
        ]
        _, (i, j) = min(costs, key=lambda x: x[0])

    return float(dtw_matrix[n, m]), path_length


def compute_matrix(
    signals: np.ndarray,
    metric: str = "correlation"
) -> DistanceMatrixResult:
    """
    Compute distance matrix for multiple signals.

    Args:
        signals: 2D array (n_signals, n_observations)
        metric: 'euclidean' | 'cosine' | 'correlation'

    Returns:
        DistanceMatrixResult
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy import stats

    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_signals = signals.shape[0]

    # Euclidean matrix
    euclidean_matrix = squareform(pdist(signals, metric='euclidean'))

    # Cosine matrix
    cosine_matrix = squareform(pdist(signals, metric='cosine'))

    # Correlation distance matrix
    corr_dist_matrix = np.zeros((n_signals, n_signals))
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            r, _ = stats.pearsonr(signals[i], signals[j])
            d = 1.0 - abs(r)
            corr_dist_matrix[i, j] = corr_dist_matrix[j, i] = d

    # Statistics (use correlation distance as primary)
    mask = ~np.eye(n_signals, dtype=bool)
    off_diag = corr_dist_matrix[mask]

    mean_dist = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    median_dist = float(np.median(off_diag)) if len(off_diag) > 0 else 0.0
    dispersion = float(np.std(off_diag)) if len(off_diag) > 0 else 0.0

    # Diameter
    diameter = float(np.max(off_diag)) if len(off_diag) > 0 else 0.0
    if len(off_diag) > 0:
        max_idx = np.argmax(corr_dist_matrix)
        diameter_pair = (max_idx // n_signals, max_idx % n_signals)
    else:
        diameter_pair = (0, 0)

    return DistanceMatrixResult(
        euclidean_matrix=euclidean_matrix,
        cosine_matrix=cosine_matrix,
        correlation_distance_matrix=corr_dist_matrix,
        mean_distance=mean_dist,
        median_distance=median_dist,
        distance_dispersion=dispersion,
        diameter=diameter,
        diameter_pair=diameter_pair
    )
