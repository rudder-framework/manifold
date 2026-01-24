"""
Geodesic Curvature Engine
=========================

Measures manifold embedding quality by comparing geodesic to Euclidean distances.

This helps determine whether linear methods (PCA) are sufficient or if
nonlinear manifold learning methods (Isomap, UMAP) are needed.

Key insight:
    - κ_geo ≈ 0: Flat manifold (PCA sufficient)
    - κ_geo > 0: Curved manifold (nonlinear methods needed)
    - κ_geo < 0: Indicates estimation issues

Usage:
    from prism.manifold_geometry.geodesic_curvature import compute

    result = compute(signals)
    if result.curvature > 0.2:
        print("Manifold is curved - use nonlinear embedding")
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import shortest_path


@dataclass
class GeodesicCurvatureResult:
    """Result from geodesic curvature computation."""

    # Main metric
    curvature: float              # Mean geodesic/euclidean ratio - 1
    curvature_std: float          # Variance in curvature

    # Distance matrices
    geodesic_distances: np.ndarray   # Shortest path distances
    euclidean_distances: np.ndarray  # Direct distances

    # Ratios
    mean_ratio: float             # Mean(geodesic/euclidean)
    max_ratio: float              # Max ratio (worst case)

    # Quality indicators
    reconstruction_error: float   # How well geodesic approximates true manifold
    linearity_score: float        # 0-1, higher = more linear

    # Recommendation
    use_nonlinear: bool           # Should use nonlinear methods?
    recommendation: str           # Human-readable recommendation


def _build_knn_graph(
    X: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Build k-nearest neighbor graph for geodesic estimation.

    Args:
        X: (n_samples, n_features) data matrix
        k: Number of neighbors

    Returns:
        Distance matrix (inf where not connected)
    """
    n = X.shape[0]

    # Compute all pairwise distances
    D = squareform(pdist(X))

    # Build kNN graph
    graph = np.full((n, n), np.inf)
    np.fill_diagonal(graph, 0)

    for i in range(n):
        # Find k nearest neighbors
        distances = D[i]
        neighbors = np.argsort(distances)[1:k+1]  # Exclude self

        for j in neighbors:
            graph[i, j] = D[i, j]
            graph[j, i] = D[j, i]  # Symmetric

    return graph


def compute(
    signals: np.ndarray,
    k_neighbors: int = 5,
    curvature_threshold: float = 0.2
) -> GeodesicCurvatureResult:
    """
    Compute geodesic curvature for signal correlation manifold.

    Compares geodesic distances (shortest path through kNN graph)
    to Euclidean distances. Higher ratio indicates more curved manifold.

    Args:
        signals: (n_signals, n_observations) array
        k_neighbors: Number of neighbors for graph
        curvature_threshold: Threshold for recommending nonlinear methods

    Returns:
        GeodesicCurvatureResult
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]

    if n_signals < 3:
        return GeodesicCurvatureResult(
            curvature=0.0,
            curvature_std=0.0,
            geodesic_distances=np.zeros((n_signals, n_signals)),
            euclidean_distances=np.zeros((n_signals, n_signals)),
            mean_ratio=1.0,
            max_ratio=1.0,
            reconstruction_error=0.0,
            linearity_score=1.0,
            use_nonlinear=False,
            recommendation="Insufficient signals for curvature estimation"
        )

    # Use correlation-based distance
    corr_matrix = np.corrcoef(signals)
    euclidean_dist = np.sqrt(2 * (1 - corr_matrix))  # Correlation distance
    np.fill_diagonal(euclidean_dist, 0)

    # Build kNN graph
    k = min(k_neighbors, n_signals - 1)
    knn_graph = _build_knn_graph(signals.T, k)  # Transpose for sample-based

    # Actually, for signal correlation manifold, use correlation distance
    graph = np.full((n_signals, n_signals), np.inf)
    np.fill_diagonal(graph, 0)

    for i in range(n_signals):
        distances = euclidean_dist[i]
        neighbors = np.argsort(distances)[1:k+1]
        for j in neighbors:
            graph[i, j] = euclidean_dist[i, j]
            graph[j, i] = euclidean_dist[j, i]

    # Compute geodesic distances (shortest paths)
    geodesic_dist = shortest_path(graph, method='auto', directed=False)

    # Handle disconnected components
    geodesic_dist[np.isinf(geodesic_dist)] = np.nan

    # Compute ratios (geodesic / euclidean)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = geodesic_dist / euclidean_dist
        ratios[euclidean_dist == 0] = 1.0  # Same point
        ratios[np.isnan(ratios)] = 1.0  # Disconnected

    # Upper triangular (excluding diagonal)
    upper_mask = np.triu(np.ones((n_signals, n_signals), dtype=bool), k=1)
    valid_ratios = ratios[upper_mask]
    valid_ratios = valid_ratios[~np.isnan(valid_ratios) & ~np.isinf(valid_ratios)]

    if len(valid_ratios) == 0:
        return GeodesicCurvatureResult(
            curvature=0.0,
            curvature_std=0.0,
            geodesic_distances=geodesic_dist,
            euclidean_distances=euclidean_dist,
            mean_ratio=1.0,
            max_ratio=1.0,
            reconstruction_error=0.0,
            linearity_score=1.0,
            use_nonlinear=False,
            recommendation="Unable to compute geodesic distances"
        )

    mean_ratio = np.mean(valid_ratios)
    max_ratio = np.max(valid_ratios)
    curvature = mean_ratio - 1.0  # 0 = flat, >0 = curved
    curvature_std = np.std(valid_ratios)

    # Reconstruction error (how well does geodesic approximate true distance)
    reconstruction_error = np.mean(np.abs(valid_ratios - 1.0))

    # Linearity score (inverse of curvature, bounded)
    linearity_score = 1.0 / (1.0 + curvature * 5)

    # Recommendation
    use_nonlinear = curvature > curvature_threshold

    if curvature < 0.1:
        recommendation = "Manifold is nearly flat - PCA/linear methods sufficient"
    elif curvature < curvature_threshold:
        recommendation = "Manifold has mild curvature - linear methods may work"
    elif curvature < 0.5:
        recommendation = "Manifold is curved - consider Isomap or UMAP"
    else:
        recommendation = "Manifold is highly curved - nonlinear methods required"

    return GeodesicCurvatureResult(
        curvature=float(curvature),
        curvature_std=float(curvature_std),
        geodesic_distances=geodesic_dist,
        euclidean_distances=euclidean_dist,
        mean_ratio=float(mean_ratio),
        max_ratio=float(max_ratio),
        reconstruction_error=float(reconstruction_error),
        linearity_score=float(linearity_score),
        use_nonlinear=use_nonlinear,
        recommendation=recommendation
    )


def estimate_intrinsic_dimension(
    signals: np.ndarray,
    method: str = "correlation"
) -> Tuple[float, float]:
    """
    Estimate intrinsic dimension of signal manifold.

    Args:
        signals: (n_signals, n_observations) array
        method: "correlation" (default) or "mle"

    Returns:
        (intrinsic_dim, linear_dim) tuple
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]

    # Linear dimension from PCA
    corr_matrix = np.corrcoef(signals)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[::-1]  # Descending

    # Effective dimension (entropy-based)
    total = np.sum(eigenvalues)
    probs = eigenvalues / total
    probs = probs[probs > 1e-10]
    entropy = -np.sum(probs * np.log(probs))
    linear_dim = np.exp(entropy)

    # For correlation dimension, would need longer computation
    # Approximation: count eigenvalues > threshold
    threshold = total * 0.01  # 1% of variance
    intrinsic_dim = np.sum(eigenvalues > threshold)

    return float(intrinsic_dim), float(linear_dim)
