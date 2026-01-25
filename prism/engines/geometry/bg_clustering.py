"""
Clustering Analysis
===================

Groups signals by behavioral similarity.

Methods:
    - Hierarchical: Agglomerative clustering with dendrogram
    - K-Means: Partition-based clustering
    - DBSCAN: Density-based (finds arbitrary shapes)

Clustering reveals which signals "move together" and helps identify
regime-specific groupings.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ClusteringResult:
    """Output from clustering analysis"""

    # Cluster assignments
    labels: np.ndarray              # Cluster label for each signal
    n_clusters: int

    # Hierarchy (if hierarchical)
    linkage_matrix: Optional[np.ndarray] = None
    dendrogram_order: Optional[List[int]] = None

    # Quality metrics
    silhouette_score: float = 0.0   # -1 to 1, higher is better
    calinski_harabasz: float = 0.0  # Higher is better

    # Cluster sizes
    cluster_sizes: List[int] = None

    # Centroids (if applicable)
    centroids: Optional[np.ndarray] = None


@dataclass
class ClusterProfile:
    """Profile of a single cluster"""

    cluster_id: int
    n_members: int
    member_indices: List[int]

    # Internal correlation
    mean_internal_correlation: float
    min_internal_correlation: float

    # Centroid characteristics (if signals have typology)
    centroid: Optional[np.ndarray] = None


def compute_hierarchical(
    distance_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    method: str = "ward"
) -> ClusteringResult:
    """
    Hierarchical clustering from distance matrix.

    Args:
        distance_matrix: Pairwise distances (n_signals, n_signals)
        n_clusters: Number of clusters (if specified)
        distance_threshold: Cut dendrogram at this height
        method: Linkage method ('ward', 'average', 'complete', 'single')

    Returns:
        ClusteringResult
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    n_signals = distance_matrix.shape[0]

    if n_signals < 2:
        return ClusteringResult(
            labels=np.array([0]),
            n_clusters=1,
            cluster_sizes=[1]
        )

    # Convert to condensed form
    condensed = squareform(distance_matrix, checks=False)

    # Linkage
    Z = linkage(condensed, method=method)

    # Cut tree
    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    elif distance_threshold is not None:
        labels = fcluster(Z, distance_threshold, criterion='distance') - 1
    else:
        # Auto-select: use elbow method on distances
        labels = fcluster(Z, 0.7 * np.max(Z[:, 2]), criterion='distance') - 1

    n_clusters_found = len(np.unique(labels))

    # Dendrogram order
    dend = dendrogram(Z, no_plot=True)
    dend_order = list(dend['leaves'])

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = [int(counts[unique == i][0]) if i in unique else 0
                     for i in range(n_clusters_found)]

    # Quality metrics
    silhouette = _compute_silhouette(distance_matrix, labels)

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters_found,
        linkage_matrix=Z,
        dendrogram_order=dend_order,
        silhouette_score=silhouette,
        cluster_sizes=cluster_sizes
    )


def compute_from_signals(
    signals: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "hierarchical"
) -> ClusteringResult:
    """
    Cluster signals directly (computes distance internally).

    Args:
        signals: 2D array (n_signals, n_observations)
        n_clusters: Number of clusters
        method: 'hierarchical' | 'kmeans' | 'dbscan'

    Returns:
        ClusteringResult
    """
    from scipy import stats

    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_signals = signals.shape[0]

    # Compute correlation distance matrix
    dist_matrix = np.zeros((n_signals, n_signals))
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            r, _ = stats.pearsonr(signals[i], signals[j])
            d = 1.0 - abs(r)
            dist_matrix[i, j] = dist_matrix[j, i] = d

    if method == "hierarchical":
        return compute_hierarchical(dist_matrix, n_clusters=n_clusters)

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        k = n_clusters or min(5, n_signals)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(signals)

        silhouette = _compute_silhouette(dist_matrix, labels)
        unique, counts = np.unique(labels, return_counts=True)

        return ClusteringResult(
            labels=labels,
            n_clusters=len(unique),
            silhouette_score=silhouette,
            cluster_sizes=[int(c) for c in counts],
            centroids=kmeans.cluster_centers_
        )

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(metric='precomputed', eps=0.5, min_samples=2)
        labels = dbscan.fit_predict(dist_matrix)

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette = _compute_silhouette(dist_matrix, labels) if n_clusters_found > 1 else 0.0

        unique, counts = np.unique(labels[labels >= 0], return_counts=True)

        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters_found,
            silhouette_score=silhouette,
            cluster_sizes=[int(c) for c in counts] if len(counts) > 0 else []
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from distance matrix."""
    n = len(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= n:
        return 0.0

    silhouettes = []

    for i in range(n):
        label_i = labels[i]

        # a(i) = mean distance to same cluster
        same_cluster = labels == label_i
        same_cluster[i] = False
        if np.sum(same_cluster) > 0:
            a_i = np.mean(distance_matrix[i, same_cluster])
        else:
            a_i = 0.0

        # b(i) = min mean distance to other clusters
        b_i = np.inf
        for label in unique_labels:
            if label != label_i:
                other_cluster = labels == label
                if np.sum(other_cluster) > 0:
                    mean_dist = np.mean(distance_matrix[i, other_cluster])
                    b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            b_i = 0.0

        # Silhouette
        max_ab = max(a_i, b_i)
        s_i = (b_i - a_i) / max_ab if max_ab > 0 else 0.0
        silhouettes.append(s_i)

    return float(np.mean(silhouettes))


def get_cluster_profiles(
    signals: np.ndarray,
    labels: np.ndarray,
    signal_ids: Optional[List[str]] = None
) -> List[ClusterProfile]:
    """
    Generate profiles for each cluster.

    Args:
        signals: Signal data (n_signals, n_obs)
        labels: Cluster assignments
        signal_ids: Optional signal identifiers

    Returns:
        List of ClusterProfile objects
    """
    from scipy import stats

    profiles = []
    unique_labels = np.unique(labels[labels >= 0])

    for label in unique_labels:
        member_mask = labels == label
        member_indices = list(np.where(member_mask)[0])
        n_members = len(member_indices)

        # Internal correlations
        if n_members > 1:
            cluster_signals = signals[member_mask]
            internal_corrs = []
            for i in range(n_members):
                for j in range(i+1, n_members):
                    r, _ = stats.pearsonr(cluster_signals[i], cluster_signals[j])
                    internal_corrs.append(r)
            mean_internal = float(np.mean(internal_corrs))
            min_internal = float(np.min(internal_corrs))
        else:
            mean_internal = 1.0
            min_internal = 1.0

        # Centroid
        centroid = np.mean(signals[member_mask], axis=0) if n_members > 0 else None

        profiles.append(ClusterProfile(
            cluster_id=int(label),
            n_members=n_members,
            member_indices=member_indices,
            mean_internal_correlation=mean_internal,
            min_internal_correlation=min_internal,
            centroid=centroid
        ))

    return profiles
