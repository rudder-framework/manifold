"""
Mode Discovery
==============

Discover behavioral modes from Laplace field similarity.

Modes are clusters of signals with similar Laplace field patterns,
indicating shared behavioral characteristics.
"""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

from prism.modules.signals.types import LaplaceField
from prism.geometry.coupling import compute_coupling_matrix, compute_affinity_matrix


def discover_modes(
    field_vectors: np.ndarray,
    s_values: np.ndarray,
    n_clusters: int = None,
    min_cluster_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discover behavioral modes from Laplace field similarity.

    Args:
        field_vectors: [n_signals, n_s] complex array of F(s) values
        s_values: Laplace s-values
        n_clusters: Number of clusters (if None, auto-determine)
        min_cluster_size: Minimum signals per cluster

    Returns:
        (mode_labels, mode_coherence)
        - mode_labels: Cluster label per signal
        - mode_coherence: Coherence score per mode (higher = tighter cluster)
    """
    n_signals = field_vectors.shape[0]

    if n_signals < 3:
        return np.zeros(n_signals, dtype=int), np.array([1.0])

    # Compute coupling matrix
    coupling = compute_coupling_matrix(field_vectors, s_values)

    # Convert to affinity
    affinity = compute_affinity_matrix(coupling, threshold=0.1)

    # Determine number of clusters if not specified
    if n_clusters is None:
        n_clusters = _estimate_n_clusters(affinity, min_cluster_size)

    if n_clusters < 2:
        return np.zeros(n_signals, dtype=int), np.array([1.0])

    # Spectral clustering on affinity matrix
    try:
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize',
        )
        mode_labels = clustering.fit_predict(affinity)
    except Exception:
        # Fallback: all same cluster
        return np.zeros(n_signals, dtype=int), np.array([1.0])

    # Compute coherence per mode
    mode_coherence = _compute_mode_coherence(coupling, mode_labels)

    return mode_labels, mode_coherence


def _estimate_n_clusters(
    affinity: np.ndarray,
    min_cluster_size: int,
) -> int:
    """Estimate optimal number of clusters from affinity matrix."""
    n_signals = affinity.shape[0]

    if n_signals < 4:
        return 1

    best_score = -1
    best_k = 2

    for k in range(2, min(6, n_signals)):
        try:
            clustering = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                random_state=42,
            )
            labels = clustering.fit_predict(affinity)

            # Check minimum cluster size
            unique, counts = np.unique(labels, return_counts=True)
            if min(counts) < min_cluster_size:
                continue

            # Compute silhouette score
            distance = 1 - affinity
            score = silhouette_score(distance, labels, metric='precomputed')

            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def _compute_mode_coherence(
    coupling: np.ndarray,
    mode_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute coherence score for each mode.

    Coherence = average within-mode coupling.
    """
    unique_modes = np.unique(mode_labels)
    coherence = []

    for mode in unique_modes:
        mask = mode_labels == mode
        if np.sum(mask) < 2:
            coherence.append(1.0)
            continue

        # Within-mode coupling
        mode_coupling = coupling[np.ix_(mask, mask)]
        # Average off-diagonal
        n = mode_coupling.shape[0]
        off_diag_sum = mode_coupling.sum() - np.trace(mode_coupling)
        coherence.append(off_diag_sum / (n * (n - 1)))

    return np.array(coherence)


def discover_modes_from_fields(
    fields: Dict[str, LaplaceField],
    t: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Discover modes at time t from a collection of fields.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        t: Timestamp to analyze

    Returns:
        (mode_labels, mode_coherence, signal_ids)
    """
    signal_ids = sorted(fields.keys())
    s_values = list(fields.values())[0].s_values

    # Get F(s) for each signal at time t
    field_vectors = np.array([fields[sid].at(t) for sid in signal_ids])

    mode_labels, mode_coherence = discover_modes(field_vectors, s_values)

    return mode_labels, mode_coherence, signal_ids


def track_mode_evolution(
    fields: Dict[str, LaplaceField],
    timestamps: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Track how modes evolve over time.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        timestamps: Timestamps to analyze

    Returns:
        Dict with:
        - 'n_modes': Number of modes at each timestamp
        - 'mean_coherence': Average mode coherence at each timestamp
        - 'labels': [n_t, n_signals] mode labels over time
    """
    n_modes = []
    mean_coherence = []
    all_labels = []

    for t in timestamps:
        labels, coherence, _ = discover_modes_from_fields(fields, float(t))
        n_modes.append(len(np.unique(labels)))
        mean_coherence.append(np.mean(coherence))
        all_labels.append(labels)

    return {
        'n_modes': np.array(n_modes),
        'mean_coherence': np.array(mean_coherence),
        'labels': np.array(all_labels),
    }
