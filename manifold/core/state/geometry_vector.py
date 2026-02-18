"""
Geometry Vector Engine — Unified centroid + dispersion + per-engine centroids + trajectory views.

Used at Scale 1 (signals -> cohort_vector) and Scale 2 (cohorts -> system_vector).
Pure computation — numpy/dict in, dict out. No file I/O.
"""

import numpy as np
from typing import List, Dict, Optional, Any

from manifold.core.state.centroid import compute as compute_centroid
from manifold.core.state.trajectory_views import (
    compute_fourier_view,
    compute_hilbert_view,
    compute_laplacian_view,
    compute_wavelet_view,
)


def compute_geometry_vector(
    entity_matrix: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    trajectories: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Unified centroid + dispersion + per-engine centroids + trajectory views.

    Args:
        entity_matrix: N_entities x D_features matrix (signals at Scale 1, cohorts at Scale 2)
        feature_names: Names of features (columns of entity_matrix)
        feature_groups: Dict mapping engine names to feature lists
        trajectories: Optional {entity_id: {feature: array}} for trajectory views

    Returns:
        Dict with dispersion metrics, per-engine centroids, and trajectory view columns
    """
    state = compute_centroid(entity_matrix, min_signals=1)

    if state['n_signals'] < 1:
        return {}

    result = {
        'n_signals': state['n_signals'],
        'mean_distance': state['mean_distance'],
        'max_distance': state['max_distance'],
        'std_distance': state['std_distance'],
    }

    # Per-engine centroids
    feature_set = set(feature_names)
    for engine_name, features in feature_groups.items():
        available = [f for f in features if f in feature_set]
        if len(available) < 2:
            continue
        indices = [feature_names.index(f) for f in available]
        sub_matrix = entity_matrix[:, indices]
        engine_state = compute_centroid(sub_matrix, min_signals=1)
        for j, feat in enumerate(available):
            result[f'state_{engine_name}_{feat}'] = engine_state['centroid'][j]

    # Trajectory views (if provided)
    if trajectories:
        result.update(compute_fourier_view(trajectories, feature_names))
        result.update(compute_hilbert_view(trajectories, feature_names))
        result.update(compute_laplacian_view(trajectories, feature_names))
        result.update(compute_wavelet_view(trajectories, feature_names))

    return result
