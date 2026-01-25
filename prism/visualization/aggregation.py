"""
Aggregation Functions
=====================

Cohort summaries and signal clustering.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


AXES = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
        'derivatives', 'recurrence', 'discontinuity', 'momentum']


def aggregate_cohort_typology(profiles: pd.DataFrame) -> Dict:
    """
    Aggregate multiple signal profiles into cohort summary.

    Args:
        profiles: DataFrame with signal_id and axis columns

    Returns:
        Dict with mean, std, min, max for each axis
    """
    return {
        'mean': {a: float(profiles[a].mean()) for a in AXES if a in profiles.columns},
        'std': {a: float(profiles[a].std()) for a in AXES if a in profiles.columns},
        'min': {a: float(profiles[a].min()) for a in AXES if a in profiles.columns},
        'max': {a: float(profiles[a].max()) for a in AXES if a in profiles.columns},
        'n_signals': len(profiles),
    }


def cluster_signals(
    profiles: pd.DataFrame,
    n_clusters: Optional[int] = None,
    method: str = 'kmeans',
) -> Dict:
    """
    Cluster signals by typology similarity.

    Args:
        profiles: DataFrame with signal_id and axis columns
        n_clusters: Number of clusters (auto-detect if None)
        method: Clustering method ('kmeans', 'hierarchical')

    Returns:
        Dict with cluster assignments, colors, centroids
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    X = profiles[AXES].values

    if len(X) < 3:
        # Not enough data to cluster
        return {
            'assignments': {profiles['signal_id'].iloc[i]: 0 for i in range(len(profiles))},
            'colors': {0: 'hsl(180, 70%, 50%)'},
            'n_clusters': 1,
            'centroids': [X.mean(axis=0).tolist()] if len(X) > 0 else [[0.5] * 9],
        }

    # Auto-detect optimal number of clusters
    if n_clusters is None:
        best_k, best_score = 2, -1
        max_k = min(8, len(X) - 1)

        for k in range(2, max_k + 1):
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)

            labels = model.fit_predict(X)

            # Need at least 2 clusters for silhouette
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score

        n_clusters = best_k

    # Fit final model
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_.tolist()
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        # Compute centroids manually for hierarchical
        centroids = []
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                centroids.append(X[mask].mean(axis=0).tolist())
            else:
                centroids.append([0.5] * 9)

    # Assign colors to clusters (HSL color wheel)
    cluster_colors = {
        i: f'hsl({int(360 * i / n_clusters)}, 70%, 50%)'
        for i in range(n_clusters)
    }

    return {
        'assignments': dict(zip(profiles['signal_id'], labels.tolist())),
        'colors': cluster_colors,
        'n_clusters': n_clusters,
        'centroids': centroids,
    }


def compute_distinctiveness(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distinctiveness score for each signal.

    Distinctiveness = mean absolute deviation from 0.5 (neutral)
    Higher = more characteristic signal.

    Args:
        profiles: DataFrame with signal_id and axis columns

    Returns:
        DataFrame with signal_id and distinctiveness columns
    """
    result = profiles[['signal_id']].copy()

    scores = profiles[AXES].values
    distinctiveness = np.mean(np.abs(scores - 0.5), axis=1)

    result['distinctiveness'] = distinctiveness
    return result


def find_dominant_trait(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Find dominant trait for each signal.

    Args:
        profiles: DataFrame with signal_id and axis columns

    Returns:
        DataFrame with signal_id, dominant_trait, trait_score columns
    """
    result = profiles[['signal_id']].copy()

    scores = profiles[AXES].values
    dominant_idx = np.argmax(scores, axis=1)

    result['dominant_trait'] = [AXES[i] for i in dominant_idx]
    result['trait_score'] = [scores[i, dominant_idx[i]] for i in range(len(scores))]

    return result
