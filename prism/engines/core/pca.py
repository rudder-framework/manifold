"""
PCA Engine
==========

Principal Component Analysis for structure analysis.

CANONICAL INTERFACE:
    def compute(observations: pd.DataFrame) -> pd.DataFrame
    Input:  [entity_id, signal_id, I, y]
    Output: [I, manifold_x, manifold_y, manifold_z, explained_variance]

Measures:
- Variance explained by each component
- Loading matrix (signal weights)
- Effective dimensionality
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional


def compute(observations: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Compute PCA embedding.

    CANONICAL INTERFACE:
        Input:  observations [entity_id, signal_id, I, y]
        Output: primitives [I, manifold_x, manifold_y, manifold_z, explained_variance]

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]
        n_components: Number of PCA components (default: 3)

    Returns:
        DataFrame with PCA coordinates per time point
    """
    # Pivot to wide format: rows=I (time), cols=signal_id, values=y
    # Use first entity if multiple exist
    entity_id = observations['entity_id'].iloc[0]
    entity_obs = observations[observations['entity_id'] == entity_id]

    try:
        wide = entity_obs.pivot(index='I', columns='signal_id', values='y')
    except Exception:
        # Handle duplicate indices
        wide = entity_obs.groupby(['I', 'signal_id'])['y'].mean().unstack()

    # Drop rows/cols with all NaN
    wide = wide.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if wide.empty or wide.shape[0] < 3 or wide.shape[1] < 2:
        return pd.DataFrame({
            'entity_id': [entity_id],
            'I': [0],
            'manifold_x': [0.0],
            'manifold_y': [0.0],
            'manifold_z': [0.0],
            'explained_variance': [0.0],
        })

    # Fill remaining NaN with column means
    wide = wide.fillna(wide.mean())

    # Compute PCA
    n_comp = min(n_components, wide.shape[0], wide.shape[1])
    pca = PCA(n_components=n_comp)

    try:
        coords = pca.fit_transform(wide.values)
    except Exception:
        return pd.DataFrame({
            'entity_id': [entity_id],
            'I': [0],
            'manifold_x': [0.0],
            'manifold_y': [0.0],
            'manifold_z': [0.0],
            'explained_variance': [0.0],
        })

    # Pad to 3D if needed
    if coords.shape[1] < 3:
        padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
        coords = np.hstack([coords, padding])

    explained = sum(pca.explained_variance_ratio_)

    return pd.DataFrame({
        'entity_id': entity_id,
        'I': wide.index,
        'manifold_x': coords[:, 0],
        'manifold_y': coords[:, 1],
        'manifold_z': coords[:, 2],
        'explained_variance': explained,
    })


def compute_loadings(observations: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Compute PCA loadings (signal weights).

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]
        n_components: Number of PCA components (default: 3)

    Returns:
        DataFrame with loadings per signal [signal_id, pc1, pc2, pc3]
    """
    entity_id = observations['entity_id'].iloc[0]
    entity_obs = observations[observations['entity_id'] == entity_id]

    try:
        wide = entity_obs.pivot(index='I', columns='signal_id', values='y')
    except Exception:
        wide = entity_obs.groupby(['I', 'signal_id'])['y'].mean().unstack()

    wide = wide.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if wide.empty or wide.shape[0] < 3 or wide.shape[1] < 2:
        return pd.DataFrame({'signal_id': [], 'pc1': [], 'pc2': [], 'pc3': []})

    wide = wide.fillna(wide.mean())

    n_comp = min(n_components, wide.shape[0], wide.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(wide.values)

    loadings = pca.components_.T

    # Pad loadings if needed
    if loadings.shape[1] < 3:
        padding = np.zeros((loadings.shape[0], 3 - loadings.shape[1]))
        loadings = np.hstack([loadings, padding])

    return pd.DataFrame({
        'signal_id': wide.columns,
        'pc1': loadings[:, 0],
        'pc2': loadings[:, 1],
        'pc3': loadings[:, 2],
    })


def compute_variance(observations: pd.DataFrame, n_components: int = None) -> pd.DataFrame:
    """
    Compute variance explained by each component.

    Args:
        observations: DataFrame with columns [entity_id, signal_id, I, y]
        n_components: Number of components (default: all)

    Returns:
        DataFrame with variance per component [component, variance, cumulative]
    """
    entity_id = observations['entity_id'].iloc[0]
    entity_obs = observations[observations['entity_id'] == entity_id]

    try:
        wide = entity_obs.pivot(index='I', columns='signal_id', values='y')
    except Exception:
        wide = entity_obs.groupby(['I', 'signal_id'])['y'].mean().unstack()

    wide = wide.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if wide.empty or wide.shape[0] < 3 or wide.shape[1] < 2:
        return pd.DataFrame({'component': [], 'variance': [], 'cumulative': []})

    wide = wide.fillna(wide.mean())

    n_comp = n_components or min(wide.shape[0], wide.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(wide.values)

    variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)

    return pd.DataFrame({
        'component': range(1, len(variance) + 1),
        'variance': variance,
        'cumulative': cumulative,
    })
