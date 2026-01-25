"""
2D Projection Methods
=====================

Project high-dimensional typology vectors to 2D for visualization.
"""

from typing import Optional
import numpy as np
import pandas as pd


AXES = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
        'derivatives', 'recurrence', 'discontinuity', 'momentum']


def project_typology_umap(
    profiles: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Project 9D typology vectors to 2D via UMAP.

    Args:
        profiles: DataFrame with signal_id and axis columns
        n_neighbors: UMAP parameter for local neighborhood
        min_dist: UMAP parameter for minimum distance
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with signal_id, proj_x, proj_y columns
    """
    try:
        import umap
    except ImportError:
        # Fallback to PCA if UMAP not installed
        return project_typology_pca(profiles)

    X = profiles[AXES].values

    if len(X) < 3:
        # Not enough data for UMAP
        result = profiles[['signal_id']].copy()
        result['proj_x'] = np.random.randn(len(X))
        result['proj_y'] = np.random.randn(len(X))
        return result

    # Adjust n_neighbors if needed
    n_neighbors = min(n_neighbors, len(X) - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    result = profiles[['signal_id']].copy()
    result['proj_x'] = embedding[:, 0]
    result['proj_y'] = embedding[:, 1]

    return result


def project_typology_tsne(
    profiles: pd.DataFrame,
    perplexity: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Project 9D typology vectors to 2D via t-SNE.

    Args:
        profiles: DataFrame with signal_id and axis columns
        perplexity: t-SNE perplexity (auto-adjusted if None)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with signal_id, proj_x, proj_y columns
    """
    from sklearn.manifold import TSNE

    X = profiles[AXES].values

    if len(X) < 3:
        # Not enough data for t-SNE
        result = profiles[['signal_id']].copy()
        result['proj_x'] = np.random.randn(len(X))
        result['proj_y'] = np.random.randn(len(X))
        return result

    # Auto-adjust perplexity
    if perplexity is None:
        perplexity = min(30, len(X) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X)

    result = profiles[['signal_id']].copy()
    result['proj_x'] = embedding[:, 0]
    result['proj_y'] = embedding[:, 1]

    return result


def project_typology_pca(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project 9D typology vectors to 2D via PCA.

    Fallback method when UMAP/t-SNE not available.

    Args:
        profiles: DataFrame with signal_id and axis columns

    Returns:
        DataFrame with signal_id, proj_x, proj_y columns
    """
    from sklearn.decomposition import PCA

    X = profiles[AXES].values

    if len(X) < 2:
        result = profiles[['signal_id']].copy()
        result['proj_x'] = 0.0
        result['proj_y'] = 0.0
        return result

    n_components = min(2, len(X), len(AXES))
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(X)

    # Pad if only 1 component
    if embedding.shape[1] < 2:
        embedding = np.hstack([embedding, np.zeros((len(X), 1))])

    result = profiles[['signal_id']].copy()
    result['proj_x'] = embedding[:, 0]
    result['proj_y'] = embedding[:, 1]

    return result
