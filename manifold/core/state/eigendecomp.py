"""
Eigendecomposition Engine (State Geometry).

Delegates to pmtvs state_eigendecomp and bootstrap_effective_dim.
"""

import warnings

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Literal

from manifold.primitives.matrix.state import (
    state_eigendecomp,
    bootstrap_effective_dim as _bootstrap_effective_dim,
)
from manifold.primitives.pairwise.regression import linear_regression


def compute(
    signal_matrix: np.ndarray,
    centroid: np.ndarray = None,
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 2,
) -> Dict[str, Any]:
    """
    Compute cohort geometry (eigenvalues) from signal matrix.

    Args:
        signal_matrix: 2D array (n_signals, n_features)
        centroid: Pre-computed centroid. If None, computed from data.
        norm_method: Normalization before SVD
        min_signals: Minimum valid signals required

    Returns:
        dict with eigenvalues, effective_dim, derived metrics, loadings
    """
    return state_eigendecomp(
        signal_matrix,
        centroid=centroid,
        norm_method=norm_method,
        min_signals=min_signals,
    )


def _empty_result(D: int) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'eigenvalues': np.full(D, np.nan),
        'explained_ratio': np.full(D, np.nan),
        'total_variance': np.nan,
        'effective_dim': np.nan,
        'eigenvalue_entropy': np.nan,
        'eigenvalue_entropy_normalized': np.nan,
        'condition_number': np.nan,
        'ratio_2_1': np.nan,
        'ratio_3_1': np.nan,
        'principal_components': None,
        'signal_loadings': None,
        'n_signals': 0,
        'n_features': D,
    }


def compute_from_signal_vector(
    signal_vector: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    group_cols: List[str] = ['unit_id', 'signal_0_end'],
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",
    min_signals: int = 3,
) -> pl.DataFrame:
    """
    Compute cohort geometry from signal_vector.parquet.

    Args:
        signal_vector: DataFrame with signal features
        feature_columns: Which columns to use as features
        group_cols: Columns to group by
        norm_method: Normalization method
        min_signals: Minimum signals per group

    Returns:
        DataFrame with eigenvalues for each group
    """
    if feature_columns is None:
        feature_columns = [
            col for col in signal_vector.columns
            if col not in ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id', 'cohort']
            and signal_vector[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

    results = []

    for group_keys, group_df in signal_vector.group_by(group_cols):
        matrix = group_df.select(feature_columns).to_numpy()
        geom = compute(matrix, norm_method=norm_method, min_signals=min_signals)

        row = dict(zip(group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
        row['effective_dim'] = geom['effective_dim']
        row['total_variance'] = geom['total_variance']
        row['condition_number'] = geom['condition_number']
        row['eigenvalue_entropy'] = geom['eigenvalue_entropy']
        row['eigenvalue_entropy_normalized'] = geom['eigenvalue_entropy_normalized']
        row['ratio_2_1'] = geom['ratio_2_1']
        row['ratio_3_1'] = geom['ratio_3_1']
        row['n_signals'] = geom['n_signals']

        eig = geom['eigenvalues']
        for i in range(min(5, len(eig))):
            row[f'eigenvalue_{i}'] = eig[i] if not np.isnan(eig[i]) else None

        exp_ratio = geom['explained_ratio']
        for i in range(min(3, len(exp_ratio))):
            row[f'explained_ratio_{i}'] = exp_ratio[i] if not np.isnan(exp_ratio[i]) else None

        results.append(row)

    return pl.DataFrame(results).sort(group_cols)


def enforce_eigenvector_continuity(
    eigenvectors_current: np.ndarray,
    eigenvectors_previous: np.ndarray,
) -> np.ndarray:
    """Ensure eigenvectors maintain consistent orientation across windows."""
    if eigenvectors_previous is None:
        return eigenvectors_current

    if eigenvectors_current.shape != eigenvectors_previous.shape:
        return eigenvectors_current

    corrected = eigenvectors_current.copy()
    n_vecs = corrected.shape[1] if corrected.ndim == 2 else 1

    if corrected.ndim == 1:
        if np.dot(corrected, eigenvectors_previous) < 0:
            corrected *= -1
        return corrected

    for j in range(n_vecs):
        dot = np.dot(eigenvectors_previous[:, j], corrected[:, j])
        if dot < 0:
            corrected[:, j] *= -1

    return corrected


def bootstrap_effective_dim(
    signal_matrix: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap confidence interval for effective dimensionality."""
    return _bootstrap_effective_dim(
        signal_matrix,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def enforce_eigenvector_continuity_sequence(
    eigenvectors_sequence: list,
) -> list:
    """Ensure eigenvector continuity across a sequence of windows."""
    if not eigenvectors_sequence:
        return eigenvectors_sequence

    corrected = [eigenvectors_sequence[0]]

    for i in range(1, len(eigenvectors_sequence)):
        prev = corrected[i - 1]
        curr = eigenvectors_sequence[i]

        if prev is None or curr is None:
            corrected.append(curr)
            continue

        corrected.append(enforce_eigenvector_continuity(curr, prev))

    return corrected


def compute_effective_dim_trend(
    effective_dims: np.ndarray,
) -> Dict[str, float]:
    """Compute trend statistics on effective dimension over time."""
    valid = ~np.isnan(effective_dims)
    if np.sum(valid) < 4:
        return {
            'eff_dim_slope': np.nan,
            'eff_dim_r2': np.nan,
        }

    x = np.arange(len(effective_dims))[valid].astype(float)
    y = effective_dims[valid]

    slope, _, r2, _ = linear_regression(x, y)

    return {
        'eff_dim_slope': slope,
        'eff_dim_r2': r2,
    }
