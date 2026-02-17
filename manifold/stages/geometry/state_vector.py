"""
02: State Vector Entry Point
============================

Pure orchestration - calls centroid engine from engines/state/centroid.py.
Computes WHERE the system is (centroid = mean position in feature space).

Four views per system window:
  1. Centroid  — mean feature values across signals (existing)
  2. Fourier   — spectral analysis of per-signal feature trajectories
  3. Hilbert   — envelope (amplitude modulation) of feature trajectories
  4. Laplacian — graph coupling structure across signals

Stages: signal_vector.parquet → cohort_vector.parquet

The SHAPE (eigenvalues, effective_dim) is computed in cohort_geometry (stage 03).
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any

from manifold.core.state.geometry_vector import compute_geometry_vector
from manifold.io.writer import write_output


# Feature groups for per-engine centroids
try:
    from manifold.core.geometry.config import DEFAULT_FEATURE_GROUPS, FALLBACK_FEATURES
except ImportError:
    DEFAULT_FEATURE_GROUPS = {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    }
    FALLBACK_FEATURES = ['kurtosis', 'skewness', 'crest_factor']


def _extract_trajectories(
    window_rows: pl.DataFrame,
    feature_cols: List[str],
    signal_col: str = 'signal_id',
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract per-signal feature trajectories from signal_vector rows
    within a system window.

    Groups by signal_id, sorts by signal_0_center, extracts each feature
    as a 1D array.

    Returns:
        {signal_id: {feature_name: trajectory_array}}
    """
    trajectories = {}

    for (sig_id,), sig_df in window_rows.group_by([signal_col]):
        sig_sorted = sig_df.sort('signal_0_center')
        traj = {}
        for feat in feature_cols:
            if feat in sig_sorted.columns:
                arr = sig_sorted[feat].to_numpy().astype(np.float64)
                # Drop NaNs from edges
                valid = np.isfinite(arr)
                if valid.sum() > 0:
                    traj[feat] = arr[valid]
        if traj:
            trajectories[sig_id] = traj

    return trajectories




def compute_cohort_vector(
    signal_vector_path: str,
    typology_path: str,
    data_path: str = ".",
    feature_groups: Optional[Dict[str, List[str]]] = None,
    compute_per_engine: bool = True,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute state vector (centroids + trajectory views) from signal vector.

    The state vector captures WHERE the system is in feature space (centroid)
    plus HOW features are evolving (fourier, hilbert, laplacian views).
    The SHAPE (eigenvalues, effective_dim) is computed in cohort_geometry (stage 03).

    Args:
        signal_vector_path: Path to signal_vector.parquet
        typology_path: Path to typology.parquet
        data_path: Root data directory
        feature_groups: Dict mapping engine names to feature lists
        compute_per_engine: Compute per-engine centroids
        verbose: Print progress

    Returns:
        State vector DataFrame with centroids and trajectory views per window
    """
    if verbose:
        print("=" * 70)
        print("02: STATE VECTOR - Centroids + trajectory views")
        print("=" * 70)

    # Load data
    signal_vector = pl.read_parquet(signal_vector_path)
    typology = pl.read_parquet(typology_path)

    # Get active signals (non-constant)
    if 'is_constant' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('is_constant') == False)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    elif 'signal_std' in typology.columns:
        active_signals = (
            typology
            .filter(pl.col('signal_std') > 1e-10)
            .select('signal_id')
            .to_series()
            .to_list()
        )
    else:
        active_signals = typology['signal_id'].unique().to_list()

    # Filter to active signals
    signal_col = 'signal_id' if 'signal_id' in signal_vector.columns else 'signal_name'
    signal_vector = signal_vector.filter(pl.col(signal_col).is_in(active_signals))

    # Identify features
    meta_cols = ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id', 'signal_name', 'n_samples', 'window_size', 'cohort']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    if verbose:
        print(f"Active signals: {len(active_signals)}")
        print(f"Available features: {len(all_features)}")

    # Determine feature groups
    if feature_groups is None:
        feature_groups = {}
        for name, features in DEFAULT_FEATURE_GROUPS.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups:
            fallback = [f for f in FALLBACK_FEATURES if f in all_features]
            if len(fallback) >= 2:
                feature_groups['full'] = fallback
            else:
                feature_groups['full'] = all_features[:3] if len(all_features) >= 2 else all_features

    # Composite features (union of all groups)
    composite_features = list(set(f for features in feature_groups.values() for f in features))
    composite_features = [f for f in composite_features if f in all_features]

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")
        print(f"Composite features: {len(composite_features)}")
        print()

    # Require signal_0_end column
    if 'signal_0_end' not in signal_vector.columns:
        raise ValueError("Missing required column 'signal_0_end'. Use temporal signal_vector.")

    # Determine grouping columns
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'signal_0_end'] if has_cohort else ['signal_0_end']

    # Process each group
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    if verbose:
        print(f"Processing {n_groups} groups...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, s0_end = group_key if isinstance(group_key, tuple) else (group_key, None)
        else:
            cohort = None
            s0_end = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Build composite matrix
        available_composite = [f for f in composite_features if f in group.columns]
        if len(available_composite) < 2:
            continue

        composite_matrix = group.select(available_composite).to_numpy()

        # Drop rows where ALL features are NaN (signals with no applicable engines)
        valid_rows = np.isfinite(composite_matrix).any(axis=1)
        composite_matrix = composite_matrix[valid_rows]

        # Need at least 1 signal with data
        if composite_matrix.shape[0] < 1:
            continue

        # Pass through signal_0 columns from the group
        s0_start = group['signal_0_start'].to_list()[0] if 'signal_0_start' in group.columns else None
        s0_center = group['signal_0_center'].to_list()[0] if 'signal_0_center' in group.columns else None

        # Extract trajectories for fourier/hilbert/laplacian views
        trajectories = None
        if s0_start is not None and s0_end is not None:
            if has_cohort and cohort is not None:
                window_rows = signal_vector.filter(
                    (pl.col('cohort') == cohort) &
                    (pl.col('signal_0_center') >= s0_start) &
                    (pl.col('signal_0_center') <= s0_end)
                )
            else:
                window_rows = signal_vector.filter(
                    (pl.col('signal_0_center') >= s0_start) &
                    (pl.col('signal_0_center') <= s0_end)
                )
            trajectories = _extract_trajectories(window_rows, composite_features, signal_col) or None

        # Unified centroid + dispersion + per-engine centroids + trajectory views
        gv = compute_geometry_vector(
            composite_matrix,
            available_composite,
            feature_groups if compute_per_engine else {},
            trajectories=trajectories,
        )
        if not gv:
            continue

        # Build result row
        row = {
            'signal_0_end': s0_end,
            'signal_0_start': s0_start,
            'signal_0_center': s0_center,
        }
        if cohort:
            row['cohort'] = cohort
        if unit_id:
            row['unit_id'] = unit_id
        row.update(gv)

        results.append(row)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)
    write_output(result, data_path, 'cohort_vector', verbose=verbose)

    return result


# Alias for run_pipeline.py compatibility
def run(
    signal_vector_path: str,
    data_path: str = ".",
    typology_path: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run cohort vector computation (wrapper for compute_cohort_vector)."""
    return compute_cohort_vector(
        signal_vector_path,
        typology_path,
        data_path,
        verbose=verbose,
    )


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python state_vector.py <signal_vector.parquet> <typology.parquet> [output.parquet]")
        sys.exit(1)

    signal_path = sys.argv[1]
    typology_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "cohort_vector.parquet"

    compute_cohort_vector(signal_path, typology_path, output_path)


if __name__ == "__main__":
    main()
