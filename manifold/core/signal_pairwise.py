"""
ENGINES Signal Pairwise Engine

Signal pairwise computes relationships BETWEEN signals at each index.
This captures the internal structure of the signal cloud.

Computes per pair, per engine, per index:
- Correlation (do they move together in feature space?)
- Distance (how far apart?)
- Cosine similarity (same direction?)

REQUIRES: signal_vector.parquet + cohort_vector.parquet (for context)

N signals → N²/2 unique pairs per index
N ≈ 14 → ~91 pairs per index
TRACTABLE (not the N² across time we avoided)

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to primitives.
All mathematical operations are performed by directly-imported primitive functions.

Pipeline:
    signal_vector + cohort_vector → signal_pairwise.parquet → dynamics.parquet
"""

import numpy as np
import polars as pl
from typing import List, Dict, Optional, Any, Tuple
from itertools import combinations

# Import primitives for all mathematical computation
from manifold.primitives.individual.similarity import (
    euclidean_distance, cosine_similarity, correlation_coefficient,
)

# Import configuration
from manifold.config import get_config


# ============================================================
# CONFIGURATION-DRIVEN DEFAULTS
# ============================================================

def _get_feature_groups() -> Dict[str, List[str]]:
    """Get feature groups from config."""
    config = get_config()
    return config.get('geometry.feature_groups', {
        'shape': ['kurtosis', 'skewness', 'crest_factor'],
        'complexity': ['permutation_entropy', 'hurst', 'acf_lag1'],
        'spectral': ['spectral_entropy', 'spectral_centroid', 'band_low_rel', 'band_mid_rel', 'band_high_rel'],
    })


def _get_pairwise_config() -> Dict[str, Any]:
    """Get pairwise computation config."""
    config = get_config()
    return {
        'high_correlation_threshold': config.get('pairwise.correlation.high_threshold', 0.8),
        'moderate_correlation_threshold': config.get('pairwise.correlation.moderate_threshold', 0.5),
        'min_correlation': config.get('pairwise.coupling.min_correlation', 0.1),
    }


def _get_thresholds() -> Dict[str, Any]:
    """Get numerical thresholds."""
    config = get_config()
    epsilon = config.get('thresholds.numerical.epsilon', 1e-10)
    if isinstance(epsilon, str):
        epsilon = float(epsilon)
    return {
        'epsilon': epsilon,
    }


# Legacy alias for compatibility
DEFAULT_FEATURE_GROUPS = _get_feature_groups()


# ============================================================
# PAIRWISE COMPUTATION (Python)
# ============================================================

def compute_pairwise_at_index(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    centroid: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Compute pairwise relationships between all signals at single index.

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Args:
        signal_matrix: N_signals × D_features
        signal_ids: Names of signals
        centroid: Optional centroid for relative metrics

    Returns:
        List of dicts, one per pair
    """
    # Get config values
    thresholds = _get_thresholds()
    epsilon = thresholds['epsilon']

    N, D = signal_matrix.shape
    results = []

    # Precompute distances to centroid if provided
    if centroid is not None:
        # Use ENGINES primitive for each distance
        centroid_distances = np.array([
            euclidean_distance(signal_matrix[i], centroid)
            for i in range(N)
        ])
        centroid_norm = euclidean_distance(centroid, np.zeros_like(centroid))
        if centroid_norm > epsilon:
            # Projections onto centroid direction
            projections = (signal_matrix @ centroid) / centroid_norm
        else:
            projections = np.zeros(N)
    else:
        centroid_distances = None
        projections = None

    # Compute all pairs
    for i, j in combinations(range(N), 2):
        signal_a = signal_matrix[i]
        signal_b = signal_matrix[j]
        name_a = signal_ids[i]
        name_b = signal_ids[j]

        # Skip if either signal is invalid
        if not (np.isfinite(signal_a).all() and np.isfinite(signal_b).all()):
            continue

        # ─────────────────────────────────────────────────
        # DISTANCE (Euclidean) → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        distance = euclidean_distance(signal_a, signal_b)

        # ─────────────────────────────────────────────────
        # COSINE SIMILARITY → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        cos_sim = cosine_similarity(signal_a, signal_b)

        # ─────────────────────────────────────────────────
        # CORRELATION → ENGINES PRIMITIVE
        # ─────────────────────────────────────────────────
        if D > 1:
            correlation = correlation_coefficient(signal_a, signal_b)
            # Handle NaN from correlation_coefficient (can happen with constant signals)
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = cos_sim  # For 1D, same as cosine

        # ─────────────────────────────────────────────────
        # RELATIVE TO STATE (if centroid provided)
        # ─────────────────────────────────────────────────
        if centroid_distances is not None:
            dist_a_to_state = centroid_distances[i]
            dist_b_to_state = centroid_distances[j]

            # Are both close to state?
            mean_dist = np.mean(centroid_distances[np.isfinite(centroid_distances)])
            both_close = (dist_a_to_state < mean_dist) and (dist_b_to_state < mean_dist)
            both_far = (dist_a_to_state > mean_dist) and (dist_b_to_state > mean_dist)

            # Same side of state? (same sign of projection)
            if projections is not None:
                same_side = (projections[i] * projections[j]) > 0
            else:
                same_side = None

            # Distance difference to state
            state_distance_diff = abs(dist_a_to_state - dist_b_to_state)
        else:
            both_close = None
            both_far = None
            same_side = None
            state_distance_diff = None

        results.append({
            'signal_a': name_a,
            'signal_b': name_b,
            'distance': distance,
            'cosine_similarity': cos_sim,
            'correlation': correlation,
            'both_close_to_state': both_close,
            'both_far_from_state': both_far,
            'same_side_of_state': same_side,
            'state_distance_diff': state_distance_diff,
        })

    return results


def compute_signal_pairwise(
    signal_vector: pl.DataFrame,
    cohort_vector: pl.DataFrame,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    eigenvector_gating: Optional[Dict] = None,
    coloading_threshold: float = 0.1,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute signal pairwise relationships.

    Uses eigenvector co-loading to gate expensive pairwise operations.
    If two signals have high co-loading on same PC, run Granger causality.
    Otherwise, correlation/mutual_info can be derived from loadings.

    Args:
        signal_vector: Signal vector DataFrame
        cohort_vector: State vector DataFrame
        feature_groups: Dict mapping engine names to feature lists
        eigenvector_gating: Pre-built dict of eigenvector loadings for gating
        coloading_threshold: Threshold for PC co-loading to trigger Granger (default 0.5)
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL PAIRWISE ENGINE")
        print("Signal-to-signal relationships")
        print("=" * 70)

    if eigenvector_gating is None:
        eigenvector_gating = {}

    # Identify features
    meta_cols = ['unit_id', 'signal_0_start', 'signal_0_end', 'signal_0_center', 'signal_id']
    all_features = [c for c in signal_vector.columns if c not in meta_cols]

    # Determine feature groups
    if feature_groups is None:
        default_groups = _get_feature_groups()
        feature_groups = {}
        for name, features in default_groups.items():
            available = [f for f in features if f in all_features]
            if len(available) >= 2:
                feature_groups[name] = available

        if not feature_groups and len(all_features) >= 2:
            feature_groups['full'] = all_features[:3]

    # I is REQUIRED
    if 'signal_0_end' not in signal_vector.columns:
        raise ValueError("Missing required column 'signal_0_end'. Use temporal signal_vector.")

    if verbose:
        print(f"Feature groups: {list(feature_groups.keys())}")

        # Estimate output size
        n_signals = signal_vector.select('signal_id').unique().height
        n_pairs = n_signals * (n_signals - 1) // 2
        n_indices = signal_vector.select(['signal_0_end']).unique().height
        n_engines = len(feature_groups)
        print(f"Signals: {n_signals} → Pairs: {n_pairs}")
        print(f"Indices: {n_indices}")
        print(f"Estimated rows: {n_pairs * n_indices * n_engines:,}")
        print()

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in signal_vector.columns
    group_cols = ['cohort', 'signal_0_end'] if has_cohort else ['signal_0_end']

    # Process each (cohort, I) or just I
    results = []
    groups = signal_vector.group_by(group_cols, maintain_order=True)
    n_groups = signal_vector.select(group_cols).unique().height

    if verbose:
        if has_cohort:
            n_cohorts = signal_vector['cohort'].n_unique()
            print(f"Processing {n_groups} (cohort, I) groups across {n_cohorts} cohorts...")
        else:
            print(f"Processing {n_groups} time points...")

    for i, (group_key, group) in enumerate(groups):
        if has_cohort:
            cohort, s0_end = group_key if isinstance(group_key, tuple) else (None, group_key)
        else:
            cohort = None
            s0_end = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'].to_list()[0] if 'unit_id' in group.columns else ''

        # Get state vector for this (cohort, signal_0_end) or just signal_0_end
        if has_cohort and cohort:
            state_row = cohort_vector.filter(
                (pl.col('cohort') == cohort) & (pl.col('signal_0_end') == s0_end)
            )
        else:
            state_row = cohort_vector.filter(pl.col('signal_0_end') == s0_end)

        signal_ids = group['signal_id'].to_list()

        # Compute pairwise for each engine
        for engine_name, features in feature_groups.items():
            available = [f for f in features if f in group.columns]
            if len(available) < 2:
                continue

            # Get signal matrix
            matrix = group.select(available).to_numpy()

            # Get centroid from cohort_vector (if available)
            centroid_cols = [f'state_{engine_name}_{f}' for f in available]
            centroid_available = [c for c in centroid_cols if c in state_row.columns]

            if len(state_row) > 0 and len(centroid_available) == len(available):
                centroid = state_row.select(centroid_available).to_numpy().flatten()
            else:
                centroid = np.mean(matrix[np.isfinite(matrix).all(axis=1)], axis=0) if len(matrix) > 0 else None

            # Get eigenvector loadings for this (cohort, I, engine) if available
            gating_key = (cohort, s0_end, engine_name)
            pc1_loadings = eigenvector_gating.get(gating_key, {})

            # Compute pairwise
            pairs = compute_pairwise_at_index(matrix, signal_ids, centroid)

            # Build result rows
            for pair in pairs:
                signal_a = pair['signal_a']
                signal_b = pair['signal_b']

                # Check eigenvector co-loading (if available)
                # High co-loading = both signals load strongly onto same PC
                # This indicates correlation without needing full pairwise compute
                pc1_a = pc1_loadings.get(signal_a, 0.0)
                pc1_b = pc1_loadings.get(signal_b, 0.0)

                # Co-loading: product of absolute loadings (both high = high product)
                coloading = abs(pc1_a * pc1_b)

                # Flag for Granger causality: high co-loading means they're related,
                # but we need Granger to determine direction
                needs_granger = coloading > coloading_threshold

                row = {
                    'signal_0_end': s0_end,
                    'signal_a': signal_a,
                    'signal_b': signal_b,
                    'engine': engine_name,
                    'distance': pair['distance'],
                    'cosine_similarity': pair['cosine_similarity'],
                    'correlation': pair['correlation'],
                    'pc1_coloading': float(coloading),
                    'needs_granger': needs_granger,
                    'both_close_to_state': pair['both_close_to_state'],
                    'both_far_from_state': pair['both_far_from_state'],
                    'same_side_of_state': pair['same_side_of_state'],
                    'state_distance_diff': pair['state_distance_diff'],
                }
                # Include cohort if available
                if cohort:
                    row['cohort'] = cohort
                if unit_id:
                    row['unit_id'] = unit_id
                results.append(row)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_groups}...")

    # Build DataFrame
    result = pl.DataFrame(results)

    return result

