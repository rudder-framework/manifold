"""
Stage 25: Cohort Vector (System Vector)
=======================================

Cross-cohort centroid per window via unified geometry vector engine.
Reads cohort_geometry, pivots to wide per-cohort rows, then computes
the centroid across cohorts at each window — the same computation
stage 02 does across signals.

Inputs:
    - cohort_geometry.parquet (from stage 03)

Outputs:
    - system_vector.parquet (one row per signal_0_end window)
"""

import numpy as np
import polars as pl
from pathlib import Path

from manifold.core.fleet.pivot import pivot_cohort_geometry
from manifold.core.state.geometry_vector import compute_geometry_vector
from manifold.io.writer import write_output


def run(
    cohort_geometry_path: str,
    data_path: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute cross-cohort centroid per window.

    1. Read cohort_geometry.parquet
    2. Pivot to wide per-cohort rows via pivot_cohort_geometry()
    3. Group by signal_0_end (all cohorts at each window)
    4. Call compute_geometry_vector() with entities=cohorts

    Args:
        cohort_geometry_path: Path to cohort_geometry.parquet
        data_path: Data directory for output
        verbose: Print progress

    Returns:
        System vector DataFrame (one row per window)
    """
    if verbose:
        print("=" * 70)
        print("STAGE 25: SYSTEM VECTOR")
        print("Cross-cohort centroid per window")
        print("=" * 70)

    sg_path = Path(cohort_geometry_path)
    if not sg_path.exists():
        if verbose:
            print(f"  cohort_geometry.parquet not found: {sg_path}")
        write_output(pl.DataFrame(), data_path, 'system_vector', verbose=verbose)
        return pl.DataFrame()

    sg = pl.read_parquet(str(sg_path))

    if verbose:
        print(f"  Loaded cohort_geometry: {sg.shape}")

    if len(sg) == 0 or 'engine' not in sg.columns:
        if verbose:
            print("  Empty or missing engine column — skipping")
        write_output(pl.DataFrame(), data_path, 'system_vector', verbose=verbose)
        return pl.DataFrame()

    # Pivot to wide per-cohort rows
    cv = pivot_cohort_geometry(sg)

    if len(cv) == 0:
        if verbose:
            print("  Pivot produced empty result — skipping")
        write_output(pl.DataFrame(), data_path, 'system_vector', verbose=verbose)
        return pl.DataFrame()

    if verbose:
        print(f"  Pivoted: {cv.shape}")
        print(f"  Cohorts: {cv['cohort'].n_unique()}, Windows: {cv['signal_0_end'].n_unique()}")

    # Identify feature columns (everything except cohort and coordinate columns)
    meta_cols = {'cohort', 'signal_0_end', 'signal_0_start', 'signal_0_center'}
    feature_cols = [c for c in cv.columns if c not in meta_cols]

    # Process each window: compute centroid across cohorts
    s0_values = sorted(cv['signal_0_end'].unique().to_list())
    results = []

    for s0_end in s0_values:
        window = cv.filter(pl.col('signal_0_end') == s0_end)
        n_cohorts = len(window)

        if n_cohorts < 1:
            continue

        # Build feature matrix: rows = cohorts, columns = features
        matrix = window.select(feature_cols).to_numpy().astype(float)

        # Drop rows where ALL features are NaN
        valid_rows = np.isfinite(matrix).any(axis=1)
        matrix_clean = matrix[valid_rows]

        if matrix_clean.shape[0] < 1:
            continue

        # Compute centroid + dispersion (no per-engine centroids at Scale 2)
        gv = compute_geometry_vector(
            matrix_clean,
            feature_cols,
            {},
        )
        if not gv:
            continue

        # Pass through coordinate columns
        s0_start = window['signal_0_start'].to_list()[0] if 'signal_0_start' in window.columns else None
        s0_center = window['signal_0_center'].to_list()[0] if 'signal_0_center' in window.columns else None

        row = {
            'signal_0_end': s0_end,
            'signal_0_start': s0_start,
            'signal_0_center': s0_center,
            'n_cohorts': n_cohorts,
        }
        row.update(gv)
        results.append(row)

    result = pl.DataFrame(results) if results else pl.DataFrame()

    if verbose:
        print(f"  System vector: {result.shape}")

    write_output(result, data_path, 'system_vector', verbose=verbose)
    return result
