"""
Stage 35: Observation Geometry Entry Point
==========================================

Per-cycle geometric scoring against the baseline established in stage 34.
This is the real-time health indicator — computable from a single observation.

Inputs:
    - observations.parquet
    - cohort_baseline.parquet (from stage 34)

Output:
    - observation_geometry.parquet

Per (cohort, I), computes:
    - centroid_distance: ||x(I) - centroid_baseline||  (how far from healthy)
    - centroid_distance_norm: normalized by sqrt(n_signals)
    - pc1_projection: x(I) . PC1  (where along primary degradation axis)
    - pc2_projection: x(I) . PC2  (lateral displacement)
    - mahalanobis_approx: weighted distance using baseline eigenvalues
    - sensor_norm: ||x(I)||  (total magnitude)

Mathematical foundation:
    Given baseline centroid c, baseline std s, principal directions V:
    x_norm(I) = (x(I) - c) / s
    centroid_distance = ||x_norm(I)||
    pc_k_projection = x_norm(I) . V_k
    mahalanobis_approx = sqrt(sum((pc_k_projection)^2 / eigenvalue_k))
"""

import argparse
import numpy as np
import polars as pl
import json
from pathlib import Path
from typing import Optional

from manifold.io.writer import write_output


def _score_cohort(cohort_data, cohort, centroid, baseline_std, pcs, signal_cols, eigenvalues):
    """Score a single cohort's observations against a baseline. Returns list of dicts."""
    try:
        wide = cohort_data.pivot(
            values='value', index='I', on='signal_id',
        ).sort('I')
    except Exception:
        return []

    if wide is None or len(wide) < 1:
        return []

    i_values = wide['I'].to_numpy()

    available_cols = [c for c in signal_cols if c in wide.columns]
    if len(available_cols) < 2:
        return []

    x = np.zeros((len(wide), len(signal_cols)))
    for j, col in enumerate(signal_cols):
        if col in wide.columns:
            vals = wide[col].to_numpy().astype(float)
            nans = np.isnan(vals)
            if nans.any() and not nans.all():
                vals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], vals[~nans])
            elif nans.all():
                vals[:] = centroid[j]
            x[:, j] = vals
        else:
            x[:, j] = centroid[j]

    x_norm = (x - centroid) / baseline_std

    dists = np.sqrt(np.sum(x_norm ** 2, axis=1))
    dists_norm = dists / np.sqrt(len(signal_cols))

    pc1_proj = x_norm @ pcs[0] if len(pcs) > 0 else np.zeros(len(x))
    pc2_proj = x_norm @ pcs[1] if len(pcs) > 1 else np.zeros(len(x))

    if len(pcs) > 0 and eigenvalues:
        projections = x_norm @ np.array(pcs[:len(eigenvalues)]).T
        mahal = np.sqrt(np.sum(projections ** 2 / np.array(eigenvalues), axis=1))
    else:
        mahal = dists

    sensor_norm = np.sqrt(np.sum(x ** 2, axis=1))

    rows = []
    for i in range(len(x)):
        rows.append({
            'cohort': cohort,
            'I': int(i_values[i]),
            'centroid_distance': float(dists[i]),
            'centroid_distance_norm': float(dists_norm[i]),
            'pc1_projection': float(pc1_proj[i]),
            'pc2_projection': float(pc2_proj[i]),
            'mahalanobis_approx': float(mahal[i]),
            'sensor_norm': float(sensor_norm[i]),
        })
    return rows


def run(
    observations_path: str,
    baseline_path: str,
    data_path: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute per-cycle geometry relative to baseline.

    Args:
        observations_path: Path to observations.parquet
        baseline_path: Path to cohort_baseline.parquet (from stage 34)
        output_path: Output path for observation_geometry.parquet
        verbose: Print progress

    Returns:
        Observation geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 35: OBSERVATION GEOMETRY")
        print("Per-cycle geometric scoring against healthy baseline")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)
    baseline = pl.read_parquet(baseline_path)

    if verbose:
        print(f"Observations: {obs.shape}")
        print(f"Baselines: {len(baseline)} cohorts")

    if len(baseline) == 0:
        if verbose:
            print("  Empty baseline — skipping")
        write_output(pl.DataFrame(), data_path, 'observation_geometry', verbose=verbose)
        return pl.DataFrame()

    has_cohort = 'cohort' in obs.columns
    results = []

    # Detect fleet mode: single row with cohort='fleet'
    is_fleet = len(baseline) == 1 and baseline['cohort'][0] == 'fleet'

    if is_fleet:
        # ─── FLEET MODE: one baseline applied to all cohorts ───
        bl_row = baseline.row(0, named=True)
        centroid = np.array(json.loads(bl_row['centroid_json']))
        baseline_std = np.array(json.loads(bl_row['std_json']))
        pcs = np.array(json.loads(bl_row['principal_directions_json']))
        signal_cols = json.loads(bl_row['signal_ids_json'])

        eigenvalues = []
        for k in range(min(len(pcs), 5)):
            ev = bl_row.get(f'baseline_eigenvalue_{k+1}')
            eigenvalues.append(ev if ev is not None and ev > 1e-12 else 1.0)

        cohorts = sorted(obs['cohort'].unique().to_list()) if has_cohort else ['all']

        if verbose:
            print(f"  Fleet mode: applying 1 baseline to {len(cohorts)} cohorts")

        for cohort_idx, cohort in enumerate(cohorts):
            if verbose and cohort_idx % 20 == 0:
                print(f"  Scoring cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

            cohort_data = obs.filter(pl.col('cohort') == cohort) if has_cohort else obs
            scored = _score_cohort(cohort_data, cohort, centroid, baseline_std,
                                   pcs, signal_cols, eigenvalues)
            results.extend(scored)
    else:
        # ─── PER-COHORT MODE: each cohort has its own baseline ───
        for row in baseline.iter_rows(named=True):
            cohort = row['cohort']
            centroid = np.array(json.loads(row['centroid_json']))
            baseline_std = np.array(json.loads(row['std_json']))
            pcs = np.array(json.loads(row['principal_directions_json']))
            signal_cols = json.loads(row['signal_ids_json'])

            eigenvalues = []
            for k in range(min(len(pcs), 5)):
                ev = row.get(f'baseline_eigenvalue_{k+1}')
                eigenvalues.append(ev if ev is not None and ev > 1e-12 else 1.0)

            if verbose and len(results) % 2000 == 0:
                n_cohorts_done = len(set(r['cohort'] for r in results)) if results else 0
                print(f"  Scoring cohort {n_cohorts_done + 1}/{len(baseline)}: {cohort}")

            cohort_data = obs.filter(pl.col('cohort') == cohort) if has_cohort else obs
            scored = _score_cohort(cohort_data, cohort, centroid, baseline_std,
                                   pcs, signal_cols, eigenvalues)
            results.extend(scored)

    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'I': pl.Int64,
            'centroid_distance': pl.Float64,
            'centroid_distance_norm': pl.Float64,
            'pc1_projection': pl.Float64,
            'pc2_projection': pl.Float64,
            'mahalanobis_approx': pl.Float64,
            'sensor_norm': pl.Float64,
        })

    write_output(result, data_path, 'observation_geometry', verbose=verbose)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 35: Observation Geometry (Per-Cycle Health Score)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes per-cycle distance from healthy baseline for each cohort.

Key outputs:
  centroid_distance:     How far from healthy (Euclidean)
  pc1_projection:        Where along primary degradation axis
  mahalanobis_approx:    Distance weighted by baseline eigenvalues

Example:
  python -m engines.entry_points.stage_35_observation_geometry \\
      observations.parquet --baseline cohort_baseline.parquet \\
      -o observation_geometry.parquet
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--baseline', required=True,
                        help='Path to cohort_baseline.parquet')
    parser.add_argument('-o', '--output', default='observation_geometry.parquet',
                        help='Output path (default: observation_geometry.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.baseline,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
