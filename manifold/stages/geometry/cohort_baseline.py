"""
Stage 34: Fleet Baseline Entry Point
=====================================

SVD on early-life observations from the FLEET to establish a shared healthy
reference. All cohorts (train AND test) are scored against this same baseline.

Two modes:
    fleet (default): ONE baseline from all cohorts' early-life pooled together.
                     Use for single-regime datasets (FD001, FD003).
    per_cohort:      One baseline per cohort (original behavior).
                     Use when each cohort has genuinely different healthy states.

Inputs:
    - observations.parquet

Output:
    - cohort_baseline.parquet (one row in fleet mode, one per cohort in per_cohort)

Columns:
    - baseline_centroid: mean sensor vector from early-life
    - baseline_eigenvalues: SVD spectrum of early-life covariance
    - baseline_effective_dim: dimensionality of healthy behavior
    - baseline_total_variance: total energy in healthy state
    - principal_directions: first k eigenvectors (stored as JSON)

Mathematical foundation:
    Given observations X_early (first 20% of each cohort's lifecycle),
    pool all cohorts: X_pool = [X_early_1; X_early_2; ... X_early_N],
    center X_c = X_pool - mean(X_pool),
    normalize by std, compute SVD: X_c = U S V^T,
    eigenvalues = S^2 / (n-1),
    effective_dim = (sum(eigenvalues))^2 / sum(eigenvalues^2).
"""

import argparse
import numpy as np
import polars as pl
import json
from pathlib import Path
from typing import Optional


BASELINE_FRACTION = 0.20  # First 20% of lifecycle


def _pivot_cohort(obs: pl.DataFrame, cohort: str, has_cohort: bool) -> Optional[np.ndarray]:
    """Pivot a single cohort to wide matrix, return (x, signal_cols) or None."""
    if has_cohort:
        cohort_data = obs.filter(pl.col('cohort') == cohort)
    else:
        cohort_data = obs

    try:
        wide = cohort_data.pivot(
            values='value', index='I', on='signal_id',
        ).sort('I')
    except Exception:
        return None

    if wide is None or len(wide) < 5:
        return None

    signal_cols = sorted([c for c in wide.columns if c != 'I'])
    x = wide.select(signal_cols).to_numpy().astype(float)

    # Handle NaN: interpolate
    for j in range(x.shape[1]):
        col = x[:, j]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            col[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], col[~nans])
            x[:, j] = col
        elif nans.all():
            x[:, j] = 0.0

    return x, signal_cols


def _compute_baseline(x_baseline: np.ndarray) -> dict:
    """Compute SVD baseline from a matrix of early-life observations."""
    centroid = np.mean(x_baseline, axis=0)
    x_centered = x_baseline - centroid
    baseline_std = np.std(x_baseline, axis=0)
    baseline_std[baseline_std < 1e-12] = 1.0

    x_normed = x_centered / baseline_std

    if x_normed.shape[0] < 2 or x_normed.shape[1] < 2:
        return None

    try:
        U, S, Vt = np.linalg.svd(x_normed, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    n = x_baseline.shape[0]
    eigenvalues = (S ** 2) / max(1, n - 1)
    total_variance = float(np.sum(eigenvalues))

    if total_variance > 0:
        effective_dim = float(total_variance ** 2 / np.sum(eigenvalues ** 2))
        explained = eigenvalues / total_variance
    else:
        effective_dim = 0.0
        explained = eigenvalues

    k = min(5, len(S))
    principal_directions = Vt[:k].tolist()

    return {
        'centroid': centroid,
        'std': baseline_std,
        'eigenvalues': eigenvalues,
        'explained': explained,
        'effective_dim': effective_dim,
        'total_variance': total_variance,
        'principal_directions': principal_directions,
        'n_baseline_cycles': n,
    }


def run(
    observations_path: str,
    output_path: str = "cohort_baseline.parquet",
    baseline_fraction: float = BASELINE_FRACTION,
    mode: str = "fleet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute healthy baseline via SVD on early-life observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for cohort_baseline.parquet
        baseline_fraction: Fraction of lifecycle to use as baseline (default 0.20)
        mode: "fleet" (one shared baseline) or "per_cohort" (one per cohort)
        verbose: Print progress

    Returns:
        Baseline DataFrame
    """
    if verbose:
        print("=" * 70)
        print(f"STAGE 34: {'FLEET' if mode == 'fleet' else 'PER-COHORT'} BASELINE")
        print(f"SVD on early-life observations ({'shared' if mode == 'fleet' else 'individual'} reference)")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded observations: {obs.shape}")

    has_cohort = 'cohort' in obs.columns
    cohorts = sorted(obs['cohort'].unique().to_list()) if has_cohort else ['all']
    signals = sorted(obs['signal_id'].unique().to_list())

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Signals: {len(signals)}")
        print(f"Baseline fraction: {baseline_fraction}")
        print(f"Mode: {mode}")

    results = []

    if mode == "fleet":
        # ─── FLEET MODE: pool early-life from ALL cohorts ───
        pooled_baselines = []
        signal_cols_ref = None

        for cohort_idx, cohort in enumerate(cohorts):
            if verbose and cohort_idx % 20 == 0:
                print(f"  Collecting cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

            result = _pivot_cohort(obs, cohort, has_cohort)
            if result is None:
                continue

            x, signal_cols = result
            if signal_cols_ref is None:
                signal_cols_ref = signal_cols
            elif signal_cols != signal_cols_ref:
                # Reorder to match reference
                reorder = []
                for sc in signal_cols_ref:
                    if sc in signal_cols:
                        reorder.append(signal_cols.index(sc))
                    else:
                        reorder.append(-1)
                x_reordered = np.zeros((len(x), len(signal_cols_ref)))
                for j, idx in enumerate(reorder):
                    if idx >= 0:
                        x_reordered[:, j] = x[:, idx]
                x = x_reordered

            # Take first fraction
            n_baseline = max(3, int(len(x) * baseline_fraction))
            pooled_baselines.append(x[:n_baseline])

        if not pooled_baselines:
            if verbose:
                print("  No valid cohorts found")
            result = pl.DataFrame(schema={
                'cohort': pl.Utf8, 'n_baseline_cycles': pl.Int64,
                'baseline_effective_dim': pl.Float64,
            })
            result.write_parquet(output_path)
            return result

        x_pooled = np.vstack(pooled_baselines)
        if verbose:
            print(f"\n  Pooled baseline: {x_pooled.shape[0]} cycles from {len(pooled_baselines)} cohorts")

        bl = _compute_baseline(x_pooled)
        if bl is None:
            if verbose:
                print("  SVD failed on pooled data")
            result = pl.DataFrame(schema={'cohort': pl.Utf8})
            result.write_parquet(output_path)
            return result

        # Single fleet row — cohort='fleet' signals it applies to all
        row = {
            'cohort': 'fleet',
            'n_baseline_cycles': bl['n_baseline_cycles'],
            'n_total_cycles': sum(len(b) for b in pooled_baselines),
            'n_signals': len(signal_cols_ref),
            'n_cohorts_pooled': len(pooled_baselines),
            'baseline_effective_dim': bl['effective_dim'],
            'baseline_total_variance': bl['total_variance'],
            'baseline_eigenvalue_1': float(bl['eigenvalues'][0]) if len(bl['eigenvalues']) > 0 else None,
            'baseline_eigenvalue_2': float(bl['eigenvalues'][1]) if len(bl['eigenvalues']) > 1 else None,
            'baseline_eigenvalue_3': float(bl['eigenvalues'][2]) if len(bl['eigenvalues']) > 2 else None,
            'baseline_explained_1': float(bl['explained'][0]) if len(bl['explained']) > 0 else None,
            'baseline_explained_2': float(bl['explained'][1]) if len(bl['explained']) > 1 else None,
            'baseline_explained_3': float(bl['explained'][2]) if len(bl['explained']) > 2 else None,
            'baseline_condition_number': float(bl['eigenvalues'][0] / bl['eigenvalues'][-1]) if bl['eigenvalues'][-1] > 1e-12 else None,
            'centroid_json': json.dumps(bl['centroid'].tolist()),
            'std_json': json.dumps(bl['std'].tolist()),
            'principal_directions_json': json.dumps(bl['principal_directions']),
            'signal_ids_json': json.dumps(signal_cols_ref),
        }
        results.append(row)

    else:
        # ─── PER-COHORT MODE: original behavior ───
        for cohort_idx, cohort in enumerate(cohorts):
            if verbose and cohort_idx % 20 == 0:
                print(f"  Processing cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

            pivot_result = _pivot_cohort(obs, cohort, has_cohort)
            if pivot_result is None:
                continue

            x, signal_cols = pivot_result
            n_baseline = max(3, int(len(x) * baseline_fraction))
            x_baseline = x[:n_baseline]

            bl = _compute_baseline(x_baseline)
            if bl is None:
                continue

            row = {
                'cohort': cohort,
                'n_baseline_cycles': bl['n_baseline_cycles'],
                'n_total_cycles': len(x),
                'n_signals': len(signal_cols),
                'baseline_effective_dim': bl['effective_dim'],
                'baseline_total_variance': bl['total_variance'],
                'baseline_eigenvalue_1': float(bl['eigenvalues'][0]) if len(bl['eigenvalues']) > 0 else None,
                'baseline_eigenvalue_2': float(bl['eigenvalues'][1]) if len(bl['eigenvalues']) > 1 else None,
                'baseline_eigenvalue_3': float(bl['eigenvalues'][2]) if len(bl['eigenvalues']) > 2 else None,
                'baseline_explained_1': float(bl['explained'][0]) if len(bl['explained']) > 0 else None,
                'baseline_explained_2': float(bl['explained'][1]) if len(bl['explained']) > 1 else None,
                'baseline_explained_3': float(bl['explained'][2]) if len(bl['explained']) > 2 else None,
                'baseline_condition_number': float(bl['eigenvalues'][0] / bl['eigenvalues'][-1]) if bl['eigenvalues'][-1] > 1e-12 else None,
                'centroid_json': json.dumps(bl['centroid'].tolist()),
                'std_json': json.dumps(bl['std'].tolist()),
                'principal_directions_json': json.dumps(bl['principal_directions']),
                'signal_ids_json': json.dumps(signal_cols),
            }
            results.append(row)

    if results:
        result_df = pl.DataFrame(results)
    else:
        result_df = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'n_baseline_cycles': pl.Int64,
            'n_total_cycles': pl.Int64,
            'n_signals': pl.Int64,
            'baseline_effective_dim': pl.Float64,
            'baseline_total_variance': pl.Float64,
        })

    result_df.write_parquet(output_path)

    if verbose:
        print(f"\nShape: {result_df.shape}")
        if len(result_df) > 0:
            print(f"  Mean baseline effective_dim: {result_df['baseline_effective_dim'].mean():.2f}")
            print(f"  Mean baseline total_variance: {result_df['baseline_total_variance'].mean():.2f}")
        print()
        print("-" * 50)
        print(f"  {Path(output_path).absolute()}")
        print("-" * 50)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 34: Fleet/Cohort Baseline (Healthy Reference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SVD on early-life observations to establish healthy reference.

Fleet mode (default): pools all cohorts' early-life into ONE baseline.
Per-cohort mode: computes individual baselines per cohort.

Example:
  # Fleet baseline (recommended for single-regime datasets)
  python -m engines.entry_points.stage_34_cohort_baseline \\
      observations.parquet -o cohort_baseline.parquet --mode fleet

  # Per-cohort baseline
  python -m engines.entry_points.stage_34_cohort_baseline \\
      observations.parquet -o cohort_baseline.parquet --mode per_cohort
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='cohort_baseline.parquet',
                        help='Output path (default: cohort_baseline.parquet)')
    parser.add_argument('--baseline-fraction', type=float, default=BASELINE_FRACTION,
                        help=f'Fraction of lifecycle for baseline (default: {BASELINE_FRACTION})')
    parser.add_argument('--mode', choices=['fleet', 'per_cohort'], default='fleet',
                        help='Baseline mode: fleet (shared) or per_cohort (individual)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        baseline_fraction=args.baseline_fraction,
        mode=args.mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
