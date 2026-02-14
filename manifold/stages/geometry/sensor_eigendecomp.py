"""
Stage 20: Rolling Sensor Eigendecomposition
============================================

Per-cohort rolling eigendecomposition of the sensor observation matrix.
Captures how sensor inter-correlation structure evolves over the index range.

Key finding: effective_dim of sensor covariance correlates r=0.849 with
remaining useful life across N=249 turbofan engines (FD_004).

Input:
    observations.parquet (cohort, signal_id, I, value)

Output:
    sensor_eigendecomp.parquet (one row per cohort per window)

Algorithm (2-level rolling):
    1. Per cohort: pivot observations to wide (rows=I, cols=signal_id)
    2. Level 1: Rolling mean of raw obs (window=agg_window, stride=agg_stride)
       -> one aggregated row per stride step, each = sensor mean over window
    3. Level 2: At each aggregated row, eigendecomp over the most recent
       `lookback` aggregated rows (expanding until capped)
    4. Enforce eigenvector continuity across sequential windows
    5. Output per-window eigendecomp results

Uses existing eigendecomp.compute() primitive -- zero new math.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path

from manifold.core.state.eigendecomp import (
    compute as compute_eigendecomp,
    enforce_eigenvector_continuity,
)
from manifold.io.writer import write_output


def _aggregate_sensor_means(matrix: np.ndarray, i_values: np.ndarray,
                            agg_window: int, agg_stride: int):
    """
    Level 1: compute rolling per-sensor means.

    Args:
        matrix: (n_obs, n_signals) raw observation matrix
        i_values: (n_obs,) I index values
        agg_window: number of raw observations per aggregation window
        agg_stride: stride between aggregation windows

    Returns:
        agg_matrix: (n_windows, n_signals) sensor means per window
        agg_i: (n_windows,) I value at end of each window
    """
    n_obs = len(matrix)
    effective_window = min(agg_window, n_obs)
    if effective_window < 10:
        effective_window = n_obs

    agg_rows = []
    agg_i = []

    for start in range(0, n_obs - effective_window + 1, agg_stride):
        end = start + effective_window
        window = matrix[start:end]

        # Per-sensor mean (ignoring NaN)
        with np.errstate(all='ignore'):
            means = np.nanmean(window, axis=0)

        agg_rows.append(means)
        agg_i.append(i_values[end - 1])

    if not agg_rows:
        return np.empty((0, matrix.shape[1])), np.empty(0, dtype=int)

    return np.array(agg_rows), np.array(agg_i)


def run(
    observations_path: str,
    data_path: str = ".",
    agg_window: int = 30,
    agg_stride: int = 5,
    lookback: int = 30,
    norm_method: str = "zscore",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Rolling eigendecomposition of sensor matrix per cohort (2-level).

    Level 1: Aggregate raw observations into rolling sensor means.
    Level 2: Eigendecomp over lookback window of aggregated rows.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for sensor_eigendecomp.parquet
        agg_window: Raw observation aggregation window size
        agg_stride: Stride between aggregation windows
        lookback: Number of aggregated rows for eigendecomp
        norm_method: Normalization for eigendecomp ("zscore", "robust", "mad", "none")
        verbose: Print progress

    Returns:
        DataFrame with per-cohort, per-window eigendecomp results
    """
    if verbose:
        print("=" * 70)
        print("STAGE 20: ROLLING SENSOR EIGENDECOMPOSITION")
        print("Per-cohort eigendecomp of sensor observation matrix")
        print("=" * 70)

    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"\nLoaded observations: {obs.shape}")

    has_cohort = 'cohort' in obs.columns
    cohorts = sorted(obs['cohort'].unique().to_list()) if has_cohort else ['all']

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Level 1: agg_window={agg_window}, agg_stride={agg_stride}")
        print(f"Level 2: lookback={lookback}, norm={norm_method}")

    all_results = []

    for ci, cohort in enumerate(cohorts):
        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        # Pivot to wide: rows=I, cols=signal_id, values=value
        try:
            wide = cohort_data.pivot(
                values='value',
                index='I',
                on='signal_id',
            ).sort('I')
        except Exception:
            continue

        signal_cols = sorted([c for c in wide.columns if c != 'I'])
        n_signals = len(signal_cols)

        if n_signals < 2:
            continue

        i_values = wide['I'].to_numpy()
        matrix_full = wide.select(signal_cols).to_numpy()

        # Level 1: aggregate into sensor means
        agg_matrix, agg_i = _aggregate_sensor_means(
            matrix_full, i_values, agg_window, agg_stride
        )

        if len(agg_matrix) < 3:
            continue

        # Level 2: rolling eigendecomp over aggregated rows
        prev_pcs = None

        for i in range(len(agg_matrix)):
            # Expanding-then-capped lookback (matches ML behavior)
            lb = min(lookback, i + 1)
            window = agg_matrix[i - lb + 1:i + 1]

            # Remove columns with NaN
            valid_cols = ~np.any(np.isnan(window), axis=0)
            window_clean = window[:, valid_cols]

            # Always output a row (NaN when eigendecomp can't compute)
            if window_clean.shape[0] < 3 or window_clean.shape[1] < 2:
                all_results.append({
                    'cohort': cohort,
                    'I': int(agg_i[i]),
                    'n_signals': n_signals,
                    'n_samples': window_clean.shape[0],
                    'effective_dim': np.nan,
                    'eff_dim_entropy': np.nan,
                    'eigenvalue_entropy': np.nan,
                    'eigenvalue_entropy_normalized': np.nan,
                    'total_variance': np.nan,
                    'condition_number': np.nan,
                    'ratio_2_1': np.nan,
                    'ratio_3_1': np.nan,
                    'energy_concentration': np.nan,
                })
                continue

            # Eigendecomp (uses existing primitive)
            result = compute_eigendecomp(
                window_clean,
                norm_method=norm_method,
                min_signals=2,
            )

            if np.isnan(result['effective_dim']):
                all_results.append({
                    'cohort': cohort,
                    'I': int(agg_i[i]),
                    'n_signals': n_signals,
                    'n_samples': window_clean.shape[0],
                    'effective_dim': np.nan,
                    'eff_dim_entropy': np.nan,
                    'eigenvalue_entropy': np.nan,
                    'eigenvalue_entropy_normalized': np.nan,
                    'total_variance': np.nan,
                    'condition_number': np.nan,
                    'ratio_2_1': np.nan,
                    'ratio_3_1': np.nan,
                    'energy_concentration': np.nan,
                })
                continue

            # Eigenvector continuity
            current_pcs = result.get('principal_components')
            if current_pcs is not None and prev_pcs is not None:
                try:
                    current_pcs = enforce_eigenvector_continuity(current_pcs, prev_pcs)
                except (ValueError, IndexError):
                    pass
            prev_pcs = current_pcs

            # Entropy-based eff_dim: exp(-sum(p*log(p)))
            eigenvalues = result['eigenvalues']
            total = np.sum(eigenvalues)
            if total > 0:
                p = eigenvalues[eigenvalues > 0] / total
                eff_dim_entropy = float(np.exp(-np.sum(p * np.log(p))))
            else:
                eff_dim_entropy = np.nan

            # Build row (I = end of aggregation window)
            row = {
                'cohort': cohort,
                'I': int(agg_i[i]),
                'n_signals': result['n_signals'],
                'n_samples': window_clean.shape[0],
                'effective_dim': result['effective_dim'],
                'eff_dim_entropy': eff_dim_entropy,
                'eigenvalue_entropy': result['eigenvalue_entropy'],
                'eigenvalue_entropy_normalized': result['eigenvalue_entropy_normalized'],
                'total_variance': result['total_variance'],
                'condition_number': result['condition_number'],
                'ratio_2_1': result['ratio_2_1'],
                'ratio_3_1': result['ratio_3_1'],
            }

            # Top eigenvalues
            for k in range(min(5, len(eigenvalues))):
                val = eigenvalues[k]
                row[f'eigenvalue_{k}'] = float(val) if np.isfinite(val) else None

            # Top explained ratios
            exp_ratio = result['explained_ratio']
            for k in range(min(3, len(exp_ratio))):
                val = exp_ratio[k]
                row[f'explained_ratio_{k}'] = float(val) if np.isfinite(val) else None

            # Energy concentration: eigenvalue_0 / total
            row['energy_concentration'] = float(eigenvalues[0] / total) if total > 0 and len(eigenvalues) > 0 else None

            all_results.append(row)

        if verbose and (ci + 1) % 50 == 0:
            print(f"  Processed {ci + 1}/{len(cohorts)} cohorts ({len(all_results)} windows)")

    # Build output
    if all_results:
        df = pl.DataFrame(all_results)
    else:
        df = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'I': pl.UInt32,
            'n_signals': pl.Int64,
            'n_samples': pl.Int64,
            'effective_dim': pl.Float64,
            'eff_dim_entropy': pl.Float64,
            'eigenvalue_entropy': pl.Float64,
            'eigenvalue_entropy_normalized': pl.Float64,
            'total_variance': pl.Float64,
            'condition_number': pl.Float64,
            'ratio_2_1': pl.Float64,
            'ratio_3_1': pl.Float64,
        })

    write_output(df, data_path, 'sensor_eigendecomp', verbose=verbose)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 20: Rolling Sensor Eigendecomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Per-cohort rolling eigendecomposition of sensor observation matrix.

Algorithm (2-level rolling):
  Level 1: Aggregate raw obs into rolling sensor means
           (agg_window obs per mean, agg_stride between windows)
  Level 2: Eigendecomp over lookback aggregated rows
           (expanding-then-capped window)

Output schema:
  cohort, I, n_signals, n_samples,
  effective_dim (participation ratio), eff_dim_entropy (exp-entropy),
  eigenvalue_entropy, total_variance, condition_number,
  ratio_2_1, ratio_3_1, eigenvalue_0..4, explained_ratio_0..2,
  energy_concentration

Example:
  python -m engines.entry_points.stage_20_sensor_eigendecomp \\
      observations.parquet -o sensor_eigendecomp.parquet \\
      --agg-window 30 --agg-stride 5 --lookback 30
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='sensor_eigendecomp.parquet',
                        help='Output path (default: sensor_eigendecomp.parquet)')
    parser.add_argument('--agg-window', type=int, default=30,
                        help='Level 1 aggregation window size (default: 30)')
    parser.add_argument('--agg-stride', type=int, default=5,
                        help='Level 1 aggregation stride (default: 5)')
    parser.add_argument('--lookback', type=int, default=30,
                        help='Level 2 eigendecomp lookback rows (default: 30)')
    parser.add_argument('--norm', default='zscore',
                        choices=['zscore', 'robust', 'mad', 'none'],
                        help='Normalization method (default: zscore)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        args.output,
        agg_window=args.agg_window,
        agg_stride=args.agg_stride,
        lookback=args.lookback,
        norm_method=args.norm,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
