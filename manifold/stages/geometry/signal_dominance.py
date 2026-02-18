"""
Stage 03b: Signal Dominance
============================

Ranks signals by PC1 loading magnitude per feature engine.
Derived from cohort_signal_positions.parquet (stage 03 sidecar).

Input:
    - cohort_signal_positions.parquet

Output:
    - signal_dominance.parquet (parameterization/)
"""

import polars as pl
from pathlib import Path

from manifold.io.writer import write_output


def run(
    cohort_signal_positions_path: str,
    data_path: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Rank signals by PC1 loading magnitude.

    Args:
        cohort_signal_positions_path: Path to cohort_signal_positions.parquet
        data_path: Root data directory (for write_output)
        verbose: Print progress

    Returns:
        Signal dominance DataFrame
    """
    path = Path(cohort_signal_positions_path)
    if not path.exists():
        if verbose:
            print("  Skipped (cohort_signal_positions.parquet not found)")
        write_output(pl.DataFrame(), data_path, 'signal_dominance', verbose=verbose)
        return pl.DataFrame()

    positions = pl.read_parquet(cohort_signal_positions_path)
    if positions.is_empty():
        if verbose:
            print("  Skipped (cohort_signal_positions.parquet is empty)")
        write_output(pl.DataFrame(), data_path, 'signal_dominance', verbose=verbose)
        return pl.DataFrame()

    # Per-engine aggregation: group by (engine, signal_id)
    per_engine = (
        positions
        .with_columns(pl.col('pc1_loading').abs().alias('_abs_pc1'))
        .group_by(['engine', 'signal_id'])
        .agg([
            pl.col('_abs_pc1').mean().alias('mean_abs_pc1_loading'),
            pl.col('_abs_pc1').sort_by('signal_0_end').last().alias('final_abs_pc1_loading'),
            pl.col('pc1_loading').mean().sign().cast(pl.Int8).alias('pc1_loading_sign'),
            pl.col('pc1_loading').count().cast(pl.UInt32).alias('n_windows'),
        ])
    )

    # Rank within each engine by mean_abs_pc1_loading descending
    per_engine = per_engine.with_columns(
        pl.col('mean_abs_pc1_loading')
        .rank(method='ordinal', descending=True)
        .over('engine')
        .cast(pl.UInt32)
        .alias('dominance_rank')
    )

    # Aggregate rows: average across engines per signal
    agg = (
        per_engine
        .group_by('signal_id')
        .agg([
            pl.col('mean_abs_pc1_loading').mean(),
            pl.col('final_abs_pc1_loading').sort_by('n_windows').last(),
            pl.col('pc1_loading_sign').sort_by('n_windows').last(),
            pl.col('n_windows').sum(),
        ])
        .with_columns(pl.lit('aggregate').alias('engine'))
        .with_columns(
            pl.col('mean_abs_pc1_loading')
            .rank(method='ordinal', descending=True)
            .cast(pl.UInt32)
            .alias('dominance_rank')
        )
    )

    # Concatenate and sort
    result = (
        pl.concat([per_engine, agg], how='diagonal_relaxed')
        .select([
            'engine', 'signal_id', 'mean_abs_pc1_loading',
            'final_abs_pc1_loading', 'dominance_rank', 'pc1_loading_sign', 'n_windows',
        ])
        .sort(['engine', 'dominance_rank'])
    )

    write_output(result, data_path, 'signal_dominance', verbose=verbose)
    return result
