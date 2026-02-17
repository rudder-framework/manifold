"""
Stage 25: Cohort Vector
=======================

Pivots cohort_geometry engine rows into wide cohort feature vectors.
Each (cohort, signal_0_end) gets one row with {engine}_{metric} columns.

This creates the input required by fleet stages (26-31).

Inputs:
    - cohort_geometry.parquet (from stage 03)

Outputs:
    - cohort_vector.parquet (one row per cohort per signal_0_end window)
"""

import polars as pl
from pathlib import Path

from manifold.io.writer import write_output


# Metrics to pivot from cohort_geometry (per engine)
PIVOT_METRICS = [
    'effective_dim',
    'eigenvalue_1',
    'total_variance',
    'condition_number',
    'ratio_2_1',
    'ratio_3_1',
    'eigenvalue_entropy_norm',
    'explained_1',
    'explained_2',
    'explained_3',
    'n_signals',
]


def run(
    cohort_geometry_path: str,
    data_path: str = ".",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Pivot cohort_geometry by engine into wide cohort feature vectors.

    Args:
        cohort_geometry_path: Path to cohort_geometry.parquet
        data_path: Data directory for output
        verbose: Print progress

    Returns:
        Cohort vector DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 25: COHORT VECTOR")
        print("Pivot cohort_geometry engines to wide feature vectors")
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

    # Ensure cohort column exists (single-cohort domains may omit it)
    if 'cohort' not in sg.columns:
        sg = sg.with_columns(pl.lit('').alias('cohort'))

    engines = sorted(sg['engine'].unique().to_list())
    if verbose:
        print(f"  Engines: {engines}")

    # Filter to metrics that exist in the data
    available_metrics = [m for m in PIVOT_METRICS if m in sg.columns]

    # Pivot: for each engine, create {engine}_{metric} columns
    # Use conditional aggregation via filter
    agg_exprs = []
    for engine in engines:
        for metric in available_metrics:
            col_name = f"{engine}_{metric}"
            agg_exprs.append(
                pl.col(metric).filter(pl.col('engine') == engine).first().alias(col_name)
            )

    # Pass through coordinate columns if available
    coord_passthrough = []
    for coord_col in ['signal_0_start', 'signal_0_center']:
        if coord_col in sg.columns:
            coord_passthrough.append(
                pl.col(coord_col).first().alias(coord_col)
            )

    result = sg.group_by('cohort', 'signal_0_end').agg(agg_exprs + coord_passthrough).sort('cohort', 'signal_0_end')

    # Add derived columns: eff_dim_rate (diff within cohort)
    rate_exprs = []
    for engine in engines:
        dim_col = f"{engine}_effective_dim"
        rate_col = f"{engine}_eff_dim_rate"
        if dim_col in result.columns:
            rate_exprs.append(
                (pl.col(dim_col) - pl.col(dim_col).shift(1)).over('cohort').alias(rate_col)
            )

    if rate_exprs:
        result = result.with_columns(rate_exprs)

    if verbose:
        print(f"  Cohort vector: {result.shape}")
        print(f"  Cohorts: {result['cohort'].n_unique()}, Windows: {result['signal_0_end'].n_unique()}")

    write_output(result, data_path, 'system_vector', verbose=verbose)
    return result
