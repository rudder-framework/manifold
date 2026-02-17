"""
Fleet Pivot Helper — pivot cohort_geometry to wide per-cohort rows.

Shared by stages 26-31 which need per-cohort rows with {engine}_{metric} columns.
Pure computation — DataFrame in, DataFrame out. No file I/O.
"""

import polars as pl

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


def pivot_cohort_geometry(cohort_geometry: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot long-format cohort_geometry (one row per cohort x engine x window)
    into wide format (one row per cohort x window with {engine}_{metric} columns).

    Also adds derived eff_dim_rate columns.

    Args:
        cohort_geometry: DataFrame from cohort_geometry.parquet

    Returns:
        Wide-format DataFrame with one row per (cohort, signal_0_end)
    """
    sg = cohort_geometry

    if len(sg) == 0 or 'engine' not in sg.columns:
        return pl.DataFrame()

    # Ensure cohort column exists (single-cohort domains may omit it)
    if 'cohort' not in sg.columns:
        sg = sg.with_columns(pl.lit('').alias('cohort'))

    engines = sorted(sg['engine'].unique().to_list())
    available_metrics = [m for m in PIVOT_METRICS if m in sg.columns]

    # Pivot: for each engine, create {engine}_{metric} columns
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

    result = (
        sg.group_by('cohort', 'signal_0_end')
        .agg(agg_exprs + coord_passthrough)
        .sort('cohort', 'signal_0_end')
    )

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

    return result
