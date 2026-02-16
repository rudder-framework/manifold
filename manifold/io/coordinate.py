"""
Coordinate utilities — translate I (sequential index) to physical coordinates.

When a manifest contains a `coordinate` block, every output parquet gets
coordinate columns (e.g., `t`, `t_start`, `t_center`) added transparently
by the writer layer. No engine or stage changes required.

Supported source types:
    sample_rate — coord = I / sample_rate  (pure arithmetic)
    column      — lookup from observations  (join on cohort + I)
"""

import polars as pl
from typing import Any, Dict, Optional


# Stages that represent point events (no window concept).
# These get only {col}, not {col}_start or {col}_center.
POINT_EVENT_STAGES = {'breaks', 'observation_geometry', 'cohort_baseline'}


def get_coordinate_config(
    manifest: Dict[str, Any],
    observations_path: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse coordinate block from manifest and build a ready-to-use config.

    Args:
        manifest: Parsed manifest dict (must contain 'coordinate' key to activate).
        observations_path: Path to observations.parquet (required for source=column).

    Returns:
        Config dict with keys:
            column, unit, label, source, sample_rate, system_window
            lookup (pl.DataFrame) — only for source=column
        or None if no coordinate block in manifest.
    """
    coord_block = manifest.get('coordinate')
    if not coord_block:
        return None

    source = coord_block.get('source')
    column = coord_block.get('column')

    if not source:
        raise ValueError("coordinate block requires 'source' (sample_rate or column)")
    if not column:
        raise ValueError("coordinate block requires 'column' (output column name)")

    config = {
        'column': column,
        'unit': coord_block.get('unit', ''),
        'label': coord_block.get('label', column),
        'source': source,
        'system_window': manifest.get('system', {}).get('window', 1),
    }

    if source == 'sample_rate':
        sample_rate = coord_block.get('sample_rate')
        if not sample_rate:
            raise ValueError("coordinate source=sample_rate requires 'sample_rate'")
        config['sample_rate'] = float(sample_rate)

    elif source == 'column':
        source_column = coord_block.get('source_column', column)
        if not observations_path:
            raise ValueError("coordinate source=column requires observations_path")
        obs = pl.read_parquet(observations_path)
        if source_column not in obs.columns:
            raise ValueError(
                f"coordinate source_column '{source_column}' not in observations "
                f"(available: {obs.columns})"
            )
        # Build (cohort, I) -> coord_value lookup
        group_cols = ['I']
        if 'cohort' in obs.columns:
            group_cols = ['cohort', 'I']
        lookup = (
            obs.group_by(group_cols)
            .agg(pl.col(source_column).first().alias('_coord_value'))
        )
        config['lookup'] = lookup
        config['source_column'] = source_column

    else:
        raise ValueError(f"Unknown coordinate source: '{source}' (expected sample_rate or column)")

    return config


def tag_coordinates(
    df: pl.DataFrame,
    coord_config: Dict[str, Any],
    stage_name: str = None,
) -> pl.DataFrame:
    """
    Add coordinate columns to a DataFrame that has an I column.

    For windowed stages: adds {col}, {col}_start, {col}_center.
    For point-event stages: adds {col} only.

    Args:
        df: DataFrame with I column.
        coord_config: Config dict from get_coordinate_config().
        stage_name: Name of the stage (used to detect point-event stages).

    Returns:
        New DataFrame with coordinate columns added.
    """
    if len(df) == 0 or 'I' not in df.columns:
        return df

    col = coord_config['column']
    source = coord_config['source']
    system_window = coord_config['system_window']
    is_point_event = stage_name in POINT_EVENT_STAGES

    if source == 'sample_rate':
        rate = coord_config['sample_rate']
        # Always add the main coordinate column
        df = df.with_columns(
            (pl.col('I').cast(pl.Float64) / rate).alias(col)
        )

        if not is_point_event:
            # Compute window_start_I: use explicit column if available, else derive
            if 'window_start_I' in df.columns:
                start_expr = pl.col('window_start_I').cast(pl.Float64) / rate
            else:
                start_expr = (
                    (pl.col('I').cast(pl.Float64) - system_window + 1)
                    .clip(lower_bound=0)
                    / rate
                )
            center_I_expr = (
                (pl.col('I').cast(pl.Float64) + start_expr * rate) / 2.0 / rate
            )
            # Simpler: center = (start + end) / 2
            df = df.with_columns(
                start_expr.alias(f'{col}_start'),
            )
            df = df.with_columns(
                ((pl.col(f'{col}_start') + pl.col(col)) / 2.0).alias(f'{col}_center'),
            )

    elif source == 'column':
        lookup = coord_config['lookup']
        has_cohort = 'cohort' in df.columns and 'cohort' in lookup.columns

        join_on = ['cohort', 'I'] if has_cohort else ['I']
        df = df.join(lookup, on=join_on, how='left')
        df = df.rename({'_coord_value': col})

        if not is_point_event:
            if 'window_start_I' in df.columns:
                # Join again on window_start_I to get start coordinate
                start_lookup = lookup.rename({'I': '_start_I', '_coord_value': f'{col}_start'})
                start_join_on = (['cohort', '_start_I'] if has_cohort
                                 else ['_start_I'])
                df = df.with_columns(
                    pl.col('window_start_I').alias('_start_I')
                )
                df = df.join(start_lookup, left_on=start_join_on if not has_cohort else ['cohort', '_start_I'],
                             right_on=start_join_on, how='left')
                df = df.drop('_start_I')
            else:
                # Derive window_start_I from I and system_window
                df = df.with_columns(
                    (pl.col('I') - system_window + 1).clip(lower_bound=0).alias('_start_I')
                )
                start_lookup = lookup.rename({'I': '_start_I', '_coord_value': f'{col}_start'})
                start_join_on = ['cohort', '_start_I'] if has_cohort else ['_start_I']
                df = df.join(start_lookup, on=start_join_on, how='left')
                df = df.drop('_start_I')

            df = df.with_columns(
                ((pl.col(f'{col}_start') + pl.col(col)) / 2.0).alias(f'{col}_center'),
            )

    return df
