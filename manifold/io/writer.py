"""
Writer — all parquet writes go through here.

No other module should call df.write_parquet directly.
"""

import polars as pl
from pathlib import Path

from manifold.io.reader import STAGE_DIRS


def write_output(
    df: pl.DataFrame,
    data_path: str,
    name: str,
    verbose: bool = True,
) -> Path:
    """
    Write a stage output to the correct subdirectory.

    Args:
        df: DataFrame to write
        data_path: Root data directory (e.g., domains/rossler)
        name: Output name (e.g., 'signal_vector', 'state_geometry')
        verbose: Print path on write

    Returns:
        Path to written file
    """
    from manifold.io.reader import output_path
    path = output_path(data_path, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))

    if verbose:
        print(f"  -> {path} ({len(df)} rows)")

    return path


def write_sidecar(
    df: pl.DataFrame,
    data_path: str,
    parent_name: str,
    sidecar_suffix: str,
    verbose: bool = True,
) -> Path:
    """
    Write a sidecar file alongside its parent.

    Example: write_sidecar(df, path, 'state_geometry', 'loadings')
    -> output/2_system_state/state_geometry_loadings.parquet
    """
    from manifold.io.reader import output_path
    parent_path = output_path(data_path, parent_name)
    sidecar_path = parent_path.parent / f"{parent_name}_{sidecar_suffix}.parquet"
    df.write_parquet(str(sidecar_path))

    if verbose:
        print(f"  -> {sidecar_path} ({len(df)} rows, sidecar)")

    return sidecar_path
