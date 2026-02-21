"""
Writer — all parquet writes go through here.

No other module should call df.write_parquet directly.
"""

import polars as pl
from pathlib import Path

from manifold.io.reader import STAGE_DIRS


def _safe_write(df: pl.DataFrame, path: Path, verbose: bool = True, metadata: dict = None) -> bool:
    """
    Guard against writing invalid parquet files.

    Returns True if a file was written, False if skipped.
    """
    if df is None:
        return False

    if len(df.columns) == 0:
        if verbose:
            print(f"  !! Skipped {path} (empty schema — 0 columns)")
        return False

    kw = {"metadata": metadata} if metadata else {}

    if df.height == 0:
        # Schema-only parquet: columns defined, 0 rows.
        # DuckDB/Polars can read this; downstream SQL won't crash.
        df.head(0).write_parquet(str(path), **kw)
        return True

    df.write_parquet(str(path), **kw)
    return True


def write_output(
    df: pl.DataFrame,
    data_path: str,
    name: str,
    verbose: bool = True,
    metadata: dict = None,
) -> Path:
    """
    Write a stage output to the correct subdirectory.

    Args:
        df: DataFrame to write (None or empty-schema → skip)
        data_path: Root data directory (e.g., domains/rossler)
        name: Output name (e.g., 'signal_vector', 'cohort_geometry')
        verbose: Print path on write
        metadata: Optional Parquet file-level key-value metadata

    Returns:
        Path to written file, or None if skipped
    """
    from manifold.io.reader import output_path

    path = output_path(data_path, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not _safe_write(df, path, verbose=verbose, metadata=metadata):
        return None

    if verbose:
        print(f"  -> {path} ({len(df)} rows)")

    return path


def write_sidecar(
    df: pl.DataFrame,
    data_path: str,
    parent_name: str,
    sidecar_suffix: str,
    verbose: bool = True,
    metadata: dict = None,
) -> Path:
    """
    Write a sidecar file alongside its parent.

    Example: write_sidecar(df, path, 'velocity_field', 'components')
    -> output/cohort/cohort_dynamics/velocity_field_components.parquet
    """
    from manifold.io.reader import output_path

    sidecar_name = f"{parent_name}_{sidecar_suffix}"

    parent_path = output_path(data_path, parent_name)
    sidecar_path = parent_path.parent / f"{sidecar_name}.parquet"

    if not _safe_write(df, sidecar_path, verbose=verbose, metadata=metadata):
        return None

    if verbose:
        print(f"  -> {sidecar_path} ({len(df)} rows, sidecar)")

    return sidecar_path
