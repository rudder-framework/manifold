"""
Stage 05: Signal Geometry Entry Point
=====================================

Orchestration - reads parquets, calls core engine, writes output.

Inputs:
    - signal_vector.parquet
    - cohort_vector.parquet
    - cohort_geometry.parquet (optional, for principal components)

Output:
    - signal_geometry.parquet

Computes per-signal relationships to system state:
    - Distance to state centroid
    - Coherence to first principal component
    - Contribution (projection magnitude)
    - Residual (orthogonal component)
"""

import polars as pl
from typing import Optional

from manifold.core.signal_geometry import compute_signal_geometry
from manifold.io.writer import write_output


def run(
    signal_vector_path: str,
    cohort_vector_path: str,
    data_path: str = ".",
    cohort_geometry_path: Optional[str] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal geometry computation.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        cohort_vector_path: Path to cohort_vector.parquet
        data_path: Root data directory (for write_output)
        cohort_geometry_path: Path to cohort_geometry.parquet (for PCs)
        verbose: Print progress

    Returns:
        Signal geometry DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 05: SIGNAL GEOMETRY")
        print("Per-signal relationships to system state")
        print("=" * 70)

    signal_vector = pl.read_parquet(signal_vector_path)
    cohort_vector = pl.read_parquet(cohort_vector_path)

    result = compute_signal_geometry(
        signal_vector,
        cohort_vector,
        cohort_geometry_path=cohort_geometry_path,
        verbose=verbose,
    )

    write_output(result, data_path, 'signal_geometry', verbose=verbose)

    return result
