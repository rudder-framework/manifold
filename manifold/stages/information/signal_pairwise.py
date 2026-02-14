"""
Stage 06: Signal Pairwise Entry Point
=====================================

Pure orchestration - calls engines/signal_pairwise.py for computation.

Inputs:
    - signal_vector.parquet
    - state_vector.parquet
    - state_geometry.parquet (optional, for eigenvector gating)

Output:
    - signal_pairwise.parquet

Computes pairwise relationships between signals:
    - Correlation
    - Distance
    - Cosine similarity
    - PC co-loading (for Granger gating)
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.signal_pairwise import compute_signal_pairwise
from manifold.io.reader import output_path as resolve_output_path


def run(
    signal_vector_path: str,
    state_vector_path: str,
    data_path: str = ".",
    state_geometry_path: Optional[str] = None,
    coloading_threshold: float = 0.1,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal pairwise computation with eigenvector gating.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        data_path: Root data directory (for write_output)
        state_geometry_path: Path to state_geometry.parquet (for PC gating)
        coloading_threshold: Threshold for PC co-loading to flag Granger
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 06: SIGNAL PAIRWISE")
        print("Pairwise relationships with eigenvector gating")
        print("=" * 70)

    out = str(resolve_output_path(data_path, 'signal_pairwise'))
    result = compute_signal_pairwise(
        signal_vector_path,
        state_vector_path,
        out,
        state_geometry_path=state_geometry_path,
        coloading_threshold=coloading_threshold,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 06: Signal Pairwise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes pairwise relationships between signals with eigenvector gating.

Uses PC co-loading to determine which pairs need Granger causality:
  - High co-loading (both load onto same PC) → run Granger
  - Low co-loading → skip expensive causality compute

Example:
  python -m engines.entry_points.stage_06_signal_pairwise \\
      signal_vector.parquet state_vector.parquet \\
      --state-geometry state_geometry.parquet \\
      -o signal_pairwise.parquet
"""
    )
    parser.add_argument('signal_vector', help='Path to signal_vector.parquet')
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('--state-geometry', help='Path to state_geometry.parquet (for PC gating)')
    parser.add_argument('--coloading-threshold', type=float, default=0.1,
                        help='Threshold for PC co-loading (default: 0.1)')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.signal_vector,
        args.state_vector,
        args.data_path,
        state_geometry_path=args.state_geometry,
        coloading_threshold=args.coloading_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
