"""
Stage 07: Geometry Dynamics Entry Point
=======================================

Pure orchestration - calls engines/geometry_dynamics.py for computation.

Inputs:
    - state_geometry.parquet

Output:
    - geometry_dynamics.parquet

Computes differential geometry of state evolution:
    - d1: velocity (first derivative)
    - d2: acceleration (second derivative)
    - d3: jerk (third derivative)
    - Curvature metrics
    - Collapse indicators
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.geometry_dynamics import compute_geometry_dynamics
from manifold.io.reader import output_path as resolve_output_path


def run(
    state_geometry_path: str,
    data_path: str = ".",
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run geometry dynamics computation.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        data_path: Root data directory (for write_output)
        dt: Time step for derivatives (from config if None)
        smooth_window: Smoothing window (from config if None)
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 07: GEOMETRY DYNAMICS")
        print("Differential geometry of state evolution (d1/d2/d3)")
        print("=" * 70)

    out = str(resolve_output_path(data_path, 'geometry_dynamics'))
    result = compute_geometry_dynamics(
        state_geometry_path,
        out,
        dt=dt,
        smooth_window=smooth_window,
        verbose=verbose,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 07: Geometry Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes differential geometry of state evolution:
  - d1 (velocity): rate of change of geometry
  - d2 (acceleration): rate of change of velocity
  - d3 (jerk): rate of change of acceleration
  - Collapse indicators: sustained loss of degrees of freedom

Example:
  python -m engines.entry_points.stage_07_geometry_dynamics \\
      state_geometry.parquet -o geometry_dynamics.parquet
"""
    )
    parser.add_argument('state_geometry', help='Path to state_geometry.parquet')
    parser.add_argument('--dt', type=float, help='Time step for derivatives')
    parser.add_argument('--smooth-window', type=int, help='Smoothing window')
    parser.add_argument('-o', '--output', default='geometry_dynamics.parquet',
                        help='Output path (default: geometry_dynamics.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_geometry,
        args.output,
        dt=args.dt,
        smooth_window=args.smooth_window,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
