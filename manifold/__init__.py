"""
Manifold — dynamical systems computation engine.

Public API:
    from manifold import run
    run(observations_path, manifest_path, output_dir)

Three layers:
    manifold.core       Compute engines (DataFrames → DataFrames)
    manifold.primitives Pure math (numpy → float)
    manifold.io         Parquet I/O (reader, writer, manifest)

Five stage groups:
    manifold.stages.vector      Signal features
    manifold.stages.geometry    System state
    manifold.stages.dynamics    Trajectory evolution
    manifold.stages.information Pairwise relationships
    manifold.stages.energy      Fleet analysis
"""

from manifold.run import run

__all__ = ["run"]
