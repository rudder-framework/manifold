"""
Manifold — dynamical systems computation engine.

Public API:
    from manifold import run
    run(observations_path, manifest_path, output_dir)

Two layers + external math:
    manifold.stages     Runners — orchestrate I/O (read parquet, call engines, write parquet)
    manifold.core       Engines — compute (DataFrames in, DataFrames out, no file I/O)
    pmtvs (external)    Math — numpy in, numbers out (pip install pmtvs)

Also:
    manifold.io         Parquet I/O (reader, writer, manifest)
    manifold.config     Configuration management (defaults, domains, environments)
    manifold.features   Feature extraction (trajectory fingerprints)
    manifold.validation Input validation (sequential signal_0, no nulls)

Five stage groups (27 stages total):
    manifold.stages.vector      Signal features (00a, 00, 01, 33)
    manifold.stages.geometry    System state (02, 03, 03b, 05, 07)
    manifold.stages.dynamics    Trajectory evolution (08, 08_lyapunov, 09a, 15, 17, 21, 22, 23, 36)
    manifold.stages.information Pairwise relationships (06, 10)
    manifold.stages.energy      Fleet analysis (25, 26, 27, 28, 30, 31, 32)
"""

from manifold.run import run

__all__ = ["run"]
