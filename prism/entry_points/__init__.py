"""
PRISM Entry Points

Runners that orchestrate pipeline stages.

- pipeline.py: Full pipeline orchestrator (python -m prism)
- signal_vector.py: Wrapper for prism.signal_vector (manifest-driven)
- sql_runner.py: Executes SQL-based computations
"""

from prism.entry_points.signal_vector import (
    run_signal_vector,
    compute_signal_vector_temporal,
    compute_signal_vector_temporal_sql,
    ManifestReader,
)
from prism.entry_points.sql_runner import SQLRunner
from prism.entry_points.pipeline import main, run_full_pipeline

__all__ = [
    'main',
    'run_full_pipeline',
    'run_signal_vector',
    'compute_signal_vector_temporal',
    'compute_signal_vector_temporal_sql',
    'ManifestReader',
    'SQLRunner',
]
