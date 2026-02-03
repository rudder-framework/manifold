"""
Entry point wrapper for signal_vector.

Uses the manifest-driven runner from prism.signal_vector.
"""

from prism.signal_vector import (
    run_signal_vector,
    ManifestReader,
    SignalConfig,
    list_engines,
    sliding_windows,
    process_signal,
)

# Re-export old names for backwards compatibility
compute_signal_vector_temporal = run_signal_vector
compute_signal_vector_temporal_sql = run_signal_vector  # Same runner, manifest-driven

__all__ = [
    'run_signal_vector',
    'compute_signal_vector_temporal',
    'compute_signal_vector_temporal_sql',
    'ManifestReader',
    'SignalConfig',
    'list_engines',
    'sliding_windows',
    'process_signal',
]
