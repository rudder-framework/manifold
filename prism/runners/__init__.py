"""
PRISM Runners

Orchestration only - read manifest, call engines, write output.
"""

from .signal_vector import run_signal_vector

__all__ = ['run_signal_vector']
