"""
PRISM Signal States
===================

Unified state-based architecture for tracking signals through four analytical layers.
Each signal is tracked through: Typology -> Geometry -> Dynamics -> Mechanics

The signal_states table provides:
- Per-window state strings for each layer
- Hash-based change detection
- Validation flags for mechanics stability
"""

from .models import SignalState
from .state_builders import (
    compute_typology_state,
    compute_geometry_state,
    compute_dynamics_state,
    compute_mechanics_state,
)
from .validation import validate_mechanics_stability, validate_cohort_alignment
from .orchestrator import compute_signal_states

__all__ = [
    "SignalState",
    "compute_typology_state",
    "compute_geometry_state",
    "compute_dynamics_state",
    "compute_mechanics_state",
    "validate_mechanics_stability",
    "validate_cohort_alignment",
    "compute_signal_states",
]
