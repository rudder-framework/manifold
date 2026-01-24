"""
Dynamical Systems
=================

One of the four ORTHON analytical frameworks:

    Signal Typology     → What IS this signal?
    Structural Geometry → What is its STRUCTURE?
    Dynamical Systems   → How does the SYSTEM evolve? (this framework)
    Causal Mechanics    → What DRIVES the system?

The Four Dimensions of Dynamical Analysis:
    - Regime     - What dynamical state is the system in?
    - Stability  - Is the system stable or transitioning?
    - Trajectory - Where is the system heading?
    - Attractor  - What states does it tend toward?

Key Questions:
    - Is the system in a coupled, decoupled, or transitioning regime?
    - Is the current state stable, evolving, or critical?
    - Is the trajectory converging, diverging, or oscillating?
    - Are there attractors (fixed points, limit cycles, strange attractors)?

Usage:
    >>> from prism.dynamical_systems import run_dynamical_systems, analyze_dynamics
    >>>
    >>> # Full analysis
    >>> results = run_dynamical_systems(geometry_history, timestamps)
    >>> print(results['regime'])
    >>> print(results['stability'])
    >>> print(results['engine_recommendations'])
    >>>
    >>> # Quick single analysis
    >>> result = analyze_dynamics(geometry_df)
    >>> print(result.typology.summary)

Architecture:
    dynamical_systems/
        __init__.py         # This file
        orchestrator.py     # Routes + formats
        models.py           # DynamicsVector, DynamicsTypology
        engine_mapping.py   # Engine selection
"""

__version__ = "1.0.0"
__author__ = "Ørthon Project"

# Models (dataclasses and enums)
from .models import (
    # Enums
    RegimeClass,
    StabilityClass,
    TrajectoryClass,
    AttractorClass,
    # Legacy dataclasses (one summary per entity)
    DynamicsVector,
    DynamicsTypology,
    DynamicalSystemsOutput,
    # New dataclasses (state + transitions per window)
    DynamicsState,
    DynamicsTransition,
    NUMERIC_THRESHOLDS,
)

# Orchestrator (main API)
from .orchestrator import (
    run_dynamical_systems,
    analyze_single_entity,
    detect_regime_transition,
    get_trajectory_fingerprint,
    trajectory_distance,
    DynamicalSystemsFramework,
    analyze_dynamics,
    DIMENSION_NAMES,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_regime_classification,
    get_stability_classification,
    get_trajectory_classification,
    should_escalate_to_mechanics,
    ENGINE_MAP,
    STABILITY_THRESHOLDS,
)

# Transitions
from .transitions import (
    detect_transitions,
    detect_all_transitions,
    filter_transitions_by_severity,
    filter_transitions_by_type,
    get_bifurcations,
    get_collapses,
    get_escalation_candidates,
    summarize_transitions,
)

# State computation
from .state_computation import (
    compute_state,
    compute_state_from_vector,
    compute_states_for_entity,
    classify_regime,
    classify_stability,
    classify_trajectory,
    classify_attractor,
    state_to_typology,
)

__all__ = [
    # Version
    "__version__",

    # Enums
    "RegimeClass",
    "StabilityClass",
    "TrajectoryClass",
    "AttractorClass",

    # Legacy dataclasses (one summary per entity)
    "DynamicsVector",
    "DynamicsTypology",
    "DynamicalSystemsOutput",

    # New dataclasses (state + transitions per window)
    "DynamicsState",
    "DynamicsTransition",
    "NUMERIC_THRESHOLDS",

    # Orchestrator API
    "run_dynamical_systems",
    "analyze_single_entity",
    "detect_regime_transition",
    "get_trajectory_fingerprint",
    "trajectory_distance",
    "DynamicalSystemsFramework",
    "analyze_dynamics",
    "DIMENSION_NAMES",

    # Engine mapping
    "select_engines",
    "get_regime_classification",
    "get_stability_classification",
    "get_trajectory_classification",
    "should_escalate_to_mechanics",
    "ENGINE_MAP",
    "STABILITY_THRESHOLDS",

    # Transitions
    "detect_transitions",
    "detect_all_transitions",
    "filter_transitions_by_severity",
    "filter_transitions_by_type",
    "get_bifurcations",
    "get_collapses",
    "get_escalation_candidates",
    "summarize_transitions",

    # State computation
    "compute_state",
    "compute_state_from_vector",
    "compute_states_for_entity",
    "classify_regime",
    "classify_stability",
    "classify_trajectory",
    "classify_attractor",
    "state_to_typology",
]

# Backwards compatibility
DynamicalSystemsLayer = DynamicalSystemsFramework
