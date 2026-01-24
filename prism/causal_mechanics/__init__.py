"""
Causal Mechanics
================

One of the four ORTHON analytical frameworks:

    Signal Typology     → What IS this signal?
    Structural Geometry → What is its STRUCTURE?
    Dynamical Systems   → How does the SYSTEM evolve?
    Causal Mechanics    → What DRIVES the system? (this framework)

The Five Physics-Inspired Dimensions:
    - Energy       - Is energy conserved? (Hamiltonian)
    - Motion       - What are the equations of motion? (Lagrangian)
    - Equilibrium  - Spontaneous or forced? (Gibbs free energy)
    - Cycles       - What are the rotational dynamics? (Angular momentum)
    - Flow         - How does momentum propagate? (Momentum flux)

Key Questions:
    - Is the system conservative, driven, or dissipative?
    - Is it spontaneously equilibrating or externally forced?
    - Is flow laminar, transitional, or turbulent?
    - What is the energy dominance (kinetic vs potential)?

Usage:
    >>> from prism.causal_mechanics import run_causal_mechanics, analyze_mechanics
    >>>
    >>> # Full analysis from signal
    >>> results = run_causal_mechanics(signal_data, entity_id='unit_1')
    >>> print(results['energy_class'])
    >>> print(results['system_class'])
    >>> print(results['engine_recommendations'])
    >>>
    >>> # Quick single analysis
    >>> result = analyze_mechanics(my_array)
    >>> print(result.typology.summary)

Architecture:
    causal_mechanics/
        __init__.py         # This file
        orchestrator.py     # Routes + formats
        models.py           # MechanicsVector, MechanicsTypology
        engine_mapping.py   # Engine selection
"""

__version__ = "1.0.0"
__author__ = "Ørthon Project"

# Models (dataclasses and enums)
from .models import (
    # Enums
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
    # Legacy dataclasses (one summary per signal)
    MechanicsVector,
    MechanicsTypology,
    CausalMechanicsOutput,
    # New dataclasses (state + transitions per window)
    MechanicsState,
    MechanicsTransition,
    MECHANICS_THRESHOLDS,
)

# Orchestrator (main API)
from .orchestrator import (
    run_causal_mechanics,
    analyze_single_signal,
    detect_energy_transition,
    get_mechanics_fingerprint,
    mechanics_distance,
    classify_system,
    CausalMechanicsFramework,
    analyze_mechanics,
    DIMENSION_NAMES,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_energy_classification,
    get_equilibrium_classification,
    get_flow_classification,
    get_intervention_recommendations,
    ENGINE_MAP,
    ENERGY_THRESHOLDS,
    INTERVENTION_MAP,
)

# Transitions
from .transitions import (
    detect_transitions as detect_mechanics_transitions,
    detect_all_transitions as detect_all_mechanics_transitions,
    filter_transitions_by_severity as filter_mechanics_by_severity,
    filter_transitions_by_type as filter_mechanics_by_type,
    get_turbulence_events,
    get_energy_events,
    get_critical_transitions,
    validate_mechanics_stability,
    summarize_transitions as summarize_mechanics_transitions,
)

# State computation
from .state_computation import (
    compute_state as compute_mechanics_state,
    compute_state_from_vector as compute_mechanics_state_from_vector,
    compute_states_for_signal,
    classify_energy,
    classify_equilibrium,
    classify_flow,
    classify_orbit,
    state_to_typology as mechanics_state_to_typology,
    validate_state as validate_mechanics_state,
)

__all__ = [
    # Version
    "__version__",

    # Enums
    "EnergyClass",
    "EquilibriumClass",
    "FlowClass",
    "OrbitClass",
    "DominanceClass",

    # Legacy dataclasses (one summary per signal)
    "MechanicsVector",
    "MechanicsTypology",
    "CausalMechanicsOutput",

    # New dataclasses (state + transitions per window)
    "MechanicsState",
    "MechanicsTransition",
    "MECHANICS_THRESHOLDS",

    # Orchestrator API
    "run_causal_mechanics",
    "analyze_single_signal",
    "detect_energy_transition",
    "get_mechanics_fingerprint",
    "mechanics_distance",
    "classify_system",
    "CausalMechanicsFramework",
    "analyze_mechanics",
    "DIMENSION_NAMES",

    # Engine mapping
    "select_engines",
    "get_energy_classification",
    "get_equilibrium_classification",
    "get_flow_classification",
    "get_intervention_recommendations",
    "ENGINE_MAP",
    "ENERGY_THRESHOLDS",
    "INTERVENTION_MAP",

    # Transitions
    "detect_mechanics_transitions",
    "detect_all_mechanics_transitions",
    "filter_mechanics_by_severity",
    "filter_mechanics_by_type",
    "get_turbulence_events",
    "get_energy_events",
    "get_critical_transitions",
    "validate_mechanics_stability",
    "summarize_mechanics_transitions",

    # State computation
    "compute_mechanics_state",
    "compute_mechanics_state_from_vector",
    "compute_states_for_signal",
    "classify_energy",
    "classify_equilibrium",
    "classify_flow",
    "classify_orbit",
    "mechanics_state_to_typology",
    "validate_mechanics_state",
]

# Backwards compatibility
CausalMechanicsLayer = CausalMechanicsFramework
SystemPhysicsLayer = CausalMechanicsFramework
