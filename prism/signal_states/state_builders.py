"""
State Builders
==============

Functions to compute state strings from layer outputs.

State String Formats:
- Typology: Pipe-delimited dominant characteristics ("persistent|periodic|clustered")
- Geometry: Dot-delimited enum values ("MODULAR.STABLE.CLEAR_LEADER")
- Dynamics: Dot-delimited enum values ("COUPLED.EVOLVING.CONVERGING.FIXED_POINT")
- Mechanics: Dot-delimited enum values ("CONSERVATIVE.APPROACHING.LAMINAR.CIRCULAR")
"""

from typing import Dict, Any, List, Optional


# =============================================================================
# TYPOLOGY STATE BUILDER
# =============================================================================

def compute_typology_state(profile: Dict[str, float], threshold: float = 0.6) -> str:
    """
    Compute typology state from 9-axis profile.

    The state string contains pipe-delimited dominant characteristics
    (axes scoring above threshold).

    Args:
        profile: Dict with axis scores (0-1 normalized)
        threshold: Score above which axis is considered dominant

    Returns:
        State string like "persistent|periodic|clustered"

    Axis mappings (high score meaning):
        memory -> persistent (vs forgetful)
        information -> entropic (vs predictable)
        frequency -> periodic (vs aperiodic)
        volatility -> clustered (vs stable)
        wavelet -> multiscale (vs single_scale)
        derivatives -> spiky (vs smooth)
        recurrence -> returning (vs wandering)
        discontinuity -> step_like (vs continuous)
        momentum -> trending (vs reverting)
    """
    axis_labels = {
        'memory': 'persistent',
        'information': 'entropic',
        'frequency': 'periodic',
        'volatility': 'clustered',
        'wavelet': 'multiscale',
        'derivatives': 'spiky',
        'recurrence': 'returning',
        'discontinuity': 'step_like',
        'momentum': 'trending',
    }

    # Low score labels (for bi-directional classification)
    axis_labels_low = {
        'memory': 'forgetful',
        'information': 'predictable',
        'frequency': 'aperiodic',
        'volatility': 'stable',
        'wavelet': 'single_scale',
        'derivatives': 'smooth',
        'recurrence': 'wandering',
        'discontinuity': 'continuous',
        'momentum': 'reverting',
    }

    dominant = []
    for axis, high_label in axis_labels.items():
        score = profile.get(axis, 0.5)
        if score >= threshold:
            dominant.append(high_label)
        elif score <= (1 - threshold):
            dominant.append(axis_labels_low[axis])

    if not dominant:
        # All axes are moderate - use the highest scoring axis
        best_axis = max(axis_labels.keys(), key=lambda a: abs(profile.get(a, 0.5) - 0.5))
        best_score = profile.get(best_axis, 0.5)
        if best_score >= 0.5:
            dominant.append(axis_labels[best_axis])
        else:
            dominant.append(axis_labels_low[best_axis])

    return "|".join(sorted(dominant))


# =============================================================================
# GEOMETRY STATE BUILDER
# =============================================================================

def compute_geometry_state(
    topology_class: str,
    stability_class: str,
    leadership_class: str
) -> str:
    """
    Compute geometry state from structural geometry classifications.

    Args:
        topology_class: TopologyClass value (e.g., "modular", "hierarchical")
        stability_class: ManifoldStabilityClass value (e.g., "stable", "weakening")
        leadership_class: LeadershipClass value (e.g., "clear_leader", "bidirectional")

    Returns:
        State string like "MODULAR.STABLE.CLEAR_LEADER"
    """
    parts = [
        topology_class.upper() if topology_class else "SPARSE",
        stability_class.upper() if stability_class else "STABLE",
        leadership_class.upper() if leadership_class else "CONTEMPORANEOUS",
    ]
    return ".".join(parts)


def geometry_state_from_output(output: Dict[str, Any]) -> str:
    """
    Extract geometry state from StructuralGeometryOutput dict.

    Args:
        output: Dict from StructuralGeometryOutput.to_dict()

    Returns:
        State string
    """
    return compute_geometry_state(
        output.get('topology_class', 'sparse'),
        output.get('stability_class', 'stable'),
        output.get('leadership_class', 'contemporaneous'),
    )


# =============================================================================
# DYNAMICS STATE BUILDER
# =============================================================================

def compute_dynamics_state(
    regime_class: str,
    stability_class: str,
    trajectory_class: str,
    attractor_class: str
) -> str:
    """
    Compute dynamics state from dynamical systems classifications.

    Args:
        regime_class: RegimeClass value (e.g., "coupled", "decoupled")
        stability_class: DynamicsStabilityClass value (e.g., "stable", "evolving")
        trajectory_class: TrajectoryClass value (e.g., "converging", "diverging")
        attractor_class: AttractorClass value (e.g., "fixed_point", "limit_cycle")

    Returns:
        State string like "COUPLED.EVOLVING.CONVERGING.FIXED_POINT"
    """
    parts = [
        regime_class.upper() if regime_class else "MODERATE",
        stability_class.upper() if stability_class else "STABLE",
        trajectory_class.upper() if trajectory_class else "WANDERING",
        attractor_class.upper() if attractor_class else "NONE",
    ]
    return ".".join(parts)


def dynamics_state_from_output(output: Dict[str, Any]) -> str:
    """
    Extract dynamics state from DynamicalSystemsOutput dict.

    Args:
        output: Dict from DynamicalSystemsOutput.to_dict()

    Returns:
        State string
    """
    return compute_dynamics_state(
        output.get('regime_class', 'moderate'),
        output.get('stability_class', 'stable'),
        output.get('trajectory_class', 'wandering'),
        output.get('attractor_class', 'none'),
    )


# =============================================================================
# MECHANICS STATE BUILDER
# =============================================================================

def compute_mechanics_state(
    energy_class: str,
    equilibrium_class: str,
    flow_class: str,
    orbit_class: str
) -> str:
    """
    Compute mechanics state from causal mechanics classifications.

    Args:
        energy_class: EnergyClass value (e.g., "conservative", "driven")
        equilibrium_class: EquilibriumClass value (e.g., "approaching", "departing")
        flow_class: FlowClass value (e.g., "laminar", "turbulent")
        orbit_class: OrbitClass value (e.g., "circular", "irregular")

    Returns:
        State string like "CONSERVATIVE.APPROACHING.LAMINAR.CIRCULAR"
    """
    parts = [
        energy_class.upper() if energy_class else "CONSERVATIVE",
        equilibrium_class.upper() if equilibrium_class else "AT_EQUILIBRIUM",
        flow_class.upper() if flow_class else "LAMINAR",
        orbit_class.upper() if orbit_class else "LINEAR",
    ]
    return ".".join(parts)


def mechanics_state_from_output(output: Dict[str, Any]) -> str:
    """
    Extract mechanics state from CausalMechanicsOutput dict.

    Args:
        output: Dict from CausalMechanicsOutput.to_dict()

    Returns:
        State string
    """
    return compute_mechanics_state(
        output.get('energy_class', 'conservative'),
        output.get('equilibrium_class', 'at_equilibrium'),
        output.get('flow_class', 'laminar'),
        output.get('orbit_class', 'linear'),
    )


# =============================================================================
# STATE PARSING
# =============================================================================

def parse_typology_state(state: str) -> List[str]:
    """
    Parse typology state string into list of characteristics.

    Args:
        state: State string like "persistent|periodic|clustered"

    Returns:
        List of characteristics
    """
    if not state:
        return []
    return state.split("|")


def parse_dotted_state(state: str) -> List[str]:
    """
    Parse dot-delimited state string into list of values.

    Args:
        state: State string like "MODULAR.STABLE.CLEAR_LEADER"

    Returns:
        List of values
    """
    if not state:
        return []
    return state.split(".")


def states_match(state1: str, state2: str) -> bool:
    """
    Check if two state strings are equivalent.

    Args:
        state1: First state string
        state2: Second state string

    Returns:
        True if states are identical
    """
    return state1 == state2


def compute_state_distance(state1: str, state2: str) -> int:
    """
    Compute edit distance between two state strings.

    For dot-delimited states, counts number of differing components.

    Args:
        state1: First state string
        state2: Second state string

    Returns:
        Number of differing components
    """
    if "|" in state1 or "|" in state2:
        # Typology states - use set difference
        set1 = set(parse_typology_state(state1))
        set2 = set(parse_typology_state(state2))
        return len(set1.symmetric_difference(set2))
    else:
        # Dotted states - count differing positions
        parts1 = parse_dotted_state(state1)
        parts2 = parse_dotted_state(state2)
        # Pad to same length
        max_len = max(len(parts1), len(parts2))
        parts1 = parts1 + [""] * (max_len - len(parts1))
        parts2 = parts2 + [""] * (max_len - len(parts2))
        return sum(1 for p1, p2 in zip(parts1, parts2) if p1 != p2)
