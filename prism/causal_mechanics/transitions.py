"""
Causal Mechanics Transition Detection
=====================================

Detects meaningful transitions between consecutive MechanicsState windows.

Transition detection follows these rules:

1. Categorical Fields (energy, equilibrium, flow, orbit):
   - Any change emits a transition

2. Numeric Fields (energy_conservation, equilibrium_distance, turbulence_intensity, orbit_stability):
   - Change must exceed threshold to emit transition

3. Severity Classification:
   - mild: Small change or minor categorical shift
   - moderate: Significant change or categorical change
   - severe: Major categorical flip or extreme change

4. Transition Types:
   - energy_injection: Energy regime changed to driven
   - energy_dissipation: Energy regime changed to dissipative
   - equilibrium_departure: System moved away from equilibrium
   - equilibrium_approach: System moved toward equilibrium
   - turbulence_onset: Flow became turbulent
   - laminarization: Flow became laminar
   - orbit_destabilization: Orbit became irregular
   - orbit_stabilization: Orbit became stable
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from prism.causal_mechanics.models import (
    MechanicsState,
    MechanicsTransition,
    MECHANICS_THRESHOLDS,
)


def detect_transitions(
    current: MechanicsState,
    previous: MechanicsState,
    thresholds: Optional[Dict[str, float]] = None
) -> List[MechanicsTransition]:
    """
    Detect transitions between two consecutive MechanicsState windows.

    Args:
        current: Current window state
        previous: Previous window state
        thresholds: Optional custom thresholds

    Returns:
        List of MechanicsTransition objects for meaningful changes
    """
    if thresholds is None:
        thresholds = MECHANICS_THRESHOLDS

    transitions = []

    # Check categorical fields
    categorical_fields = [
        ("energy", _get_energy_transition_type),
        ("equilibrium", _get_equilibrium_transition_type),
        ("flow", _get_flow_transition_type),
        ("orbit", _get_orbit_transition_type),
    ]

    for field, type_func in categorical_fields:
        current_val = getattr(current, field, None)
        previous_val = getattr(previous, field, None)

        if current_val != previous_val:
            transition_type, severity = type_func(previous_val, current_val)
            trans = MechanicsTransition(
                entity_id=current.entity_id,
                unit_id=current.unit_id,
                signal_id=current.signal_id,
                window_idx=current.window_idx,
                timestamp=current.timestamp,
                field=field,
                from_value=str(previous_val),
                to_value=str(current_val),
                delta=None,
                transition_type=transition_type,
                severity=severity,
            )
            transitions.append(trans)

    # Check numeric fields
    numeric_fields = [
        "energy_conservation",
        "equilibrium_distance",
        "turbulence_intensity",
        "orbit_stability",
    ]

    for field in numeric_fields:
        current_val = getattr(current, field, 0.5)
        previous_val = getattr(previous, field, 0.5)

        threshold = thresholds.get(field, 0.15)
        delta = abs(current_val - previous_val)

        if delta > threshold:
            transition_type = _get_numeric_transition_type(
                field, previous_val, current_val
            )
            severity = _classify_numeric_severity(delta, threshold)

            trans = MechanicsTransition(
                entity_id=current.entity_id,
                unit_id=current.unit_id,
                signal_id=current.signal_id,
                window_idx=current.window_idx,
                timestamp=current.timestamp,
                field=field,
                from_value=f"{previous_val:.4f}",
                to_value=f"{current_val:.4f}",
                delta=delta,
                transition_type=transition_type,
                severity=severity,
            )
            transitions.append(trans)

    return transitions


def _get_energy_transition_type(from_val: str, to_val: str) -> tuple:
    """Determine transition type and severity for energy changes."""

    # Major transitions
    if from_val == "conservative" and to_val == "driven":
        return "energy_injection", "moderate"
    elif from_val == "conservative" and to_val == "dissipative":
        return "energy_dissipation", "moderate"
    elif from_val == "conservative" and to_val == "fluctuating":
        return "energy_destabilization", "severe"
    elif to_val == "conservative" and from_val in ["driven", "dissipative"]:
        return "energy_stabilization", "mild"
    elif to_val == "fluctuating":
        return "energy_destabilization", "moderate"
    elif from_val == "fluctuating":
        return "energy_stabilization", "mild"
    else:
        return "energy_shift", "mild"


def _get_equilibrium_transition_type(from_val: str, to_val: str) -> tuple:
    """Determine transition type and severity for equilibrium changes."""

    if from_val == "at_equilibrium" and to_val == "departing":
        return "equilibrium_departure", "moderate"
    elif from_val == "at_equilibrium" and to_val == "forced":
        return "equilibrium_forcing", "severe"
    elif to_val == "at_equilibrium":
        return "equilibrium_approach", "mild"
    elif from_val == "approaching" and to_val == "departing":
        return "equilibrium_reversal", "moderate"
    elif to_val == "forced":
        return "equilibrium_forcing", "moderate"
    else:
        return "equilibrium_shift", "mild"


def _get_flow_transition_type(from_val: str, to_val: str) -> tuple:
    """Determine transition type and severity for flow changes."""

    if from_val == "laminar" and to_val == "turbulent":
        return "turbulence_onset", "severe"
    elif from_val == "turbulent" and to_val == "laminar":
        return "laminarization", "moderate"
    elif to_val == "turbulent":
        return "turbulence_onset", "moderate"
    elif to_val == "laminar":
        return "laminarization", "mild"
    else:
        return "flow_transition", "mild"


def _get_orbit_transition_type(from_val: str, to_val: str) -> tuple:
    """Determine transition type and severity for orbit changes."""

    if from_val in ["circular", "elliptical"] and to_val == "irregular":
        return "orbit_destabilization", "moderate"
    elif from_val == "irregular" and to_val in ["circular", "elliptical"]:
        return "orbit_stabilization", "mild"
    elif from_val == "circular" and to_val == "linear":
        return "orbit_collapse", "moderate"
    elif from_val == "linear" and to_val in ["circular", "elliptical"]:
        return "orbit_formation", "mild"
    else:
        return "orbit_shift", "mild"


def _get_numeric_transition_type(
    field: str,
    from_val: float,
    to_val: float
) -> str:
    """Determine transition type for numeric field changes."""

    if field == "energy_conservation":
        if to_val < from_val:
            return "energy_destabilization"
        else:
            return "energy_stabilization"

    elif field == "equilibrium_distance":
        if to_val > from_val:
            return "equilibrium_departure"
        else:
            return "equilibrium_approach"

    elif field == "turbulence_intensity":
        if to_val > from_val:
            return "turbulence_increase"
        else:
            return "turbulence_decrease"

    elif field == "orbit_stability":
        if to_val < from_val:
            return "orbit_destabilization"
        else:
            return "orbit_stabilization"

    return "shift"


def _classify_numeric_severity(delta: float, threshold: float) -> str:
    """Classify severity based on delta relative to threshold."""
    if delta > 3 * threshold:
        return "severe"
    elif delta > 2 * threshold:
        return "moderate"
    else:
        return "mild"


def detect_all_transitions(
    states: List[MechanicsState],
    thresholds: Optional[Dict[str, float]] = None
) -> List[MechanicsTransition]:
    """
    Detect all transitions across a sequence of states.

    Args:
        states: List of MechanicsState objects, sorted by window_idx
        thresholds: Optional custom thresholds

    Returns:
        List of all MechanicsTransition objects
    """
    all_transitions = []

    for i in range(1, len(states)):
        transitions = detect_transitions(states[i], states[i-1], thresholds)
        all_transitions.extend(transitions)

    return all_transitions


def filter_transitions_by_severity(
    transitions: List[MechanicsTransition],
    min_severity: str = "mild"
) -> List[MechanicsTransition]:
    """
    Filter transitions by minimum severity.

    Args:
        transitions: List of transitions
        min_severity: Minimum severity to include ('mild', 'moderate', 'severe')

    Returns:
        Filtered list of transitions
    """
    severity_levels = {"mild": 0, "moderate": 1, "severe": 2}
    min_level = severity_levels.get(min_severity, 0)

    return [
        t for t in transitions
        if severity_levels.get(t.severity, 0) >= min_level
    ]


def filter_transitions_by_type(
    transitions: List[MechanicsTransition],
    transition_types: List[str]
) -> List[MechanicsTransition]:
    """
    Filter transitions by type.

    Args:
        transitions: List of transitions
        transition_types: List of types to include

    Returns:
        Filtered list of transitions
    """
    return [t for t in transitions if t.transition_type in transition_types]


def get_turbulence_events(
    transitions: List[MechanicsTransition]
) -> List[MechanicsTransition]:
    """Extract turbulence-related transitions."""
    return filter_transitions_by_type(
        transitions,
        ["turbulence_onset", "turbulence_increase"]
    )


def get_energy_events(
    transitions: List[MechanicsTransition]
) -> List[MechanicsTransition]:
    """Extract energy-related transitions."""
    return filter_transitions_by_type(
        transitions,
        ["energy_injection", "energy_dissipation", "energy_destabilization", "energy_stabilization"]
    )


def get_critical_transitions(
    transitions: List[MechanicsTransition]
) -> List[MechanicsTransition]:
    """
    Get critical transitions indicating system degradation.

    Returns transitions that suggest potential failure:
    - Severe turbulence onset
    - Energy destabilization
    - Orbit destabilization
    - Equilibrium forcing
    """
    critical_types = [
        "turbulence_onset",
        "energy_destabilization",
        "orbit_destabilization",
        "equilibrium_forcing",
        "equilibrium_departure",
    ]

    return [
        t for t in transitions
        if t.transition_type in critical_types and t.severity in ["moderate", "severe"]
    ]


def validate_mechanics_stability(
    current_state: str,
    previous_state: str
) -> tuple:
    """
    Validate that mechanics state is stable (no unexpected changes).

    Args:
        current_state: Current state string (e.g., "CONSERVATIVE.APPROACHING.LAMINAR.CIRCULAR")
        previous_state: Previous state string

    Returns:
        (is_stable, alert_message)
    """
    if current_state == previous_state:
        return True, ""

    # Parse state strings
    current_parts = current_state.upper().split(".")
    previous_parts = previous_state.upper().split(".")

    if len(current_parts) != 4 or len(previous_parts) != 4:
        return False, "Invalid state string format"

    changes = []
    fields = ["energy", "equilibrium", "flow", "orbit"]

    for i, field in enumerate(fields):
        if current_parts[i] != previous_parts[i]:
            changes.append(f"{field}: {previous_parts[i]} -> {current_parts[i]}")

    if changes:
        alert = f"Mechanics state changed: {'; '.join(changes)}"
        return False, alert

    return True, ""


def summarize_transitions(
    transitions: List[MechanicsTransition]
) -> Dict[str, Any]:
    """
    Summarize a list of transitions.

    Returns:
        Dictionary with summary statistics
    """
    if not transitions:
        return {
            "total": 0,
            "by_severity": {"mild": 0, "moderate": 0, "severe": 0},
            "by_type": {},
            "by_field": {},
            "entities_affected": 0,
            "signals_affected": 0,
            "critical_count": 0,
        }

    by_severity = {"mild": 0, "moderate": 0, "severe": 0}
    by_type = {}
    by_field = {}
    entities = set()
    signals = set()

    critical_types = [
        "turbulence_onset", "energy_destabilization",
        "orbit_destabilization", "equilibrium_forcing"
    ]
    critical_count = 0

    for t in transitions:
        by_severity[t.severity] = by_severity.get(t.severity, 0) + 1
        by_type[t.transition_type] = by_type.get(t.transition_type, 0) + 1
        by_field[t.field] = by_field.get(t.field, 0) + 1
        entities.add(t.entity_id)
        signals.add(t.signal_id)

        if t.transition_type in critical_types:
            critical_count += 1

    return {
        "total": len(transitions),
        "by_severity": by_severity,
        "by_type": by_type,
        "by_field": by_field,
        "entities_affected": len(entities),
        "signals_affected": len(signals),
        "critical_count": critical_count,
    }
