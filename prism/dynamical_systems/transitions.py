"""
Dynamical Systems Transition Detection
======================================

Detects meaningful transitions between consecutive DynamicsState windows.

Transition detection follows these rules:

1. Categorical Fields (trajectory, attractor):
   - Any change emits a transition

2. Numeric Fields (stability, predictability, coupling, memory):
   - Change must exceed threshold to emit transition
   - Thresholds defined per metric

3. Severity Classification:
   - mild: Delta > threshold but < 2x threshold
   - moderate: Delta > 2x threshold OR sign change
   - severe: Categorical flip OR delta > 3x threshold OR stability crosses zero

4. Transition Types:
   - bifurcation: Stability crossed zero (stable → unstable)
   - collapse: Predictability or coupling dropped sharply
   - recovery: Metrics improving after previous decline
   - shift: Categorical change (trajectory or attractor type)
   - flip: Memory crossed 0.5 (persistent ↔ anti-persistent)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from prism.dynamical_systems.models import DynamicsState, DynamicsTransition
from prism.config.thresholds import (
    DYNAMICS_TRANSITION,
    TRANSITION_NUMERIC,
    is_meaningful_change,
    classify_severity,
    get_transition_type,
)


# Default thresholds if config not available
DEFAULT_THRESHOLDS = {
    "stability": 0.2,       # 20% of range
    "predictability": 0.15,
    "coupling": 0.15,
    "memory": 0.1,
}


def detect_transitions(
    current: DynamicsState,
    previous: DynamicsState,
    thresholds: Optional[Dict[str, float]] = None
) -> List[DynamicsTransition]:
    """
    Detect transitions between two consecutive DynamicsState windows.

    Args:
        current: Current window state
        previous: Previous window state
        thresholds: Optional custom thresholds (uses defaults if None)

    Returns:
        List of DynamicsTransition objects for meaningful changes
    """
    if thresholds is None:
        try:
            thresholds = TRANSITION_NUMERIC.get("dynamics", DEFAULT_THRESHOLDS)
        except Exception:
            thresholds = DEFAULT_THRESHOLDS

    transitions = []

    # Check categorical fields
    for field in ["trajectory", "attractor"]:
        current_val = getattr(current, field, None)
        previous_val = getattr(previous, field, None)

        if current_val != previous_val:
            trans = _create_categorical_transition(
                current, previous, field, previous_val, current_val
            )
            transitions.append(trans)

    # Check numeric fields
    for field in ["stability", "predictability", "coupling", "memory"]:
        current_val = getattr(current, field, 0.5)
        previous_val = getattr(previous, field, 0.5)

        threshold = thresholds.get(field, 0.15)
        delta = abs(current_val - previous_val)

        if delta > threshold:
            trans = _create_numeric_transition(
                current, previous, field, previous_val, current_val, delta, threshold
            )
            transitions.append(trans)

    return transitions


def _create_categorical_transition(
    current: DynamicsState,
    previous: DynamicsState,
    field: str,
    from_value: str,
    to_value: str
) -> DynamicsTransition:
    """Create transition for categorical field change."""

    # Determine transition type
    if field == "trajectory":
        transition_type = "shift"
        # Check for specific trajectory transitions
        if from_value == "converging" and to_value in ["diverging", "chaotic"]:
            severity = "severe"
        elif to_value == "chaotic":
            severity = "moderate"
        else:
            severity = "mild"
    elif field == "attractor":
        transition_type = "shift"
        # Loss of attractor is severe
        if from_value != "none" and to_value == "none":
            severity = "severe"
        elif from_value == "limit_cycle" and to_value == "strange":
            severity = "severe"
        elif to_value == "strange":
            severity = "moderate"
        else:
            severity = "mild"
    else:
        transition_type = "shift"
        severity = "mild"

    return DynamicsTransition(
        entity_id=current.entity_id,
        unit_id=current.unit_id,
        window_idx=current.window_idx,
        timestamp=current.timestamp,
        field=field,
        from_value=str(from_value),
        to_value=str(to_value),
        delta=None,
        transition_type=transition_type,
        severity=severity,
    )


def _create_numeric_transition(
    current: DynamicsState,
    previous: DynamicsState,
    field: str,
    from_value: float,
    to_value: float,
    delta: float,
    threshold: float
) -> DynamicsTransition:
    """Create transition for numeric field change."""

    # Determine transition type
    if field == "stability":
        # Check for bifurcation (sign change)
        if (from_value > 0 and to_value < 0) or (from_value < 0 and to_value > 0):
            transition_type = "bifurcation"
            severity = "severe"
        elif to_value < 0 and from_value >= 0:
            transition_type = "collapse"
            severity = "moderate"
        elif to_value > 0 and from_value <= 0:
            transition_type = "recovery"
            severity = "moderate"
        else:
            transition_type = "shift"
            severity = _classify_numeric_severity(delta, threshold)

    elif field == "predictability":
        if to_value < from_value - 2 * threshold:
            transition_type = "collapse"
            severity = "moderate" if delta < 3 * threshold else "severe"
        elif to_value > from_value + 2 * threshold:
            transition_type = "recovery"
            severity = "mild"
        else:
            transition_type = "shift"
            severity = _classify_numeric_severity(delta, threshold)

    elif field == "coupling":
        if to_value < from_value - 2 * threshold:
            transition_type = "collapse"
            severity = "moderate" if delta < 3 * threshold else "severe"
        elif to_value > from_value + 2 * threshold:
            transition_type = "recovery"
            severity = "mild"
        else:
            transition_type = "shift"
            severity = _classify_numeric_severity(delta, threshold)

    elif field == "memory":
        # Check for flip across 0.5 (persistent ↔ anti-persistent)
        if (from_value > 0.5 and to_value < 0.5) or (from_value < 0.5 and to_value > 0.5):
            transition_type = "flip"
            severity = "moderate"
        else:
            transition_type = "shift"
            severity = _classify_numeric_severity(delta, threshold)

    else:
        transition_type = "shift"
        severity = _classify_numeric_severity(delta, threshold)

    return DynamicsTransition(
        entity_id=current.entity_id,
        unit_id=current.unit_id,
        window_idx=current.window_idx,
        timestamp=current.timestamp,
        field=field,
        from_value=f"{from_value:.4f}",
        to_value=f"{to_value:.4f}",
        delta=delta,
        transition_type=transition_type,
        severity=severity,
    )


def _classify_numeric_severity(delta: float, threshold: float) -> str:
    """Classify severity based on delta relative to threshold."""
    if delta > 3 * threshold:
        return "severe"
    elif delta > 2 * threshold:
        return "moderate"
    else:
        return "mild"


def detect_all_transitions(
    states: List[DynamicsState],
    thresholds: Optional[Dict[str, float]] = None
) -> List[DynamicsTransition]:
    """
    Detect all transitions across a sequence of states.

    Args:
        states: List of DynamicsState objects, sorted by window_idx
        thresholds: Optional custom thresholds

    Returns:
        List of all DynamicsTransition objects
    """
    all_transitions = []

    for i in range(1, len(states)):
        transitions = detect_transitions(states[i], states[i-1], thresholds)
        all_transitions.extend(transitions)

    return all_transitions


def filter_transitions_by_severity(
    transitions: List[DynamicsTransition],
    min_severity: str = "mild"
) -> List[DynamicsTransition]:
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
    transitions: List[DynamicsTransition],
    transition_types: List[str]
) -> List[DynamicsTransition]:
    """
    Filter transitions by type.

    Args:
        transitions: List of transitions
        transition_types: List of types to include

    Returns:
        Filtered list of transitions
    """
    return [t for t in transitions if t.transition_type in transition_types]


def get_bifurcations(transitions: List[DynamicsTransition]) -> List[DynamicsTransition]:
    """Extract only bifurcation transitions (stability sign changes)."""
    return filter_transitions_by_type(transitions, ["bifurcation"])


def get_collapses(transitions: List[DynamicsTransition]) -> List[DynamicsTransition]:
    """Extract only collapse transitions."""
    return filter_transitions_by_type(transitions, ["collapse"])


def get_escalation_candidates(
    transitions: List[DynamicsTransition]
) -> List[str]:
    """
    Get entity IDs that should be escalated to Causal Mechanics.

    Returns entities with severe transitions or bifurcations.
    """
    escalate_ids = set()

    for t in transitions:
        if t.severity == "severe" or t.transition_type == "bifurcation":
            escalate_ids.add(t.entity_id)

    return list(escalate_ids)


def summarize_transitions(
    transitions: List[DynamicsTransition]
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
        }

    by_severity = {"mild": 0, "moderate": 0, "severe": 0}
    by_type = {}
    by_field = {}
    entities = set()

    for t in transitions:
        by_severity[t.severity] = by_severity.get(t.severity, 0) + 1
        by_type[t.transition_type] = by_type.get(t.transition_type, 0) + 1
        by_field[t.field] = by_field.get(t.field, 0) + 1
        entities.add(t.entity_id)

    return {
        "total": len(transitions),
        "by_severity": by_severity,
        "by_type": by_type,
        "by_field": by_field,
        "entities_affected": len(entities),
    }
