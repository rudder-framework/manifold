"""
Signal States Validation
========================

Validation functions for signal state consistency and cohort alignment.

Key principle: Mechanics states should be STABLE over time for a given signal.
Unexpected changes indicate either:
1. A genuine regime change (rare, should trigger alert)
2. A measurement/computation issue (should be investigated)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .state_builders import parse_dotted_state, compute_state_distance


# =============================================================================
# MECHANICS STABILITY VALIDATION
# =============================================================================

# Expected stable components (should not change between windows)
MECHANICS_STABLE_POSITIONS = [0, 1]  # energy_class, equilibrium_class

# Components that may evolve (but still warrant attention)
MECHANICS_EVOLVING_POSITIONS = [2, 3]  # flow_class, orbit_class

# Valid transition pairs (from -> to)
VALID_MECHANICS_TRANSITIONS = {
    # Energy transitions (usually stable, but can change)
    ("CONSERVATIVE", "DRIVEN"): "energy_injection",
    ("CONSERVATIVE", "DISSIPATIVE"): "energy_loss",
    ("DRIVEN", "CONSERVATIVE"): "energy_stabilized",
    ("DRIVEN", "DISSIPATIVE"): "energy_depleting",
    ("DISSIPATIVE", "CONSERVATIVE"): "energy_recovered",
    ("DISSIPATIVE", "DRIVEN"): "external_forcing",
    ("FLUCTUATING", "CONSERVATIVE"): "stabilizing",
    ("FLUCTUATING", "DRIVEN"): "forcing_onset",

    # Equilibrium transitions
    ("APPROACHING", "AT_EQUILIBRIUM"): "equilibrium_reached",
    ("AT_EQUILIBRIUM", "DEPARTING"): "destabilizing",
    ("DEPARTING", "APPROACHING"): "restabilizing",
    ("FORCED", "APPROACHING"): "forcing_removed",

    # Flow transitions
    ("LAMINAR", "TRANSITIONAL"): "flow_destabilizing",
    ("TRANSITIONAL", "TURBULENT"): "turbulence_onset",
    ("TRANSITIONAL", "LAMINAR"): "flow_stabilizing",
    ("TURBULENT", "TRANSITIONAL"): "turbulence_subsiding",

    # Orbit transitions
    ("CIRCULAR", "ELLIPTICAL"): "orbit_perturbed",
    ("ELLIPTICAL", "IRREGULAR"): "orbit_destabilizing",
    ("ELLIPTICAL", "CIRCULAR"): "orbit_stabilizing",
    ("IRREGULAR", "ELLIPTICAL"): "orbit_recovering",
    ("LINEAR", "CIRCULAR"): "rotation_onset",
    ("LINEAR", "ELLIPTICAL"): "rotation_onset",
}


def validate_mechanics_stability(
    current_state: str,
    previous_state: str
) -> Tuple[bool, str]:
    """
    Validate that mechanics state hasn't changed unexpectedly.

    Mechanics states should generally be stable across time windows.
    Changes indicate either genuine regime changes or computation issues.

    Args:
        current_state: Current window mechanics state
        previous_state: Previous window mechanics state

    Returns:
        Tuple of (is_stable, explanation)
        - is_stable: True if no unexpected changes
        - explanation: Human-readable explanation of any changes
    """
    if not previous_state or not current_state:
        return True, "No previous state to compare"

    if current_state == previous_state:
        return True, "State unchanged"

    current_parts = parse_dotted_state(current_state)
    previous_parts = parse_dotted_state(previous_state)

    # Pad to 4 components
    while len(current_parts) < 4:
        current_parts.append("")
    while len(previous_parts) < 4:
        previous_parts.append("")

    changes = []
    alerts = []
    position_names = ["energy", "equilibrium", "flow", "orbit"]

    for i, (curr, prev) in enumerate(zip(current_parts, previous_parts)):
        if curr != prev:
            transition_key = (prev, curr)
            if transition_key in VALID_MECHANICS_TRANSITIONS:
                reason = VALID_MECHANICS_TRANSITIONS[transition_key]
                changes.append(f"{position_names[i]}: {prev} -> {curr} ({reason})")
            else:
                # Unexpected transition
                if i in MECHANICS_STABLE_POSITIONS:
                    alerts.append(f"UNEXPECTED {position_names[i]}: {prev} -> {curr}")
                else:
                    changes.append(f"{position_names[i]}: {prev} -> {curr}")

    if alerts:
        return False, "; ".join(alerts + changes)
    elif changes:
        return True, "; ".join(changes)
    else:
        return True, "State unchanged"


def get_mechanics_stability_level(
    current_state: str,
    previous_state: str
) -> str:
    """
    Get stability level for mechanics transition.

    Args:
        current_state: Current window mechanics state
        previous_state: Previous window mechanics state

    Returns:
        One of: "stable", "evolving", "warning", "critical"
    """
    if not previous_state or current_state == previous_state:
        return "stable"

    current_parts = parse_dotted_state(current_state)
    previous_parts = parse_dotted_state(previous_state)

    # Pad to 4 components
    while len(current_parts) < 4:
        current_parts.append("")
    while len(previous_parts) < 4:
        previous_parts.append("")

    n_changes = sum(1 for c, p in zip(current_parts, previous_parts) if c != p)

    # Check if changes are in stable positions
    stable_changes = sum(
        1 for i in MECHANICS_STABLE_POSITIONS
        if i < len(current_parts) and i < len(previous_parts)
        and current_parts[i] != previous_parts[i]
    )

    if stable_changes > 0:
        if n_changes >= 3:
            return "critical"
        else:
            return "warning"
    elif n_changes > 0:
        return "evolving"
    else:
        return "stable"


# =============================================================================
# COHORT ALIGNMENT VALIDATION
# =============================================================================

@dataclass
class CohortValidation:
    """Result of cohort alignment validation."""
    cohort_id: str = ""
    cohort_name: str = ""

    # Member counts
    n_members: int = 0
    n_aligned: int = 0
    n_divergent: int = 0

    # Alignment metrics
    alignment_score: float = 1.0      # 0-1, 1 = all aligned
    dominant_state: str = ""          # Most common state in cohort
    divergent_units: List[str] = field(default_factory=list)

    # Details
    state_distribution: Dict[str, int] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    validated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cohort_id': self.cohort_id,
            'cohort_name': self.cohort_name,
            'n_members': self.n_members,
            'n_aligned': self.n_aligned,
            'n_divergent': self.n_divergent,
            'alignment_score': self.alignment_score,
            'dominant_state': self.dominant_state,
            'divergent_units': self.divergent_units,
            'state_distribution': self.state_distribution,
            'alerts': self.alerts,
            'validated_at': self.validated_at.isoformat(),
        }


def validate_cohort_alignment(
    cohort_id: str,
    cohort_name: str,
    member_states: Dict[str, str],
    state_type: str = "mechanics"
) -> CohortValidation:
    """
    Validate that cohort members have aligned states.

    User-defined cohorts should have members that behave similarly.
    This validates whether the cohort grouping is appropriate.

    Args:
        cohort_id: Cohort identifier
        cohort_name: Cohort display name
        member_states: Dict of {unit_id: state_string}
        state_type: Which state to validate ("mechanics", "dynamics", etc.)

    Returns:
        CohortValidation with alignment analysis
    """
    validation = CohortValidation(
        cohort_id=cohort_id,
        cohort_name=cohort_name,
        n_members=len(member_states),
    )

    if not member_states:
        return validation

    # Count state occurrences
    state_counts: Dict[str, int] = {}
    for unit_id, state in member_states.items():
        state_counts[state] = state_counts.get(state, 0) + 1

    validation.state_distribution = state_counts

    # Find dominant state
    dominant_state = max(state_counts.keys(), key=lambda s: state_counts[s])
    validation.dominant_state = dominant_state
    validation.n_aligned = state_counts[dominant_state]
    validation.n_divergent = validation.n_members - validation.n_aligned

    # Identify divergent units
    for unit_id, state in member_states.items():
        if state != dominant_state:
            validation.divergent_units.append(unit_id)

    # Compute alignment score
    validation.alignment_score = validation.n_aligned / validation.n_members

    # Generate alerts
    if validation.alignment_score < 0.5:
        validation.alerts.append(
            f"Cohort '{cohort_name}' has low alignment ({validation.alignment_score:.1%}). "
            f"Consider splitting into sub-cohorts."
        )
    elif validation.alignment_score < 0.8:
        validation.alerts.append(
            f"Cohort '{cohort_name}' has moderate alignment ({validation.alignment_score:.1%}). "
            f"{validation.n_divergent} members diverge from dominant state."
        )

    return validation


def find_natural_cohorts(
    unit_states: Dict[str, str],
    min_cohort_size: int = 2
) -> Dict[str, List[str]]:
    """
    Discover natural cohorts from state alignment.

    Groups units that share the same state into potential cohorts.

    Args:
        unit_states: Dict of {unit_id: state_string}
        min_cohort_size: Minimum members for a valid cohort

    Returns:
        Dict of {state_string: [unit_ids]}
    """
    state_groups: Dict[str, List[str]] = {}

    for unit_id, state in unit_states.items():
        if state not in state_groups:
            state_groups[state] = []
        state_groups[state].append(unit_id)

    # Filter by minimum size
    return {
        state: units
        for state, units in state_groups.items()
        if len(units) >= min_cohort_size
    }


def compare_cohort_to_discovered(
    user_cohort: Dict[str, str],
    discovered_cohorts: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Compare user-defined cohort to discovered natural groupings.

    Args:
        user_cohort: Dict of {unit_id: state_string} for user cohort
        discovered_cohorts: Dict of {state: [unit_ids]} from find_natural_cohorts

    Returns:
        Comparison analysis
    """
    user_units = set(user_cohort.keys())

    overlaps = {}
    for state, discovered_units in discovered_cohorts.items():
        discovered_set = set(discovered_units)
        overlap = user_units & discovered_set
        if overlap:
            overlaps[state] = {
                'overlap_count': len(overlap),
                'overlap_ratio': len(overlap) / len(user_units),
                'units': list(overlap),
            }

    # Find best matching discovered cohort
    best_match = None
    best_ratio = 0.0
    for state, info in overlaps.items():
        if info['overlap_ratio'] > best_ratio:
            best_ratio = info['overlap_ratio']
            best_match = state

    return {
        'user_cohort_size': len(user_units),
        'overlaps': overlaps,
        'best_match_state': best_match,
        'best_match_ratio': best_ratio,
        'recommendation': (
            "User cohort aligns well with discovered grouping"
            if best_ratio >= 0.8
            else "User cohort may benefit from re-evaluation"
        ),
    }
