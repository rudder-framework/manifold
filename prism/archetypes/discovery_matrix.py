"""
Discovery Matrix
================

Differential diagnosis for regime transitions.

The discovery matrix tracks which axes are moving vs stable,
enabling identification of transition type and early warnings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..models.enums import TransitionType


# =============================================================================
# TRANSITION THRESHOLDS
# =============================================================================

# Movement thresholds (change per window that indicates "moving")
MOVEMENT_THRESHOLDS = {
    'memory': 0.10,       # Hurst changing by 0.10
    'information': 0.15,  # Entropy changing
    'recurrence': 0.15,   # Determinism changing
    'volatility': 0.10,   # Persistence changing
    'frequency': 0.10,    # Centroid shifting
    'dynamics': 0.05,     # Lyapunov very sensitive
    'energy': 0.15,       # Hamiltonian trend
    'wavelet': 0.20,      # Scale shift
    'derivatives': 0.10,  # Sign change rate
}

# Axis display names
AXIS_NAMES = {
    'memory': 'Memory',
    'information': 'Information',
    'recurrence': 'Recurrence',
    'volatility': 'Volatility',
    'frequency': 'Frequency',
    'dynamics': 'Dynamics',
    'energy': 'Energy',
    'wavelet': 'Wavelet Scale',
    'derivatives': 'Motion',
}


# =============================================================================
# DISCOVERY MATRIX
# =============================================================================

@dataclass
class AxisDelta:
    """Change in a single axis between windows."""
    axis: str
    previous: float
    current: float
    delta: float
    threshold: float
    is_moving: bool
    direction: str  # 'increasing', 'decreasing', 'stable'


DISCOVERY_MATRIX: Dict[str, Dict] = {
    # Key patterns identified by which axes are moving together

    "trend_exhaustion": {
        "description": "Trend losing momentum - prepare for reversal or consolidation",
        "moving": ["memory", "energy"],
        "stable": ["volatility", "dynamics"],
        "pattern": {
            "memory": "decreasing",  # H dropping toward 0.5
            "energy": "decreasing",  # Hamiltonian dissipating
        },
        "alert_level": "warning",
        "recommended_action": "Monitor for reversal signals"
    },

    "volatility_regime_shift": {
        "description": "Volatility structure changing - new regime emerging",
        "moving": ["volatility"],
        "stable": ["memory", "frequency"],
        "pattern": {
            "volatility": "any",  # Either direction significant
        },
        "alert_level": "warning",
        "recommended_action": "Reassess risk models"
    },

    "chaos_onset": {
        "description": "System approaching chaotic dynamics",
        "moving": ["dynamics", "information"],
        "stable": ["volatility"],
        "pattern": {
            "dynamics": "increasing",   # Lyapunov going positive
            "information": "increasing", # Entropy rising
        },
        "alert_level": "critical",
        "recommended_action": "Immediate investigation required"
    },

    "stability_return": {
        "description": "System returning to stable regime",
        "moving": ["dynamics", "volatility"],
        "stable": ["memory"],
        "pattern": {
            "dynamics": "decreasing",   # Lyapunov going negative
            "volatility": "decreasing", # Volatility subsiding
        },
        "alert_level": "info",
        "recommended_action": "Confirm stability before resuming normal operation"
    },

    "frequency_shift": {
        "description": "Dominant frequency changing - check for mechanical issues",
        "moving": ["frequency", "wavelet"],
        "stable": ["memory", "volatility"],
        "pattern": {
            "frequency": "any",
            "wavelet": "any",
        },
        "alert_level": "warning",
        "recommended_action": "Check for resonance or mechanical changes"
    },

    "energy_accumulation": {
        "description": "Energy building in system - potential breakout",
        "moving": ["energy"],
        "stable": ["volatility", "dynamics"],
        "pattern": {
            "energy": "increasing",
        },
        "alert_level": "warning",
        "recommended_action": "Monitor for sudden release"
    },

    "energy_dissipation": {
        "description": "System losing energy - damping or decay",
        "moving": ["energy", "dynamics"],
        "stable": ["memory"],
        "pattern": {
            "energy": "decreasing",
            "dynamics": "decreasing",
        },
        "alert_level": "info",
        "recommended_action": "Normal if expected, investigate if not"
    },

    "multi_axis_transition": {
        "description": "Multiple axes changing simultaneously - major regime change",
        "moving": ["memory", "volatility", "dynamics"],  # 3+ axes
        "stable": [],
        "pattern": {},  # Any direction
        "alert_level": "critical",
        "recommended_action": "Full system review - significant behavioral change"
    },

    "recurrence_breakdown": {
        "description": "Pattern structure breaking down",
        "moving": ["recurrence", "information"],
        "stable": ["frequency"],
        "pattern": {
            "recurrence": "decreasing",
            "information": "increasing",
        },
        "alert_level": "warning",
        "recommended_action": "Check for new fault modes"
    },

    "memory_emergence": {
        "description": "Correlation structure emerging from random",
        "moving": ["memory"],
        "stable": ["volatility", "frequency"],
        "pattern": {
            "memory": "any",  # Moving away from 0.5
        },
        "alert_level": "info",
        "recommended_action": "Monitor for trend or mean-reversion establishment"
    },
}


def compute_axis_deltas(
    previous: Dict[str, float],
    current: Dict[str, float]
) -> List[AxisDelta]:
    """
    Compute changes between two fingerprint snapshots.

    Args:
        previous: Previous window's axis values
        current: Current window's axis values

    Returns:
        List of AxisDelta objects for each axis
    """
    deltas = []

    for axis, threshold in MOVEMENT_THRESHOLDS.items():
        prev_val = previous.get(axis, 0.5)
        curr_val = current.get(axis, 0.5)
        delta = curr_val - prev_val

        is_moving = abs(delta) > threshold

        if delta > threshold:
            direction = "increasing"
        elif delta < -threshold:
            direction = "decreasing"
        else:
            direction = "stable"

        deltas.append(AxisDelta(
            axis=axis,
            previous=prev_val,
            current=curr_val,
            delta=delta,
            threshold=threshold,
            is_moving=is_moving,
            direction=direction
        ))

    return deltas


def diagnose_differential(
    previous_fingerprint: np.ndarray,
    current_fingerprint: np.ndarray,
    previous_extras: Optional[Dict] = None,
    current_extras: Optional[Dict] = None
) -> Dict:
    """
    Perform differential diagnosis between two windows.

    Args:
        previous_fingerprint: Previous 6D/7D fingerprint
        current_fingerprint: Current 6D/7D fingerprint
        previous_extras: Optional dict with wavelet/derivatives metrics
        current_extras: Optional dict with wavelet/derivatives metrics

    Returns:
        Dictionary with:
            - transition_type: TransitionType enum
            - axes_moving: List of moving axis names
            - axes_stable: List of stable axis names
            - matched_pattern: Name of matched pattern (if any)
            - diagnosis: Human-readable diagnosis
            - alert_level: 'info', 'warning', 'critical'
            - recommended_action: Suggested response
    """
    # Build axis dictionaries
    axis_keys = ['memory', 'information', 'recurrence', 'volatility', 'frequency', 'dynamics']
    if len(previous_fingerprint) >= 7:
        axis_keys.append('energy')

    previous_dict = {k: float(previous_fingerprint[i]) for i, k in enumerate(axis_keys)}
    current_dict = {k: float(current_fingerprint[i]) for i, k in enumerate(axis_keys)}

    # Add extras if provided
    if previous_extras:
        previous_dict.update(previous_extras)
    if current_extras:
        current_dict.update(current_extras)

    # Compute deltas
    deltas = compute_axis_deltas(previous_dict, current_dict)

    # Categorize axes
    moving_axes = [d.axis for d in deltas if d.is_moving]
    stable_axes = [d.axis for d in deltas if not d.is_moving]

    # Determine transition type
    n_moving = len(moving_axes)
    if n_moving == 0:
        transition_type = TransitionType.NONE
    elif n_moving >= 3:
        transition_type = TransitionType.IN_PROGRESS
    elif n_moving >= 1:
        transition_type = TransitionType.APPROACHING
    else:
        transition_type = TransitionType.NONE

    # Match against known patterns
    matched_pattern = None
    alert_level = "info"
    recommended_action = "No action required"
    diagnosis = "Signal is stable"

    for pattern_name, pattern in DISCOVERY_MATRIX.items():
        pattern_moving = set(pattern["moving"])
        pattern_stable = set(pattern.get("stable", []))

        # Check if moving axes match
        if pattern_moving.issubset(set(moving_axes)):
            # Check direction patterns if specified
            direction_match = True
            for axis, expected_dir in pattern.get("pattern", {}).items():
                if expected_dir != "any":
                    actual = next((d for d in deltas if d.axis == axis), None)
                    if actual and actual.direction != expected_dir:
                        direction_match = False
                        break

            if direction_match:
                matched_pattern = pattern_name
                alert_level = pattern["alert_level"]
                recommended_action = pattern["recommended_action"]
                diagnosis = pattern["description"]
                break

    # Generate detailed diagnosis if no pattern matched
    if matched_pattern is None and n_moving > 0:
        moving_names = [AXIS_NAMES.get(a, a) for a in moving_axes]
        diagnosis = f"Movement detected in: {', '.join(moving_names)}"
        if n_moving >= 3:
            alert_level = "warning"
            recommended_action = "Multiple axes changing - monitor closely"

    return {
        'transition_type': transition_type,
        'axes_moving': moving_axes,
        'axes_stable': stable_axes,
        'matched_pattern': matched_pattern,
        'diagnosis': diagnosis,
        'alert_level': alert_level,
        'recommended_action': recommended_action,
        'axis_deltas': {d.axis: {'delta': d.delta, 'direction': d.direction} for d in deltas}
    }


def generate_summary(
    typology_result: Dict,
    differential: Optional[Dict] = None
) -> str:
    """
    Generate human-readable summary from typology and differential.

    Args:
        typology_result: Output from archetype matching
        differential: Optional differential diagnosis

    Returns:
        Multi-line summary string
    """
    lines = []

    # Archetype line
    archetype = typology_result.get('archetype', 'Unknown')
    confidence = typology_result.get('confidence', 0.0)
    lines.append(f"Archetype: {archetype} (confidence: {confidence:.1%})")

    # Secondary if close
    if typology_result.get('secondary_archetype'):
        secondary = typology_result['secondary_archetype']
        boundary = typology_result.get('boundary_proximity', 1.0)
        if boundary < 0.5:
            lines.append(f"  Near boundary with: {secondary}")

    # Differential if provided
    if differential:
        transition = differential.get('transition_type', TransitionType.NONE)
        if transition != TransitionType.NONE:
            lines.append(f"Transition: {transition.value}")
            lines.append(f"  {differential.get('diagnosis', '')}")

            moving = differential.get('axes_moving', [])
            if moving:
                moving_names = [AXIS_NAMES.get(a, a) for a in moving]
                lines.append(f"  Moving axes: {', '.join(moving_names)}")

            alert = differential.get('alert_level', 'info')
            if alert in ('warning', 'critical'):
                lines.append(f"  ⚠️ {differential.get('recommended_action', '')}")

    return "\n".join(lines)


def generate_alerts(
    typology_result: Dict,
    differential: Optional[Dict] = None,
    vector: Optional[Dict] = None
) -> List[str]:
    """
    Generate list of alert strings for UI display.

    Args:
        typology_result: Archetype matching result
        differential: Differential diagnosis
        vector: Raw vector metrics

    Returns:
        List of alert strings
    """
    alerts = []

    # Boundary proximity alert
    boundary = typology_result.get('boundary_proximity', 1.0)
    if boundary < 0.3:
        secondary = typology_result.get('secondary_archetype', 'unknown')
        alerts.append(f"Near archetype boundary with {secondary}")

    # Differential alerts
    if differential:
        alert_level = differential.get('alert_level', 'info')
        if alert_level == 'critical':
            alerts.append(f"CRITICAL: {differential.get('diagnosis', 'Major regime change')}")
        elif alert_level == 'warning':
            alerts.append(f"WARNING: {differential.get('diagnosis', 'Regime changing')}")

    # Vector-based alerts (if provided)
    if vector:
        # Hamiltonian not conserved
        if not vector.get('energy_conserved', True):
            trend = vector.get('hamiltonian_trend', 0)
            if trend > 0:
                alerts.append("Energy accumulating - system being driven")
            else:
                alerts.append("Energy dissipating - check for damping source")

        # High volatility
        persistence = vector.get('garch_persistence', 0)
        if persistence > 0.98:
            alerts.append("Volatility integrated - shocks are permanent")

        # Chaotic dynamics
        lyapunov = vector.get('lyapunov_exponent', 0)
        if lyapunov > 0.1:
            alerts.append("Chaotic dynamics detected - sensitive to initial conditions")

        # Discontinuities accelerating
        if vector.get('discontinuity_accelerating', False):
            alerts.append("Discontinuities accelerating - increasing instability")

    return alerts
