"""
Dynamical Systems Models
========================

Dataclasses and enums for Dynamical Systems framework output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class RegimeClass(Enum):
    """Dynamical regime classification"""
    COUPLED = "coupled"                 # Strong correlation, moving together
    DECOUPLED = "decoupled"             # Weak correlation, independent
    MODERATE = "moderate"               # Intermediate coupling
    TRANSITIONING = "transitioning"     # Active regime change


class DynamicsStabilityClass(Enum):
    """System stability classification (Dynamical Systems layer)"""
    STABLE = "stable"                   # Minimal change
    EVOLVING = "evolving"               # Gradual change
    UNSTABLE = "unstable"               # Rapid change
    CRITICAL = "critical"               # Near bifurcation


# Backwards compatibility alias
StabilityClass = DynamicsStabilityClass


class TrajectoryClass(Enum):
    """Trajectory classification"""
    CONVERGING = "converging"           # Moving toward attractor
    DIVERGING = "diverging"             # Moving away from attractor
    OSCILLATING = "oscillating"         # Periodic motion
    WANDERING = "wandering"             # No clear direction


class AttractorClass(Enum):
    """Attractor type classification"""
    FIXED_POINT = "fixed_point"         # Single stable state
    LIMIT_CYCLE = "limit_cycle"         # Periodic attractor
    STRANGE = "strange"                 # Chaotic attractor
    NONE = "none"                       # No clear attractor


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class DynamicsVector:
    """
    Numerical measurements from dynamical systems analysis.
    This is the DATA output - consumed by downstream frameworks.
    """

    # === IDENTIFICATION ===
    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: str = ""
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === REGIME METRICS ===
    correlation_level: float = 0.0      # Current coupling strength
    correlation_change: float = 0.0     # Rate of correlation change
    regime_duration: int = 0            # Time in current regime

    # === STABILITY METRICS ===
    stability_index: float = 0.0        # Overall stability [0,1]
    volatility_trend: float = 0.0       # Change in volatility
    density_change: float = 0.0         # Network density evolution

    # === TRAJECTORY METRICS ===
    trajectory_direction: float = 0.0   # Angle in phase space
    trajectory_speed: float = 0.0       # Rate of movement
    trajectory_curvature: float = 0.0   # Change in direction

    # === ATTRACTOR METRICS ===
    attractor_distance: float = 0.0     # Distance from nearest attractor
    basin_depth: float = 0.0            # Stability of current basin

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'correlation_level': self.correlation_level,
            'correlation_change': self.correlation_change,
            'regime_duration': self.regime_duration,
            'stability_index': self.stability_index,
            'volatility_trend': self.volatility_trend,
            'density_change': self.density_change,
            'trajectory_direction': self.trajectory_direction,
            'trajectory_speed': self.trajectory_speed,
            'trajectory_curvature': self.trajectory_curvature,
            'attractor_distance': self.attractor_distance,
            'basin_depth': self.basin_depth,
        }


@dataclass
class DynamicsTypology:
    """
    Classification output from dynamical systems analysis.
    This is the INTERPRETATION output - consumed by humans/reports.
    """

    # === IDENTIFICATION ===
    entity_id: str = ""
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === CLASSIFICATIONS ===
    regime_class: RegimeClass = RegimeClass.MODERATE
    stability_class: DynamicsStabilityClass = DynamicsStabilityClass.STABLE
    trajectory_class: TrajectoryClass = TrajectoryClass.WANDERING
    attractor_class: AttractorClass = AttractorClass.NONE

    # === SUMMARY ===
    summary: str = ""
    confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'regime_class': self.regime_class.value,
            'stability_class': self.stability_class.value,
            'trajectory_class': self.trajectory_class.value,
            'attractor_class': self.attractor_class.value,
            'summary': self.summary,
            'confidence': self.confidence,
            'alerts': self.alerts,
        }


@dataclass
class DynamicalSystemsOutput:
    """Combined output from Dynamical Systems framework."""
    vector: DynamicsVector
    typology: DynamicsTypology
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Combined dictionary for full output."""
        result = self.vector.to_dict()
        result.update(self.typology.to_dict())
        result['metadata'] = self.metadata
        return result


# =============================================================================
# NEW ARCHITECTURE: STATE + TRANSITIONS
# =============================================================================

@dataclass
class DynamicsState:
    """
    State of dynamical system at a single window.

    The 6 Dynamics Metrics:
        - trajectory: Where is it going? (categorical)
        - stability: Will perturbations grow? (-1 to 1)
        - attractor: What does it settle toward? (categorical)
        - predictability: How far can we forecast? (0-1)
        - coupling: How do signals drive each other? (0-1)
        - memory: Does past influence future? (0-1, 0.5 = random walk)
    """
    entity_id: str
    unit_id: str = ""  # Defaults to entity_id if not set
    window_idx: int = 0
    timestamp: Optional[Any] = None

    # Categorical states
    trajectory: str = "stationary"   # converging | diverging | periodic | chaotic | stationary
    attractor: str = "none"          # fixed_point | limit_cycle | strange | none

    # Numeric metrics (all normalized)
    stability: float = 0.5           # -1 to 1, >0 stable, <0 unstable
    predictability: float = 0.5      # 0-1, 1=deterministic, 0=random
    coupling: float = 0.5            # 0-1, 1=fully coupled, 0=independent
    memory: float = 0.5              # 0-1, 0.5=random walk, >0.5=persistent, <0.5=anti-persistent

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'window_idx': self.window_idx,
            'timestamp': self.timestamp,
            'trajectory': self.trajectory,
            'attractor': self.attractor,
            'stability': self.stability,
            'predictability': self.predictability,
            'coupling': self.coupling,
            'memory': self.memory,
        }

    def state_string(self) -> str:
        """Generate dot-delimited state string for signal_states table."""
        # Format: REGIME.STABILITY.TRAJECTORY.ATTRACTOR
        regime = "COUPLED" if self.coupling > 0.6 else "DECOUPLED" if self.coupling < 0.4 else "MODERATE"
        stab = "STABLE" if self.stability > 0.3 else "UNSTABLE" if self.stability < -0.3 else "EVOLVING"
        return f"{regime}.{stab}.{self.trajectory.upper()}.{self.attractor.upper()}"


@dataclass
class DynamicsTransition:
    """
    A meaningful state change between consecutive windows.

    Transition Types:
        - bifurcation: Stability crossed zero (stable → unstable)
        - collapse: Predictability or coupling dropped sharply
        - recovery: Metrics improving after previous decline
        - shift: Categorical change (trajectory or attractor type)
        - flip: Memory crossed 0.5 (persistent ↔ anti-persistent)

    Severity Classification:
        - mild: Delta > threshold but < 2x threshold
        - moderate: Delta > 2x threshold OR sign change
        - severe: Categorical flip OR delta > 3x threshold OR stability crosses zero
    """
    entity_id: str
    unit_id: str = ""  # Defaults to entity_id if not set
    window_idx: int = 0
    timestamp: Optional[Any] = None

    field: str = ""              # which metric changed
    from_value: str = ""         # previous value (string for flexibility)
    to_value: str = ""           # new value
    delta: Optional[float] = None  # numeric change magnitude (if applicable)

    transition_type: str = "shift"   # bifurcation | collapse | recovery | shift | flip
    severity: str = "mild"           # mild | moderate | severe

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'window_idx': self.window_idx,
            'timestamp': self.timestamp,
            'field': self.field,
            'from_value': self.from_value,
            'to_value': self.to_value,
            'delta': self.delta,
            'transition_type': self.transition_type,
            'severity': self.severity,
        }


# Thresholds for "meaningful" numeric changes
NUMERIC_THRESHOLDS = {
    "stability": 0.2,       # 20% of range
    "predictability": 0.15,
    "coupling": 0.15,
    "memory": 0.1
}
