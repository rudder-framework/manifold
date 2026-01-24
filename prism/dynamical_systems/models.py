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
