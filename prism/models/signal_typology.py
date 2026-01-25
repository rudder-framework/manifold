"""
Signal Typology
===============

Classification and interpretation output.
This is the INFORMATION layer output.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import numpy as np

from .enums import (
    MemoryClass, InformationClass, RecurrenceClass,
    VolatilityClass, FrequencyClass, DynamicsClass,
    EnergyClass, TransitionType
)
from .signal_vector import SignalVector


@dataclass
class SignalTypology:
    """
    Classification and interpretation.

    This is the INFORMATION output of signal_typology.
    Contains the "what type of signal is this?" answer.
    """

    # Identification
    entity_id: str = "unknown"
    signal_id: str = "unknown"
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)
    n_observations: int = 0

    # Axis classifications
    memory_class: MemoryClass = MemoryClass.RANDOM
    information_class: InformationClass = InformationClass.MODERATE
    recurrence_class: RecurrenceClass = RecurrenceClass.TRANSITIONAL
    volatility_class: VolatilityClass = VolatilityClass.PERSISTENT
    frequency_class: FrequencyClass = FrequencyClass.BROADBAND
    dynamics_class: DynamicsClass = DynamicsClass.STABLE
    energy_class: EnergyClass = EnergyClass.CONSERVATIVE

    # Archetype classification
    archetype: str = "Unknown"
    archetype_distance: float = 0.0
    secondary_archetype: str = "Unknown"
    secondary_distance: float = 0.0
    boundary_proximity: float = 1.0  # 0 = at boundary, 1 = far

    # 6D fingerprint (normalized axis values)
    fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(6))

    # Transition detection
    regime_transition: TransitionType = TransitionType.NONE
    axes_moving: List[str] = field(default_factory=list)
    axes_stable: List[str] = field(default_factory=list)
    transition_diagnosis: str = ""

    # Human-readable output
    summary: str = ""
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Export to dictionary for serialization."""
        return {
            # Identity
            'entity_id': self.entity_id,
            'signal_id': self.signal_id,
            'window_start': self.window_start.isoformat() if hasattr(self.window_start, 'isoformat') else str(self.window_start),
            'window_end': self.window_end.isoformat() if hasattr(self.window_end, 'isoformat') else str(self.window_end),
            'n_observations': self.n_observations,
            # Classifications
            'memory_class': self.memory_class.value,
            'information_class': self.information_class.value,
            'recurrence_class': self.recurrence_class.value,
            'volatility_class': self.volatility_class.value,
            'frequency_class': self.frequency_class.value,
            'dynamics_class': self.dynamics_class.value,
            'energy_class': self.energy_class.value,
            # Archetype
            'archetype': self.archetype,
            'archetype_distance': self.archetype_distance,
            'secondary_archetype': self.secondary_archetype,
            'secondary_distance': self.secondary_distance,
            'boundary_proximity': self.boundary_proximity,
            'fingerprint': self.fingerprint.tolist(),
            # Transition
            'regime_transition': self.regime_transition.value,
            'axes_moving': self.axes_moving,
            'axes_stable': self.axes_stable,
            'transition_diagnosis': self.transition_diagnosis,
            # Human
            'summary': self.summary,
            'alerts': self.alerts,
            'confidence': self.confidence,
        }


@dataclass
class SignalTypologyOutput:
    """
    Complete output from Signal Typology layer.

    Contains BOTH:
        - vector: Numerical measurements (DATA)
        - typology: Classification (INFORMATION)
    """
    vector: SignalVector = field(default_factory=SignalVector)
    typology: SignalTypology = field(default_factory=SignalTypology)

    def to_dict(self) -> Dict:
        """Combined dictionary for full output."""
        result = self.vector.to_dict()
        result.update(self.typology.to_dict())
        return result
