"""
PRISM Data Models
=================

Dataclasses for signal analysis output.

Core Models:
    - SignalVector: Numerical measurements for downstream layers
    - SignalTypology: Classification and interpretation
    - SignalTypologyOutput: Combined vector + typology

Axis Models:
    - MemoryAxis, InformationAxis, RecurrenceAxis
    - VolatilityAxis, FrequencyAxis, DynamicsAxis
    - MomentumAxis, WaveletAxis, DerivativesAxis

Enums:
    - MemoryClass, InformationClass, RecurrenceClass
    - VolatilityClass, FrequencyClass, DynamicsClass
    - EnergyClass, TransitionType
"""

from .signal_vector import SignalVector
from .signal_typology import SignalTypology, SignalTypologyOutput
from .axes import (
    MemoryAxis,
    InformationAxis,
    RecurrenceAxis,
    VolatilityAxis,
    FrequencyAxis,
    DynamicsAxis,
    MomentumAxis,
    WaveletAxis,
    DerivativesAxis,
    DiscontinuityData,
)
from .enums import (
    MemoryClass,
    InformationClass,
    RecurrenceClass,
    VolatilityClass,
    FrequencyClass,
    DynamicsClass,
    EnergyClass,
    WaveletClass,
    DerivativesClass,
    ACFDecayType,
    TransitionType,
)

__all__ = [
    # Core
    'SignalVector',
    'SignalTypology',
    'SignalTypologyOutput',
    # Axes
    'MemoryAxis',
    'InformationAxis',
    'RecurrenceAxis',
    'VolatilityAxis',
    'FrequencyAxis',
    'DynamicsAxis',
    'MomentumAxis',
    'WaveletAxis',
    'DerivativesAxis',
    'DiscontinuityData',
    # Enums
    'MemoryClass',
    'InformationClass',
    'RecurrenceClass',
    'VolatilityClass',
    'FrequencyClass',
    'DynamicsClass',
    'EnergyClass',
    'WaveletClass',
    'DerivativesClass',
    'ACFDecayType',
    'TransitionType',
]
