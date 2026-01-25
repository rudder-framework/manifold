"""
Axis Dataclasses
================

Dataclasses for each orthogonal axis measurement.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .enums import (
    MemoryClass, InformationClass, RecurrenceClass,
    VolatilityClass, FrequencyClass, DynamicsClass,
    EnergyClass, WaveletClass, DerivativesClass,
    ACFDecayType
)


@dataclass
class MemoryAxis:
    """Axis 1: Memory structure measurements"""
    hurst_exponent: float = 0.5
    hurst_confidence: float = 0.0
    hurst_method: str = "dfa"
    acf_decay_type: ACFDecayType = ACFDecayType.EXPONENTIAL
    acf_half_life: float = 0.0
    spectral_slope: float = 0.0
    spectral_slope_r2: float = 0.0
    memory_class: MemoryClass = MemoryClass.RANDOM


@dataclass
class InformationAxis:
    """Axis 2: Information/entropy measurements"""
    entropy_permutation: float = 0.0
    entropy_sample: float = 0.0
    entropy_rate: float = 0.0
    information_class: InformationClass = InformationClass.MODERATE


@dataclass
class RecurrenceAxis:
    """Axis 3: Recurrence Quantification Analysis measurements"""
    determinism: float = 0.0
    laminarity: float = 0.0
    entropy: float = 0.0
    recurrence_rate: float = 0.0
    trapping_time: float = 0.0
    max_diagonal: int = 0
    avg_diagonal: float = 0.0
    recurrence_class: RecurrenceClass = RecurrenceClass.TRANSITIONAL


@dataclass
class VolatilityAxis:
    """Axis 4: Volatility/amplitude dynamics measurements"""
    # GARCH parameters
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    garch_omega: float = 0.0
    garch_persistence: float = 0.0
    garch_unconditional: float = 0.0
    # Realized volatility
    realized_vol: float = 0.0
    bipower_variation: float = 0.0
    jump_component: float = 0.0
    jump_ratio: float = 0.0
    # Hilbert amplitude
    hilbert_amplitude_mean: float = 0.0
    hilbert_amplitude_std: float = 0.0
    # Classification
    volatility_class: VolatilityClass = VolatilityClass.PERSISTENT


@dataclass
class FrequencyAxis:
    """Axis 5: Spectral characteristics measurements"""
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_low_high_ratio: float = 0.0
    spectral_rolloff: float = 0.0
    frequency_class: FrequencyClass = FrequencyClass.BROADBAND


@dataclass
class DynamicsAxis:
    """Axis 6: Dynamical systems measurements"""
    lyapunov_exponent: float = 0.0
    lyapunov_confidence: float = 0.0
    embedding_dimension: int = 2
    correlation_dimension: float = 0.0
    dynamics_class: DynamicsClass = DynamicsClass.STABLE


@dataclass
class MomentumAxis:
    """Axis 7: Momentum/Energy measurements (Hamiltonian)"""
    # Derivatives
    d1_mean: float = 0.0
    d1_std: float = 0.0
    d2_mean: float = 0.0
    d2_std: float = 0.0
    momentum_strength: float = 0.0
    acceleration_regime: str = "neutral"
    # Energy
    kinetic_energy_mean: float = 0.0
    potential_energy_mean: float = 0.0
    hamiltonian_mean: float = 0.0
    hamiltonian_trend: float = 0.0
    energy_conserved: bool = True
    kinetic_ratio: float = 0.5
    # Angular momentum
    angular_momentum_mean: float = 0.0
    is_periodic: bool = False
    # Classification
    energy_class: EnergyClass = EnergyClass.CONSERVATIVE


@dataclass
class WaveletAxis:
    """Multi-scale wavelet decomposition measurements"""
    energy_by_scale: List[float] = field(default_factory=list)
    dominant_scale: int = 0
    scale_entropy: float = 0.0
    energy_ratio_low_high: float = 1.0
    detail_mean: float = 0.0
    detail_std: float = 0.0
    detail_kurtosis: float = 0.0
    approx_slope: float = 0.0
    approx_curvature: float = 0.0
    scale_shift_detected: bool = False
    scale_shift_direction: int = 0
    wavelet_class: WaveletClass = WaveletClass.BALANCED


@dataclass
class DerivativesAxis:
    """Motion dynamics measurements"""
    d1_mean: float = 0.0
    d1_std: float = 0.0
    d1_max: float = 0.0
    d2_mean: float = 0.0
    d2_std: float = 0.0
    d2_max: float = 0.0
    d3_mean: float = 0.0
    d3_std: float = 0.0
    momentum_strength: float = 0.0
    acceleration_regime: str = "neutral"
    smoothness: float = 1.0
    d1_sign_changes: int = 0
    d2_sign_changes: int = 0
    sign_change_rate: float = 0.0
    derivatives_class: DerivativesClass = DerivativesClass.STATIONARY


@dataclass
class DiscontinuityData:
    """Structural discontinuity detection results"""
    # Dirac (impulse)
    dirac_detected: bool = False
    dirac_count: int = 0
    dirac_max_magnitude: float = 0.0
    dirac_mean_magnitude: float = 0.0
    dirac_mean_half_life: float = 0.0
    dirac_up_ratio: float = 0.5
    # Heaviside (step)
    heaviside_detected: bool = False
    heaviside_count: int = 0
    heaviside_max_magnitude: float = 0.0
    heaviside_mean_magnitude: float = 0.0
    heaviside_up_ratio: float = 0.5
    # Structural
    total_count: int = 0
    mean_interval: float = 0.0
    interval_cv: float = 0.0
    dominant_period: float = 0.0
    is_accelerating: bool = False
