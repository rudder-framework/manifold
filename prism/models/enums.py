"""
Classification Enums
====================

Enumerations for axis classification and transition types.
"""

from enum import Enum


class MemoryClass(Enum):
    """Axis 1: Memory structure classification"""
    ANTI_PERSISTENT = "anti_persistent"   # H < 0.45: mean-reverting
    RANDOM = "random"                      # 0.45 <= H <= 0.55: no memory
    PERSISTENT = "persistent"              # H > 0.55: trending


class InformationClass(Enum):
    """Axis 2: Information/entropy classification"""
    LOW = "low"           # Highly structured, predictable
    MODERATE = "moderate" # Mixed structure
    HIGH = "high"         # High randomness, complex


class RecurrenceClass(Enum):
    """Axis 3: Recurrence structure classification"""
    DETERMINISTIC = "deterministic"   # DET > 0.7: strong pattern repetition
    TRANSITIONAL = "transitional"     # 0.4 <= DET <= 0.7: mixed
    STOCHASTIC = "stochastic"         # DET < 0.4: weak/no patterns


class VolatilityClass(Enum):
    """Axis 4: Volatility persistence classification"""
    DISSIPATING = "dissipating"   # α+β < 0.85: shocks fade quickly
    PERSISTENT = "persistent"      # 0.85 <= α+β < 0.99: shocks linger
    INTEGRATED = "integrated"      # α+β >= 0.99: shocks permanent


class FrequencyClass(Enum):
    """Axis 5: Spectral character classification"""
    NARROWBAND = "narrowband"     # Dominant frequency, low bandwidth
    BROADBAND = "broadband"       # No dominant frequency, high bandwidth
    ONE_OVER_F = "one_over_f"     # 1/f noise, power-law spectrum


class DynamicsClass(Enum):
    """Axis 6: Dynamical stability classification"""
    STABLE = "stable"             # λ < -0.05: converging to attractor
    EDGE_OF_CHAOS = "edge_of_chaos"  # -0.05 <= λ <= 0.05: critical
    CHAOTIC = "chaotic"           # λ > 0.05: sensitive dependence


class EnergyClass(Enum):
    """Axis 7: Energy/Hamiltonian classification"""
    CONSERVATIVE = "conservative"  # H approximately constant (stable)
    DRIVEN = "driven"              # H increasing (external input)
    DISSIPATIVE = "dissipative"    # H decreasing (energy loss)


class WaveletClass(Enum):
    """Wavelet scale classification"""
    LOW_FREQUENCY_DOMINANT = "low_frequency_dominant"
    HIGH_FREQUENCY_DOMINANT = "high_frequency_dominant"
    BALANCED = "balanced"


class DerivativesClass(Enum):
    """Derivatives/motion classification"""
    STATIONARY = "stationary"       # Low velocity/acceleration
    SMOOTH_MOTION = "smooth_motion" # Moderate velocity, low jerk
    JERKY_MOTION = "jerky_motion"   # High jerk, abrupt changes


class ACFDecayType(Enum):
    """Autocorrelation decay type"""
    EXPONENTIAL = "exponential"   # Short-range dependence
    POWER_LAW = "power_law"       # Long-range dependence


class TransitionType(Enum):
    """Regime transition status"""
    NONE = "none"                 # Stable in current regime
    APPROACHING = "approaching"   # Moving toward boundary
    IN_PROGRESS = "in_progress"   # Currently crossing boundary
    COMPLETED = "completed"       # Recently changed regime
