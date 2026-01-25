"""
Signal Vector
=============

The numerical vector that downstream layers consume.
This replaces what vector.py used to produce.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import numpy as np


@dataclass
class SignalVector:
    """
    Numerical measurements for downstream layers.

    This is the DATA output of signal_typology.
    Geometry and phase layers consume this vector.
    """

    # Identification
    entity_id: str = "unknown"
    signal_id: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)

    # Memory axis
    hurst_exponent: float = 0.5
    hurst_confidence: float = 0.0
    acf_decay_type: str = "exponential"
    acf_half_life: float = 0.0
    spectral_slope: float = 0.0
    spectral_slope_r2: float = 0.0

    # Information axis
    entropy_permutation: float = 0.0
    entropy_sample: float = 0.0
    entropy_rate: float = 0.0

    # Recurrence axis
    rqa_determinism: float = 0.0
    rqa_laminarity: float = 0.0
    rqa_entropy: float = 0.0
    rqa_recurrence_rate: float = 0.0
    rqa_trapping_time: float = 0.0
    rqa_max_diagonal: int = 0
    rqa_avg_diagonal: float = 0.0

    # Volatility axis
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    garch_persistence: float = 0.0
    garch_omega: float = 0.0
    garch_unconditional: float = 0.0
    realized_vol: float = 0.0
    bipower_variation: float = 0.0
    jump_component: float = 0.0
    hilbert_amplitude_mean: float = 0.0
    hilbert_amplitude_std: float = 0.0

    # Frequency axis
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_low_high_ratio: float = 0.0
    spectral_rolloff: float = 0.0

    # Wavelet (multi-scale)
    wavelet_dominant_scale: int = 0
    wavelet_scale_entropy: float = 0.0
    wavelet_energy_ratio: float = 1.0
    wavelet_energy_by_scale: List[float] = field(default_factory=list)

    # Dynamics axis
    lyapunov_exponent: float = 0.0
    lyapunov_confidence: float = 0.0
    embedding_dimension: int = 2
    correlation_dimension: float = 0.0

    # Momentum/Energy axis
    d1_mean: float = 0.0
    d1_std: float = 0.0
    d2_mean: float = 0.0
    d2_std: float = 0.0
    momentum_strength: float = 0.0
    kinetic_energy_mean: float = 0.0
    potential_energy_mean: float = 0.0
    hamiltonian_mean: float = 0.0
    hamiltonian_trend: float = 0.0
    energy_conserved: bool = True

    # Discontinuity
    dirac_detected: bool = False
    dirac_count: int = 0
    dirac_max_magnitude: float = 0.0
    dirac_mean_half_life: float = 0.0
    heaviside_detected: bool = False
    heaviside_count: int = 0
    heaviside_max_magnitude: float = 0.0
    heaviside_mean_magnitude: float = 0.0
    discontinuity_mean_interval: float = 0.0
    discontinuity_accelerating: bool = False

    def to_array(self) -> np.ndarray:
        """Flatten to numpy array for geometry calculations."""
        return np.array([
            # Memory
            self.hurst_exponent, self.acf_half_life, self.spectral_slope,
            # Information
            self.entropy_permutation, self.entropy_sample, self.entropy_rate,
            # Recurrence
            self.rqa_determinism, self.rqa_laminarity, self.rqa_entropy,
            self.rqa_trapping_time, self.rqa_max_diagonal,
            # Volatility
            self.garch_alpha, self.garch_beta, self.garch_persistence,
            self.realized_vol, self.bipower_variation, self.jump_component,
            self.hilbert_amplitude_mean, self.hilbert_amplitude_std,
            # Frequency
            self.spectral_centroid, self.spectral_bandwidth,
            self.wavelet_dominant_scale, self.wavelet_scale_entropy,
            # Dynamics
            self.lyapunov_exponent, self.embedding_dimension,
            self.correlation_dimension,
            # Momentum
            self.d1_mean, self.d1_std, self.d2_mean, self.d2_std,
            self.momentum_strength, self.hamiltonian_mean, self.hamiltonian_trend,
            # Discontinuity
            self.dirac_count, self.dirac_max_magnitude,
            self.heaviside_count, self.heaviside_max_magnitude,
        ])

    def to_dict(self) -> Dict:
        """For parquet/SQL storage."""
        return {
            # Identity
            'entity_id': self.entity_id,
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            # Memory
            'hurst_exponent': self.hurst_exponent,
            'hurst_confidence': self.hurst_confidence,
            'acf_decay_type': self.acf_decay_type,
            'acf_half_life': self.acf_half_life,
            'spectral_slope': self.spectral_slope,
            'spectral_slope_r2': self.spectral_slope_r2,
            # Information
            'entropy_permutation': self.entropy_permutation,
            'entropy_sample': self.entropy_sample,
            'entropy_rate': self.entropy_rate,
            # Recurrence
            'rqa_determinism': self.rqa_determinism,
            'rqa_laminarity': self.rqa_laminarity,
            'rqa_entropy': self.rqa_entropy,
            'rqa_recurrence_rate': self.rqa_recurrence_rate,
            'rqa_trapping_time': self.rqa_trapping_time,
            'rqa_max_diagonal': self.rqa_max_diagonal,
            'rqa_avg_diagonal': self.rqa_avg_diagonal,
            # Volatility
            'garch_alpha': self.garch_alpha,
            'garch_beta': self.garch_beta,
            'garch_persistence': self.garch_persistence,
            'garch_omega': self.garch_omega,
            'garch_unconditional': self.garch_unconditional,
            'realized_vol': self.realized_vol,
            'bipower_variation': self.bipower_variation,
            'jump_component': self.jump_component,
            'hilbert_amplitude_mean': self.hilbert_amplitude_mean,
            'hilbert_amplitude_std': self.hilbert_amplitude_std,
            # Frequency
            'spectral_centroid': self.spectral_centroid,
            'spectral_bandwidth': self.spectral_bandwidth,
            'spectral_low_high_ratio': self.spectral_low_high_ratio,
            'spectral_rolloff': self.spectral_rolloff,
            # Wavelet
            'wavelet_dominant_scale': self.wavelet_dominant_scale,
            'wavelet_scale_entropy': self.wavelet_scale_entropy,
            'wavelet_energy_ratio': self.wavelet_energy_ratio,
            # Dynamics
            'lyapunov_exponent': self.lyapunov_exponent,
            'lyapunov_confidence': self.lyapunov_confidence,
            'embedding_dimension': self.embedding_dimension,
            'correlation_dimension': self.correlation_dimension,
            # Momentum
            'd1_mean': self.d1_mean,
            'd1_std': self.d1_std,
            'd2_mean': self.d2_mean,
            'd2_std': self.d2_std,
            'momentum_strength': self.momentum_strength,
            'kinetic_energy_mean': self.kinetic_energy_mean,
            'potential_energy_mean': self.potential_energy_mean,
            'hamiltonian_mean': self.hamiltonian_mean,
            'hamiltonian_trend': self.hamiltonian_trend,
            'energy_conserved': self.energy_conserved,
            # Discontinuity
            'dirac_detected': self.dirac_detected,
            'dirac_count': self.dirac_count,
            'dirac_max_magnitude': self.dirac_max_magnitude,
            'dirac_mean_half_life': self.dirac_mean_half_life,
            'heaviside_detected': self.heaviside_detected,
            'heaviside_count': self.heaviside_count,
            'heaviside_max_magnitude': self.heaviside_max_magnitude,
            'heaviside_mean_magnitude': self.heaviside_mean_magnitude,
            'discontinuity_mean_interval': self.discontinuity_mean_interval,
            'discontinuity_accelerating': self.discontinuity_accelerating,
        }

    @classmethod
    def feature_names(cls) -> List[str]:
        """Return ordered list of feature names for ML."""
        instance = cls()
        d = instance.to_dict()
        # Remove non-numeric fields
        return [k for k, v in d.items()
                if k not in ('entity_id', 'signal_id', 'timestamp', 'acf_decay_type')]
