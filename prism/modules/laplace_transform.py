"""
Running Laplace Transform
=========================

Incremental Laplace transform that updates O(1) per observation.

F(s, t) = ∫₀ᵗ f(τ) e^(-sτ) dτ

With O(1) update rule:
F(s, t+Δt) = F(s, t) + f(t+Δt) × e^(-s×(t+Δt)) × Δt

This provides the temporal backbone for PRISM - capturing both
time evolution and frequency content in a single computation.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from prism.modules.signals.types import LaplaceField, DenseSignal


# =============================================================================
# RUNNING LAPLACE TRANSFORM
# =============================================================================

@dataclass
class RunningLaplace:
    """
    Incremental Laplace transform with O(1) update per observation.

    The Laplace transform captures frequency content while preserving
    causality - only past observations contribute to F(s, t).

    Usage:
        laplace = RunningLaplace(s_values=[0.01, 0.1, 1.0, 10.0])
        for t, value in observations:
            laplace.update(t, value)
        field = laplace.get_field()

    Parameters:
        s_values: Complex frequency values to compute
                  Real part = decay rate, Imaginary part = oscillation
        normalize: Whether to normalize by time span (default True)
    """
    s_values: np.ndarray = field(default_factory=lambda: np.array([
        0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    ]))
    normalize: bool = True

    # Internal state (initialized on first update)
    _signal_id: str = ""
    _timestamps: List[datetime] = field(default_factory=list)
    _values: List[float] = field(default_factory=list)
    _field_history: List[np.ndarray] = field(default_factory=list)
    _current_field: Optional[np.ndarray] = None
    _t0: Optional[float] = None
    _initialized: bool = False

    def __post_init__(self):
        self.s_values = np.asarray(self.s_values, dtype=np.complex128)
        if len(self.s_values) == 0:
            raise ValueError("s_values must not be empty")

    def reset(self, signal_id: str = "") -> None:
        """Reset state for new signal."""
        self._signal_id = signal_id
        self._timestamps = []
        self._values = []
        self._field_history = []
        self._current_field = np.zeros(len(self.s_values), dtype=np.complex128)
        self._t0 = None
        self._initialized = False

    def _to_numeric_time(self, t: Union[datetime, float, int]) -> float:
        """Convert timestamp to numeric value."""
        if isinstance(t, datetime):
            return t.timestamp()
        return float(t)

    def update(self, t: Union[datetime, float, int], value: float) -> np.ndarray:
        """
        Update Laplace transform with new observation.

        F(s, t+Δt) = F(s, t) + f(t+Δt) × e^(-s×(t+Δt)) × Δt

        Returns the current field values (one per s value).
        """
        t_numeric = self._to_numeric_time(t)

        if not self._initialized:
            self._current_field = np.zeros(len(self.s_values), dtype=np.complex128)
            self._t0 = t_numeric
            self._initialized = True

        # Compute relative time from start
        tau = t_numeric - self._t0

        # Compute Δt
        if len(self._timestamps) > 0:
            t_prev = self._to_numeric_time(self._timestamps[-1])
            dt = t_numeric - t_prev
        else:
            dt = 1.0  # First observation, assume unit time step

        # Update: F(s, t) += value × e^(-s×tau) × dt
        exponential = np.exp(-self.s_values * tau)
        self._current_field += value * exponential * dt

        # Store history
        self._timestamps.append(t)
        self._values.append(value)
        self._field_history.append(self._current_field.copy())

        return self._current_field.copy()

    def get_current(self) -> np.ndarray:
        """Get current Laplace field values."""
        if self._current_field is None:
            return np.zeros(len(self.s_values), dtype=np.complex128)
        return self._current_field.copy()

    def get_field(self) -> LaplaceField:
        """
        Get complete LaplaceField structure.

        Returns 2D field [n_timestamps × n_s] with all history.
        """
        if len(self._timestamps) == 0:
            return LaplaceField(
                signal_id=self._signal_id,
                timestamps=np.array([]),
                s_values=self.s_values,
                field=np.zeros((0, len(self.s_values)), dtype=np.complex128),
            )

        # Stack history into 2D array [n_t × n_s]
        field_array = np.row_stack(self._field_history)

        # Normalize if requested
        if self.normalize and len(self._timestamps) > 1:
            t_span = self._to_numeric_time(self._timestamps[-1]) - self._to_numeric_time(self._timestamps[0])
            if t_span > 0:
                field_array = field_array / t_span

        return LaplaceField(
            signal_id=self._signal_id,
            timestamps=np.array(self._timestamps),
            s_values=self.s_values,
            field=field_array,
        )

    def get_magnitude_at(self, t: Union[datetime, float, int]) -> np.ndarray:
        """Get magnitude |F(s,t)| at specific timestamp."""
        idx = None
        for i, ts in enumerate(self._timestamps):
            if self._to_numeric_time(ts) == self._to_numeric_time(t):
                idx = i
                break

        if idx is None:
            return np.full(len(self.s_values), np.nan)

        return np.abs(self._field_history[idx])

    def get_dominant_frequency_at(self, t: Union[datetime, float, int]) -> float:
        """Get dominant frequency (s with max magnitude) at timestamp."""
        magnitudes = self.get_magnitude_at(t)
        if np.all(np.isnan(magnitudes)):
            return np.nan
        idx = np.argmax(magnitudes)
        return float(np.real(self.s_values[idx]))


# =============================================================================
# BATCH COMPUTATION
# =============================================================================

def compute_laplace_field(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    s_values: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> LaplaceField:
    """
    Compute Laplace field for a complete signal.

    This is the batch version - use RunningLaplace for streaming.

    Parameters:
        signal_id: Signal identifier
        timestamps: Array of timestamps
        values: Array of signal values
        s_values: Complex frequency values (default: logarithmic range)
        normalize: Whether to normalize by time span

    Returns:
        LaplaceField with complete 2D structure
    """
    if s_values is None:
        # Default logarithmic range covering multiple scales
        s_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    laplace = RunningLaplace(s_values=s_values, normalize=normalize)
    laplace.reset(signal_id)

    for t, v in zip(timestamps, values):
        laplace.update(t, v)

    return laplace.get_field()


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def laplace_gradient(field: LaplaceField) -> DenseSignal:
    """
    Compute temporal gradient of Laplace field.

    ∂F/∂t represents the "velocity" in Laplace space -
    how fast the frequency content is changing.

    Returns the norm of gradient across all frequencies.
    """
    grad = field.gradient_t()  # [n_t × n_s]
    grad_norm = np.linalg.norm(grad, axis=1)  # Norm across s values

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_grad",
        timestamps=field.timestamps,
        values=grad_norm.real,  # Real part of gradient norm
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'gradient_norm'},
    )


def laplace_divergence(field: LaplaceField) -> DenseSignal:
    """
    Compute divergence in frequency space.

    Positive divergence = energy dispersing (healthy)
    Negative divergence = energy concentrating (degradation)
    """
    div = field.divergence_at_t()  # Returns array of divergence per timestamp

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_div",
        timestamps=field.timestamps,
        values=div.real,  # Real part of divergence
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'divergence'},
    )


def laplace_energy(field: LaplaceField) -> DenseSignal:
    """
    Total energy in Laplace field at each timestamp.

    E(t) = Σ_s |F(s,t)|²

    Increasing energy = more pronounced frequency components
    Decreasing energy = signal becoming more uniform
    """
    energy = field.total_energy

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_energy",
        timestamps=field.timestamps,
        values=energy,
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'total_energy'},
    )


# =============================================================================
# SCALE-SPECIFIC ANALYSIS
# =============================================================================

def decompose_by_scale(
    field: LaplaceField,
    scale_boundaries: Optional[List[float]] = None,
) -> List[DenseSignal]:
    """
    Decompose Laplace field by frequency scales.

    Parameters:
        field: LaplaceField to decompose
        scale_boundaries: Boundaries between scales (default: [0.1, 1.0, 10.0])

    Returns:
        List of DenseSignals, one per scale band
    """
    if scale_boundaries is None:
        scale_boundaries = [0.1, 1.0, 10.0]

    s_real = np.real(field.s_values)
    signals = []

    # Add lower bound
    boundaries = [0.0] + list(scale_boundaries) + [np.inf]

    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (s_real >= low) & (s_real < high)

        if not np.any(mask):
            continue

        # Extract magnitude for this scale band
        # field.magnitude has shape [n_t × n_s], so sum over masked s values
        band_magnitude = np.sum(field.magnitude[:, mask], axis=1)

        scale_name = f"scale_{low:.2f}_{high:.2f}"
        signals.append(DenseSignal(
            signal_id=f"{field.signal_id}_{scale_name}",
            timestamps=field.timestamps,
            values=band_magnitude,
            source_signal=field.signal_id,
            engine='laplace',
            parameters={'scale_low': low, 'scale_high': high},
        ))

    return signals


def frequency_shift(field: LaplaceField) -> DenseSignal:
    """
    Track shift in dominant frequency over time.

    Frequency shift is an early indicator of regime change -
    the system starts resonating at different frequencies before
    behavior visibly changes.
    """
    dominant = field.dominant_frequency()

    # Compute shift as first difference
    shifts = np.diff(dominant.values, prepend=dominant.values[0])

    return DenseSignal(
        signal_id=f"{field.signal_id}_freq_shift",
        timestamps=field.timestamps,
        values=shifts,
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'frequency_shift'},
    )
