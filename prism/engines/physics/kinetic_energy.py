"""
Kinetic Energy: Energy of Motion
================================

T = 1/2 mv^2 (Kinetic energy)

Measures the energy contained in the movement of the system.

Time series interpretation:
    - v = dx/dt = rate of change
    - T = 1/2(dx/dt)^2 = energy in the movement

High kinetic energy = rapid movement
Low kinetic energy = slow or stationary

Academic reference:
    - Standard classical mechanics
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class KineticEnergyResult:
    """Output from kinetic energy analysis"""

    # Time series
    kinetic: np.ndarray           # T(t) = 1/2 mv^2
    velocity: np.ndarray          # v(t) = dx/dt

    # Statistics
    T_mean: float
    T_std: float
    T_max: float
    T_min: float

    # Velocity statistics
    v_mean: float
    v_std: float
    v_abs_mean: float             # Mean speed (absolute velocity)

    # Energy state
    energy_level: str             # 'low' | 'moderate' | 'high'


def compute(
    series: np.ndarray,
    mass: float = 1.0
) -> KineticEnergyResult:
    """
    Compute kinetic energy.

    Args:
        series: 1D time series (position)
        mass: Effective mass (default: 1.0)

    Returns:
        KineticEnergyResult with kinetic energy analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 2:
        return KineticEnergyResult(
            kinetic=np.zeros(n),
            velocity=np.zeros(n),
            T_mean=0.0, T_std=0.0, T_max=0.0, T_min=0.0,
            v_mean=0.0, v_std=0.0, v_abs_mean=0.0,
            energy_level="low"
        )

    # Velocity (first derivative)
    velocity = np.gradient(series)

    # Kinetic energy: T = 1/2 mv^2
    T = 0.5 * mass * velocity**2

    # Statistics
    T_mean = float(np.mean(T))
    T_std = float(np.std(T))
    T_max = float(np.max(T))
    T_min = float(np.min(T))

    v_mean = float(np.mean(velocity))
    v_std = float(np.std(velocity))
    v_abs_mean = float(np.mean(np.abs(velocity)))

    # Energy level classification (relative to series variance)
    series_var = np.var(series)
    if series_var > 0:
        normalized_T = T_mean / series_var
        if normalized_T < 0.1:
            energy_level = "low"
        elif normalized_T > 0.5:
            energy_level = "high"
        else:
            energy_level = "moderate"
    else:
        energy_level = "low"

    return KineticEnergyResult(
        kinetic=T,
        velocity=velocity,
        T_mean=T_mean,
        T_std=T_std,
        T_max=T_max,
        T_min=T_min,
        v_mean=v_mean,
        v_std=v_std,
        v_abs_mean=v_abs_mean,
        energy_level=energy_level
    )
