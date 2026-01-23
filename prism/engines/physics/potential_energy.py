"""
Potential Energy: Energy of Position
====================================

V = 1/2 k(x - x0)^2 (Harmonic potential)

Measures the energy stored in the displacement from equilibrium.

Time series interpretation:
    - x0 = equilibrium point (mean, moving average, or specified)
    - x - x0 = displacement from equilibrium
    - V = energy stored in the displacement

High potential energy = far from equilibrium
Low potential energy = near equilibrium

The potential energy creates a "restoring force" that pulls
the system back toward equilibrium.

Academic reference:
    - Standard classical mechanics (harmonic oscillator)
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class PotentialEnergyResult:
    """Output from potential energy analysis"""

    # Time series
    potential: np.ndarray         # V(t) = 1/2 k(x - x0)^2
    displacement: np.ndarray      # x(t) - x0

    # Statistics
    V_mean: float
    V_std: float
    V_max: float
    V_min: float

    # Displacement statistics
    displacement_mean: float      # Should be ~0 if equilibrium is correct
    displacement_std: float
    displacement_max: float       # Maximum excursion

    # Equilibrium
    equilibrium: float            # The equilibrium point used
    equilibrium_type: str         # 'static' | 'rolling' | 'specified'

    # Energy state
    energy_level: str             # 'low' | 'moderate' | 'high'

    # Restoring force
    mean_restoring_force: float   # -k * displacement (average)


def compute(
    series: np.ndarray,
    equilibrium: Optional[Union[float, np.ndarray]] = None,
    spring_constant: float = 1.0,
    rolling_window: Optional[int] = None
) -> PotentialEnergyResult:
    """
    Compute potential energy.

    Args:
        series: 1D time series (position)
        equilibrium: Reference point (default: mean of series)
        spring_constant: k in V = 1/2 k(x-x0)^2 (default: 1.0)
        rolling_window: If specified, use rolling mean as equilibrium

    Returns:
        PotentialEnergyResult with potential energy analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 2:
        return PotentialEnergyResult(
            potential=np.zeros(n),
            displacement=np.zeros(n),
            V_mean=0.0, V_std=0.0, V_max=0.0, V_min=0.0,
            displacement_mean=0.0, displacement_std=0.0, displacement_max=0.0,
            equilibrium=0.0, equilibrium_type="static",
            energy_level="low", mean_restoring_force=0.0
        )

    # Determine equilibrium
    if equilibrium is not None:
        eq = equilibrium
        eq_type = "specified"
    elif rolling_window is not None and rolling_window < n:
        from scipy.ndimage import uniform_filter1d
        eq = uniform_filter1d(series.astype(float), size=rolling_window, mode='nearest')
        eq_type = "rolling"
    else:
        eq = np.mean(series)
        eq_type = "static"

    # Displacement from equilibrium
    displacement = series - eq

    # Potential energy: V = 1/2 k(x - x0)^2
    V = 0.5 * spring_constant * displacement**2

    # Statistics
    V_mean = float(np.mean(V))
    V_std = float(np.std(V))
    V_max = float(np.max(V))
    V_min = float(np.min(V))

    displacement_mean = float(np.mean(displacement))
    displacement_std = float(np.std(displacement))
    displacement_max = float(np.max(np.abs(displacement)))

    # Equilibrium value (scalar for reporting)
    if isinstance(eq, np.ndarray):
        equilibrium_val = float(np.mean(eq))
    else:
        equilibrium_val = float(eq)

    # Energy level classification
    series_var = np.var(series)
    if series_var > 0:
        normalized_V = V_mean / series_var
        if normalized_V < 0.1:
            energy_level = "low"
        elif normalized_V > 0.5:
            energy_level = "high"
        else:
            energy_level = "moderate"
    else:
        energy_level = "low"

    # Mean restoring force: F = -kx
    mean_restoring_force = float(-spring_constant * np.mean(displacement))

    return PotentialEnergyResult(
        potential=V,
        displacement=displacement,
        V_mean=V_mean,
        V_std=V_std,
        V_max=V_max,
        V_min=V_min,
        displacement_mean=displacement_mean,
        displacement_std=displacement_std,
        displacement_max=displacement_max,
        equilibrium=equilibrium_val,
        equilibrium_type=eq_type,
        energy_level=energy_level,
        mean_restoring_force=mean_restoring_force
    )
