"""
PRISM Physics Engines
=====================

System physics computations based on classical mechanics and thermodynamics.

Engines:
    - hamiltonian: Total energy (H = T + V), conservation detection
    - lagrangian: Action principle (L = T - V)
    - kinetic_energy: Energy of motion (T = 1/2 mv^2)
    - potential_energy: Energy of position (V = 1/2 k(x-x0)^2)
    - gibbs_free_energy: Spontaneity and equilibrium (G = H - TS)
    - angular_momentum: Cyclical dynamics (L = q x p)
    - momentum_flux: Flow dynamics (Navier-Stokes inspired)
    - derivatives: Velocity, acceleration, jerk analysis

Key insight:
    The Hamiltonian is the canary - when energy stops being conserved,
    something fundamental has changed in the system.

    Gibbs free energy tells you WHERE the system WANTS TO GO.
    dG < 0 -> spontaneous movement toward equilibrium.
"""

from typing import Dict, Any, Optional
import numpy as np

from .hamiltonian import compute as compute_hamiltonian, HamiltonianResult
from .lagrangian import compute as compute_lagrangian, LagrangianResult
from .kinetic_energy import compute as compute_kinetic, KineticEnergyResult
from .potential_energy import compute as compute_potential, PotentialEnergyResult
from .gibbs_free_energy import compute as compute_gibbs, GibbsResult
from .angular_momentum import compute as compute_angular_momentum, AngularMomentumResult
from .momentum_flux import compute as compute_momentum_flux, MomentumFluxResult
from .derivatives import compute as compute_derivatives


# =============================================================================
# Dict Adapters for signal_typology.py compatibility
# =============================================================================

def kinetic_energy_dict(series: np.ndarray, mass: float = 1.0) -> Dict[str, Any]:
    """
    Compute kinetic energy and return as dict for signal_typology compatibility.

    Args:
        series: 1D time series
        mass: Effective mass

    Returns:
        Dict with keys: mean, std, max, trend, total
    """
    result = compute_kinetic(series, mass=mass)
    return {
        'mean': result.T_mean,
        'std': result.T_std,
        'max': result.T_max,
        'trend': 0.0,  # Not computed in dataclass version
        'total': result.T_mean * len(series),
    }


def potential_energy_dict(
    series: np.ndarray,
    equilibrium: Optional[float] = None,
    spring_constant: float = 1.0
) -> Dict[str, Any]:
    """
    Compute potential energy and return as dict for signal_typology compatibility.

    Args:
        series: 1D time series
        equilibrium: Equilibrium point (default: mean)
        spring_constant: Spring constant k

    Returns:
        Dict with keys: mean, std, max, trend, total
    """
    result = compute_potential(series, equilibrium=equilibrium, spring_constant=spring_constant)
    return {
        'mean': result.V_mean,
        'std': result.V_std,
        'max': result.V_max,
        'trend': 0.0,  # Not computed in dataclass version
        'total': result.V_mean * len(series),
    }


def hamiltonian_dict(
    series: np.ndarray,
    equilibrium: Optional[float] = None,
    conservation_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Compute Hamiltonian and return as dict for signal_typology compatibility.

    Args:
        series: 1D time series
        equilibrium: Equilibrium point for potential energy
        conservation_threshold: Threshold for conservation detection

    Returns:
        Dict with keys: mean, std, trend, trend_normalized, conserved, regime, kinetic_ratio, energy_flow
    """
    result = compute_hamiltonian(series, equilibrium=equilibrium)
    return {
        'mean': result.H_mean,
        'std': result.H_std,
        'trend': result.H_trend,
        'trend_normalized': result.H_trend / (result.H_mean + 1e-10) if result.H_mean > 0 else 0.0,
        'conserved': result.conserved,
        'regime': result.regime,
        'kinetic_ratio': result.T_V_ratio / (1 + result.T_V_ratio) if result.T_V_ratio != float('inf') else 1.0,
        'energy_flow': 0.0,  # Not computed in dataclass version
    }


__all__ = [
    # Compute functions (dataclass returns)
    'compute_hamiltonian',
    'compute_lagrangian',
    'compute_kinetic',
    'compute_potential',
    'compute_gibbs',
    'compute_angular_momentum',
    'compute_momentum_flux',
    'compute_derivatives',

    # Dict adapters for signal_typology compatibility
    'kinetic_energy_dict',
    'potential_energy_dict',
    'hamiltonian_dict',

    # Result dataclasses
    'HamiltonianResult',
    'LagrangianResult',
    'KineticEnergyResult',
    'PotentialEnergyResult',
    'GibbsResult',
    'AngularMomentumResult',
    'MomentumFluxResult',
]
