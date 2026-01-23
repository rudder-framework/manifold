"""
Lagrangian: Action Principle
============================

L = T - V (Kinetic MINUS Potential)

The Lagrangian is the foundation of analytical mechanics.
Hamilton's equations are derived FROM the Lagrangian via Legendre transform.

Physical interpretation:
    - L > 0: Kinetic dominates (system is moving more than displaced)
    - L < 0: Potential dominates (system is displaced more than moving)
    - L ~ 0: Balanced exchange between kinetic and potential

Action = integral(L dt)
The actual path minimizes the action (principle of least action).

Academic references:
    - Landau & Lifshitz, Mechanics
    - Goldstein, Classical Mechanics
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class LagrangianResult:
    """Output from Lagrangian analysis"""

    # Time series
    lagrangian: np.ndarray        # L(t) = T(t) - V(t)
    kinetic: np.ndarray           # T(t)
    potential: np.ndarray         # V(t)

    # Action
    action: float                 # integral(L dt) (total action)
    action_rate: float            # Average action per unit time

    # Statistics
    L_mean: float
    L_std: float
    L_trend: float

    # Dominance
    kinetic_dominant_fraction: float  # Fraction of time L > 0
    potential_dominant_fraction: float  # Fraction of time L < 0

    # Classification
    dominance: str                # 'kinetic' | 'potential' | 'balanced'


def compute(
    series: np.ndarray,
    equilibrium: Optional[float] = None,
    mass: float = 1.0,
    spring_constant: float = 1.0
) -> LagrangianResult:
    """
    Compute Lagrangian and action.

    Args:
        series: 1D time series (position q)
        equilibrium: Reference point for potential energy (default: mean)
        mass: Effective mass (default: 1.0)
        spring_constant: k in V = 1/2 k(x-x0)^2 (default: 1.0)

    Returns:
        LagrangianResult with Lagrangian analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 3:
        return LagrangianResult(
            lagrangian=np.zeros(n),
            kinetic=np.zeros(n),
            potential=np.zeros(n),
            action=0.0,
            action_rate=0.0,
            L_mean=0.0,
            L_std=0.0,
            L_trend=0.0,
            kinetic_dominant_fraction=0.5,
            potential_dominant_fraction=0.5,
            dominance="balanced"
        )

    # Velocity
    velocity = np.gradient(series)

    # Equilibrium
    if equilibrium is None:
        equilibrium = np.mean(series)

    # Kinetic energy: T = 1/2 mv^2
    T = 0.5 * mass * velocity**2

    # Potential energy: V = 1/2 k(x - x0)^2
    displacement = series - equilibrium
    V = 0.5 * spring_constant * displacement**2

    # Lagrangian: L = T - V
    L = T - V

    # Action (integral of L over time)
    action = float(np.trapezoid(L))

    # Statistics
    L_mean = float(np.mean(L))
    L_std = float(np.std(L))
    L_trend = float(np.polyfit(np.arange(n), L, 1)[0])

    # Action rate (average action per period)
    action_rate = action / n if n > 0 else 0.0

    # Dominance fractions
    kinetic_dominant_fraction = float(np.mean(L > 0))
    potential_dominant_fraction = float(np.mean(L < 0))

    # Classification
    if kinetic_dominant_fraction > 0.6:
        dominance = "kinetic"
    elif potential_dominant_fraction > 0.6:
        dominance = "potential"
    else:
        dominance = "balanced"

    return LagrangianResult(
        lagrangian=L,
        kinetic=T,
        potential=V,
        action=action,
        action_rate=action_rate,
        L_mean=L_mean,
        L_std=L_std,
        L_trend=L_trend,
        kinetic_dominant_fraction=kinetic_dominant_fraction,
        potential_dominant_fraction=potential_dominant_fraction,
        dominance=dominance
    )
