"""
Angular Momentum: Cyclical Dynamics
===================================

L = r x p = q x (m * dq/dt)

In 2D phase space (position vs momentum): L = q * p (scalar)

Measures "rotation" in phase space:
    - L > 0: Counter-clockwise rotation (position and momentum same sign)
    - L < 0: Clockwise rotation (position and momentum opposite sign)
    - |L| large: Strong cyclical behavior
    - |L| small: Weak or no cyclical behavior

For oscillators:
    - Constant |L| -> circular orbit (perfect oscillation)
    - Varying |L| -> elliptical or irregular orbit
    - L sign changes -> direction reversals in phase space

Physical insight:
    Angular momentum is conserved in central force systems.
    When L changes -> external torque (forcing) or friction.

Academic references:
    - Goldstein, Classical Mechanics (Chapter on Central Forces)
    - Strogatz, Nonlinear Dynamics and Chaos
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AngularMomentumResult:
    """Output from angular momentum analysis"""

    # Time series
    angular_momentum: np.ndarray  # L(t) = q(t) * p(t)
    position: np.ndarray          # q(t) relative to equilibrium
    momentum: np.ndarray          # p(t) = m * v(t)

    # Statistics
    L_mean: float                 # Mean (signed)
    L_std: float                  # Variability
    L_abs_mean: float             # Mean magnitude
    L_max: float
    L_min: float

    # Cyclical character
    rotation_direction: str       # 'counterclockwise' | 'clockwise' | 'mixed'
    sign_changes: int             # Number of direction reversals
    sign_change_rate: float       # Reversals per unit time

    # Orbit shape (in phase space)
    orbit_circularity: float      # 0 = linear, 1 = circular
    orbit_stability: float        # Consistency of |L|

    # Conservation
    L_conserved: bool             # Is |L| approximately constant?
    L_cv: float                   # Coefficient of variation of |L|


def compute(
    series: np.ndarray,
    equilibrium: Optional[float] = None,
    mass: float = 1.0,
    conservation_threshold: float = 0.15
) -> AngularMomentumResult:
    """
    Compute angular momentum in phase space.

    Args:
        series: 1D time series (position)
        equilibrium: Reference point (default: mean)
        mass: Effective mass (default: 1.0)
        conservation_threshold: CV threshold for conservation (default: 0.15)

    Returns:
        AngularMomentumResult with angular momentum analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 3:
        return AngularMomentumResult(
            angular_momentum=np.zeros(n),
            position=np.zeros(n),
            momentum=np.zeros(n),
            L_mean=0.0, L_std=0.0, L_abs_mean=0.0, L_max=0.0, L_min=0.0,
            rotation_direction="mixed",
            sign_changes=0, sign_change_rate=0.0,
            orbit_circularity=0.0, orbit_stability=0.0,
            L_conserved=True, L_cv=0.0
        )

    # Position relative to equilibrium
    if equilibrium is None:
        equilibrium = np.mean(series)
    q = series - equilibrium

    # Velocity and momentum
    velocity = np.gradient(series)
    p = mass * velocity

    # Angular momentum: L = q x p (scalar in 2D phase space)
    L = q * p

    # Statistics
    L_mean = float(np.mean(L))
    L_std = float(np.std(L))
    L_abs = np.abs(L)
    L_abs_mean = float(np.mean(L_abs))
    L_max = float(np.max(L))
    L_min = float(np.min(L))

    # Rotation direction
    if L_abs_mean > 1e-10:
        direction_ratio = L_mean / L_abs_mean
        if direction_ratio > 0.3:
            rotation_direction = "counterclockwise"
        elif direction_ratio < -0.3:
            rotation_direction = "clockwise"
        else:
            rotation_direction = "mixed"
    else:
        rotation_direction = "mixed"

    # Sign changes (direction reversals)
    signs = np.sign(L)
    sign_changes = int(np.sum(np.diff(signs) != 0))
    sign_change_rate = sign_changes / n if n > 0 else 0.0

    # Orbit circularity - improved detection for oscillators
    # For a perfect oscillator, L oscillates between positive and negative
    # but has consistent magnitude at peaks
    #
    # Better approach: measure consistency of |L| at its peaks
    # rather than overall min/max ratio

    series_std = float(np.std(series))
    has_rotation = L_abs_mean > 0.01 * series_std

    if has_rotation:
        # Find peaks in |L| to measure orbit consistency
        # A circular orbit has consistent peak magnitudes
        L_abs_smooth = np.convolve(L_abs, np.ones(3)/3, mode='same')

        # Use CV of |L| as inverse circularity measure
        # Low CV = consistent magnitude = circular
        # High CV = varying magnitude = elliptical/irregular
        L_cv = float(np.std(L_abs) / L_abs_mean) if L_abs_mean > 1e-10 else float('inf')

        # Also check sign change pattern - regular oscillation has periodic sign changes
        if sign_change_rate > 0.02:  # Has regular direction changes
            # Periodic sign changes suggest oscillatory behavior
            # Circularity based on CV: CV=0 -> circularity=1, CV=2 -> circularity~0.33
            orbit_circularity = 1.0 / (1.0 + L_cv)
        else:
            # Few sign changes - more linear or trending
            orbit_circularity = 0.5 / (1.0 + L_cv)
    else:
        # No significant angular momentum - linear motion
        orbit_circularity = 0.0
        L_cv = float('inf')

    # Orbit stability: inverse of CV of |L|
    orbit_stability = 1.0 / (1.0 + L_cv) if L_cv < float('inf') else 0.0

    # Conservation check
    L_conserved = L_cv < conservation_threshold

    return AngularMomentumResult(
        angular_momentum=L,
        position=q,
        momentum=p,
        L_mean=L_mean,
        L_std=L_std,
        L_abs_mean=L_abs_mean,
        L_max=L_max,
        L_min=L_min,
        rotation_direction=rotation_direction,
        sign_changes=sign_changes,
        sign_change_rate=sign_change_rate,
        orbit_circularity=orbit_circularity,
        orbit_stability=orbit_stability,
        L_conserved=L_conserved,
        L_cv=L_cv
    )
