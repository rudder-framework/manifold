"""
Hamiltonian: Total Mechanical Energy
====================================

H = T + V (Kinetic + Potential)

In a conservative system, H is constant.
When H changes -> energy injection or dissipation -> regime shift.

This is the CANARY. When H stops being conserved, something fundamental changed.

Time series interpretation:
    - T = 1/2(dx/dt)^2 = kinetic energy (energy in movement)
    - V = 1/2(x - x_bar)^2 = potential energy (deviation from equilibrium)
    - H = T + V = total mechanical energy

Key insight:
    - dH/dt ~ 0 -> Conservative system (closed, stable)
    - dH/dt > 0 -> Energy injection (driven system, external force)
    - dH/dt < 0 -> Energy dissipation (damped system, friction)

Academic references:
    - Goldstein, Classical Mechanics
    - Arnold, Mathematical Methods of Classical Mechanics
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class HamiltonianResult:
    """Output from Hamiltonian analysis"""

    # Time series
    kinetic: np.ndarray           # T(t)
    potential: np.ndarray         # V(t)
    hamiltonian: np.ndarray       # H(t) = T(t) + V(t)

    # Statistics
    T_mean: float
    T_std: float
    V_mean: float
    V_std: float
    H_mean: float
    H_std: float

    # Conservation
    H_trend: float                # dH/dt (linear trend)
    H_cv: float                   # Coefficient of variation
    conserved: bool               # Is H approximately constant?

    # Classification
    regime: str                   # 'conservative' | 'driven' | 'dissipative' | 'fluctuating'

    # Energy partition
    T_V_ratio: float              # Kinetic/Potential ratio
    dominant_energy: str          # 'kinetic' | 'potential' | 'balanced'


def compute(
    series: np.ndarray,
    equilibrium: Optional[float] = None,
    mass: float = 1.0,
    spring_constant: float = 1.0,
    window: Optional[int] = None,
    conservation_cv_threshold: float = 0.8,  # Tighter: non-oscillators need low CV
    conservation_trend_threshold: float = 0.002,
    driven_threshold: float = 0.0005,  # Looser: catch weak trends
    dissipative_threshold: float = -0.003  # Looser: catch weak damping
) -> HamiltonianResult:
    """
    Compute Hamiltonian and classify energy regime.

    Args:
        series: 1D time series (position q)
        equilibrium: Reference point for potential energy (default: mean)
        mass: Effective mass (default: 1.0)
        spring_constant: k in V = 1/2 k(x-x0)^2 (default: 1.0)
        window: Rolling window for equilibrium (default: static mean)
        conservation_cv_threshold: CV threshold for conservation (default: 0.10)
        conservation_trend_threshold: Trend threshold for conservation (default: 0.01)

    Returns:
        HamiltonianResult with full energy analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 3:
        # Return minimal result for short series
        return HamiltonianResult(
            kinetic=np.zeros(n),
            potential=np.zeros(n),
            hamiltonian=np.zeros(n),
            T_mean=0.0, T_std=0.0,
            V_mean=0.0, V_std=0.0,
            H_mean=0.0, H_std=0.0,
            H_trend=0.0, H_cv=0.0,
            conserved=True,
            regime="conservative",
            T_V_ratio=1.0,
            dominant_energy="balanced"
        )

    # Velocity (first derivative)
    velocity = np.gradient(series)

    # Equilibrium point
    if equilibrium is None:
        if window is not None and window < n:
            from scipy.ndimage import uniform_filter1d
            equilibrium = uniform_filter1d(series.astype(float), size=window, mode='nearest')
        else:
            equilibrium = np.mean(series)

    # Kinetic energy: T = 1/2 mv^2
    T = 0.5 * mass * velocity**2

    # Potential energy: V = 1/2 k(x - x0)^2
    if isinstance(equilibrium, np.ndarray):
        displacement = series - equilibrium
    else:
        displacement = series - equilibrium
    V = 0.5 * spring_constant * displacement**2

    # Hamiltonian: H = T + V
    H = T + V

    # Statistics
    T_mean, T_std = float(np.mean(T)), float(np.std(T))
    V_mean, V_std = float(np.mean(V)), float(np.std(V))
    H_mean, H_std = float(np.mean(H)), float(np.std(H))

    # Conservation check
    # For constant signals, H is near zero but perfectly conserved
    if H_mean < 1e-10 and H_std < 1e-10:
        H_cv = 0.0  # Perfect conservation (constant signal)
    elif H_mean > 1e-10:
        H_cv = H_std / H_mean
    else:
        H_cv = float('inf')

    # Trend (linear regression)
    t = np.arange(n)
    H_trend = float(np.polyfit(t, H, 1)[0])

    # Normalized trend (relative to mean)
    H_trend_normalized = H_trend / H_mean if H_mean > 1e-10 else 0

    # Conservation: Trend is the PRIMARY indicator
    # In oscillators, H varies between T and V but doesn't trend
    # CV can be high for oscillators but trend should be near zero
    trend_is_stable = abs(H_trend_normalized) < conservation_trend_threshold
    cv_is_bounded = H_cv < conservation_cv_threshold

    # Regime classification
    # Priority order: driven > dissipative > conservative > fluctuating
    #
    # Detect true oscillators via velocity autocorrelation
    # Oscillators have periodic velocity (high autocorrelation at lag > 1)
    # Random signals have uncorrelated velocity
    if n > 10:
        vel_autocorr = np.corrcoef(velocity[:-1], velocity[1:])[0, 1]
        if np.isnan(vel_autocorr):
            vel_autocorr = 0.0
    else:
        vel_autocorr = 0.0

    # True oscillators have negative velocity autocorrelation (velocity reverses)
    # or strong positive autocorrelation (smooth periodic motion)
    is_oscillatory = abs(vel_autocorr) > 0.7

    # Velocity bias detection (key for driven vs random walk)
    # Driven system: velocity has consistent direction (high bias)
    # Random walk: velocity is random (zero bias)
    vel_mean = np.mean(velocity)
    vel_std = np.std(velocity)
    velocity_bias = abs(vel_mean) / vel_std if vel_std > 1e-10 else 0.0

    # Linear trend analysis using R² to distinguish trend from random walk
    # Trend + Noise: high R² (signal follows trend closely)
    # Random Walk: low R² (signal is noisy, trend fit is poor)
    signal_slope, signal_intercept = np.polyfit(t, series, 1)
    trend_fit = signal_slope * t + signal_intercept
    ss_res = np.sum((series - trend_fit)**2)
    ss_tot = np.sum((series - np.mean(series))**2)
    trend_r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    # Net displacement for oscillatory signals
    # Driven oscillator: significant net drift from start to end
    net_displacement = series[-1] - series[0]
    signal_amplitude = np.std(series)
    normalized_displacement = abs(net_displacement) / signal_amplitude if signal_amplitude > 1e-10 else 0.0

    # Driven detection:
    # - Non-oscillatory: high R² means real trend (not random walk)
    #   R² > 0.7 distinguishes trend from random walk (random walk ~0.6)
    # - Oscillatory: net displacement AND positive H_trend means driven oscillator
    #   (without positive H_trend check, damped oscillators get misclassified)
    is_driven_nonoscillatory = not is_oscillatory and trend_r2 > 0.7
    is_driven_oscillatory = (is_oscillatory and
                             normalized_displacement > 1.0 and
                             H_trend_normalized > 0)
    is_driven = is_driven_nonoscillatory or is_driven_oscillatory
    is_dissipating = H_trend_normalized < dissipative_threshold

    # Check dissipation BEFORE driven for oscillatory signals
    # (damped oscillators can have displacement but negative H_trend)
    if is_dissipating and is_oscillatory:
        regime = "dissipative"
        conserved = False
    elif is_driven:
        regime = "driven"
        conserved = False
    elif is_dissipating:
        regime = "dissipative"
        conserved = False
    elif trend_is_stable:
        # Trend is stable - check if conservative or fluctuating
        if is_oscillatory:
            # For oscillators, high CV is normal, just need stable trend
            conserved = True
            regime = "conservative"
        elif cv_is_bounded:
            # For non-oscillators, need both stable trend AND bounded CV
            conserved = True
            regime = "conservative"
        else:
            # Non-oscillator with high CV = fluctuating
            conserved = False
            regime = "fluctuating"
    else:
        regime = "fluctuating"
        conserved = False

    # Energy partition
    T_V_ratio = T_mean / V_mean if V_mean > 1e-10 else float('inf')

    if T_V_ratio > 2:
        dominant_energy = "kinetic"
    elif T_V_ratio < 0.5:
        dominant_energy = "potential"
    else:
        dominant_energy = "balanced"

    return HamiltonianResult(
        kinetic=T,
        potential=V,
        hamiltonian=H,
        T_mean=T_mean,
        T_std=T_std,
        V_mean=V_mean,
        V_std=V_std,
        H_mean=H_mean,
        H_std=H_std,
        H_trend=H_trend,
        H_cv=H_cv,
        conserved=conserved,
        regime=regime,
        T_V_ratio=T_V_ratio,
        dominant_energy=dominant_energy
    )
