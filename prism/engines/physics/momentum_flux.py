"""
Momentum Flux: Navier-Stokes Inspired Flow Dynamics
===================================================

Inspired by the momentum conservation equation (Navier-Stokes):

d(rho*v)/dt + div(rho*v*v) = -grad(p) + mu*laplacian(v) + f

For time series, we measure analogous properties:
    - Rate of momentum change (dp/dt)
    - Momentum "pressure" (local accumulation)
    - "Viscous" drag (resistance to momentum change)
    - External forcing detection

This captures HOW momentum flows through the system.

Flow regimes:
    - Laminar: Smooth, predictable momentum flow
    - Transitional: Between laminar and turbulent
    - Turbulent: Chaotic, unpredictable momentum fluctuations

Reynolds number proxy:
    Re = inertial forces / viscous forces
    High Re -> turbulent
    Low Re -> laminar

Academic references:
    - Batchelor, "An Introduction to Fluid Dynamics"
    - Bird, Stewart, Lightfoot, "Transport Phenomena" (ChemE bible)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MomentumFluxResult:
    """Output from momentum flux analysis"""

    # Time series
    momentum: np.ndarray          # p(t) = m * v(t)
    momentum_rate: np.ndarray     # dp/dt (force)
    momentum_pressure: np.ndarray # p^2 (local momentum intensity)

    # Statistics
    p_mean: float                 # Mean momentum
    p_std: float                  # Momentum variability
    flux_mean: float              # Mean dp/dt (average force)
    flux_std: float               # Force variability

    # Flow character
    inertial: bool                # Momentum mostly conserved (persistent)
    viscous: bool                 # Strong drag (momentum opposes rate)
    forced: bool                  # External forcing detected (trend in dp/dt)

    # Turbulence analysis
    turbulent: bool               # High flux variability relative to mean
    turbulence_intensity: float   # flux_std / |flux_mean|

    # Reynolds-like analysis
    reynolds_proxy: float         # Inertial / viscous ratio

    # Flow classification
    flow_regime: str              # 'laminar' | 'transitional' | 'turbulent'
    forcing_type: str             # 'inertial' | 'viscous' | 'forced' | 'mixed'


def compute(
    series: np.ndarray,
    mass: float = 1.0,
    inertial_threshold: float = 0.5,
    viscous_threshold: float = -0.3,
    turbulence_threshold: float = 5.0,
    reynolds_laminar: float = 10.0,
    reynolds_turbulent: float = 100.0
) -> MomentumFluxResult:
    """
    Compute momentum flux characteristics (Navier-Stokes inspired).

    Args:
        series: 1D time series (position)
        mass: Effective mass (default: 1.0)
        inertial_threshold: Autocorrelation threshold for inertial (default: 0.5)
        viscous_threshold: Drag correlation threshold (default: -0.3)
        turbulence_threshold: Flux CV threshold for turbulence (default: 3.0)
        reynolds_laminar: Re threshold for laminar (default: 10)
        reynolds_turbulent: Re threshold for turbulent (default: 100)

    Returns:
        MomentumFluxResult with momentum flux analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < 5:
        return MomentumFluxResult(
            momentum=np.zeros(n),
            momentum_rate=np.zeros(n),
            momentum_pressure=np.zeros(n),
            p_mean=0.0, p_std=0.0, flux_mean=0.0, flux_std=0.0,
            inertial=False, viscous=False, forced=False,
            turbulent=False, turbulence_intensity=0.0,
            reynolds_proxy=1.0,
            flow_regime="laminar", forcing_type="mixed"
        )

    # Velocity and momentum
    velocity = np.gradient(series)
    momentum = mass * velocity

    # Momentum rate (dp/dt = force)
    momentum_rate = np.gradient(momentum)

    # Momentum "pressure" (p^2 - local momentum intensity)
    momentum_pressure = momentum**2

    # Statistics
    p_mean = float(np.mean(momentum))
    p_std = float(np.std(momentum))
    flux_mean = float(np.mean(momentum_rate))
    flux_std = float(np.std(momentum_rate))

    # === INERTIAL CHECK ===
    # Inertial: momentum is autocorrelated (persistent)
    if n > 2:
        momentum_autocorr = np.corrcoef(momentum[:-1], momentum[1:])[0, 1]
        if np.isnan(momentum_autocorr):
            momentum_autocorr = 0.0
    else:
        momentum_autocorr = 0.0

    inertial = momentum_autocorr > inertial_threshold

    # === VISCOUS CHECK ===
    # Viscous: momentum rate opposes momentum (drag/friction)
    if n > 2:
        drag_correlation = np.corrcoef(momentum[:-1], momentum_rate[1:])[0, 1]
        if np.isnan(drag_correlation):
            drag_correlation = 0.0
    else:
        drag_correlation = 0.0

    viscous = drag_correlation < viscous_threshold

    # === FORCING CHECK ===
    # Forced: momentum rate has a trend (systematic external force)
    rate_trend = float(np.polyfit(np.arange(n), momentum_rate, 1)[0])
    forced = abs(rate_trend) > 0.01 * flux_std if flux_std > 0 else False

    # === TURBULENCE CHECK ===
    # Use momentum-normalized measure when flux_mean is near zero
    # This prevents false turbulence detection in decaying signals
    flux_mean_threshold = 0.01 * p_std if p_std > 0 else 1e-10

    if abs(flux_mean) > flux_mean_threshold:
        # Normal case: ratio of variability to mean
        turbulence_intensity = flux_std / abs(flux_mean)
    else:
        # Near-zero mean: normalize by momentum std instead
        # This gives a more stable measure for oscillatory/decaying signals
        turbulence_intensity = flux_std / (p_std + 1e-10)

    # Also check for actual chaotic behavior via autocorrelation
    if n > 10:
        flux_autocorr = np.corrcoef(momentum_rate[:-1], momentum_rate[1:])[0, 1]
        if np.isnan(flux_autocorr):
            flux_autocorr = 0.0
        # Turbulent flow has low autocorrelation in force
        is_chaotic_flux = flux_autocorr < 0.3
    else:
        is_chaotic_flux = False

    # Turbulent requires BOTH high intensity AND chaotic flux pattern
    turbulent = (turbulence_intensity > turbulence_threshold) and is_chaotic_flux

    # === REYNOLDS PROXY ===
    # Re ~ inertial forces / viscous forces
    # Inertial ~ p^2 (momentum squared)
    # Viscous ~ |dp/dt| (rate of momentum change)
    inertial_force = float(np.mean(momentum**2))
    viscous_force = float(np.mean(np.abs(momentum_rate))) + 1e-10
    reynolds_proxy = inertial_force / viscous_force

    # === FLOW REGIME CLASSIFICATION ===
    if reynolds_proxy < reynolds_laminar and not turbulent:
        flow_regime = "laminar"
    elif reynolds_proxy > reynolds_turbulent or turbulent:
        flow_regime = "turbulent"
    else:
        flow_regime = "transitional"

    # === FORCING TYPE CLASSIFICATION ===
    if inertial and not viscous and not forced:
        forcing_type = "inertial"
    elif viscous and not forced:
        forcing_type = "viscous"
    elif forced:
        forcing_type = "forced"
    else:
        forcing_type = "mixed"

    return MomentumFluxResult(
        momentum=momentum,
        momentum_rate=momentum_rate,
        momentum_pressure=momentum_pressure,
        p_mean=p_mean,
        p_std=p_std,
        flux_mean=flux_mean,
        flux_std=flux_std,
        inertial=inertial,
        viscous=viscous,
        forced=forced,
        turbulent=turbulent,
        turbulence_intensity=turbulence_intensity,
        reynolds_proxy=reynolds_proxy,
        flow_regime=flow_regime,
        forcing_type=forcing_type
    )
