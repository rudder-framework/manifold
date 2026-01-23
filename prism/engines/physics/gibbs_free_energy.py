"""
Gibbs Free Energy: Spontaneity and Equilibrium
==============================================

G = H - TS (Enthalpy minus Temperature x Entropy)

ChemE's key equation for "will this happen spontaneously?"

dG < 0 -> Spontaneous (system will move that direction naturally)
dG = 0 -> At equilibrium
dG > 0 -> Non-spontaneous (requires external energy input)

Time series interpretation:
    - H = Hamiltonian (total mechanical energy)
    - T = "Temperature" -> volatility (intensity of random fluctuations)
    - S = Information entropy (disorder/complexity)
    - G = H - T*S = "Free energy available for directed work"

Key insight:
    When G is decreasing -> system moving toward equilibrium spontaneously
    When G is increasing -> system being driven away from equilibrium
    When G is flat -> system at equilibrium (or steady state)

This is the ChemE addition that regular physics thermo doesn't emphasize.
It tells you not just WHERE the system is, but WHERE IT WANTS TO GO.

Academic references:
    - Gibbs, "On the Equilibrium of Heterogeneous Substances" (1876)
    - Smith, Van Ness, Abbott, "Introduction to Chemical Engineering Thermodynamics"
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class GibbsResult:
    """Output from Gibbs free energy analysis"""

    # Time series
    gibbs: np.ndarray             # G(t) = H(t) - T(t)*S(t)
    enthalpy: np.ndarray          # H(t) (Hamiltonian proxy)
    temperature: np.ndarray       # T(t) (volatility proxy)
    entropy: np.ndarray           # S(t) (information entropy proxy)

    # Statistics
    G_mean: float
    G_std: float
    G_trend: float                # dG/dt (direction of spontaneous change)

    # Components
    H_mean: float                 # Mean enthalpy
    T_mean: float                 # Mean temperature (volatility)
    S_mean: float                 # Mean entropy
    TS_mean: float                # Mean temperature-entropy product

    # Spontaneity analysis
    delta_G: float                # G_final - G_initial
    spontaneous: bool             # Is dG < 0?

    # Equilibrium analysis
    equilibrium_distance: float   # How far from equilibrium
    moving_toward_equilibrium: bool  # Is G decreasing?

    # Classification
    equilibrium_class: str        # 'approaching' | 'at_equilibrium' | 'departing' | 'forced'


def compute(
    series: np.ndarray,
    entropy_window: int = 20,
    volatility_window: int = 20,
    equilibrium: Optional[float] = None
) -> GibbsResult:
    """
    Compute Gibbs free energy analogue for time series.

    The mapping:
        - H (enthalpy) -> Hamiltonian (total mechanical energy)
        - T (temperature) -> Local volatility (randomness intensity)
        - S (entropy) -> Local information entropy

    Args:
        series: 1D time series
        entropy_window: Window for local entropy calculation
        volatility_window: Window for local volatility (temperature)
        equilibrium: Reference point for potential energy (default: mean)

    Returns:
        GibbsResult with Gibbs free energy analysis
    """
    series = np.asarray(series).flatten()
    n = len(series)

    if n < max(entropy_window, volatility_window) + 5:
        return GibbsResult(
            gibbs=np.zeros(n),
            enthalpy=np.zeros(n),
            temperature=np.ones(n),
            entropy=np.zeros(n),
            G_mean=0.0, G_std=0.0, G_trend=0.0,
            H_mean=0.0, T_mean=1.0, S_mean=0.0, TS_mean=0.0,
            delta_G=0.0, spontaneous=False,
            equilibrium_distance=0.0, moving_toward_equilibrium=False,
            equilibrium_class="at_equilibrium"
        )

    # === ENTHALPY (H) = Hamiltonian ===
    velocity = np.gradient(series)
    if equilibrium is None:
        equilibrium = np.mean(series)

    T_kinetic = 0.5 * velocity**2
    V_potential = 0.5 * (series - equilibrium)**2
    H = T_kinetic + V_potential

    # === TEMPERATURE = Rolling volatility (local randomness intensity) ===
    from scipy.ndimage import uniform_filter1d

    rolling_mean = uniform_filter1d(series.astype(float), size=volatility_window, mode='nearest')
    rolling_var = uniform_filter1d((series - rolling_mean)**2, size=volatility_window, mode='nearest')
    temperature = np.sqrt(np.maximum(rolling_var, 1e-10))

    # === ENTROPY = Local permutation entropy proxy ===
    entropy = np.zeros(n)

    for i in range(entropy_window, n):
        segment = series[i-entropy_window:i]

        # Simple entropy measure: normalized histogram entropy
        diffs = np.diff(segment)
        if np.std(diffs) > 1e-10:
            normalized = diffs / np.std(diffs)
            # Bin into categories
            hist, _ = np.histogram(normalized, bins=10, range=(-3, 3))
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy[i] = -np.sum(hist * np.log(hist)) / np.log(10)  # Normalize by max
            else:
                entropy[i] = 0.0
        else:
            entropy[i] = 0.0

    # Fill initial values
    if entropy_window < n:
        entropy[:entropy_window] = entropy[entropy_window]

    # Normalize entropy to [0, 1] range approximately
    entropy = np.clip(entropy, 0, 1)

    # === GIBBS FREE ENERGY: G = H - T*S ===
    G = H - temperature * entropy

    # Statistics
    G_mean = float(np.mean(G))
    G_std = float(np.std(G))

    # Trend (dG/dt via linear regression)
    t = np.arange(n)
    G_trend = float(np.polyfit(t, G, 1)[0])

    # Component means
    H_mean = float(np.mean(H))
    T_mean = float(np.mean(temperature))
    S_mean = float(np.mean(entropy))
    TS_mean = float(np.mean(temperature * entropy))

    # Spontaneity: dG < 0 means spontaneous
    delta_G = float(G[-1] - G[0])
    spontaneous = delta_G < 0

    # Equilibrium analysis - use AMPLITUDE trend as primary indicator
    # A system is approaching equilibrium if its amplitude is decreasing
    displacement = series - equilibrium
    amplitude = np.abs(displacement)
    amplitude_trend = float(np.polyfit(t, amplitude, 1)[0])
    amplitude_trend_normalized = amplitude_trend / (np.mean(amplitude) + 1e-10)

    equilibrium_distance = float(np.mean(amplitude))
    moving_toward_equilibrium = amplitude_trend < 0

    # Classification based on amplitude dynamics
    amplitude_threshold = 0.005

    if abs(amplitude_trend_normalized) < amplitude_threshold:
        # Amplitude stable - check if at equilibrium or oscillating
        if equilibrium_distance < 0.1 * np.std(series):
            equilibrium_class = "at_equilibrium"
        else:
            equilibrium_class = "at_equilibrium"  # Stable oscillation
    elif amplitude_trend_normalized < -amplitude_threshold:
        # Amplitude decreasing - approaching equilibrium
        equilibrium_class = "approaching"
    elif amplitude_trend_normalized > amplitude_threshold:
        # Amplitude increasing - departing from equilibrium
        if G_trend > 0 and not spontaneous:
            equilibrium_class = "forced"  # Being driven away
        else:
            equilibrium_class = "departing"
    else:
        equilibrium_class = "at_equilibrium"

    return GibbsResult(
        gibbs=G,
        enthalpy=H,
        temperature=temperature,
        entropy=entropy,
        G_mean=G_mean,
        G_std=G_std,
        G_trend=G_trend,
        H_mean=H_mean,
        T_mean=T_mean,
        S_mean=S_mean,
        TS_mean=TS_mean,
        delta_G=delta_G,
        spontaneous=spontaneous,
        equilibrium_distance=equilibrium_distance,
        moving_toward_equilibrium=moving_toward_equilibrium,
        equilibrium_class=equilibrium_class
    )
