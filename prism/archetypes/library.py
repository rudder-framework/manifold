"""
Archetype Library
=================

Behavioral archetype definitions with 6D fingerprint ranges.

Each archetype is defined by expected ranges on the six orthogonal axes:
    - Memory: Hurst exponent (0=anti-persistent, 0.5=random, 1=persistent)
    - Information: Entropy (0=deterministic, 1=maximum entropy)
    - Recurrence: Determinism (0=stochastic, 1=deterministic)
    - Volatility: GARCH persistence (0=dissipating, 1=integrated)
    - Frequency: Spectral centroid (0=low freq, 1=high freq)
    - Dynamics: Lyapunov (0=stable, 0.5=edge, 1=chaotic)

Plus the Energy axis:
    - Energy: Hamiltonian conservation (0=dissipative, 0.5=conservative, 1=driven)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class AxisRange:
    """Expected range for a single axis."""
    low: float
    high: float
    weight: float = 1.0  # Importance weight for matching

    def contains(self, value: float) -> bool:
        """Check if value falls within range."""
        return self.low <= value <= self.high

    def distance(self, value: float) -> float:
        """Distance from range (0 if inside, positive if outside)."""
        if value < self.low:
            return self.low - value
        elif value > self.high:
            return value - self.high
        return 0.0


@dataclass
class Archetype:
    """
    Behavioral archetype definition.

    An archetype represents a characteristic signal behavior pattern
    defined by expected ranges on each measurement axis.
    """
    name: str
    description: str

    # Six orthogonal axes (normalized 0-1)
    memory: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))
    information: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))
    recurrence: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))
    volatility: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))
    frequency: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))
    dynamics: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))

    # Energy axis
    energy: AxisRange = field(default_factory=lambda: AxisRange(0.0, 1.0))

    # Metadata
    typical_domains: List[str] = field(default_factory=list)
    warning_signs: List[str] = field(default_factory=list)

    def distance(self, fingerprint: np.ndarray) -> float:
        """
        Compute weighted distance from fingerprint to archetype.

        Args:
            fingerprint: 6D or 7D normalized axis values

        Returns:
            Weighted Euclidean distance (0 = perfect match)
        """
        axes = [self.memory, self.information, self.recurrence,
                self.volatility, self.frequency, self.dynamics]

        if len(fingerprint) >= 7:
            axes.append(self.energy)

        total_weight = sum(ax.weight for ax in axes[:len(fingerprint)])
        weighted_distance = 0.0

        for i, ax in enumerate(axes[:len(fingerprint)]):
            d = ax.distance(fingerprint[i])
            weighted_distance += (d * ax.weight) ** 2

        return np.sqrt(weighted_distance / total_weight) if total_weight > 0 else 0.0

    def match_score(self, fingerprint: np.ndarray) -> float:
        """
        Compute match score (1 = perfect match, 0 = no match).

        Args:
            fingerprint: 6D or 7D normalized axis values

        Returns:
            Match score between 0 and 1
        """
        d = self.distance(fingerprint)
        # Convert distance to score using exponential decay
        return np.exp(-d * 2)


# =============================================================================
# ARCHETYPE DEFINITIONS
# =============================================================================

ARCHETYPES: Dict[str, Archetype] = {

    # -------------------------------------------------------------------------
    # TRENDING PATTERNS
    # -------------------------------------------------------------------------

    "stable_trend": Archetype(
        name="Stable Trend",
        description="Persistent directional movement with low volatility. "
                    "Classic trending behavior - momentum sustains.",
        memory=AxisRange(0.6, 1.0, weight=1.5),      # High persistence (H > 0.6)
        information=AxisRange(0.2, 0.5),              # Low-moderate entropy
        recurrence=AxisRange(0.4, 0.7),               # Moderate determinism
        volatility=AxisRange(0.0, 0.4),               # Low volatility
        frequency=AxisRange(0.0, 0.4),                # Low frequency dominant
        dynamics=AxisRange(0.0, 0.3),                 # Stable
        energy=AxisRange(0.4, 0.6),                   # Conservative
        typical_domains=["trending_market", "gradual_degradation", "ramp_signal"],
        warning_signs=["volatility_increase", "momentum_decay"]
    ),

    "momentum_decay": Archetype(
        name="Momentum Decay",
        description="Previously trending signal losing momentum. "
                    "Early warning of trend exhaustion or reversal.",
        memory=AxisRange(0.45, 0.65, weight=1.5),    # Transitioning from persistent
        information=AxisRange(0.4, 0.7),              # Increasing entropy
        recurrence=AxisRange(0.3, 0.6),               # Weakening patterns
        volatility=AxisRange(0.3, 0.6),               # Moderate volatility
        frequency=AxisRange(0.3, 0.6),                # Mixed frequencies
        dynamics=AxisRange(0.2, 0.5),                 # Approaching edge
        energy=AxisRange(0.2, 0.5, weight=1.5),       # Dissipating energy
        typical_domains=["trend_exhaustion", "bearing_runout", "pump_cavitation"],
        warning_signs=["energy_dissipating", "acceleration_negative"]
    ),

    "trending_volatile": Archetype(
        name="Trending Volatile",
        description="Persistent direction with high volatility. "
                    "Trend exists but with significant fluctuations.",
        memory=AxisRange(0.55, 0.85),                 # Persistent
        information=AxisRange(0.4, 0.7),              # Moderate-high entropy
        recurrence=AxisRange(0.2, 0.5),               # Lower determinism
        volatility=AxisRange(0.6, 1.0, weight=1.5),   # High volatility
        frequency=AxisRange(0.3, 0.7),                # Mixed
        dynamics=AxisRange(0.2, 0.6),                 # Moderate stability
        energy=AxisRange(0.5, 0.8),                   # Driven system
        typical_domains=["volatile_market", "turbulent_flow", "unstable_combustion"],
        warning_signs=["volatility_integrated", "jump_detected"]
    ),

    # -------------------------------------------------------------------------
    # MEAN REVERSION PATTERNS
    # -------------------------------------------------------------------------

    "mean_reversion_stable": Archetype(
        name="Mean Reversion Stable",
        description="Anti-persistent with stable volatility. "
                    "Classic oscillation around equilibrium.",
        memory=AxisRange(0.0, 0.45, weight=1.5),     # Anti-persistent (H < 0.45)
        information=AxisRange(0.3, 0.6),              # Moderate entropy
        recurrence=AxisRange(0.5, 0.8),               # High determinism
        volatility=AxisRange(0.0, 0.4),               # Low volatility
        frequency=AxisRange(0.4, 0.7),                # Mid frequencies
        dynamics=AxisRange(0.0, 0.3),                 # Stable
        energy=AxisRange(0.4, 0.6),                   # Conservative
        typical_domains=["thermostat", "servo_control", "homeostasis"],
        warning_signs=["determinism_dropping", "volatility_increasing"]
    ),

    "mean_reversion_volatile": Archetype(
        name="Mean Reversion Volatile",
        description="Anti-persistent with high volatility. "
                    "Oscillation with irregular amplitude.",
        memory=AxisRange(0.0, 0.45, weight=1.5),     # Anti-persistent
        information=AxisRange(0.5, 0.8),              # Higher entropy
        recurrence=AxisRange(0.3, 0.6),               # Moderate determinism
        volatility=AxisRange(0.5, 1.0, weight=1.5),   # High volatility
        frequency=AxisRange(0.3, 0.7),                # Mixed
        dynamics=AxisRange(0.2, 0.5),                 # Moderate
        energy=AxisRange(0.3, 0.7),                   # Variable
        typical_domains=["noisy_control", "turbulent_mixing", "erratic_sensor"],
        warning_signs=["volatility_integrated", "recurrence_dropping"]
    ),

    # -------------------------------------------------------------------------
    # RANDOM / NOISY PATTERNS
    # -------------------------------------------------------------------------

    "random_walk": Archetype(
        name="Random Walk",
        description="No memory structure - pure stochastic process. "
                    "Classic Brownian motion behavior.",
        memory=AxisRange(0.45, 0.55, weight=2.0),    # H ≈ 0.5 (critical)
        information=AxisRange(0.6, 0.9),              # High entropy
        recurrence=AxisRange(0.1, 0.4),               # Low determinism
        volatility=AxisRange(0.3, 0.7),               # Variable
        frequency=AxisRange(0.4, 0.7),                # Broadband
        dynamics=AxisRange(0.3, 0.6),                 # Edge of chaos
        energy=AxisRange(0.4, 0.6),                   # Conservative
        typical_domains=["efficient_market", "brownian_motion", "white_noise"],
        warning_signs=["memory_emerging", "determinism_increasing"]
    ),

    "consolidation": Archetype(
        name="Consolidation",
        description="Range-bound movement with decreasing volatility. "
                    "Energy building before potential breakout.",
        memory=AxisRange(0.35, 0.55),                 # Low to random
        information=AxisRange(0.3, 0.6),              # Moderate entropy
        recurrence=AxisRange(0.5, 0.8),               # High determinism (range-bound)
        volatility=AxisRange(0.0, 0.4, weight=1.5),   # Decreasing volatility
        frequency=AxisRange(0.2, 0.5),                # Lower frequencies
        dynamics=AxisRange(0.0, 0.4),                 # Stable
        energy=AxisRange(0.4, 0.7),                   # Slight accumulation
        typical_domains=["market_consolidation", "steady_state", "equilibrium"],
        warning_signs=["volatility_increasing", "energy_accumulating"]
    ),

    # -------------------------------------------------------------------------
    # CHAOTIC / COMPLEX PATTERNS
    # -------------------------------------------------------------------------

    "chaotic": Archetype(
        name="Chaotic",
        description="Positive Lyapunov exponent - sensitive dependence. "
                    "Deterministic chaos, not random.",
        memory=AxisRange(0.3, 0.7),                   # Variable
        information=AxisRange(0.6, 0.9),              # High complexity
        recurrence=AxisRange(0.4, 0.7),               # Moderate (strange attractor)
        volatility=AxisRange(0.4, 0.8),               # Moderate-high
        frequency=AxisRange(0.4, 0.8),                # Broadband
        dynamics=AxisRange(0.7, 1.0, weight=2.0),     # Chaotic (λ > 0)
        energy=AxisRange(0.3, 0.7),                   # Variable
        typical_domains=["turbulence", "weather", "combustion_instability"],
        warning_signs=["lyapunov_increasing", "dimension_increasing"]
    ),

    "edge_of_chaos": Archetype(
        name="Edge of Chaos",
        description="Critical dynamics - between order and chaos. "
                    "Maximum computational capacity, high sensitivity.",
        memory=AxisRange(0.4, 0.6),                   # Near random
        information=AxisRange(0.5, 0.8),              # High complexity
        recurrence=AxisRange(0.4, 0.7),               # Moderate
        volatility=AxisRange(0.3, 0.7),               # Variable
        frequency=AxisRange(0.3, 0.7),                # Broadband
        dynamics=AxisRange(0.4, 0.6, weight=2.0),     # Edge (λ ≈ 0)
        energy=AxisRange(0.4, 0.6),                   # Near conservative
        typical_domains=["critical_system", "phase_transition", "bifurcation"],
        warning_signs=["dynamics_shifting", "intermittency"]
    ),

    # -------------------------------------------------------------------------
    # TRANSITION PATTERNS
    # -------------------------------------------------------------------------

    "regime_transition": Archetype(
        name="Regime Transition",
        description="Active state change in progress. "
                    "Multiple axes moving simultaneously.",
        memory=AxisRange(0.3, 0.7),                   # Unstable
        information=AxisRange(0.5, 0.9),              # High entropy
        recurrence=AxisRange(0.2, 0.6),               # Breaking patterns
        volatility=AxisRange(0.5, 1.0),               # Elevated
        frequency=AxisRange(0.3, 0.8),                # Shifting
        dynamics=AxisRange(0.3, 0.7),                 # Unstable
        energy=AxisRange(0.0, 0.4, weight=2.0),       # NOT conservative (key indicator)
        typical_domains=["phase_change", "failure_onset", "market_crash"],
        warning_signs=["multiple_axes_moving", "hamiltonian_not_conserved"]
    ),

    "post_shock_recovery": Archetype(
        name="Post-Shock Recovery",
        description="System recovering from discontinuity. "
                    "Elevated volatility with mean-reverting character.",
        memory=AxisRange(0.2, 0.5),                   # Mean-reverting tendency
        information=AxisRange(0.4, 0.7),              # Moderate-high
        recurrence=AxisRange(0.3, 0.6),               # Rebuilding patterns
        volatility=AxisRange(0.6, 1.0, weight=1.5),   # Elevated but decaying
        frequency=AxisRange(0.4, 0.8),                # High frequencies present
        dynamics=AxisRange(0.1, 0.5),                 # Stabilizing
        energy=AxisRange(0.2, 0.5),                   # Dissipating
        typical_domains=["crash_recovery", "fault_clearance", "restart"],
        warning_signs=["volatility_not_decaying", "new_shock_detected"]
    ),

    # -------------------------------------------------------------------------
    # PERIODIC PATTERNS
    # -------------------------------------------------------------------------

    "periodic": Archetype(
        name="Periodic",
        description="Regular cyclic behavior - dominant frequency. "
                    "Classic oscillatory or rotational signal.",
        memory=AxisRange(0.2, 0.6),                   # Variable
        information=AxisRange(0.1, 0.4, weight=1.5),  # Low entropy (predictable)
        recurrence=AxisRange(0.7, 1.0, weight=1.5),   # High determinism
        volatility=AxisRange(0.0, 0.4),               # Low volatility
        frequency=AxisRange(0.0, 0.3, weight=1.5),    # Narrowband
        dynamics=AxisRange(0.0, 0.3),                 # Stable
        energy=AxisRange(0.45, 0.55),                 # Conservative
        typical_domains=["rotation", "heartbeat", "seasonal", "ac_power"],
        warning_signs=["frequency_shifting", "amplitude_varying"]
    ),

    "quasi_periodic": Archetype(
        name="Quasi-Periodic",
        description="Multiple incommensurate frequencies. "
                    "Regular but non-repeating - torus attractor.",
        memory=AxisRange(0.3, 0.6),                   # Moderate
        information=AxisRange(0.3, 0.6),              # Moderate
        recurrence=AxisRange(0.5, 0.8),               # High but not maximal
        volatility=AxisRange(0.1, 0.5),               # Low-moderate
        frequency=AxisRange(0.3, 0.6),                # Multiple peaks
        dynamics=AxisRange(0.0, 0.4),                 # Stable
        energy=AxisRange(0.4, 0.6),                   # Conservative
        typical_domains=["modulation", "beat_frequency", "coupled_oscillators"],
        warning_signs=["frequency_locking", "chaos_onset"]
    ),
}


def get_archetype(name: str) -> Optional[Archetype]:
    """Get archetype by name (case-insensitive, underscores optional)."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    return ARCHETYPES.get(key)


def list_archetypes() -> List[str]:
    """Return list of all archetype names."""
    return list(ARCHETYPES.keys())
