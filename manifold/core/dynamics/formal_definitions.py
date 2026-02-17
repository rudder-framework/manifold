"""
Formal Definitions for Dynamical Systems Analysis

Classification framework for attractor types and stability analysis.

Based on:
- Strogatz (2015) Nonlinear Dynamics and Chaos
- Scheffer (2009) Critical Transitions in Nature and Society
- Ashwin et al. (2012) Tipping points in open systems

ENGINES computes metrics. Prime interprets regime transitions.
"""

from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS: CLASSIFICATION CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class AttractorType(str, Enum):
    """Type of attractor in phase space."""
    FIXED_POINT = "fixed_point"           # Single stable equilibrium
    LIMIT_CYCLE = "limit_cycle"           # Periodic orbit
    TORUS = "torus"                       # Quasi-periodic (multiple frequencies)
    STRANGE_ATTRACTOR = "strange_attractor"  # Chaotic (deterministic chaos)
    NO_ATTRACTOR = "no_attractor"         # Transient / unbounded


class StabilityType(str, Enum):
    """Stability classification based on Lyapunov exponent."""
    ASYMPTOTICALLY_STABLE = "asymptotically_stable"  # λ < -0.1
    STABLE = "stable"                                 # -0.1 < λ < -0.01
    MARGINALLY_STABLE = "marginally_stable"          # -0.01 < λ < 0.01
    WEAKLY_UNSTABLE = "weakly_unstable"              # 0.01 < λ < 0.1
    UNSTABLE = "unstable"                            # λ > 0.1 or diverging
    CHAOTIC = "chaotic"                              # λ > 0.1 with strange attractor


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_stability(lyapunov: float) -> StabilityType:
    """Classify stability based on Lyapunov exponent."""
    if lyapunov < -0.1:
        return StabilityType.ASYMPTOTICALLY_STABLE
    elif lyapunov < -0.01:
        return StabilityType.STABLE
    elif lyapunov < 0.01:
        return StabilityType.MARGINALLY_STABLE
    elif lyapunov < 0.1:
        return StabilityType.WEAKLY_UNSTABLE
    else:
        return StabilityType.CHAOTIC
