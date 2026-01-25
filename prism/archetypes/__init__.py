"""
Archetype Library
=================

Behavioral archetype definitions and matching algorithms.

Archetypes:
    - Stable Trend
    - Momentum Decay
    - Trending Volatile
    - Mean Reversion Stable
    - Mean Reversion Volatile
    - Random Walk
    - Consolidation
    - Chaotic
    - Edge of Chaos
    - Regime Transition
    - Post-Shock Recovery
    - Periodic
    - Quasi-Periodic
"""

from .library import ARCHETYPES, Archetype
from .matching import match_archetype, compute_fingerprint, compute_boundary_proximity
from .discovery_matrix import (
    DISCOVERY_MATRIX,
    diagnose_differential,
    generate_summary,
    generate_alerts,
)

__all__ = [
    'ARCHETYPES',
    'Archetype',
    'match_archetype',
    'compute_fingerprint',
    'compute_boundary_proximity',
    'DISCOVERY_MATRIX',
    'diagnose_differential',
    'generate_summary',
    'generate_alerts',
]
