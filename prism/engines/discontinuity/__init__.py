"""
Discontinuity Engines
=====================

Structural discontinuity detection:
- dirac: Impulse detection (transient shocks that decay)
- heaviside: Step detection (permanent level shifts)
- structural: Interval analysis, acceleration detection
"""

from .dirac import compute as compute_dirac
from .heaviside import compute as compute_heaviside
from .structural import compute as compute_structural

__all__ = [
    'compute_dirac',
    'compute_heaviside',
    'compute_structural',
]
