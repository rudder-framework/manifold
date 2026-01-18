"""
PRISM Geometry Module
=====================

Compute system geometry at each timestamp from Laplace fields.
"""

from prism.geometry.snapshot import compute_geometry_at_t, compute_geometry_trajectory
from prism.geometry.coupling import compute_coupling_matrix
from prism.geometry.divergence import compute_divergence
from prism.geometry.modes import discover_modes

__all__ = [
    'compute_geometry_at_t',
    'compute_geometry_trajectory',
    'compute_coupling_matrix',
    'compute_divergence',
    'discover_modes',
]
