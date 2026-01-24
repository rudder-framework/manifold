"""
Manifold Geometry
=================

One of the four ORTHON analytical frameworks:

    Signal Typology     -> What IS this signal?
    Manifold Geometry   -> What is its STRUCTURE? (this framework)
    Dynamical Systems   -> How does the SYSTEM evolve?
    Causal Mechanics    -> What DRIVES the system?

Manifold Geometry analyzes relational structure between signals using
differential geometry concepts including Ricci curvature:
    - Correlation: Who moves with whom?
    - Clustering: What groups exist?
    - Network: What is the topology?
    - Curvature: How bent is the manifold? (fragility indicator)
    - Causality: Who leads whom?
    - Decoupling: Are relationships breaking?

Key insight (Sandhu/Tannenbaum 2016):
    Curvature is negatively correlated with network fragility.
    Curvature DROP = system becoming more fragile = regime change incoming.

Usage:
    >>> from prism.manifold_geometry import run_manifold_geometry, analyze_geometry
    >>>
    >>> # Full analysis
    >>> results = run_manifold_geometry(signals, signal_ids)
    >>> print(results['topology'])
    >>> print(results['curvature_ollivier'])
    >>>
    >>> # Quick analysis
    >>> result = analyze_geometry(signals)
    >>> print(result.typology.summary)

Architecture:
    manifold_geometry/
        __init__.py         # This file
        orchestrator.py     # Routes + formats
        models.py           # GeometryVector, GeometryTypology
        engine_mapping.py   # Engine selection
        ollivier_ricci.py   # Gold standard curvature (expensive)
        forman_ricci.py     # Fast approximation curvature
        geodesic_curvature.py # Manifold embedding curvature
        geometry_state.py   # 6-metric state computation
"""

__version__ = "2.0.0"
__author__ = "ORTHON Project"

# Models (dataclasses and enums)
from .models import (
    # Enums
    TopologyClass,
    GeometryStabilityClass,
    StabilityClass,  # Backwards compat alias
    LeadershipClass,
    # Dataclasses
    GeometryVector,
    GeometryTypology,
    ManifoldGeometryOutput,
    StructuralGeometryOutput,  # Backwards compat alias
)

# Orchestrator
from .orchestrator import (
    run_manifold_geometry,
    run_structural_geometry,  # Backwards compat alias
    analyze_geometry,
    ManifoldGeometryFramework,
    StructuralGeometryFramework,  # Backwards compat alias
    get_structure_fingerprint,
    structure_distance,
    detect_structure_change,
)

# Engine mapping
from .engine_mapping import (
    select_engines,
    get_topology_classification,
    get_stability_classification,
    get_leadership_classification,
    ENGINE_MAP,
    TOPOLOGY_THRESHOLDS,
)

# Curvature engines
try:
    from .ollivier_ricci import (
        compute as compute_ollivier_ricci,
        compute_temporal as compute_ollivier_temporal,
        detect_curvature_anomaly,
        OllivierRicciResult,
    )
    from .forman_ricci import (
        compute as compute_forman_ricci,
        compute_temporal as compute_forman_temporal,
        identify_critical_edges,
        compute_ricci_flow,
        FormanRicciResult,
    )
    CURVATURE_AVAILABLE = True
except ImportError:
    CURVATURE_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    "CURVATURE_AVAILABLE",

    # Enums
    "TopologyClass",
    "GeometryStabilityClass",
    "StabilityClass",
    "LeadershipClass",

    # Dataclasses
    "GeometryVector",
    "GeometryTypology",
    "ManifoldGeometryOutput",
    "StructuralGeometryOutput",

    # Orchestrator
    "run_manifold_geometry",
    "run_structural_geometry",
    "analyze_geometry",
    "ManifoldGeometryFramework",
    "StructuralGeometryFramework",
    "get_structure_fingerprint",
    "structure_distance",
    "detect_structure_change",

    # Engine mapping
    "select_engines",
    "get_topology_classification",
    "get_stability_classification",
    "get_leadership_classification",
    "ENGINE_MAP",
    "TOPOLOGY_THRESHOLDS",

    # Curvature (if available)
    "compute_ollivier_ricci",
    "compute_forman_ricci",
    "compute_ollivier_temporal",
    "compute_forman_temporal",
    "detect_curvature_anomaly",
    "identify_critical_edges",
    "compute_ricci_flow",
    "OllivierRicciResult",
    "FormanRicciResult",
]
