"""
Structural Geometry
===================

One of the four ORTHON analytical frameworks:

    Signal Typology     → What IS this signal?
    Structural Geometry → What is its STRUCTURE? (this framework)
    Dynamical Systems   → How does the SYSTEM evolve?
    Causal Mechanics    → What DRIVES the system?

Structural Geometry analyzes relational structure between signals:
    - Correlation: Who moves with whom?
    - Clustering: What groups exist?
    - Network: What is the topology?
    - Causality: Who leads whom?
    - Decoupling: Are relationships breaking?

Usage:
    >>> from prism.structural_geometry import run_structural_geometry, analyze_geometry
    >>>
    >>> # Full analysis
    >>> results = run_structural_geometry(signals, signal_ids)
    >>> print(results['topology'])
    >>> print(results['hub_names'])
    >>>
    >>> # Quick analysis
    >>> result = analyze_geometry(signals)
    >>> print(result.typology.summary)

Architecture:
    structural_geometry/
        __init__.py         # This file
        orchestrator.py     # Routes + formats
        models.py           # GeometryVector, GeometryTypology
        engine_mapping.py   # Engine selection
"""

__version__ = "1.0.0"
__author__ = "Ørthon Project"

# Models (dataclasses and enums)
from .models import (
    # Enums
    TopologyClass,
    StabilityClass,
    LeadershipClass,
    # Dataclasses
    GeometryVector,
    GeometryTypology,
    StructuralGeometryOutput,
)

# Orchestrator
from .orchestrator import (
    run_structural_geometry,
    analyze_geometry,
    StructuralGeometryFramework,
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

__all__ = [
    # Version
    "__version__",

    # Enums
    "TopologyClass",
    "StabilityClass",
    "LeadershipClass",

    # Dataclasses
    "GeometryVector",
    "GeometryTypology",
    "StructuralGeometryOutput",

    # Orchestrator
    "run_structural_geometry",
    "analyze_geometry",
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
]
