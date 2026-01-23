"""
PRISM Analytical Layers
=======================

Pure orchestrators that call engines and produce meaning.
Contains ZERO computation - all computation lives in engines/.

Layers:
    1. signal_typology: What type of signal is this?
    2. system_physics: What are the physical properties?
    3. behavioral_geometry: How do signals relate?
    4. phase_state: What regime?
    5. dynamical_systems: Stable or bifurcating?
    6. derivatives: Where is it heading?
    7. thesis_summary: Unified narrative

Rule: If you see `np.` or `scipy.` in a layer file, STOP.
      That computation belongs in an engine.
"""

from .signal_typology import SignalTypologyLayer
from .system_physics import (
    SystemPhysicsLayer,
    SystemPhysicsOutput,
    PhysicsVector,
    PhysicsTypology,
    analyze_physics,
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass
)
from .behavioral_geometry import (
    BehavioralGeometryLayer,
    BehavioralGeometryOutput,
    GeometryVector,
    GeometryTypology,
    analyze_geometry,
    TopologyClass,
    StabilityClass,
    LeadershipClass
)

__all__ = [
    # Signal Typology
    'SignalTypologyLayer',
    # System Physics
    'SystemPhysicsLayer',
    'SystemPhysicsOutput',
    'PhysicsVector',
    'PhysicsTypology',
    'analyze_physics',
    'EnergyClass',
    'EquilibriumClass',
    'FlowClass',
    'OrbitClass',
    'DominanceClass',
    # Behavioral Geometry
    'BehavioralGeometryLayer',
    'BehavioralGeometryOutput',
    'GeometryVector',
    'GeometryTypology',
    'analyze_geometry',
    'TopologyClass',
    'StabilityClass',
    'LeadershipClass',
]
