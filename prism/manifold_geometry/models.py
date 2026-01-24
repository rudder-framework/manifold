"""
Structural Geometry Models
==========================

Dataclasses and enums for Structural Geometry framework output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class TopologyClass(Enum):
    """Network topology classification"""
    HIGHLY_CONNECTED = "highly_connected"   # Dense, most nodes connected
    MODULAR = "modular"                     # Clear clusters
    HIERARCHICAL = "hierarchical"           # Hub-and-spoke
    FRAGMENTED = "fragmented"               # Disconnected components
    SPARSE = "sparse"                       # Few connections


class GeometryStabilityClass(Enum):
    """Relationship stability classification (Structural Geometry layer)"""
    STABLE = "stable"                       # Relationships holding
    WEAKENING = "weakening"                 # Correlations declining
    RESTRUCTURING = "restructuring"         # Clusters changing
    BREAKING = "breaking"                   # Active decoupling
    BROKEN = "broken"                       # Major relationships lost


# Backwards compatibility alias
StabilityClass = GeometryStabilityClass


class LeadershipClass(Enum):
    """Causal structure classification"""
    CLEAR_LEADER = "clear_leader"           # One dominant signal
    MULTIPLE_LEADERS = "multiple_leaders"   # Several leading signals
    BIDIRECTIONAL = "bidirectional"         # Mutual causality
    CONTEMPORANEOUS = "contemporaneous"     # Move together
    FRAGMENTED = "fragmented"               # No clear structure


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class GeometryVector:
    """
    Numerical measurements from structural geometry analysis.
    This is the DATA output - consumed by downstream frameworks.
    """

    # === IDENTIFICATION ===
    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: str = ""
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)
    n_signals: int = 0

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === CORRELATION STRUCTURE ===
    mean_correlation: float = 0.0
    median_correlation: float = 0.0
    correlation_dispersion: float = 0.0
    n_strong_pairs: int = 0
    n_weak_pairs: int = 0

    # Eigenvalue analysis
    variance_explained_1: float = 0.0       # First PC
    effective_dimension: float = 1.0        # How many independent signals

    # === CLUSTERING ===
    n_clusters: int = 1
    silhouette_score: float = 0.0
    mean_internal_correlation: float = 0.0

    # === NETWORK TOPOLOGY ===
    network_density: float = 0.0
    mean_degree: float = 0.0
    transitivity: float = 0.0               # Clustering coefficient
    n_components: int = 1
    n_hubs: int = 0

    # === CAUSALITY ===
    n_causal_pairs: int = 0
    n_bidirectional: int = 0
    mean_lead_lag: float = 0.0              # Average optimal lag

    # === DECOUPLING ===
    n_decoupled_pairs: int = 0
    n_severe_decouplings: int = 0
    decoupling_rate: float = 0.0            # Fraction of pairs decoupled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        # Ensure unit_id is set
        if not result.get('unit_id') and result.get('entity_id'):
            result['unit_id'] = result['entity_id']
        return result


@dataclass
class GeometryTypology:
    """
    Classification from structural geometry analysis.
    This is the INFORMATION output - human-interpretable.
    """

    # === IDENTIFICATION ===
    entity_id: str = ""
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === CLASSIFICATIONS ===
    topology_class: TopologyClass = TopologyClass.SPARSE
    stability_class: GeometryStabilityClass = GeometryStabilityClass.STABLE
    leadership_class: LeadershipClass = LeadershipClass.CONTEMPORANEOUS

    # === CLUSTER INFO ===
    cluster_assignments: np.ndarray = None
    cluster_names: List[str] = None

    # === HUB IDENTIFICATION ===
    hub_indices: List[int] = field(default_factory=list)
    hub_names: List[str] = field(default_factory=list)

    # === LEADER IDENTIFICATION ===
    leader_indices: List[int] = field(default_factory=list)
    leader_names: List[str] = field(default_factory=list)

    # === DECOUPLING ALERTS ===
    decoupling_alerts: List[str] = field(default_factory=list)

    # === HUMAN-READABLE ===
    summary: str = ""
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'topology_class': self.topology_class.value,
            'stability_class': self.stability_class.value,
            'leadership_class': self.leadership_class.value,
            'hub_indices': self.hub_indices,
            'hub_names': self.hub_names,
            'leader_indices': self.leader_indices,
            'leader_names': self.leader_names,
            'decoupling_alerts': self.decoupling_alerts,
            'summary': self.summary,
            'alerts': self.alerts,
            'confidence': self.confidence
        }


@dataclass
class StructuralGeometryOutput:
    """Complete output from Structural Geometry framework"""
    vector: GeometryVector
    typology: GeometryTypology

    # Detailed results (optional, for deep inspection)
    correlation_matrix: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    adjacency_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Combined dictionary for full output."""
        result = self.vector.to_dict()
        result.update(self.typology.to_dict())
        return result


# Backwards compatibility aliases
BehavioralGeometryOutput = StructuralGeometryOutput
ManifoldGeometryOutput = StructuralGeometryOutput
