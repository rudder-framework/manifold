"""
Structural Geometry Orchestrator
================================

One of four ORTHON analytical frameworks: What is the STRUCTURE?

Analyzes relational structure between signals:
    - Correlation: Who moves with whom?
    - Clustering: What groups exist?
    - Network: What is the topology?
    - Causality: Who leads whom?
    - Decoupling: Are relationships breaking?

Architecture:
    orchestrator.py (this file - routes + formats)
        │
        ▼
    engines/* (computations)
        │
        ▼
    engine_mapping.py (selects engines)

Usage:
    from prism.structural_geometry import run_structural_geometry

    results = run_structural_geometry(signals, signal_ids)
    print(results['topology'])
    print(results['engine_recommendations'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .models import (
    TopologyClass,
    StabilityClass,
    LeadershipClass,
    GeometryVector,
    GeometryTypology,
    StructuralGeometryOutput,
)
from .engine_mapping import (
    select_engines,
    get_topology_classification,
    get_stability_classification,
    get_leadership_classification,
)


def run_structural_geometry(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    entity_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Structural Geometry analysis.

    Args:
        signals: 2D array (n_signals, n_observations)
        signal_ids: Optional signal identifiers
        entity_id: Entity identifier
        config: Optional configuration overrides

    Returns:
        {
            'topology': topology classification,
            'stability': stability classification,
            'leadership': leadership classification,
            'vector': GeometryVector as dict,
            'typology': GeometryTypology as dict,
            'engine_recommendations': [engines],
            'hub_names': [...],
            'leader_names': [...],
            'alerts': [...],
            'metadata': {...}
        }
    """
    config = config or {}

    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_signals = signals.shape[0]

    if signal_ids is None:
        signal_ids = [f"signal_{i}" for i in range(n_signals)]

    # Use the framework class for core analysis
    framework = StructuralGeometryFramework(config=config)
    output = framework.analyze(signals, signal_ids, entity_id)

    # Get classifications as strings
    topology = output.typology.topology_class.value.upper()
    stability = output.typology.stability_class.value.upper()
    leadership = output.typology.leadership_class.value.upper()

    # Build state for engine selection
    state = {
        'topology': topology,
        'stability': stability,
        'leadership': leadership,
        'n_clusters': output.vector.n_clusters,
        'network_density': output.vector.network_density,
    }

    # Select engines
    engines = select_engines(state)

    return {
        'topology': topology,
        'stability': stability,
        'leadership': leadership,
        'vector': output.vector.to_dict(),
        'typology': output.typology.to_dict(),
        'engine_recommendations': engines,
        'hub_names': output.typology.hub_names,
        'leader_names': output.typology.leader_names,
        'alerts': output.typology.alerts,
        'summary': output.typology.summary,
        'confidence': output.typology.confidence,
        'metadata': {
            'entity_id': entity_id,
            'n_signals': n_signals,
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


class StructuralGeometryFramework:
    """
    Structural Geometry Framework: What is the STRUCTURE?

    Pure orchestrator. Calls geometry engines, classifies results.
    Contains ZERO computation - only coordination and classification logic.

    Answers:
        - Correlation: Who moves with whom?
        - Clustering: What groups exist?
        - Network: What is the topology?
        - Causality: Who leads whom?
        - Decoupling: Are relationships breaking?
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        self.historical_window = self.config.get('historical_window', 100)
        self.recent_window = self.config.get('recent_window', 20)

    def analyze(
        self,
        signals: np.ndarray,
        signal_ids: Optional[List[str]] = None,
        entity_id: str = "",
        previous: Optional[StructuralGeometryOutput] = None
    ) -> StructuralGeometryOutput:
        """
        Analyze relationships between multiple signals.

        Args:
            signals: 2D array (n_signals, n_observations)
            signal_ids: Optional signal identifiers
            entity_id: Entity identifier
            previous: Previous output (for change detection)

        Returns:
            StructuralGeometryOutput with vector and typology
        """
        # Import engines (deferred to avoid circular imports)
        from prism.engines.geometry import (
            bg_correlation as correlation,
            bg_distance as distance,
            bg_clustering as clustering,
            bg_network as network,
            bg_granger as granger,
            bg_lead_lag as lead_lag,
            bg_decoupling as decoupling
        )

        signals = np.asarray(signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_signals = signals.shape[0]

        if signal_ids is None:
            signal_ids = [f"signal_{i}" for i in range(n_signals)]

        # === CALL ALL ENGINES ===
        corr_result = correlation.compute_matrix(signals)
        dist_result = distance.compute_matrix(signals)

        cluster_result = clustering.compute_from_signals(signals)
        cluster_profiles = clustering.get_cluster_profiles(
            signals, cluster_result.labels, signal_ids
        )

        network_result = network.compute(
            corr_result.pearson_matrix,
            threshold=self.correlation_threshold
        )
        node_profiles = network.get_node_profiles(network_result, signal_ids)

        granger_result = granger.compute_matrix(signals)
        lead_lag_result = lead_lag.compute_matrix(signals)

        decoupling_result = decoupling.compute_matrix(
            signals, signal_ids,
            self.historical_window, self.recent_window
        )

        # === BUILD VECTOR ===
        vector = self._build_vector(
            entity_id, n_signals, corr_result, cluster_result, cluster_profiles,
            network_result, granger_result, lead_lag_result, decoupling_result
        )

        # === CLASSIFY ===
        typology = self._classify(
            entity_id, signal_ids, corr_result, cluster_result, network_result,
            node_profiles, granger_result, lead_lag_result, decoupling_result
        )

        return StructuralGeometryOutput(
            vector=vector,
            typology=typology,
            correlation_matrix=corr_result.pearson_matrix,
            distance_matrix=dist_result.correlation_distance_matrix,
            cluster_labels=cluster_result.labels,
            adjacency_matrix=network_result.adjacency_matrix
        )

    def _build_vector(
        self, entity_id, n_signals, corr, cluster, cluster_profiles,
        network, granger, lead_lag, decoupling
    ) -> GeometryVector:
        """Assemble GeometryVector from engine outputs."""

        # Mean internal correlation across clusters
        if cluster_profiles:
            mean_internal = np.mean([p.mean_internal_correlation for p in cluster_profiles])
        else:
            mean_internal = 0.0

        # Mean lead/lag
        mean_lag = np.mean(np.abs(lead_lag.lag_matrix[lead_lag.lag_matrix != 0])) \
            if np.any(lead_lag.lag_matrix != 0) else 0.0

        # Decoupling rate
        n_possible_pairs = n_signals * (n_signals - 1) / 2
        decoupling_rate = decoupling.n_decoupled_pairs / n_possible_pairs \
            if n_possible_pairs > 0 else 0.0

        return GeometryVector(
            timestamp=datetime.now(),
            entity_id=entity_id,
            n_signals=n_signals,

            # Correlation
            mean_correlation=corr.mean_correlation,
            median_correlation=corr.median_correlation,
            correlation_dispersion=corr.correlation_dispersion,
            n_strong_pairs=corr.n_strong_pairs,
            n_weak_pairs=corr.n_weak_pairs,
            variance_explained_1=corr.variance_explained_1,
            effective_dimension=corr.effective_dimension,

            # Clustering
            n_clusters=cluster.n_clusters,
            silhouette_score=cluster.silhouette_score,
            mean_internal_correlation=float(mean_internal),

            # Network
            network_density=network.density,
            mean_degree=network.mean_degree,
            transitivity=network.transitivity,
            n_components=network.n_components,
            n_hubs=len(network.hub_indices),

            # Causality
            n_causal_pairs=granger.n_causal_pairs,
            n_bidirectional=granger.n_bidirectional,
            mean_lead_lag=float(mean_lag),

            # Decoupling
            n_decoupled_pairs=decoupling.n_decoupled_pairs,
            n_severe_decouplings=decoupling.n_severe,
            decoupling_rate=float(decoupling_rate)
        )

    def _classify(
        self, entity_id, signal_ids, corr, cluster, network,
        node_profiles, granger, lead_lag, decoupling
    ) -> GeometryTypology:
        """Convert measurements to classification."""

        # Topology class
        topology_str = get_topology_classification(
            network.density,
            cluster.n_clusters,
            cluster.silhouette_score,
            len(network.hub_indices),
            network.n_components
        )
        topology_class = TopologyClass(topology_str.lower())

        # Stability class
        stability_str = get_stability_classification(
            decoupling.n_decoupled_pairs,
            decoupling.n_severe,
            len(signal_ids)
        )
        stability_class = StabilityClass(stability_str.lower())

        # Leadership class
        leadership_str = get_leadership_classification(
            granger.n_causal_pairs,
            granger.n_bidirectional,
            granger.top_causers,
            corr.mean_correlation
        )
        leadership_class = LeadershipClass(leadership_str.lower())

        # Hub names
        hub_names = [signal_ids[i] if i < len(signal_ids) else f"signal_{i}"
                     for i in network.hub_indices]

        # Leader names
        leader_names = [signal_ids[i] if i < len(signal_ids) else f"signal_{i}"
                        for i in granger.top_causers]

        # Generate summary and alerts
        summary, alerts = self._generate_summary(
            topology_class, stability_class, leadership_class,
            network, decoupling, hub_names, leader_names
        )

        # Add decoupling alerts
        alerts.extend(decoupling.alerts)

        # Confidence
        confidence = self._compute_confidence(corr, cluster, network)

        return GeometryTypology(
            entity_id=entity_id,
            topology_class=topology_class,
            stability_class=stability_class,
            leadership_class=leadership_class,
            cluster_assignments=cluster.labels,
            hub_indices=network.hub_indices,
            hub_names=hub_names,
            leader_indices=granger.top_causers,
            leader_names=leader_names,
            decoupling_alerts=decoupling.alerts,
            summary=summary,
            alerts=alerts,
            confidence=confidence
        )

    def _generate_summary(
        self, topology, stability, leadership,
        network, decoupling, hub_names, leader_names
    ) -> tuple:
        """Generate human-readable summary and alerts."""

        alerts = []

        # Stability alerts
        if stability == StabilityClass.BREAKING:
            alerts.append("ACTIVE DECOUPLING: Relationships breaking down")
        elif stability == StabilityClass.WEAKENING:
            alerts.append("Correlation structure weakening")

        # Topology alerts
        if topology == TopologyClass.FRAGMENTED:
            alerts.append("Network fragmented: disconnected components")

        # Leadership
        if leader_names:
            leaders_str = ", ".join(leader_names[:3])
            alerts.append(f"Leaders: {leaders_str}")

        # Hubs
        if hub_names:
            hubs_str = ", ".join(hub_names[:3])
            alerts.append(f"Hubs: {hubs_str}")

        summary = f"**{topology.value.replace('_', ' ').title()}** | {stability.value} | {leadership.value.replace('_', ' ')}"

        return summary, alerts

    def _compute_confidence(self, corr, cluster, network) -> float:
        """Compute overall classification confidence."""

        # Higher confidence when structure is clear
        corr_conf = min(corr.variance_explained_1 * 2, 1.0)
        cluster_conf = cluster.silhouette_score if cluster.silhouette_score > 0 else 0.3
        network_conf = network.transitivity if network.density > 0.1 else 0.3

        confidence = (corr_conf + cluster_conf + network_conf) / 3

        return float(np.clip(confidence, 0, 1))


def analyze_geometry(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    entity_id: str = "",
) -> StructuralGeometryOutput:
    """
    Convenience function for quick structural geometry analysis.

    Example:
        result = analyze_geometry(my_signals)
        print(result.typology.summary)
        print(result.typology.hub_names)
    """
    framework = StructuralGeometryFramework()
    return framework.analyze(signals, signal_ids, entity_id)


def get_structure_fingerprint(state: Dict[str, Any]) -> np.ndarray:
    """
    Convert structural state to a fingerprint vector.

    Args:
        state: Dict with topology, stability, leadership

    Returns:
        numpy array encoding the state
    """
    topology_map = {
        'HIGHLY_CONNECTED': 1.0, 'MODULAR': 0.75, 'HIERARCHICAL': 0.5,
        'FRAGMENTED': 0.25, 'SPARSE': 0.0
    }
    stability_map = {
        'STABLE': 1.0, 'RESTRUCTURING': 0.75, 'WEAKENING': 0.5,
        'BREAKING': 0.25, 'BROKEN': 0.0
    }
    leadership_map = {
        'CLEAR_LEADER': 1.0, 'MULTIPLE_LEADERS': 0.75, 'BIDIRECTIONAL': 0.5,
        'CONTEMPORANEOUS': 0.25, 'FRAGMENTED': 0.0
    }

    return np.array([
        topology_map.get(state.get('topology', '').upper(), 0.5),
        stability_map.get(state.get('stability', '').upper(), 0.5),
        leadership_map.get(state.get('leadership', '').upper(), 0.5),
    ])


def structure_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute distance between two structure fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, ~1.7 = maximally different)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_structure_change(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect structural change between two states.

    Args:
        previous_state: Previous structural state
        current_state: Current structural state
        threshold: Change threshold for flagging

    Returns:
        Dict with change detection results
    """
    fp_prev = get_structure_fingerprint(previous_state)
    fp_curr = get_structure_fingerprint(current_state)

    distance = structure_distance(fp_prev, fp_curr)

    changes = {}
    for dim in ['topology', 'stability', 'leadership']:
        changes[dim] = {
            'previous': previous_state.get(dim, ''),
            'current': current_state.get(dim, ''),
            'changed': previous_state.get(dim, '') != current_state.get(dim, ''),
        }

    changed_dims = [d for d, c in changes.items() if c['changed']]

    if not changed_dims:
        change_type = 'NONE'
    elif 'topology' in changed_dims:
        change_type = 'TOPOLOGY_CHANGE'
    elif 'stability' in changed_dims:
        change_type = 'STABILITY_CHANGE'
    else:
        change_type = 'LEADERSHIP_CHANGE'

    return {
        'change_detected': distance >= threshold,
        'change_type': change_type,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'changed_dimensions': changed_dims,
    }


# Backwards compatibility
BehavioralGeometryFramework = StructuralGeometryFramework
BehavioralGeometryLayer = StructuralGeometryFramework

# New naming (manifold_geometry)
ManifoldGeometryFramework = StructuralGeometryFramework
run_manifold_geometry = run_structural_geometry
