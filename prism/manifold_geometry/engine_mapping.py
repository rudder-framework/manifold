"""
Structural Geometry Engine Mapping
==================================

Maps structural state to recommended analysis engines.

Categories:
    - Attractor Shape: phase_space, embedding, correlation_dimension
    - Density Structure: density, mode_detection
    - Boundary Behavior: range_analysis, extreme_clustering
    - Trajectory Patterns: path_complexity, directional_bias

Usage:
    from prism.structural_geometry.engine_mapping import select_engines

    state = {
        'topology': 'MODULAR',
        'stability': 'STABLE',
        'n_clusters': 3,
        'network_density': 0.4,
    }

    engines = select_engines(state)
"""

from typing import Dict, List


# =============================================================================
# THRESHOLDS
# =============================================================================

TOPOLOGY_THRESHOLDS = {
    'highly_connected_density': 0.7,
    'modular_silhouette': 0.3,
    'hierarchical_density': 0.5,
}

STABILITY_THRESHOLDS = {
    'breaking_severe': 1,
    'weakening_pairs': 0.25,  # Fraction of pairs
}


# =============================================================================
# ENGINE RECOMMENDATIONS
# =============================================================================

ENGINE_MAP = {
    # Topology-based engines
    'HIGHLY_CONNECTED': [
        'correlation_analysis',
        'factor_model',
        'pca_decomposition',
        'common_driver_detection',
    ],
    'MODULAR': [
        'cluster_analysis',
        'community_detection',
        'block_model',
        'inter_cluster_dynamics',
    ],
    'HIERARCHICAL': [
        'hub_analysis',
        'influence_propagation',
        'cascade_detection',
        'leader_follower',
    ],
    'FRAGMENTED': [
        'component_analysis',
        'bridge_detection',
        'reconnection_potential',
    ],
    'SPARSE': [
        'pairwise_analysis',
        'weak_tie_detection',
        'emerging_structure',
    ],

    # Stability-based engines
    'STABLE': [
        'steady_state_analysis',
        'persistence_metrics',
    ],
    'WEAKENING': [
        'trend_analysis',
        'correlation_decay',
        'early_warning',
    ],
    'RESTRUCTURING': [
        'cluster_evolution',
        'membership_change',
        'regime_detection',
    ],
    'BREAKING': [
        'decoupling_analysis',
        'shock_propagation',
        'contagion_risk',
    ],
    'BROKEN': [
        'recovery_potential',
        'new_equilibrium',
    ],

    # Leadership-based engines
    'CLEAR_LEADER': [
        'leader_tracking',
        'influence_measurement',
        'follower_analysis',
    ],
    'MULTIPLE_LEADERS': [
        'multi_hub_analysis',
        'competition_detection',
        'coordination_metrics',
    ],
    'BIDIRECTIONAL': [
        'feedback_analysis',
        'mutual_causation',
        'cycle_detection',
    ],
    'CONTEMPORANEOUS': [
        'common_factor',
        'synchronization_analysis',
    ],

    # Attractor shape engines
    'ATTRACTOR_SHAPE': [
        'phase_space',
        'embedding',
        'correlation_dimension',
    ],

    # Density structure engines
    'DENSITY_STRUCTURE': [
        'density',
        'mode_detection',
    ],

    # Boundary behavior engines
    'BOUNDARY_BEHAVIOR': [
        'range_analysis',
        'extreme_clustering',
    ],

    # Trajectory pattern engines
    'TRAJECTORY_PATTERNS': [
        'path_complexity',
        'directional_bias',
    ],
}

# Compound engines for specific combinations
COMPOUND_ENGINES = {
    ('MODULAR', 'BREAKING'): [
        'cluster_fragmentation',
        'modular_collapse',
    ],
    ('HIERARCHICAL', 'CLEAR_LEADER'): [
        'hub_vulnerability',
        'succession_analysis',
    ],
    ('FRAGMENTED', 'BROKEN'): [
        'systemic_failure',
        'recovery_path',
    ],
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def select_engines(state: Dict) -> List[str]:
    """
    Select recommended engines based on structural state.

    Args:
        state: Dict with topology, stability, leadership keys

    Returns:
        Prioritized list of recommended engines
    """
    engines = []

    topology = state.get('topology', 'SPARSE').upper()
    stability = state.get('stability', 'STABLE').upper()
    leadership = state.get('leadership', 'CONTEMPORANEOUS').upper()

    # Add topology engines
    if topology in ENGINE_MAP:
        engines.extend(ENGINE_MAP[topology])

    # Add stability engines
    if stability in ENGINE_MAP:
        engines.extend(ENGINE_MAP[stability])

    # Add leadership engines
    if leadership in ENGINE_MAP:
        engines.extend(ENGINE_MAP[leadership])

    # Always add core structural engines
    engines.extend(ENGINE_MAP.get('ATTRACTOR_SHAPE', []))
    engines.extend(ENGINE_MAP.get('DENSITY_STRUCTURE', []))

    # Add compound engines for special combinations
    for combo, combo_engines in COMPOUND_ENGINES.items():
        if all(c in [topology, stability, leadership] for c in combo):
            engines.extend(combo_engines)

    # Deduplicate while preserving order
    seen = set()
    prioritized = []
    for e in engines:
        if e not in seen:
            seen.add(e)
            prioritized.append(e)

    return prioritized


def get_topology_classification(
    density: float,
    n_clusters: int,
    silhouette: float,
    n_hubs: int,
    n_components: int,
) -> str:
    """
    Classify network topology from metrics.

    Args:
        density: Network density [0, 1]
        n_clusters: Number of clusters
        silhouette: Cluster quality score
        n_hubs: Number of hub nodes
        n_components: Number of disconnected components

    Returns:
        Topology classification string
    """
    if density > TOPOLOGY_THRESHOLDS['highly_connected_density']:
        return 'HIGHLY_CONNECTED'
    elif n_clusters > 2 and silhouette > TOPOLOGY_THRESHOLDS['modular_silhouette']:
        return 'MODULAR'
    elif n_hubs > 0 and density < TOPOLOGY_THRESHOLDS['hierarchical_density']:
        return 'HIERARCHICAL'
    elif n_components > 1:
        return 'FRAGMENTED'
    else:
        return 'SPARSE'


def get_stability_classification(
    n_decoupled_pairs: int,
    n_severe: int,
    n_signals: int,
) -> str:
    """
    Classify relationship stability.

    Args:
        n_decoupled_pairs: Number of decoupled pairs
        n_severe: Number of severe decouplings
        n_signals: Total number of signals

    Returns:
        Stability classification string
    """
    n_possible_pairs = n_signals * (n_signals - 1) / 2
    decoupling_rate = n_decoupled_pairs / n_possible_pairs if n_possible_pairs > 0 else 0

    if n_severe > STABILITY_THRESHOLDS['breaking_severe']:
        return 'BREAKING'
    elif decoupling_rate > STABILITY_THRESHOLDS['weakening_pairs']:
        return 'WEAKENING'
    elif n_decoupled_pairs > 0:
        return 'RESTRUCTURING'
    else:
        return 'STABLE'


def get_leadership_classification(
    n_causal_pairs: int,
    n_bidirectional: int,
    top_causers: List[int],
    mean_correlation: float,
) -> str:
    """
    Classify causal/leadership structure.

    Args:
        n_causal_pairs: Number of significant Granger pairs
        n_bidirectional: Number of bidirectional relationships
        top_causers: Indices of top causal signals
        mean_correlation: Average correlation

    Returns:
        Leadership classification string
    """
    if n_causal_pairs > 0:
        if n_bidirectional > n_causal_pairs // 2:
            return 'BIDIRECTIONAL'
        elif len(set(top_causers)) == 1:
            return 'CLEAR_LEADER'
        else:
            return 'MULTIPLE_LEADERS'
    elif mean_correlation > 0.5:
        return 'CONTEMPORANEOUS'
    else:
        return 'FRAGMENTED'
