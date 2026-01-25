"""
Network Topology Analysis
=========================

Treats signals as nodes and correlations as edges to extract
network-theoretic properties.

Key metrics:
    - Centrality: Which signals are most connected/influential
    - Density: How interconnected is the network
    - Modularity: How clustered is the structure
    - Hubs: Central nodes that connect clusters

Network analysis reveals market structure and systemically important signals.
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class NetworkResult:
    """Output from network analysis"""

    # Adjacency matrix
    adjacency_matrix: np.ndarray    # Thresholded correlation

    # Global metrics
    density: float                  # Edge count / possible edges
    mean_degree: float              # Average connections per node
    transitivity: float             # Global clustering coefficient

    # Centrality measures (per node)
    degree_centrality: np.ndarray   # Number of connections
    eigenvector_centrality: np.ndarray  # Importance of connections
    betweenness_centrality: np.ndarray  # Bridge importance

    # Structure
    n_components: int               # Connected components
    largest_component_size: int     # Size of biggest component

    # Hubs
    hub_indices: List[int]          # Most central nodes
    hub_threshold: float            # Centrality threshold used


@dataclass
class NodeProfile:
    """Profile for a single node (signal)"""

    node_idx: int
    signal_id: str

    degree: int
    degree_centrality: float
    eigenvector_centrality: float
    betweenness_centrality: float

    neighbors: List[int]

    role: str  # 'hub' | 'bridge' | 'peripheral' | 'isolated'


def compute(
    correlation_matrix: np.ndarray,
    threshold: float = 0.5,
    signal_ids: Optional[List[str]] = None
) -> NetworkResult:
    """
    Build network from correlation matrix and compute metrics.

    Args:
        correlation_matrix: Pairwise correlations (n_signals, n_signals)
        threshold: Edge threshold (|r| >= threshold creates edge)
        signal_ids: Optional signal identifiers

    Returns:
        NetworkResult
    """
    n = correlation_matrix.shape[0]

    if n < 2:
        return NetworkResult(
            adjacency_matrix=np.zeros((1, 1)),
            density=0.0, mean_degree=0.0, transitivity=0.0,
            degree_centrality=np.array([0.0]),
            eigenvector_centrality=np.array([0.0]),
            betweenness_centrality=np.array([0.0]),
            n_components=1, largest_component_size=1,
            hub_indices=[], hub_threshold=0.0
        )

    # Build adjacency matrix
    adj = (np.abs(correlation_matrix) >= threshold).astype(float)
    np.fill_diagonal(adj, 0)  # No self-loops

    # Density
    n_edges = np.sum(adj) / 2
    n_possible = n * (n - 1) / 2
    density = n_edges / n_possible if n_possible > 0 else 0.0

    # Degree
    degrees = np.sum(adj, axis=1)
    mean_degree = float(np.mean(degrees))

    # Degree centrality
    degree_centrality = degrees / (n - 1) if n > 1 else degrees

    # Eigenvector centrality (dominant eigenvector of adjacency)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        idx = np.argmax(eigenvalues)
        eig_cent = np.abs(eigenvectors[:, idx])
        eig_cent = eig_cent / np.max(eig_cent) if np.max(eig_cent) > 0 else eig_cent
    except:
        eig_cent = degree_centrality.copy()

    # Betweenness centrality (simplified)
    between_cent = _compute_betweenness(adj)

    # Transitivity (global clustering coefficient)
    transitivity = _compute_transitivity(adj)

    # Connected components
    n_components, component_sizes = _find_components(adj)
    largest_component = max(component_sizes) if component_sizes else 0

    # Identify hubs (top 10% by eigenvector centrality)
    hub_thresh = np.percentile(eig_cent, 90)
    hub_indices = list(np.where(eig_cent >= hub_thresh)[0])

    return NetworkResult(
        adjacency_matrix=adj,
        density=float(density),
        mean_degree=mean_degree,
        transitivity=transitivity,
        degree_centrality=degree_centrality,
        eigenvector_centrality=eig_cent,
        betweenness_centrality=between_cent,
        n_components=n_components,
        largest_component_size=largest_component,
        hub_indices=hub_indices,
        hub_threshold=float(hub_thresh)
    )


def _compute_transitivity(adj: np.ndarray) -> float:
    """Compute global clustering coefficient."""
    n = adj.shape[0]

    triangles = 0
    triples = 0

    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue

        # Count triangles through node i
        for j_idx, j in enumerate(neighbors):
            for k_node in neighbors[j_idx+1:]:
                triples += 1
                if adj[j, k_node] > 0:
                    triangles += 1

    return triangles / triples if triples > 0 else 0.0


def _compute_betweenness(adj: np.ndarray) -> np.ndarray:
    """Simplified betweenness centrality using BFS."""
    n = adj.shape[0]
    betweenness = np.zeros(n)

    for s in range(n):
        # BFS from s
        dist = np.full(n, -1)
        dist[s] = 0
        paths = np.zeros(n)
        paths[s] = 1

        queue = [s]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)

            for w in np.where(adj[v] > 0)[0]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    paths[w] += paths[v]

        # Accumulate
        delta = np.zeros(n)
        for w in reversed(order[1:]):
            for v in np.where(adj[w] > 0)[0]:
                if dist[v] == dist[w] - 1:
                    delta[v] += (paths[v] / paths[w]) * (1 + delta[w]) if paths[w] > 0 else 0
            if w != s:
                betweenness[w] += delta[w]

    # Normalize
    if n > 2:
        betweenness = betweenness / ((n - 1) * (n - 2))

    return betweenness


def _find_components(adj: np.ndarray) -> tuple:
    """Find connected components."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS
        component = []
        queue = [start]
        visited[start] = True

        while queue:
            v = queue.pop(0)
            component.append(v)

            for w in np.where(adj[v] > 0)[0]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)

        components.append(len(component))

    return len(components), components


def get_node_profiles(
    network: NetworkResult,
    signal_ids: Optional[List[str]] = None
) -> List[NodeProfile]:
    """
    Generate profiles for each node.

    Args:
        network: NetworkResult from compute()
        signal_ids: Optional signal identifiers

    Returns:
        List of NodeProfile objects
    """
    n = network.adjacency_matrix.shape[0]
    profiles = []

    for i in range(n):
        sig_id = signal_ids[i] if signal_ids and i < len(signal_ids) else f"signal_{i}"

        degree = int(np.sum(network.adjacency_matrix[i]))
        neighbors = list(np.where(network.adjacency_matrix[i] > 0)[0])

        # Classify role
        if i in network.hub_indices:
            role = "hub"
        elif network.betweenness_centrality[i] > np.median(network.betweenness_centrality):
            role = "bridge"
        elif degree == 0:
            role = "isolated"
        else:
            role = "peripheral"

        profiles.append(NodeProfile(
            node_idx=i,
            signal_id=sig_id,
            degree=degree,
            degree_centrality=float(network.degree_centrality[i]),
            eigenvector_centrality=float(network.eigenvector_centrality[i]),
            betweenness_centrality=float(network.betweenness_centrality[i]),
            neighbors=neighbors,
            role=role
        ))

    return profiles
