"""
Ollivier-Ricci Curvature Engine
===============================

Gold standard discrete Ricci curvature for network/signal analysis.

Based on optimal transport theory - measures how "expensive" it is to move
probability mass between neighborhoods of connected nodes.

Key insight (Sandhu/Tannenbaum 2016):
    Curvature is negatively correlated with network fragility.
    Curvature DROP = system becoming more fragile = regime change incoming.

This is the EARLY WARNING SYSTEM for manifold geometry.

References:
    - Ollivier (2009). "Ricci curvature of Markov chains on metric spaces"
    - Sandhu et al. (2016). "Ricci curvature: An economic indicator for 
      market fragility and systemic risk." Science Advances.
    - Gosztolai & Arnaudon (2021). "Unfolding the multiscale structure of 
      networks with dynamical Ollivier-Ricci curvature." Nature Communications.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import linprog


@dataclass
class OllivierRicciResult:
    """Result from Ollivier-Ricci curvature computation"""
    
    # Edge curvatures
    edge_curvatures: np.ndarray      # Curvature per edge
    
    # Summary statistics
    mean_curvature: float            # Average across all edges
    min_curvature: float             # Most negative (most fragile)
    max_curvature: float             # Most positive (most robust)
    curvature_std: float             # Variance in curvature
    
    # Distribution
    n_positive: int                  # Edges with κ > 0
    n_negative: int                  # Edges with κ < 0
    n_flat: int                      # Edges with κ ≈ 0
    
    # Fragility indicator
    fragility_score: float           # 0-1 scale, higher = more fragile
    curvature_sign: str              # "positive" | "negative" | "mixed" | "flat"
    
    # Node-level aggregation
    node_curvatures: np.ndarray      # Average curvature per node


def _wasserstein_1d(p: np.ndarray, q: np.ndarray, 
                    cost_matrix: np.ndarray) -> float:
    """
    Compute 1-Wasserstein distance between discrete distributions.
    
    Uses linear programming to solve optimal transport problem.
    
    Args:
        p: Source distribution (sums to 1)
        q: Target distribution (sums to 1)
        cost_matrix: Pairwise costs between support points
        
    Returns:
        Wasserstein-1 distance
    """
    n = len(p)
    m = len(q)
    
    # Flatten cost matrix for LP
    c = cost_matrix.flatten()
    
    # Constraints: row sums = p, column sums = q
    # Build constraint matrix
    A_eq = np.zeros((n + m, n * m))
    
    # Row sum constraints
    for i in range(n):
        A_eq[i, i*m:(i+1)*m] = 1
    
    # Column sum constraints  
    for j in range(m):
        A_eq[n + j, j::m] = 1
    
    b_eq = np.concatenate([p, q])
    
    # Solve LP
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    if result.success:
        return result.fun
    else:
        # Fallback to simpler approximation
        return np.sum(np.abs(p - q)) * np.mean(cost_matrix)


def _build_correlation_graph(signals: np.ndarray, 
                             threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build weighted graph from signal correlations.
    
    Args:
        signals: (n_signals, n_observations) array
        threshold: Minimum |correlation| to create edge
        
    Returns:
        adjacency: Weighted adjacency matrix
        edges: List of (i, j) edge pairs
    """
    n_signals = signals.shape[0]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(signals)
    
    # Create adjacency (use absolute correlation as weight)
    adjacency = np.abs(corr_matrix)
    adjacency[adjacency < threshold] = 0
    np.fill_diagonal(adjacency, 0)
    
    # Extract edges
    edges = []
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            if adjacency[i, j] > 0:
                edges.append((i, j))
    
    return adjacency, edges


def _node_distribution(adjacency: np.ndarray, node: int, 
                       alpha: float = 0.5) -> np.ndarray:
    """
    Compute probability distribution on neighborhood of node.
    
    Uses lazy random walk: with probability alpha stay at node,
    with probability (1-alpha) move to neighbor proportional to edge weight.
    
    Args:
        adjacency: Weighted adjacency matrix
        node: Node index
        alpha: Laziness parameter (0.5 standard)
        
    Returns:
        Distribution over all nodes (mostly zeros except neighborhood)
    """
    n = adjacency.shape[0]
    dist = np.zeros(n)
    
    # Self-loop probability
    dist[node] = alpha
    
    # Neighbor probabilities
    neighbors = adjacency[node]
    neighbor_sum = np.sum(neighbors)
    
    if neighbor_sum > 0:
        dist += (1 - alpha) * neighbors / neighbor_sum
    else:
        dist[node] = 1.0  # Isolated node
    
    return dist


def compute_edge_curvature(adjacency: np.ndarray, 
                           edge: Tuple[int, int],
                           alpha: float = 0.5) -> float:
    """
    Compute Ollivier-Ricci curvature for a single edge.
    
    κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
    
    Where:
        W₁ = Wasserstein-1 distance
        μₓ, μᵧ = probability distributions on neighborhoods
        d(x,y) = graph distance (here: 1/weight or 1 for unweighted)
    
    Args:
        adjacency: Weighted adjacency matrix
        edge: (node_i, node_j) tuple
        alpha: Laziness parameter for distributions
        
    Returns:
        Curvature value (can be negative)
    """
    i, j = edge
    n = adjacency.shape[0]
    
    # Get distributions on neighborhoods
    mu_i = _node_distribution(adjacency, i, alpha)
    mu_j = _node_distribution(adjacency, j, alpha)
    
    # Graph distance matrix (shortest paths)
    # For simplicity, use 1/weight as distance
    with np.errstate(divide='ignore'):
        distance_matrix = np.where(adjacency > 0, 1.0 / adjacency, np.inf)
    np.fill_diagonal(distance_matrix, 0)
    
    # Floyd-Warshall for shortest paths (small graphs)
    if n <= 100:
        for k in range(n):
            for ii in range(n):
                for jj in range(n):
                    if distance_matrix[ii, k] + distance_matrix[k, jj] < distance_matrix[ii, jj]:
                        distance_matrix[ii, jj] = distance_matrix[ii, k] + distance_matrix[k, jj]
    
    # Replace inf with large value
    distance_matrix[np.isinf(distance_matrix)] = n * 10
    
    # Edge distance
    d_ij = distance_matrix[i, j]
    if d_ij == 0:
        d_ij = 1.0
    
    # Compute Wasserstein distance
    W1 = _wasserstein_1d(mu_i, mu_j, distance_matrix)
    
    # Curvature
    kappa = 1.0 - W1 / d_ij
    
    return kappa


def compute(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    correlation_threshold: float = 0.3,
    alpha: float = 0.5
) -> OllivierRicciResult:
    """
    Compute Ollivier-Ricci curvature for signal correlation network.
    
    This is the EXPENSIVE but ACCURATE curvature measure.
    Use for:
        - Final analysis
        - Regime change detection
        - Fragility assessment
    
    Args:
        signals: (n_signals, n_observations) array
        signal_ids: Optional signal identifiers
        correlation_threshold: Minimum |correlation| for edge
        alpha: Laziness parameter (0.5 standard)
        
    Returns:
        OllivierRicciResult with curvatures and fragility metrics
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]
    
    # Build graph
    adjacency, edges = _build_correlation_graph(signals, correlation_threshold)
    
    if len(edges) == 0:
        # No edges - fully disconnected
        return OllivierRicciResult(
            edge_curvatures=np.array([]),
            mean_curvature=0.0,
            min_curvature=0.0,
            max_curvature=0.0,
            curvature_std=0.0,
            n_positive=0,
            n_negative=0,
            n_flat=0,
            fragility_score=1.0,  # Maximum fragility
            curvature_sign="flat",
            node_curvatures=np.zeros(n_signals)
        )
    
    # Compute curvature for each edge
    edge_curvatures = np.array([
        compute_edge_curvature(adjacency, edge, alpha) 
        for edge in edges
    ])
    
    # Summary statistics
    mean_curv = np.mean(edge_curvatures)
    min_curv = np.min(edge_curvatures)
    max_curv = np.max(edge_curvatures)
    std_curv = np.std(edge_curvatures)
    
    # Distribution analysis
    flat_threshold = 0.05
    n_positive = np.sum(edge_curvatures > flat_threshold)
    n_negative = np.sum(edge_curvatures < -flat_threshold)
    n_flat = len(edge_curvatures) - n_positive - n_negative
    
    # Curvature sign classification
    if n_positive > 2 * n_negative:
        curvature_sign = "positive"
    elif n_negative > 2 * n_positive:
        curvature_sign = "negative"
    elif n_flat > (n_positive + n_negative):
        curvature_sign = "flat"
    else:
        curvature_sign = "mixed"
    
    # Fragility score (higher = more fragile)
    # Based on Sandhu et al.: fragility negatively correlated with curvature
    # Normalize to 0-1 scale
    fragility_score = 1.0 / (1.0 + np.exp(mean_curv * 5))  # Sigmoid transform
    
    # Node-level curvature (average of incident edges)
    node_curvatures = np.zeros(n_signals)
    node_counts = np.zeros(n_signals)
    
    for idx, (i, j) in enumerate(edges):
        node_curvatures[i] += edge_curvatures[idx]
        node_curvatures[j] += edge_curvatures[idx]
        node_counts[i] += 1
        node_counts[j] += 1
    
    node_counts[node_counts == 0] = 1  # Avoid division by zero
    node_curvatures /= node_counts
    
    return OllivierRicciResult(
        edge_curvatures=edge_curvatures,
        mean_curvature=float(mean_curv),
        min_curvature=float(min_curv),
        max_curvature=float(max_curv),
        curvature_std=float(std_curv),
        n_positive=int(n_positive),
        n_negative=int(n_negative),
        n_flat=int(n_flat),
        fragility_score=float(fragility_score),
        curvature_sign=curvature_sign,
        node_curvatures=node_curvatures
    )


def compute_temporal(
    signals: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
    **kwargs
) -> List[OllivierRicciResult]:
    """
    Compute Ollivier-Ricci curvature over sliding windows.
    
    Tracks curvature evolution - key for detecting regime changes.
    
    Args:
        signals: (n_signals, n_observations) array
        window_size: Size of each window
        stride: Step between windows
        **kwargs: Passed to compute()
        
    Returns:
        List of OllivierRicciResult, one per window
    """
    n_obs = signals.shape[1]
    results = []
    
    for start in range(0, n_obs - window_size + 1, stride):
        window = signals[:, start:start + window_size]
        result = compute(window, **kwargs)
        results.append(result)
    
    return results


def detect_curvature_anomaly(
    curvature_history: List[float],
    lookback: int = 10,
    zscore_threshold: float = 2.0
) -> Tuple[bool, float]:
    """
    Detect anomalous curvature drop (fragility increasing).
    
    Args:
        curvature_history: List of mean curvatures over time
        lookback: Window for baseline
        zscore_threshold: Threshold for anomaly
        
    Returns:
        (is_anomaly, z_score)
    """
    if len(curvature_history) < lookback + 1:
        return False, 0.0
    
    baseline = curvature_history[-lookback-1:-1]
    current = curvature_history[-1]
    
    mean_baseline = np.mean(baseline)
    std_baseline = np.std(baseline)
    
    if std_baseline < 1e-6:
        return False, 0.0
    
    zscore = (current - mean_baseline) / std_baseline
    
    # Negative z-score = curvature dropped = fragility increased
    is_anomaly = zscore < -zscore_threshold
    
    return is_anomaly, float(zscore)
