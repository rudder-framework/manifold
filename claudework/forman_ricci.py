"""
Forman-Ricci Curvature Engine
=============================

Fast discrete Ricci curvature for large-scale network analysis.

Forman-Ricci is a combinatorial curvature based on edge degrees and triangles.
It is 10-100x faster than Ollivier-Ricci while maintaining high correlation
(r > 0.7) in most real-world networks.

Use this for:
    - Real-time monitoring
    - Large signal sets (100+ signals)
    - Initial screening before expensive Ollivier-Ricci
    - Edge importance analysis

References:
    - Forman (2003). "Bochner's method for cell complexes and combinatorial 
      Ricci curvature." Discrete & Computational Geometry.
    - Sreejith et al. (2016). "Forman curvature for complex networks."
      Journal of Statistical Mechanics.
    - Samal et al. (2018). "Comparative analysis of two discretizations of 
      Ricci curvature for complex networks." Scientific Reports.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class FormanRicciResult:
    """Result from Forman-Ricci curvature computation"""
    
    # Edge curvatures
    edge_curvatures: np.ndarray      # Curvature per edge
    edge_list: List[Tuple[int, int]] # Edge indices
    
    # Summary statistics
    mean_curvature: float
    min_curvature: float
    max_curvature: float
    curvature_std: float
    
    # Distribution
    n_positive: int
    n_negative: int
    n_flat: int
    
    # Derived metrics
    curvature_sign: str              # "positive" | "negative" | "mixed" | "flat"
    fragility_score: float           # 0-1 scale
    
    # Node-level
    node_curvatures: np.ndarray
    
    # Augmented version (includes triangles)
    augmented_curvatures: Optional[np.ndarray] = None


def _count_triangles(adjacency: np.ndarray, edge: Tuple[int, int]) -> int:
    """
    Count triangles containing an edge.
    
    A triangle exists if nodes i, j share a common neighbor k.
    
    Args:
        adjacency: Binary or weighted adjacency matrix
        edge: (i, j) edge tuple
        
    Returns:
        Number of triangles containing this edge
    """
    i, j = edge
    n = adjacency.shape[0]
    
    # Find common neighbors
    neighbors_i = set(np.where(adjacency[i] > 0)[0])
    neighbors_j = set(np.where(adjacency[j] > 0)[0])
    
    # Remove i and j themselves
    neighbors_i.discard(i)
    neighbors_i.discard(j)
    neighbors_j.discard(i)
    neighbors_j.discard(j)
    
    # Common neighbors = triangles
    common = neighbors_i & neighbors_j
    
    return len(common)


def _compute_forman_curvature_edge(
    adjacency: np.ndarray,
    edge: Tuple[int, int],
    include_triangles: bool = True
) -> float:
    """
    Compute Forman-Ricci curvature for a single edge.
    
    Basic formula (unweighted):
        F(e) = 4 - d(v1) - d(v2)
    
    Where d(v) is the degree of vertex v.
    
    Augmented formula (includes triangles):
        F(e) = 4 - d(v1) - d(v2) + 3 * #triangles(e)
    
    For weighted graphs:
        F(e) = w(e) * [w(v1)/sqrt(sum_e1 w) + w(v2)/sqrt(sum_e2 w)] 
               - w(e) * [sum parallel edges] + w(e) * [sum triangles]
    
    Args:
        adjacency: Adjacency matrix (weighted or binary)
        edge: (i, j) edge tuple
        include_triangles: Whether to use augmented formula
        
    Returns:
        Forman-Ricci curvature value
    """
    i, j = edge
    
    # Edge weight
    w_e = adjacency[i, j]
    if w_e == 0:
        return 0.0
    
    # Node degrees (weighted)
    d_i = np.sum(adjacency[i])
    d_j = np.sum(adjacency[j])
    
    # Basic Forman curvature
    # F(e) = w(e) * (1/sqrt(w(i)) + 1/sqrt(w(j))) - w(e)
    # Simplified for correlation networks:
    F = 4 - d_i - d_j
    
    # Augmented: add triangle contribution
    if include_triangles:
        n_triangles = _count_triangles(adjacency > 0, edge)
        F += 3 * n_triangles
    
    return F


def compute(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    correlation_threshold: float = 0.3,
    include_triangles: bool = True
) -> FormanRicciResult:
    """
    Compute Forman-Ricci curvature for signal correlation network.
    
    This is the FAST curvature measure - O(E) complexity.
    Use for:
        - Real-time monitoring
        - Large networks
        - Initial screening
    
    Args:
        signals: (n_signals, n_observations) array
        signal_ids: Optional signal identifiers  
        correlation_threshold: Minimum |correlation| for edge
        include_triangles: Use augmented formula (recommended)
        
    Returns:
        FormanRicciResult with curvatures and metrics
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(signals)
    
    # Build adjacency (absolute correlation as weight)
    adjacency = np.abs(corr_matrix)
    adjacency[adjacency < correlation_threshold] = 0
    np.fill_diagonal(adjacency, 0)
    
    # Extract edges
    edges = []
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            if adjacency[i, j] > 0:
                edges.append((i, j))
    
    if len(edges) == 0:
        return FormanRicciResult(
            edge_curvatures=np.array([]),
            edge_list=[],
            mean_curvature=0.0,
            min_curvature=0.0,
            max_curvature=0.0,
            curvature_std=0.0,
            n_positive=0,
            n_negative=0,
            n_flat=0,
            curvature_sign="flat",
            fragility_score=1.0,
            node_curvatures=np.zeros(n_signals)
        )
    
    # Compute curvature for each edge (FAST - just degree lookups)
    edge_curvatures = np.array([
        _compute_forman_curvature_edge(adjacency, edge, include_triangles)
        for edge in edges
    ])
    
    # Also compute without triangles for comparison
    if include_triangles:
        basic_curvatures = np.array([
            _compute_forman_curvature_edge(adjacency, edge, False)
            for edge in edges
        ])
    else:
        basic_curvatures = None
    
    # Summary statistics
    mean_curv = np.mean(edge_curvatures)
    min_curv = np.min(edge_curvatures)
    max_curv = np.max(edge_curvatures)
    std_curv = np.std(edge_curvatures)
    
    # Distribution (different thresholds than Ollivier due to different scale)
    flat_threshold = 0.5
    n_positive = np.sum(edge_curvatures > flat_threshold)
    n_negative = np.sum(edge_curvatures < -flat_threshold)
    n_flat = len(edge_curvatures) - n_positive - n_negative
    
    # Sign classification
    if n_positive > 2 * n_negative:
        curvature_sign = "positive"
    elif n_negative > 2 * n_positive:
        curvature_sign = "negative"
    elif n_flat > (n_positive + n_negative):
        curvature_sign = "flat"
    else:
        curvature_sign = "mixed"
    
    # Fragility score (normalize Forman to 0-1 scale)
    # Forman curvature can range widely, so use tanh normalization
    fragility_score = 0.5 * (1 - np.tanh(mean_curv / 10))
    
    # Node-level curvature
    node_curvatures = np.zeros(n_signals)
    node_counts = np.zeros(n_signals)
    
    for idx, (i, j) in enumerate(edges):
        node_curvatures[i] += edge_curvatures[idx]
        node_curvatures[j] += edge_curvatures[idx]
        node_counts[i] += 1
        node_counts[j] += 1
    
    node_counts[node_counts == 0] = 1
    node_curvatures /= node_counts
    
    return FormanRicciResult(
        edge_curvatures=edge_curvatures,
        edge_list=edges,
        mean_curvature=float(mean_curv),
        min_curvature=float(min_curv),
        max_curvature=float(max_curv),
        curvature_std=float(std_curv),
        n_positive=int(n_positive),
        n_negative=int(n_negative),
        n_flat=int(n_flat),
        curvature_sign=curvature_sign,
        fragility_score=float(fragility_score),
        node_curvatures=node_curvatures,
        augmented_curvatures=edge_curvatures if include_triangles else None
    )


def compute_temporal(
    signals: np.ndarray,
    window_size: int = 50,
    stride: int = 10,
    **kwargs
) -> List[FormanRicciResult]:
    """
    Compute Forman-Ricci curvature over sliding windows.
    
    Args:
        signals: (n_signals, n_observations) array
        window_size: Size of each window
        stride: Step between windows
        **kwargs: Passed to compute()
        
    Returns:
        List of FormanRicciResult, one per window
    """
    n_obs = signals.shape[1]
    results = []
    
    for start in range(0, n_obs - window_size + 1, stride):
        window = signals[:, start:start + window_size]
        result = compute(window, **kwargs)
        results.append(result)
    
    return results


def identify_critical_edges(
    result: FormanRicciResult,
    n_critical: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Identify most critical edges (lowest curvature = most fragile).
    
    These are the "bottleneck" edges that, if broken, would most
    affect network connectivity.
    
    Args:
        result: FormanRicciResult from compute()
        n_critical: Number of critical edges to return
        
    Returns:
        List of (edge, curvature) tuples, sorted by curvature (ascending)
    """
    if len(result.edge_curvatures) == 0:
        return []
    
    # Sort by curvature (most negative first)
    indices = np.argsort(result.edge_curvatures)
    
    critical = []
    for idx in indices[:n_critical]:
        edge = result.edge_list[idx]
        curv = result.edge_curvatures[idx]
        critical.append((edge, float(curv)))
    
    return critical


def compute_ricci_flow(
    signals: np.ndarray,
    n_iterations: int = 10,
    step_size: float = 0.1,
    correlation_threshold: float = 0.3
) -> List[FormanRicciResult]:
    """
    Simulate Ricci flow on the correlation network.
    
    Ricci flow evolves edge weights to "smooth out" curvature differences.
    Edges with negative curvature get strengthened, positive get weakened.
    
    This reveals the natural community structure of the network.
    
    Args:
        signals: (n_signals, n_observations) array
        n_iterations: Number of flow iterations
        step_size: Flow step size
        correlation_threshold: Initial edge threshold
        
    Returns:
        List of FormanRicciResult at each iteration
    """
    signals = np.asarray(signals)
    n_signals = signals.shape[0]
    
    # Initial correlation matrix
    corr_matrix = np.corrcoef(signals)
    adjacency = np.abs(corr_matrix)
    adjacency[adjacency < correlation_threshold] = 0
    np.fill_diagonal(adjacency, 0)
    
    results = []
    
    for iteration in range(n_iterations):
        # Get edges
        edges = []
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                if adjacency[i, j] > 0:
                    edges.append((i, j))
        
        if len(edges) == 0:
            break
        
        # Compute current curvatures
        curvatures = np.array([
            _compute_forman_curvature_edge(adjacency, edge, True)
            for edge in edges
        ])
        
        # Update edge weights based on curvature
        # Ricci flow: dw/dt = -κ * w
        for idx, (i, j) in enumerate(edges):
            kappa = curvatures[idx]
            w = adjacency[i, j]
            
            # Flow update
            new_w = w * (1 - step_size * kappa / 10)  # Normalized
            new_w = np.clip(new_w, 0.01, 1.0)  # Keep bounded
            
            adjacency[i, j] = adjacency[j, i] = new_w
        
        # Compute result for this iteration
        # (Recompute on modified adjacency)
        result = FormanRicciResult(
            edge_curvatures=curvatures,
            edge_list=edges,
            mean_curvature=float(np.mean(curvatures)),
            min_curvature=float(np.min(curvatures)),
            max_curvature=float(np.max(curvatures)),
            curvature_std=float(np.std(curvatures)),
            n_positive=int(np.sum(curvatures > 0.5)),
            n_negative=int(np.sum(curvatures < -0.5)),
            n_flat=int(np.sum(np.abs(curvatures) <= 0.5)),
            curvature_sign="mixed",
            fragility_score=0.5 * (1 - np.tanh(np.mean(curvatures) / 10)),
            node_curvatures=np.zeros(n_signals)
        )
        results.append(result)
    
    return results


def compare_with_ollivier(
    forman_result: FormanRicciResult,
    ollivier_curvatures: np.ndarray
) -> Dict[str, float]:
    """
    Compare Forman-Ricci with Ollivier-Ricci curvatures.
    
    Useful for validating that Forman is appropriate for your data.
    Correlation should be > 0.7 for most real-world networks.
    
    Args:
        forman_result: Result from Forman compute()
        ollivier_curvatures: Edge curvatures from Ollivier compute()
        
    Returns:
        Dictionary with correlation and comparison metrics
    """
    from scipy import stats
    
    forman = forman_result.edge_curvatures
    ollivier = ollivier_curvatures
    
    if len(forman) != len(ollivier):
        raise ValueError("Curvature arrays must have same length")
    
    if len(forman) < 3:
        return {
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "sign_agreement": 0.0,
            "mean_diff": 0.0
        }
    
    # Pearson correlation
    pearson_r, _ = stats.pearsonr(forman, ollivier)
    
    # Spearman (rank) correlation
    spearman_r, _ = stats.spearmanr(forman, ollivier)
    
    # Sign agreement
    forman_sign = np.sign(forman)
    ollivier_sign = np.sign(ollivier)
    sign_agreement = np.mean(forman_sign == ollivier_sign)
    
    # Mean absolute difference (after normalization)
    forman_norm = (forman - np.mean(forman)) / (np.std(forman) + 1e-6)
    ollivier_norm = (ollivier - np.mean(ollivier)) / (np.std(ollivier) + 1e-6)
    mean_diff = np.mean(np.abs(forman_norm - ollivier_norm))
    
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "sign_agreement": float(sign_agreement),
        "mean_diff": float(mean_diff)
    }
