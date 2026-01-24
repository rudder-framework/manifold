"""
Geometry State
==============

Computes the 6-metric manifold geometry state at each timestamp.

The geometry state vector captures:
    1. DIMENSION    - How complex?
    2. TOPOLOGY     - What shape class?
    3. CURVATURE    - How bent?
    4. COHERENCE    - How tight?
    5. STABILITY    - How robust?
    6. ORIENTATION  - Is there a direction?

This provides a comprehensive view of manifold structure for each window.

Usage:
    from prism.manifold_geometry.geometry_state import compute_geometry_state

    state = compute_geometry_state(signals, signal_ids)
    print(state['curvature_sign'])  # "positive" | "negative" | "mixed" | "flat"
    print(state['topology_class'])  # "linear" | "bounded" | "periodic" | "attractor"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats


@dataclass
class GeometryState:
    """
    6-metric geometry state at a specific timestamp.

    Captures the complete manifold structure in a single object.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: str = ""
    unit_id: str = ""

    # === 1. DIMENSION ===
    dim_linear: int = 0              # PCA effective dimension
    dim_intrinsic: float = 0.0       # Nonlinear (estimated)
    dim_fractal: float = 0.0         # Correlation dimension

    # === 2. TOPOLOGY ===
    betti_0: int = 1                 # Connected components
    betti_1: int = 0                 # Holes/loops
    topology_class: str = "linear"   # "linear" | "bounded" | "periodic" | "attractor"

    # === 3. CURVATURE ===
    curvature_ollivier: float = 0.0  # Ollivier-Ricci (network)
    curvature_forman: float = 0.0    # Forman-Ricci (fast)
    curvature_geodesic: float = 0.0  # Manifold embedding
    curvature_sign: str = "flat"     # "positive" | "negative" | "mixed" | "flat"

    # === 4. COHERENCE ===
    coherence_tightness: float = 0.0     # Silhouette score
    coherence_uniformity: float = 0.0    # Distribution evenness
    reconstruction_error: float = 0.0    # Manifold fit quality

    # === 5. STABILITY ===
    stability_persistence: float = 0.0   # Topological persistence
    stability_curvature_var: float = 0.0 # Curvature variance over time

    # === 6. ORIENTATION ===
    anisotropy: float = 0.0          # Eigenvalue ratio (0=isotropic, 1=directional)
    principal_direction: Optional[np.ndarray] = None

    # === META ===
    n_signals: int = 0
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            # Dimension
            'dim_linear': self.dim_linear,
            'dim_intrinsic': self.dim_intrinsic,
            'dim_fractal': self.dim_fractal,
            # Topology
            'betti_0': self.betti_0,
            'betti_1': self.betti_1,
            'topology_class': self.topology_class,
            # Curvature
            'curvature_ollivier': self.curvature_ollivier,
            'curvature_forman': self.curvature_forman,
            'curvature_geodesic': self.curvature_geodesic,
            'curvature_sign': self.curvature_sign,
            # Coherence
            'coherence_tightness': self.coherence_tightness,
            'coherence_uniformity': self.coherence_uniformity,
            'reconstruction_error': self.reconstruction_error,
            # Stability
            'stability_persistence': self.stability_persistence,
            'stability_curvature_var': self.stability_curvature_var,
            # Orientation
            'anisotropy': self.anisotropy,
            # Meta
            'n_signals': self.n_signals,
            'computed_at': self.computed_at.isoformat() if isinstance(self.computed_at, datetime) else str(self.computed_at),
        }


def compute_geometry_state(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None,
    entity_id: str = "",
    timestamp: Optional[datetime] = None,
    compute_ollivier: bool = False,  # Expensive, default off
) -> GeometryState:
    """
    Compute 6-metric geometry state for a signal window.

    Args:
        signals: (n_signals, n_observations) array
        signal_ids: Optional signal identifiers
        entity_id: Entity identifier
        timestamp: Timestamp for this window
        compute_ollivier: Whether to compute expensive Ollivier-Ricci

    Returns:
        GeometryState with all 6 metric categories
    """
    signals = np.asarray(signals)
    n_signals, n_obs = signals.shape

    state = GeometryState(
        timestamp=timestamp or datetime.now(),
        entity_id=entity_id,
        unit_id=entity_id,
        n_signals=n_signals,
    )

    if n_signals < 2:
        return state

    # === 1. DIMENSION ===
    state.dim_linear, state.dim_intrinsic = _compute_dimensions(signals)
    state.dim_fractal = _estimate_fractal_dimension(signals)

    # === 2. TOPOLOGY ===
    state.betti_0, state.betti_1, state.topology_class = _compute_topology(signals)

    # === 3. CURVATURE ===
    # Forman (fast)
    try:
        from .forman_ricci import compute as compute_forman
        forman_result = compute_forman(signals, signal_ids)
        state.curvature_forman = forman_result.mean_curvature
        state.curvature_sign = forman_result.curvature_sign
    except Exception:
        pass

    # Ollivier (expensive, optional)
    if compute_ollivier:
        try:
            from .ollivier_ricci import compute as compute_ollivier_fn
            ollivier_result = compute_ollivier_fn(signals, signal_ids)
            state.curvature_ollivier = ollivier_result.mean_curvature
            if state.curvature_sign == "flat":
                state.curvature_sign = ollivier_result.curvature_sign
        except Exception:
            pass

    # Geodesic
    try:
        from .geodesic_curvature import compute as compute_geodesic
        geodesic_result = compute_geodesic(signals)
        state.curvature_geodesic = geodesic_result.curvature
        state.reconstruction_error = geodesic_result.reconstruction_error
    except Exception:
        pass

    # === 4. COHERENCE ===
    state.coherence_tightness, state.coherence_uniformity = _compute_coherence(signals)

    # === 5. STABILITY ===
    state.stability_persistence = _compute_persistence(signals)

    # === 6. ORIENTATION ===
    state.anisotropy, state.principal_direction = _compute_orientation(signals)

    return state


def _compute_dimensions(signals: np.ndarray) -> tuple:
    """Compute linear and intrinsic dimensions."""
    n_signals = signals.shape[0]

    # Correlation matrix
    corr = np.corrcoef(signals)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = eigenvalues[::-1]  # Descending
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Linear dimension (PCA)
    total_var = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total_var
    dim_linear = np.searchsorted(cumulative, 0.95) + 1
    dim_linear = min(dim_linear, n_signals)

    # Intrinsic dimension (entropy-based)
    probs = eigenvalues / total_var
    probs = probs[probs > 1e-10]
    entropy = -np.sum(probs * np.log(probs))
    dim_intrinsic = np.exp(entropy)

    return int(dim_linear), float(dim_intrinsic)


def _estimate_fractal_dimension(signals: np.ndarray) -> float:
    """Estimate correlation/fractal dimension."""
    n_signals = signals.shape[0]

    if n_signals < 4:
        return float(n_signals)

    # Use correlation matrix eigenvalue spectrum
    corr = np.corrcoef(signals)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = eigenvalues[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Fractal dimension estimate from eigenvalue decay
    # D ≈ sum(λ_i^2) / (sum(λ_i))^2 * n
    sum_sq = np.sum(eigenvalues ** 2)
    sum_val = np.sum(eigenvalues)

    if sum_val > 0:
        participation_ratio = sum_val ** 2 / (n_signals * sum_sq)
        dim_fractal = participation_ratio * n_signals
    else:
        dim_fractal = 1.0

    return float(dim_fractal)


def _compute_topology(signals: np.ndarray) -> tuple:
    """Compute topological invariants."""
    n_signals = signals.shape[0]

    # Correlation matrix
    corr = np.abs(np.corrcoef(signals))
    np.fill_diagonal(corr, 0)

    # Betti_0: Connected components (via thresholded graph)
    threshold = 0.3
    adjacency = (corr > threshold).astype(int)

    # Simple component counting via BFS
    visited = set()
    n_components = 0

    for i in range(n_signals):
        if i not in visited:
            n_components += 1
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    neighbors = np.where(adjacency[node] > 0)[0]
                    stack.extend([n for n in neighbors if n not in visited])

    betti_0 = n_components

    # Betti_1: Estimate loops (triangles indicate potential loops)
    n_triangles = 0
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            if adjacency[i, j]:
                for k in range(j+1, n_signals):
                    if adjacency[j, k] and adjacency[i, k]:
                        n_triangles += 1

    betti_1 = max(0, n_triangles - n_signals + betti_0)  # Euler characteristic

    # Topology class
    if betti_0 > 1:
        topology_class = "fragmented"
    elif betti_1 > 0:
        topology_class = "periodic"  # Has loops
    elif np.mean(corr) > 0.5:
        topology_class = "bounded"  # Tightly connected
    else:
        topology_class = "linear"

    return betti_0, betti_1, topology_class


def _compute_coherence(signals: np.ndarray) -> tuple:
    """Compute coherence metrics."""
    n_signals = signals.shape[0]

    # Correlation matrix
    corr = np.corrcoef(signals)

    # Tightness: How clustered are correlations?
    upper_tri = corr[np.triu_indices(n_signals, k=1)]
    tightness = 1.0 - np.std(upper_tri) if len(upper_tri) > 0 else 0.0

    # Uniformity: How evenly distributed are correlation values?
    if len(upper_tri) > 0:
        hist, _ = np.histogram(upper_tri, bins=10, range=(-1, 1))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(10)
        uniformity = entropy / max_entropy
    else:
        uniformity = 0.0

    return float(tightness), float(uniformity)


def _compute_persistence(signals: np.ndarray) -> float:
    """Compute topological persistence (stability of structure)."""
    n_signals = signals.shape[0]

    # Correlation matrix
    corr = np.abs(np.corrcoef(signals))
    np.fill_diagonal(corr, 0)

    # Persistence: How much does structure change with threshold?
    thresholds = np.linspace(0.1, 0.9, 9)
    component_counts = []

    for thresh in thresholds:
        adjacency = (corr > thresh).astype(int)
        visited = set()
        n_comp = 0
        for i in range(n_signals):
            if i not in visited:
                n_comp += 1
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        neighbors = np.where(adjacency[node] > 0)[0]
                        stack.extend([n for n in neighbors if n not in visited])
        component_counts.append(n_comp)

    # Persistence = inverse of variability
    if len(component_counts) > 1:
        persistence = 1.0 / (1.0 + np.std(component_counts))
    else:
        persistence = 0.5

    return float(persistence)


def _compute_orientation(signals: np.ndarray) -> tuple:
    """Compute orientation (anisotropy and principal direction)."""
    n_signals = signals.shape[0]

    # PCA on correlation matrix
    corr = np.corrcoef(signals)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Anisotropy: ratio of first to sum of rest
    if np.sum(eigenvalues[1:]) > 0:
        anisotropy = eigenvalues[0] / np.sum(eigenvalues)
    else:
        anisotropy = 1.0

    # Principal direction
    principal_direction = eigenvectors[:, 0]

    return float(anisotropy), principal_direction


def geometry_state_to_string(state: GeometryState) -> str:
    """
    Convert geometry state to a compact string representation.

    Format: "TOPOLOGY.CURVATURE_SIGN.DIM_LINEAR"
    Example: "BOUNDED.POSITIVE.3"
    """
    return f"{state.topology_class.upper()}.{state.curvature_sign.upper()}.{state.dim_linear}"
