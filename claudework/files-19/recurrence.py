#!/usr/bin/env python3
"""
Recurrence Analysis Engine
==========================

Computes attractor classification via Recurrence Quantification Analysis (RQA).

Recurrence plots reveal the structure of dynamical systems:
- Fixed point: high recurrence rate, high determinism
- Limit cycle: diagonal lines (deterministic periodicity)
- Strange attractor: complex structure, high entropy
- No attractor: low recurrence, low structure

Reference: Marwan et al. (2007) "Recurrence plots for the analysis of complex systems"
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.computation import RQAComputation
    PYRQA_AVAILABLE = True
except ImportError:
    PYRQA_AVAILABLE = False


@dataclass
class RecurrenceResult:
    """Result of recurrence quantification analysis."""
    attractor: str            # fixed_point | limit_cycle | strange | none
    
    # Core RQA metrics
    recurrence_rate: float    # RR: fraction of recurrent points
    determinism: float        # DET: fraction on diagonal lines
    laminarity: float         # LAM: fraction on vertical lines
    entropy: float            # ENTR: Shannon entropy of diagonal line lengths
    
    # Derived metrics
    trapping_time: float      # TT: average vertical line length
    max_diagonal: int         # Lmax: longest diagonal line
    divergence: float         # DIV: 1/Lmax (inverse stability)
    
    confidence: float
    method: str


def compute(signal: np.ndarray,
            emb_dim: int = 10,
            delay: int = 1,
            radius: Optional[float] = None,
            min_samples: int = 100) -> RecurrenceResult:
    """
    Compute recurrence quantification analysis for a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    emb_dim : int
        Embedding dimension
    delay : int
        Time delay for embedding
    radius : float, optional
        Neighborhood radius (auto if None, typically 10% of std)
    min_samples : int
        Minimum samples required
        
    Returns
    -------
    RecurrenceResult
        RQA metrics and attractor classification
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    
    # Validate
    if n < min_samples:
        return _empty_result("insufficient_data")
    
    # Remove NaN/Inf
    signal = signal[np.isfinite(signal)]
    if len(signal) < min_samples:
        return _empty_result("insufficient_valid_data")
    
    # Standardize
    std = np.std(signal)
    if std < 1e-10:
        return RecurrenceResult(
            attractor="fixed_point",
            recurrence_rate=1.0,
            determinism=1.0,
            laminarity=1.0,
            entropy=0.0,
            trapping_time=float('inf'),
            max_diagonal=len(signal),
            divergence=0.0,
            confidence=1.0,
            method="constant_signal"
        )
    
    signal = (signal - np.mean(signal)) / std
    
    # Auto-compute radius if not provided
    if radius is None:
        radius = 0.1 * std  # 10% of standard deviation
    
    # Compute using pyrqa if available
    if PYRQA_AVAILABLE:
        return _compute_pyrqa(signal, emb_dim, delay, radius)
    else:
        return _compute_manual(signal, emb_dim, delay, radius)


def _empty_result(method: str) -> RecurrenceResult:
    """Return empty result for error cases."""
    return RecurrenceResult(
        attractor="none",
        recurrence_rate=0.0,
        determinism=0.0,
        laminarity=0.0,
        entropy=0.0,
        trapping_time=0.0,
        max_diagonal=0,
        divergence=1.0,
        confidence=0.0,
        method=method
    )


def _compute_pyrqa(signal: np.ndarray, emb_dim: int, 
                   delay: int, radius: float) -> RecurrenceResult:
    """Compute using pyrqa library."""
    try:
        ts = TimeSeries(signal, embedding_dimension=emb_dim, time_delay=delay)
        settings = Settings(ts, neighbourhood=FixedRadius(radius))
        
        computation = RQAComputation.create(settings)
        result = computation.run()
        
        # Extract metrics
        RR = result.recurrence_rate
        DET = result.determinism
        LAM = result.laminarity
        ENTR = result.entropy_diagonal_lines
        TT = result.trapping_time
        Lmax = result.longest_diagonal_line
        DIV = result.divergence
        
        # Classify attractor
        attractor = _classify_attractor(RR, DET, LAM, ENTR)
        
        return RecurrenceResult(
            attractor=attractor,
            recurrence_rate=float(RR),
            determinism=float(DET),
            laminarity=float(LAM),
            entropy=float(ENTR),
            trapping_time=float(TT),
            max_diagonal=int(Lmax),
            divergence=float(DIV),
            confidence=0.9,
            method="pyrqa"
        )
        
    except Exception as e:
        return _compute_manual(signal, emb_dim, delay, radius)


def _compute_manual(signal: np.ndarray, emb_dim: int,
                    delay: int, radius: float) -> RecurrenceResult:
    """
    Manual recurrence quantification analysis.
    
    Steps:
    1. Embed signal in phase space
    2. Compute recurrence matrix
    3. Extract diagonal and vertical line statistics
    """
    n = len(signal)
    
    # Phase space reconstruction
    n_vectors = n - (emb_dim - 1) * delay
    if n_vectors < 20:
        return _empty_result("insufficient_for_embedding")
    
    # Build embedded vectors
    embedded = np.zeros((n_vectors, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = signal[i * delay : i * delay + n_vectors]
    
    # Compute recurrence matrix (distance-based)
    # Use subsampling for large n to avoid memory issues
    max_points = 500
    if n_vectors > max_points:
        indices = np.linspace(0, n_vectors - 1, max_points).astype(int)
        embedded = embedded[indices]
        n_vectors = max_points
    
    # Distance matrix
    recurrence_matrix = np.zeros((n_vectors, n_vectors), dtype=bool)
    
    for i in range(n_vectors):
        distances = np.linalg.norm(embedded - embedded[i], axis=1)
        recurrence_matrix[i] = distances < radius
    
    # Recurrence rate
    n_recurrent = np.sum(recurrence_matrix) - n_vectors  # Exclude diagonal
    RR = n_recurrent / (n_vectors * (n_vectors - 1))
    
    # Diagonal lines analysis
    diagonal_lengths = _extract_diagonal_lengths(recurrence_matrix)
    DET, ENTR, Lmax = _analyze_lines(diagonal_lengths, n_recurrent)
    
    # Vertical lines analysis
    vertical_lengths = _extract_vertical_lengths(recurrence_matrix)
    LAM, TT = _analyze_vertical_lines(vertical_lengths, n_recurrent)
    
    # Divergence
    DIV = 1.0 / Lmax if Lmax > 0 else 1.0
    
    # Classify attractor
    attractor = _classify_attractor(RR, DET, LAM, ENTR)
    
    return RecurrenceResult(
        attractor=attractor,
        recurrence_rate=float(RR),
        determinism=float(DET),
        laminarity=float(LAM),
        entropy=float(ENTR),
        trapping_time=float(TT),
        max_diagonal=int(Lmax),
        divergence=float(DIV),
        confidence=0.7,  # Lower confidence for manual method
        method="manual"
    )


def _extract_diagonal_lengths(R: np.ndarray, min_length: int = 2) -> list:
    """Extract lengths of diagonal lines from recurrence matrix."""
    n = R.shape[0]
    lengths = []
    
    # Check all diagonals (excluding main diagonal)
    for offset in range(1, n):
        # Upper diagonal
        diag = np.diag(R, offset)
        lengths.extend(_get_line_lengths(diag, min_length))
        
        # Lower diagonal
        diag = np.diag(R, -offset)
        lengths.extend(_get_line_lengths(diag, min_length))
    
    return lengths


def _extract_vertical_lengths(R: np.ndarray, min_length: int = 2) -> list:
    """Extract lengths of vertical lines from recurrence matrix."""
    n = R.shape[0]
    lengths = []
    
    for col in range(n):
        column = R[:, col]
        lengths.extend(_get_line_lengths(column, min_length))
    
    return lengths


def _get_line_lengths(arr: np.ndarray, min_length: int) -> list:
    """Get lengths of consecutive True values in boolean array."""
    lengths = []
    count = 0
    
    for val in arr:
        if val:
            count += 1
        else:
            if count >= min_length:
                lengths.append(count)
            count = 0
    
    if count >= min_length:
        lengths.append(count)
    
    return lengths


def _analyze_lines(lengths: list, n_recurrent: int) -> Tuple[float, float, int]:
    """Analyze diagonal line statistics."""
    if not lengths or n_recurrent == 0:
        return 0.0, 0.0, 0
    
    # Determinism: fraction of recurrent points forming diagonal lines
    points_on_diagonals = sum(lengths)
    DET = points_on_diagonals / n_recurrent if n_recurrent > 0 else 0
    
    # Entropy of line length distribution
    total = len(lengths)
    if total > 0:
        counts = {}
        for L in lengths:
            counts[L] = counts.get(L, 0) + 1
        
        probs = [c / total for c in counts.values()]
        ENTR = -sum(p * np.log(p) for p in probs if p > 0)
    else:
        ENTR = 0.0
    
    # Maximum diagonal line length
    Lmax = max(lengths) if lengths else 0
    
    return float(DET), float(ENTR), Lmax


def _analyze_vertical_lines(lengths: list, n_recurrent: int) -> Tuple[float, float]:
    """Analyze vertical line statistics."""
    if not lengths or n_recurrent == 0:
        return 0.0, 0.0
    
    # Laminarity: fraction on vertical lines
    points_on_verticals = sum(lengths)
    LAM = points_on_verticals / n_recurrent if n_recurrent > 0 else 0
    
    # Trapping time: average vertical line length
    TT = np.mean(lengths) if lengths else 0
    
    return float(LAM), float(TT)


def _classify_attractor(RR: float, DET: float, LAM: float, ENTR: float) -> str:
    """
    Classify attractor type based on RQA metrics.
    
    Rules (based on Marwan et al. 2007):
    - fixed_point: High RR, high DET, low ENTR
    - limit_cycle: High DET, periodic diagonal lines
    - strange: Moderate DET, high ENTR (complex structure)
    - none: Low RR (no recurrence)
    """
    # No attractor if recurrence is too low
    if RR < 0.01:
        return "none"
    
    # Fixed point: very high determinism, low entropy
    if DET > 0.95 and ENTR < 0.5:
        return "fixed_point"
    
    # Limit cycle: high determinism, moderate entropy
    if DET > 0.8 and ENTR < 2.0:
        return "limit_cycle"
    
    # Strange attractor: moderate determinism, high entropy
    if DET > 0.3 and ENTR > 1.5:
        return "strange"
    
    # Default based on determinism
    if DET > 0.5:
        return "limit_cycle"
    elif DET > 0.2:
        return "strange"
    else:
        return "none"


def compute_from_geometry(geometry: dict, 
                          prev_trajectory: Optional[str] = None) -> RecurrenceResult:
    """
    Estimate attractor type from geometry window data.
    
    When raw signal isn't available, use geometry metrics as proxy.
    
    Parameters
    ----------
    geometry : dict
        Geometry window with correlation/clustering metrics
    prev_trajectory : str, optional
        Trajectory classification (helps inform attractor type)
        
    Returns
    -------
    RecurrenceResult
        Estimated attractor with proxy metrics
    """
    # Extract proxy metrics
    mean_corr = geometry.get("mean_correlation", 0.5)
    silhouette = geometry.get("silhouette_score", 0.0)
    n_clusters = geometry.get("n_clusters", 1)
    stability_class = geometry.get("stability_class", "")
    
    # Estimate RQA metrics from proxies
    # High correlation → high recurrence rate
    RR_est = abs(mean_corr) * 0.8 + 0.1
    
    # Good clustering → deterministic structure
    DET_est = max(0, silhouette) * 0.5 + 0.3
    
    # Many clusters → complex structure → higher entropy
    ENTR_est = min(2.0, np.log(n_clusters + 1)) if n_clusters > 1 else 0.5
    
    # Stability affects laminarity
    LAM_est = 0.7 if "STABLE" in stability_class.upper() else 0.3
    
    # Classify based on estimated metrics
    attractor = _classify_attractor(RR_est, DET_est, LAM_est, ENTR_est)
    
    # Override with trajectory if available
    if prev_trajectory:
        if prev_trajectory == "chaotic":
            attractor = "strange"
        elif prev_trajectory == "periodic":
            attractor = "limit_cycle"
        elif prev_trajectory == "converging" and DET_est > 0.8:
            attractor = "fixed_point"
    
    return RecurrenceResult(
        attractor=attractor,
        recurrence_rate=RR_est,
        determinism=DET_est,
        laminarity=LAM_est,
        entropy=ENTR_est,
        trapping_time=0.0,
        max_diagonal=0,
        divergence=1.0 - DET_est,
        confidence=0.5,  # Lower confidence for proxy method
        method="geometry_proxy"
    )
