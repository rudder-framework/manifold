#!/usr/bin/env python3
"""
Permutation Entropy Engine
==========================

Computes predictability via permutation entropy.

Permutation entropy (Bandt & Pompe, 2002) measures complexity through
ordinal patterns - robust to noise and monotonic transformations.

Output:
- entropy: 0 (deterministic) to 1 (random)
- predictability: 1 - entropy (inverted for intuitive interpretation)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from math import factorial

try:
    import ordpy
    ORDPY_AVAILABLE = True
except ImportError:
    ORDPY_AVAILABLE = False


@dataclass
class PermutationEntropyResult:
    """Result of permutation entropy computation."""
    entropy: float                # Normalized entropy [0, 1]
    predictability: float         # 1 - entropy [0, 1]
    complexity: float             # Statistical complexity [0, 1]
    pattern_distribution: dict    # Count of each ordinal pattern
    dominant_pattern: Optional[Tuple[int, ...]]  # Most frequent pattern
    n_patterns_observed: int      # Number of unique patterns seen
    n_patterns_possible: int      # d! possible patterns
    confidence: float             # Data quality indicator
    method: str


def compute(signal: np.ndarray,
            order: int = 3,
            delay: int = 1,
            min_samples: int = 50) -> PermutationEntropyResult:
    """
    Compute permutation entropy for a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    order : int
        Embedding dimension (pattern length). Typically 3-7.
        - order=3: 6 possible patterns (fast, less sensitive)
        - order=5: 120 possible patterns (slower, more sensitive)
    delay : int
        Time delay between elements in pattern
    min_samples : int
        Minimum samples required
        
    Returns
    -------
    PermutationEntropyResult
        Contains entropy, predictability, complexity, and pattern info
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
    
    # Check for constant signal
    if np.std(signal) < 1e-10:
        return PermutationEntropyResult(
            entropy=0.0,
            predictability=1.0,
            complexity=0.0,
            pattern_distribution={},
            dominant_pattern=(0, 1, 2)[:order],
            n_patterns_observed=1,
            n_patterns_possible=factorial(order),
            confidence=1.0,
            method="constant_signal"
        )
    
    # Compute using ordpy if available
    if ORDPY_AVAILABLE:
        return _compute_ordpy(signal, order, delay)
    else:
        return _compute_manual(signal, order, delay)


def _empty_result(method: str) -> PermutationEntropyResult:
    """Return empty result for error cases."""
    return PermutationEntropyResult(
        entropy=0.5,
        predictability=0.5,
        complexity=0.0,
        pattern_distribution={},
        dominant_pattern=None,
        n_patterns_observed=0,
        n_patterns_possible=0,
        confidence=0.0,
        method=method
    )


def _compute_ordpy(signal: np.ndarray, order: int, delay: int) -> PermutationEntropyResult:
    """Compute using ordpy library."""
    try:
        # Permutation entropy (normalized)
        pe = ordpy.permutation_entropy(signal, dx=order, taux=delay)
        
        # Complexity-entropy plane
        H, C = ordpy.complexity_entropy(signal, dx=order, taux=delay)
        
        # Get pattern distribution
        patterns, probs = ordpy.ordinal_distribution(signal, dx=order, taux=delay)
        
        pattern_dist = {tuple(p): float(prob) for p, prob in zip(patterns, probs)}
        
        # Find dominant pattern
        if pattern_dist:
            dominant = max(pattern_dist.keys(), key=lambda k: pattern_dist[k])
        else:
            dominant = None
        
        n_possible = factorial(order)
        n_observed = len([p for p in probs if p > 0])
        
        # Confidence based on number of patterns observed vs possible
        confidence = min(1.0, n_observed / (n_possible * 0.5))
        
        return PermutationEntropyResult(
            entropy=float(pe),
            predictability=float(1 - pe),
            complexity=float(C),
            pattern_distribution=pattern_dist,
            dominant_pattern=dominant,
            n_patterns_observed=n_observed,
            n_patterns_possible=n_possible,
            confidence=confidence,
            method="ordpy"
        )
        
    except Exception as e:
        return _empty_result(f"ordpy_error: {str(e)[:50]}")


def _compute_manual(signal: np.ndarray, order: int, delay: int) -> PermutationEntropyResult:
    """Manual implementation of permutation entropy."""
    n = len(signal)
    n_patterns = n - (order - 1) * delay
    
    if n_patterns < 10:
        return _empty_result("insufficient_for_patterns")
    
    # Extract ordinal patterns
    pattern_counts = {}
    
    for i in range(n_patterns):
        # Get the pattern of values
        indices = [i + j * delay for j in range(order)]
        values = signal[indices]
        
        # Convert to ordinal pattern (rank order)
        pattern = tuple(np.argsort(values))
        
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Convert to probabilities
    total = sum(pattern_counts.values())
    pattern_probs = {k: v / total for k, v in pattern_counts.items()}
    
    # Compute entropy
    probs = list(pattern_probs.values())
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    
    # Normalize by maximum entropy (log(d!))
    n_possible = factorial(order)
    max_entropy = np.log(n_possible)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Statistical complexity (Jensen-Shannon complexity)
    # C = H * D, where D is disequilibrium from uniform distribution
    uniform_prob = 1.0 / n_possible
    disequilibrium = sum((p - uniform_prob) ** 2 for p in probs)
    complexity = normalized_entropy * disequilibrium * n_possible
    complexity = min(1.0, complexity)  # Normalize roughly
    
    # Dominant pattern
    dominant = max(pattern_counts.keys(), key=lambda k: pattern_counts[k]) if pattern_counts else None
    
    n_observed = len(pattern_counts)
    confidence = min(1.0, n_observed / (n_possible * 0.5))
    
    return PermutationEntropyResult(
        entropy=float(normalized_entropy),
        predictability=float(1 - normalized_entropy),
        complexity=float(complexity),
        pattern_distribution=pattern_probs,
        dominant_pattern=dominant,
        n_patterns_observed=n_observed,
        n_patterns_possible=n_possible,
        confidence=confidence,
        method="manual"
    )


def compute_weighted(signal: np.ndarray,
                     order: int = 3,
                     delay: int = 1) -> PermutationEntropyResult:
    """
    Compute weighted permutation entropy.
    
    Weighted PE accounts for amplitude differences, not just order.
    More sensitive to signal dynamics than standard PE.
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    n_patterns = n - (order - 1) * delay
    
    if n_patterns < 10:
        return _empty_result("insufficient_for_weighted")
    
    # Extract patterns with weights
    pattern_weights = {}
    
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        values = signal[indices]
        
        pattern = tuple(np.argsort(values))
        
        # Weight = variance of the pattern values
        weight = np.var(values)
        
        if pattern not in pattern_weights:
            pattern_weights[pattern] = []
        pattern_weights[pattern].append(weight)
    
    # Weighted probabilities
    total_weight = sum(sum(w) for w in pattern_weights.values())
    if total_weight < 1e-10:
        return _empty_result("zero_weight")
    
    pattern_probs = {k: sum(v) / total_weight for k, v in pattern_weights.items()}
    
    # Compute weighted entropy
    probs = list(pattern_probs.values())
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    
    n_possible = factorial(order)
    max_entropy = np.log(n_possible)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    dominant = max(pattern_probs.keys(), key=lambda k: pattern_probs[k]) if pattern_probs else None
    
    return PermutationEntropyResult(
        entropy=float(normalized_entropy),
        predictability=float(1 - normalized_entropy),
        complexity=0.0,  # Not computed for weighted
        pattern_distribution=pattern_probs,
        dominant_pattern=dominant,
        n_patterns_observed=len(pattern_probs),
        n_patterns_possible=n_possible,
        confidence=min(1.0, len(pattern_probs) / (n_possible * 0.5)),
        method="weighted_manual"
    )


def compute_multiscale(signal: np.ndarray,
                       order: int = 3,
                       scales: List[int] = None) -> dict:
    """
    Compute permutation entropy at multiple time scales.
    
    Useful for detecting structure at different temporal resolutions.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    order : int
        Pattern length
    scales : List[int]
        Coarse-graining scales. Default [1, 2, 4, 8]
        
    Returns
    -------
    dict with entropy at each scale
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    results = {}
    
    for scale in scales:
        if scale == 1:
            coarsened = signal
        else:
            # Coarse-grain by averaging
            n_new = len(signal) // scale
            coarsened = np.mean(signal[:n_new * scale].reshape(-1, scale), axis=1)
        
        if len(coarsened) < 50:
            results[scale] = None
            continue
        
        result = compute(coarsened, order=order)
        results[scale] = {
            "entropy": result.entropy,
            "predictability": result.predictability,
            "complexity": result.complexity
        }
    
    return {
        "scales": scales,
        "results": results,
        "method": "multiscale_pe"
    }
