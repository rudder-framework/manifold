"""
Discrete Entropy Engine.

Delegates to pmtvs discrete_entropy and transition_matrix primitives.
"""

import numpy as np
from typing import Dict, Any
from manifold.primitives.individual.discrete import (
    discrete_entropy,
    transition_matrix,
)


MIN_SAMPLES = 4


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute discrete entropy measures.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    n_bins : int
        Number of bins for continuous signals.

    Returns
    -------
    dict
        Entropy measures for discrete signals.
    """
    y = np.asarray(y).flatten()
    mask = ~np.isnan(y)
    y_clean = y[mask]

    if len(y_clean) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} non-NaN samples, got {len(y_clean)}")

    # Shannon entropy (scalar)
    shannon = float(discrete_entropy(y_clean))

    # Normalized entropy
    unique_vals = np.unique(y_clean)
    k = min(len(unique_vals), n_bins)
    max_entropy = np.log2(k) if k > 1 else 1.0
    normalized = float(shannon / max_entropy) if max_entropy > 0 else 0.0

    # Transition matrix for conditional entropy
    tm = transition_matrix(y_clean)
    conditional = tm.get('entropy', 0.0)

    # Entropy rate ≈ conditional entropy for stationary Markov chains
    entropy_rate = conditional

    # Excess entropy
    excess = shannon - conditional

    return {
        'shannon_entropy': shannon,
        'normalized_entropy': normalized,
        'conditional_entropy': float(conditional),
        'entropy_rate': float(entropy_rate),
        'excess_entropy': float(excess),
    }
