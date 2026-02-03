"""
Entropy Engine.

Imports from primitives/individual/entropy.py (canonical).
"""

import numpy as np
from typing import Dict
from prism.primitives.individual.entropy import (
    sample_entropy,
    permutation_entropy,
    approximate_entropy,
)


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy measures of signal.

    Args:
        y: Signal values

    Returns:
        dict with sample_entropy, permutation_entropy, approximate_entropy
    """
    result = {
        'sample_entropy': np.nan,
        'permutation_entropy': np.nan,
        'approximate_entropy': np.nan
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 50:
        return result

    # Try antropy library first (most accurate)
    try:
        from antropy import sample_entropy as ant_sampen
        from antropy import perm_entropy, app_entropy

        result['sample_entropy'] = float(ant_sampen(y, order=2, metric='chebyshev'))
        result['permutation_entropy'] = float(perm_entropy(y, order=3, normalize=True))
        result['approximate_entropy'] = float(app_entropy(y, order=2, metric='chebyshev'))
        return result

    except ImportError:
        pass

    # Fall back to primitives
    result['sample_entropy'] = sample_entropy(y, m=2)
    result['permutation_entropy'] = permutation_entropy(y, order=3, normalize=True)
    result['approximate_entropy'] = approximate_entropy(y, m=2)

    return result
