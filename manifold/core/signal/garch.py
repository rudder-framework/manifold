"""
GARCH Engine.

Delegates to pmtvs garch primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.volatility import garch


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute GARCH(1,1) parameters of signal.

    Args:
        y: Signal values

    Returns:
        dict with garch_omega, garch_alpha, garch_beta, garch_persistence
    """
    return garch(y)
