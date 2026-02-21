"""
GARCH Engine.

Delegates to pmtvs garch primitive.
"""

import numpy as np
from typing import Dict
# TODO: needs pmtvs export — garch


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Compute GARCH(1,1) parameters of signal.

    Args:
        y: Signal values

    Returns:
        dict with garch_omega, garch_alpha, garch_beta, garch_persistence
    """
    # garch not yet in pmtvs
    return {
        'garch_omega': np.nan,
        'garch_alpha': np.nan,
        'garch_beta': np.nan,
        'garch_persistence': np.nan,
    }
