"""
Hidden Markov Model Engine.

Delegates to pmtvs hmm_fit primitive.
"""

import numpy as np
from typing import Dict
# TODO: needs pmtvs export — hmm_fit


def compute(y: np.ndarray, max_states: int = 5) -> Dict[str, float]:
    """
    Fit Gaussian HMM and extract regime metrics.

    Args:
        y: 1D signal array
        max_states: Maximum number of hidden states to consider

    Returns:
        dict with hmm_* prefixed keys
    """
    # hmm_fit not yet in pmtvs
    return {
        'hmm_n_states': np.nan,
        'hmm_current_state': np.nan,
        'hmm_current_state_prob': np.nan,
        'hmm_state_entropy': np.nan,
        'hmm_transition_rate': np.nan,
        'hmm_bic': np.nan,
        'hmm_log_likelihood': np.nan,
    }
