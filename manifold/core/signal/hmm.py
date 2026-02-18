"""
Hidden Markov Model Engine.

Delegates to pmtvs hmm_fit primitive.
"""

import numpy as np
from typing import Dict
from manifold.primitives.individual.discrete import hmm_fit


def compute(y: np.ndarray, max_states: int = 5) -> Dict[str, float]:
    """
    Fit Gaussian HMM and extract regime metrics.

    Args:
        y: 1D signal array
        max_states: Maximum number of hidden states to consider

    Returns:
        dict with hmm_* prefixed keys
    """
    r = hmm_fit(y, n_states=max_states)

    # Map unprefixed pmtvs keys to hmm_-prefixed keys expected by stages
    return {
        'hmm_n_states': r.get('n_states', np.nan),
        'hmm_current_state': r.get('current_state', np.nan),
        'hmm_current_state_prob': r.get('current_state_prob', np.nan),
        'hmm_state_entropy': r.get('state_entropy', np.nan),
        'hmm_transition_rate': r.get('transition_rate', np.nan),
        'hmm_bic': r.get('bic', np.nan),
        'hmm_log_likelihood': r.get('log_likelihood', np.nan),
    }
