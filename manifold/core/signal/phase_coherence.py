"""
Phase Coherence Engine.

Delegates to pmtvs phase_coherence primitive.
"""

import numpy as np
from typing import Dict
from manifold.core._pmtvs import hilbert_transform
# TODO: needs pmtvs export — phase_coherence


def compute(y: np.ndarray, n_segments: int = 4) -> Dict[str, float]:
    """
    Measure phase coherence.

    Args:
        y: Signal values
        n_segments: Number of segments to compare

    Returns:
        dict with phase_coherence, coherence_std, coherence_trend
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < n_segments * 4:
        return {
            'phase_coherence': np.nan,
            'coherence_std': np.nan,
            'coherence_trend': np.nan,
        }

    try:
        analytic = hilbert_transform(y)
        phase = np.angle(analytic)

        seg_len = n // n_segments
        coherences = []
        for i in range(n_segments - 1):
            p1 = phase[i * seg_len:(i + 1) * seg_len]
            p2 = phase[(i + 1) * seg_len:(i + 2) * seg_len]
            min_len = min(len(p1), len(p2))
            diff = p1[:min_len] - p2[:min_len]
            coh = float(np.abs(np.mean(np.exp(1j * diff))))
            coherences.append(coh)

        coherences = np.array(coherences)
        return {
            'phase_coherence': float(np.mean(coherences)),
            'coherence_std': float(np.std(coherences)),
            'coherence_trend': float(coherences[-1] - coherences[0]) if len(coherences) > 1 else np.nan,
        }
    except Exception:
        return {
            'phase_coherence': np.nan,
            'coherence_std': np.nan,
            'coherence_trend': np.nan,
        }
