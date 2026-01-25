"""
Archetype Matching
==================

Fingerprint computation and archetype matching algorithms.

The fingerprint is a 6D (or 7D) normalized vector representing
the signal's position in behavioral space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .library import ARCHETYPES, Archetype


# =============================================================================
# NORMALIZATION PARAMETERS
# =============================================================================

# Axis normalization ranges (raw value → 0-1)
NORMALIZATION = {
    'memory': {
        'metric': 'hurst_exponent',
        'min': 0.0,
        'max': 1.0,
        'description': 'Hurst exponent: 0=anti-persistent, 0.5=random, 1=persistent'
    },
    'information': {
        'metric': 'entropy_permutation',
        'min': 0.0,
        'max': 1.0,  # Already normalized permutation entropy
        'description': 'Permutation entropy: 0=deterministic, 1=maximum entropy'
    },
    'recurrence': {
        'metric': 'rqa_determinism',
        'min': 0.0,
        'max': 1.0,
        'description': 'RQA determinism: 0=stochastic, 1=deterministic'
    },
    'volatility': {
        'metric': 'garch_persistence',
        'min': 0.0,
        'max': 1.0,
        'description': 'GARCH persistence (α+β): 0=dissipating, 1=integrated'
    },
    'frequency': {
        'metric': 'spectral_centroid',
        'min': 0.0,
        'max': 0.5,  # Normalized frequency (Nyquist = 0.5)
        'description': 'Spectral centroid: 0=low freq, 0.5=Nyquist'
    },
    'dynamics': {
        'metric': 'lyapunov_exponent',
        'min': -0.5,
        'max': 0.5,
        'description': 'Lyapunov exponent: negative=stable, 0=edge, positive=chaotic'
    },
    'energy': {
        'metric': 'hamiltonian_trend',
        'min': -1.0,
        'max': 1.0,
        'description': 'Hamiltonian trend: negative=dissipative, 0=conservative, positive=driven'
    }
}


def normalize_value(value: float, axis: str) -> float:
    """
    Normalize a raw metric value to 0-1 range for fingerprint.

    Args:
        value: Raw metric value
        axis: Axis name (memory, information, etc.)

    Returns:
        Normalized value in [0, 1]
    """
    if axis not in NORMALIZATION:
        return float(np.clip(value, 0, 1))

    params = NORMALIZATION[axis]
    v_min, v_max = params['min'], params['max']

    # Handle special cases for dynamics (centered at 0)
    if axis == 'dynamics':
        # Map [-0.5, 0.5] → [0, 1]
        normalized = (value - v_min) / (v_max - v_min)
    elif axis == 'energy':
        # Map [-1, 1] → [0, 1]
        normalized = (value - v_min) / (v_max - v_min)
    elif axis == 'frequency':
        # Map [0, 0.5] → [0, 1]
        normalized = value / v_max
    else:
        # Standard 0-1 normalization
        normalized = (value - v_min) / (v_max - v_min) if v_max > v_min else 0.5

    return float(np.clip(normalized, 0, 1))


def compute_fingerprint(
    hurst: float = 0.5,
    entropy: float = 0.5,
    determinism: float = 0.5,
    persistence: float = 0.5,
    centroid: float = 0.25,
    lyapunov: float = 0.0,
    hamiltonian_trend: float = 0.0,
    include_energy: bool = True
) -> np.ndarray:
    """
    Compute normalized 6D or 7D fingerprint from raw metrics.

    Args:
        hurst: Hurst exponent (0-1)
        entropy: Permutation entropy (0-1)
        determinism: RQA determinism (0-1)
        persistence: GARCH persistence α+β (0-1)
        centroid: Spectral centroid (normalized frequency)
        lyapunov: Lyapunov exponent
        hamiltonian_trend: Hamiltonian trend (energy axis)
        include_energy: Whether to include 7th energy dimension

    Returns:
        Normalized fingerprint array
    """
    fingerprint = [
        normalize_value(hurst, 'memory'),
        normalize_value(entropy, 'information'),
        normalize_value(determinism, 'recurrence'),
        normalize_value(persistence, 'volatility'),
        normalize_value(centroid, 'frequency'),
        normalize_value(lyapunov, 'dynamics'),
    ]

    if include_energy:
        fingerprint.append(normalize_value(hamiltonian_trend, 'energy'))

    return np.array(fingerprint)


def compute_fingerprint_from_vector(vector: Dict) -> np.ndarray:
    """
    Compute fingerprint from a SignalVector dictionary.

    Args:
        vector: Dictionary with vector metrics

    Returns:
        7D normalized fingerprint
    """
    return compute_fingerprint(
        hurst=vector.get('hurst_exponent', 0.5),
        entropy=vector.get('entropy_permutation', 0.5),
        determinism=vector.get('rqa_determinism', 0.5),
        persistence=vector.get('garch_persistence', 0.5),
        centroid=vector.get('spectral_centroid', 0.25),
        lyapunov=vector.get('lyapunov_exponent', 0.0),
        hamiltonian_trend=vector.get('hamiltonian_trend', 0.0),
        include_energy=True
    )


def match_archetype(
    fingerprint: np.ndarray,
    candidates: Optional[List[str]] = None
) -> Tuple[str, float, str, float]:
    """
    Match fingerprint to closest archetype.

    Args:
        fingerprint: 6D or 7D normalized fingerprint
        candidates: Optional list of archetype names to consider

    Returns:
        Tuple of (primary_name, primary_score, secondary_name, secondary_score)
    """
    if candidates is None:
        search_archetypes = ARCHETYPES
    else:
        search_archetypes = {k: v for k, v in ARCHETYPES.items() if k in candidates}

    scores = []
    for name, archetype in search_archetypes.items():
        score = archetype.match_score(fingerprint)
        distance = archetype.distance(fingerprint)
        scores.append((name, score, distance))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    if len(scores) >= 2:
        return (scores[0][0], scores[0][1], scores[1][0], scores[1][1])
    elif len(scores) == 1:
        return (scores[0][0], scores[0][1], "Unknown", 0.0)
    else:
        return ("Unknown", 0.0, "Unknown", 0.0)


def compute_boundary_proximity(
    fingerprint: np.ndarray,
    archetype_name: str
) -> float:
    """
    Compute proximity to archetype boundary.

    Returns value between 0 and 1:
        - 0: At boundary (could easily transition)
        - 1: Far from boundary (solidly in archetype)

    Args:
        fingerprint: Normalized fingerprint
        archetype_name: Name of the archetype

    Returns:
        Boundary proximity score
    """
    archetype = ARCHETYPES.get(archetype_name)
    if archetype is None:
        return 0.5

    # Compute distance to each axis boundary
    axes = [
        archetype.memory,
        archetype.information,
        archetype.recurrence,
        archetype.volatility,
        archetype.frequency,
        archetype.dynamics
    ]

    if len(fingerprint) >= 7:
        axes.append(archetype.energy)

    min_margin = float('inf')

    for i, ax in enumerate(axes[:len(fingerprint)]):
        value = fingerprint[i]
        if ax.low <= value <= ax.high:
            # Distance to nearest boundary
            margin_low = value - ax.low
            margin_high = ax.high - value
            margin = min(margin_low, margin_high)
            min_margin = min(min_margin, margin)
        else:
            # Outside range - at or past boundary
            return 0.0

    # Normalize: typical range width is 0.3-0.5
    # So margin of 0.15 is "deep inside"
    max_expected_margin = 0.15
    proximity = min(min_margin / max_expected_margin, 1.0)

    return float(proximity)


def get_archetype_for_fingerprint(fingerprint: np.ndarray) -> Dict:
    """
    Get full archetype matching result.

    Args:
        fingerprint: Normalized fingerprint

    Returns:
        Dictionary with:
            - archetype: Primary archetype name
            - archetype_distance: Distance to primary
            - secondary_archetype: Second-best match
            - secondary_distance: Distance to secondary
            - boundary_proximity: How close to boundary
            - confidence: Overall matching confidence
    """
    primary, primary_score, secondary, secondary_score = match_archetype(fingerprint)

    # Confidence based on separation between primary and secondary
    if primary_score > 0:
        separation = (primary_score - secondary_score) / primary_score
    else:
        separation = 0.0

    boundary = compute_boundary_proximity(fingerprint, primary)

    # Overall confidence combines match quality, separation, and boundary distance
    confidence = primary_score * (0.5 + 0.5 * separation) * (0.5 + 0.5 * boundary)

    archetype_obj = ARCHETYPES.get(primary)
    primary_distance = archetype_obj.distance(fingerprint) if archetype_obj else 1.0

    secondary_obj = ARCHETYPES.get(secondary)
    secondary_distance = secondary_obj.distance(fingerprint) if secondary_obj else 1.0

    return {
        'archetype': primary,
        'archetype_distance': primary_distance,
        'secondary_archetype': secondary,
        'secondary_distance': secondary_distance,
        'boundary_proximity': boundary,
        'confidence': confidence
    }


def fingerprint_to_string(fingerprint: np.ndarray) -> str:
    """
    Convert fingerprint to human-readable string.

    Args:
        fingerprint: Normalized fingerprint

    Returns:
        Formatted string representation
    """
    axis_names = ['M', 'I', 'R', 'V', 'F', 'D', 'E']

    parts = []
    for i, val in enumerate(fingerprint[:len(axis_names)]):
        parts.append(f"{axis_names[i]}:{val:.2f}")

    return " ".join(parts)
