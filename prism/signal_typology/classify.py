"""
Signal Typology Classification
==============================

Runtime classification of normalized 0-1 scores to human-readable labels.

Principle: Data = math. Labels = rendering.
- Scores stored in parquet (0-1 range)
- Classification computed at query/display time
- Threshold changes don't require recomputation
- Signal evolution reflected immediately

Each axis maps to 5 levels:
    [0.00, 0.25) → strong low    (e.g., "forgetful")
    [0.25, 0.40) → weak low      (e.g., "weak forgetful")
    [0.40, 0.60) → indeterminate
    [0.60, 0.75) → weak high     (e.g., "weak persistent")
    [0.75, 1.00] → strong high   (e.g., "persistent")
"""

from typing import Dict, List, Optional, Tuple


# Labels for each axis: [strong_low, weak_low, indeterminate, weak_high, strong_high]
AXIS_LABELS = {
    'memory': ['forgetful', 'weak forgetful', 'indeterminate', 'weak persistent', 'persistent'],
    'information': ['predictable', 'weak predictable', 'indeterminate', 'weak entropic', 'entropic'],
    'frequency': ['aperiodic', 'weak aperiodic', 'indeterminate', 'weak periodic', 'periodic'],
    'volatility': ['stable', 'weak stable', 'indeterminate', 'weak clustered', 'clustered'],
    'wavelet': ['single-scale', 'weak single-scale', 'indeterminate', 'weak multi-scale', 'multi-scale'],
    'derivatives': ['smooth', 'weak smooth', 'indeterminate', 'weak spiky', 'spiky'],
    'recurrence': ['wandering', 'weak wandering', 'indeterminate', 'weak returning', 'returning'],
    'discontinuity': ['continuous', 'weak continuous', 'indeterminate', 'weak step-like', 'step-like'],
    'momentum': ['reverting', 'weak reverting', 'indeterminate', 'weak trending', 'trending'],
}

# Axis descriptions for UI
AXIS_DESCRIPTIONS = {
    'memory': 'Long-term autocorrelation structure',
    'information': 'Entropy and predictability',
    'frequency': 'Periodic vs aperiodic behavior',
    'volatility': 'Variance clustering (GARCH effects)',
    'wavelet': 'Multi-scale structure',
    'derivatives': 'Smoothness vs spikiness',
    'recurrence': 'Phase space recurrence patterns',
    'discontinuity': 'Step changes and level shifts',
    'momentum': 'Trending vs mean-reverting behavior',
}

# Semantic poles for each axis
AXIS_POLES = {
    'memory': ('Forgetful', 'Persistent'),
    'information': ('Predictable', 'Entropic'),
    'frequency': ('Aperiodic', 'Periodic'),
    'volatility': ('Stable', 'Clustered'),
    'wavelet': ('Single-scale', 'Multi-scale'),
    'derivatives': ('Smooth', 'Spiky'),
    'recurrence': ('Wandering', 'Returning'),
    'discontinuity': ('Continuous', 'Step-like'),
    'momentum': ('Reverting', 'Trending'),
}

# Default thresholds: [t1, t2, t3, t4]
DEFAULT_THRESHOLDS = [0.25, 0.40, 0.60, 0.75]

# Alternative threshold presets
THRESHOLD_PRESETS = {
    'default': [0.25, 0.40, 0.60, 0.75],
    'strict': [0.20, 0.35, 0.65, 0.80],   # Wider indeterminate zone
    'loose': [0.30, 0.45, 0.55, 0.70],    # Narrower indeterminate zone
    'binary': [0.50, 0.50, 0.50, 0.50],   # Simple high/low split
}

# Per-axis threshold overrides (optional)
AXIS_THRESHOLDS = {
    'memory': [0.30, 0.45, 0.55, 0.70],      # Hurst has known distribution around 0.5
    'volatility': DEFAULT_THRESHOLDS,
    'frequency': DEFAULT_THRESHOLDS,
    'information': DEFAULT_THRESHOLDS,
    'wavelet': DEFAULT_THRESHOLDS,
    'derivatives': [0.20, 0.35, 0.65, 0.80], # Kurtosis can be extreme
    'recurrence': DEFAULT_THRESHOLDS,
    'discontinuity': [0.15, 0.30, 0.50, 0.70], # Level shifts are rare
    'momentum': [0.30, 0.45, 0.55, 0.70],    # Same as memory
}


def classify(
    score: float,
    axis: str,
    thresholds: Optional[List[float]] = None,
    use_axis_thresholds: bool = True,
) -> str:
    """
    Map 0-1 score to classification label.

    Args:
        score: Normalized 0-1 score
        axis: One of the 9 axis names
        thresholds: Optional custom thresholds [t1, t2, t3, t4]
        use_axis_thresholds: If True and no thresholds given, use axis-specific defaults

    Returns:
        Classification label string
    """
    if axis not in AXIS_LABELS:
        return 'unknown'

    labels = AXIS_LABELS[axis]

    # Determine thresholds
    if thresholds is None:
        if use_axis_thresholds and axis in AXIS_THRESHOLDS:
            thresholds = AXIS_THRESHOLDS[axis]
        else:
            thresholds = DEFAULT_THRESHOLDS

    # Classify
    if score < thresholds[0]:
        return labels[0]
    elif score < thresholds[1]:
        return labels[1]
    elif score < thresholds[2]:
        return labels[2]
    elif score < thresholds[3]:
        return labels[3]
    else:
        return labels[4]


def classify_profile(
    profile: Dict[str, float],
    thresholds: Optional[List[float]] = None,
    use_axis_thresholds: bool = True,
) -> Dict[str, str]:
    """
    Classify all axes in a profile.

    Args:
        profile: Dict with axis scores (0-1)
        thresholds: Optional custom thresholds for all axes
        use_axis_thresholds: If True, use axis-specific thresholds

    Returns:
        Dict with classification labels: {'{axis}_class': label}
    """
    result = {}

    for axis in AXIS_LABELS.keys():
        if axis in profile:
            label = classify(
                profile[axis],
                axis,
                thresholds,
                use_axis_thresholds
            )
            result[f'{axis}_class'] = label

    return result


def get_classification_level(score: float, thresholds: Optional[List[float]] = None) -> int:
    """
    Get numeric classification level (0-4).

    Args:
        score: Normalized 0-1 score
        thresholds: Optional custom thresholds

    Returns:
        Integer level 0-4
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    if score < thresholds[0]:
        return 0
    elif score < thresholds[1]:
        return 1
    elif score < thresholds[2]:
        return 2
    elif score < thresholds[3]:
        return 3
    else:
        return 4


def get_pole_label(score: float, axis: str) -> str:
    """
    Get simplified two-pole label.

    Args:
        score: Normalized 0-1 score
        axis: Axis name

    Returns:
        Either low pole or high pole label
    """
    if axis not in AXIS_POLES:
        return 'unknown'

    low, high = AXIS_POLES[axis]
    return high if score >= 0.5 else low


def summarize_profile(profile: Dict[str, float]) -> str:
    """
    Generate a one-line summary of the profile.

    Args:
        profile: Dict with axis scores

    Returns:
        Summary string like "persistent, entropic, periodic"
    """
    # Get strong classifications only (levels 0 or 4)
    strong = []
    for axis in AXIS_LABELS.keys():
        if axis in profile:
            level = get_classification_level(profile[axis])
            if level == 0:
                strong.append(AXIS_LABELS[axis][0])
            elif level == 4:
                strong.append(AXIS_LABELS[axis][4])

    if strong:
        return ', '.join(strong)
    else:
        return 'indeterminate across all axes'


def get_dominant_characteristics(
    profile: Dict[str, float],
    n: int = 3
) -> List[Tuple[str, str, float]]:
    """
    Get the N most extreme (non-indeterminate) characteristics.

    Args:
        profile: Dict with axis scores
        n: Number of characteristics to return

    Returns:
        List of (axis, label, score) tuples, sorted by extremity
    """
    characteristics = []

    for axis in AXIS_LABELS.keys():
        if axis in profile:
            score = profile[axis]
            # Extremity = distance from 0.5
            extremity = abs(score - 0.5)
            label = classify(score, axis)
            characteristics.append((axis, label, score, extremity))

    # Sort by extremity (most extreme first)
    characteristics.sort(key=lambda x: x[3], reverse=True)

    # Return top N without extremity value
    return [(axis, label, score) for axis, label, score, _ in characteristics[:n]]


def format_profile_table(profile: Dict[str, float]) -> str:
    """
    Format profile as ASCII table for display.

    Args:
        profile: Dict with axis scores

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"{'Axis':<15} {'Score':>6} {'Classification':<20}")
    lines.append("-" * 45)

    for axis in AXIS_LABELS.keys():
        if axis in profile:
            score = profile[axis]
            label = classify(score, axis)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"{axis:<15} {score:>6.3f} [{bar}] {label}")

    return "\n".join(lines)
