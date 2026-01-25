"""
Color Encoding Strategies
=========================

Three strategies for mapping typology vectors to colors:
1. PCA → RGB: Similar signals get similar colors
2. Dominant Trait: Categorical colors by strongest axis
3. Direct Mapping: 3 key axes map directly to RGB
"""

from typing import Dict
import numpy as np
import pandas as pd


AXES = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
        'derivatives', 'recurrence', 'discontinuity', 'momentum']


# Categorical colors for each trait
TRAIT_COLORS = {
    'memory': '#4C78A8',       # Blue
    'information': '#F58518',  # Orange
    'frequency': '#E45756',    # Red
    'volatility': '#72B7B2',   # Teal
    'wavelet': '#54A24B',      # Green
    'derivatives': '#EECA3B',  # Yellow
    'recurrence': '#B279A2',   # Purple
    'discontinuity': '#FF9DA6',  # Pink
    'momentum': '#9D755D',     # Brown
}


def typology_to_color_pca(profiles: pd.DataFrame) -> Dict[str, str]:
    """
    Map typology profiles to colors via PCA.
    Similar signals → similar colors.

    Args:
        profiles: DataFrame with signal_id and axis columns

    Returns:
        Dict mapping signal_id to hex color
    """
    from sklearn.decomposition import PCA

    X = profiles[AXES].values

    if len(X) < 2:
        # Not enough data for PCA
        return {profiles['signal_id'].iloc[0]: '#808080'}

    # Reduce 9D → 3D
    n_components = min(3, len(X), len(AXES))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    # Pad to 3D if needed
    if components.shape[1] < 3:
        padding = np.zeros((len(X), 3 - components.shape[1]))
        components = np.hstack([components, padding])

    # Normalize to 0-255
    mins = components.min(axis=0)
    maxs = components.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (components - mins) / ranges
    rgb = (normalized * 255).astype(int)

    colors = {}
    for i, signal_id in enumerate(profiles['signal_id']):
        r, g, b = rgb[i]
        colors[signal_id] = f'#{r:02x}{g:02x}{b:02x}'

    return colors


def typology_to_color_dominant(profile: Dict) -> str:
    """
    Color based on dominant trait.

    Args:
        profile: Dict with axis scores

    Returns:
        Hex color string
    """
    scores = [profile.get(a, 0) for a in AXES]
    dominant = AXES[int(np.argmax(scores))]
    return TRAIT_COLORS[dominant]


def typology_to_color_direct(profile: Dict) -> str:
    """
    Direct mapping of 3 interpretable axes to RGB.

    R = volatility (red = unstable)
    G = frequency (green = periodic)
    B = memory (blue = persistent)

    Args:
        profile: Dict with axis scores

    Returns:
        Hex color string
    """
    r = int(255 * profile.get('volatility', 0.5))
    g = int(255 * profile.get('frequency', 0.5))
    b = int(255 * profile.get('memory', 0.5))

    # Ensure valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f'#{r:02x}{g:02x}{b:02x}'


def get_color_for_score(score: float, colormap: str = 'viridis') -> str:
    """
    Get color for a 0-1 score using matplotlib colormap.

    Args:
        score: Value 0-1
        colormap: Matplotlib colormap name

    Returns:
        Hex color string
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        rgba = cmap(score)
        r, g, b = [int(x * 255) for x in rgba[:3]]
        return f'#{r:02x}{g:02x}{b:02x}'
    except ImportError:
        # Fallback: grayscale
        v = int(score * 255)
        return f'#{v:02x}{v:02x}{v:02x}'
