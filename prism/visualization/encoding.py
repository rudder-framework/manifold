"""
Visual Encoding Dataclasses
===========================

Core data structures for signal visual encoding.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np


AXES = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
        'derivatives', 'recurrence', 'discontinuity', 'momentum']

SHAPES = ['circle', 'square', 'triangle', 'diamond', 'star',
          'hexagon', 'pentagon', 'cross', 'arrow']


@dataclass
class SignalVisualEncoding:
    """Visual encoding for a single signal."""

    signal_id: str

    # Color signature (from typology)
    color_rgb: Tuple[int, int, int] = (128, 128, 128)
    color_hex: str = "#808080"

    # 2D projection coordinates (for scatter/map views)
    proj_x: float = 0.0
    proj_y: float = 0.0

    # Glyph parameters
    glyph_shape: str = "circle"
    glyph_size: float = 0.5
    glyph_rotation: float = 0.0

    # Radar chart data
    radar_values: List[float] = field(default_factory=list)
    radar_labels: List[str] = field(default_factory=lambda: AXES.copy())

    # Classification summary
    dominant_trait: str = "indeterminate"
    trait_strength: float = 0.5


@dataclass
class SystemVisualEncoding:
    """Visual encoding for system-level (multi-signal) results."""

    # Node positions (for graph layouts)
    node_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Edge data (for relationship graphs)
    edges: List[Dict] = field(default_factory=list)

    # Heatmap data
    matrix: Optional[np.ndarray] = None
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)

    # Cluster assignments
    clusters: Dict[str, int] = field(default_factory=dict)
    cluster_colors: Dict[int, str] = field(default_factory=dict)


def encode_signal(profile: Dict, signal_id: str = "unknown") -> SignalVisualEncoding:
    """
    Create visual encoding from a typology profile.

    Args:
        profile: Dict with axis scores (0-1)
        signal_id: Signal identifier

    Returns:
        SignalVisualEncoding with computed visual properties
    """
    from .colors import typology_to_color_direct

    # Get axis scores
    scores = [profile.get(ax, 0.5) for ax in AXES]

    # Color from direct mapping
    color_hex = typology_to_color_direct(profile)
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)

    # Glyph from typology
    glyph = typology_to_glyph(profile)

    # Dominant trait
    dominant_idx = int(np.argmax(scores))
    dominant_trait = AXES[dominant_idx]
    trait_strength = scores[dominant_idx]

    return SignalVisualEncoding(
        signal_id=signal_id,
        color_rgb=(r, g, b),
        color_hex=color_hex,
        glyph_shape=glyph['shape'],
        glyph_size=glyph['size'],
        glyph_rotation=glyph['rotation'],
        radar_values=scores,
        radar_labels=AXES.copy(),
        dominant_trait=dominant_trait,
        trait_strength=trait_strength,
    )


def typology_to_glyph(profile: Dict) -> Dict:
    """
    Encode typology as glyph parameters.

    Shape: dominant trait category
    Size: overall signal "strength" (mean absolute deviation from 0.5)
    Rotation: momentum direction
    """
    scores = [profile.get(a, 0.5) for a in AXES]

    # Shape from dominant trait
    dominant_idx = int(np.argmax(scores))
    shape = SHAPES[dominant_idx % len(SHAPES)]

    # Size from distinctiveness (how far from neutral)
    distinctiveness = np.mean([abs(s - 0.5) for s in scores])
    size = 0.3 + (distinctiveness * 1.4)  # Range 0.3-1.0

    # Rotation from momentum
    momentum = profile.get('momentum', 0.5)
    rotation = (momentum - 0.5) * 180  # -90 to +90 degrees

    return {
        'shape': shape,
        'size': float(min(size, 1.0)),
        'rotation': float(rotation),
    }
