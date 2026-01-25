"""
PRISM Visual Encoding Layer
===========================

Transforms framework outputs into visual representations.

Framework → Vector → Visual Encoding → Render

Modules:
    encoding.py    - Visual encoding dataclasses
    colors.py      - Color mapping strategies
    projection.py  - 2D projection (UMAP, t-SNE)
    aggregation.py - Cohort summaries and clustering
"""

from .encoding import SignalVisualEncoding, SystemVisualEncoding, encode_signal
from .colors import (
    typology_to_color_pca,
    typology_to_color_dominant,
    typology_to_color_direct,
    TRAIT_COLORS,
)
from .projection import project_typology_umap, project_typology_tsne
from .aggregation import aggregate_cohort_typology, cluster_signals

__all__ = [
    # Encoding
    "SignalVisualEncoding",
    "SystemVisualEncoding",
    "encode_signal",
    # Colors
    "typology_to_color_pca",
    "typology_to_color_dominant",
    "typology_to_color_direct",
    "TRAIT_COLORS",
    # Projection
    "project_typology_umap",
    "project_typology_tsne",
    # Aggregation
    "aggregate_cohort_typology",
    "cluster_signals",
]
