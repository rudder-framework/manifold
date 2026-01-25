# PR: Manifold Geometry Refactor + Curvature Engines

## Summary

This PR introduces three major changes:
1. **Rename** `structural_geometry` → `manifold_geometry`
2. **Add** two discrete Ricci curvature engines (Ollivier-Ricci, Forman-Ricci)
3. **Implement** the 6-metric geometry state architecture

## Background

Academic research (Sandhu/Tannenbaum 2016, Samal et al. 2018) demonstrates that discrete Ricci curvature serves as a "crash hallmark" for system fragility—curvature changes predict regime shifts before they manifest in individual signals.

ORTHON's geometry layer now computes manifold state at each timestamp, producing interpretable metrics that feed into dynamics and mechanics layers.

---

## Files Changed

### Renamed
- `prism/entry_points/structural_geometry.py` → `prism/entry_points/manifold_geometry.py`
- `prism/structural_geometry/` → `prism/manifold_geometry/`

### Added
- `prism/manifold_geometry/ollivier_ricci.py` — Gold standard curvature (expensive)
- `prism/manifold_geometry/forman_ricci.py` — Fast approximation curvature
- `prism/manifold_geometry/geodesic_curvature.py` — Manifold embedding curvature
- `prism/manifold_geometry/geometry_state.py` — 6-metric state computation

### Modified
- `prism/manifold_geometry/__init__.py` — Export new engines
- `data/` schema — `structural_geometry.parquet` → `manifold_geometry.parquet`

---

## Architecture

### Signal State Vector (per signal, per timestamp)

```python
geometry_state = {
    "timestamp": t,
    
    # 1. DIMENSION — "How complex?"
    "dim_linear": int,           # PCA effective dimension
    "dim_intrinsic": float,      # Nonlinear (Isomap/UMAP)
    "dim_fractal": float,        # Correlation dimension
    
    # 2. TOPOLOGY — "What shape class?"
    "betti_0": int,              # Connected components
    "betti_1": int,              # Holes/loops
    "topology_class": str,       # "linear" | "bounded" | "periodic" | "attractor"
    
    # 3. CURVATURE — "How bent?"
    "curvature_ollivier": float, # Ollivier-Ricci (network)
    "curvature_forman": float,   # Forman-Ricci (fast)
    "curvature_geodesic": float, # Manifold embedding
    "curvature_sign": str,       # "positive" | "negative" | "mixed" | "flat"
    
    # 4. COHERENCE — "How tight?"
    "coherence_tightness": float,    # Silhouette score
    "coherence_uniformity": float,   # Distribution evenness
    "reconstruction_error": float,   # Manifold fit quality
    
    # 5. STABILITY — "How robust?"
    "stability_persistence": float,  # Topological persistence
    "stability_curvature_var": float,# Curvature variance over time
    
    # 6. ORIENTATION — "Is there a direction?"
    "anisotropy": float,         # Eigenvalue ratio (0=isotropic, 1=directional)
}
```

---

## Curvature Engines

### Ollivier-Ricci Curvature

**Purpose:** Gold standard for system fragility detection

**Math:** Measures optimal transport cost between node neighborhoods
```
κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
```
Where W₁ is Wasserstein-1 distance between probability distributions on neighborhoods.

**Interpretation:**
- κ > 0: Locally "spherical" — neighbors are closer than expected
- κ < 0: Locally "hyperbolic" — neighbors are farther than expected  
- κ → 0: Locally "flat" — grid-like structure

**Use case:** When precision matters. Curvature drop = fragility increase.

### Forman-Ricci Curvature

**Purpose:** Fast approximation for large networks

**Math:** Combinatorial formula based on node degrees
```
F(e) = w(e) * (w(v₁)/√(Σw(e₁)) + w(v₂)/√(Σw(e₂)) - Σ(parallel edges) - Σ(triangles))
```

**Interpretation:** Highly correlated with Ollivier-Ricci in most networks, but O(E) vs O(E × V²).

**Use case:** Real-time monitoring, large signal sets, initial screening.

### Geodesic Curvature

**Purpose:** Measures manifold embedding quality

**Math:** Ratio of geodesic to Euclidean distance
```
κ_geo = mean(d_geodesic(i,j) / d_euclidean(i,j)) - 1
```

**Interpretation:**
- κ_geo ≈ 0: Flat manifold (PCA sufficient)
- κ_geo > 0: Curved manifold (nonlinear methods needed)

---

## Integration with Existing Code

### Decoupling Detection (`bg_decoupling.py`)

Curvature provides EARLY WARNING before decoupling occurs:

```python
# Current: Detect decoupling after it happens
result = bg_decoupling.compute(x, y)

# New: Predict decoupling via curvature
curvature_history = [compute_ollivier_ricci(graph_t) for t in windows]
if curvature_dropping(curvature_history):
    alert("Fragility increasing — decoupling likely")
```

### Entry Point Changes

```python
# OLD
from prism.entry_points.structural_geometry import main

# NEW  
from prism.entry_points.manifold_geometry import main
```

Output parquet columns added:
- `curvature_ollivier`
- `curvature_forman`
- `curvature_geodesic`
- `curvature_sign`
- `dim_fractal`
- `anisotropy`

---

## Testing

```bash
# Run on FD_002
python -m prism.entry_points.manifold_geometry --data-dir data/fd002

# Compare curvatures
python -m prism.manifold_geometry.validate --compare-curvatures

# Benchmark speed
python -m prism.manifold_geometry.benchmark --n-signals 100
```

Expected results:
- Forman-Ricci: 10-100x faster than Ollivier-Ricci
- Correlation between curvatures: r > 0.7 for most networks
- Curvature drops precede decoupling events by 5-20 windows

---

## Migration

1. Update imports: `structural_geometry` → `manifold_geometry`
2. Rename output files: `structural_geometry.parquet` → `manifold_geometry.parquet`
3. Add new columns to downstream consumers
4. Update Streamlit dashboards

---

## References

1. Sandhu, Georgiou, Tannenbaum (2016). "Ricci curvature: An economic indicator for market fragility and systemic risk." Science Advances.
2. Samal et al. (2018). "Comparative analysis of two discretizations of Ricci curvature for complex networks." Scientific Reports.
3. Gosztolai & Arnaudon (2021). "Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature." Nature Communications.

---

## Acceptance Criteria

- [ ] `structural_geometry` renamed to `manifold_geometry` throughout codebase
- [ ] Ollivier-Ricci curvature engine implemented and tested
- [ ] Forman-Ricci curvature engine implemented and tested
- [ ] Geodesic curvature ratio implemented
- [ ] 6-metric geometry state output from entry point
- [ ] Curvature correlation validated (Forman vs Ollivier r > 0.7)
- [ ] Performance benchmark: Forman 10x+ faster than Ollivier
- [ ] Integration with decoupling detection documented
- [ ] Downstream parquet schema updated
