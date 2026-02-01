# PRISM Engine Inventory Report

## Executive Summary

| Category | Count | Status |
|----------|-------|--------|
| ORTHON Expected | 53 | Legacy spec |
| New Manifest (scale-invariant) | 16 | Active |
| Python Engine Files | 65 | Implemented |
| SQL Engine Files | 26 | Implemented |
| Missing from spec | 21 | Gap |

---

## Architecture Shift

```
OLD ARCHITECTURE:
  53 engines → 12 output parquets
  All engines run on all signals
  Scale-dependent metrics included

NEW ARCHITECTURE:
  Typology → Signal Vector → State Vector → Geometry
  Only scale-invariant engines
  Typology determines which engines run per signal
  Eigenvalues capture signal cloud SHAPE
```

---

## New Geometry Pipeline

| Stage | File | Description |
|-------|------|-------------|
| 1 | typology.parquet | Signal characterization (smoothness, periodicity, etc.) |
| 2 | signal_vector.parquet | Per-signal features (scale-invariant only) |
| 3 | state_vector.parquet | System state with eigenvalues, effective_dim |
| 4 | state_geometry.parquet | Eigenvalues per engine per index |
| 5 | signal_geometry.parquet | Per-signal distance to state centroid |
| 6 | signal_pairwise.parquet | Pairwise signal relationships (N²/2) |

---

## Engine Categories

### Core (Always Run)
- kurtosis
- skewness
- crest_factor

### Typology-Guided (Conditional)
| Signal Type | Engines |
|-------------|---------|
| SMOOTH | rolling_kurtosis, rolling_entropy, rolling_crest_factor, rolling_skewness |
| NOISY | kurtosis, entropy, crest_factor |
| IMPULSIVE | kurtosis, crest_factor, peak_ratio, rolling_kurtosis |
| PERIODIC | harmonics_ratio, band_ratios, spectral_centroid, spectral_entropy |
| APERIODIC | entropy, hurst |
| NON_STATIONARY | rolling_* engines only |
| TRENDING | hurst, rate_of_change_ratio |

### Deprecated (Scale-Dependent)
- rms, peak, mean, std
- rolling_rms, rolling_mean, rolling_std, rolling_range
- envelope, rolling_envelope
- total_power, harmonic_2x, harmonic_3x

---

## Gap Analysis

### Missing Engines (21)

| Category | Missing |
|----------|---------|
| Distribution | histogram, percentiles, iqr, mad, coefficient_of_variation |
| Spectral | fft, psd, spectral_spread, spectral_rolloff, spectral_flatness, spectral_peaks, bandwidth |
| Shape | shape_factor, impulse_factor, margin_factor |
| Dynamics | correlation_dimension, dfa |
| Relationships | cross_correlation, coherence, phase_coupling, dtw_distance |

### Assessment

| Gap | Priority | Notes |
|-----|----------|-------|
| Distribution metrics | Low | Scale-dependent, covered by kurtosis/skewness |
| Spectral peaks/rolloff | Low | Covered by spectral_centroid, spectral_entropy |
| DFA | Medium | Alternative to Hurst, may add |
| Phase coupling | Medium | Important for multi-signal systems |
| DTW distance | Low | Computationally expensive |
| correlation_dimension | Low | Covered by attractor_dimension |

---

## Output File Mapping

| New Pipeline | Old Spec | Notes |
|--------------|----------|-------|
| typology.parquet | (none) | NEW: signal characterization |
| signal_vector.parquet | primitives.parquet | Per-signal features |
| state_vector.parquet | (none) | NEW: eigenvalues, effective_dim |
| state_geometry.parquet | geometry.parquet | Shape metrics per engine |
| signal_geometry.parquet | primitives.parquet | Signal-to-state relationships |
| signal_pairwise.parquet | primitives_pairs.parquet | Pairwise metrics |
| (pending) | dynamics.parquet | Temporal evolution |
| (pending) | information_flow.parquet | Transfer entropy, Granger |
| (pending) | topology.parquet | Betti numbers, persistence |
| zscore.parquet | zscore.parquet | Unchanged |
| statistics.parquet | statistics.parquet | Unchanged |
| correlation.parquet | correlation.parquet | Unchanged |
| regime_assignment.parquet | regime_assignment.parquet | Unchanged |

---

## Next Steps

### High Priority
1. **Temporal signal_vector** - Add rolling features with I column for dynamics
2. **Dynamics integration** - Connect to existing dynamics_runner.py
3. **Topology integration** - Connect to existing topology_runner.py

### Medium Priority
4. **DFA engine** - Detrended Fluctuation Analysis
5. **Phase coupling** - For oscillatory systems
6. **Information flow** - Integrate transfer_entropy, granger

### Low Priority
7. **Distribution metrics** - If needed for specific domains
8. **DTW distance** - If time-warping alignment needed

---

## File Inventory

### Runners/Orchestration (4)
- typology_engine.py
- dynamics_runner.py
- information_flow_runner.py
- topology_runner.py

### New Geometry Pipeline (4)
- state_vector.py
- state_geometry.py
- signal_geometry.py
- signal_pairwise.py

### Signal Engines (26 core files)
kurtosis, skewness, crest_factor, entropy, hurst, spectral, harmonics, frequency_bands, peak, rms, rate_of_change, lyapunov, attractor, granger, cointegration, transfer_entropy, mutual_info, correlation, garch, dmd, envelope, basin, lof, cycle_counting, pulsation_index, time_constant

### Rolling Engines (16)
rolling_kurtosis, rolling_skewness, rolling_entropy, rolling_crest_factor, rolling_hurst, rolling_lyapunov, rolling_volatility, rolling_pulsation, rolling_mean, rolling_std, rolling_rms, rolling_range, rolling_envelope, manifold, derivatives, stability

### Advanced Engines (4)
causality_engine, topology_engine, emergence_engine, integration_engine

### Dynamics Engines (4)
lyapunov_engine, attractor_engine, recurrence_engine, bifurcation_engine

### Physics Engines (5)
energy_engine, mass_engine, momentum_engine, constitutive_engine, physics_stack

### Structure Engines (6)
covariance_engine, eigenvalue_engine, koopman_engine, spectral_engine, wavelet_engine, structure_runner

### Statistics Engines (4)
baseline_engine, anomaly_engine, fleet_engine, summary_engine

### SQL Engines (26)
zscore, statistics, correlation, regime_assignment, + 22 report files

---

## Conclusion

The new architecture shifts from "run all 53 engines" to "typology-guided scale-invariant engines". This reduces compute, eliminates scale-dependent features, and adds eigenvalue-based geometry capture.

**Key insight**: Eigenvalues encode the SHAPE of the signal cloud. Multi-mode detection and engine disagreement are now first-class outputs.

**Credit**: Avery Rudder - "Laplace transform IS the state engine"
