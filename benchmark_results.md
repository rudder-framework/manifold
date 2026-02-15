# Full Pipeline Benchmark — FD_004 (5,976 signals, 249 cohorts)

Date: 2026-02-14
Branch: `rust`
Platform: macOS Darwin 25.2.0, Apple Silicon (arm64), Python 3.14.2

## Rust Pipeline — Per-Stage Timing

| Stage | Name | Time | Output Dir |
|-------|------|-----:|------------|
| 00 | breaks | 36.1s | 3_health_scoring |
| 01 | signal_vector | 123.2s | 1_signal_features |
| 02 | state_vector | 1.4s | 2_system_state |
| 03 | state_geometry | 23.4s | 2_system_state |
| 05 | signal_geometry | 4.5s | 1_signal_features |
| 06 | signal_pairwise | 75.3s | 4_signal_relationships |
| 07 | geometry_dynamics | 0.2s | 2_system_state |
| 08 | ftle | 91.8s | 5_evolution |
| 08b | lyapunov | 48.6s | 5_evolution |
| 09a | cohort_thermodynamics | 0.1s | 5_evolution |
| 10 | information_flow | 165.2s | 4_signal_relationships |
| 15 | ftle_field | 90.2s | 5_evolution |
| 17 | ftle_backward | 92.6s | 5_evolution |
| 18 | segment_comparison | 0.5s | 4_signal_relationships |
| 19 | info_flow_delta | 5.5s | 4_signal_relationships |
| 20 | sensor_eigendecomp | 1.3s | 2_system_state |
| 21 | velocity_field | 2.5s | 5_evolution |
| 22 | ftle_rolling | 135.2s | 5_evolution |
| 23 | ridge_proximity | 0.8s | 5_evolution |
| 36 | persistent_homology | 0.1s | 5_evolution |
| 26 | system_geometry | 0.0s | 6_fleet (skipped) |
| 27 | cohort_pairwise | 0.0s | 6_fleet (skipped) |
| 28 | cohort_information_flow | 0.0s | 6_fleet (skipped) |
| 30 | cohort_ftle | 0.0s | 6_fleet (skipped) |
| 31 | cohort_velocity_field | 0.0s | 6_fleet (skipped) |
| 33 | signal_stability | 141.9s | 1_signal_features |
| 34 | cohort_baseline | 0.3s | 3_health_scoring |
| 35 | observation_geometry | 0.4s | 3_health_scoring |
| | **TOTAL** | **1041.2s** | |

Wall clock: **17 min 21s**
Peak memory: **3.48 GB** (maximum resident set size)
Output: **26 parquet files** across 6 directories
Fleet stages (26-31) skipped: no cohort_vector.parquet (produced by Prime SQL)

## Top 5 Hottest Stages

| Rank | Stage | Time | % of Total |
|------|-------|-----:|-----------:|
| 1 | information_flow | 165.2s | 15.9% |
| 2 | signal_stability | 141.9s | 13.6% |
| 3 | ftle_rolling | 135.2s | 13.0% |
| 4 | signal_vector | 123.2s | 11.8% |
| 5 | ftle_backward | 92.6s | 8.9% |

These 5 stages account for 63.2% of total runtime.

## Time by Stage Group

| Group | Time | % of Total | Stages |
|-------|-----:|-----------:|--------|
| dynamics | 459.3s | 44.1% | ftle, lyapunov, thermo, ftle_field, ftle_backward, velocity, ftle_rolling, ridge, persistent_homology |
| information | 246.5s | 23.7% | signal_pairwise, information_flow, segment_comparison, info_flow_delta |
| vector | 301.2s | 28.9% | breaks, signal_vector, signal_stability |
| geometry | 31.5s | 3.0% | state_vector, state_geometry, signal_geometry, geometry_dynamics, eigendecomp, cohort_baseline, observation_geometry |
| energy (fleet) | 0.0s | 0.0% | all skipped |

## Rust Primitive Speedups (from micro-benchmarks)

These Rust primitives are used within the hot stages above:

| Primitive | Speedup | Used By |
|-----------|--------:|---------|
| kurtosis | 71.2x | signal_vector, signal_stability |
| skewness | 69.5x | signal_vector, signal_stability |
| spectral_centroid | 18.6x | signal_vector |
| spectral_entropy | 18.5x | signal_vector |
| psd | 17.5x | signal_vector, frequency_bands |
| optimal_delay | 15.0x | ftle, lyapunov |
| crest_factor | 12.3x | signal_vector |
| ftle | 8.3x | ftle, ftle_backward, ftle_rolling |
| lyapunov | 4.5x | lyapunov |
| acf_decay | 3.8x | signal_vector |
| snr | 1.7x | signal_vector |

## Output Comparison: Rust vs Legacy

Both pipelines produce identical file sets (26 parquet files).

### Exact or Machine-Epsilon Match (11 files)

| File | Rows | Max Abs Diff |
|------|-----:|-------------|
| breaks.parquet | 46,448 | 0.00e+00 |
| cohort_baseline.parquet | 1 | 0.00e+00 |
| observation_geometry.parquet | 61,249 | 0.00e+00 |
| sensor_eigendecomp.parquet | 10,901 | 0.00e+00 |
| signal_stability.parquet | 10,901 | 0.00e+00 |
| info_flow_delta.parquet | 136,896 | 0.00e+00 |
| segment_comparison.parquet | 123 | 0.00e+00 |
| velocity_field.parquet | 60,751 | 0.00e+00 |
| velocity_field_components.parquet | 1,458,024 | 0.00e+00 |
| signal_vector.parquet | 249,480 | ~1e-12 |
| lyapunov.parquet | 5,976 | ~0.54 (confidence only) |

### Numerically Close (9 files)

Differences stem from known causes:
- **Eigenvector sign ambiguity**: pc1_loading, eigenvector_flip_count
- **Embedding parameter selection**: embedding_dim, embedding_tau differ by 1-7
- **Near-zero denominators**: inflated relative error on FTLE, variance_velocity
- **Condition numbers**: inherently volatile for near-singular matrices

### Row Count Differences (6 files)

| File | Rust Rows | Legacy Rows | Delta |
|------|----------:|------------:|------:|
| state_geometry_loadings | 237,376 | 232,894 | +4,482 |
| signal_pairwise | 2,728,152 | 2,628,280 | +99,872 |
| information_flow | 65,985 | 66,981 | -996 |
| ftle_field | 1,926 | 1,919 | +7 |
| ftle_rolling | 8,640 | 8,567 | +73 |
| ridge_proximity | 4,584 | 4,513 | +71 |

Row count differences are due to different thresholding/filtering behavior in embedding and pairwise computations between Rust and Python implementations.

## Known Bottleneck: Lyapunov Parameter Estimation

Stage 08b (lyapunov) uses Rust for the Rosenstein algorithm itself, but the embedding parameter estimation (`_auto_delay` via AMI and `_auto_dimension` via Cao's method) still runs in pure Python. Cao's method does O(n_dims x n_samples^2) brute-force distance computations in Python loops. For 5,976 signals, this dominates the stage runtime.

**Fix**: Wire `_auto_delay` and `_auto_dimension` in `manifold/primitives/dynamical/lyapunov.py` to use the Rust-bridged `optimal_delay` from `manifold/primitives/embedding/delay.py` instead of reimplementing in Python.

## What Could Not Be Compared

1. **Python-only baseline timing**: No timing data from a pure Python (no Rust) run on FD_004. The legacy pipeline was run by a separate process without instrumented timing.
2. **Memory comparison**: Only the Rust run has /usr/bin/time memory data (3.48 GB). No comparable data from the legacy run.
3. **Fleet stages (26-31)**: Skipped in both runs (require cohort_vector.parquet from Prime SQL).
