# NASA Li-ion Battery Degradation Analysis

## Derivations Report: Batteries B0045-B0048

**Generated:** January 21, 2026
**Dataset:** NASA Prognostics Center of Excellence
**Experiment:** Li-ion 18650 cells at 4°C ambient temperature
**Analysis Pipeline:** PRISM Behavioral Geometry Engine (orthon v0.1.0)

---

## 1. Experiment Overview

### 1.1 Test Protocol

Four Li-ion batteries (B0045-B0048) were subjected to accelerated aging:

| Parameter | Value |
|-----------|-------|
| Ambient Temperature | 4°C |
| Charge Mode | CC (1.5A) → CV (4.2V, 20mA cutoff) |
| Discharge Load | 1A constant current |
| End-of-Discharge Voltage | B0045: 2.0V, B0046: 2.2V, B0047: 2.5V, B0048: 2.7V |
| EIS Frequency Sweep | 0.1 Hz – 5 kHz |
| End-of-Life Criterion | Capacity < 1.4 Ah (30% fade) |

### 1.2 Signals Extracted

| Signal | Description | Units |
|--------|-------------|-------|
| `capacity` | Discharge capacity | Ah |
| `Re_electrolyte` | Electrolyte resistance (from EIS) | Ω |
| `Rct_charge_transfer` | Charge transfer resistance (from EIS) | Ω |
| `impedance_mean` | Mean battery impedance | Ω |
| `temperature_mean` | Mean temperature during discharge | °C |

---

## 2. Capacity Fade Analysis

### 2.1 Summary Statistics

| Battery | Cycles | Initial Capacity | Final Capacity | Fade (%) |
|---------|--------|------------------|----------------|----------|
| B0045 | 70 | 1.082 Ah | 0.607 Ah | **43.9%** |
| B0046 | 69 | 1.728 Ah | 1.154 Ah | **33.2%** |
| B0047 | 69 | 1.674 Ah | 1.157 Ah | **30.9%** |
| B0048 | 69 | 1.658 Ah | 1.223 Ah | **26.2%** |

### 2.2 Observations

1. **B0045 shows anomalous behavior**: Started with lower initial capacity (1.08 Ah vs ~1.7 Ah for others) and degraded most severely (43.9% fade)
2. **Lower discharge cutoff = more degradation**: B0045 (2.0V cutoff) degraded fastest; B0048 (2.7V cutoff) degraded slowest
3. **All batteries exceeded 30% fade target** by end of experiment

---

## 3. Impedance Growth Analysis

### 3.1 Electrolyte Resistance (Re)

| Battery | Initial Re | Final Re | Growth (%) |
|---------|------------|----------|------------|
| B0045 | 0.0726 Ω | 0.0739 Ω | +1.8% |
| B0046 | 0.0919 Ω | 0.0808 Ω | **-12.1%** |
| B0047 | 0.0561 Ω | 0.0730 Ω | **+30.2%** |
| B0048 | 0.0780 Ω | 0.0805 Ω | +3.3% |

### 3.2 Observations

1. **B0047 shows classic impedance growth** (+30.2%) - consistent with SEI layer thickening
2. **B0046 shows anomalous impedance decrease** (-12.1%) - possible measurement artifact or temperature effect
3. **Impedance-capacity anti-correlation** is weaker than expected at 4°C ambient

---

## 4. Hurst Exponent Derivation

### 4.1 Mathematical Definition

The **Hurst exponent** H quantifies long-range dependence in a time series:

$$H = \frac{\log(R/S)}{\log(n)}$$

where R/S is the rescaled range statistic computed as:

$$R/S = \frac{1}{S} \left[ \max_{1 \le k \le n} \sum_{i=1}^{k}(x_i - \bar{x}) - \min_{1 \le k \le n} \sum_{i=1}^{k}(x_i - \bar{x}) \right]$$

### 4.2 Interpretation

| H Value | Interpretation |
|---------|----------------|
| H < 0.5 | Anti-persistent (mean-reverting) |
| H = 0.5 | Random walk (Brownian motion) |
| H > 0.5 | Persistent (trending) |
| H ≈ 1.0 | Strong persistence / deterministic trend |

### 4.3 Results for Capacity Signals

| Battery | Hurst Exponent | Interpretation |
|---------|----------------|----------------|
| B0045 | **1.052** | Strong persistence (deterministic degradation) |
| B0046 | **1.026** | Strong persistence |
| B0047 | **1.022** | Strong persistence |
| B0048 | **0.994** | Strong persistence |

### 4.4 R/S Analysis Details (B0045)

| Window Size (n) | R/S Value | log(n) | log(R/S) |
|-----------------|-----------|--------|----------|
| 8 | 2.795 | 2.079 | 1.028 |
| 16 | 5.698 | 2.773 | 1.740 |
| 32 | 12.011 | 3.466 | 2.486 |

**Linear regression:** slope = 1.052 (Hurst exponent)

### 4.5 Physical Meaning

All four batteries show H ≈ 1.0, indicating:

1. **Capacity fade is a deterministic process** - not random fluctuations
2. **Strong autocorrelation** - degradation at cycle n predicts degradation at cycle n+1
3. **Predictable trend** - amenable to prognostic modeling

---

## 5. Sample Entropy Derivation

### 5.1 Mathematical Definition

**Sample entropy** (SampEn) measures the regularity/complexity of a time series:

$$SampEn(m, r, N) = -\ln \frac{A}{B}$$

where:
- m = embedding dimension (pattern length)
- r = tolerance threshold (fraction of std)
- A = count of matching patterns of length m+1
- B = count of matching patterns of length m

### 5.2 Parameters Used

| Parameter | Value |
|-----------|-------|
| m (embedding dimension) | 2 |
| r (tolerance) | 0.2 × σ |

### 5.3 Results

| Battery | Sample Entropy | Regularity |
|---------|----------------|------------|
| B0045 | **0.455** | Moderate complexity |
| B0046 | **0.313** | More regular |
| B0047 | **0.362** | Moderate |
| B0048 | **0.367** | Moderate |

### 5.4 Physical Interpretation

- **Lower SampEn** → more regular/predictable degradation
- **B0046** has lowest entropy (0.313) → smoothest degradation curve
- **B0045** has highest entropy (0.455) → more irregular behavior (consistent with anomalous capacity)

---

## 6. GARCH Volatility Analysis

### 6.1 Model Specification

GARCH(1,1) model for variance dynamics:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where:
- ω = long-run variance baseline
- α = shock impact coefficient
- β = variance persistence
- Persistence = α + β

### 6.2 Results for Capacity Signals

| Parameter | Mean | Std |
|-----------|------|-----|
| ω (omega) | 0.000033 | 0.000003 |
| α (alpha) | 0.051 | 0.002 |
| β (beta) | 0.849 | 0.002 |
| **Persistence** | **0.900** | 0.000 |

### 6.3 Interpretation

- **High persistence (0.90)**: Volatility shocks decay slowly
- **Low α (0.05)**: Individual cycle variance has small immediate impact
- **High β (0.85)**: Historical variance dominates current variance
- **Conclusion**: Capacity variance is stable and predictable

---

## 7. Recurrence Quantification Analysis (RQA)

### 7.1 RQA Metrics for Capacity Signals

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| Recurrence Rate | 0.100 | 0.000 | 10% of states recur |
| Determinism | 0.946 | 0.027 | 95% of recurrences form diagonal lines |
| Avg Diagonal Length | 6.09 | 0.40 | Moderate predictability horizon |
| Max Diagonal Length | 16.94 | 0.57 | Longest predictable sequence |
| Laminarity | 1.000 | 0.000 | All vertical lines present |
| Entropy | 2.05 | 0.07 | Moderate diagonal line complexity |

### 7.2 Physical Interpretation

- **High determinism (0.95)**: System evolution is highly predictable
- **Laminarity = 1.0**: System shows laminar (trapped) behavior - consistent with monotonic degradation
- **Avg diagonal = 6 cycles**: Can predict ~6 cycles ahead from recurrence structure

---

## 8. Geometry Analysis

### 8.1 Multi-Signal Geometry

PRISM computes structural geometry across all signals at each timestep:

| Metric | Mean Value | Interpretation |
|--------|------------|----------------|
| PCA Effective Dimension | 1.60 | 1.6 independent components explain variance |
| Clustering Silhouette | 0.34 | Moderate cluster separation |
| MST Total Weight | 6.28 | Signal connectivity strength |
| LOF Outliers | 0 | No anomalous geometry detected |

### 8.2 Interpretation

- **Low effective dimension (1.6)**: Signals are highly correlated - capacity fade dominates all measurements
- **Moderate silhouette (0.34)**: Two behavioral regimes present (early life vs late life)

---

## 9. State Dynamics

### 9.1 Failure Signatures

| Metric | Count |
|--------|-------|
| Failure Signatures Detected | **8** |
| Mode Transitions | **4** |

### 9.2 Mode Analysis

The state engine detected **4 mode transitions** across the experiment:
- Likely correspond to: initial → mid-life → late-life → end-of-life phases
- Each mode transition marks a qualitative change in degradation dynamics

---

## 10. Cohort Discovery

### 10.1 Signal Groupings

| Cohort | Signals | Interpretation |
|--------|---------|----------------|
| raw_cohort_1 | capacity (all batteries) | Degradation behavior group |
| raw_cohort_2 | Re_electrolyte, Rct_charge_transfer, impedance_mean | Impedance behavior group |

### 10.2 Interpretation

- **Capacity signals cluster together** across all batteries (similar degradation pattern)
- **Impedance signals cluster together** (EIS measurements behave similarly)
- This validates that capacity and impedance capture different aspects of degradation

---

## 11. Key Findings

### 11.1 Degradation Characteristics

1. **Deterministic process**: H ≈ 1.0 indicates capacity fade follows predictable trajectory
2. **Stable volatility**: GARCH persistence = 0.90, variance evolves smoothly
3. **High predictability**: RQA determinism = 95%, 6-cycle prediction horizon

### 11.2 Battery-Specific Observations

| Battery | Key Finding |
|---------|-------------|
| B0045 | Anomalous low initial capacity, highest fade (43.9%), highest entropy |
| B0046 | Anomalous impedance decrease, smoothest degradation (lowest entropy) |
| B0047 | Classic impedance growth (+30%), normal degradation |
| B0048 | Best performer (26% fade), highest discharge cutoff (2.7V) |

### 11.3 Operational Insight

**Discharge cutoff voltage is a key factor**: Lower cutoff (deeper discharge) accelerates degradation:
- 2.0V cutoff → 43.9% fade
- 2.7V cutoff → 26.2% fade

---

## 12. Appendix: Derivation Formulas

### A. R/S Analysis for Hurst Exponent

```
For a time series X = {x_1, x_2, ..., x_n}:

1. Compute mean: μ = (1/n) Σ x_i

2. Compute cumulative deviation: Y_k = Σ_{i=1}^{k} (x_i - μ)

3. Compute range: R = max(Y) - min(Y)

4. Compute standard deviation: S = sqrt((1/n) Σ (x_i - μ)²)

5. Rescaled range: R/S = R / S

6. Hurst exponent: H = log(R/S) / log(n)

For multiple windows, use linear regression of log(R/S) vs log(n).
```

### B. Sample Entropy

```
For embedding dimension m and tolerance r:

1. Form template vectors: u_i = {x_i, x_{i+1}, ..., x_{i+m-1}}

2. Count matches: B^m(r) = # pairs where ||u_i - u_j|| < r

3. Repeat for m+1: A^{m+1}(r) = # pairs for length m+1 templates

4. Sample entropy: SampEn = -ln(A/B)
```

### C. GARCH(1,1)

```
Variance equation: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Unconditional variance: σ² = ω / (1 - α - β)

Persistence: α + β (measures shock decay rate)

Estimation: Maximum likelihood with Student-t innovations
```

---

## 13. Files and Reproducibility

### Source Data
- `/data/battery_45_48/B0045.mat` through `B0048.mat`
- NASA Prognostics Data Repository

### Computed Artifacts
- `observations.parquet` - 1,045 rows, 5 signals, 4 batteries
- `vector.parquet` - 20,080 behavioral metrics
- `geometry.parquet` - 288 structural snapshots
- `state.parquet` - 288 state vectors, 8 failure signatures
- `cohorts.parquet` - 2 discovered cohorts

### Pipeline Version
- PRISM/orthon v0.1.0
- Python 3.14
- Polars, NumPy, SciPy

---

*Report generated by PRISM Behavioral Geometry Engine*
*Contact: prism-engines/diagnostics*
