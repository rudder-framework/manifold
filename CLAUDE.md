# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⛔ STOP: DATA COMPLIANCE IS ORTHON'S RESPONSIBILITY

**ORTHON delivers data. PRISM computes. The contract is non-negotiable.**

If `observations.parquet` or `manifest.yaml` does not comply with PRISM's schema:

1. **STOP** - Do not proceed with computation
2. **REJECT** - Return an error explaining what's wrong
3. **DO NOT** adjust PRISM's codebase or schema to accommodate bad data

**Bad data from ORTHON is ORTHON's problem to fix.**

```
❌ WRONG: "The data has column 'sensor_name' instead of 'signal_id',
          let me update PRISM to accept both..."

✅ RIGHT: "The data has column 'sensor_name' instead of 'signal_id'.
          ORTHON must rename this column before sending to PRISM."
```

**PRs adjust the codebase. Bad data does not.**

If you find yourself about to modify PRISM to handle non-compliant data, STOP and ask: "Should ORTHON fix this instead?"

The answer is almost always YES.

---

## CRITICAL: PRISM ↔ ORTHON Architecture

**PRISM is an HTTP service ONLY. NOT a pip install. NO code sharing with ORTHON.**

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│     ORTHON      │ ──────────────────▶   │      PRISM      │
│   (Frontend)    │   POST /compute       │  (Compute API)  │
│   Streamlit     │ ◀──────────────────   │  localhost:8100 │
│                 │   {status, parquets}  │                 │
└─────────────────┘                       └─────────────────┘
        │                                         │
        │ reads                                   │ writes
        ▼                                         ▼
   ~/prism/data/*.parquet                  ~/prism/data/*.parquet
```

**ORTHON creates observations.parquet. PRISM only reads it.**

---

## What PRISM Is

A behavioral geometry engine for signal topology analysis. PRISM transforms raw observations into eigenvalue-based state representations that capture the SHAPE of signal distributions.

**Core Philosophy:**
- Typology-guided engine selection (not all engines run on all signals)
- Scale-invariant features only (ratios, entropy, kurtosis - no absolute values)
- Eigenvalue-based state representation via SVD
- Geometry dynamics for trajectory analysis

---

## Architecture (v2)

```
observations.parquet
       ↓
┌──────────────────────────────────────────────────────────────┐
│                    TYPOLOGY ENGINE                           │
│  Signal characterization: smoothness, periodicity, tails     │
│  Output: typology.parquet                                    │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                   SIGNAL VECTOR                              │
│  Per-signal features (scale-invariant only)                  │
│  Output: signal_vector.parquet (with I column for temporal)  │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                    STATE VECTOR                              │
│  System state via eigenvalues (SVD)                          │
│  - Eigenvalues capture SHAPE of signal cloud                 │
│  - effective_dim from participation ratio                    │
│  - Multi-mode detection                                      │
│  Output: state_vector.parquet                                │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                  GEOMETRY LAYER                              │
│  signal_geometry.parquet  - signal-to-state relationships    │
│  signal_pairwise.parquet  - pairwise signal relationships    │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│               GEOMETRY DYNAMICS LAYER                        │
│  "You have position. You have shape. Here are derivatives."  │
│  geometry_dynamics.parquet - velocity, acceleration, jerk    │
│  Collapse detection, trajectory classification               │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                  DYNAMICS LAYER                              │
│  dynamics.parquet         - Lyapunov, RQA, attractor         │
│  information_flow.parquet - Transfer entropy, Granger        │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│                   SQL LAYER                                  │
│  zscore.parquet           - Normalized metrics               │
│  statistics.parquet       - Summary statistics               │
│  correlation.parquet      - Correlation matrix               │
│  regime_assignment.parquet - State labels                    │
└──────────────────────────────────────────────────────────────┘
```

---

## CLI Commands

```bash
# Run full pipeline on a data directory
python -m prism data/cmapss

# Run individual stages
python -m prism typology data/cmapss
python -m prism signal-vector data/cmapss
python -m prism signal-vector-temporal data/cmapss
python -m prism state-vector data/cmapss
python -m prism geometry data/cmapss
python -m prism dynamics data/cmapss

# Run legacy 53-engine pipeline
python -m prism --legacy data/manifest.yaml
```

---

## Input: observations.parquet (Schema v2.0)

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| signal_id | str | What signal (temp, pressure, return) |
| I | UInt32 | Index 0,1,2,3... per unit/signal. Sequential, no gaps. |
| value | Float64 | The measurement |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| unit_id | str | Label for grouping. Blank is fine. "bananas" is fine. |

### About unit_id

**unit_id is OPTIONAL.** It's just a sticky note for humans.

- PRISM passes unit_id through to output for SQL filtering
- unit_id has ZERO effect on compute
- unit_id can be blank, null, "pump_1", "friday_data", "bananas" - whatever
- unit_id is NOT an index (that's what I is for)
- If no unit_id provided, PRISM uses blank ""

**DO NOT validate unit_id contents. DO NOT require unit_id.**

### Example
```
unit_id | signal_id | I | value
--------|-----------|---|------
pump_1  | temp      | 0 | 45.2
pump_1  | temp      | 1 | 45.4
pump_1  | pressure  | 0 | 101.3
pump_1  | pressure  | 1 | 101.5
        | temp      | 0 | 30.1   ← blank unit_id is fine
```

**If data is not in this format, ORTHON must transform it first.**

---

## Output Files

| File | Description | Rows |
|------|-------------|------|
| typology.parquet | Signal characterization | units × signals |
| signal_vector.parquet | Per-signal features (with I column) | units × signals × time |
| state_vector.parquet | System state (eigenvalues) | units |
| signal_geometry.parquet | Signal-to-state relationships | units × signals |
| signal_pairwise.parquet | Pairwise relationships | units × N²/2 pairs |
| geometry_dynamics.parquet | Derivatives of geometry evolution | units × engines × time |
| dynamics.parquet | Lyapunov, RQA, attractor | units × signals |
| information_flow.parquet | Transfer entropy, Granger | units × N² pairs |
| zscore.parquet | Normalized metrics | observations |
| statistics.parquet | Summary statistics | units × signals |
| correlation.parquet | Correlation matrix | units × signals² |
| regime_assignment.parquet | State labels | observations |

---

## Key Concepts

### Typology-Guided Engine Selection

Not all engines run on all signals. The typology engine classifies each signal first:

| Classification | Engines Selected |
|----------------|------------------|
| SMOOTH | rolling_kurtosis, rolling_entropy, rolling_crest_factor |
| NOISY | kurtosis, entropy, crest_factor (larger window) |
| IMPULSIVE | kurtosis, crest_factor, peak_ratio |
| PERIODIC | harmonics_ratio, band_ratios, spectral_centroid |
| APERIODIC | entropy, hurst |
| NON_STATIONARY | rolling engines only (global stats meaningless) |

### Scale-Invariant Features Only

All features are scale-invariant (ratios, entropy, kurtosis). Absolute values deprecated:
- rms, peak, mean, std
- rolling_rms, rolling_mean, rolling_std, rolling_range
- envelope, total_power

### Eigenvalue-Based State

The state vector uses SVD to compute eigenvalues of the signal distribution:

```python
# Centroid (position in feature space)
centroid = mean(signal_matrix, axis=0)

# Eigenvalues (shape of signal cloud)
U, S, Vt = svd(signal_matrix - centroid)
eigenvalues = S² / (N - 1)

# Effective dimension (participation ratio)
effective_dim = (Σλ)² / Σλ²

# Multi-mode detection
is_multimode = (λ₂/λ₁ > 0.5) and (n_significant_modes >= 2)
```

**Key insight:** For two signals to occupy the same state, they must match across ALL feature dimensions. Eigenvalues capture this shape.

### Geometry Dynamics (Differential Geometry)

The geometry dynamics engine computes derivatives of the geometry evolution:

```python
# First derivative (velocity/tangent)
dx/dt = (x[t+1] - x[t-1]) / (2*dt)

# Second derivative (acceleration/curvature)
d²x/dt² = (x[t+1] - 2*x[t] + x[t-1]) / dt²

# Third derivative (jerk/torsion)
d³x/dt³ = derivative of acceleration

# Curvature
κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
```

**Trajectory Classification:**
| Type | Meaning |
|------|---------|
| STABLE | Low velocity and acceleration |
| CONVERGING | Moving toward equilibrium |
| DIVERGING | Moving away from equilibrium |
| OSCILLATING | Velocity changes sign periodically |
| CHAOTIC | High variability in derivatives |
| COLLAPSING | Sustained loss of effective dimension |
| EXPANDING | Sustained gain in effective dimension |

**Collapse Detection:** Identifies when effective_dim has sustained negative velocity - the system is losing degrees of freedom.

---

## Directory Structure

```
prism/
├── prism/
│   ├── cli.py                    # Main CLI
│   ├── signal_vector.py          # Aggregate signal features
│   ├── signal_vector_temporal.py # Temporal signal features
│   ├── sql_runner.py             # SQL engine runner
│   │
│   ├── engines/
│   │   ├── engine_manifest.yaml  # Scale-invariant engine config
│   │   ├── typology_engine.py    # Signal characterization
│   │   ├── state_vector.py       # Eigenvalue-based state
│   │   ├── signal_geometry.py    # Signal-to-state relationships
│   │   ├── signal_pairwise.py    # Pairwise relationships
│   │   ├── geometry_dynamics.py  # Differential geometry (derivatives)
│   │   ├── dynamics_runner.py    # Lyapunov, RQA
│   │   ├── information_flow_runner.py  # Transfer entropy, Granger
│   │   ├── signal/               # Individual signal engines
│   │   ├── rolling/              # Rolling window engines
│   │   └── sql/                  # SQL engines
│   │
│   ├── primitives/               # Low-level computation primitives
│   │   ├── individual/           # Per-signal analysis
│   │   ├── pairwise/             # Between-signal analysis
│   │   ├── matrix/               # Matrix operations (SVD, PCA)
│   │   ├── information/          # Information theory
│   │   ├── topology/             # Topological analysis
│   │   └── dynamical/            # Dynamical systems
│   │
│   └── _legacy/                  # Legacy 53-engine pipeline
│       ├── runner.py
│       ├── python_runner.py
│       └── ram_manager.py
│
├── data/
│   └── cmapss/                   # C-MAPSS turbofan dataset
│       ├── observations.parquet
│       ├── typology.parquet
│       ├── signal_vector.parquet
│       └── state_vector.parquet
│
└── ENGINE_INVENTORY.md           # Full engine inventory
```

---

## Rules

1. **Typology runs first** - classifies signals before any computation
2. **Scale-invariant features only** - no absolute values (rms, peak, mean, std deprecated)
3. **Insufficient data → return NaN, never skip** - engines must not fail
4. **No domain-specific logic in PRISM** - PRISM is domain-agnostic
5. **No interpretation in PRISM** - compute only, no decisions
6. **Slow compute is fine** - publication-grade results take time

---

## Do NOT

- Import PRISM into ORTHON (HTTP only)
- Use deprecated scale-dependent engines (rms, peak, mean, std, etc.)
- Skip typology classification
- Make decisions about result interpretation
- Create observations.parquet (ORTHON's job)

---

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary), DuckDB (SQL engines)
- **Core:** NumPy, SciPy, scikit-learn
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx

---

## Session Recovery

```bash
# Start PRISM (from repo root, using venv)
cd ~/prism
./venv/bin/python -m prism data/cmapss

# Or run legacy pipeline:
./venv/bin/python -m prism --legacy data/manifest.yaml
```

---

## DO NOT TOUCH

- ORTHON code lives in `~/orthon/` - let CC ORTHON handle it
- Never `pip install prism` - PRISM is HTTP only
- Never create observations.parquet - ORTHON's job

---

## Credits

- **Avery Rudder** - "Laplace transform IS the state engine" - eigenvalue insight
- Architecture: Typology-guided, scale-invariant, eigenvalue-based
