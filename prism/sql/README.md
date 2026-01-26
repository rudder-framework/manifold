# PRISM SQL Pipeline

**CRITICAL**: Every computation MUST write to parquet. No silent failures. No data loss.

## Architecture

```
INPUT.parquet → SQL Pipeline → OUTPUT/*.parquet
                    ↓
              PRISM Engines (UDFs)
```

SQL handles:
- Orchestration
- Data persistence
- Validation
- Logging

PRISM handles:
- Irreducible computations (hurst, lyapunov, entropy, etc.)

## Execution Order

```
00_load/           → Load raw data
01_calculus/       → Pure SQL: derivatives, curvature, arc length
02_signal_class/   → Pure SQL: classify signal types
03_signal_typology/→ SQL + PRISM: hurst, entropy, spectral
04_behavioral_geometry/ → SQL + PRISM: PCA, clustering, DTW
05_dynamical_systems/   → SQL + PRISM: lyapunov, basin, attractor
06_causal_mechanics/    → SQL + PRISM: granger, transfer entropy
```

## Usage

```bash
# Run full pipeline
python run_all.py /path/to/observations.parquet

# Run full pipeline with custom output dir
python run_all.py /path/to/observations.parquet ./my_outputs/

# Validate outputs
python validate_outputs.py ./outputs/
```

## Output Files

| File | Description |
|------|-------------|
| `calculus.parquet` | Derivatives, curvature, arc length |
| `signal_class.parquet` | Signal classification (intensive/extensive/rate/state) |
| `signal_typology.parquet` | Behavioral metrics (hurst, entropy, etc.) |
| `behavioral_geometry.parquet` | Pairwise relationships |
| `dynamical_systems.parquet` | Regime detection, stability |
| `causal_mechanics.parquet` | Granger causality, transfer entropy |
| `manifest.json` | Row counts, timestamps |

## Data Persistence Pattern

Every `_write_*.sql` file follows this pattern:

```sql
-- Step 1: Create output table
CREATE OR REPLACE TABLE output AS SELECT ...;

-- Step 2: Validate (FAIL if invalid)
SELECT CASE WHEN COUNT(*) = 0 THEN error('FATAL: 0 rows') END FROM output;

-- Step 3: Write parquet
COPY output TO 'outputs/file.parquet' (FORMAT PARQUET);

-- Step 4: Confirm and log
INSERT INTO _write_log (file, rows, written_at) ...;
```

## Dependencies

- DuckDB >= 0.9.0
- Python >= 3.10
- PRISM engines (for stages 03+)
