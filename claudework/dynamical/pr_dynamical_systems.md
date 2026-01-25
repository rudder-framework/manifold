# PR: Dynamical Systems Refactor — State + Transitions Architecture

## Summary

Refactor dynamical systems from "one summary per entity" to "state per window + transitions only when meaningful."

Aligns with geometry layer pattern:
- **States table**: 6 metrics at each timestamp
- **Transitions table**: Only when something changes

## Background

Current implementation computes dynamics over full history and returns one summary row per entity. This loses temporal granularity and buries transitions in a count.

New implementation tracks state evolution and surfaces transitions as first-class queryable events.

---

## The 6 Dynamics Metrics

| Metric | Question | Range | Math |
|--------|----------|-------|------|
| **trajectory** | Where is it going? | categorical | Phase space velocity direction |
| **stability** | Will perturbations grow? | -1 to 1 | Lyapunov exponent (normalized) |
| **attractor** | What does it settle toward? | categorical | Attractor classification |
| **predictability** | How far can we forecast? | 0-1 | Permutation entropy |
| **coupling** | How do signals drive each other? | 0-1 | Mean correlation / Granger strength |
| **memory** | Does past influence future? | 0-1 | Hurst exponent |

---

## Schema

### dynamics_states (one row per entity per window)

```sql
CREATE TABLE dynamics_states (
    entity_id TEXT NOT NULL,
    window_idx INT NOT NULL,
    timestamp TIMESTAMP,
    
    -- Categorical states
    trajectory TEXT,       -- converging | diverging | periodic | chaotic | stationary
    attractor TEXT,        -- fixed_point | limit_cycle | strange | none
    
    -- Numeric metrics (all 0-1 or -1 to 1 normalized)
    stability FLOAT,       -- >0 stable, <0 unstable
    predictability FLOAT,  -- 1=deterministic, 0=random
    coupling FLOAT,        -- 1=fully coupled, 0=independent
    memory FLOAT,          -- 0.5=random walk, >0.5=persistent, <0.5=anti-persistent
    
    PRIMARY KEY (entity_id, window_idx)
);

CREATE INDEX idx_dynamics_states_entity ON dynamics_states(entity_id);
CREATE INDEX idx_dynamics_states_time ON dynamics_states(timestamp);
```

### dynamics_transitions (only when state changes)

```sql
CREATE TABLE dynamics_transitions (
    entity_id TEXT NOT NULL,
    window_idx INT NOT NULL,
    timestamp TIMESTAMP,
    
    field TEXT NOT NULL,        -- which metric changed
    from_value TEXT,            -- previous value (string for flexibility)
    to_value TEXT,              -- new value
    delta FLOAT,                -- numeric change magnitude (if applicable)
    
    transition_type TEXT,       -- bifurcation | collapse | recovery | shift | flip
    severity TEXT,              -- mild | moderate | severe
    
    PRIMARY KEY (entity_id, window_idx, field)
);

CREATE INDEX idx_dynamics_transitions_severity ON dynamics_transitions(severity);
CREATE INDEX idx_dynamics_transitions_type ON dynamics_transitions(transition_type);
```

---

## Transition Detection Logic

### Categorical Fields (trajectory, attractor)

```python
if state_t.trajectory != state_t_minus_1.trajectory:
    emit_transition(field="trajectory", ...)
```

### Numeric Fields (stability, predictability, coupling, memory)

```python
# Thresholds for "meaningful" change
THRESHOLDS = {
    "stability": 0.2,      # 20% of range
    "predictability": 0.15,
    "coupling": 0.15,
    "memory": 0.1
}

delta = abs(state_t.stability - state_t_minus_1.stability)
if delta > THRESHOLDS["stability"]:
    emit_transition(field="stability", delta=delta, ...)
```

### Severity Classification

| Severity | Criteria |
|----------|----------|
| **mild** | Delta > threshold but < 2x threshold |
| **moderate** | Delta > 2x threshold OR sign change |
| **severe** | Categorical flip OR delta > 3x threshold OR stability crosses zero |

### Transition Types

| Type | Meaning |
|------|---------|
| **bifurcation** | Stability crossed zero (stable → unstable) |
| **collapse** | Predictability or coupling dropped sharply |
| **recovery** | Metrics improving after previous decline |
| **shift** | Categorical change (trajectory or attractor type) |
| **flip** | Memory crossed 0.5 (persistent ↔ anti-persistent) |

---

## File Changes

### Renamed/Restructured
- `prism/dynamical_systems/` — Core computation modules

### Added
- `prism/dynamical_systems/metrics/trajectory.py`
- `prism/dynamical_systems/metrics/stability.py`
- `prism/dynamical_systems/metrics/attractor.py`
- `prism/dynamical_systems/metrics/predictability.py`
- `prism/dynamical_systems/metrics/coupling.py`
- `prism/dynamical_systems/metrics/memory.py`
- `prism/dynamical_systems/transitions.py` — Transition detection logic

### Modified
- `prism/entry_points/dynamical_systems.py` — New architecture

### Output Files
- `data/dynamics_states.parquet` (replaces `dynamical_systems.parquet`)
- `data/dynamics_transitions.parquet` (new)

---

## Integration with Geometry Layer

Geometry transitions can trigger dynamics recomputation:

```python
# If geometry shows fragility increase, check dynamics
geometry_transitions = query("SELECT * FROM geometry_transitions WHERE field = 'curvature' AND severity = 'severe'")

for gt in geometry_transitions:
    # Dynamics at same window likely shows instability
    dynamics = query(f"SELECT * FROM dynamics_states WHERE entity_id = '{gt.entity_id}' AND window_idx = {gt.window_idx}")
    
    if dynamics.stability < 0:
        alert(f"Confirmed: geometry fragility + dynamics instability at {gt.timestamp}")
```

---

## Escalation to Mechanics

Simple query replaces buried boolean:

```sql
-- Entities needing mechanics analysis
SELECT DISTINCT entity_id 
FROM dynamics_transitions 
WHERE severity = 'severe'
   OR transition_type = 'bifurcation'
```

---

## Example Output

### dynamics_states.parquet

| entity_id | window_idx | trajectory | stability | attractor | predictability | coupling | memory |
|-----------|------------|------------|-----------|-----------|----------------|----------|--------|
| engine_001 | 0 | converging | 0.82 | limit_cycle | 0.71 | 0.65 | 0.58 |
| engine_001 | 1 | converging | 0.79 | limit_cycle | 0.69 | 0.63 | 0.57 |
| engine_001 | 2 | diverging | -0.15 | none | 0.42 | 0.31 | 0.52 |
| engine_001 | 3 | chaotic | -0.31 | strange | 0.28 | 0.22 | 0.49 |

### dynamics_transitions.parquet

| entity_id | window_idx | field | from_value | to_value | delta | transition_type | severity |
|-----------|------------|-------|------------|----------|-------|-----------------|----------|
| engine_001 | 2 | trajectory | converging | diverging | — | shift | moderate |
| engine_001 | 2 | stability | 0.79 | -0.15 | 0.94 | bifurcation | severe |
| engine_001 | 2 | attractor | limit_cycle | none | — | shift | severe |
| engine_001 | 2 | coupling | 0.63 | 0.31 | 0.32 | collapse | moderate |
| engine_001 | 3 | trajectory | diverging | chaotic | — | shift | moderate |
| engine_001 | 3 | attractor | none | strange | — | shift | moderate |

---

## Migration

1. Run new entry point: `python -m prism.entry_points.dynamical_systems`
2. Produces two parquet files instead of one
3. Update downstream consumers to query new schema
4. Deprecate `n_regime_changes` summary field (now queryable from transitions)

---

## Acceptance Criteria

- [ ] dynamics_states.parquet produced with 6 metrics per window
- [ ] dynamics_transitions.parquet produced only for meaningful changes
- [ ] Transition severity correctly classified
- [ ] Bifurcation detection (stability sign change) working
- [ ] Integration test: geometry curvature drop → dynamics instability correlation
- [ ] Escalation query returns correct entities
