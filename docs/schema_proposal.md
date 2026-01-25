# ORTHON Schema Proposal: Separating Entity, Signal, and Grouping Concepts

## The Problem

The term "cohort" is overloaded:
- Sometimes means "signals with similar behavior" (auto-discovered)
- Sometimes means "entities grouped for analysis" (user-defined)
- Sometimes means "the whole population" (implicit)

## Proposed Hierarchy

```
POPULATION (implicit)
    └── All entities in the dataset
    └── e.g., "FD002 train set" = 260 engines

COHORT (user-defined)
    └── Named group of entities for comparison
    └── e.g., "healthy_fleet", "degraded_fleet", "maintenance_batch_1"
    └── User assigns entities to cohorts

ENTITY (from data)
    └── The thing that fails/degrades
    └── e.g., engine FD002_U001

SIGNAL (from data)
    └── A measurement on an entity
    └── e.g., T2 temperature sensor

SIGNAL_CLASS (auto-discovered)
    └── Signals with similar behavioral fingerprints
    └── e.g., "temperature_sensors", "pressure_sensors", "speed_sensors"
```

## Schema Changes

### 1. observations.parquet (unchanged)
```
entity_id   | String  | The unit (engine, bearing)
signal_id   | String  | The sensor/measurement
timestamp   | Float64 | Time (cycles, seconds)
value       | Float64 | Raw reading
```

### 2. NEW: cohorts.parquet (user-defined entity groupings)
```
cohort_id   | String   | User-defined group name
entity_id   | String   | Entity in this cohort
assigned_at | Datetime | When assigned
assigned_by | String   | "user" | "auto" | "rule"
metadata    | JSON     | Optional user notes
```

Example:
```
cohort_id       | entity_id    | assigned_by
----------------|--------------|------------
healthy         | FD002_U001   | user
healthy         | FD002_U002   | user
degraded        | FD002_U050   | user
high_stress     | FD002_U100   | rule
```

### 3. NEW: signal_classes.parquet (auto-discovered signal groupings)
```
class_id        | String   | Auto-generated class name
signal_id       | String   | Signal in this class
similarity      | Float64  | How well it fits (0-1)
discovered_at   | Datetime | When discovered
method          | String   | "behavioral" | "correlation" | "spectral"
```

Example:
```
class_id            | signal_id | similarity
--------------------|-----------|------------
thermal_sensors     | T2        | 0.95
thermal_sensors     | T24       | 0.92
thermal_sensors     | T30       | 0.88
pressure_sensors    | P2        | 0.97
pressure_sensors    | P15       | 0.94
speed_sensors       | Nf        | 0.91
speed_sensors       | Nc        | 0.89
```

### 4. NEW: populations.parquet (dataset-level metadata)
```
population_id   | String   | Dataset identifier
entity_count    | Int64    | Number of entities
signal_count    | Int64    | Signals per entity
time_range      | String   | Min-max timestamp
source          | String   | "cmapss_fd002", "femto", etc.
loaded_at       | Datetime | When ingested
```

## How This Changes Analysis

### Before (muddied)
```python
# What does "cohort" mean here?
cohorts = discover_cohorts(observations)  # Signals? Entities? Both?
state = compute_cohort_state(cohorts)     # Aggregating what?
```

### After (clear)
```python
# Entity groupings (user-defined)
cohorts = load_cohorts()  # User's entity groups
cohorts = assign_cohort(entity_id="FD002_U001", cohort_id="healthy")

# Signal classifications (auto-discovered)
signal_classes = discover_signal_classes(observations)  # Groups similar sensors

# Analysis is explicit about what it's aggregating
entity_state = compute_entity_state(entity_id)           # One engine
cohort_state = compute_cohort_state(cohort_id)           # User's group
population_state = compute_population_state(population_id)  # All engines
signal_class_behavior = analyze_signal_class(class_id)   # Sensor group
```

## User Workflows

### Workflow 1: "I want to analyze engine 47"
```python
entity_id = "FD002_U047"
typology = compute_signal_typology(entity_id)
geometry = compute_structural_geometry(entity_id)
dynamics = compute_dynamical_systems(entity_id)
```

### Workflow 2: "I want to compare healthy vs degraded engines"
```python
# User defines cohorts
assign_cohort("FD002_U001", "healthy")
assign_cohort("FD002_U002", "healthy")
assign_cohort("FD002_U050", "degraded")
assign_cohort("FD002_U051", "degraded")

# Compare
healthy_profile = aggregate_cohort_profile("healthy")
degraded_profile = aggregate_cohort_profile("degraded")
comparison = compare_cohorts("healthy", "degraded")
```

### Workflow 3: "Which sensors behave similarly?"
```python
# Auto-discover signal classes
classes = discover_signal_classes()
# Returns: thermal_sensors, pressure_sensors, speed_sensors, etc.

# Analyze a class
thermal_behavior = analyze_signal_class("thermal_sensors")
```

### Workflow 4: "Show me fleet-wide patterns"
```python
# Population-level analysis
population = "FD002_train"
fleet_health = compute_population_health(population)
anomalies = detect_population_anomalies(population)
```

## Default Behavior

If user doesn't define cohorts:
- `cohort_id = "default"` contains all entities
- Analysis proceeds as before
- No breaking changes

## Entry Points

```bash
# Entity-level (existing, unchanged)
python -m orthon.signal_typology --entity FD002_U001

# Cohort management (new)
python -m orthon.cohorts assign FD002_U001 healthy
python -m orthon.cohorts assign FD002_U002 healthy
python -m orthon.cohorts list
python -m orthon.cohorts compare healthy degraded

# Signal class discovery (renamed from cohort discovery)
python -m orthon.signal_classes discover
python -m orthon.signal_classes show thermal_sensors

# Population-level (new)
python -m orthon.population summary
python -m orthon.population anomalies
```

## Migration Path

1. Rename existing `cohort.py` discovery → `signal_class_discovery.py`
2. Rename output `cohorts.parquet` → `signal_classes.parquet`
3. Create new `cohorts.parquet` for user-defined entity groups
4. Add `populations.parquet` for dataset metadata
5. Update downstream code to use explicit concepts

## Summary Table

| Concept | Definition | Source | Granularity |
|---------|-----------|--------|-------------|
| **Population** | All entities in dataset | Implicit from data | Dataset |
| **Cohort** | Named entity group | User-defined | Group of entities |
| **Entity** | Individual unit | From data | Single unit |
| **Signal** | Measurement type | From data | Single sensor |
| **Signal Class** | Similar signals | Auto-discovered | Group of signals |
