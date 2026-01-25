#!/usr/bin/env python3
"""
Dynamical Systems Entry Point (Pure Orchestrator)
=================================================

Orchestrates dynamical systems analysis by calling engine modules.

Pipeline: signals → signal_typology → manifold_geometry → dynamical_systems → causal_mechanics

Input:
    - data/manifold_geometry.parquet (geometry per window)

Output:
    - data/dynamics_states.parquet (6 metrics per entity per window)
    - data/dynamics_transitions.parquet (only when state changes)

Engines:
    - prism/dynamical_systems/engines/takens.py → trajectory
    - prism/dynamical_systems/engines/lyapunov.py → stability
    - prism/dynamical_systems/engines/recurrence.py → attractor
    - prism/dynamical_systems/engines/permutation_entropy.py → predictability
    - prism/dynamical_systems/engines/granger.py → coupling
    - prism/dynamical_systems/engines/hurst.py → memory

Usage:
    python -m prism.entry_points.dynamical_systems
    python -m prism.entry_points.dynamical_systems --force
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import polars as pl


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DynamicsState:
    """State of dynamical system at a single window."""
    entity_id: str
    window_idx: int
    timestamp: Optional[Any]
    
    # Categorical
    trajectory: str      # converging | diverging | periodic | chaotic | stationary
    attractor: str       # fixed_point | limit_cycle | strange | none
    
    # Numeric (all normalized)
    stability: float     # -1 to 1, >0 stable
    predictability: float  # 0-1, higher = more predictable
    coupling: float      # 0-1, higher = more coupled
    memory: float        # 0-1, 0.5 = random walk


@dataclass 
class DynamicsTransition:
    """A meaningful state change."""
    entity_id: str
    window_idx: int
    timestamp: Optional[Any]
    
    field: str           # which metric changed
    from_value: str      # previous (as string)
    to_value: str        # new (as string)
    delta: Optional[float]  # numeric change if applicable
    
    transition_type: str  # bifurcation | collapse | recovery | shift | flip
    severity: str        # mild | moderate | severe


# =============================================================================
# Engine Imports (with fallbacks)
# =============================================================================

def _import_engines():
    """Import engine modules with fallback to local versions."""
    engines = {}
    
    # Try prism package first, then local
    try:
        from prism.dynamical_systems.engines import takens
        engines['takens'] = takens
    except ImportError:
        try:
            import takens
            engines['takens'] = takens
        except ImportError:
            engines['takens'] = None
    
    try:
        from prism.dynamical_systems.engines import lyapunov
        engines['lyapunov'] = lyapunov
    except ImportError:
        try:
            import lyapunov
            engines['lyapunov'] = lyapunov
        except ImportError:
            engines['lyapunov'] = None
    
    try:
        from prism.dynamical_systems.engines import recurrence
        engines['recurrence'] = recurrence
    except ImportError:
        try:
            import recurrence
            engines['recurrence'] = recurrence
        except ImportError:
            engines['recurrence'] = None
    
    try:
        from prism.dynamical_systems.engines import permutation_entropy
        engines['permutation_entropy'] = permutation_entropy
    except ImportError:
        try:
            import permutation_entropy
            engines['permutation_entropy'] = permutation_entropy
        except ImportError:
            engines['permutation_entropy'] = None
    
    try:
        from prism.dynamical_systems.engines import granger
        engines['granger'] = granger
    except ImportError:
        try:
            import granger
            engines['granger'] = granger
        except ImportError:
            engines['granger'] = None
    
    try:
        from prism.dynamical_systems.engines import hurst
        engines['hurst'] = hurst
    except ImportError:
        try:
            import hurst
            engines['hurst'] = hurst
        except ImportError:
            engines['hurst'] = None
    
    return engines


# =============================================================================
# State Computation (calls engines)
# =============================================================================

def compute_state(entity_id: str,
                  geometry_history: List[Dict],
                  window_idx: int,
                  timestamp: Optional[Any],
                  engines: Dict) -> DynamicsState:
    """
    Compute full dynamics state for a single window.
    
    Calls each engine and assembles the state vector.
    """
    geometry = geometry_history[window_idx] if window_idx < len(geometry_history) else {}
    
    # === TRAJECTORY (Takens embedding) ===
    if engines.get('takens'):
        result = engines['takens'].compute_from_geometry(geometry_history, window_idx)
        trajectory = result.trajectory
    else:
        trajectory = _fallback_trajectory(geometry_history, window_idx)
    
    # === STABILITY (Lyapunov exponent) ===
    # At geometry layer, we don't have raw signal - use proxy from geometry trends
    if engines.get('lyapunov'):
        # Would need raw signal - use geometry proxy instead
        stability = _stability_from_geometry(geometry_history, window_idx)
    else:
        stability = _stability_from_geometry(geometry_history, window_idx)
    
    # === ATTRACTOR (Recurrence analysis) ===
    if engines.get('recurrence'):
        result = engines['recurrence'].compute_from_geometry(geometry, trajectory)
        attractor = result.attractor
    else:
        attractor = _fallback_attractor(geometry, trajectory)
    
    # === PREDICTABILITY (Permutation entropy) ===
    if engines.get('permutation_entropy'):
        predictability = _predictability_from_geometry(geometry_history, window_idx)
    else:
        predictability = _predictability_from_geometry(geometry_history, window_idx)
    
    # === COUPLING (Granger causality) ===
    if engines.get('granger'):
        result = engines['granger'].compute_from_geometry(geometry)
        coupling = result.coupling
    else:
        coupling = _fallback_coupling(geometry)
    
    # === MEMORY (Hurst exponent) ===
    if engines.get('hurst'):
        memory = _memory_from_geometry(geometry_history, window_idx)
    else:
        memory = _memory_from_geometry(geometry_history, window_idx)
    
    return DynamicsState(
        entity_id=entity_id,
        window_idx=window_idx,
        timestamp=timestamp,
        trajectory=trajectory,
        attractor=attractor,
        stability=stability,
        predictability=predictability,
        coupling=coupling,
        memory=memory
    )


# =============================================================================
# Geometry-Based Proxies (when raw signal unavailable)
# =============================================================================

def _fallback_trajectory(geometry_history: List[Dict], window_idx: int) -> str:
    """Estimate trajectory from geometry trends."""
    if window_idx < 2:
        return "stationary"
    
    lookback = min(5, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    if len(recent) < 2:
        return "stationary"
    
    correlations = [g.get("mean_correlation", 0) for g in recent]
    densities = [g.get("network_density", 0) for g in recent]
    
    x = np.arange(len(correlations))
    corr_slope = np.polyfit(x, correlations, 1)[0] if len(correlations) > 1 else 0
    density_slope = np.polyfit(x, densities, 1)[0] if len(densities) > 1 else 0
    corr_var = np.var(correlations) if len(correlations) > 1 else 0
    
    if corr_var > 0.05:
        return "chaotic"
    elif corr_slope > 0.02 and density_slope > 0.02:
        return "converging"
    elif corr_slope < -0.02 and density_slope < -0.02:
        return "diverging"
    else:
        return "stationary"


def _stability_from_geometry(geometry_history: List[Dict], window_idx: int) -> float:
    """Estimate stability from geometry evolution."""
    if window_idx < 3:
        return 0.5
    
    lookback = min(10, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0.5) for g in recent]
    
    if len(correlations) < 3:
        return 0.5
    
    # Compute successive differences
    diffs = np.diff(correlations)
    
    # Growth rate of perturbations
    growth_rates = []
    for i in range(1, len(diffs)):
        if abs(diffs[i-1]) > 0.001:
            growth_rates.append(diffs[i] / diffs[i-1])
    
    if not growth_rates:
        return 0.5
    
    avg_growth = np.mean(growth_rates)
    stability = -np.tanh(avg_growth - 1)
    
    return float(np.clip(stability, -1, 1))


def _fallback_attractor(geometry: Dict, trajectory: str) -> str:
    """Estimate attractor from geometry and trajectory."""
    mean_corr = geometry.get("mean_correlation", 0.5)
    silhouette = geometry.get("silhouette_score", 0.0)
    n_clusters = geometry.get("n_clusters", 1)
    
    if trajectory == "chaotic":
        return "strange"
    elif trajectory == "periodic":
        return "limit_cycle"
    elif trajectory == "converging" and abs(mean_corr) > 0.8:
        return "fixed_point"
    elif silhouette > 0.5:
        return "limit_cycle"
    else:
        return "none"


def _predictability_from_geometry(geometry_history: List[Dict], window_idx: int) -> float:
    """Estimate predictability from geometry pattern entropy."""
    if window_idx < 5:
        return 0.5
    
    lookback = min(20, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0) for g in recent]
    
    if len(correlations) < 5:
        return 0.5
    
    # Simplified permutation entropy on geometry series
    pattern_counts = {}
    for i in range(len(correlations) - 2):
        triplet = correlations[i:i+3]
        pattern = tuple(np.argsort(triplet))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    if not pattern_counts:
        return 0.5
    
    total = sum(pattern_counts.values())
    probs = [c / total for c in pattern_counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    
    max_entropy = np.log(6)  # 3! patterns
    normalized = entropy / max_entropy
    
    return float(1 - normalized)


def _fallback_coupling(geometry: Dict) -> float:
    """Estimate coupling from geometry correlation/density."""
    mean_corr = abs(geometry.get("mean_correlation", 0.5))
    density = geometry.get("network_density", 0.5)
    
    coupling = 0.7 * mean_corr + 0.3 * density
    return float(np.clip(coupling, 0, 1))


def _memory_from_geometry(geometry_history: List[Dict], window_idx: int) -> float:
    """Estimate memory (Hurst) from geometry trends."""
    if window_idx < 10:
        return 0.5
    
    lookback = min(50, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0.5) for g in recent]
    
    if len(correlations) < 10:
        return 0.5
    
    # Simplified R/S analysis
    series = np.array(correlations)
    mean = np.mean(series)
    deviations = series - mean
    cumulative = np.cumsum(deviations)
    
    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(series)
    
    if S < 1e-6:
        return 0.5
    
    RS = R / S
    n = len(series)
    
    if RS > 0 and n > 1:
        H = np.log(RS) / np.log(n)
        H = np.clip(H, 0, 1)
    else:
        H = 0.5
    
    return float(H)


# =============================================================================
# Transition Detection
# =============================================================================

NUMERIC_THRESHOLDS = {
    "stability": 0.2,
    "predictability": 0.15,
    "coupling": 0.15,
    "memory": 0.1
}


def detect_transitions(prev_state: DynamicsState, 
                       curr_state: DynamicsState) -> List[DynamicsTransition]:
    """Detect meaningful transitions between consecutive states."""
    transitions = []
    
    # Categorical fields
    for field in ["trajectory", "attractor"]:
        prev_val = getattr(prev_state, field)
        curr_val = getattr(curr_state, field)
        
        if prev_val != curr_val:
            severity = _classify_categorical_severity(field, prev_val, curr_val)
            transitions.append(DynamicsTransition(
                entity_id=curr_state.entity_id,
                window_idx=curr_state.window_idx,
                timestamp=curr_state.timestamp,
                field=field,
                from_value=str(prev_val),
                to_value=str(curr_val),
                delta=None,
                transition_type="shift",
                severity=severity
            ))
    
    # Numeric fields
    for field, threshold in NUMERIC_THRESHOLDS.items():
        prev_val = getattr(prev_state, field)
        curr_val = getattr(curr_state, field)
        delta = curr_val - prev_val
        
        if abs(delta) > threshold:
            transition_type, severity = _classify_numeric_transition(
                field, prev_val, curr_val, delta, threshold
            )
            transitions.append(DynamicsTransition(
                entity_id=curr_state.entity_id,
                window_idx=curr_state.window_idx,
                timestamp=curr_state.timestamp,
                field=field,
                from_value=f"{prev_val:.3f}",
                to_value=f"{curr_val:.3f}",
                delta=delta,
                transition_type=transition_type,
                severity=severity
            ))
    
    return transitions


def _classify_categorical_severity(field: str, prev: str, curr: str) -> str:
    """Classify severity of categorical change."""
    severe_transitions = {
        ("trajectory", "converging", "chaotic"),
        ("trajectory", "stationary", "chaotic"),
        ("trajectory", "periodic", "chaotic"),
        ("attractor", "fixed_point", "strange"),
        ("attractor", "limit_cycle", "strange"),
        ("attractor", "fixed_point", "none"),
        ("attractor", "limit_cycle", "none"),
    }
    
    if (field, prev, curr) in severe_transitions:
        return "severe"
    
    if field == "attractor":
        return "moderate"
    
    return "mild"


def _classify_numeric_transition(field: str, prev: float, curr: float,
                                  delta: float, threshold: float) -> tuple:
    """Classify numeric transition type and severity."""
    abs_delta = abs(delta)
    
    # Stability sign change = bifurcation
    if field == "stability" and prev * curr < 0:
        return "bifurcation", "severe"
    
    # Memory crossing 0.5
    if field == "memory" and (prev - 0.5) * (curr - 0.5) < 0:
        return "flip", "moderate"
    
    # Determine type
    if field in ["predictability", "coupling", "stability"]:
        transition_type = "collapse" if delta < 0 else "recovery"
    else:
        transition_type = "shift"
    
    # Severity
    if abs_delta > 3 * threshold:
        severity = "severe"
    elif abs_delta > 2 * threshold:
        severity = "moderate"
    else:
        severity = "mild"
    
    return transition_type, severity


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute Dynamical Systems (orchestrator)")
    parser.add_argument("--force", action="store_true", help="Recompute all")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--entity", type=str, default=None, help="[TESTING] Only process specific entity")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    # Try new name first, fall back to old
    geometry_path = data_dir / "manifold_geometry.parquet"
    if not geometry_path.exists():
        geometry_path = data_dir / "structural_geometry.parquet"
    
    states_path = data_dir / "dynamics_states.parquet"
    transitions_path = data_dir / "dynamics_transitions.parquet"

    # Check for geometry input
    if not geometry_path.exists():
        print(f"ERROR: {geometry_path} not found")
        print("Run manifold_geometry first:")
        print("  python -m prism.entry_points.manifold_geometry")
        sys.exit(1)

    print(f"Loading geometry from {geometry_path}")
    df = pl.read_parquet(geometry_path)
    print(f"  {len(df):,} geometry windows loaded")

    # Get unique entities
    entity_ids = df["entity_id"].unique().sort().to_list()
    print(f"  {len(entity_ids)} unique entities")

    # Testing mode filters
    if args.entity:
        if not args.testing:
            print("ERROR: --entity requires --testing flag")
            sys.exit(1)
        entity_ids = [e for e in entity_ids if e in args.entity.split(",")]

    # Import engines
    print("Loading engines...")
    engines = _import_engines()
    loaded = [k for k, v in engines.items() if v is not None]
    missing = [k for k, v in engines.items() if v is None]
    print(f"  Loaded: {loaded}")
    if missing:
        print(f"  Missing (using fallbacks): {missing}")

    all_states = []
    all_transitions = []

    for entity_id in entity_ids:
        # Get geometry history for this entity (sorted by window)
        entity_df = df.filter(pl.col("entity_id") == entity_id).sort("window_idx")
        n_windows = len(entity_df)

        if n_windows < 2:
            print(f"  {entity_id}: SKIP (only {n_windows} window)")
            continue

        # Build geometry history as list of dicts
        geometry_history = []
        timestamps = []

        for row in entity_df.iter_rows(named=True):
            geometry_history.append({
                "mean_correlation": row.get("mean_correlation", 0.0),
                "network_density": row.get("network_density", 0.0),
                "n_clusters": row.get("n_clusters", 1),
                "n_signals": row.get("n_signals", 0),
                "silhouette_score": row.get("silhouette_score", 0.0),
                "n_hubs": row.get("n_hubs", 0),
                "n_decoupled_pairs": row.get("n_decoupled_pairs", 0),
                "n_causal_pairs": row.get("n_causal_pairs", 0),
                "n_bidirectional": row.get("n_bidirectional", 0),
                "topology_class": row.get("topology_class", ""),
                "stability_class": row.get("stability_class", ""),
                "curvature_forman": row.get("curvature_forman", 0.0),
                "curvature_ollivier": row.get("curvature_ollivier", 0.0),
            })
            timestamps.append(row.get("timestamp"))

        # Compute state at each window
        prev_state = None
        entity_transitions = []
        
        for w in range(n_windows):
            state = compute_state(
                entity_id=entity_id,
                geometry_history=geometry_history,
                window_idx=w,
                timestamp=timestamps[w] if w < len(timestamps) else None,
                engines=engines
            )
            all_states.append(asdict(state))
            
            # Detect transitions
            if prev_state is not None:
                transitions = detect_transitions(prev_state, state)
                entity_transitions.extend(transitions)
            
            prev_state = state
        
        # Add entity transitions to global list
        for t in entity_transitions:
            all_transitions.append(asdict(t))
        
        # Summary
        n_transitions = len(entity_transitions)
        n_severe = sum(1 for t in entity_transitions if t.severity == "severe")
        print(f"  {entity_id}: {n_windows} windows → {n_transitions} transitions ({n_severe} severe)")

    # Convert to DataFrames
    if all_states:
        states_df = pl.DataFrame(all_states)
        print(f"\nWriting {len(states_df)} states to {states_path}")
        states_df.write_parquet(states_path)
    else:
        print("\nNo states computed")
        return

    if all_transitions:
        transitions_df = pl.DataFrame(all_transitions)
        print(f"Writing {len(transitions_df)} transitions to {transitions_path}")
        transitions_df.write_parquet(transitions_path)
    else:
        print("No transitions detected (stable system)")
        transitions_df = pl.DataFrame({
            "entity_id": [], "window_idx": [], "timestamp": [],
            "field": [], "from_value": [], "to_value": [],
            "delta": [], "transition_type": [], "severity": []
        })
        transitions_df.write_parquet(transitions_path)

    # Summary
    print("\n" + "=" * 60)
    print("DYNAMICAL SYSTEMS COMPLETE")
    print("=" * 60)
    print(f"  Entities processed: {len(entity_ids)}")
    print(f"  Total states: {len(all_states)}")
    print(f"  Total transitions: {len(all_transitions)}")
    
    if all_transitions:
        severity_counts = {}
        for t in all_transitions:
            sev = t["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print("\nTransition Severity:")
        for sev in ["mild", "moderate", "severe"]:
            print(f"  {sev}: {severity_counts.get(sev, 0)}")
        
        type_counts = {}
        for t in all_transitions:
            tt = t["transition_type"]
            type_counts[tt] = type_counts.get(tt, 0) + 1
        
        print("\nTransition Types:")
        for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {tt}: {count}")
        
        bifurcations = [t for t in all_transitions 
                       if t["transition_type"] == "bifurcation" or t["severity"] == "severe"]
        if bifurcations:
            escalate_entities = set(t["entity_id"] for t in bifurcations)
            print(f"\nEscalate to mechanics: {len(escalate_entities)} entities")


if __name__ == "__main__":
    main()
