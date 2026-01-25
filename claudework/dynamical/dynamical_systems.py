#!/usr/bin/env python3
"""
Dynamical Systems Entry Point
=============================

Analyzes temporal evolution of manifold geometry windows.

Pipeline: signals → signal_typology → manifold_geometry → dynamical_systems → causal_mechanics

Input:
    - data/manifold_geometry.parquet (geometry per window)

Output:
    - data/dynamics_states.parquet (6 metrics per entity per window)
    - data/dynamics_transitions.parquet (only when state changes)

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
# Metric Computation
# =============================================================================

def compute_trajectory(geometry_history: List[Dict], window_idx: int) -> str:
    """
    Classify trajectory based on geometry evolution.
    
    Uses correlation and density trends to determine direction.
    """
    if window_idx < 2:
        return "stationary"
    
    # Look at recent trend (last 3-5 windows)
    lookback = min(5, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    if len(recent) < 2:
        return "stationary"
    
    # Extract correlation trend
    correlations = [g.get("mean_correlation", 0) for g in recent]
    densities = [g.get("network_density", 0) for g in recent]
    
    # Compute slopes
    x = np.arange(len(correlations))
    corr_slope = np.polyfit(x, correlations, 1)[0] if len(correlations) > 1 else 0
    density_slope = np.polyfit(x, densities, 1)[0] if len(densities) > 1 else 0
    
    # Variance in recent windows (chaos indicator)
    corr_var = np.var(correlations) if len(correlations) > 1 else 0
    
    # Classification logic
    if corr_var > 0.05:  # High variance = chaotic
        return "chaotic"
    elif corr_slope > 0.02 and density_slope > 0.02:
        return "converging"
    elif corr_slope < -0.02 and density_slope < -0.02:
        return "diverging"
    elif abs(corr_slope) < 0.01 and corr_var < 0.01:
        # Check for periodicity
        if _detect_periodicity(correlations):
            return "periodic"
        return "stationary"
    else:
        return "stationary"


def _detect_periodicity(values: List[float], threshold: float = 0.7) -> bool:
    """Simple periodicity detection via autocorrelation."""
    if len(values) < 6:
        return False
    
    values = np.array(values)
    values = values - np.mean(values)
    
    # Autocorrelation at lag 2-3
    for lag in [2, 3]:
        if len(values) > lag:
            autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if abs(autocorr) > threshold:
                return True
    return False


def compute_stability(geometry_history: List[Dict], window_idx: int) -> float:
    """
    Compute stability index based on geometry evolution.
    
    Approximates Lyapunov exponent from correlation dynamics.
    Returns: -1 to 1, where >0 is stable, <0 is unstable.
    """
    if window_idx < 3:
        return 0.5  # Neutral
    
    lookback = min(10, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    # Track "perturbation growth" via correlation changes
    correlations = [g.get("mean_correlation", 0.5) for g in recent]
    
    if len(correlations) < 3:
        return 0.5
    
    # Compute successive differences (proxy for perturbation evolution)
    diffs = np.diff(correlations)
    
    # Growth rate of perturbations
    growth_rates = []
    for i in range(1, len(diffs)):
        if abs(diffs[i-1]) > 0.001:
            growth_rates.append(diffs[i] / diffs[i-1])
    
    if not growth_rates:
        return 0.5
    
    # Average growth rate -> Lyapunov-like exponent
    avg_growth = np.mean(growth_rates)
    
    # Normalize to -1, 1 using tanh
    stability = -np.tanh(avg_growth - 1)  # Centered at growth=1
    
    return float(np.clip(stability, -1, 1))


def compute_attractor(geometry_history: List[Dict], window_idx: int,
                      trajectory: str) -> str:
    """
    Classify attractor type based on long-term behavior.
    """
    if window_idx < 5:
        return "none"
    
    lookback = min(20, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0) for g in recent]
    n_clusters = [g.get("n_clusters", 1) for g in recent]
    
    if len(correlations) < 5:
        return "none"
    
    corr_var = np.var(correlations)
    corr_mean = np.mean(correlations)
    cluster_var = np.var(n_clusters)
    
    # Classification
    if trajectory == "chaotic" or corr_var > 0.1:
        return "strange"
    elif trajectory == "periodic" or (cluster_var < 0.5 and corr_var < 0.02):
        return "limit_cycle"
    elif trajectory == "converging" and corr_var < 0.01:
        return "fixed_point"
    else:
        return "none"


def compute_predictability(geometry_history: List[Dict], window_idx: int) -> float:
    """
    Compute predictability via permutation entropy approximation.
    
    Returns: 0 (random) to 1 (deterministic)
    """
    if window_idx < 5:
        return 0.5
    
    lookback = min(20, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0) for g in recent]
    
    if len(correlations) < 5:
        return 0.5
    
    # Simplified permutation entropy
    # Count ordinal patterns in triplets
    pattern_counts = {}
    for i in range(len(correlations) - 2):
        triplet = correlations[i:i+3]
        # Get rank order
        pattern = tuple(np.argsort(triplet))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    if not pattern_counts:
        return 0.5
    
    # Compute entropy
    total = sum(pattern_counts.values())
    probs = [c / total for c in pattern_counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    
    # Normalize by max entropy (log(6) for 3! patterns)
    max_entropy = np.log(6)
    normalized_entropy = entropy / max_entropy
    
    # Predictability = 1 - normalized entropy
    return float(1 - normalized_entropy)


def compute_coupling(geometry_history: List[Dict], window_idx: int) -> float:
    """
    Compute coupling strength from geometry.
    
    Uses mean correlation as proxy for coupling.
    """
    if window_idx < 0 or window_idx >= len(geometry_history):
        return 0.5
    
    g = geometry_history[window_idx]
    
    # Use mean correlation directly (already 0-1 ish)
    mean_corr = abs(g.get("mean_correlation", 0.5))
    
    # Also factor in network density
    density = g.get("network_density", 0.5)
    
    # Combined coupling metric
    coupling = 0.7 * mean_corr + 0.3 * density
    
    return float(np.clip(coupling, 0, 1))


def compute_memory(geometry_history: List[Dict], window_idx: int) -> float:
    """
    Compute memory (persistence) via Hurst exponent approximation.
    
    Returns: 0-1 where 0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent
    """
    if window_idx < 10:
        return 0.5
    
    lookback = min(50, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    correlations = [g.get("mean_correlation", 0.5) for g in recent]
    
    if len(correlations) < 10:
        return 0.5
    
    # Simplified R/S analysis for Hurst exponent
    series = np.array(correlations)
    n = len(series)
    
    # Compute rescaled range
    mean = np.mean(series)
    deviations = series - mean
    cumulative = np.cumsum(deviations)
    
    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(series)
    
    if S < 1e-6:
        return 0.5
    
    RS = R / S
    
    # Hurst exponent approximation: H = log(R/S) / log(n)
    if RS > 0 and n > 1:
        H = np.log(RS) / np.log(n)
        H = np.clip(H, 0, 1)
    else:
        H = 0.5
    
    return float(H)


def compute_state(entity_id: str, 
                  geometry_history: List[Dict],
                  window_idx: int,
                  timestamp: Optional[Any] = None) -> DynamicsState:
    """
    Compute full dynamics state for a single window.
    """
    trajectory = compute_trajectory(geometry_history, window_idx)
    stability = compute_stability(geometry_history, window_idx)
    attractor = compute_attractor(geometry_history, window_idx, trajectory)
    predictability = compute_predictability(geometry_history, window_idx)
    coupling = compute_coupling(geometry_history, window_idx)
    memory = compute_memory(geometry_history, window_idx)
    
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
# Transition Detection
# =============================================================================

# Thresholds for "meaningful" change
NUMERIC_THRESHOLDS = {
    "stability": 0.2,
    "predictability": 0.15,
    "coupling": 0.15,
    "memory": 0.1
}


def detect_transitions(prev_state: DynamicsState, 
                       curr_state: DynamicsState) -> List[DynamicsTransition]:
    """
    Detect meaningful transitions between consecutive states.
    """
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
    
    # Severe transitions
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
    
    # Moderate: any other change in attractor
    if field == "attractor":
        return "moderate"
    
    return "mild"


def _classify_numeric_transition(field: str, prev: float, curr: float,
                                  delta: float, threshold: float) -> tuple:
    """Classify numeric transition type and severity."""
    
    abs_delta = abs(delta)
    
    # Special case: stability sign change = bifurcation
    if field == "stability" and prev * curr < 0:
        return "bifurcation", "severe"
    
    # Special case: memory crossing 0.5
    if field == "memory" and (prev - 0.5) * (curr - 0.5) < 0:
        return "flip", "moderate"
    
    # Determine type based on direction
    if field in ["predictability", "coupling", "stability"]:
        if delta < 0:
            transition_type = "collapse"
        else:
            transition_type = "recovery"
    else:
        transition_type = "shift"
    
    # Severity based on magnitude
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
    parser = argparse.ArgumentParser(description="Compute Dynamical Systems (state + transitions)")
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
                "topology_class": row.get("topology_class", ""),
                "stability_class": row.get("stability_class", ""),
                # New curvature fields if available
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
                timestamp=timestamps[w] if w < len(timestamps) else None
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
        
        # Summary for this entity
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
        # Write empty transitions file
        transitions_df = pl.DataFrame({
            "entity_id": [],
            "window_idx": [],
            "timestamp": [],
            "field": [],
            "from_value": [],
            "to_value": [],
            "delta": [],
            "transition_type": [],
            "severity": []
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
        # Severity distribution
        severity_counts = {}
        for t in all_transitions:
            sev = t["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print("\nTransition Severity:")
        for sev in ["mild", "moderate", "severe"]:
            print(f"  {sev}: {severity_counts.get(sev, 0)}")
        
        # Type distribution
        type_counts = {}
        for t in all_transitions:
            tt = t["transition_type"]
            type_counts[tt] = type_counts.get(tt, 0) + 1
        
        print("\nTransition Types:")
        for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {tt}: {count}")
        
        # Escalation candidates
        bifurcations = [t for t in all_transitions 
                       if t["transition_type"] == "bifurcation" or t["severity"] == "severe"]
        if bifurcations:
            escalate_entities = set(t["entity_id"] for t in bifurcations)
            print(f"\nEscalate to mechanics: {len(escalate_entities)} entities")


if __name__ == "__main__":
    main()
