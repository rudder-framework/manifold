"""
Dynamical Systems State Computation
===================================

Computes DynamicsState from engine outputs.

Takes outputs from various engines and computes the 6-metric state:
    - trajectory: Phase space direction
    - stability: Lyapunov-based stability
    - attractor: Attractor type
    - predictability: Entropy-based predictability
    - coupling: Correlation-based coupling
    - memory: Hurst-based memory

Each metric is normalized to its expected range.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from prism.dynamical_systems.models import (
    DynamicsState,
    DynamicsVector,
    DynamicsTypology,
    RegimeClass,
    DynamicsStabilityClass,
    TrajectoryClass,
    AttractorClass,
)


def compute_state(
    entity_id: str,
    window_idx: int,
    engine_outputs: Dict[str, Any],
    timestamp: Optional[Any] = None,
    unit_id: Optional[str] = None,
) -> DynamicsState:
    """
    Compute DynamicsState from engine outputs.

    Args:
        entity_id: Entity identifier
        window_idx: Window index
        engine_outputs: Dictionary of engine name -> output dict
        timestamp: Optional timestamp
        unit_id: Optional unit identifier

    Returns:
        DynamicsState object

    Expected engine outputs:
        - "lyapunov": {"largest_lyapunov": float, ...}
        - "entropy": {"permutation_entropy": float, ...}
        - "hurst": {"hurst_exponent": float, ...}
        - "correlation": {"mean_correlation": float, ...}
        - "granger": {"n_causal_pairs": int, ...}
        - "trajectory": {"direction": str, "speed": float, ...}
        - "attractor": {"type": str, "dimension": float, ...}
        - "embedding": {"embedding_dimension": int, ...}
        - "phase_space": {"correlation_dimension": float, ...}
    """

    # Extract metrics from engine outputs
    trajectory = _compute_trajectory(engine_outputs)
    stability = _compute_stability(engine_outputs)
    attractor = _compute_attractor(engine_outputs)
    predictability = _compute_predictability(engine_outputs)
    coupling = _compute_coupling(engine_outputs)
    memory = _compute_memory(engine_outputs)

    return DynamicsState(
        entity_id=entity_id,
        unit_id=unit_id or entity_id,
        window_idx=window_idx,
        timestamp=timestamp,
        trajectory=trajectory,
        attractor=attractor,
        stability=stability,
        predictability=predictability,
        coupling=coupling,
        memory=memory,
    )


def _compute_trajectory(outputs: Dict[str, Any]) -> str:
    """
    Compute trajectory classification.

    Maps to: converging | diverging | periodic | chaotic | stationary
    """
    # Try trajectory engine first
    traj = outputs.get("trajectory", {})
    if "direction" in traj:
        direction = traj["direction"]
        if direction in ["converging", "diverging", "periodic", "chaotic", "stationary"]:
            return direction

    # Try phase space engine
    phase = outputs.get("phase_space", {})
    attractor_type = phase.get("attractor_type", "")

    if attractor_type == "fixed_point":
        return "converging"
    elif attractor_type == "limit_cycle":
        return "periodic"
    elif attractor_type == "strange":
        return "chaotic"

    # Try Lyapunov exponent
    lyap = outputs.get("lyapunov", {})
    largest = lyap.get("largest_lyapunov", 0)

    if largest > 0.1:
        return "diverging"
    elif largest < -0.1:
        return "converging"

    # Try embedding quality
    embed = outputs.get("embedding", {})
    recurrence = embed.get("recurrence_rate", 0.5)

    if recurrence > 0.7:
        return "periodic"

    # Default
    return "stationary"


def _compute_stability(outputs: Dict[str, Any]) -> float:
    """
    Compute stability metric (-1 to 1).

    Based on Lyapunov exponent:
        - Negative exponent = stable (positive stability)
        - Positive exponent = unstable (negative stability)
    """
    # Primary: Lyapunov exponent
    lyap = outputs.get("lyapunov", {})
    largest = lyap.get("largest_lyapunov", None)

    if largest is not None and not np.isnan(largest):
        # Normalize: typical range -1 to 1
        # Negative Lyapunov = positive stability
        stability = -np.tanh(largest)
        return float(np.clip(stability, -1, 1))

    # Fallback: phase space
    phase = outputs.get("phase_space", {})
    if "orbit_stability" in phase:
        return float(phase["orbit_stability"] * 2 - 1)  # Map 0-1 to -1 to 1

    # Fallback: embedding
    embed = outputs.get("embedding", {})
    if "recurrence_rate" in embed:
        return float(embed["recurrence_rate"] * 2 - 1)

    # Fallback: correlation stability
    corr = outputs.get("correlation", {})
    if "stability_index" in corr:
        return float(corr["stability_index"] * 2 - 1)

    return 0.5  # Neutral default


def _compute_attractor(outputs: Dict[str, Any]) -> str:
    """
    Compute attractor classification.

    Maps to: fixed_point | limit_cycle | strange | none
    """
    # Try phase space engine
    phase = outputs.get("phase_space", {})
    if "attractor_type" in phase:
        att = phase["attractor_type"]
        if att in ["fixed_point", "limit_cycle", "strange", "none"]:
            return att

    # Try embedding engine
    embed = outputs.get("embedding", {})
    if "attractor_type" in embed:
        att = embed["attractor_type"]
        if att in ["fixed_point", "limit_cycle", "strange", "none"]:
            return att

    # Infer from correlation dimension
    corr_dim = phase.get("correlation_dimension") or embed.get("correlation_dimension")
    if corr_dim is not None and not np.isnan(corr_dim):
        if corr_dim < 0.5:
            return "fixed_point"
        elif corr_dim < 1.5:
            return "limit_cycle"
        elif corr_dim > 2.5:
            return "strange"

    # Try Lyapunov
    lyap = outputs.get("lyapunov", {})
    largest = lyap.get("largest_lyapunov", 0)
    if largest > 0.3:
        return "strange"
    elif largest < -0.3:
        return "fixed_point"

    return "none"


def _compute_predictability(outputs: Dict[str, Any]) -> float:
    """
    Compute predictability (0 to 1).

    Based on permutation entropy:
        - Low entropy = high predictability
        - High entropy = low predictability
    """
    # Primary: permutation entropy
    ent = outputs.get("entropy", {})
    perm_ent = ent.get("permutation_entropy", None)

    if perm_ent is not None and not np.isnan(perm_ent):
        # Permutation entropy is typically 0-1 (normalized)
        # Invert: high entropy = low predictability
        return float(np.clip(1 - perm_ent, 0, 1))

    # Try sample entropy
    samp_ent = ent.get("sample_entropy", None)
    if samp_ent is not None and not np.isnan(samp_ent):
        # Sample entropy typically 0-3+
        # Normalize and invert
        return float(np.clip(1 - samp_ent / 3, 0, 1))

    # Fallback: RQA determinism
    rqa = outputs.get("rqa", {})
    det = rqa.get("determinism", None)
    if det is not None:
        return float(np.clip(det, 0, 1))

    # Fallback: embedding quality
    embed = outputs.get("embedding", {})
    if "determinism" in embed:
        return float(np.clip(embed["determinism"], 0, 1))

    return 0.5  # Neutral default


def _compute_coupling(outputs: Dict[str, Any]) -> float:
    """
    Compute coupling strength (0 to 1).

    Based on correlation and Granger causality.
    """
    # Primary: mean correlation
    corr = outputs.get("correlation", {})
    mean_corr = corr.get("mean_correlation", None)

    if mean_corr is not None and not np.isnan(mean_corr):
        # Correlation is -1 to 1, but coupling uses absolute value
        coupling = abs(mean_corr)
        return float(np.clip(coupling, 0, 1))

    # Try Granger causality
    granger = outputs.get("granger", {})
    if "coupling_strength" in granger:
        return float(np.clip(granger["coupling_strength"], 0, 1))

    # Fallback: network density
    network = outputs.get("network", {})
    density = network.get("density", None)
    if density is not None:
        return float(np.clip(density, 0, 1))

    # Fallback: mutual information
    mi = outputs.get("mutual_information", {})
    if "normalized_mi" in mi:
        return float(np.clip(mi["normalized_mi"], 0, 1))

    return 0.5  # Neutral default


def _compute_memory(outputs: Dict[str, Any]) -> float:
    """
    Compute memory (0 to 1).

    Based on Hurst exponent:
        - H = 0.5: random walk (no memory)
        - H > 0.5: persistent (positive memory)
        - H < 0.5: anti-persistent (negative memory)
    """
    # Primary: Hurst exponent
    hurst = outputs.get("hurst", {})
    H = hurst.get("hurst_exponent", None) or hurst.get("H", None)

    if H is not None and not np.isnan(H):
        # Hurst is typically 0-1
        return float(np.clip(H, 0, 1))

    # Try DFA
    dfa = outputs.get("dfa", {})
    if "alpha" in dfa:
        # DFA alpha is similar to Hurst
        return float(np.clip(dfa["alpha"], 0, 1))

    # Fallback: ACF decay
    acf = outputs.get("acf", {})
    if "decay_rate" in acf:
        # Slow decay = more memory
        # Normalize: typical decay is 0-1
        return float(np.clip(1 - acf["decay_rate"], 0, 1))

    # Fallback: spectral slope
    spectral = outputs.get("spectral", {})
    if "spectral_slope" in spectral:
        # Steeper negative slope = more memory
        slope = spectral["spectral_slope"]
        # Typical range: -2 to 0
        # Map to 0-1
        return float(np.clip((abs(slope) / 2), 0, 1))

    return 0.5  # Neutral default (random walk)


def compute_state_from_vector(
    vector: DynamicsVector,
    window_idx: int = 0,
) -> DynamicsState:
    """
    Compute DynamicsState from a DynamicsVector.

    This is useful when you have the vector already computed.
    """
    # Map vector metrics to state metrics
    # Trajectory from correlation change
    if vector.correlation_change > 0.1:
        trajectory = "converging"
    elif vector.correlation_change < -0.1:
        trajectory = "diverging"
    else:
        trajectory = "stationary"

    # Stability from stability index
    stability = vector.stability_index * 2 - 1  # Map 0-1 to -1 to 1

    # Attractor from trajectory direction
    if vector.trajectory_curvature > 0.5:
        attractor = "limit_cycle"
    elif vector.trajectory_speed > 0.5:
        attractor = "strange"
    else:
        attractor = "none"

    # Predictability (placeholder - needs entropy input)
    predictability = 0.5

    # Coupling from correlation level
    coupling = abs(vector.correlation_level)

    # Memory (placeholder - needs Hurst input)
    memory = 0.5

    return DynamicsState(
        entity_id=vector.entity_id,
        unit_id=vector.unit_id,
        window_idx=window_idx,
        timestamp=vector.timestamp,
        trajectory=trajectory,
        attractor=attractor,
        stability=float(np.clip(stability, -1, 1)),
        predictability=predictability,
        coupling=float(np.clip(coupling, 0, 1)),
        memory=memory,
    )


def compute_states_for_entity(
    entity_id: str,
    window_outputs: List[Dict[str, Any]],
    timestamps: Optional[List[Any]] = None,
    unit_id: Optional[str] = None,
) -> List[DynamicsState]:
    """
    Compute DynamicsState for each window of an entity.

    Args:
        entity_id: Entity identifier
        window_outputs: List of engine output dicts, one per window
        timestamps: Optional list of timestamps
        unit_id: Optional unit identifier

    Returns:
        List of DynamicsState objects
    """
    states = []

    for i, outputs in enumerate(window_outputs):
        ts = timestamps[i] if timestamps and i < len(timestamps) else None

        state = compute_state(
            entity_id=entity_id,
            window_idx=i,
            engine_outputs=outputs,
            timestamp=ts,
            unit_id=unit_id,
        )
        states.append(state)

    return states


def classify_regime(state: DynamicsState) -> RegimeClass:
    """Classify regime from state metrics."""
    if state.coupling > 0.6:
        return RegimeClass.COUPLED
    elif state.coupling < 0.3:
        return RegimeClass.DECOUPLED
    else:
        return RegimeClass.MODERATE


def classify_stability(state: DynamicsState) -> DynamicsStabilityClass:
    """Classify stability from state metrics."""
    if state.stability > 0.3:
        return DynamicsStabilityClass.STABLE
    elif state.stability > 0:
        return DynamicsStabilityClass.EVOLVING
    elif state.stability > -0.3:
        return DynamicsStabilityClass.UNSTABLE
    else:
        return DynamicsStabilityClass.CRITICAL


def classify_trajectory(state: DynamicsState) -> TrajectoryClass:
    """Classify trajectory from state."""
    mapping = {
        "converging": TrajectoryClass.CONVERGING,
        "diverging": TrajectoryClass.DIVERGING,
        "periodic": TrajectoryClass.OSCILLATING,
        "chaotic": TrajectoryClass.WANDERING,
        "stationary": TrajectoryClass.CONVERGING,
    }
    return mapping.get(state.trajectory, TrajectoryClass.WANDERING)


def classify_attractor(state: DynamicsState) -> AttractorClass:
    """Classify attractor from state."""
    mapping = {
        "fixed_point": AttractorClass.FIXED_POINT,
        "limit_cycle": AttractorClass.LIMIT_CYCLE,
        "strange": AttractorClass.STRANGE,
        "none": AttractorClass.NONE,
    }
    return mapping.get(state.attractor, AttractorClass.NONE)


def state_to_typology(state: DynamicsState) -> DynamicsTypology:
    """Convert DynamicsState to DynamicsTypology."""
    return DynamicsTypology(
        entity_id=state.entity_id,
        unit_id=state.unit_id,
        regime_class=classify_regime(state),
        stability_class=classify_stability(state),
        trajectory_class=classify_trajectory(state),
        attractor_class=classify_attractor(state),
        summary=f"State: {state.state_string()}",
        confidence=0.8,
    )
