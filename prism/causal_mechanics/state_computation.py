"""
Causal Mechanics State Computation
==================================

Computes MechanicsState from physics engine outputs.

Takes outputs from physics engines and computes the 4-metric state:
    - energy: Energy regime (conservative, driven, dissipative, fluctuating)
    - equilibrium: Equilibrium tendency
    - flow: Flow regime (laminar, transitional, turbulent)
    - orbit: Phase space orbit type

Plus 4 numeric metrics:
    - energy_conservation: How constant is total energy?
    - equilibrium_distance: Distance from equilibrium
    - turbulence_intensity: Flow chaos level
    - orbit_stability: How consistent is the orbit?
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from prism.causal_mechanics.models import (
    MechanicsState,
    MechanicsTransition,
    MechanicsVector,
    MechanicsTypology,
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
)


def compute_state(
    entity_id: str,
    signal_id: str,
    window_idx: int,
    engine_outputs: Dict[str, Any],
    timestamp: Optional[Any] = None,
    unit_id: Optional[str] = None,
) -> MechanicsState:
    """
    Compute MechanicsState from physics engine outputs.

    Args:
        entity_id: Entity identifier
        signal_id: Signal identifier
        window_idx: Window index
        engine_outputs: Dictionary of engine name -> output dict
        timestamp: Optional timestamp
        unit_id: Optional unit identifier

    Returns:
        MechanicsState object

    Expected engine outputs:
        - "hamiltonian": {"H_mean": float, "H_std": float, "H_trend": float, ...}
        - "gibbs": {"G_mean": float, "G_trend": float, "delta_G": float, ...}
        - "momentum_flux": {"reynolds_proxy": float, "turbulence_intensity": float, ...}
        - "angular_momentum": {"orbit_circularity": float, "orbit_stability": float, ...}
        - "kinetic_energy": {"T_mean": float, ...}
        - "potential_energy": {"V_mean": float, ...}
    """

    # Extract metrics from engine outputs
    energy, energy_conservation = _compute_energy(engine_outputs)
    equilibrium, equilibrium_distance = _compute_equilibrium(engine_outputs)
    flow, turbulence_intensity = _compute_flow(engine_outputs)
    orbit, orbit_stability = _compute_orbit(engine_outputs)

    return MechanicsState(
        entity_id=entity_id,
        unit_id=unit_id or entity_id,
        signal_id=signal_id,
        window_idx=window_idx,
        timestamp=timestamp,
        energy=energy,
        equilibrium=equilibrium,
        flow=flow,
        orbit=orbit,
        energy_conservation=energy_conservation,
        equilibrium_distance=equilibrium_distance,
        turbulence_intensity=turbulence_intensity,
        orbit_stability=orbit_stability,
    )


def _compute_energy(outputs: Dict[str, Any]) -> tuple:
    """
    Compute energy regime and conservation metric.

    Returns:
        (energy_class: str, energy_conservation: float)
    """
    # Get Hamiltonian metrics
    ham = outputs.get("hamiltonian", {})
    H_mean = ham.get("H_mean", 0)
    H_std = ham.get("H_std", 0)
    H_trend = ham.get("H_trend", 0)
    H_cv = ham.get("H_cv", 0)
    energy_conserved = ham.get("energy_conserved", True)

    # Compute energy conservation metric (0-1)
    # Lower CV = better conservation
    if H_mean != 0:
        conservation = 1 - min(abs(H_cv), 1)
    else:
        conservation = 1.0 if H_std < 0.01 else 0.5

    # Classify energy regime
    if energy_conserved or H_cv < 0.05:
        energy_class = "conservative"
    elif H_trend > 0.1:
        energy_class = "driven"
    elif H_trend < -0.1:
        energy_class = "dissipative"
    else:
        energy_class = "fluctuating"

    return energy_class, float(np.clip(conservation, 0, 1))


def _compute_equilibrium(outputs: Dict[str, Any]) -> tuple:
    """
    Compute equilibrium tendency and distance metric.

    Returns:
        (equilibrium_class: str, equilibrium_distance: float)
    """
    # Get Gibbs metrics
    gibbs = outputs.get("gibbs", {})
    G_mean = gibbs.get("G_mean", 0)
    G_trend = gibbs.get("G_trend", 0)
    delta_G = gibbs.get("delta_G", 0)
    spontaneous = gibbs.get("spontaneous", False)

    # Compute equilibrium distance (0-1)
    # Based on |delta_G| - larger magnitude = farther from equilibrium
    # Normalize by typical range
    distance = min(abs(delta_G) / 10, 1.0)

    # Classify equilibrium tendency
    if abs(G_trend) < 0.01 and abs(delta_G) < 0.1:
        equilibrium_class = "at_equilibrium"
    elif G_trend < -0.05 or spontaneous:
        equilibrium_class = "approaching"
    elif G_trend > 0.05:
        equilibrium_class = "departing"
    else:
        equilibrium_class = "forced"

    return equilibrium_class, float(np.clip(distance, 0, 1))


def _compute_flow(outputs: Dict[str, Any]) -> tuple:
    """
    Compute flow regime and turbulence intensity.

    Returns:
        (flow_class: str, turbulence_intensity: float)
    """
    # Get momentum flux metrics
    flux = outputs.get("momentum_flux", {})
    reynolds = flux.get("reynolds_proxy", 1.0)
    turbulence = flux.get("turbulence_intensity", 0.0)
    is_turbulent = flux.get("turbulent", False)

    # Normalize turbulence intensity (0-1)
    turb_normalized = float(np.clip(turbulence, 0, 1))

    # Classify flow regime
    if is_turbulent or reynolds > 4000:
        flow_class = "turbulent"
    elif reynolds > 2000 or turbulence > 0.3:
        flow_class = "transitional"
    else:
        flow_class = "laminar"

    return flow_class, turb_normalized


def _compute_orbit(outputs: Dict[str, Any]) -> tuple:
    """
    Compute orbit type and stability.

    Returns:
        (orbit_class: str, orbit_stability: float)
    """
    # Get angular momentum metrics
    ang = outputs.get("angular_momentum", {})
    circularity = ang.get("orbit_circularity", 0.0)
    stability = ang.get("orbit_stability", 0.0)
    L_conserved = ang.get("L_conserved", True)

    # Classify orbit type
    if circularity > 0.8:
        orbit_class = "circular"
    elif circularity > 0.4:
        orbit_class = "elliptical"
    elif circularity > 0.1:
        orbit_class = "irregular"
    else:
        orbit_class = "linear"

    # Orbit stability (0-1)
    if L_conserved:
        orbit_stability = max(stability, 0.8)
    else:
        orbit_stability = stability

    return orbit_class, float(np.clip(orbit_stability, 0, 1))


def compute_state_from_vector(
    vector: MechanicsVector,
    window_idx: int = 0,
) -> MechanicsState:
    """
    Compute MechanicsState from a MechanicsVector.

    This is useful when you have the vector already computed.
    """
    # Energy classification
    if vector.energy_conserved:
        energy = "conservative"
        energy_conservation = 1 - min(vector.H_cv, 1)
    elif vector.H_trend > 0.1:
        energy = "driven"
        energy_conservation = 1 - min(vector.H_cv, 1)
    elif vector.H_trend < -0.1:
        energy = "dissipative"
        energy_conservation = 1 - min(vector.H_cv, 1)
    else:
        energy = "fluctuating"
        energy_conservation = 0.5

    # Equilibrium classification
    if vector.spontaneous:
        equilibrium = "approaching"
    elif abs(vector.G_trend) < 0.05:
        equilibrium = "at_equilibrium"
    elif vector.G_trend > 0:
        equilibrium = "departing"
    else:
        equilibrium = "forced"

    equilibrium_distance = min(abs(vector.delta_G) / 10, 1.0)

    # Flow classification
    if vector.turbulent:
        flow = "turbulent"
    elif vector.reynolds_proxy > 2000:
        flow = "transitional"
    else:
        flow = "laminar"

    turbulence_intensity = vector.turbulence_intensity

    # Orbit classification
    if vector.orbit_circularity > 0.8:
        orbit = "circular"
    elif vector.orbit_circularity > 0.4:
        orbit = "elliptical"
    elif vector.orbit_circularity > 0.1:
        orbit = "irregular"
    else:
        orbit = "linear"

    orbit_stability = vector.orbit_stability

    return MechanicsState(
        entity_id=vector.entity_id,
        unit_id=vector.unit_id,
        signal_id=vector.signal_id,
        window_idx=window_idx,
        timestamp=vector.timestamp,
        energy=energy,
        equilibrium=equilibrium,
        flow=flow,
        orbit=orbit,
        energy_conservation=float(np.clip(energy_conservation, 0, 1)),
        equilibrium_distance=float(np.clip(equilibrium_distance, 0, 1)),
        turbulence_intensity=float(np.clip(turbulence_intensity, 0, 1)),
        orbit_stability=float(np.clip(orbit_stability, 0, 1)),
    )


def compute_states_for_signal(
    entity_id: str,
    signal_id: str,
    window_outputs: List[Dict[str, Any]],
    timestamps: Optional[List[Any]] = None,
    unit_id: Optional[str] = None,
) -> List[MechanicsState]:
    """
    Compute MechanicsState for each window of a signal.

    Args:
        entity_id: Entity identifier
        signal_id: Signal identifier
        window_outputs: List of engine output dicts, one per window
        timestamps: Optional list of timestamps
        unit_id: Optional unit identifier

    Returns:
        List of MechanicsState objects
    """
    states = []

    for i, outputs in enumerate(window_outputs):
        ts = timestamps[i] if timestamps and i < len(timestamps) else None

        state = compute_state(
            entity_id=entity_id,
            signal_id=signal_id,
            window_idx=i,
            engine_outputs=outputs,
            timestamp=ts,
            unit_id=unit_id,
        )
        states.append(state)

    return states


def classify_energy(state: MechanicsState) -> EnergyClass:
    """Classify energy from state."""
    mapping = {
        "conservative": EnergyClass.CONSERVATIVE,
        "driven": EnergyClass.DRIVEN,
        "dissipative": EnergyClass.DISSIPATIVE,
        "fluctuating": EnergyClass.FLUCTUATING,
    }
    return mapping.get(state.energy, EnergyClass.FLUCTUATING)


def classify_equilibrium(state: MechanicsState) -> EquilibriumClass:
    """Classify equilibrium from state."""
    mapping = {
        "approaching": EquilibriumClass.APPROACHING,
        "at_equilibrium": EquilibriumClass.AT_EQUILIBRIUM,
        "departing": EquilibriumClass.DEPARTING,
        "forced": EquilibriumClass.FORCED,
    }
    return mapping.get(state.equilibrium, EquilibriumClass.AT_EQUILIBRIUM)


def classify_flow(state: MechanicsState) -> FlowClass:
    """Classify flow from state."""
    mapping = {
        "laminar": FlowClass.LAMINAR,
        "transitional": FlowClass.TRANSITIONAL,
        "turbulent": FlowClass.TURBULENT,
    }
    return mapping.get(state.flow, FlowClass.LAMINAR)


def classify_orbit(state: MechanicsState) -> OrbitClass:
    """Classify orbit from state."""
    mapping = {
        "circular": OrbitClass.CIRCULAR,
        "elliptical": OrbitClass.ELLIPTICAL,
        "irregular": OrbitClass.IRREGULAR,
        "linear": OrbitClass.LINEAR,
    }
    return mapping.get(state.orbit, OrbitClass.LINEAR)


def state_to_typology(state: MechanicsState) -> MechanicsTypology:
    """Convert MechanicsState to MechanicsTypology."""

    # Determine dominance
    if state.energy == "driven":
        dominant_energy = DominanceClass.KINETIC
    elif state.energy == "dissipative":
        dominant_energy = DominanceClass.POTENTIAL
    else:
        dominant_energy = DominanceClass.BALANCED

    # Determine system class
    system_class = _compute_system_class(state)

    return MechanicsTypology(
        entity_id=state.entity_id,
        unit_id=state.unit_id,
        signal_id=state.signal_id,
        energy_class=classify_energy(state),
        dominant_energy=dominant_energy,
        motion_class=dominant_energy,
        equilibrium_class=classify_equilibrium(state),
        spontaneous=(state.equilibrium == "approaching"),
        rotation_direction="mixed",
        orbit_class=classify_orbit(state),
        flow_class=classify_flow(state),
        forcing_type="mixed",
        system_class=system_class,
        summary=f"State: {state.state_string()}",
        confidence=0.8,
    )


def _compute_system_class(state: MechanicsState) -> str:
    """Compute overall system classification from state."""

    # Simple heuristic based on state combination
    if state.energy == "conservative" and state.flow == "laminar":
        if state.orbit == "circular":
            return "Harmonic Oscillator"
        elif state.orbit == "elliptical":
            return "Quasi-Periodic System"
        else:
            return "Conservative System"

    if state.energy == "driven":
        if state.flow == "turbulent":
            return "Turbulent Driven System"
        else:
            return "Forced System"

    if state.energy == "dissipative":
        if state.equilibrium == "approaching":
            return "Damped System"
        else:
            return "Dissipating System"

    if state.flow == "turbulent":
        return "Chaotic System"

    return "Complex System"


def validate_state(state: MechanicsState) -> Dict[str, Any]:
    """
    Validate that a MechanicsState has consistent values.

    Returns dictionary with validation results.
    """
    issues = []

    # Check categorical values
    valid_energy = ["conservative", "driven", "dissipative", "fluctuating"]
    valid_equilibrium = ["approaching", "at_equilibrium", "departing", "forced"]
    valid_flow = ["laminar", "transitional", "turbulent"]
    valid_orbit = ["circular", "elliptical", "irregular", "linear"]

    if state.energy not in valid_energy:
        issues.append(f"Invalid energy: {state.energy}")
    if state.equilibrium not in valid_equilibrium:
        issues.append(f"Invalid equilibrium: {state.equilibrium}")
    if state.flow not in valid_flow:
        issues.append(f"Invalid flow: {state.flow}")
    if state.orbit not in valid_orbit:
        issues.append(f"Invalid orbit: {state.orbit}")

    # Check numeric ranges
    if not 0 <= state.energy_conservation <= 1:
        issues.append(f"energy_conservation out of range: {state.energy_conservation}")
    if not 0 <= state.equilibrium_distance <= 1:
        issues.append(f"equilibrium_distance out of range: {state.equilibrium_distance}")
    if not 0 <= state.turbulence_intensity <= 1:
        issues.append(f"turbulence_intensity out of range: {state.turbulence_intensity}")
    if not 0 <= state.orbit_stability <= 1:
        issues.append(f"orbit_stability out of range: {state.orbit_stability}")

    # Check consistency
    if state.energy == "conservative" and state.energy_conservation < 0.5:
        issues.append("Inconsistent: conservative energy but low conservation")
    if state.flow == "turbulent" and state.turbulence_intensity < 0.3:
        issues.append("Inconsistent: turbulent flow but low turbulence intensity")
    if state.orbit == "circular" and state.orbit_stability < 0.5:
        issues.append("Inconsistent: circular orbit but low stability")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "state_string": state.state_string(),
    }
