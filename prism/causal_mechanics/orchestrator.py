"""
Causal Mechanics Orchestrator
=============================

One of four ORTHON analytical frameworks: What DRIVES the system?

The Five Physics-Inspired Dimensions:
    1. Energy       - Is energy conserved? (Hamiltonian)
    2. Motion       - What are the equations of motion? (Lagrangian)
    3. Equilibrium  - Spontaneous or forced? (Gibbs free energy)
    4. Cycles       - What are the rotational dynamics? (Angular momentum)
    5. Flow         - How does momentum propagate? (Momentum flux)

Architecture:
    causal_mechanics/
        orchestrator.py (this file - routes + formats)
            │
            ▼
        models.py (MechanicsVector, MechanicsTypology)
            │
            ▼
        engines/physics/* (computations)
            │
            ▼
        engine_mapping.py (action recommendations)

Usage:
    from prism.causal_mechanics import run_causal_mechanics, analyze_mechanics

    results = run_causal_mechanics(signal_data, entity_id='unit_1')
    print(results['energy_class'])
    print(results['intervention_recommendations'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .models import (
    EnergyClass,
    EquilibriumClass,
    FlowClass,
    OrbitClass,
    DominanceClass,
    MechanicsVector,
    MechanicsTypology,
    CausalMechanicsOutput,
)
from .engine_mapping import (
    select_engines,
    get_energy_classification,
    get_equilibrium_classification,
    get_flow_classification,
    get_intervention_recommendations,
)


# Dimension names (canonical order)
DIMENSION_NAMES = [
    'energy',
    'motion',
    'equilibrium',
    'cycles',
    'flow',
]


class CausalMechanicsFramework:
    """
    Causal Mechanics Framework: What DRIVES the system?

    Pure orchestrator. Calls physics engines, classifies results.
    Contains ZERO computation - only coordination and classification logic.

    Answers:
        - Is energy conserved? (Hamiltonian)
        - What type of motion? (Lagrangian)
        - Is it spontaneously equilibrating? (Gibbs)
        - What are the cyclical dynamics? (Angular momentum)
        - How does momentum flow? (Momentum flux)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def analyze(
        self,
        series: np.ndarray,
        entity_id: str = "unknown",
        signal_id: str = "unknown",
        previous: Optional[CausalMechanicsOutput] = None
    ) -> CausalMechanicsOutput:
        """
        Analyze causal mechanics of a signal.

        Args:
            series: 1D numpy array
            entity_id: Entity identifier
            signal_id: Signal identifier
            previous: Previous window output (for transition detection)

        Returns:
            CausalMechanicsOutput with vector and typology
        """
        # Import engines (deferred to avoid circular imports)
        from ..engines.physics import (
            hamiltonian, lagrangian, kinetic_energy, potential_energy,
            gibbs_free_energy, angular_momentum, momentum_flux
        )

        series = np.asarray(series).flatten()

        # === CALL ALL ENGINES ===
        H_result = hamiltonian.compute(series)
        L_result = lagrangian.compute(series)
        KE_result = kinetic_energy.compute(series)
        PE_result = potential_energy.compute(series)
        G_result = gibbs_free_energy.compute(series)
        AM_result = angular_momentum.compute(series)
        MF_result = momentum_flux.compute(series)

        # === BUILD VECTOR ===
        vector = self._build_vector(
            entity_id, signal_id,
            H_result, L_result, KE_result, PE_result,
            G_result, AM_result, MF_result
        )

        # === CLASSIFY ===
        typology = self._classify(
            entity_id, signal_id, vector,
            H_result, L_result, G_result, AM_result, MF_result
        )

        return CausalMechanicsOutput(vector=vector, typology=typology)

    def _build_vector(
        self, entity_id, signal_id,
        H, L, KE, PE, G, AM, MF
    ) -> MechanicsVector:
        """Assemble MechanicsVector from engine outputs."""

        return MechanicsVector(
            entity_id=entity_id,
            signal_id=signal_id,
            timestamp=datetime.now(),

            # Hamiltonian
            H_mean=H.H_mean,
            H_std=H.H_std,
            H_trend=H.H_trend,
            H_cv=H.H_cv,
            T_mean=H.T_mean,
            V_mean=H.V_mean,
            T_V_ratio=H.T_V_ratio,
            energy_conserved=H.conserved,

            # Lagrangian
            L_mean=L.L_mean,
            action=L.action,
            action_rate=L.action_rate,
            kinetic_dominant_fraction=L.kinetic_dominant_fraction,

            # Gibbs
            G_mean=G.G_mean,
            G_trend=G.G_trend,
            delta_G=G.delta_G,
            temperature_mean=G.T_mean,
            entropy_mean=G.S_mean,
            spontaneous=G.spontaneous,

            # Angular momentum
            angular_L_mean=AM.L_mean,
            angular_L_abs_mean=AM.L_abs_mean,
            sign_change_rate=AM.sign_change_rate,
            orbit_circularity=AM.orbit_circularity,
            orbit_stability=AM.orbit_stability,
            L_conserved=AM.L_conserved,

            # Momentum flux
            p_mean=MF.p_mean,
            p_std=MF.p_std,
            flux_mean=MF.flux_mean,
            flux_std=MF.flux_std,
            reynolds_proxy=MF.reynolds_proxy,
            turbulence_intensity=MF.turbulence_intensity,
            inertial=MF.inertial,
            viscous=MF.viscous,
            forced=MF.forced,
            turbulent=MF.turbulent
        )

    def _classify(
        self, entity_id, signal_id, vector,
        H, L, G, AM, MF
    ) -> MechanicsTypology:
        """Convert measurements to classification."""

        # Energy class (from Hamiltonian)
        energy_class = EnergyClass(H.regime)

        # Dominant energy
        if H.dominant_energy == "kinetic":
            dominant_energy = DominanceClass.KINETIC
        elif H.dominant_energy == "potential":
            dominant_energy = DominanceClass.POTENTIAL
        else:
            dominant_energy = DominanceClass.BALANCED

        # Motion class (from Lagrangian)
        if L.dominance == "kinetic":
            motion_class = DominanceClass.KINETIC
        elif L.dominance == "potential":
            motion_class = DominanceClass.POTENTIAL
        else:
            motion_class = DominanceClass.BALANCED

        # Equilibrium class (from Gibbs)
        equilibrium_class = EquilibriumClass(G.equilibrium_class)

        # Rotation direction
        rotation_direction = AM.rotation_direction

        # Orbit class
        if AM.orbit_circularity > 0.8:
            orbit_class = OrbitClass.CIRCULAR
        elif AM.orbit_circularity > 0.4:
            orbit_class = OrbitClass.ELLIPTICAL
        elif AM.L_abs_mean > 0.1:
            orbit_class = OrbitClass.IRREGULAR
        else:
            orbit_class = OrbitClass.LINEAR

        # Flow class
        flow_class = FlowClass(MF.flow_regime)

        # Forcing type
        forcing_type = MF.forcing_type

        # System class (composite)
        system_class = self._derive_system_class(
            energy_class, equilibrium_class, flow_class, orbit_class
        )

        # Summary and alerts
        summary, alerts = self._generate_summary(
            energy_class, equilibrium_class, flow_class, H, G
        )

        # Confidence
        confidence = self._compute_confidence(H, G, MF)

        return MechanicsTypology(
            entity_id=entity_id,
            signal_id=signal_id,
            energy_class=energy_class,
            dominant_energy=dominant_energy,
            motion_class=motion_class,
            equilibrium_class=equilibrium_class,
            spontaneous=G.spontaneous,
            rotation_direction=rotation_direction,
            orbit_class=orbit_class,
            flow_class=flow_class,
            forcing_type=forcing_type,
            system_class=system_class,
            summary=summary,
            alerts=alerts,
            confidence=confidence
        )

    def _derive_system_class(
        self, energy: EnergyClass, equilibrium: EquilibriumClass,
        flow: FlowClass, orbit: OrbitClass
    ) -> str:
        """Derive overall system classification from components."""

        # Priority-based classification
        if energy == EnergyClass.CONSERVATIVE:
            if orbit == OrbitClass.CIRCULAR:
                return "Stable Oscillator"
            elif orbit == OrbitClass.ELLIPTICAL:
                return "Quasi-Periodic"
            elif flow == FlowClass.LAMINAR:
                return "Conservative Laminar"
            else:
                return "Conservative"

        elif energy == EnergyClass.DRIVEN:
            if flow == FlowClass.TURBULENT:
                return "Turbulent Driven"
            elif equilibrium == EquilibriumClass.DEPARTING:
                return "Forced Unstable"
            else:
                return "Driven System"

        elif energy == EnergyClass.DISSIPATIVE:
            if equilibrium == EquilibriumClass.APPROACHING:
                return "Damped Equilibrating"
            else:
                return "Dissipative"

        elif flow == FlowClass.TURBULENT:
            return "Turbulent"

        elif equilibrium == EquilibriumClass.APPROACHING:
            return "Equilibrating"

        elif equilibrium == EquilibriumClass.DEPARTING:
            return "Destabilizing"

        else:
            return "Transitional"

    def _generate_summary(
        self, energy: EnergyClass, equilibrium: EquilibriumClass,
        flow: FlowClass, H, G
    ) -> tuple:
        """Generate human-readable summary and alerts."""

        alerts = []

        # Energy alerts
        if not H.conserved:
            if H.regime == "driven":
                alerts.append("Energy injection - system is being driven")
            elif H.regime == "dissipative":
                alerts.append("Energy dissipating - momentum fading")

        # Equilibrium alerts
        if equilibrium == EquilibriumClass.DEPARTING:
            alerts.append("Moving away from equilibrium - instability risk")
        elif equilibrium == EquilibriumClass.FORCED:
            alerts.append("Forced regime - unsustainable without external input")

        # Flow alerts
        if flow == FlowClass.TURBULENT:
            alerts.append("Turbulent flow - high unpredictability")

        # Gibbs spontaneity
        if G.spontaneous and G.moving_toward_equilibrium:
            alerts.append("Spontaneously equilibrating")

        summary = f"**{energy.value.title()}** | {equilibrium.value.replace('_', ' ')} | {flow.value} flow"

        return summary, alerts

    def _compute_confidence(self, H, G, MF) -> float:
        """Compute overall classification confidence."""

        # Higher confidence when measurements are clear
        energy_conf = 1.0 / (1.0 + H.H_cv) if H.H_cv < float('inf') else 0.5
        gibbs_conf = min(abs(G.G_trend) * 100 + 0.3, 1.0)
        flow_conf = 0.3 if MF.turbulent else 0.8

        confidence = (energy_conf + gibbs_conf + flow_conf) / 3

        return float(np.clip(confidence, 0, 1))


def run_causal_mechanics(
    signal: np.ndarray,
    entity_id: str = "",
    signal_id: str = "",
    dynamics_state: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Causal Mechanics analysis.

    Args:
        signal: 1D numpy array of signal values
        entity_id: Entity identifier
        signal_id: Signal identifier
        dynamics_state: Optional state from Dynamical Systems framework
        config: Optional configuration overrides

    Returns:
        {
            'energy_class': energy regime classification,
            'equilibrium_class': equilibrium state classification,
            'flow_class': flow regime classification,
            'orbit_class': orbit classification,
            'dominant_energy': kinetic vs potential,
            'system_class': overall system classification,
            'vector': MechanicsVector as dict,
            'typology': MechanicsTypology as dict,
            'engine_recommendations': [engines],
            'intervention_recommendations': {...},
            'alerts': [...],
            'metadata': {...}
        }
    """
    config = config or {}

    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=float)

    if len(signal) < 30:
        return _empty_result(entity_id, signal_id, "Insufficient data (need >= 30 points)")

    # Use framework for core analysis
    framework = CausalMechanicsFramework(config=config)
    output = framework.analyze(signal, entity_id=entity_id, signal_id=signal_id)

    # Get vector and typology
    vector = output.vector
    typology = output.typology

    # Build state for engine selection
    state = {
        'energy_class': typology.energy_class.value.upper(),
        'equilibrium_class': typology.equilibrium_class.value.upper(),
        'flow_class': typology.flow_class.value.upper(),
        'orbit_class': typology.orbit_class.value.upper(),
        'motion_class': typology.motion_class.value.upper(),
        'H_cv': vector.H_cv,
        'H_trend': vector.H_trend,
        'delta_G': vector.delta_G,
        'temperature': vector.temperature_mean,
        'spontaneous': vector.spontaneous,
        'reynolds_proxy': vector.reynolds_proxy,
        'turbulence_intensity': vector.turbulence_intensity,
    }

    # Select engines
    engines = select_engines(state)

    # Get intervention recommendations
    interventions = get_intervention_recommendations(state)

    # Include dynamics context if provided
    if dynamics_state:
        state['dynamics_regime'] = dynamics_state.get('regime', '')
        state['dynamics_stability'] = dynamics_state.get('stability', '')

    return {
        'energy_class': typology.energy_class.value.upper(),
        'equilibrium_class': typology.equilibrium_class.value.upper(),
        'flow_class': typology.flow_class.value.upper(),
        'orbit_class': typology.orbit_class.value.upper(),
        'dominant_energy': typology.dominant_energy.value.upper(),
        'motion_class': typology.motion_class.value.upper(),
        'system_class': typology.system_class,
        'vector': vector.to_dict(),
        'typology': typology.to_dict(),
        'engine_recommendations': engines,
        'intervention_recommendations': interventions,
        'alerts': typology.alerts,
        'summary': typology.summary,
        'confidence': typology.confidence,
        'metadata': {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_observations': len(signal),
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


def analyze_mechanics(
    series: np.ndarray,
    entity_id: str = "unknown",
    signal_id: str = "unknown"
) -> CausalMechanicsOutput:
    """
    Convenience function for quick causal mechanics analysis.

    Example:
        result = analyze_mechanics(my_series)
        print(result.typology.summary)
        print(result.typology.system_class)
    """
    framework = CausalMechanicsFramework()
    return framework.analyze(series, entity_id, signal_id)


def analyze_single_signal(
    signal: np.ndarray,
    entity_id: str = "",
    signal_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single signal.

    Args:
        signal: 1D numpy array
        entity_id: Entity identifier
        signal_id: Signal identifier
        config: Optional configuration

    Returns:
        Dict with mechanics analysis results
    """
    return run_causal_mechanics(signal, entity_id, signal_id, config=config)


def get_mechanics_fingerprint(state: Dict[str, Any]) -> np.ndarray:
    """
    Convert mechanics state to a fingerprint vector.

    Args:
        state: Dict with energy_class, equilibrium_class, flow_class, etc.

    Returns:
        numpy array encoding the state
    """
    # Encode each dimension as a value
    energy_map = {'CONSERVATIVE': 1.0, 'FLUCTUATING': 0.66, 'DISSIPATIVE': 0.33, 'DRIVEN': 0.0}
    equilibrium_map = {'AT_EQUILIBRIUM': 1.0, 'APPROACHING': 0.75, 'FORCED': 0.25, 'DEPARTING': 0.0}
    flow_map = {'LAMINAR': 1.0, 'TRANSITIONAL': 0.5, 'TURBULENT': 0.0}
    orbit_map = {'CIRCULAR': 1.0, 'ELLIPTICAL': 0.75, 'LINEAR': 0.5, 'IRREGULAR': 0.0}
    motion_map = {'BALANCED': 0.5, 'KINETIC': 1.0, 'POTENTIAL': 0.0}

    return np.array([
        energy_map.get(state.get('energy_class', '').upper(), 0.5),
        equilibrium_map.get(state.get('equilibrium_class', '').upper(), 0.5),
        flow_map.get(state.get('flow_class', '').upper(), 0.5),
        orbit_map.get(state.get('orbit_class', '').upper(), 0.5),
        motion_map.get(state.get('motion_class', '').upper(), 0.5),
    ])


def mechanics_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute distance between two mechanics fingerprints.

    Args:
        fp1: First fingerprint vector
        fp2: Second fingerprint vector

    Returns:
        Distance (0 = identical, ~2.2 = maximally different)
    """
    return float(np.linalg.norm(fp1 - fp2))


def detect_energy_transition(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect energy/mechanics transition between two states.

    Args:
        previous_state: Previous mechanics state
        current_state: Current mechanics state
        threshold: Change threshold for flagging transition

    Returns:
        Dict with transition detection results
    """
    fp_prev = get_mechanics_fingerprint(previous_state)
    fp_curr = get_mechanics_fingerprint(current_state)

    distance = mechanics_distance(fp_prev, fp_curr)

    # Find what changed
    changes = {}
    dim_keys = ['energy_class', 'equilibrium_class', 'flow_class', 'orbit_class', 'motion_class']

    for dim in dim_keys:
        changes[dim] = {
            'previous': previous_state.get(dim, ''),
            'current': current_state.get(dim, ''),
            'changed': previous_state.get(dim, '') != current_state.get(dim, ''),
        }

    changed_dims = [d for d, c in changes.items() if c['changed']]

    # Classify transition type
    if not changed_dims:
        transition_type = 'NONE'
    elif 'energy_class' in changed_dims:
        transition_type = 'ENERGY_TRANSITION'
    elif 'equilibrium_class' in changed_dims:
        transition_type = 'EQUILIBRIUM_SHIFT'
    elif 'flow_class' in changed_dims:
        transition_type = 'FLOW_REGIME_CHANGE'
    else:
        transition_type = 'MINOR_CHANGE'

    return {
        'transition_detected': distance >= threshold,
        'transition_type': transition_type,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'changed_dimensions': changed_dims,
    }


def classify_system(state: Dict[str, Any]) -> str:
    """
    Derive overall system classification from mechanics state.

    Args:
        state: Dict with energy_class, equilibrium_class, flow_class, orbit_class

    Returns:
        System classification string
    """
    energy = state.get('energy_class', '').upper()
    equilibrium = state.get('equilibrium_class', '').upper()
    flow = state.get('flow_class', '').upper()
    orbit = state.get('orbit_class', '').upper()

    # Priority-based classification
    if energy == 'CONSERVATIVE':
        if orbit == 'CIRCULAR':
            return 'Stable Oscillator'
        elif orbit == 'ELLIPTICAL':
            return 'Quasi-Periodic'
        elif flow == 'LAMINAR':
            return 'Conservative Laminar'
        else:
            return 'Conservative'

    elif energy == 'DRIVEN':
        if flow == 'TURBULENT':
            return 'Turbulent Driven'
        elif equilibrium == 'DEPARTING':
            return 'Forced Unstable'
        else:
            return 'Driven System'

    elif energy == 'DISSIPATIVE':
        if equilibrium == 'APPROACHING':
            return 'Damped Equilibrating'
        else:
            return 'Dissipative'

    elif flow == 'TURBULENT':
        return 'Turbulent'

    elif equilibrium == 'APPROACHING':
        return 'Equilibrating'

    elif equilibrium == 'DEPARTING':
        return 'Destabilizing'

    else:
        return 'Transitional'


def _empty_result(entity_id: str, signal_id: str, error: str) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'energy_class': 'UNDETERMINED',
        'equilibrium_class': 'UNDETERMINED',
        'flow_class': 'UNDETERMINED',
        'orbit_class': 'UNDETERMINED',
        'dominant_energy': 'UNDETERMINED',
        'motion_class': 'UNDETERMINED',
        'system_class': 'Unknown',
        'vector': {},
        'typology': {},
        'engine_recommendations': [],
        'intervention_recommendations': {'recommended_actions': [], 'timing': 'MONITOR', 'methods': [], 'urgency': 'LOW'},
        'alerts': [],
        'summary': '',
        'confidence': 0.0,
        'metadata': {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'error': error,
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


# Backwards compatibility aliases
CausalMechanicsLayer = CausalMechanicsFramework
SystemPhysicsLayer = CausalMechanicsFramework
analyze_physics = analyze_mechanics
