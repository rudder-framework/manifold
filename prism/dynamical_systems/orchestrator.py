"""
Dynamical Systems Orchestrator
==============================

One of four ORTHON analytical frameworks: How does the SYSTEM evolve?

Analyzes temporal evolution of structural geometry:
    - Regime: Coupled / Decoupled / Transitioning
    - Stability: Stable / Evolving / Unstable / Critical
    - Trajectory: Converging / Diverging / Oscillating / Wandering
    - Attractor: Fixed point / Limit cycle / Strange / None

Architecture:
    dynamical_systems/
        orchestrator.py (this file - routes + formats)
            │
            ▼
        models.py (DynamicsVector, DynamicsTypology)
            │
            ▼
        engines/* (computations)
            │
            ▼
        engine_mapping.py (selects engines)

Usage:
    from prism.dynamical_systems import run_dynamical_systems

    results = run_dynamical_systems(geometry_history, timestamps)
    print(results['regime'])
    print(results['engine_recommendations'])
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .models import (
    RegimeClass,
    StabilityClass,
    TrajectoryClass,
    AttractorClass,
    DynamicsVector,
    DynamicsTypology,
    DynamicalSystemsOutput,
)
from .engine_mapping import (
    select_engines,
    get_regime_classification,
    get_stability_classification,
    get_trajectory_classification,
    should_escalate_to_mechanics,
)


# Dimension names (canonical order)
DIMENSION_NAMES = [
    'regime',
    'stability',
    'trajectory',
    'attractor',
]


def run_dynamical_systems(
    geometry_history: List[Dict[str, float]],
    timestamps: Optional[List[datetime]] = None,
    entity_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for Dynamical Systems analysis.

    This is a PURE ORCHESTRATOR - it only:
    1. Routes to DynamicalSystemsFramework for classification
    2. Routes to engine_mapping for engine selection
    3. Assembles the output structure

    For per-window state computation with transitions, use:
        python -m prism.entry_points.dynamical_systems

    Args:
        geometry_history: List of geometry measurements over time
            Each dict should contain: mean_correlation, network_density, n_clusters, etc.
        timestamps: Optional timestamps for each measurement
        entity_id: Entity identifier
        config: Optional configuration overrides

    Returns:
        {
            'regime': current regime classification,
            'stability': current stability classification,
            'trajectory': trajectory classification,
            'attractor': attractor classification,
            'vector': DynamicsVector as dict,
            'typology': DynamicsTypology as dict,
            'engine_recommendations': [engines],
            'escalate_to_mechanics': bool,
            'transition_history': [...],
            'metadata': {...}
        }
    """
    config = config or {}

    if len(geometry_history) < 2:
        return _empty_result(entity_id, "Insufficient data (need >= 2 geometry snapshots)")

    # Route to framework for classification (pure orchestration)
    framework = DynamicalSystemsFramework(entity_id=entity_id, config=config)
    output = framework.analyze(geometry_history, timestamps)

    # Get classification strings from typology
    regime = output.typology.regime_class.value.upper()
    stability = output.typology.stability_class.value.upper()
    trajectory = output.typology.trajectory_class.value.upper()
    attractor = output.typology.attractor_class.value.upper()

    # Build state for engine selection (structure only)
    state = {
        'regime': regime,
        'stability': stability,
        'trajectory': trajectory,
        'attractor': attractor,
        'correlation_change': output.vector.correlation_change,
        'density_change': output.vector.density_change,
    }

    # Route to engine_mapping for engine selection
    engines = select_engines(state)

    # Route to engine_mapping for escalation check
    escalate = should_escalate_to_mechanics(state)

    # Build simple transition history from geometry sequence
    # (Detailed transitions are in dynamics_transitions.parquet from entry point)
    transition_history = _build_simple_transition_history(geometry_history)

    return {
        'regime': regime,
        'stability': stability,
        'trajectory': trajectory,
        'attractor': attractor,
        'vector': output.vector.to_dict(),
        'typology': output.typology.to_dict(),
        'engine_recommendations': engines,
        'escalate_to_mechanics': escalate,
        'transition_history': transition_history,
        'metadata': {
            'entity_id': entity_id,
            'n_observations': len(geometry_history),
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


class DynamicalSystemsFramework:
    """
    Dynamical Systems Framework: How does the SYSTEM evolve?

    Pure orchestrator. Routes to engine_mapping for classification.
    Contains ZERO computation - only coordination and classification logic.

    NOTE: For per-window state computation, use the entry point:
        python -m prism.entry_points.dynamical_systems

    This framework provides legacy summary-style output for compatibility.

    Answers:
        - Regime: What coupling state is the system in?
        - Stability: Is the system stable or transitioning?
        - Trajectory: Where is the system heading?
        - Attractor: What states does it tend toward?
    """

    def __init__(
        self,
        entity_id: str = "",
        config: Optional[Dict] = None,
    ):
        self.entity_id = entity_id
        self.config = config or {}

    def analyze(
        self,
        geometry_history: List[Dict[str, float]],
        timestamps: Optional[List[datetime]] = None,
    ) -> DynamicalSystemsOutput:
        """
        Analyze dynamical systems from geometry history.

        This is a pure orchestrator - it only routes to classification
        functions in engine_mapping.py. No computation happens here.

        Args:
            geometry_history: List of geometry measurements over time
            timestamps: Optional timestamps for each measurement

        Returns:
            DynamicalSystemsOutput with vector and typology
        """
        if len(geometry_history) < 2:
            return self._empty_output()

        # Extract values from geometry history (no computation, just extraction)
        correlations = [g.get('mean_correlation', 0.0) for g in geometry_history]
        densities = [g.get('network_density', 0.0) for g in geometry_history]

        # Get current and recent values
        current_corr = correlations[-1]
        recent_corr_change = correlations[-1] - correlations[-2] if len(correlations) >= 2 else 0.0
        recent_density_change = densities[-1] - densities[-2] if len(densities) >= 2 else 0.0

        # Route to engine_mapping for classification (pure routing, no computation)
        regime_str = get_regime_classification(current_corr, recent_corr_change)
        regime_class = RegimeClass(regime_str.lower())

        stability_str = get_stability_classification(recent_corr_change, recent_density_change)
        stability_class = StabilityClass(stability_str.lower())

        trajectory_str = get_trajectory_classification(correlations, densities)
        trajectory_class = TrajectoryClass(trajectory_str.lower())

        # Build output structures (assembly only)
        vector = DynamicsVector(
            timestamp=timestamps[-1] if timestamps else datetime.now(),
            entity_id=self.entity_id,
            correlation_level=current_corr,
            correlation_change=recent_corr_change,
            density_change=recent_density_change,
            # Note: stability_index and trajectory_speed computed by engine_mapping
            stability_index=0.5,  # Default, engine_mapping provides classification
            trajectory_speed=0.0,  # Default, engine_mapping provides classification
        )

        typology = DynamicsTypology(
            entity_id=self.entity_id,
            regime_class=regime_class,
            stability_class=stability_class,
            trajectory_class=trajectory_class,
            summary=self._generate_summary(regime_class, stability_class, trajectory_class),
            confidence=self._estimate_confidence(len(geometry_history)),
        )

        return DynamicalSystemsOutput(
            vector=vector,
            typology=typology,
            metadata={'n_observations': len(geometry_history)},
        )

    def _estimate_confidence(self, n_observations: int) -> float:
        """Estimate confidence based on data quantity (no computation)."""
        if n_observations < 5:
            return 0.3
        elif n_observations < 10:
            return 0.5
        elif n_observations < 20:
            return 0.7
        else:
            return 0.9

    def _generate_summary(
        self,
        regime: RegimeClass,
        stability: StabilityClass,
        trajectory: TrajectoryClass,
    ) -> str:
        """Generate human-readable summary (string formatting only)."""
        return f"{regime.value.title()} regime, {stability.value} system, {trajectory.value} trajectory"

    def _empty_output(self) -> DynamicalSystemsOutput:
        """Return empty output for insufficient data."""
        return DynamicalSystemsOutput(
            vector=DynamicsVector(entity_id=self.entity_id),
            typology=DynamicsTypology(entity_id=self.entity_id, summary="Insufficient data"),
            metadata={'error': 'insufficient_data'},
        )


def analyze_dynamics(
    geometry_history: List[Dict[str, float]],
    entity_id: str = "",
    timestamps: Optional[List[datetime]] = None,
    config: Optional[Dict] = None,
) -> DynamicalSystemsOutput:
    """
    Convenience function for dynamical systems analysis.

    Args:
        geometry_history: List of geometry measurements
        entity_id: Entity identifier
        timestamps: Optional timestamps
        config: Optional configuration

    Returns:
        DynamicalSystemsOutput
    """
    framework = DynamicalSystemsFramework(entity_id=entity_id, config=config)
    return framework.analyze(geometry_history, timestamps)


def analyze_single_entity(
    geometry_df,
    entity_id: str = "",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing a single entity from a DataFrame.

    Args:
        geometry_df: DataFrame with geometry columns
        entity_id: Entity identifier
        config: Optional configuration

    Returns:
        Dict with dynamics analysis results
    """
    geometry_history = []

    for row in geometry_df.iter_rows(named=True):
        geometry_history.append({
            'mean_correlation': row.get('mean_correlation', 0.0),
            'network_density': row.get('network_density', 0.0),
            'n_clusters': row.get('n_clusters', 0),
            'n_signals': row.get('n_signals', 0),
        })

    timestamps = None
    if 'timestamp' in geometry_df.columns:
        timestamps = geometry_df['timestamp'].to_list()

    return run_dynamical_systems(geometry_history, timestamps, entity_id, config)


def get_trajectory_fingerprint(state: Dict[str, Any]) -> np.ndarray:
    """
    Convert dynamical state to a fingerprint vector.

    Args:
        state: Dict with regime, stability, trajectory, attractor

    Returns:
        numpy array encoding the state
    """
    regime_map = {'COUPLED': 1.0, 'MODERATE': 0.5, 'DECOUPLED': 0.0, 'TRANSITIONING': 0.75}
    stability_map = {'STABLE': 1.0, 'EVOLVING': 0.66, 'UNSTABLE': 0.33, 'CRITICAL': 0.0}
    trajectory_map = {'CONVERGING': 1.0, 'OSCILLATING': 0.66, 'WANDERING': 0.33, 'DIVERGING': 0.0}
    attractor_map = {'FIXED_POINT': 1.0, 'LIMIT_CYCLE': 0.75, 'STRANGE': 0.25, 'NONE': 0.5}

    return np.array([
        regime_map.get(state.get('regime', '').upper(), 0.5),
        stability_map.get(state.get('stability', '').upper(), 0.5),
        trajectory_map.get(state.get('trajectory', '').upper(), 0.5),
        attractor_map.get(state.get('attractor', '').upper(), 0.5),
    ])


def trajectory_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute distance between two trajectory fingerprints."""
    return float(np.linalg.norm(fp1 - fp2))


def detect_regime_transition(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Detect regime transition between two states."""
    fp_prev = get_trajectory_fingerprint(previous_state)
    fp_curr = get_trajectory_fingerprint(current_state)

    distance = trajectory_distance(fp_prev, fp_curr)

    changes = {}
    for dim in DIMENSION_NAMES:
        changes[dim] = {
            'previous': previous_state.get(dim, ''),
            'current': current_state.get(dim, ''),
            'changed': previous_state.get(dim, '') != current_state.get(dim, ''),
        }

    changed_dims = [d for d, c in changes.items() if c['changed']]

    if not changed_dims:
        transition_type = 'NONE'
    elif 'regime' in changed_dims:
        transition_type = 'REGIME_SHIFT'
    elif 'stability' in changed_dims:
        transition_type = 'STABILITY_CHANGE'
    elif 'trajectory' in changed_dims:
        transition_type = 'TRAJECTORY_SHIFT'
    else:
        transition_type = 'ATTRACTOR_CHANGE'

    return {
        'transition_detected': distance >= threshold,
        'transition_type': transition_type,
        'distance': distance,
        'threshold': threshold,
        'changes': changes,
        'changed_dimensions': changed_dims,
    }


def _build_simple_transition_history(
    geometry_history: List[Dict[str, float]],
) -> List[Dict]:
    """
    Build simple transition history by routing each window to engine_mapping.

    For detailed transitions with severity classification, use:
        python -m prism.entry_points.dynamical_systems
    """
    if len(geometry_history) < 2:
        return []

    history = []

    for i in range(1, len(geometry_history)):
        current = geometry_history[i]
        previous = geometry_history[i - 1]

        # Extract values (no computation, just extraction)
        corr = current.get('mean_correlation', 0.0)
        corr_change = corr - previous.get('mean_correlation', 0.0)
        density_change = current.get('network_density', 0.0) - previous.get('network_density', 0.0)

        # Route to engine_mapping for classification
        regime = get_regime_classification(corr, corr_change)
        stability = get_stability_classification(corr_change, density_change)

        history.append({
            'index': i,
            'regime': regime,
            'stability': stability,
            'correlation': corr,
            'correlation_change': corr_change,
            'density_change': density_change,
        })

    return history


# Keep old function name for backwards compatibility
def _build_transition_history(
    correlations: List[float],
    densities: List[float],
    corr_changes: np.ndarray,
    density_changes: np.ndarray,
) -> List[Dict]:
    """Legacy function - converts to new format."""
    geometry_history = [
        {'mean_correlation': c, 'network_density': d}
        for c, d in zip(correlations, densities)
    ]
    return _build_simple_transition_history(geometry_history)


def _empty_result(entity_id: str, error: str) -> Dict[str, Any]:
    """Return empty result for insufficient data."""
    return {
        'regime': 'UNDETERMINED',
        'stability': 'UNDETERMINED',
        'trajectory': 'UNDETERMINED',
        'attractor': 'UNDETERMINED',
        'vector': {},
        'typology': {},
        'engine_recommendations': [],
        'escalate_to_mechanics': False,
        'transition_history': [],
        'metadata': {
            'entity_id': entity_id,
            'error': error,
            'version': '1.0.0',
            'computed_at': datetime.now().isoformat(),
        }
    }


# Backwards compatibility
DynamicalSystemsLayer = DynamicalSystemsFramework
