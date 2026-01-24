"""
Signal States Models
====================

Core dataclasses for the unified signal state architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib


@dataclass
class SignalState:
    """
    Unified signal state at a specific window.

    Tracks a signal through all four ORTHON analytical layers:
    - Typology: What IS this signal? (9-axis profile)
    - Geometry: How are signals RELATED? (topology/stability/leadership)
    - Dynamics: How is the system EVOLVING? (regime/stability/trajectory/attractor)
    - Mechanics: What are the PHYSICS? (energy/equilibrium/flow/orbit)

    Primary key: (signal_id, unit_id, window_idx)
    """

    # === IDENTIFICATION ===
    signal_id: str = ""
    unit_id: str = ""
    window_idx: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # === LAYER STATES (dot-delimited enum values) ===
    typology_state: str = ""      # "persistent|periodic|clustered"
    geometry_state: str = ""      # "MODULAR.STABLE.CLEAR_LEADER"
    dynamics_state: str = ""      # "COUPLED.EVOLVING.CONVERGING.FIXED_POINT"
    mechanics_state: str = ""     # "CONSERVATIVE.APPROACHING.LAMINAR.CIRCULAR"

    # === VALIDATION ===
    mechanics_stable: bool = True  # Mechanics should be stable across time
    stability_notes: str = ""      # Explanation if not stable

    # === METADATA ===
    state_hash: str = ""           # Hash for change detection
    computed_at: datetime = field(default_factory=datetime.now)

    def compute_hash(self) -> str:
        """Compute hash from all state strings for change detection."""
        combined = f"{self.typology_state}|{self.geometry_state}|{self.dynamics_state}|{self.mechanics_state}"
        self.state_hash = hashlib.md5(combined.encode()).hexdigest()[:16]
        return self.state_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'unit_id': self.unit_id,
            'window_idx': self.window_idx,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'typology_state': self.typology_state,
            'geometry_state': self.geometry_state,
            'dynamics_state': self.dynamics_state,
            'mechanics_state': self.mechanics_state,
            'mechanics_stable': self.mechanics_stable,
            'stability_notes': self.stability_notes,
            'state_hash': self.state_hash,
            'computed_at': self.computed_at.isoformat() if isinstance(self.computed_at, datetime) else str(self.computed_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalState":
        """Create SignalState from dictionary."""
        state = cls()
        state.signal_id = data.get('signal_id', '')
        state.unit_id = data.get('unit_id', '')
        state.window_idx = data.get('window_idx', 0)

        ts = data.get('timestamp')
        if isinstance(ts, str):
            state.timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            state.timestamp = ts

        state.typology_state = data.get('typology_state', '')
        state.geometry_state = data.get('geometry_state', '')
        state.dynamics_state = data.get('dynamics_state', '')
        state.mechanics_state = data.get('mechanics_state', '')
        state.mechanics_stable = data.get('mechanics_stable', True)
        state.stability_notes = data.get('stability_notes', '')
        state.state_hash = data.get('state_hash', '')

        ca = data.get('computed_at')
        if isinstance(ca, str):
            state.computed_at = datetime.fromisoformat(ca)
        elif isinstance(ca, datetime):
            state.computed_at = ca

        return state


@dataclass
class StateTransition:
    """
    Records a change in signal state between windows.

    Used for tracking regime changes and validating mechanics stability.
    """

    signal_id: str = ""
    unit_id: str = ""
    from_window: int = 0
    to_window: int = 0

    # Which layers changed
    typology_changed: bool = False
    geometry_changed: bool = False
    dynamics_changed: bool = False
    mechanics_changed: bool = False

    # Previous states
    prev_typology: str = ""
    prev_geometry: str = ""
    prev_dynamics: str = ""
    prev_mechanics: str = ""

    # New states
    new_typology: str = ""
    new_geometry: str = ""
    new_dynamics: str = ""
    new_mechanics: str = ""

    # Validation
    is_expected: bool = True       # Was this transition expected?
    alert_level: str = "info"      # "info" | "warning" | "critical"
    explanation: str = ""

    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'unit_id': self.unit_id,
            'from_window': self.from_window,
            'to_window': self.to_window,
            'typology_changed': self.typology_changed,
            'geometry_changed': self.geometry_changed,
            'dynamics_changed': self.dynamics_changed,
            'mechanics_changed': self.mechanics_changed,
            'prev_typology': self.prev_typology,
            'prev_geometry': self.prev_geometry,
            'prev_dynamics': self.prev_dynamics,
            'prev_mechanics': self.prev_mechanics,
            'new_typology': self.new_typology,
            'new_geometry': self.new_geometry,
            'new_dynamics': self.new_dynamics,
            'new_mechanics': self.new_mechanics,
            'is_expected': self.is_expected,
            'alert_level': self.alert_level,
            'explanation': self.explanation,
            'detected_at': self.detected_at.isoformat() if isinstance(self.detected_at, datetime) else str(self.detected_at),
        }
