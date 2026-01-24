"""
Causal Mechanics Models
=======================

Dataclasses and enums for Causal Mechanics framework output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class EnergyClass(Enum):
    """Energy regime classification (from Hamiltonian)"""
    CONSERVATIVE = "conservative"   # H is constant - closed system
    DRIVEN = "driven"               # H increasing - energy injection
    DISSIPATIVE = "dissipative"     # H decreasing - energy loss
    FLUCTUATING = "fluctuating"     # H varies irregularly


class EquilibriumClass(Enum):
    """Equilibrium tendency classification (from Gibbs)"""
    APPROACHING = "approaching"       # Moving toward equilibrium (dG < 0)
    AT_EQUILIBRIUM = "at_equilibrium" # At equilibrium (G stable)
    DEPARTING = "departing"           # Moving away from equilibrium
    FORCED = "forced"                 # Externally driven (non-spontaneous)


class FlowClass(Enum):
    """Flow regime classification (from momentum flux)"""
    LAMINAR = "laminar"             # Smooth, predictable flow
    TRANSITIONAL = "transitional"   # Between laminar and turbulent
    TURBULENT = "turbulent"         # Chaotic, unpredictable flow


class OrbitClass(Enum):
    """Phase space orbit classification (from angular momentum)"""
    CIRCULAR = "circular"           # Constant |L|, perfect oscillation
    ELLIPTICAL = "elliptical"       # Varying |L|, regular oscillation
    IRREGULAR = "irregular"         # Complex orbit shape
    LINEAR = "linear"               # No rotation, L ~ 0


class DominanceClass(Enum):
    """Energy dominance classification"""
    KINETIC = "kinetic"             # Motion dominates
    POTENTIAL = "potential"         # Position dominates
    BALANCED = "balanced"           # Equal contribution


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class MechanicsVector:
    """
    Numerical measurements from causal mechanics.
    This is the DATA output - consumed by downstream frameworks.
    """

    # === IDENTIFICATION ===
    entity_id: str = "unknown"
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)
    signal_id: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === HAMILTONIAN (Energy) ===
    H_mean: float = 0.0               # Mean total energy
    H_std: float = 0.0                # Energy variability
    H_trend: float = 0.0              # dH/dt (energy injection rate)
    H_cv: float = 0.0                 # Coefficient of variation
    T_mean: float = 0.0               # Mean kinetic energy
    V_mean: float = 0.0               # Mean potential energy
    T_V_ratio: float = 1.0            # Kinetic / Potential ratio
    energy_conserved: bool = True     # Is H approximately constant?

    # === LAGRANGIAN (Motion) ===
    L_mean: float = 0.0               # Mean Lagrangian
    action: float = 0.0               # Total action (integral L dt)
    action_rate: float = 0.0          # Action per unit time
    kinetic_dominant_fraction: float = 0.5  # Fraction of time T > V

    # === GIBBS (Equilibrium) ===
    G_mean: float = 0.0               # Mean Gibbs free energy
    G_trend: float = 0.0              # dG/dt
    delta_G: float = 0.0              # G_final - G_initial
    temperature_mean: float = 1.0     # Mean "temperature" (volatility)
    entropy_mean: float = 0.0         # Mean entropy
    spontaneous: bool = False         # Is dG < 0?

    # === ANGULAR MOMENTUM (Cycles) ===
    angular_L_mean: float = 0.0       # Mean angular momentum (signed)
    angular_L_abs_mean: float = 0.0   # Mean |L|
    sign_change_rate: float = 0.0     # Direction reversals per period
    orbit_circularity: float = 0.0    # 0 = linear, 1 = circular
    orbit_stability: float = 0.0      # Consistency of |L|
    L_conserved: bool = True          # Is angular momentum conserved?

    # === MOMENTUM FLUX (Flow) ===
    p_mean: float = 0.0               # Mean momentum
    p_std: float = 0.0                # Momentum variability
    flux_mean: float = 0.0            # Mean dp/dt
    flux_std: float = 0.0             # Force variability
    reynolds_proxy: float = 1.0       # Inertial / viscous ratio
    turbulence_intensity: float = 0.0 # Flux variability measure
    inertial: bool = False            # Momentum persistent?
    viscous: bool = False             # Drag present?
    forced: bool = False              # External forcing?
    turbulent: bool = False           # Turbulent regime?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        # Ensure unit_id is set
        if not result.get('unit_id') and result.get('entity_id'):
            result['unit_id'] = result['entity_id']
        return result


@dataclass
class MechanicsTypology:
    """
    Classification from causal mechanics.
    This is the INFORMATION output - human-interpretable.
    """

    # === IDENTIFICATION ===
    entity_id: str = "unknown"
    unit_id: str = ""  # New: unit_id (defaults to entity_id if not set)
    signal_id: str = "unknown"

    def __post_init__(self):
        # unit_id defaults to entity_id for backwards compatibility
        if not self.unit_id and self.entity_id:
            self.unit_id = self.entity_id

    # === ENERGY REGIME ===
    energy_class: EnergyClass = EnergyClass.CONSERVATIVE
    dominant_energy: DominanceClass = DominanceClass.BALANCED

    # === MOTION REGIME ===
    motion_class: DominanceClass = DominanceClass.BALANCED

    # === EQUILIBRIUM REGIME ===
    equilibrium_class: EquilibriumClass = EquilibriumClass.AT_EQUILIBRIUM
    spontaneous: bool = False

    # === CYCLICAL REGIME ===
    rotation_direction: str = "mixed"    # counterclockwise | clockwise | mixed
    orbit_class: OrbitClass = OrbitClass.LINEAR

    # === FLOW REGIME ===
    flow_class: FlowClass = FlowClass.LAMINAR
    forcing_type: str = "mixed"          # inertial | viscous | forced | mixed

    # === COMPOSITE ===
    system_class: str = "Unknown"        # Overall system classification

    # === HUMAN-READABLE ===
    summary: str = ""
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'unit_id': self.unit_id if self.unit_id else self.entity_id,
            'signal_id': self.signal_id,
            'energy_class': self.energy_class.value,
            'dominant_energy': self.dominant_energy.value,
            'motion_class': self.motion_class.value,
            'equilibrium_class': self.equilibrium_class.value,
            'spontaneous': self.spontaneous,
            'rotation_direction': self.rotation_direction,
            'orbit_class': self.orbit_class.value,
            'flow_class': self.flow_class.value,
            'forcing_type': self.forcing_type,
            'system_class': self.system_class,
            'summary': self.summary,
            'alerts': self.alerts,
            'confidence': self.confidence
        }


@dataclass
class CausalMechanicsOutput:
    """Complete output from Causal Mechanics framework"""
    vector: MechanicsVector
    typology: MechanicsTypology

    def to_dict(self) -> Dict[str, Any]:
        """Combined dictionary for full output."""
        result = self.vector.to_dict()
        result.update(self.typology.to_dict())
        return result


# Backwards compatibility aliases
PhysicsVector = MechanicsVector
PhysicsTypology = MechanicsTypology
SystemPhysicsOutput = CausalMechanicsOutput
