"""Geometry group — system state, eigendecomposition, health scoring."""
from manifold.stages.geometry.state_vector import run as run_state_vector
from manifold.stages.geometry.state_geometry import run as run_state_geometry
from manifold.stages.geometry.signal_geometry import run as run_signal_geometry
from manifold.stages.geometry.geometry_dynamics import run as run_geometry_dynamics
from manifold.stages.geometry.sensor_eigendecomp import run as run_sensor_eigendecomp
from manifold.stages.geometry.cohort_baseline import run as run_cohort_baseline
from manifold.stages.geometry.observation_geometry import run as run_observation_geometry
