"""
PRISM Testing Module

Synthetic physics data with KNOWN parameters for engine validation.

Usage:
    # Generate all test data
    python -m prism.testing.generate

    # Validate physics engines
    python -m prism.testing.validate

    # Generate specific level
    from prism.testing.generate import generate_spring_mass_damper
    df, ground_truth = generate_spring_mass_damper()

Curriculum Levels:
    Level 0: Raw time series (statistics, entropy, memory, spectral)
    Level 2: Mechanical systems (kinetic/potential energy, hamiltonian, momentum)
    Level 3: Thermodynamic processes (gibbs free energy, enthalpy)
    Level 4: Velocity fields (Navier-Stokes, TKE, energy spectrum)
"""

from .generate import (
    generate_level0_random_walk,
    generate_level0_oscillatory,
    generate_spring_mass_damper,
    generate_pendulum,
    generate_ideal_gas_process,
    generate_polytropic_process,
    generate_synthetic_turbulence,
    generate_channel_flow,
    generate_all_test_data,
    SpringMassDamperParams,
)

__all__ = [
    'generate_level0_random_walk',
    'generate_level0_oscillatory',
    'generate_spring_mass_damper',
    'generate_pendulum',
    'generate_ideal_gas_process',
    'generate_polytropic_process',
    'generate_synthetic_turbulence',
    'generate_channel_flow',
    'generate_all_test_data',
    'SpringMassDamperParams',
]
