#!/usr/bin/env python3
"""
PRISM Physics Engine Validator

Validates physics engines against synthetic data with KNOWN parameters.

Usage:
    python -m prism.testing.validate
    python -m prism.testing.validate --engine kinetic_energy
    python -m prism.testing.validate --level 2
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse

# Local data path
TEST_DATA_DIR = Path(__file__).parent


@dataclass
class ValidationResult:
    """Result from validating an engine."""
    engine: str
    passed: bool
    max_error: float
    mean_error: float
    n_samples: int
    message: str


def validate_kinetic_energy(
    velocity: np.ndarray,
    mass: float,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate kinetic energy: T = ½mv²"""
    from prism.engines.physics.kinetic_energy import compute

    result = compute(velocity, mass=mass)
    computed = result['kinetic_energy']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='kinetic_energy',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(velocity),
        message=f"T = ½mv² | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_potential_energy(
    position: np.ndarray,
    spring_constant: float,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate potential energy: V = ½kx²"""
    from prism.engines.physics.potential_energy import compute

    result = compute(position, spring_constant=spring_constant)
    computed = result['potential_energy']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='potential_energy',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(position),
        message=f"V = ½kx² | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_hamiltonian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    spring_constant: float,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate Hamiltonian: H = T + V"""
    from prism.engines.physics.hamiltonian import compute

    result = compute(position, velocity, mass=mass, spring_constant=spring_constant)
    computed = result['hamiltonian']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='hamiltonian',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(position),
        message=f"H = T + V | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_lagrangian(
    position: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    spring_constant: float,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate Lagrangian: L = T - V"""
    from prism.engines.physics.lagrangian import compute

    result = compute(position, velocity, mass=mass, spring_constant=spring_constant)
    computed = result['lagrangian']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='lagrangian',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(position),
        message=f"L = T - V | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_momentum(
    velocity: np.ndarray,
    mass: float,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate momentum: p = mv"""
    from prism.engines.physics.momentum import compute

    result = compute(velocity, mass=mass)
    computed = result['momentum']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='momentum',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(velocity),
        message=f"p = mv | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_gibbs_free_energy(
    enthalpy: np.ndarray,
    temperature: np.ndarray,
    entropy: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate Gibbs free energy: G = H - TS"""
    from prism.engines.physics.gibbs_free_energy import compute

    result = compute(temperature=temperature, enthalpy=enthalpy, entropy=entropy)
    computed = result['gibbs_free_energy']

    error = np.abs(computed - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)

    passed = np.allclose(computed, expected, rtol=rtol)

    return ValidationResult(
        engine='gibbs_free_energy',
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        n_samples=len(enthalpy),
        message=f"G = H - TS | max_err={max_error:.2e}" if passed else f"FAILED: max_err={max_error:.2e}",
    )


def validate_level2_spring_mass_damper() -> List[ValidationResult]:
    """
    Validate physics engines against spring-mass-damper test data.

    Ground truth parameters:
        mass = 2.0 kg
        spring_constant = 50.0 N/m
    """
    try:
        import polars as pl
        df = pl.read_parquet(TEST_DATA_DIR / 'level2_spring_mass_damper.parquet')
        get_col = lambda col: df[col].to_numpy()
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(TEST_DATA_DIR / 'level2_spring_mass_damper.parquet')
        get_col = lambda col: df[col].to_numpy()

    data_path = TEST_DATA_DIR / 'level2_spring_mass_damper.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")

    # Known parameters
    mass = 2.0  # kg
    spring_constant = 50.0  # N/m

    # Extract signals (use true values for validation)
    position = get_col('position_true')
    velocity = get_col('velocity_true')

    results = []

    # Kinetic energy
    results.append(validate_kinetic_energy(
        velocity=velocity,
        mass=mass,
        expected=get_col('kinetic_energy_true'),
    ))

    # Potential energy
    results.append(validate_potential_energy(
        position=position,
        spring_constant=spring_constant,
        expected=get_col('potential_energy_true'),
    ))

    # Hamiltonian
    results.append(validate_hamiltonian(
        position=position,
        velocity=velocity,
        mass=mass,
        spring_constant=spring_constant,
        expected=get_col('hamiltonian_true'),
    ))

    # Lagrangian
    results.append(validate_lagrangian(
        position=position,
        velocity=velocity,
        mass=mass,
        spring_constant=spring_constant,
        expected=get_col('lagrangian_true'),
    ))

    # Momentum
    results.append(validate_momentum(
        velocity=velocity,
        mass=mass,
        expected=get_col('momentum_true'),
    ))

    return results


def validate_level3_thermodynamic(process: str = 'isothermal') -> List[ValidationResult]:
    """
    Validate Gibbs free energy against thermodynamic test data.
    """
    data_path = TEST_DATA_DIR / f'level3_{process}.parquet'

    if not data_path.exists():
        # Generate if not exists
        from .generate import generate_ideal_gas_process
        df, _ = generate_ideal_gas_process(process_type=process)
        get_col = lambda col: df[col].to_numpy()
    else:
        try:
            import polars as pl
            df = pl.read_parquet(data_path)
            get_col = lambda col: df[col].to_numpy()
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(data_path)
            get_col = lambda col: df[col].to_numpy()

    results = []

    # Gibbs free energy
    results.append(validate_gibbs_free_energy(
        enthalpy=get_col('enthalpy_J'),
        temperature=get_col('temperature_K'),
        entropy=get_col('entropy_J_K'),
        expected=get_col('gibbs_free_energy_J'),
    ))

    return results


def run_all_validations() -> Dict[str, List[ValidationResult]]:
    """Run all physics engine validations."""
    results = {}

    print("\n" + "=" * 60)
    print("PRISM PHYSICS ENGINE VALIDATION")
    print("=" * 60)

    # Level 2: Spring-Mass-Damper
    print("\n--- Level 2: Spring-Mass-Damper ---")
    print("Parameters: m=2.0 kg, k=50.0 N/m")

    try:
        level2_results = validate_level2_spring_mass_damper()
        results['level2'] = level2_results

        for r in level2_results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {status} {r.engine:20} {r.message}")

    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    except ImportError as e:
        print(f"  [SKIP] Missing engine: {e}")

    # Level 3: Thermodynamic
    print("\n--- Level 3: Thermodynamic ---")

    try:
        level3_results = validate_level3_thermodynamic('isothermal')
        results['level3'] = level3_results

        for r in level3_results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {status} {r.engine:20} {r.message}")

    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    except ImportError as e:
        print(f"  [SKIP] Missing engine: {e}")

    # Summary
    print("\n" + "=" * 60)
    total_passed = sum(r.passed for level_results in results.values() for r in level_results)
    total_tests = sum(len(level_results) for level_results in results.values())

    if total_tests > 0:
        print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
        if total_passed == total_tests:
            print("All physics engines validated!")
    else:
        print("No tests run (missing test data)")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="PRISM Physics Engine Validator")
    parser.add_argument("--engine", help="Validate specific engine")
    parser.add_argument("--level", type=int, help="Validate specific level (0, 2, 3, 4)")
    parser.add_argument("--generate", action="store_true", help="Generate test data first")

    args = parser.parse_args()

    if args.generate:
        from .generate import generate_all_test_data
        generate_all_test_data(TEST_DATA_DIR)

    run_all_validations()


if __name__ == '__main__':
    main()
