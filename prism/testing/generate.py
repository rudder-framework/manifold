#!/usr/bin/env python3
"""
PRISM Test Data Generator — Curriculum Levels 0-4

Generates synthetic physics data with KNOWN parameters for validation.
Also fetches real datasets where available.

The principle: If we KNOW the parameters, we can validate the engines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import requests
import zipfile
import io


# =============================================================================
# LEVEL 0: RAW TIME SERIES (Synthetic + Real)
# =============================================================================

def generate_level0_random_walk(
    n_samples: int = 1000,
    n_signals: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate random walk time series.
    No physics knowledge, just statistics.
    """
    np.random.seed(seed)
    
    t = np.arange(n_samples)
    
    data = {'timestamp': t}
    for i in range(n_signals):
        # Random walk with drift
        drift = np.random.uniform(-0.01, 0.01)
        noise = np.random.randn(n_samples)
        signal = np.cumsum(noise) * 0.1 + drift * t
        data[f'sensor_{i+1}'] = signal
    
    return pd.DataFrame(data)


def generate_level0_oscillatory(
    n_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate oscillatory signals with noise.
    Tests spectral analysis, entropy, stationarity.
    """
    np.random.seed(seed)
    
    t = np.arange(n_samples) * dt
    
    # Multiple frequency components
    freq1, freq2, freq3 = 2.0, 5.0, 12.0  # Hz
    
    signal1 = np.sin(2 * np.pi * freq1 * t)
    signal2 = 0.5 * np.sin(2 * np.pi * freq2 * t)
    signal3 = 0.25 * np.sin(2 * np.pi * freq3 * t)
    
    combined = signal1 + signal2 + signal3 + 0.1 * np.random.randn(n_samples)
    
    return pd.DataFrame({
        'timestamp': t,
        'signal': combined,
        'signal_clean': signal1 + signal2 + signal3,
    }), {
        'frequencies_hz': [freq1, freq2, freq3],
        'dt': dt,
        'sampling_rate_hz': 1/dt,
    }


# =============================================================================
# LEVEL 1-2: MECHANICAL SYSTEMS WITH KNOWN PARAMETERS
# =============================================================================

@dataclass
class SpringMassDamperParams:
    """Parameters for spring-mass-damper system."""
    mass: float = 1.0           # kg
    spring_constant: float = 10.0  # N/m
    damping: float = 0.5        # N·s/m
    
    @property
    def natural_frequency(self) -> float:
        """ω_n = √(k/m) [rad/s]"""
        return np.sqrt(self.spring_constant / self.mass)
    
    @property
    def damping_ratio(self) -> float:
        """ζ = c / (2√(km))"""
        return self.damping / (2 * np.sqrt(self.spring_constant * self.mass))
    
    @property
    def damped_frequency(self) -> float:
        """ω_d = ω_n √(1 - ζ²)"""
        zeta = self.damping_ratio
        if zeta < 1:
            return self.natural_frequency * np.sqrt(1 - zeta**2)
        return 0.0  # Overdamped


def generate_spring_mass_damper(
    params: SpringMassDamperParams = None,
    x0: float = 1.0,
    v0: float = 0.0,
    t_max: float = 10.0,
    dt: float = 0.001,
    add_noise: bool = False,
    noise_std: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate spring-mass-damper data with KNOWN parameters.
    
    Equation: m·x'' + c·x' + k·x = 0
    
    Solution depends on damping ratio ζ:
        ζ < 1: Underdamped (oscillatory decay)
        ζ = 1: Critically damped
        ζ > 1: Overdamped
    
    Returns DataFrame with position, velocity, acceleration, and
    all KNOWN physics quantities for validation.
    """
    if params is None:
        params = SpringMassDamperParams()
    
    np.random.seed(seed)
    
    m = params.mass
    k = params.spring_constant
    c = params.damping
    
    omega_n = params.natural_frequency
    zeta = params.damping_ratio
    
    t = np.arange(0, t_max, dt)
    n = len(t)
    
    # Solve based on damping regime
    if zeta < 1:
        # Underdamped: x(t) = e^(-ζω_n·t) [A·cos(ω_d·t) + B·sin(ω_d·t)]
        omega_d = params.damped_frequency
        
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        
        exp_decay = np.exp(-zeta * omega_n * t)
        x = exp_decay * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        v = exp_decay * (
            -zeta * omega_n * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
            + omega_d * (-A * np.sin(omega_d * t) + B * np.cos(omega_d * t))
        )
        
    elif zeta == 1:
        # Critically damped: x(t) = (A + B·t)·e^(-ω_n·t)
        A = x0
        B = v0 + omega_n * x0
        
        exp_decay = np.exp(-omega_n * t)
        x = (A + B * t) * exp_decay
        v = (B - omega_n * (A + B * t)) * exp_decay
        
    else:
        # Overdamped: x(t) = A·e^(r1·t) + B·e^(r2·t)
        r1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
        
        A = (v0 - r2 * x0) / (r1 - r2)
        B = (r1 * x0 - v0) / (r1 - r2)
        
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        v = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
    
    # Acceleration from equation of motion: a = -(c/m)·v - (k/m)·x
    a = -(c/m) * v - (k/m) * x
    
    # Add measurement noise if requested
    if add_noise:
        x_measured = x + noise_std * np.random.randn(n)
        v_measured = v + noise_std * np.random.randn(n)
    else:
        x_measured = x
        v_measured = v
    
    # Compute TRUE physics quantities
    kinetic_energy = 0.5 * m * v**2
    potential_energy = 0.5 * k * x**2
    total_energy = kinetic_energy + potential_energy
    momentum = m * v
    hamiltonian = kinetic_energy + potential_energy
    lagrangian = kinetic_energy - potential_energy
    
    # Spring force
    force = -k * x - c * v
    
    df = pd.DataFrame({
        'timestamp': t,
        
        # Measured (possibly noisy)
        'position': x_measured,
        'velocity': v_measured,
        
        # True values
        'position_true': x,
        'velocity_true': v,
        'acceleration_true': a,
        
        # TRUE physics (for validation)
        'kinetic_energy_true': kinetic_energy,
        'potential_energy_true': potential_energy,
        'total_energy_true': total_energy,
        'momentum_true': momentum,
        'hamiltonian_true': hamiltonian,
        'lagrangian_true': lagrangian,
        'force_true': force,
    })
    
    # Ground truth for validation
    ground_truth = {
        'mass_kg': m,
        'spring_constant_N_per_m': k,
        'damping_N_s_per_m': c,
        'natural_frequency_rad_s': omega_n,
        'damping_ratio': zeta,
        'initial_position_m': x0,
        'initial_velocity_m_s': v0,
        'dt_s': dt,
        
        # Expected energy at t=0
        'initial_kinetic_energy_J': 0.5 * m * v0**2,
        'initial_potential_energy_J': 0.5 * k * x0**2,
        'initial_total_energy_J': 0.5 * m * v0**2 + 0.5 * k * x0**2,
        
        # Damping regime
        'regime': 'underdamped' if zeta < 1 else 'critically_damped' if zeta == 1 else 'overdamped',
    }
    
    return df, ground_truth


def generate_pendulum(
    length: float = 1.0,        # m
    mass: float = 1.0,         # kg
    theta0: float = 0.3,       # rad (initial angle)
    omega0: float = 0.0,       # rad/s (initial angular velocity)
    g: float = 9.81,           # m/s²
    damping: float = 0.0,      # damping coefficient
    t_max: float = 10.0,
    dt: float = 0.001,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate simple pendulum data.
    
    Small angle approximation: θ'' + (g/L)·θ = 0
    Period: T = 2π√(L/g)
    
    Returns position in Cartesian coordinates for angular momentum calculation.
    """
    np.random.seed(seed)
    
    t = np.arange(0, t_max, dt)
    n = len(t)
    
    # Natural frequency
    omega_n = np.sqrt(g / length)
    
    # Small angle solution
    if damping == 0:
        theta = theta0 * np.cos(omega_n * t) + (omega0 / omega_n) * np.sin(omega_n * t)
        theta_dot = -theta0 * omega_n * np.sin(omega_n * t) + omega0 * np.cos(omega_n * t)
    else:
        # Damped pendulum (underdamped approximation)
        zeta = damping / (2 * mass * omega_n)
        omega_d = omega_n * np.sqrt(1 - zeta**2) if zeta < 1 else omega_n
        exp_decay = np.exp(-zeta * omega_n * t)
        theta = exp_decay * theta0 * np.cos(omega_d * t)
        theta_dot = -exp_decay * theta0 * (zeta * omega_n * np.cos(omega_d * t) + omega_d * np.sin(omega_d * t))
    
    # Convert to Cartesian (pivot at origin)
    x = length * np.sin(theta)
    y = -length * np.cos(theta)  # y is down
    
    vx = length * theta_dot * np.cos(theta)
    vy = length * theta_dot * np.sin(theta)
    
    # For 3D angular momentum, we need z-component
    z = np.zeros(n)
    vz = np.zeros(n)
    
    # Physics quantities
    kinetic_energy = 0.5 * mass * (vx**2 + vy**2)
    potential_energy = mass * g * (y + length)  # Reference at bottom
    total_energy = kinetic_energy + potential_energy
    
    # Angular momentum: L = r × p (z-component for 2D motion)
    # L_z = x*py - y*px = m*(x*vy - y*vx)
    angular_momentum_z = mass * (x * vy - y * vx)
    
    df = pd.DataFrame({
        'timestamp': t,
        'theta': theta,
        'theta_dot': theta_dot,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'kinetic_energy_true': kinetic_energy,
        'potential_energy_true': potential_energy,
        'total_energy_true': total_energy,
        'angular_momentum_z_true': angular_momentum_z,
    })
    
    ground_truth = {
        'mass_kg': mass,
        'length_m': length,
        'g_m_s2': g,
        'natural_frequency_rad_s': omega_n,
        'period_s': 2 * np.pi / omega_n,
        'initial_angle_rad': theta0,
        'initial_angular_velocity_rad_s': omega0,
        'dt_s': dt,
    }
    
    return df, ground_truth


# =============================================================================
# LEVEL 3: THERMODYNAMIC DATA
# =============================================================================

def generate_ideal_gas_process(
    process_type: str = 'isothermal',  # isothermal, isobaric, isochoric, adiabatic
    n_moles: float = 1.0,
    T_initial: float = 300.0,  # K
    P_initial: float = 101325.0,  # Pa
    n_steps: int = 100,
    gamma: float = 1.4,  # Cp/Cv for diatomic
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate ideal gas thermodynamic data.
    
    PV = nRT
    
    Cp = (γ/(γ-1))·R [J/(mol·K)]
    Cv = (1/(γ-1))·R [J/(mol·K)]
    
    Returns T, P, V along with H, S, G for validation.
    """
    R = 8.314  # J/(mol·K)
    
    # Initial volume from ideal gas law
    V_initial = n_moles * R * T_initial / P_initial
    
    # Heat capacities
    Cp = gamma * R / (gamma - 1)  # J/(mol·K)
    Cv = R / (gamma - 1)          # J/(mol·K)
    
    if process_type == 'isothermal':
        # T = const, PV = const
        T = np.full(n_steps, T_initial)
        V = np.linspace(V_initial, 2 * V_initial, n_steps)
        P = n_moles * R * T / V
        
    elif process_type == 'isobaric':
        # P = const
        P = np.full(n_steps, P_initial)
        T = np.linspace(T_initial, 2 * T_initial, n_steps)
        V = n_moles * R * T / P
        
    elif process_type == 'isochoric':
        # V = const
        V = np.full(n_steps, V_initial)
        T = np.linspace(T_initial, 2 * T_initial, n_steps)
        P = n_moles * R * T / V
        
    elif process_type == 'adiabatic':
        # PV^γ = const, TV^(γ-1) = const
        V = np.linspace(V_initial, 2 * V_initial, n_steps)
        T = T_initial * (V_initial / V)**(gamma - 1)
        P = P_initial * (V_initial / V)**gamma
        
    else:
        raise ValueError(f"Unknown process type: {process_type}")
    
    # Thermodynamic quantities (relative to initial state)
    # Reference: standard state at T_ref = 298.15 K, P_ref = 101325 Pa
    T_ref = 298.15
    P_ref = 101325.0
    
    # Enthalpy: H = n·Cp·T (for ideal gas, H = H(T) only)
    H = n_moles * Cp * T
    H_ref = n_moles * Cp * T_ref
    delta_H = H - H_ref
    
    # Entropy: S = n·Cp·ln(T/T_ref) - n·R·ln(P/P_ref)
    S = n_moles * (Cp * np.log(T / T_ref) - R * np.log(P / P_ref))
    
    # Gibbs free energy: G = H - TS
    G = H - T * S
    
    # Internal energy: U = n·Cv·T
    U = n_moles * Cv * T
    
    df = pd.DataFrame({
        'step': np.arange(n_steps),
        'temperature_K': T,
        'pressure_Pa': P,
        'volume_m3': V,
        'enthalpy_J': H,
        'entropy_J_K': S,
        'gibbs_free_energy_J': G,
        'internal_energy_J': U,
    })
    
    ground_truth = {
        'process_type': process_type,
        'n_moles': n_moles,
        'R_J_mol_K': R,
        'gamma': gamma,
        'Cp_J_mol_K': Cp,
        'Cv_J_mol_K': Cv,
        'T_initial_K': T_initial,
        'P_initial_Pa': P_initial,
        'V_initial_m3': V_initial,
        'ideal_gas_law': 'PV = nRT',
    }
    
    return df, ground_truth


def generate_polytropic_process(
    n_moles: float = 1.0,
    T_initial: float = 300.0,  # K
    P_initial: float = 101325.0,  # Pa
    polytropic_index: float = 1.3,  # n (between 1 and γ)
    compression_ratio: float = 2.0,
    n_steps: int = 100,
    gamma: float = 1.4,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate polytropic process data: PV^n = const
    
    n = 1: Isothermal
    n = γ: Adiabatic
    1 < n < γ: Polytropic (with heat transfer)
    """
    R = 8.314
    n = polytropic_index
    
    V_initial = n_moles * R * T_initial / P_initial
    V_final = V_initial / compression_ratio
    
    V = np.linspace(V_initial, V_final, n_steps)
    
    # PV^n = const
    const = P_initial * V_initial**n
    P = const / V**n
    
    # T from ideal gas law
    T = P * V / (n_moles * R)
    
    # Heat capacities
    Cp = gamma * R / (gamma - 1)
    Cv = R / (gamma - 1)
    
    # Polytropic specific heat: C_n = Cv * (n - γ) / (n - 1)
    if n != 1:
        Cn = Cv * (n - gamma) / (n - 1)
    else:
        Cn = np.inf  # Isothermal
    
    # Work: W = (P2V2 - P1V1) / (1-n) for n ≠ 1
    # Heat: Q = Cn * ΔT
    
    H = n_moles * Cp * T
    U = n_moles * Cv * T
    
    df = pd.DataFrame({
        'step': np.arange(n_steps),
        'temperature_K': T,
        'pressure_Pa': P,
        'volume_m3': V,
        'enthalpy_J': H,
        'internal_energy_J': U,
    })
    
    ground_truth = {
        'process_type': 'polytropic',
        'polytropic_index_n': n,
        'compression_ratio': compression_ratio,
        'n_moles': n_moles,
        'gamma': gamma,
        'polytropic_heat_capacity_J_mol_K': Cn,
        'equation': f'PV^{n} = const',
    }
    
    return df, ground_truth


# =============================================================================
# LEVEL 4: VELOCITY FIELDS
# =============================================================================

def generate_synthetic_turbulence(
    nx: int = 32,
    ny: int = 32,
    nz: int = 32,
    nt: int = 10,
    L: float = 2 * np.pi,  # Domain size
    nu: float = 0.001,     # Kinematic viscosity
    energy_slope: float = -5/3,  # Kolmogorov scaling
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Generate synthetic homogeneous isotropic turbulence with Kolmogorov scaling.
    
    Energy spectrum: E(k) ∝ k^(-5/3)
    
    Returns velocity field v(x,y,z,t) and ground truth parameters.
    """
    np.random.seed(seed)
    
    dx = L / nx
    dy = L / ny
    dz = L / nz
    
    # Wavenumbers
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1  # Avoid division by zero
    
    # Generate velocity field in spectral space with Kolmogorov scaling
    # E(k) ∝ k^(-5/3) → |û(k)|² ∝ k^(-5/3) / (4πk²) ∝ k^(-11/3)
    amplitude = K**((energy_slope - 2) / 2)
    amplitude[0, 0, 0] = 0  # Zero mean
    
    # Random phases
    phase_u = np.exp(2j * np.pi * np.random.rand(nx, ny, nz))
    phase_v = np.exp(2j * np.pi * np.random.rand(nx, ny, nz))
    phase_w = np.exp(2j * np.pi * np.random.rand(nx, ny, nz))
    
    # Make divergence-free (incompressible)
    # Project: u = u - k(k·u)/|k|²
    u_hat = amplitude * phase_u
    v_hat = amplitude * phase_v
    w_hat = amplitude * phase_w
    
    k_dot_u = KX * u_hat + KY * v_hat + KZ * w_hat
    K2 = K**2
    K2[0, 0, 0] = 1
    
    u_hat -= KX * k_dot_u / K2
    v_hat -= KY * k_dot_u / K2
    w_hat -= KZ * k_dot_u / K2
    
    # Transform to physical space
    fields = {}
    
    for t_idx in range(nt):
        # Add some temporal variation (simplified)
        phase_shift = np.exp(1j * 0.1 * t_idx * K)
        
        u = np.real(np.fft.ifftn(u_hat * phase_shift))
        v = np.real(np.fft.ifftn(v_hat * phase_shift))
        w = np.real(np.fft.ifftn(w_hat * phase_shift))
        
        fields[t_idx] = {
            'u': u,
            'v': v,
            'w': w,
        }
    
    # Compute some statistics for validation
    u0 = fields[0]['u']
    v0 = fields[0]['v']
    w0 = fields[0]['w']
    
    urms = np.sqrt(np.mean(u0**2 + v0**2 + w0**2) / 3)
    tke = 0.5 * np.mean(u0**2 + v0**2 + w0**2)
    
    # Estimate dissipation (simplified)
    # ε ≈ ν * mean(|∇u|²)
    dudx = np.gradient(u0, dx, axis=0)
    dudy = np.gradient(u0, dy, axis=1)
    dudz = np.gradient(u0, dz, axis=2)
    
    grad_u_squared = dudx**2 + dudy**2 + dudz**2
    epsilon = nu * np.mean(grad_u_squared) * 15  # Approximate
    
    # Kolmogorov scales
    eta = (nu**3 / epsilon)**(1/4) if epsilon > 0 else np.inf
    tau_eta = (nu / epsilon)**(1/2) if epsilon > 0 else np.inf
    v_eta = (nu * epsilon)**(1/4) if epsilon > 0 else 0
    
    # Reynolds number
    L_integral = L / 4  # Rough estimate
    Re = urms * L_integral / nu
    
    ground_truth = {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'nt': nt,
        'domain_size_m': L,
        'dx_m': dx,
        'dy_m': dy,
        'dz_m': dz,
        'kinematic_viscosity_m2_s': nu,
        'expected_energy_slope': energy_slope,
        
        # Computed statistics
        'urms_m_s': urms,
        'turbulent_kinetic_energy_m2_s2': tke,
        'dissipation_rate_m2_s3': epsilon,
        
        # Kolmogorov scales
        'kolmogorov_length_m': eta,
        'kolmogorov_time_s': tau_eta,
        'kolmogorov_velocity_m_s': v_eta,
        
        'reynolds_number': Re,
    }
    
    return fields, ground_truth


def generate_channel_flow(
    nx: int = 64,
    ny: int = 32,
    nz: int = 64,
    Re_tau: float = 180,  # Friction Reynolds number
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Generate simplified turbulent channel flow profile.
    
    Mean profile: u+ = y+ for y+ < 5 (viscous sublayer)
                  u+ = 2.5*ln(y+) + 5.5 for y+ > 30 (log layer)
    """
    np.random.seed(seed)
    
    # Channel half-height
    delta = 1.0  # m
    
    # Friction velocity from Re_tau
    nu = 1.0 / Re_tau  # Simplified: u_tau * delta / nu = Re_tau → nu = delta/Re_tau if u_tau=1
    u_tau = 1.0
    
    # Grid
    y = np.linspace(0, 2*delta, ny)  # Wall to wall
    y_plus = y * u_tau / nu
    
    # Mean velocity profile (log law)
    u_mean = np.zeros(ny)
    for i, yp in enumerate(y_plus):
        if yp < 5:
            u_mean[i] = yp
        elif yp < 30:
            u_mean[i] = 5.0 * np.log(yp) - 3.05  # Buffer layer (approximate)
        else:
            u_mean[i] = 2.5 * np.log(yp) + 5.5
    
    u_mean *= u_tau
    
    # Add fluctuations (simplified)
    u_rms = 0.1 * u_tau * np.exp(-y_plus / 50)  # Decays away from wall
    
    # Create 3D field
    u = np.zeros((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    w = np.zeros((nx, ny, nz))
    
    for j in range(ny):
        u[:, j, :] = u_mean[j] + u_rms[j] * np.random.randn(nx, nz)
        v[:, j, :] = 0.5 * u_rms[j] * np.random.randn(nx, nz)
        w[:, j, :] = 0.5 * u_rms[j] * np.random.randn(nx, nz)
    
    fields = {
        0: {'u': u, 'v': v, 'w': w}
    }
    
    ground_truth = {
        'flow_type': 'turbulent_channel',
        'Re_tau': Re_tau,
        'friction_velocity_m_s': u_tau,
        'kinematic_viscosity_m2_s': nu,
        'channel_half_height_m': delta,
        'mean_profile': 'log_law',
        'y_plus': y_plus.tolist(),
        'u_plus_mean': (u_mean / u_tau).tolist(),
    }
    
    return fields, ground_truth


# =============================================================================
# REAL DATA FETCHERS
# =============================================================================

def fetch_tennessee_eastman(output_dir: Path = None) -> Path:
    """
    Fetch Tennessee Eastman Process dataset from GitHub.
    
    Has: Temperature, Pressure, Flow rates, Levels, Compositions
    """
    if output_dir is None:
        output_dir = Path('data/tep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GitHub raw URL
    base_url = "https://github.com/mv-per/tennessee-eastman-dataset/raw/main/simulations/mode_1/"
    
    files = [
        "mode1_normal_50.xlsx",
    ]
    
    for fname in files:
        url = base_url + fname
        output_path = output_dir / fname
        
        if not output_path.exists():
            print(f"Downloading {fname}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"  Saved to {output_path}")
            else:
                print(f"  Failed: {response.status_code}")
    
    return output_dir


def fetch_cwru_bearing(output_dir: Path = None) -> Path:
    """
    Fetch Case Western Reserve University Bearing Dataset.
    
    Has: Vibration acceleration data at 12k/48k Hz
    """
    if output_dir is None:
        output_dir = Path('data/cwru')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CWRU data is in .mat format, direct download is complex
    # For now, create a README pointing to the source
    
    readme = output_dir / 'README.md'
    readme.write_text("""# CWRU Bearing Dataset

Download from: https://engineering.case.edu/bearingdatacenter/download-data-file

This dataset contains vibration signals from bearings with:
- Inner race faults
- Outer race faults  
- Ball faults
- Normal operation

Sampling rates: 12 kHz and 48 kHz

For PRISM Level 0 (raw time series) testing.
""")
    
    return output_dir


# =============================================================================
# MASTER TEST DATA GENERATOR
# =============================================================================

def generate_all_test_data(output_dir: Path = None) -> Dict:
    """
    Generate complete test dataset for all curriculum levels.
    """
    if output_dir is None:
        output_dir = Path('data/prism_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # LEVEL 0: Raw Time Series
    # =========================================================================
    print("\n=== LEVEL 0: Raw Time Series ===")
    
    # Random walk
    df = generate_level0_random_walk()
    path = output_dir / 'level0_random_walk.parquet'
    df.to_parquet(path)
    results['level0_random_walk'] = path
    print(f"  Created: {path}")
    
    # Oscillatory
    df, meta = generate_level0_oscillatory()
    path = output_dir / 'level0_oscillatory.parquet'
    df.to_parquet(path)
    (output_dir / 'level0_oscillatory_meta.txt').write_text(str(meta))
    results['level0_oscillatory'] = path
    print(f"  Created: {path}")
    
    # =========================================================================
    # LEVEL 1-2: Mechanical with Known Parameters
    # =========================================================================
    print("\n=== LEVEL 1-2: Mechanical Systems ===")
    
    # Underdamped spring-mass-damper
    params = SpringMassDamperParams(mass=2.0, spring_constant=50.0, damping=1.0)
    df, ground_truth = generate_spring_mass_damper(params, x0=0.5, t_max=5.0)
    path = output_dir / 'level2_spring_mass_damper.parquet'
    df.to_parquet(path)
    
    # Save ground truth as JSON-like text
    gt_path = output_dir / 'level2_spring_mass_damper_ground_truth.txt'
    gt_path.write_text(str(ground_truth))
    
    results['level2_spring_mass_damper'] = {
        'data': path,
        'ground_truth': ground_truth,
    }
    print(f"  Created: {path}")
    print(f"    mass = {params.mass} kg")
    print(f"    k = {params.spring_constant} N/m")
    print(f"    ζ = {params.damping_ratio:.3f} ({ground_truth['regime']})")
    
    # Pendulum (for angular momentum)
    df, ground_truth = generate_pendulum(length=1.5, mass=0.5, theta0=0.2, t_max=10.0)
    path = output_dir / 'level2_pendulum.parquet'
    df.to_parquet(path)
    
    gt_path = output_dir / 'level2_pendulum_ground_truth.txt'
    gt_path.write_text(str(ground_truth))
    
    results['level2_pendulum'] = {
        'data': path,
        'ground_truth': ground_truth,
    }
    print(f"  Created: {path}")
    print(f"    length = {ground_truth['length_m']} m")
    print(f"    mass = {ground_truth['mass_kg']} kg")
    print(f"    period = {ground_truth['period_s']:.3f} s")
    
    # =========================================================================
    # LEVEL 3: Thermodynamic Data
    # =========================================================================
    print("\n=== LEVEL 3: Thermodynamic Processes ===")
    
    for process in ['isothermal', 'isobaric', 'isochoric', 'adiabatic']:
        df, ground_truth = generate_ideal_gas_process(
            process_type=process,
            n_moles=1.0,
            T_initial=300.0,
            P_initial=101325.0,
        )
        path = output_dir / f'level3_{process}.parquet'
        df.to_parquet(path)
        
        gt_path = output_dir / f'level3_{process}_ground_truth.txt'
        gt_path.write_text(str(ground_truth))
        
        results[f'level3_{process}'] = {
            'data': path,
            'ground_truth': ground_truth,
        }
        print(f"  Created: {path}")
    
    # Polytropic
    df, ground_truth = generate_polytropic_process(polytropic_index=1.3)
    path = output_dir / 'level3_polytropic.parquet'
    df.to_parquet(path)
    results['level3_polytropic'] = {'data': path, 'ground_truth': ground_truth}
    print(f"  Created: {path}")
    
    # =========================================================================
    # LEVEL 4: Velocity Fields
    # =========================================================================
    print("\n=== LEVEL 4: Velocity Fields ===")
    
    # Synthetic HIT (Homogeneous Isotropic Turbulence)
    fields, ground_truth = generate_synthetic_turbulence(nx=32, ny=32, nz=32, nt=5)
    
    # Save as individual arrays (parquet doesn't handle 3D well)
    field_dir = output_dir / 'level4_turbulence'
    field_dir.mkdir(exist_ok=True)
    
    for t_idx, field in fields.items():
        np.savez(
            field_dir / f't{t_idx:04d}.npz',
            u=field['u'],
            v=field['v'],
            w=field['w'],
        )
    
    gt_path = field_dir / 'ground_truth.txt'
    gt_path.write_text(str(ground_truth))
    
    results['level4_turbulence'] = {
        'data': field_dir,
        'ground_truth': ground_truth,
    }
    print(f"  Created: {field_dir}")
    print(f"    Grid: {ground_truth['nx']}×{ground_truth['ny']}×{ground_truth['nz']}")
    print(f"    Expected E(k) slope: {ground_truth['expected_energy_slope']:.2f}")
    print(f"    Re ≈ {ground_truth['reynolds_number']:.1f}")
    
    # Channel flow
    fields, ground_truth = generate_channel_flow(nx=32, ny=16, nz=32, Re_tau=180)
    
    field_dir = output_dir / 'level4_channel'
    field_dir.mkdir(exist_ok=True)
    
    for t_idx, field in fields.items():
        np.savez(
            field_dir / f't{t_idx:04d}.npz',
            u=field['u'],
            v=field['v'],
            w=field['w'],
        )
    
    gt_path = field_dir / 'ground_truth.txt'
    gt_path.write_text(str(ground_truth))
    
    results['level4_channel'] = {
        'data': field_dir,
        'ground_truth': ground_truth,
    }
    print(f"  Created: {field_dir}")
    print(f"    Re_τ = {ground_truth['Re_tau']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("TEST DATA GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    for name, info in results.items():
        if isinstance(info, dict):
            print(f"  {name}: {info['data']}")
        else:
            print(f"  {name}: {info}")
    
    return results


# =============================================================================
# PRISM CONFIG GENERATOR
# =============================================================================

def generate_prism_configs(output_dir: Path = None):
    """
    Generate PRISM config files for each test dataset.
    """
    if output_dir is None:
        output_dir = Path('data/prism_test')
    
    # Level 0 config (no physics)
    config_level0 = """# PRISM Config: Level 0 (Raw Time Series)
# No physical labels, just statistics

window_size: 50
window_stride: 25

signals:
  - name: sensor_1
  - name: sensor_2
  - name: sensor_3
  - name: sensor_4
  - name: sensor_5
"""
    
    # Level 2 config (mechanical with constants)
    config_level2 = """# PRISM Config: Level 2 (Mechanical with Constants)
# Spring-mass-damper system with known parameters

window_size: 100
window_stride: 50

signals:
  - name: position
    physical_quantity: position
    units: m
  - name: velocity
    physical_quantity: velocity
    units: m/s

constants:
  mass: 2.0              # kg
  spring_constant: 50.0  # N/m
  damping: 1.0           # N·s/m

relationships:
  mechanical:
    position: position
    velocity: velocity
"""
    
    # Level 3 config (thermodynamic)
    config_level3 = """# PRISM Config: Level 3 (Thermodynamic)
# Ideal gas process with T, P, V

window_size: 20
window_stride: 10

signals:
  - name: temperature_K
    physical_quantity: temperature
    units: K
  - name: pressure_Pa
    physical_quantity: pressure
    units: Pa
  - name: volume_m3
    physical_quantity: volume
    units: m³

constants:
  Cp: 29.1      # J/(mol·K) for diatomic ideal gas
  n_moles: 1.0  # mol

relationships:
  thermodynamic:
    temperature: temperature_K
    pressure: pressure_Pa
    volume: volume_m3
"""
    
    # Level 4 config (velocity field)
    config_level4 = """# PRISM Config: Level 4 (Velocity Field)
# Homogeneous isotropic turbulence

window_size: 1   # Each snapshot is one window
window_stride: 1

spatial:
  type: velocity_field
  dimensions: [32, 32, 32]
  u: u
  v: v
  w: w

constants:
  kinematic_viscosity: 0.001  # m²/s
  dx: 0.196                   # m (2π/32)
  dy: 0.196                   # m
  dz: 0.196                   # m
"""
    
    (output_dir / 'config_level0.yaml').write_text(config_level0)
    (output_dir / 'config_level2.yaml').write_text(config_level2)
    (output_dir / 'config_level3.yaml').write_text(config_level3)
    (output_dir / 'config_level4.yaml').write_text(config_level4)
    
    print("\nGenerated PRISM config files:")
    print(f"  {output_dir / 'config_level0.yaml'}")
    print(f"  {output_dir / 'config_level2.yaml'}")
    print(f"  {output_dir / 'config_level3.yaml'}")
    print(f"  {output_dir / 'config_level4.yaml'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/prism_test')
    
    results = generate_all_test_data(output_dir)
    generate_prism_configs(output_dir)
    
    print("\n" + "="*60)
    print("VALIDATION READY")
    print("="*60)
    print("""
Now you can validate each physics engine:

Level 0: Statistics, Entropy, Memory, Spectral
  → Use level0_*.parquet

Level 2: Kinetic Energy, Potential Energy, Hamiltonian, Lagrangian, Momentum
  → Use level2_spring_mass_damper.parquet
  → Compare computed values to *_true columns

Level 3: Gibbs Free Energy, Enthalpy
  → Use level3_*.parquet
  → Compare to known thermodynamic relations

Level 4: Navier-Stokes (vorticity, TKE, energy spectrum)
  → Use level4_turbulence/ 
  → Check E(k) slope ≈ -5/3

The ground truth is in the *_ground_truth.txt files.
""")
