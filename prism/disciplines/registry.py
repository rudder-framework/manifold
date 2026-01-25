"""
PRISM Discipline Registry

Defines available disciplines with their requirements and engines.

Each discipline specifies:
- Required/optional signals
- Required/optional constants (with units)
- Available engines
"""

DISCIPLINES = {
    "thermodynamics": {
        "name": "Thermodynamics",
        "description": "Energy, entropy, equations of state",
        "icon": "fire",
        "signals": {
            "required_any": ["temperature", "pressure"],  # Need at least one
            "optional": ["volume", "internal_energy", "enthalpy"],
        },
        "constants": {
            "optional": {
                "gas_constant": {"unit": "J/(mol·K)", "default": 8.314},
                "molar_mass": {"unit": "kg/mol"},
                "heat_capacity_cp": {"unit": "J/(kg·K)"},
                "heat_capacity_cv": {"unit": "J/(kg·K)"},
                "critical_temperature": {"unit": "K"},
                "critical_pressure": {"unit": "Pa"},
                "acentric_factor": {"unit": "dimensionless"},
            }
        },
        "engines": [
            "gibbs_free_energy",
            "enthalpy_ideal_gas",
            "entropy_ideal_gas",
            "ideal_gas",
            "van_der_waals",
            "peng_robinson",
            "clausius_clapeyron",
            "antoine_equation",
            "raoults_law",
            "henrys_law",
            "fugacity",
            "activity",
        ],
    },

    "transport": {
        "name": "Transport Phenomena",
        "description": "Heat, mass, and momentum transfer",
        "icon": "waves",
        "subdisciplines": {
            "momentum": {
                "name": "Momentum Transfer",
                "signals": {
                    "required_any": ["velocity", "flow_rate"],
                },
                "constants": {
                    "required": {
                        "density": {"unit": "kg/m³"},
                        "viscosity": {"unit": "Pa·s"},
                    },
                    "optional": {
                        "diameter": {"unit": "m"},
                        "length": {"unit": "m"},
                        "roughness": {"unit": "m"},
                    }
                },
                "engines": [
                    "reynolds",
                    "pressure_drop",
                    "friction_factor",
                    "bernoulli_equation",
                    "hagen_poiseuille",
                    "pump_power",
                ],
            },
            "heat": {
                "name": "Heat Transfer",
                "signals": {
                    "required": ["temperature"],
                },
                "constants": {
                    "required": {
                        "thermal_conductivity": {"unit": "W/(m·K)"},
                    },
                    "optional": {
                        "heat_transfer_coeff": {"unit": "W/(m²·K)"},
                        "thermal_diffusivity": {"unit": "m²/s"},
                    }
                },
                "engines": [
                    "compute_heat_flux",
                    "compute_conduction_slab",
                    "dittus_boelter",
                    "gnielinski",
                    "compute_prandtl",
                    "compute_nusselt",
                    "overall_heat_transfer_coefficient",
                ],
            },
            "mass": {
                "name": "Mass Transfer",
                "signals": {
                    "required": ["concentration"],
                },
                "constants": {
                    "required": {
                        "diffusivity": {"unit": "m²/s"},
                    },
                    "optional": {
                        "mass_transfer_coeff": {"unit": "m/s"},
                    }
                },
                "engines": [
                    "compute_molar_flux",
                    "compute_mass_transfer_slab",
                    "compute_schmidt",
                    "compute_sherwood",
                    "ranz_marshall",
                    "wilke_chang",
                ],
            },
        },
        "engines": [
            # Dimensionless numbers (need various constants)
            "compute_peclet",
            "compute_weber",
            "compute_froude",
            "compute_grashof",
            "compute_rayleigh",
            "compute_biot",
            "compute_lewis",
            "compute_stanton",
            "chilton_colburn_analogy",
        ],
    },

    "reaction": {
        "name": "Reaction Engineering",
        "description": "Kinetics, reactor design, yields",
        "icon": "flask",
        "signals": {
            "required_any": ["concentration", "conversion", "temperature"],
        },
        "constants": {
            "optional": {
                "activation_energy": {"unit": "J/mol"},
                "pre_exponential": {"unit": "1/s"},
                "reaction_order": {"unit": "dimensionless"},
                "reactor_volume": {"unit": "m³"},
                "flow_rate": {"unit": "m³/s"},
                "K_m": {"unit": "mol/m³", "description": "Michaelis constant"},
                "V_max": {"unit": "mol/(m³·s)", "description": "Maximum rate"},
            }
        },
        "engines": [
            "arrhenius",
            "arrhenius_two_temperatures",
            "power_law_rate",
            "michaelis_menten",
            "langmuir_hinshelwood",
            "conversion",
            "yield_and_selectivity",
            "batch_reactor_time",
            "cstr_volume",
            "pfr_volume",
            "residence_time_distribution",
            "equilibrium_constant",
            "compute_damkohler",
        ],
    },

    "controls": {
        "name": "Process Control",
        "description": "Dynamics, stability, feedback",
        "icon": "sliders",
        "signals": {
            "required": ["process_variable"],  # The thing being controlled
            "optional": ["setpoint", "manipulated_variable", "controller_output"],
        },
        "constants": {
            "optional": {
                "time_constant": {"unit": "s"},
                "dead_time": {"unit": "s"},
                "gain": {"unit": "dimensionless"},
                "kp": {"unit": "dimensionless", "description": "Proportional gain"},
                "ki": {"unit": "1/s", "description": "Integral gain"},
                "kd": {"unit": "s", "description": "Derivative gain"},
            }
        },
        "engines": [
            "first_order_response",
            "second_order_response",
            "time_delay_pade",
            "pid_controller",
            "ziegler_nichols_closed_loop",
            "ziegler_nichols_open_loop",
            "imc_tuning",
            "stability_margins",
            "poles_and_zeros",
            "closed_loop_transfer_function",
        ],
    },

    "mechanics": {
        "name": "Mechanical Systems",
        "description": "Vibration, fatigue, stress, energy",
        "icon": "gear",
        "signals": {
            "required_any": ["vibration", "acceleration", "displacement", "strain", "stress", "velocity"],
        },
        "constants": {
            "optional": {
                "mass": {"unit": "kg"},
                "stiffness": {"unit": "N/m"},
                "damping": {"unit": "N·s/m"},
                "youngs_modulus": {"unit": "Pa"},
                "poissons_ratio": {"unit": "dimensionless"},
                "moment_of_inertia": {"unit": "kg·m²"},
            }
        },
        "engines": [
            "compute_kinetic_energy",
            "compute_potential_energy_harmonic",
            "compute_potential_energy_gravitational",
            "compute_hamiltonian",
            "compute_lagrangian",
            "compute_linear_momentum",
            "compute_angular_momentum",
            "compute_work",
            "compute_power",
        ],
    },

    "electrical": {
        "name": "Electrical Systems",
        "description": "Impedance, power, batteries",
        "icon": "zap",
        "signals": {
            "required_any": ["voltage", "current", "power", "impedance"],
        },
        "constants": {
            "optional": {
                "resistance": {"unit": "Ω"},
                "capacitance": {"unit": "F"},
                "inductance": {"unit": "H"},
                "nominal_capacity": {"unit": "Ah"},
                "nominal_voltage": {"unit": "V"},
            }
        },
        "engines": [
            # Electrical-specific engines (to be implemented)
            "impedance_spectrum",
            "power_factor",
            "soh",  # State of Health (batteries)
            "capacity_fade",
            "internal_resistance",
        ],
    },

    "fluid_dynamics": {
        "name": "Fluid Dynamics (CFD)",
        "description": "Velocity fields, vorticity, turbulence",
        "icon": "wind",
        "signals": {
            "required": ["u", "v"],  # Velocity components
            "optional": ["w", "pressure"],
        },
        "constants": {
            "required": {
                "density": {"unit": "kg/m³"},
                "viscosity": {"unit": "Pa·s"},
            },
            "optional": {
                "reference_velocity": {"unit": "m/s"},
                "reference_length": {"unit": "m"},
            }
        },
        "spatial": True,  # Indicates Level 4 / Fields analysis
        "engines": [
            "vorticity",
            "divergence",
            "q_criterion",
            "strain_rate",
            "tke",
            "navier_stokes_residual",
        ],
    },

    "dimensionless": {
        "name": "Dimensionless Analysis",
        "description": "All dimensionless numbers",
        "icon": "hash",
        "signals": {
            "optional": ["velocity", "temperature", "concentration"],
        },
        "constants": {
            "optional": {
                "density": {"unit": "kg/m³"},
                "viscosity": {"unit": "Pa·s"},
                "thermal_conductivity": {"unit": "W/(m·K)"},
                "heat_capacity": {"unit": "J/(kg·K)"},
                "diffusivity": {"unit": "m²/s"},
                "characteristic_length": {"unit": "m"},
                "characteristic_velocity": {"unit": "m/s"},
            }
        },
        "engines": [
            "compute_prandtl",
            "compute_schmidt",
            "compute_nusselt",
            "compute_sherwood",
            "compute_peclet",
            "compute_damkohler",
            "compute_weber",
            "compute_froude",
            "compute_grashof",
            "compute_rayleigh",
            "compute_biot",
            "compute_lewis",
            "compute_stanton",
            "compute_all_dimensionless",
        ],
    },

    # === NEW ChemE DISCIPLINES ===

    "separations": {
        "name": "Separations",
        "description": "Distillation, absorption, extraction, membranes",
        "icon": "flask",
        "subdisciplines": {
            "distillation": {
                "name": "Distillation",
                "signals": {
                    "required_any": ["temperature", "composition", "vapor_fraction"],
                },
                "constants": {
                    "required": {
                        "relative_volatility": {"unit": "dimensionless", "symbol": "α"},
                    },
                    "optional": {
                        "feed_quality": {"unit": "dimensionless", "symbol": "q", "description": "q=1 saturated liquid, q=0 saturated vapor"},
                        "reflux_ratio": {"unit": "dimensionless", "symbol": "R"},
                        "min_reflux_ratio": {"unit": "dimensionless", "symbol": "R_min"},
                        "n_stages": {"unit": "dimensionless"},
                        "feed_stage": {"unit": "dimensionless"},
                        "distillate_composition": {"unit": "mol fraction", "symbol": "x_D"},
                        "bottoms_composition": {"unit": "mol fraction", "symbol": "x_B"},
                        "feed_composition": {"unit": "mol fraction", "symbol": "x_F"},
                        "murphree_efficiency": {"unit": "dimensionless", "symbol": "E_M"},
                        "tray_spacing": {"unit": "m"},
                        "column_diameter": {"unit": "m"},
                    }
                },
                "engines": [
                    "mccabe_thiele",
                    "fenske",
                    "underwood",
                    "gilliland",
                    "kirkbride",
                    "stage_efficiency",
                    "flooding_velocity",
                ],
            },
            "absorption": {
                "name": "Absorption / Stripping",
                "signals": {
                    "required_any": ["gas_concentration", "liquid_concentration"],
                },
                "constants": {
                    "required": {
                        "henrys_constant": {"unit": "Pa", "symbol": "H"},
                    },
                    "optional": {
                        "gas_flow_rate": {"unit": "mol/s", "symbol": "G"},
                        "liquid_flow_rate": {"unit": "mol/s", "symbol": "L"},
                        "absorption_factor": {"unit": "dimensionless", "symbol": "A", "description": "A = L/(mG)"},
                        "mass_transfer_coeff_gas": {"unit": "mol/(m²·s·Pa)", "symbol": "k_G"},
                        "mass_transfer_coeff_liquid": {"unit": "m/s", "symbol": "k_L"},
                        "interfacial_area": {"unit": "m²/m³", "symbol": "a"},
                        "packing_height": {"unit": "m"},
                    }
                },
                "engines": [
                    "ntu",
                    "htu",
                    "kremser",
                    "operating_line",
                    "overall_mass_transfer",
                ],
            },
            "extraction": {
                "name": "Liquid-Liquid Extraction",
                "signals": {
                    "required": ["concentration"],
                },
                "constants": {
                    "required": {
                        "partition_coefficient": {"unit": "dimensionless", "symbol": "K_D", "description": "K = C_extract/C_raffinate"},
                    },
                    "optional": {
                        "solvent_flow_rate": {"unit": "mol/s", "symbol": "S"},
                        "feed_flow_rate": {"unit": "mol/s", "symbol": "F"},
                        "extraction_factor": {"unit": "dimensionless", "symbol": "E", "description": "E = KS/F"},
                        "n_stages": {"unit": "dimensionless"},
                    }
                },
                "engines": [
                    "single_stage_extraction",
                    "cross_current",
                    "counter_current",
                    "extraction_efficiency",
                ],
            },
            "membrane": {
                "name": "Membrane Separations",
                "signals": {
                    "required_any": ["flux", "pressure", "concentration"],
                },
                "constants": {
                    "required": {
                        "permeability": {"unit": "mol/(m·s·Pa)", "symbol": "P"},
                    },
                    "optional": {
                        "membrane_thickness": {"unit": "m", "symbol": "δ"},
                        "membrane_area": {"unit": "m²", "symbol": "A"},
                        "selectivity": {"unit": "dimensionless", "symbol": "α", "description": "α = P_A/P_B"},
                        "pressure_ratio": {"unit": "dimensionless"},
                        "stage_cut": {"unit": "dimensionless", "symbol": "θ", "description": "permeate/feed"},
                        "rejection_coefficient": {"unit": "dimensionless", "symbol": "R", "description": "R = 1 - C_p/C_f"},
                    }
                },
                "engines": [
                    "permeation_flux",
                    "membrane_selectivity",
                    "concentration_polarization",
                    "spiral_wound",
                    "hollow_fiber",
                ],
            },
        },
        "engines": [],
    },

    "phase_equilibria": {
        "name": "Phase Equilibria",
        "description": "VLE, LLE, flash calculations, activity models",
        "icon": "balance",
        "subdisciplines": {
            "vle": {
                "name": "Vapor-Liquid Equilibrium",
                "signals": {
                    "required_any": ["temperature", "pressure", "composition"],
                },
                "constants": {
                    "required": {
                        "vapor_pressure_params": {"unit": "various", "description": "Antoine A, B, C or other correlation"},
                    },
                    "optional": {
                        "critical_temperature": {"unit": "K", "symbol": "T_c"},
                        "critical_pressure": {"unit": "Pa", "symbol": "P_c"},
                        "acentric_factor": {"unit": "dimensionless", "symbol": "ω"},
                        "molar_volume_liquid": {"unit": "m³/mol", "symbol": "V_L"},
                    }
                },
                "engines": [
                    "antoine",
                    "raoults_law",
                    "modified_raoults",
                    "k_value",
                    "relative_volatility",
                    "bubble_point",
                    "dew_point",
                    "txy_diagram",
                    "pxy_diagram",
                ],
            },
            "lle": {
                "name": "Liquid-Liquid Equilibrium",
                "signals": {
                    "required": ["composition"],
                },
                "constants": {
                    "optional": {
                        "binary_interaction_params": {"unit": "various", "description": "NRTL or UNIQUAC parameters"},
                        "plait_point": {"unit": "mol fraction"},
                    }
                },
                "engines": [
                    "tie_line",
                    "binodal_curve",
                    "plait_point",
                    "lever_rule",
                    "ternary_diagram",
                ],
            },
            "flash": {
                "name": "Flash Calculations",
                "signals": {
                    "required": ["composition"],
                    "required_any": ["temperature", "pressure"],
                },
                "constants": {
                    "required": {
                        "k_values": {"unit": "dimensionless", "description": "Or correlation to calculate"},
                    },
                    "optional": {
                        "feed_flow_rate": {"unit": "mol/s", "symbol": "F"},
                    }
                },
                "engines": [
                    "rachford_rice",
                    "isothermal_flash",
                    "adiabatic_flash",
                    "three_phase_flash",
                ],
            },
            "activity_models": {
                "name": "Activity Coefficient Models",
                "signals": {
                    "required": ["composition"],
                },
                "constants": {
                    "optional": {
                        "margules_params": {"unit": "dimensionless", "description": "A_12, A_21"},
                        "van_laar_params": {"unit": "dimensionless", "description": "A, B"},
                        "wilson_params": {"unit": "dimensionless", "description": "Λ_12, Λ_21"},
                        "nrtl_params": {"unit": "various", "description": "τ_12, τ_21, α"},
                        "uniquac_params": {"unit": "various", "description": "r, q, u"},
                        "unifac_groups": {"unit": "various", "description": "Group definitions"},
                    }
                },
                "engines": [
                    "ideal_solution",
                    "margules",
                    "van_laar",
                    "wilson",
                    "nrtl",
                    "uniquac",
                    "unifac",
                ],
            },
        },
        "engines": [
            "fugacity_coefficient",
            "poynting_correction",
            "gamma_phi",
        ],
    },

    "balances": {
        "name": "Material & Energy Balances",
        "description": "Conservation equations, process calculations",
        "icon": "scale",
        "subdisciplines": {
            "material": {
                "name": "Material Balances",
                "signals": {
                    "required_any": ["flow_rate", "mass_flow", "molar_flow", "concentration"],
                },
                "constants": {
                    "optional": {
                        "molecular_weight": {"unit": "kg/mol", "symbol": "M"},
                        "density": {"unit": "kg/m³", "symbol": "ρ"},
                        "stoichiometric_coefficients": {"unit": "dimensionless", "symbol": "ν"},
                        "conversion": {"unit": "dimensionless", "symbol": "X"},
                    }
                },
                "engines": [
                    "total_mass_balance",
                    "component_balance",
                    "extent_of_reaction",
                    "limiting_reactant",
                    "excess_reactant",
                    "recycle_ratio",
                    "purge_calculation",
                    "bypass_calculation",
                    "mixing_balance",
                    "splitting_balance",
                ],
            },
            "energy": {
                "name": "Energy Balances",
                "signals": {
                    "required_any": ["temperature", "enthalpy", "heat_duty"],
                },
                "constants": {
                    "optional": {
                        "heat_capacity_cp": {"unit": "J/(mol·K)", "symbol": "C_p"},
                        "heat_capacity_params": {"unit": "various", "description": "Cp = a + bT + cT² polynomial"},
                        "heat_of_reaction": {"unit": "J/mol", "symbol": "ΔH_rxn"},
                        "heat_of_formation": {"unit": "J/mol", "symbol": "ΔH_f"},
                        "heat_of_vaporization": {"unit": "J/mol", "symbol": "ΔH_vap"},
                        "heat_of_fusion": {"unit": "J/mol", "symbol": "ΔH_fus"},
                        "reference_temperature": {"unit": "K", "symbol": "T_ref", "default": 298.15},
                    }
                },
                "engines": [
                    "sensible_heat",
                    "latent_heat",
                    "enthalpy_balance",
                    "heat_of_reaction_calc",
                    "adiabatic_flame_temp",
                    "adiabatic_reaction_temp",
                    "heat_exchanger_duty",
                    "lmtd",
                    "effectiveness_ntu",
                ],
            },
            "combined": {
                "name": "Combined Balances",
                "signals": {
                    "required_any": ["flow_rate", "temperature", "composition"],
                },
                "constants": {},
                "engines": [
                    "degrees_of_freedom",
                    "process_simulation",
                    "recycle_convergence",
                ],
            },
        },
        "engines": [],
    },

    "electrochemistry": {
        "name": "Electrochemistry",
        "description": "Electrochemical cells, batteries, corrosion",
        "icon": "battery",
        "subdisciplines": {
            "thermodynamics": {
                "name": "Electrochemical Thermodynamics",
                "signals": {
                    "required_any": ["voltage", "potential", "concentration"],
                },
                "constants": {
                    "optional": {
                        "standard_potential": {"unit": "V", "symbol": "E°"},
                        "electrons_transferred": {"unit": "dimensionless", "symbol": "n"},
                        "temperature": {"unit": "K", "symbol": "T", "default": 298.15},
                        "activity_product": {"unit": "dimensionless", "symbol": "Q"},
                    }
                },
                "engines": [
                    "nernst",
                    "cell_potential",
                    "gibbs_electrochemical",
                    "equilibrium_constant_echem",
                    "pourbaix",
                ],
            },
            "kinetics": {
                "name": "Electrode Kinetics",
                "signals": {
                    "required_any": ["current", "current_density", "overpotential"],
                },
                "constants": {
                    "optional": {
                        "exchange_current_density": {"unit": "A/m²", "symbol": "i_0"},
                        "transfer_coefficient": {"unit": "dimensionless", "symbol": "α", "default": 0.5},
                        "electrons_transferred": {"unit": "dimensionless", "symbol": "n"},
                        "electrode_area": {"unit": "m²", "symbol": "A"},
                        "tafel_slope": {"unit": "V/decade", "symbol": "b"},
                    }
                },
                "engines": [
                    "butler_volmer",
                    "tafel",
                    "limiting_current",
                    "mixed_potential",
                    "polarization_curve",
                ],
            },
            "mass_transfer": {
                "name": "Electrochemical Mass Transfer",
                "signals": {
                    "required_any": ["concentration", "current_density"],
                },
                "constants": {
                    "optional": {
                        "diffusivity": {"unit": "m²/s", "symbol": "D"},
                        "boundary_layer_thickness": {"unit": "m", "symbol": "δ"},
                        "bulk_concentration": {"unit": "mol/m³", "symbol": "C_b"},
                    }
                },
                "engines": [
                    "limiting_current_density",
                    "rotating_disk",
                    "concentration_overpotential",
                ],
            },
            "applications": {
                "name": "Electrochemical Applications",
                "signals": {
                    "required_any": ["current", "voltage", "mass", "time"],
                },
                "constants": {
                    "optional": {
                        "molecular_weight": {"unit": "kg/mol", "symbol": "M"},
                        "electrons_transferred": {"unit": "dimensionless", "symbol": "n"},
                        "current_efficiency": {"unit": "dimensionless", "symbol": "η_I", "default": 1.0},
                        "corrosion_rate_constant": {"unit": "various"},
                    }
                },
                "engines": [
                    "faraday",
                    "coulombic_efficiency",
                    "energy_efficiency",
                    "corrosion_rate",
                    "electroplating_thickness",
                    "electrolysis_power",
                ],
            },
            "batteries": {
                "name": "Battery Analysis",
                "signals": {
                    "required_any": ["voltage", "current", "capacity", "soc"],
                },
                "constants": {
                    "optional": {
                        "nominal_capacity": {"unit": "Ah", "symbol": "Q_nom"},
                        "nominal_voltage": {"unit": "V", "symbol": "V_nom"},
                        "internal_resistance": {"unit": "Ω", "symbol": "R_i"},
                        "open_circuit_voltage": {"unit": "V", "symbol": "OCV"},
                        "charge_transfer_resistance": {"unit": "Ω", "symbol": "R_ct"},
                        "diffusion_coefficient": {"unit": "m²/s", "symbol": "D"},
                    }
                },
                "engines": [
                    "state_of_charge",
                    "state_of_health",
                    "c_rate",
                    "peukert",
                    "internal_resistance_calc",
                    "impedance_spectrum",
                    "capacity_fade",
                    "cycle_life",
                ],
            },
        },
        "engines": [
            "faraday_constant",
        ],
    },
}


def get_discipline(name: str) -> dict:
    """Get discipline by name."""
    return DISCIPLINES.get(name)


def list_disciplines() -> list:
    """List all available discipline names."""
    return list(DISCIPLINES.keys())


def get_all_engines() -> list:
    """Get list of all engines across all disciplines."""
    engines = set()
    for disc in DISCIPLINES.values():
        engines.update(disc.get('engines', []))
        for sub in disc.get('subdisciplines', {}).values():
            engines.update(sub.get('engines', []))
    return sorted(engines)
