"""
Unit metadata for Parquet file-level key-value metadata.

Each function returns a dict suitable for df.write_parquet(metadata={...}).
Keys use the `manifold.` prefix so they're easy to filter in DuckDB/PyArrow.
"""

import json


def _base_meta(s0_name: str, s0_unit: str, stage: str) -> dict:
    """Common manifold.* keys shared by all target files."""
    return {
        "manifold.signal_0_name": s0_name or "signal_0",
        "manifold.signal_0_unit": s0_unit or "step",
        "manifold.stage": stage,
    }


def velocity_field_units(
    s0_name: str = "",
    s0_unit: str = "",
    stage: str = "21",
) -> dict:
    """Metadata for velocity_field.parquet (stage 21).

    Inputs are z-scored so value unit = σ (standard deviations).
    """
    u = s0_unit or "step"
    col_units = {
        "speed": f"σ/{u}",
        "acceleration_magnitude": f"σ/{u}²",
        "acceleration_parallel": f"σ/{u}²",
        "acceleration_perpendicular": f"σ/{u}²",
        "curvature": "1/σ",
        "dominant_motion_fraction": "dimensionless",
        "motion_dimensionality": "dimensionless",
    }
    meta = _base_meta(s0_name, s0_unit, stage)
    meta["manifold.column_units"] = json.dumps(col_units)
    return meta


def geometry_dynamics_units(
    s0_name: str = "",
    s0_unit: str = "",
    stage: str = "07",
) -> dict:
    """Metadata for geometry_dynamics.parquet (stage 07)."""
    u = s0_unit or "step"
    col_units = {
        "effective_dim": "dimensionless",
        "effective_dim_curvature": "dimensionless",
        "effective_dim_velocity": f"1/{u}",
        "effective_dim_acceleration": f"1/{u}²",
        "effective_dim_jerk": f"1/{u}³",
        "eigenvalue_1": "σ²",
        "total_variance": "σ²",
        "eigenvalue_1_velocity": f"σ²/{u}",
        "variance_velocity": f"σ²/{u}",
        "collapse_onset_idx": "index",
        "collapse_onset_fraction": "dimensionless",
    }
    meta = _base_meta(s0_name, s0_unit, stage)
    meta["manifold.column_units"] = json.dumps(col_units)
    return meta


def cohort_geometry_units(
    s0_name: str = "",
    s0_unit: str = "",
    stage: str = "03",
) -> dict:
    """Metadata for cohort_geometry.parquet (stage 03)."""
    col_units = {
        "eigenvalue_1": "σ²",
        "eigenvalue_2": "σ²",
        "eigenvalue_3": "σ²",
        "eigenvalue_4": "σ²",
        "eigenvalue_5": "σ²",
        "total_variance": "σ²",
        "explained_1": "dimensionless",
        "explained_2": "dimensionless",
        "explained_3": "dimensionless",
        "explained_4": "dimensionless",
        "explained_5": "dimensionless",
        "effective_dim": "dimensionless",
        "eigenvalue_entropy": "nats",
        "eigenvalue_entropy_norm": "dimensionless",
        "condition_number": "dimensionless",
        "ratio_2_1": "dimensionless",
        "ratio_3_1": "dimensionless",
        "n_signals": "count",
        "n_features": "count",
        "eff_dim_std": "dimensionless",
        "eff_dim_ci_low": "dimensionless",
        "eff_dim_ci_high": "dimensionless",
    }
    meta = _base_meta(s0_name, s0_unit, stage)
    meta["manifold.column_units"] = json.dumps(col_units)
    return meta
