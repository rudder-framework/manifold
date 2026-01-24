"""
Signal States Orchestrator
==========================

Computes unified signal states by aligning outputs from all four ORTHON layers.

Windowing Alignment:
    | Layer             | Granularity       | Alignment Strategy        |
    |-------------------|-------------------|---------------------------|
    | Signal Typology   | Per signal        | Compute per-window (new)  |
    | Manifold Geometry| Per unit per window| Direct lookup            |
    | Dynamical Systems | Per unit (aggregated)| Propagate to all windows|
    | Causal Mechanics  | Per signal per window| Direct 1:1              |

Primary key: (signal_id, unit_id, window_idx)
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import polars as pl

from .models import SignalState, StateTransition
from .state_builders import (
    compute_typology_state,
    geometry_state_from_output,
    dynamics_state_from_output,
    mechanics_state_from_output,
)
from .validation import validate_mechanics_stability, get_mechanics_stability_level


def compute_signal_states(
    typology_df: Optional[pl.DataFrame] = None,
    geometry_df: Optional[pl.DataFrame] = None,
    dynamics_df: Optional[pl.DataFrame] = None,
    mechanics_df: Optional[pl.DataFrame] = None,
    validate_mechanics: bool = True,
) -> pl.DataFrame:
    """
    Compute unified signal states from layer outputs.

    Args:
        typology_df: Signal typology profile DataFrame
        geometry_df: Structural geometry output DataFrame
        dynamics_df: Dynamical systems output DataFrame
        mechanics_df: Causal mechanics output DataFrame
        validate_mechanics: Whether to validate mechanics stability

    Returns:
        DataFrame with unified signal states
    """
    now = datetime.now()
    rows = []

    # Build lookup dicts for each layer
    typology_lookup = _build_typology_lookup(typology_df)
    geometry_lookup = _build_geometry_lookup(geometry_df)
    dynamics_lookup = _build_dynamics_lookup(dynamics_df)
    mechanics_lookup = _build_mechanics_lookup(mechanics_df)

    # Get all unique (signal_id, unit_id, window_idx) combinations
    all_keys = set()
    all_keys.update(typology_lookup.keys())
    all_keys.update(geometry_lookup.keys())
    all_keys.update(mechanics_lookup.keys())

    # Also expand dynamics (which may be per-unit only)
    for key in list(all_keys):
        signal_id, unit_id, window_idx = key
        # Dynamics is keyed by unit_id only - will match all windows
        pass

    # Track previous states for validation
    previous_mechanics: Dict[Tuple[str, str], str] = {}

    # Sort by window_idx for sequential validation
    sorted_keys = sorted(all_keys, key=lambda k: (k[1], k[0], k[2]))  # unit, signal, window

    for signal_id, unit_id, window_idx in sorted_keys:
        key = (signal_id, unit_id, window_idx)

        # Get timestamp from any available source
        timestamp = _get_timestamp(
            key, typology_df, geometry_df, dynamics_df, mechanics_df
        )

        # Build state strings
        typology_state = typology_lookup.get(key, "")
        geometry_state = geometry_lookup.get(key, "")

        # Dynamics may be keyed by unit only
        dynamics_state = dynamics_lookup.get(key, "")
        if not dynamics_state:
            # Try unit-level lookup
            dynamics_state = dynamics_lookup.get(("", unit_id, -1), "")

        mechanics_state = mechanics_lookup.get(key, "")

        # Validate mechanics stability
        mechanics_stable = True
        stability_notes = ""
        prev_key = (signal_id, unit_id)

        if validate_mechanics and prev_key in previous_mechanics:
            prev_mechanics = previous_mechanics[prev_key]
            mechanics_stable, stability_notes = validate_mechanics_stability(
                mechanics_state, prev_mechanics
            )

        # Update previous state
        if mechanics_state:
            previous_mechanics[prev_key] = mechanics_state

        # Create state record
        state = SignalState(
            signal_id=signal_id,
            unit_id=unit_id,
            window_idx=window_idx,
            timestamp=timestamp,
            typology_state=typology_state,
            geometry_state=geometry_state,
            dynamics_state=dynamics_state,
            mechanics_state=mechanics_state,
            mechanics_stable=mechanics_stable,
            stability_notes=stability_notes,
            computed_at=now,
        )
        state.compute_hash()
        rows.append(state.to_dict())

    if not rows:
        # Return empty DataFrame with correct schema
        return pl.DataFrame({
            'signal_id': pl.Series([], dtype=pl.Utf8),
            'unit_id': pl.Series([], dtype=pl.Utf8),
            'window_idx': pl.Series([], dtype=pl.Int64),
            'timestamp': pl.Series([], dtype=pl.Datetime),
            'typology_state': pl.Series([], dtype=pl.Utf8),
            'geometry_state': pl.Series([], dtype=pl.Utf8),
            'dynamics_state': pl.Series([], dtype=pl.Utf8),
            'mechanics_state': pl.Series([], dtype=pl.Utf8),
            'mechanics_stable': pl.Series([], dtype=pl.Boolean),
            'stability_notes': pl.Series([], dtype=pl.Utf8),
            'state_hash': pl.Series([], dtype=pl.Utf8),
            'computed_at': pl.Series([], dtype=pl.Datetime),
        })

    return pl.DataFrame(rows)


def _build_typology_lookup(
    df: Optional[pl.DataFrame]
) -> Dict[Tuple[str, str, int], str]:
    """Build lookup from typology profile DataFrame."""
    if df is None or df.is_empty():
        return {}

    lookup = {}
    # Typology profile has axis scores - convert to state string
    axis_cols = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
                 'derivatives', 'recurrence', 'discontinuity', 'momentum']

    # Handle entity_id / unit_id
    unit_col = 'unit_id' if 'unit_id' in df.columns else 'entity_id'

    for row in df.iter_rows(named=True):
        signal_id = row.get('signal_id', '')
        unit_id = row.get(unit_col, '')
        window_idx = row.get('window_idx', 0)

        # Build profile dict from row
        profile = {ax: row.get(ax, 0.5) for ax in axis_cols if ax in row}

        if profile:
            state = compute_typology_state(profile)
            lookup[(signal_id, unit_id, window_idx)] = state

    return lookup


def _build_geometry_lookup(
    df: Optional[pl.DataFrame]
) -> Dict[Tuple[str, str, int], str]:
    """Build lookup from geometry DataFrame."""
    if df is None or df.is_empty():
        return {}

    lookup = {}
    unit_col = 'unit_id' if 'unit_id' in df.columns else 'entity_id'

    for row in df.iter_rows(named=True):
        unit_id = row.get(unit_col, '')
        window_idx = row.get('window_idx', 0)

        state = geometry_state_from_output(row)

        # Geometry is per-unit, apply to all signals for that unit
        # Use empty signal_id as wildcard
        lookup[("", unit_id, window_idx)] = state

    return lookup


def _build_dynamics_lookup(
    df: Optional[pl.DataFrame]
) -> Dict[Tuple[str, str, int], str]:
    """Build lookup from dynamics DataFrame."""
    if df is None or df.is_empty():
        return {}

    lookup = {}
    unit_col = 'unit_id' if 'unit_id' in df.columns else 'entity_id'

    for row in df.iter_rows(named=True):
        unit_id = row.get(unit_col, '')
        # Dynamics may not have window_idx (aggregated)
        window_idx = row.get('window_idx', -1)

        state = dynamics_state_from_output(row)

        # Use -1 window_idx for unit-level (propagate to all windows)
        lookup[("", unit_id, window_idx)] = state

    return lookup


def _build_mechanics_lookup(
    df: Optional[pl.DataFrame]
) -> Dict[Tuple[str, str, int], str]:
    """Build lookup from mechanics DataFrame."""
    if df is None or df.is_empty():
        return {}

    lookup = {}
    unit_col = 'unit_id' if 'unit_id' in df.columns else 'entity_id'

    for row in df.iter_rows(named=True):
        signal_id = row.get('signal_id', '')
        unit_id = row.get(unit_col, '')
        window_idx = row.get('window_idx', 0)

        state = mechanics_state_from_output(row)
        lookup[(signal_id, unit_id, window_idx)] = state

    return lookup


def _get_timestamp(
    key: Tuple[str, str, int],
    typology_df: Optional[pl.DataFrame],
    geometry_df: Optional[pl.DataFrame],
    dynamics_df: Optional[pl.DataFrame],
    mechanics_df: Optional[pl.DataFrame],
) -> datetime:
    """Get timestamp for a key from any available DataFrame."""
    signal_id, unit_id, window_idx = key
    unit_col_options = ['unit_id', 'entity_id']

    for df in [mechanics_df, typology_df, geometry_df, dynamics_df]:
        if df is None or df.is_empty():
            continue

        unit_col = 'unit_id' if 'unit_id' in df.columns else 'entity_id'

        if 'signal_id' in df.columns:
            filtered = df.filter(
                (pl.col('signal_id') == signal_id) &
                (pl.col(unit_col) == unit_id) &
                (pl.col('window_idx') == window_idx)
            )
        else:
            filtered = df.filter(
                (pl.col(unit_col) == unit_id) &
                (pl.col('window_idx') == window_idx)
            )

        if not filtered.is_empty() and 'timestamp' in filtered.columns:
            ts = filtered.select('timestamp').item()
            if ts is not None:
                return ts

    return datetime.now()


def detect_state_transitions(
    states_df: pl.DataFrame
) -> List[StateTransition]:
    """
    Detect state transitions between consecutive windows.

    Args:
        states_df: DataFrame from compute_signal_states

    Returns:
        List of StateTransition objects
    """
    transitions = []

    if states_df.is_empty():
        return transitions

    # Sort by signal, unit, window
    sorted_df = states_df.sort(['signal_id', 'unit_id', 'window_idx'])

    prev_row = None
    for row in sorted_df.iter_rows(named=True):
        if prev_row is not None:
            # Check if same signal/unit
            if (row['signal_id'] == prev_row['signal_id'] and
                row['unit_id'] == prev_row['unit_id']):

                # Check for changes
                typ_changed = row['typology_state'] != prev_row['typology_state']
                geo_changed = row['geometry_state'] != prev_row['geometry_state']
                dyn_changed = row['dynamics_state'] != prev_row['dynamics_state']
                mech_changed = row['mechanics_state'] != prev_row['mechanics_state']

                if typ_changed or geo_changed or dyn_changed or mech_changed:
                    transition = StateTransition(
                        signal_id=row['signal_id'],
                        unit_id=row['unit_id'],
                        from_window=prev_row['window_idx'],
                        to_window=row['window_idx'],
                        typology_changed=typ_changed,
                        geometry_changed=geo_changed,
                        dynamics_changed=dyn_changed,
                        mechanics_changed=mech_changed,
                        prev_typology=prev_row['typology_state'],
                        prev_geometry=prev_row['geometry_state'],
                        prev_dynamics=prev_row['dynamics_state'],
                        prev_mechanics=prev_row['mechanics_state'],
                        new_typology=row['typology_state'],
                        new_geometry=row['geometry_state'],
                        new_dynamics=row['dynamics_state'],
                        new_mechanics=row['mechanics_state'],
                    )

                    # Determine alert level
                    if mech_changed and not row.get('mechanics_stable', True):
                        transition.is_expected = False
                        transition.alert_level = "warning"
                        transition.explanation = row.get('stability_notes', '')
                    elif mech_changed:
                        transition.alert_level = "info"
                        transition.explanation = "Mechanics state evolved"

                    transitions.append(transition)

        prev_row = row

    return transitions


def load_layer_data(data_dir: Path) -> Tuple[
    Optional[pl.DataFrame],
    Optional[pl.DataFrame],
    Optional[pl.DataFrame],
    Optional[pl.DataFrame]
]:
    """
    Load layer data from parquet files.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (typology_df, geometry_df, dynamics_df, mechanics_df)
    """
    typology_path = data_dir / "signal_typology_profile.parquet"
    geometry_path = data_dir / "manifold_geometry.parquet"
    dynamics_path = data_dir / "state.parquet"  # state.parquet contains dynamics
    mechanics_path = data_dir / "causal_mechanics.parquet"

    typology_df = pl.read_parquet(typology_path) if typology_path.exists() else None
    geometry_df = pl.read_parquet(geometry_path) if geometry_path.exists() else None
    dynamics_df = pl.read_parquet(dynamics_path) if dynamics_path.exists() else None
    mechanics_df = pl.read_parquet(mechanics_path) if mechanics_path.exists() else None

    return typology_df, geometry_df, dynamics_df, mechanics_df


def run_signal_states(
    data_dir: Optional[Path] = None,
    validate_mechanics: bool = True,
) -> pl.DataFrame:
    """
    Run full signal states computation from data directory.

    Args:
        data_dir: Path to data directory (defaults to PRISM_DATA_PATH)
        validate_mechanics: Whether to validate mechanics stability

    Returns:
        DataFrame with unified signal states
    """
    if data_dir is None:
        from ..db.parquet_store import get_data_root
        data_dir = get_data_root()

    typology_df, geometry_df, dynamics_df, mechanics_df = load_layer_data(data_dir)

    return compute_signal_states(
        typology_df=typology_df,
        geometry_df=geometry_df,
        dynamics_df=dynamics_df,
        mechanics_df=mechanics_df,
        validate_mechanics=validate_mechanics,
    )
