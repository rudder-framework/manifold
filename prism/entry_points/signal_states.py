#!/usr/bin/env python3
"""
Signal States Entry Point
=========================

Computes unified signal states from all ORTHON analytical layers.

This combines outputs from:
- Signal Typology (signal_typology_profile.parquet)
- Manifold Geometry (manifold_geometry.parquet)
- Dynamical Systems (state.parquet)
- Causal Mechanics (causal_mechanics.parquet)

Into a unified view:
- data/signal_states.parquet

Each signal is tracked through four layers at each window, producing:
- typology_state: "persistent|periodic|clustered"
- geometry_state: "MODULAR.STABLE.CLEAR_LEADER"
- dynamics_state: "COUPLED.EVOLVING.CONVERGING.FIXED_POINT"
- mechanics_state: "CONSERVATIVE.APPROACHING.LAMINAR.CIRCULAR"

Usage:
    python -m prism.entry_points.signal_states
    python -m prism.entry_points.signal_states --force
    python -m prism.entry_points.signal_states --no-validate
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Compute Unified Signal States")
    parser.add_argument("--force", action="store_true", help="Recompute all states")
    parser.add_argument("--no-validate", action="store_true", help="Skip mechanics validation")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--show-transitions", action="store_true", help="Show state transitions")
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from prism.db.parquet_store import get_data_root
        data_dir = get_data_root()

    output_path = data_dir / "signal_states.parquet"

    print("=" * 60)
    print("SIGNAL STATES COMPUTATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()

    # Check for required input files
    input_files = {
        "signal_typology_profile.parquet": "Signal Typology",
        "manifold_geometry.parquet": "Manifold Geometry",
        "state.parquet": "Dynamical Systems",
        "causal_mechanics.parquet": "Causal Mechanics",
    }

    available_layers = []
    for filename, layer_name in input_files.items():
        path = data_dir / filename
        if path.exists():
            df = pl.read_parquet(path)
            print(f"  {layer_name}: {len(df):,} rows")
            available_layers.append(layer_name)
        else:
            print(f"  {layer_name}: NOT FOUND ({filename})")

    if not available_layers:
        print("\nERROR: No layer outputs found")
        print("Run the individual frameworks first:")
        print("  python -m prism.entry_points.signal_typology")
        print("  python -m prism.entry_points.structural_geometry")
        print("  python -m prism.entry_points.dynamical_systems")
        print("  python -m prism.entry_points.causal_mechanics")
        sys.exit(1)

    print(f"\n{len(available_layers)} of 4 layers available")

    # Check for existing output
    if output_path.exists() and not args.force:
        existing = pl.read_parquet(output_path)
        print(f"\nExisting signal_states.parquet: {len(existing):,} rows")
        print("Use --force to recompute")

        # Show summary
        _print_summary(existing)
        return

    print("\nComputing signal states...")

    # Import orchestrator
    from prism.signal_states.orchestrator import (
        run_signal_states,
        detect_state_transitions,
    )

    # Run computation
    validate = not args.no_validate
    states_df = run_signal_states(
        data_dir=data_dir,
        validate_mechanics=validate,
    )

    if states_df.is_empty():
        print("\nNo signal states computed - check input files")
        return

    # Write output
    print(f"\nWriting {len(states_df):,} rows to {output_path}")
    states_df.write_parquet(output_path)

    # Print summary
    _print_summary(states_df)

    # Show transitions if requested
    if args.show_transitions:
        print("\n" + "-" * 60)
        print("STATE TRANSITIONS")
        print("-" * 60)

        transitions = detect_state_transitions(states_df)

        if not transitions:
            print("  No state transitions detected")
        else:
            # Group by alert level
            warnings = [t for t in transitions if t.alert_level == "warning"]
            info = [t for t in transitions if t.alert_level == "info"]

            if warnings:
                print(f"\n  WARNINGS ({len(warnings)}):")
                for t in warnings[:10]:  # Show first 10
                    print(f"    {t.signal_id} @ {t.unit_id}: window {t.from_window} -> {t.to_window}")
                    if t.mechanics_changed:
                        print(f"      Mechanics: {t.prev_mechanics} -> {t.new_mechanics}")
                    print(f"      {t.explanation}")

            print(f"\n  INFO ({len(info)} transitions)")

    # Validation summary
    if validate:
        print("\n" + "-" * 60)
        print("MECHANICS VALIDATION")
        print("-" * 60)

        n_total = len(states_df)
        n_stable = states_df.filter(pl.col("mechanics_stable") == True).height
        n_unstable = n_total - n_stable

        print(f"  Stable: {n_stable:,} ({100*n_stable/n_total:.1f}%)")
        print(f"  Unstable: {n_unstable:,} ({100*n_unstable/n_total:.1f}%)")

        if n_unstable > 0:
            print("\n  Unstable states (first 5):")
            unstable = states_df.filter(pl.col("mechanics_stable") == False).head(5)
            for row in unstable.iter_rows(named=True):
                print(f"    {row['signal_id']} @ {row['unit_id']} window {row['window_idx']}")
                print(f"      {row['stability_notes']}")

    print("\n" + "=" * 60)
    print("SIGNAL STATES COMPLETE")
    print("=" * 60)


def _print_summary(df: pl.DataFrame) -> None:
    """Print summary of signal states DataFrame."""
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    # Counts
    n_signals = df["signal_id"].n_unique()
    n_units = df["unit_id"].n_unique()
    n_windows = df["window_idx"].n_unique() if "window_idx" in df.columns else 0

    print(f"  Signals: {n_signals:,}")
    print(f"  Units: {n_units:,}")
    print(f"  Windows: {n_windows:,}")
    print(f"  Total records: {len(df):,}")

    # State distributions
    for state_col in ["typology_state", "geometry_state", "dynamics_state", "mechanics_state"]:
        if state_col in df.columns:
            non_empty = df.filter(pl.col(state_col) != "")
            if not non_empty.is_empty():
                unique_states = non_empty[state_col].n_unique()
                print(f"\n  {state_col}:")
                print(f"    Unique states: {unique_states}")

                # Top 3 states
                top_states = (
                    non_empty
                    .group_by(state_col)
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                    .head(3)
                )
                for row in top_states.iter_rows(named=True):
                    pct = 100 * row["count"] / len(non_empty)
                    state_str = row[state_col][:40] + "..." if len(row[state_col]) > 40 else row[state_col]
                    print(f"      {state_str}: {row['count']:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
