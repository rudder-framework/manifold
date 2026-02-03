"""
Signal Vector Entry Point.

Thin orchestrator that:
1. Reads manifest
2. Loads observations
3. Calls appropriate engines per-signal with per-signal config
4. Writes output to parquet

Entry point does NOT contain compute logic - only orchestration.

================================================================================
WARNING: WINDOW LOGIC BELONGS IN THE MANIFEST - NOT HERE
================================================================================
DO NOT add any window size calculation, adaptive windowing, or default window
values to this file. All windowing parameters MUST come from the manifest.

If the manifest is missing window_size or stride, this entry point MUST fail.
No defaults. No fallbacks. No "smart" calculations.

ORTHON determines window parameters. PRISM executes what the manifest says.
Hardcoding window logic here wrecks everything downstream.
================================================================================
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Set
from collections import defaultdict


# =============================================================================
# ENGINE REQUIREMENTS
# =============================================================================
# Minimum samples required for each engine to produce valid results.
# Engines not listed default to 4 (absolute minimum for any computation).
#
# These define the "ideal" window size for accurate results. Engines receiving
# smaller windows will return NaN for those observations.
# =============================================================================

ENGINE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # FFT-based engines (need sufficient frequency resolution)
    'spectral': {'min_samples': 64},
    'harmonics': {'min_samples': 64},
    'fundamental_freq': {'min_samples': 64},
    'thd': {'min_samples': 64},
    'frequency_bands': {'min_samples': 64},
    'band_power': {'min_samples': 64},

    # Entropy engines (need sufficient data for pattern detection)
    'sample_entropy': {'min_samples': 64},
    'complexity': {'min_samples': 50},
    'approximate_entropy': {'min_samples': 30},
    'permutation_entropy': {'min_samples': 20},
    'perm_entropy': {'min_samples': 20},

    # Fractal/memory engines (need long series for scaling analysis)
    'hurst': {'min_samples': 128},
    'dfa': {'min_samples': 20},
    'memory': {'min_samples': 20},
    'acf_decay': {'min_samples': 16},

    # Statistical engines (low requirements)
    'statistics': {'min_samples': 4},
    'kurtosis': {'min_samples': 4},
    'skewness': {'min_samples': 4},
    'crest_factor': {'min_samples': 4},

    # Spectral analysis (moderate requirements)
    'snr': {'min_samples': 32},
    'phase_coherence': {'min_samples': 32},

    # Trend engines
    'trend': {'min_samples': 8},
    'mann_kendall': {'min_samples': 8},
    'rate_of_change': {'min_samples': 4},

    # Advanced engines
    'attractor': {'min_samples': 64},
    'lyapunov': {'min_samples': 128},
    'garch': {'min_samples': 64},
    'dmd': {'min_samples': 32},
    'envelope': {'min_samples': 16},
    'variance_growth': {'min_samples': 16},

    # Domain-specific
    'basin': {'min_samples': 32},
    'cycle_counting': {'min_samples': 16},
    'lof': {'min_samples': 20},
    'pulsation_index': {'min_samples': 8},
    'time_constant': {'min_samples': 16},
}

# Default minimum for unlisted engines
DEFAULT_MIN_SAMPLES = 4


def get_engine_min_samples(engine_name: str) -> int:
    """Get minimum samples required for an engine."""
    return ENGINE_REQUIREMENTS.get(engine_name, {}).get('min_samples', DEFAULT_MIN_SAMPLES)


def validate_engine_can_run(engine_name: str, window_size: int) -> bool:
    """Check if engine can run with given window size."""
    min_required = get_engine_min_samples(engine_name)
    return window_size >= min_required


def group_engines_by_window(
    engines: Dict[str, Callable],
    overrides: Dict[str, int],
    default_window: int,
) -> Dict[int, Dict[str, Callable]]:
    """
    Group engines by their required window size.

    Args:
        engines: Dict of {engine_name: engine_function}
        overrides: Dict of {engine_name: window_size} from manifest
        default_window: System default window size

    Returns:
        dict: {window_size: {engine_name: engine_function}}
    """
    groups: Dict[int, Dict[str, Callable]] = {}

    for engine_name, engine_fn in engines.items():
        # Check manifest override first, then engine requirements, then default
        if engine_name in overrides:
            window = overrides[engine_name]
        else:
            min_required = get_engine_min_samples(engine_name)
            window = max(default_window, min_required)

        if window not in groups:
            groups[window] = {}
        groups[window][engine_name] = engine_fn

    return groups


def _load_engine_registry() -> Dict[str, Callable]:
    """Load all signal engines. Each engine has a compute() method."""
    from prism.engines.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
        basin, cycle_counting, lof, pulsation_index, time_constant,
        rate_of_change, variance_growth,
        fundamental_freq, phase_coherence, snr, thd,
    )

    return {
        # Core engines
        'statistics': statistics.compute,
        'memory': memory.compute,
        'complexity': complexity.compute,
        'spectral': spectral.compute,
        'trend': trend.compute,

        # Individual statistic engines
        'kurtosis': statistics.compute_kurtosis,
        'skewness': statistics.compute_skewness,
        'crest_factor': statistics.compute_crest_factor,

        # Individual memory engines
        'hurst': hurst.compute,
        'dfa': memory.compute_dfa,
        'acf_decay': memory.compute_acf_decay,

        # Individual complexity engines
        'sample_entropy': complexity.compute_sample_entropy,
        'permutation_entropy': complexity.compute_permutation_entropy,
        'perm_entropy': complexity.compute_permutation_entropy,  # alias
        'approximate_entropy': complexity.compute_approximate_entropy,

        # Individual trend engines
        'mann_kendall': trend.compute_mann_kendall,
        'rate_of_change': trend.compute_rate_of_change,

        # Trend aliases (trend.compute returns these)
        'trend_r2': trend.compute,
        'detrend_std': trend.compute,
        'cusum': trend.compute,

        # Spectral aliases (spectral.compute returns these)
        'spectral_entropy': spectral.compute,

        # Frequency band aliases
        'band_power': frequency_bands.compute,
        'frequency_bands': frequency_bands.compute,

        # Advanced engines
        'attractor': attractor.compute,
        'lyapunov': lyapunov.compute,
        'garch': garch.compute,
        'dmd': dmd.compute,
        'envelope': envelope.compute,
        'harmonics': harmonics.compute,

        # Domain-specific engines
        'basin': basin.compute,
        'cycle_counting': cycle_counting.compute,
        'lof': lof.compute,
        'pulsation_index': pulsation_index.compute,
        'time_constant': time_constant.compute,

        # Rate of change (detailed version with mean_rate, max_rate, etc.)
        'rate_of_change_detailed': rate_of_change.compute,

        # Variance growth (non-stationarity detection)
        'variance_growth': variance_growth.compute,

        # Spectral analysis engines
        'fundamental_freq': fundamental_freq.compute,
        'phase_coherence': phase_coherence.compute,
        'snr': snr.compute,
        'thd': thd.compute,
    }


def _validate_engines(
    engine_names: List[str],
    registry: Dict[str, Callable]
) -> Tuple[List[str], List[str]]:
    """Validate engine names against registry. Returns (valid, unknown)."""
    valid = [name for name in engine_names if name in registry]
    unknown = [name for name in engine_names if name not in registry]
    return valid, unknown


def _diagnose_manifest_engines(
    manifest: Dict[str, Any],
    registry: Dict[str, Callable]
) -> Dict[str, Any]:
    """Check all manifest engines against registry."""
    all_engines: Set[str] = set()
    for cohort_signals in manifest.get('cohorts', {}).values():
        for signal_config in cohort_signals.values():
            all_engines.update(signal_config.get('engines', []))

    available = [e for e in all_engines if e in registry]
    missing = [e for e in all_engines if e not in registry]
    coverage = len(available) / len(all_engines) if all_engines else 1.0

    return {
        'available': sorted(available),
        'missing': sorted(missing),
        'coverage': coverage,
        'total_requested': len(all_engines),
    }


def run(
    observations_path: str,
    output_path: str,
    manifest: Dict[str, Any],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector computation.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path to write signal_vector.parquet
        manifest: Manifest dict from ORTHON (REQUIRED)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    _validate_manifest(manifest)

    # Get defaults from params
    params = manifest.get('params', {})
    default_window = params.get('default_window')
    default_stride = params.get('default_stride')

    # Load engine registry
    engine_registry = _load_engine_registry()

    # Validate engines upfront (PR13)
    diagnosis = _diagnose_manifest_engines(manifest, engine_registry)
    if verbose:
        if diagnosis['missing']:
            print(f"WARNING: Missing engines: {diagnosis['missing']}")
            print(f"  Coverage: {diagnosis['coverage']:.1%} ({len(diagnosis['available'])}/{diagnosis['total_requested']})")
        else:
            print(f"All {diagnosis['total_requested']} engines available")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_signals = obs['signal_id'].n_unique()
        n_obs = len(obs)
        print(f"Loaded {n_obs:,} observations across {n_signals} signals")

    # Process each signal according to its manifest config
    results = []
    error_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cohort_name, cohort_signals in manifest['cohorts'].items():
        for signal_id, signal_config in cohort_signals.items():
            # Get per-signal config from manifest
            window_size = signal_config.get('window_size') or default_window
            stride = signal_config.get('stride') or default_stride
            engine_names = signal_config.get('engines', [])
            engine_window_overrides = signal_config.get('engine_window_overrides', {})

            if window_size is None:
                raise ValueError(f"No window_size for signal '{signal_id}' and no default_window in params")
            if stride is None:
                raise ValueError(f"No stride for signal '{signal_id}' and no default_stride in params")
            if not engine_names:
                if verbose:
                    print(f"  Skipping {signal_id}: no engines specified")
                continue

            # Validate engines for this signal (PR13)
            valid_engines, unknown_engines = _validate_engines(engine_names, engine_registry)

            if unknown_engines and verbose:
                print(f"  {signal_id}: skipping unknown engines: {unknown_engines}")

            # Get engine functions
            active_engines = {
                name: engine_registry[name]
                for name in valid_engines
            }

            if verbose:
                print(f"  {signal_id}: window={window_size}, stride={stride}, engines={list(active_engines.keys())}")

            # Get signal data
            signal_data = (
                obs
                .filter(pl.col('signal_id') == signal_id)
                .sort('I')
            )

            if len(signal_data) == 0:
                if verbose:
                    print(f"    Warning: no data for signal '{signal_id}'")
                continue

            values = signal_data['value'].to_numpy()
            indices = signal_data['I'].to_numpy()

            # Compute features at each window (PR2: per-engine window expansion)
            signal_results, signal_errors = _compute_signal_features(
                signal_id=signal_id,
                values=values,
                indices=indices,
                engines=active_engines,
                window_size=window_size,
                stride=stride,
                engine_window_overrides=engine_window_overrides,
            )

            # Track errors (PR13)
            for engine_name, count in signal_errors.items():
                error_summary[signal_id][engine_name] += count

            # Convert to DataFrame immediately to preserve column schema
            if signal_results:
                signal_df = pl.DataFrame(signal_results)
                results.append(signal_df)

    # Report error summary (PR13)
    if verbose and error_summary:
        print("\nEngine errors:")
        engine_totals: Dict[str, int] = defaultdict(int)
        for signal_id, engines in error_summary.items():
            for engine, count in engines.items():
                engine_totals[engine] += count
        for engine, count in sorted(engine_totals.items()):
            print(f"  {engine}: {count} failures")

    # Concat all signal DataFrames (handles different column sets properly)
    if not results:
        df = pl.DataFrame()
    elif len(results) == 1:
        df = results[0]
    else:
        df = pl.concat(results, how='diagonal')

    # Write output
    df.write_parquet(output_path)

    if verbose:
        print(f"Wrote {len(df):,} rows to {output_path}")

    return df


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required structure."""
    if 'cohorts' not in manifest:
        raise ValueError("Manifest missing 'cohorts' section.")

    if not manifest['cohorts']:
        raise ValueError("Manifest 'cohorts' is empty.")


def _compute_signal_features(
    signal_id: str,
    values: np.ndarray,
    indices: np.ndarray,
    engines: Dict[str, Callable],
    window_size: int,
    stride: int,
    engine_window_overrides: Dict[str, int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute features for one signal with per-engine window support.

    Per-engine window expansion (PR2):
    - Engines may require larger windows than the system default
    - Each engine gets the window size it needs, but all share the same I (end index)
    - Early observations may have NaN for engines requiring larger windows

    Args:
        signal_id: Signal identifier
        values: Signal values array
        indices: I indices array
        engines: Dict of {engine_name: engine_function}
        window_size: System default window size
        stride: Window stride
        engine_window_overrides: Optional manifest overrides {engine: window_size}

    Returns:
        (results_list, error_counts) - PR13: track errors instead of silent pass
    """
    n = len(values)
    results = []
    error_counts: Dict[str, int] = defaultdict(int)

    if engine_window_overrides is None:
        engine_window_overrides = {}

    # Group engines by their required window size
    engine_groups = group_engines_by_window(engines, engine_window_overrides, window_size)

    # Find the maximum window size needed (determines when we start)
    max_window = max(engine_groups.keys()) if engine_groups else window_size

    # Track which engines need NaN for early windows
    # (when not enough data for their required window)
    def _get_null_output(engine_name: str, engine_fn: Callable) -> Dict[str, float]:
        """Get NaN output for an engine that can't run."""
        # Try to call with minimal data to get output keys, then replace with NaN
        try:
            # Call with small array to get structure
            sample_output = engine_fn(np.array([0.0, 0.0, 0.0, 0.0]))
            return {k: np.nan for k in sample_output.keys()}
        except Exception:
            return {}

    # Iterate at system stride, starting when at least system window is available
    for window_end_pos in range(window_size - 1, n, stride):
        idx = indices[window_end_pos]

        row = {
            'signal_id': signal_id,
            'I': int(idx),
        }

        # Run each engine group with appropriate window
        for req_window_size, engine_dict in engine_groups.items():
            # Calculate window start for this window size
            # All windows END at window_end_pos, but start at different positions
            window_start_pos = window_end_pos - req_window_size + 1

            # Check if we have enough data for this window
            if window_start_pos < 0:
                # Not enough data yet - fill with NaN for these engines
                for name, engine_fn in engine_dict.items():
                    null_output = _get_null_output(name, engine_fn)
                    row.update(null_output)
                continue

            # Extract window for this group
            window = values[window_start_pos:window_end_pos + 1]

            # Run engines in this group
            for name, engine_fn in engine_dict.items():
                try:
                    output = engine_fn(window)
                    for key, val in output.items():
                        row[key] = val
                except Exception:
                    error_counts[name] += 1

        results.append(row)

    return results, dict(error_counts)


def run_from_manifest(
    manifest_path: str,
    data_dir: str = None,
    output_dir: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector from manifest file.

    Args:
        manifest_path: Path to manifest.yaml
        data_dir: Directory with observations.parquet (optional, derived from manifest)
        output_dir: Directory for output (optional, derived from manifest)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    import yaml

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Derive paths from manifest if not provided
    if data_dir is None:
        # Use manifest's paths.observations or default to manifest directory
        obs_rel = manifest.get('paths', {}).get('observations', 'observations.parquet')
        obs_path = manifest_dir / obs_rel
    else:
        obs_path = Path(data_dir) / 'observations.parquet'

    if output_dir is None:
        # Use manifest directory for output
        out_path = manifest_dir / 'signal_vector.parquet'
    else:
        out_path = Path(output_dir) / 'signal_vector.parquet'

    return run(
        observations_path=str(obs_path),
        output_path=str(out_path),
        manifest=manifest,
        verbose=verbose,
    )


def main():
    """CLI entry point: python -m prism.entry_points.signal_vector <manifest.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute signal vectors from manifest',
        usage='python -m prism.entry_points.signal_vector <manifest.yaml>'
    )
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run_from_manifest(
        manifest_path=args.manifest,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
