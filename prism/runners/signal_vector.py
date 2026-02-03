"""
Signal Vector Runner

Orchestration only:
1. Read manifest
2. Load observations
3. Run engines per window per signal
4. Output signal_vector.parquet

No computation here - engines do the math.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import polars as pl
import yaml


# Engine registry - maps engine name to module
ENGINE_MODULES = {
    # Statistics (from primitives.individual.statistics)
    'kurtosis': ('prism.engines.signal.basic_stats', 'compute_kurtosis'),
    'skewness': ('prism.engines.signal.basic_stats', 'compute_skewness'),
    'crest_factor': ('prism.engines.signal.basic_stats', 'compute_crest_factor'),

    # Spectral (from primitives.individual.spectral)
    'spectral': ('prism.engines.signal.spectral', 'compute'),
    'spectral_entropy': ('prism.engines.signal.spectral', 'compute'),

    # Entropy (from primitives.individual.entropy)
    'sample_entropy': ('prism.engines.signal.entropy', 'compute'),
    'perm_entropy': ('prism.engines.signal.entropy', 'compute'),

    # Hurst (from primitives.individual.fractal)
    'hurst': ('prism.engines.signal.hurst', 'compute'),
    'acf_decay': ('prism.engines.signal.hurst', 'compute'),
}


def get_engine(name: str):
    """Get engine function by name."""
    if name not in ENGINE_MODULES:
        return None

    module_path, func_name = ENGINE_MODULES[name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError):
        return None


def run_engine(name: str, values: np.ndarray) -> Dict[str, float]:
    """Run a single engine on values."""
    engine = get_engine(name)
    if engine is None:
        return {}
    try:
        return engine(values)
    except Exception:
        return {}


def sliding_windows(n: int, window_size: int, stride: int) -> List[tuple]:
    """Generate sliding window indices."""
    windows = []
    window_I = 0
    start = 0

    while start + window_size <= n:
        end = start + window_size
        windows.append((window_I, start, end))
        window_I += 1
        start += stride

    return windows


def read_manifest(manifest_path: str) -> Dict[str, Any]:
    """Read ORTHON manifest."""
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def run_signal_vector(
    manifest_path: str,
    observations_path: str = None,
    output_path: str = None,
) -> pl.DataFrame:
    """
    Run signal vector computation from manifest.

    Args:
        manifest_path: Path to manifest.yaml
        observations_path: Override observations path (optional)
        output_path: Path to write output (optional)

    Returns:
        Polars DataFrame with signal vectors
    """
    manifest_dir = Path(manifest_path).parent
    manifest = read_manifest(manifest_path)

    print(f"Loaded manifest v{manifest.get('version', '?')}")

    # Determine observations path
    if observations_path is None:
        paths = manifest.get('paths', {})
        observations_path = paths.get('observations')
        if observations_path and not Path(observations_path).is_absolute():
            observations_path = str(manifest_dir / observations_path)

    if observations_path is None:
        raise ValueError("No observations path in manifest or arguments")

    # Load observations
    print(f"Loading observations from {observations_path}")
    obs_df = pl.read_parquet(observations_path)

    # Check required columns
    for col in ['I', 'signal_id', 'value']:
        if col not in obs_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Detect cohort column
    cohort_col = 'cohort' if 'cohort' in obs_df.columns else None

    # Get cohorts from manifest
    cohorts = manifest.get('cohorts', {})

    all_results = []

    for cohort_name, signals in cohorts.items():
        for signal_id, config in signals.items():
            # Get signal config
            engines = config.get('engines', [])
            window_size = config.get('signal_window', config.get('window_size', 128))
            stride = config.get('signal_stride', config.get('stride', 64))

            print(f"Processing {cohort_name}/{signal_id}: {len(engines)} engines, "
                  f"window={window_size}, stride={stride}")

            # Filter observations for this signal
            if cohort_col:
                mask = (obs_df['signal_id'] == signal_id) & (obs_df[cohort_col] == cohort_name)
            else:
                mask = obs_df['signal_id'] == signal_id

            signal_obs = obs_df.filter(mask).sort('I')

            if len(signal_obs) == 0:
                print(f"  Warning: No observations for {cohort_name}/{signal_id}")
                continue

            if len(signal_obs) < window_size:
                print(f"  Warning: Only {len(signal_obs)} samples, need {window_size}")
                continue

            # Extract arrays
            values = signal_obs['value'].to_numpy()
            indices = signal_obs['I'].to_numpy()

            # Generate windows
            windows = sliding_windows(len(values), window_size, stride)

            for window_I, start, end in windows:
                window_values = values[start:end]

                # Run engines
                row = {
                    'signal_id': signal_id,
                    'cohort': cohort_name,
                    'window_I': window_I,
                    'window_start': int(indices[start]),
                    'window_end': int(indices[end - 1]),
                    'n_samples': end - start,
                }

                for engine_name in engines:
                    engine_results = run_engine(engine_name, window_values)
                    row.update(engine_results)

                all_results.append(row)

            print(f"  Generated {len(windows)} windows")

    # Convert to DataFrame
    if not all_results:
        print("Warning: No results generated")
        return pl.DataFrame()

    # Collect all unique keys
    all_keys = set()
    for row in all_results:
        all_keys.update(row.keys())

    # Ensure all rows have all keys
    for row in all_results:
        for key in all_keys:
            if key not in row:
                row[key] = None

    result_df = pl.DataFrame(all_results, infer_schema_length=None)

    print(f"\nTotal: {len(result_df)} rows, {len(result_df.columns)} columns")

    # Write output
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(out_path)
        print(f"Written to {out_path}")

    return result_df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute signal vectors from manifest')
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--observations', '-o', help='Override observations path')
    parser.add_argument('--output', '-O', help='Output parquet path')

    args = parser.parse_args()

    run_signal_vector(
        args.manifest,
        observations_path=args.observations,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
