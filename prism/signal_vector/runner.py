"""
Signal Vector Runner
=====================

Reads manifest, loads observations, runs engines per window per signal.

Output: signal_vector.parquet
- signal_id
- cohort
- window_I (window index)
- window_start (first I in window)
- window_end (last I in window)
- <engine outputs...>
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import polars as pl

from .manifest_reader import ManifestReader, SignalConfig
from .engines import run_engines, list_engines


def sliding_windows(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> List[tuple]:
    """
    Generate sliding window indices.
    
    Returns: List of (window_I, start_idx, end_idx)
    """
    n = len(values)
    windows = []
    
    window_I = 0
    start = 0
    
    while start + window_size <= n:
        end = start + window_size
        windows.append((window_I, start, end))
        window_I += 1
        start += stride
    
    return windows


def process_signal(
    signal_id: str,
    cohort: str,
    values: np.ndarray,
    indices: np.ndarray,
    config: SignalConfig,
) -> List[Dict[str, Any]]:
    """
    Process a single signal through windowed engine computation.
    
    Args:
        signal_id: Signal identifier
        cohort: Cohort identifier
        values: Signal values array
        indices: Original I values
        config: Signal configuration from manifest
        
    Returns:
        List of dicts, one per window
    """
    results = []
    
    windows = sliding_windows(values, config.window_size, config.stride)
    
    for window_I, start, end in windows:
        window_values = values[start:end]
        
        # Run engines
        engine_results = run_engines(config.engines, window_values)
        
        # Build row
        row = {
            'signal_id': signal_id,
            'cohort': cohort,
            'window_I': window_I,
            'window_start': int(indices[start]),
            'window_end': int(indices[end - 1]),
            'n_samples': end - start,
        }
        row.update(engine_results)
        
        results.append(row)
    
    return results


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
    # Read manifest
    reader = ManifestReader(manifest_path)
    print(f"Loaded manifest: {reader}")
    
    # Determine observations path
    if observations_path is None:
        observations_path = reader.observations_path
    
    if observations_path is None:
        raise ValueError("No observations path in manifest or arguments")
    
    # Make path relative to manifest if needed
    manifest_dir = Path(manifest_path).parent
    obs_path = Path(observations_path)
    if not obs_path.is_absolute():
        obs_path = manifest_dir / obs_path
    
    # Load observations
    print(f"Loading observations from {obs_path}")
    obs_df = pl.read_parquet(obs_path)
    
    # Check required columns
    required = ['I', 'signal_id', 'value']
    for col in required:
        if col not in obs_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Detect cohort column
    cohort_col = 'cohort' if 'cohort' in obs_df.columns else None
    
    # Process each signal
    all_results = []
    
    for config in reader.iter_signals():
        signal_id = config.signal_id
        cohort = config.cohort
        
        print(f"Processing {cohort}/{signal_id}: {len(config.engines)} engines, "
              f"window={config.window_size}, stride={config.stride}")
        
        # Filter observations for this signal
        if cohort_col:
            mask = (obs_df['signal_id'] == signal_id) & (obs_df[cohort_col] == cohort)
        else:
            mask = obs_df['signal_id'] == signal_id
        
        signal_obs = obs_df.filter(mask).sort('I')
        
        if len(signal_obs) == 0:
            print(f"  Warning: No observations for {cohort}/{signal_id}")
            continue
        
        if len(signal_obs) < config.window_size:
            print(f"  Warning: Only {len(signal_obs)} samples, need {config.window_size}")
            continue
        
        # Extract arrays
        values = signal_obs['value'].to_numpy()
        indices = signal_obs['I'].to_numpy()
        
        # Process
        signal_results = process_signal(
            signal_id, cohort, values, indices, config
        )
        
        print(f"  Generated {len(signal_results)} windows")
        all_results.extend(signal_results)
    
    # Convert to DataFrame
    if not all_results:
        print("Warning: No results generated")
        return pl.DataFrame()
    
    # Collect all unique keys to ensure all columns are included
    all_keys = set()
    for row in all_results:
        all_keys.update(row.keys())
    
    # Ensure all rows have all keys (with None for missing)
    for row in all_results:
        for key in all_keys:
            if key not in row:
                row[key] = None
    
    # Use infer_schema_length=None to scan all rows
    result_df = pl.DataFrame(all_results, infer_schema_length=None)
    
    print(f"\nTotal: {len(result_df)} rows, {len(result_df.columns)} columns")
    
    # Write output
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(out_path)
        print(f"Written to {out_path}")
    
    return result_df


# CLI entry point
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
