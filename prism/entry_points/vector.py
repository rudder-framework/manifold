#!/usr/bin/env python3
"""
PRISM Vector Entry Point
========================

Computes signal-level metrics using ALL vector engines with proper windowing.

Engines (28 total):
    Memory (4): hurst_dfa, hurst_rs, acf_decay, spectral_slope
    Information (3): permutation_entropy, sample_entropy, entropy_rate
    Frequency (2): spectral, wavelet
    Volatility (4): garch, realized_vol, bipower_variation, hilbert_amplitude
    Recurrence (1): rqa
    Typology (8): cusum, derivative_stats, distribution, rolling_volatility,
                  seasonality, stationarity, takens, trend
    Pointwise (3): derivatives, hilbert, statistical
    Momentum (1): runs_test
    Discontinuity (3): dirac, heaviside, structural

Usage:
    python -m prism.entry_points.vector
    python -m prism.entry_points.vector --force
    python -m prism.entry_points.vector --window 100 --stride 20

Output:
    data/vector.parquet - One row per (entity, signal, window)
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS
# =============================================================================

def import_engines():
    """Import all vector engines with graceful fallbacks."""
    engines = {}

    # Memory engines
    try:
        from prism.engines.memory import (
            compute_hurst_dfa, compute_hurst_rs,
            compute_acf_decay, compute_spectral_slope
        )
        engines['hurst_dfa'] = compute_hurst_dfa
        engines['hurst_rs'] = compute_hurst_rs
        engines['acf_decay'] = compute_acf_decay
        engines['spectral_slope'] = compute_spectral_slope
        logger.info("  Memory engines: 4 loaded")
    except ImportError as e:
        logger.warning(f"  Memory engines failed: {e}")

    # Information engines
    try:
        from prism.engines.information import (
            compute_permutation_entropy, compute_sample_entropy,
            compute_entropy_rate
        )
        engines['permutation_entropy'] = compute_permutation_entropy
        engines['sample_entropy'] = compute_sample_entropy
        engines['entropy_rate'] = compute_entropy_rate
        logger.info("  Information engines: 3 loaded")
    except ImportError as e:
        logger.warning(f"  Information engines failed: {e}")

    # Frequency engines
    try:
        from prism.engines.frequency import compute_spectral, compute_wavelet
        engines['spectral'] = compute_spectral
        engines['wavelet'] = compute_wavelet
        logger.info("  Frequency engines: 2 loaded")
    except ImportError as e:
        logger.warning(f"  Frequency engines failed: {e}")

    # Volatility engines
    try:
        from prism.engines.volatility import (
            compute_garch, compute_realized_vol,
            compute_bipower_variation, compute_hilbert_amplitude
        )
        engines['garch'] = compute_garch
        engines['realized_vol'] = compute_realized_vol
        engines['bipower_variation'] = compute_bipower_variation
        engines['hilbert_amplitude'] = compute_hilbert_amplitude
        logger.info("  Volatility engines: 4 loaded")
    except ImportError as e:
        logger.warning(f"  Volatility engines failed: {e}")

    # Recurrence engines
    try:
        from prism.engines.recurrence import compute_rqa
        engines['rqa'] = compute_rqa
        logger.info("  Recurrence engines: 1 loaded")
    except ImportError as e:
        logger.warning(f"  Recurrence engines failed: {e}")

    # Typology engines
    try:
        from prism.engines.typology import (
            compute_cusum, compute_derivative_stats, compute_distribution,
            compute_rolling_volatility, compute_seasonality,
            compute_stationarity, compute_takens, compute_trend
        )
        engines['cusum'] = compute_cusum
        engines['derivative_stats'] = compute_derivative_stats
        engines['distribution'] = compute_distribution
        engines['rolling_volatility'] = compute_rolling_volatility
        engines['seasonality'] = compute_seasonality
        engines['stationarity'] = compute_stationarity
        engines['takens'] = compute_takens
        engines['trend'] = compute_trend
        logger.info("  Typology engines: 8 loaded")
    except ImportError as e:
        logger.warning(f"  Typology engines failed: {e}")

    # Pointwise engines
    try:
        from prism.engines.pointwise import (
            compute_derivatives, compute_hilbert, compute_statistical
        )
        engines['derivatives'] = compute_derivatives
        engines['hilbert'] = compute_hilbert
        engines['statistical'] = compute_statistical
        logger.info("  Pointwise engines: 3 loaded")
    except ImportError as e:
        logger.warning(f"  Pointwise engines failed: {e}")

    # Momentum engines
    try:
        from prism.engines.momentum import compute_runs_test
        engines['runs_test'] = compute_runs_test
        logger.info("  Momentum engines: 1 loaded")
    except ImportError as e:
        logger.warning(f"  Momentum engines failed: {e}")

    # Discontinuity engines
    try:
        from prism.engines.discontinuity import (
            compute_dirac, compute_heaviside, compute_structural
        )
        engines['dirac'] = compute_dirac
        engines['heaviside'] = compute_heaviside
        engines['structural'] = compute_structural
        logger.info("  Discontinuity engines: 3 loaded")
    except ImportError as e:
        logger.warning(f"  Discontinuity engines failed: {e}")

    return engines


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples': 50,
    'window_size': None,  # None = use full signal
    'stride': None,       # None = non-overlapping
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
        logger.info(f"Loaded config: min_samples={config['min_samples']}")

    return config


# =============================================================================
# WINDOWING
# =============================================================================

def generate_windows(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    min_samples: int = 50,
) -> List[Dict]:
    """
    Generate overlapping windows from a signal.

    Args:
        values: Signal values
        timestamps: Corresponding timestamps
        window_size: Window size (None = full signal)
        stride: Stride between windows (None = window_size)
        min_samples: Minimum samples per window

    Yields:
        Dict with window_idx, window_start, window_end, values, timestamps
    """
    n = len(values)

    if window_size is None or window_size >= n:
        # Single window (full signal)
        if n >= min_samples:
            yield {
                'window_idx': 0,
                'window_start': float(timestamps[0]),
                'window_end': float(timestamps[-1]),
                'values': values,
                'timestamps': timestamps,
            }
        return

    if stride is None:
        stride = window_size

    window_idx = 0
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_values = values[start:end]
        window_timestamps = timestamps[start:end]

        if len(window_values) >= min_samples:
            yield {
                'window_idx': window_idx,
                'window_start': float(window_timestamps[0]),
                'window_end': float(window_timestamps[-1]),
                'values': window_values,
                'timestamps': window_timestamps,
            }
            window_idx += 1


# =============================================================================
# RESULT FLATTENING
# =============================================================================

def flatten_result(result: Any, prefix: str) -> Dict[str, float]:
    """Flatten engine result to dict of floats."""
    flat = {}

    if result is None:
        return flat

    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif isinstance(v, np.ndarray) and v.size == 1:
                val = float(v.item())
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif hasattr(result, '__dataclass_fields__'):
        # Dataclass
        for k in result.__dataclass_fields__:
            v = getattr(result, k)
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
    elif isinstance(result, (int, float, np.integer, np.floating)):
        val = float(result)
        if np.isfinite(val):
            flat[prefix] = val

    return flat


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_vector(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute vector metrics for all signals.

    Args:
        observations: Raw observations DataFrame
        config: Domain configuration
        engines: Dict of engine_name -> compute function
        force: Recompute all

    Returns:
        DataFrame with one row per (entity, signal, window)
    """
    min_samples = config.get('min_samples', 50)
    window_size = config.get('window_size')
    stride = config.get('stride')

    # Group observations by entity+signal
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    n_signals = len(signals)
    n_engines = len(engines)
    logger.info(f"Processing {n_signals} signals with {n_engines} engines")

    results = []
    total_windows = 0

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        timestamps = np.array(row['timestamps'], dtype=float)

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        values = values[sort_idx]
        timestamps = timestamps[sort_idx]

        # Remove NaN
        valid = ~np.isnan(values)
        values = values[valid]
        timestamps = timestamps[valid]

        if len(values) < min_samples:
            continue

        # Generate windows
        for window in generate_windows(values, timestamps, window_size, stride, min_samples):
            window_values = window['values']

            row_data = {
                'entity_id': entity_id,
                'signal_id': signal_id,
                'window_idx': window['window_idx'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'n_samples': len(window_values),
            }

            # Run all engines
            for engine_name, compute_fn in engines.items():
                try:
                    result = compute_fn(window_values)
                    flat = flatten_result(result, engine_name)
                    row_data.update(flat)
                except Exception as e:
                    # Engine failed for this window, skip silently
                    pass

            results.append(row_data)
            total_windows += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals ({total_windows} windows)")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Vector: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Vector - Signal-level metrics computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')
    parser.add_argument('--window', type=int, default=None,
                        help='Window size (default: full signal)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride between windows (default: window size)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Vector Engine")
    logger.info("=" * 60)

    ensure_directory()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found at {obs_path}")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'vector.parquet'

    if output_path.exists() and not args.force:
        logger.info(f"vector.parquet exists, use --force to recompute")
        return 0

    # Load config
    config = load_config(data_path)
    if args.window:
        config['window_size'] = args.window
    if args.stride:
        config['stride'] = args.stride

    # Load engines
    logger.info("Loading engines...")
    engines = import_engines()
    logger.info(f"Total engines: {len(engines)}")

    # Load data
    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    # Compute
    start = time.time()
    df = compute_vector(observations, config, engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
