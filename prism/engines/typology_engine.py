"""
PRISM Typology Engine

First pass signal characterization. Runs BEFORE any other computation.
Determines: which vector engines to use, what window size, which signals to skip.

Output: typology.parquet

Core Metrics (all SQL-computable, all scale-invariant where possible):
1. SMOOTHNESS     - autocorr lag-1 (0-1, how continuous)
2. PERIODICITY    - autocorr secondary peaks (does it repeat)
3. TAIL BEHAVIOR  - kurtosis (>3 = impulsive, heavy tails)
4. MEMORY         - Hurst proxy (trending vs reverting)
5. STATIONARITY   - variance ratio (stable vs evolving)

Plus diagnostic metrics:
- signal_std (for constant detection)
- n_samples
- recommended_engines
- recommended_window
"""

import duckdb
import polars as pl
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Any


# Load engine manifest
MANIFEST_PATH = Path(__file__).parent / "engine_manifest.yaml"


def load_manifest() -> Dict[str, Any]:
    """Load the engine manifest from YAML."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return yaml.safe_load(f)
    else:
        # Fallback defaults
        return {
            'core': ['kurtosis', 'skewness', 'crest_factor'],
            'signal_type': {
                'SMOOTH': ['rolling_kurtosis', 'rolling_entropy', 'rolling_crest_factor'],
                'NOISY': ['kurtosis', 'entropy'],
                'IMPULSIVE': ['kurtosis', 'crest_factor', 'peak_ratio'],
                'MIXED': ['kurtosis', 'entropy', 'crest_factor'],
            },
            'periodicity': {
                'PERIODIC': ['harmonics_ratio', 'band_ratios', 'spectral_centroid'],
                'APERIODIC': ['entropy', 'hurst'],
            },
            'deprecated': ['rms', 'peak', 'total_power', 'rolling_rms', 'rolling_mean'],
        }


ENGINE_MANIFEST = load_manifest()


# Simplified version that's faster (single pass, no subqueries)
# Uses signal_id to match PRISM schema
TYPOLOGY_SQL_FAST = """
-- FAST TYPOLOGY: Single-pass core metrics
-- Approximates some metrics for speed

WITH base AS (
    SELECT
        unit_id,
        signal_id,
        I,
        value,
        LAG(value, 1) OVER w AS lag1,
        LAG(value, 10) OVER w AS lag10,
        LAG(value, 20) OVER w AS lag20,
        LAG(value, 50) OVER w AS lag50
    FROM observations
    WINDOW w AS (PARTITION BY unit_id, signal_id ORDER BY I)
)

SELECT
    unit_id,
    signal_id,

    -- Sample size
    COUNT(*) AS n_samples,

    -- Basic stats
    STDDEV(value) AS signal_std,
    AVG(value) AS signal_mean,
    MIN(value) AS signal_min,
    MAX(value) AS signal_max,

    -- 1. SMOOTHNESS (autocorr lag-1)
    CORR(value, lag1) AS smoothness,

    -- 2. PERIODICITY (autocorr lag-10 / lag-1)
    CASE
        WHEN ABS(CORR(value, lag1)) > 0.1
        THEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0)
        ELSE 0
    END AS periodicity_ratio,

    -- Raw autocorrelations
    CORR(value, lag1) AS autocorr_1,
    CORR(value, lag10) AS autocorr_10,
    CORR(value, lag20) AS autocorr_20,
    CORR(value, lag50) AS autocorr_50,

    -- 3. TAIL BEHAVIOR
    KURTOSIS(value) AS kurtosis,
    SKEWNESS(value) AS skewness,

    -- 4. MEMORY PROXY (diff std / value std)
    -- Low ratio = high memory (smooth, trending)
    -- High ratio = low memory (noisy, independent)
    STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) AS memory_proxy,

    -- Classifications
    CASE WHEN STDDEV(value) < 0.001 THEN TRUE ELSE FALSE END AS is_constant,

    -- Signal type classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN ABS(CORR(value, lag1)) > 0.95 THEN 'SMOOTH'
        WHEN ABS(CORR(value, lag1)) < 0.3 THEN 'NOISY'
        WHEN KURTOSIS(value) > 5 THEN 'IMPULSIVE'
        ELSE 'MIXED'
    END AS signal_type,

    -- Periodicity classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0) > 0.7 THEN 'PERIODIC'
        WHEN CORR(value, lag10) / NULLIF(CORR(value, lag1), 0) > 0.3 THEN 'QUASI_PERIODIC'
        ELSE 'APERIODIC'
    END AS periodicity_type,

    -- Tail classification
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN KURTOSIS(value) > 6 THEN 'HEAVY_TAILS'
        WHEN KURTOSIS(value) > 4 THEN 'MODERATE_TAILS'
        WHEN KURTOSIS(value) < 2 THEN 'LIGHT_TAILS'
        ELSE 'NORMAL_TAILS'
    END AS tail_type,

    -- Stationarity classification (variance ratio approximation)
    -- Compare variance of first vs second half using diff variance as proxy
    CASE
        WHEN STDDEV(value) < 0.001 THEN 'CONSTANT'
        WHEN STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) > 1.5 THEN 'NON_STATIONARY'
        WHEN STDDEV(value - lag1) / NULLIF(STDDEV(value), 0) < 0.3 THEN 'HIGHLY_STATIONARY'
        ELSE 'STATIONARY'
    END AS stationarity_type,

    -- Recommended window size based on autocorrelation decay
    CASE
        WHEN STDDEV(value) < 0.001 THEN NULL  -- constant, no window needed
        WHEN ABS(CORR(value, lag1)) > 0.95 THEN 10   -- very smooth, small window OK
        WHEN ABS(CORR(value, lag1)) > 0.9 THEN 15
        WHEN ABS(CORR(value, lag1)) > 0.8 THEN 20
        WHEN ABS(CORR(value, lag1)) > 0.6 THEN 30
        WHEN ABS(CORR(value, lag1)) > 0.4 THEN 50
        ELSE 100  -- noisy, need larger window
    END AS recommended_window

FROM base
WHERE lag1 IS NOT NULL
GROUP BY unit_id, signal_id
ORDER BY unit_id, signal_id
"""


def run_typology(
    observations_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Run typology engine on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path for typology.parquet output (default: same dir as input)
        verbose: Print progress and summary

    Returns:
        Polars DataFrame with typology results
    """
    observations_path = Path(observations_path)

    if output_path is None:
        output_path = observations_path.parent / "typology.parquet"
    else:
        output_path = Path(output_path)

    if verbose:
        print("=" * 70)
        print("TYPOLOGY ENGINE")
        print("=" * 70)
        print(f"Input: {observations_path}")
        print(f"Output: {output_path}")
        print()

    # Connect to DuckDB
    con = duckdb.connect()

    # Load observations
    if verbose:
        print("Loading observations...")

    con.execute(f"""
        CREATE TABLE observations AS
        SELECT * FROM read_parquet('{observations_path}')
    """)

    # Get row count
    n_rows = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    n_signals = con.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]
    n_units = con.execute("SELECT COUNT(DISTINCT unit_id) FROM observations").fetchone()[0]

    if verbose:
        print(f"  Rows: {n_rows:,}")
        print(f"  Signals: {n_signals}")
        print(f"  Units: {n_units}")
        print()

    # Run typology
    if verbose:
        print("Computing typology...")

    import time
    start = time.time()

    result = con.execute(TYPOLOGY_SQL_FAST).pl()

    elapsed = time.time() - start

    if verbose:
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Throughput: {n_rows/elapsed:,.0f} rows/sec")
        print()

    # Save to parquet
    result.write_parquet(output_path)

    if verbose:
        print(f"Saved: {output_path}")
        print()

        # Summary
        print("=" * 70)
        print("TYPOLOGY SUMMARY")
        print("=" * 70)

        # Constant signals
        constant = result.filter(pl.col('is_constant'))
        print(f"\nCONSTANT SIGNALS (skip): {len(constant)}")
        if len(constant) > 0:
            const_signals = constant['signal_id'].unique().to_list()
            print(f"  {const_signals[:10]}{'...' if len(const_signals) > 10 else ''}")

        # Signal types
        print(f"\nSIGNAL TYPES:")
        type_counts = result.group_by('signal_type').agg(pl.len().alias('count')).sort('count', descending=True)
        for row in type_counts.iter_rows():
            print(f"  {row[0]}: {row[1]}")

        # Periodicity types
        print(f"\nPERIODICITY TYPES:")
        period_counts = result.group_by('periodicity_type').agg(pl.len().alias('count')).sort('count', descending=True)
        for row in period_counts.iter_rows():
            print(f"  {row[0]}: {row[1]}")

        # Tail types
        print(f"\nTAIL TYPES:")
        tail_counts = result.group_by('tail_type').agg(pl.len().alias('count')).sort('count', descending=True)
        for row in tail_counts.iter_rows():
            print(f"  {row[0]}: {row[1]}")

        # Stationarity types
        print(f"\nSTATIONARITY TYPES:")
        stat_counts = result.group_by('stationarity_type').agg(pl.len().alias('count')).sort('count', descending=True)
        for row in stat_counts.iter_rows():
            print(f"  {row[0]}: {row[1]}")

        # Window recommendations
        active = result.filter(~pl.col('is_constant'))
        if len(active) > 0:
            print(f"\nWINDOW RECOMMENDATIONS:")
            print(f"  Min: {active['recommended_window'].min()}")
            print(f"  Max: {active['recommended_window'].max()}")
            print(f"  Median: {active['recommended_window'].median()}")

            # Suggested global window (conservative = min)
            suggested_window = int(active['recommended_window'].min())
            print(f"\n  SUGGESTED GLOBAL WINDOW: {suggested_window}")

    con.close()

    return result


def select_engines(typology_row: dict, manifest: Dict[str, Any] = None) -> List[str]:
    """
    Select engines based on typology classification for a single signal.
    Uses engine_manifest.yaml for configuration.

    Args:
        typology_row: Dict with typology metrics for one signal
        manifest: Engine manifest (uses global ENGINE_MANIFEST if not provided)

    Returns:
        List of engine names to run for this signal
    """
    if manifest is None:
        manifest = ENGINE_MANIFEST

    engines = set()

    # Skip constants
    if typology_row.get('is_constant'):
        return []

    # CORE - always include
    core = manifest.get('core', ['kurtosis', 'skewness', 'crest_factor'])
    engines.update(core)

    # By signal type
    sig_type = typology_row.get('signal_type', 'MIXED')
    signal_type_engines = manifest.get('signal_type', {})
    if sig_type in signal_type_engines:
        engines.update(signal_type_engines[sig_type])

    # By periodicity
    period_type = typology_row.get('periodicity_type', 'APERIODIC')
    periodicity_engines = manifest.get('periodicity', {})
    if period_type in periodicity_engines:
        engines.update(periodicity_engines[period_type])

    # By tail behavior
    tail_type = typology_row.get('tail_type', 'NORMAL_TAILS')
    tail_engines = manifest.get('tail_type', {})
    if tail_type in tail_engines:
        engines.update(tail_engines[tail_type])

    # By stationarity
    stationarity = typology_row.get('stationarity_type', 'STATIONARY')
    stationarity_engines = manifest.get('stationarity', {})
    if stationarity in stationarity_engines:
        stat_engs = stationarity_engines[stationarity]
        engines.update(stat_engs)

        # For non-stationary, remove global versions
        if stationarity in ['NON_STATIONARY', 'VARIANCE_INCREASING', 'VARIANCE_DECREASING']:
            engines.discard('kurtosis')
            engines.discard('entropy')

    # By memory (trending signals)
    memory_proxy = typology_row.get('memory_proxy')
    memory_engines = manifest.get('memory', {})
    if memory_proxy and memory_proxy < 0.5:
        # High memory = trending
        if 'TRENDING' in memory_engines:
            engines.update(memory_engines['TRENDING'])
    elif memory_proxy and memory_proxy > 1.5:
        # Low memory = reverting
        if 'REVERTING' in memory_engines:
            engines.update(memory_engines['REVERTING'])

    # Remove deprecated engines
    deprecated = set(manifest.get('deprecated', []))
    engines = engines - deprecated

    return list(engines)


def get_engine_recommendations(typology: pl.DataFrame) -> Dict[str, Any]:
    """
    From typology results, determine which vector engines to use.

    Returns dict with:
        - skip_signals: List of constant signals to exclude
        - active_signals: List of signals to process
        - window_size: Recommended global window
        - signal_engines: Dict mapping each signal to its recommended engines
        - all_engines: Union of all engines needed
        - engines_not_needed: Engines that no signal requires
    """

    # Signals to skip (constant)
    skip_signals = typology.filter(
        pl.col('is_constant')
    )['signal_id'].unique().to_list()

    # Active signals
    active = typology.filter(~pl.col('is_constant'))

    # Global window (conservative = minimum)
    if len(active) > 0:
        window_size = int(active['recommended_window'].min())
    else:
        window_size = 50  # default

    # Map each signal to recommended engines using select_engines
    signal_engines = {}
    all_engines = set()

    for row in active.iter_rows(named=True):
        signal = row['signal_id']
        engines = select_engines(row)
        signal_engines[signal] = engines
        all_engines.update(engines)

    # Get all valid engines from manifest
    valid_engines = set(ENGINE_MANIFEST.get('engines', {}).keys())

    # Get deprecated engines
    deprecated = set(ENGINE_MANIFEST.get('deprecated', []))

    # Engines not needed = valid engines that aren't selected
    engines_not_needed = sorted([e for e in valid_engines if e not in all_engines])

    # Also track deprecated engines that were avoided
    deprecated_avoided = sorted(list(deprecated))

    return {
        'skip_signals': skip_signals,
        'active_signals': active['signal_id'].unique().to_list(),
        'window_size': window_size,
        'signal_engines': signal_engines,
        'all_engines': sorted(list(all_engines)),
        'engines_not_needed': engines_not_needed,
        'deprecated_avoided': deprecated_avoided,
    }


def print_recommendations(typology: pl.DataFrame):
    """Print human-readable recommendations from typology."""

    recs = get_engine_recommendations(typology)

    print("=" * 70)
    print("VECTOR ENGINE RECOMMENDATIONS")
    print("=" * 70)

    print(f"\n1. SIGNALS TO SKIP ({len(recs['skip_signals'])}):")
    for sig in recs['skip_signals']:
        print(f"   - {sig}")

    print(f"\n2. ACTIVE SIGNALS ({len(recs['active_signals'])}):")
    for sig in recs['active_signals'][:10]:
        print(f"   + {sig}")
    if len(recs['active_signals']) > 10:
        print(f"   ... and {len(recs['active_signals']) - 10} more")

    print(f"\n3. GLOBAL WINDOW SIZE: {recs['window_size']}")

    print(f"\n4. ENGINES NEEDED ({len(recs['all_engines'])}):")
    for eng in sorted(recs['all_engines']):
        print(f"   * {eng}")

    print(f"\n5. ENGINES NOT NEEDED ({len(recs['engines_not_needed'])}):")
    for eng in recs['engines_not_needed'][:10]:
        print(f"   - {eng}")
    if len(recs['engines_not_needed']) > 10:
        print(f"   ... and {len(recs['engines_not_needed']) - 10} more")

    print(f"\n   DEPRECATED (scale-dependent, never used):")
    for eng in recs['deprecated_avoided'][:5]:
        print(f"   x {eng}")
    if len(recs['deprecated_avoided']) > 5:
        print(f"   ... and {len(recs['deprecated_avoided']) - 5} more")

    # Show per-signal breakdown for first few
    print(f"\n6. PER-SIGNAL ENGINES (sample):")
    for i, (sig, engines) in enumerate(recs['signal_engines'].items()):
        if i >= 5:
            print(f"   ... and {len(recs['signal_engines']) - 5} more signals")
            break
        print(f"   {sig}: {', '.join(sorted(engines)[:5])}{'...' if len(engines) > 5 else ''}")


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python typology_engine.py <observations.parquet> [output.parquet]")
        print()
        print("Runs typology analysis on observations and outputs recommendations")
        print("for vector engine selection and window sizing.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Run typology
    result = run_typology(input_path, output_path)

    # Print recommendations
    print()
    print_recommendations(result)
