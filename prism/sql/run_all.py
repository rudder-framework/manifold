"""
prism/sql/run_all.py

Executes all SQL scripts in order, validates outputs, fails fast on any error.

Usage:
    python run_all.py /path/to/observations.parquet
    python run_all.py /path/to/observations.parquet ./custom_outputs/
"""

import duckdb
from pathlib import Path
import json
from datetime import datetime
import sys

SQL_DIR = Path(__file__).parent
OUTPUT_DIR = SQL_DIR / 'outputs'

# Execution order - each tuple is (stage_dir, [scripts])
STAGES = [
    ('00_load', ['001_load_raw.sql']),
    ('01_calculus', [
        '001_first_derivative.sql',
        '002_second_derivative.sql',
        '003_curvature.sql',
        '004_laplacian.sql',
        '005_arc_length.sql',
        '_write_calculus.sql'
    ]),
    ('02_signal_class', [
        '001_from_units.sql',
        '002_from_curvature.sql',
        '003_from_sparsity.sql',
        '004_classify.sql',
        '_write_signal_class.sql'
    ]),
    ('03_signal_typology', [
        '001_persistence.sql',
        '002_periodicity.sql',
        '003_stationarity.sql',
        '004_entropy.sql',
        '005_classify.sql',
        '_write_typology.sql'
    ]),
    ('04_behavioral_geometry', [
        '001_correlation.sql',
        '002_covariance.sql',
        '003_distance.sql',
        '004_coupling.sql',
        '005_pca.sql',
        '006_clustering.sql',
        '_write_geometry.sql'
    ]),
    ('05_dynamical_systems', [
        '001_regime_detection.sql',
        '002_stability.sql',
        '003_basin.sql',
        '004_attractor.sql',
        '_write_dynamics.sql'
    ]),
    ('06_causal_mechanics', [
        '001_granger.sql',
        '002_transfer_entropy.sql',
        '003_role_assignment.sql',
        '_write_causal.sql'
    ]),
]

EXPECTED_OUTPUTS = [
    'calculus.parquet',
    'signal_class.parquet',
    'signal_typology.parquet',
    'behavioral_geometry.parquet',
    'dynamical_systems.parquet',
    'causal_mechanics.parquet',
]


def register_prism_udfs(conn: duckdb.DuckDBPyConnection):
    """Register PRISM engine functions as DuckDB UDFs."""
    import numpy as np

    # Import PRISM engines
    try:
        from prism.engines.core import hurst, fft, lyapunov, garch
        from prism.engines.core import sample_entropy, permutation_entropy
        from prism.engines.core import granger, transfer_entropy, dtw
        from prism.engines.core import pca, clustering
    except ImportError as e:
        print(f"Warning: Could not import PRISM engines: {e}")
        print("Some UDFs will not be available.")
        return

    # Helper to convert array to numpy
    def to_numpy(arr):
        if arr is None:
            return np.array([])
        return np.array(arr, dtype=float)

    # Register UDFs
    def prism_hurst(arr):
        x = to_numpy(arr)
        if len(x) < 20:
            return {'hurst': None, 'hurst_r2': None}
        result = hurst.compute(x)
        return {'hurst': result.get('hurst'), 'hurst_r2': result.get('r2')}

    def prism_fft(arr):
        x = to_numpy(arr)
        if len(x) < 16:
            return {'centroid': None, 'bandwidth': None, 'dominant_freq': None,
                    'rolloff': None, 'low_high_ratio': None, 'total_power': None}
        result = fft.compute(x)
        return result

    def prism_lyapunov(arr):
        x = to_numpy(arr)
        if len(x) < 100:
            return {'lyapunov_exponent': None, 'is_chaotic': None,
                    'is_stable': None, 'is_critical': None, 'method': 'insufficient_data'}
        result = lyapunov.compute(x)
        return result

    def prism_sample_entropy(arr):
        x = to_numpy(arr)
        if len(x) < 50:
            return {'sample_entropy': None}
        result = sample_entropy.compute(x)
        return {'sample_entropy': result.get('sample_entropy')}

    def prism_permutation_entropy(arr):
        x = to_numpy(arr)
        if len(x) < 20:
            return {'permutation_entropy': None, 'normalized_entropy': None}
        result = permutation_entropy.compute(x)
        return {
            'permutation_entropy': result.get('permutation_entropy'),
            'normalized_entropy': result.get('normalized_entropy')
        }

    def prism_granger(arr1, arr2):
        x = to_numpy(arr1)
        y = to_numpy(arr2)
        if len(x) < 30 or len(y) < 30:
            return {'f_statistic': None, 'p_value': None, 'is_significant': False}
        result = granger.compute(x, y)
        return {
            'f_statistic': result.get('f_statistic'),
            'p_value': result.get('p_value'),
            'is_significant': result.get('p_value', 1.0) < 0.05
        }

    def prism_transfer_entropy(arr1, arr2):
        x = to_numpy(arr1)
        y = to_numpy(arr2)
        if len(x) < 50 or len(y) < 50:
            return {'transfer_entropy': None, 'normalized_te': None}
        result = transfer_entropy.compute(x, y)
        return {
            'transfer_entropy': result.get('transfer_entropy'),
            'normalized_te': result.get('normalized_te')
        }

    def prism_dtw(arr1, arr2):
        x = to_numpy(arr1)
        y = to_numpy(arr2)
        if len(x) < 5 or len(y) < 5:
            return {'distance': None}
        result = dtw.compute(x, y)
        return {'distance': result.get('distance')}

    def prism_pca(matrix):
        if matrix is None or len(matrix) < 3:
            return {'explained_variance_ratio': [], 'components': [], 'n_components': 0}
        arr = np.array(matrix, dtype=float)
        result = pca.compute(arr)
        return {
            'explained_variance_ratio': result.get('explained_variance_ratio', []),
            'components': result.get('components', []),
            'n_components': result.get('n_components', 0)
        }

    def prism_clustering(matrix, k=3):
        if matrix is None or len(matrix) < k:
            return {'labels': [], 'inertia': None, 'n_clusters': 0}
        arr = np.array(matrix, dtype=float)
        result = clustering.compute(arr, n_clusters=k)
        return {
            'labels': result.get('labels', []),
            'inertia': result.get('inertia'),
            'n_clusters': k
        }

    # Register all UDFs
    try:
        conn.create_function('prism_hurst', prism_hurst)
        conn.create_function('prism_fft', prism_fft)
        conn.create_function('prism_lyapunov', prism_lyapunov)
        conn.create_function('prism_sample_entropy', prism_sample_entropy)
        conn.create_function('prism_permutation_entropy', prism_permutation_entropy)
        conn.create_function('prism_granger', prism_granger)
        conn.create_function('prism_transfer_entropy', prism_transfer_entropy)
        conn.create_function('prism_dtw', prism_dtw)
        conn.create_function('prism_pca', prism_pca)
        conn.create_function('prism_clustering', prism_clustering)
        print("PRISM UDFs registered successfully")
    except Exception as e:
        print(f"Warning: Could not register some UDFs: {e}")


def run_pipeline(input_path: str, output_dir: str = None, stages: list = None):
    """
    Execute full SQL pipeline.

    Args:
        input_path: Path to input parquet file
        output_dir: Output directory (default: sql/outputs/)
        stages: List of stage names to run (default: all)

    Returns:
        manifest dict with file info
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(':memory:')

    # Set output directory for COPY commands
    conn.execute(f"SET file_search_path='{output_dir}'")

    # Create write log table
    conn.execute("""
        CREATE TABLE _write_log (
            file VARCHAR,
            rows INTEGER,
            written_at TIMESTAMP
        )
    """)

    # Register PRISM UDFs
    register_prism_udfs(conn)

    # Load input data
    print(f"\nLoading input data from {input_path}...")
    conn.execute(f"CREATE TABLE input_data AS SELECT * FROM '{input_path}'")
    input_rows = conn.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    print(f"Loaded {input_rows:,} rows")

    # Filter stages if specified
    stages_to_run = STAGES
    if stages:
        stages_to_run = [(name, scripts) for name, scripts in STAGES if name in stages]

    # Execute each stage
    for stage_name, scripts in stages_to_run:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name}")
        print('='*60)

        stage_dir = SQL_DIR / stage_name

        for script in scripts:
            script_path = stage_dir / script
            if not script_path.exists():
                raise FileNotFoundError(f"Missing script: {script_path}")

            print(f"  Running {script}...", end=' ', flush=True)
            sql = script_path.read_text()

            # Replace relative output paths with absolute
            sql = sql.replace("'outputs/", f"'{output_dir}/")

            try:
                conn.execute(sql)
                print("OK")
            except Exception as e:
                print(f"FAILED")
                print(f"\n  Error: {e}")
                raise RuntimeError(f"Script {script} failed: {e}")

    # Validate all outputs exist
    print(f"\n{'='*60}")
    print("VALIDATION")
    print('='*60)

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'input_file': str(input_path),
        'input_rows': input_rows,
        'files': {}
    }

    missing = []
    for expected in EXPECTED_OUTPUTS:
        path = output_dir / expected
        if not path.exists():
            missing.append(expected)
            print(f"  MISSING: {expected}")
        else:
            rows = conn.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
            manifest['files'][expected] = {'rows': rows, 'path': str(path)}
            print(f"  OK: {expected}: {rows:,} rows")

    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  manifest.json written")

    # Final summary from write log
    print(f"\n{'='*60}")
    print("WRITE LOG")
    print('='*60)
    log = conn.execute("SELECT * FROM _write_log ORDER BY written_at").fetchall()
    for file, rows, written_at in log:
        print(f"  {written_at}: {file} ({rows:,} rows)")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Input:  {input_rows:,} rows")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(manifest['files'])}")

    return manifest


def run_stage(input_path: str, stage_name: str, output_dir: str = None):
    """Run a single stage (for testing)."""
    return run_pipeline(input_path, output_dir, stages=[stage_name])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <input.parquet> [output_dir]")
        print("\nStages:")
        for name, scripts in STAGES:
            print(f"  {name}: {len(scripts)} scripts")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        run_pipeline(input_path, output_dir)
    except Exception as e:
        print(f"\nFATAL: {e}")
        sys.exit(1)
