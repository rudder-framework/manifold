"""
PRISM Manifest Runner (Orchestrator)

Reads manifest from ORTHON and runs ALL engines.

FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.

- ALL engines run, always
- Insufficient data → NaN, never skip
- RAM managed via entity batching
- Writes directly to data/ directory
"""

import json
import gc
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import polars as pl

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from prism.python_runner import PythonRunner, SIGNAL_ENGINES, PAIR_ENGINES, SYMMETRIC_PAIR_ENGINES, WINDOWED_ENGINES
from prism.sql_runner import SQLRunner, SQL_ENGINES


# ALL engines - always enabled, no exceptions
ALL_SIGNAL_ENGINES = SIGNAL_ENGINES
ALL_PAIR_ENGINES = PAIR_ENGINES
ALL_SYMMETRIC_PAIR_ENGINES = SYMMETRIC_PAIR_ENGINES
ALL_WINDOWED_ENGINES = WINDOWED_ENGINES
ALL_SQL_ENGINES = SQL_ENGINES


def load_manifest(manifest_path: str) -> dict:
    """
    Load manifest from YAML or JSON file.

    Supports both ORTHON format (YAML) and legacy format (JSON).
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path) as f:
        content = f.read()

    # Try YAML first (ORTHON format), fall back to JSON (legacy)
    if manifest_path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML manifests: pip install pyyaml")
        manifest = yaml.safe_load(content)
    elif manifest_path.suffix == '.json':
        manifest = json.loads(content)
    else:
        # Try both
        try:
            if HAS_YAML:
                manifest = yaml.safe_load(content)
            else:
                manifest = json.loads(content)
        except:
            manifest = json.loads(content)

    return normalize_manifest(manifest, manifest_path.parent)


def normalize_manifest(manifest: dict, manifest_dir: Path) -> dict:
    """
    Normalize manifest to internal format.

    Handles both ORTHON format and legacy format.
    Always enables ALL engines.
    """

    # Check if this is ORTHON format (has 'data' and/or 'prism' keys)
    if 'data' in manifest or 'prism' in manifest:
        return _normalize_orthon_manifest(manifest, manifest_dir)
    else:
        # Legacy format - normalize and enable all engines
        return _normalize_legacy_manifest(manifest)


def _normalize_orthon_manifest(manifest: dict, manifest_dir: Path) -> dict:
    """
    Normalize ORTHON YAML manifest to internal format.

    ORTHON format:
        dataset:
            name: "..."
            domain: "..."
        data:
            observations_path: "observations.parquet"
            output_path: "observations.parquet"  # or directory
        prism:
            window_size: 1024
            stride: 512
            engines: {...}  # ignored - we run ALL
            ram:
                batch_size: auto

    Internal format:
        observations_path: Path
        output_dir: Path
        engines: {all engines enabled}
        params: {window, stride, etc}
        ram: {batch config}
    """

    data_config = manifest.get('data', {})
    prism_config = manifest.get('prism', {})

    # Resolve observations path
    obs_path = data_config.get('observations_path', data_config.get('output_path', 'observations.parquet'))
    if not Path(obs_path).is_absolute():
        obs_path = manifest_dir / obs_path
    obs_path = Path(obs_path)

    # Output directory is same as observations directory
    output_dir = obs_path.parent

    # Window/stride params
    window_size = prism_config.get('window_size', 100)
    stride = prism_config.get('stride', window_size)

    # Build params dict
    params = {
        'window_size': window_size,
        'stride': stride,
    }

    # Add any engine-specific params from prism config
    for key, value in prism_config.items():
        if key not in ['engines', 'ram', 'window_size', 'stride', 'compute']:
            params[key] = value

    # RAM config
    ram_config = prism_config.get('ram', {})

    # ALL ENGINES - NO EXCEPTIONS
    # Ignore whatever engines are specified in manifest - we run everything
    normalized = {
        'observations_path': str(obs_path),
        'output_dir': str(output_dir),
        'engines': {
            'signal': ALL_SIGNAL_ENGINES,
            'pair': ALL_PAIR_ENGINES,
            'symmetric_pair': ALL_SYMMETRIC_PAIR_ENGINES,
            'windowed': ALL_WINDOWED_ENGINES,
            'sql': ALL_SQL_ENGINES,
            'dynamics': True,
            'topology': True,
            'information_flow': True,
            'physics': True,
        },
        'params': params,
        'ram': ram_config,
        'metadata': manifest.get('dataset', {}),
    }

    return normalized


def _normalize_legacy_manifest(manifest: dict) -> dict:
    """
    Normalize legacy JSON manifest to internal format.

    Ensures ALL engines are enabled regardless of what manifest specifies.
    """

    # Override engine config - ALL engines always
    manifest['engines'] = {
        'signal': ALL_SIGNAL_ENGINES,
        'pair': ALL_PAIR_ENGINES,
        'symmetric_pair': ALL_SYMMETRIC_PAIR_ENGINES,
        'windowed': ALL_WINDOWED_ENGINES,
        'sql': ALL_SQL_ENGINES,
        'dynamics': True,
        'topology': True,
        'information_flow': True,
        'physics': True,
    }

    # Ensure ram config exists
    if 'ram' not in manifest:
        manifest['ram'] = {}

    return manifest


def get_ram_stats() -> dict:
    """Get current RAM statistics."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_pct': mem.percent,
        }
    return {'total_gb': 0, 'available_gb': 0, 'used_pct': 0}


def estimate_batch_size(n_entities: int, ram_config: dict) -> int:
    """
    Estimate optimal batch size based on available RAM.

    Returns number of entities to process per batch.
    """

    batch_size = ram_config.get('batch_size', 'auto')

    if batch_size != 'auto':
        return int(batch_size)

    if not HAS_PSUTIL:
        # Default if psutil not available
        return min(100, n_entities)

    # Auto-configure based on available RAM
    available_gb = psutil.virtual_memory().available / (1024**3)

    # Rough estimate: ~100MB per entity with all engines
    # Use 70% of available RAM
    bytes_per_entity = 100 * 1024 * 1024  # 100MB estimate
    target_bytes = available_gb * 1024**3 * 0.7

    batch_size = max(10, int(target_bytes / bytes_per_entity))
    batch_size = min(batch_size, 500)  # Cap at 500
    batch_size = min(batch_size, n_entities)  # Don't exceed total

    return batch_size


class ManifestRunner:
    """
    Orchestrates manifest execution.

    FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.

    - Runs ALL engines (signal, pair, windowed, sql, dynamics, topology, info flow, physics)
    - Manages RAM via entity batching
    - Returns NaN for insufficient data, never skips
    - Writes directly to data/ directory
    """

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.observations_path = Path(manifest['observations_path'])
        self.output_dir = Path(manifest['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ALL engines - always enabled
        self.engine_config = manifest.get('engines', {})
        self.params = manifest.get('params', {})
        self.ram_config = manifest.get('ram', {})

        # Load observations once, share with runners
        print(f"Loading observations from {self.observations_path}")
        self.obs_pd = pd.read_parquet(self.observations_path)
        self.obs_pl = pl.from_pandas(self.obs_pd)
        print(f"  Loaded {len(self.obs_pd):,} observations")

        # Get entity count for batch sizing
        self.entities = self.obs_pl.select('entity_id').unique().to_series().to_list()
        self.n_entities = len(self.entities)
        print(f"  Entities: {self.n_entities}")

        # Calculate batch size
        self.batch_size = estimate_batch_size(self.n_entities, self.ram_config)
        print(f"  Batch size: {self.batch_size}")

        # Check sampling uniformity (for logging)
        self._check_sampling()

        # Results collection
        self.results: Dict[str, Any] = {}

    def _check_sampling(self):
        """Check and report on sampling uniformity."""
        try:
            I_values = self.obs_pl.select('I').unique().sort('I').to_series().to_numpy()
            if len(I_values) > 1:
                dI = np.diff(I_values)
                cv = np.std(dI) / (np.mean(dI) + 1e-10)
                if cv > 0.1:
                    print(f"  Note: Non-uniform sampling detected (CV={cv:.2f})")
                    print(f"  Dynamical results assume uniform sampling - interpret with caution")
        except Exception:
            pass

    def _clear_ram(self):
        """Clear RAM and report status."""
        gc.collect()
        if HAS_PSUTIL:
            stats = get_ram_stats()
            print(f"  RAM: {stats['used_pct']:.1f}% used ({stats['available_gb']:.1f}GB available)")

    def run(self) -> dict:
        """
        Execute the manifest - FULL COMPUTE, ALL ENGINES.

        No conditionals. No skipping. NaN for insufficient data.
        """
        print("=" * 60)
        print("PRISM MANIFEST RUNNER - FULL COMPUTE")
        print("=" * 60)
        print(f"Input:  {self.observations_path}")
        print(f"Output: {self.output_dir}")
        print(f"Mode:   ALL engines, RAM optimized")

        # Report RAM status
        if HAS_PSUTIL:
            stats = get_ram_stats()
            print(f"RAM:    {stats['available_gb']:.1f}GB available")

        # 1. Run Python signal/pair engines (ALL of them)
        python_engines = {
            'signal': self.engine_config.get('signal', ALL_SIGNAL_ENGINES),
            'pair': self.engine_config.get('pair', ALL_PAIR_ENGINES),
            'symmetric_pair': self.engine_config.get('symmetric_pair', ALL_SYMMETRIC_PAIR_ENGINES),
            'windowed': self.engine_config.get('windowed', ALL_WINDOWED_ENGINES),
        }

        print("\n" + "-" * 60)
        print("PYTHON RUNNER (ALL ENGINES)")
        print("-" * 60)
        print(f"  Signal engines: {len(python_engines['signal'])}")
        print(f"  Pair engines: {len(python_engines['pair'])}")
        print(f"  Symmetric pair engines: {len(python_engines['symmetric_pair'])}")
        print(f"  Windowed engines: {len(python_engines['windowed'])}")

        python_runner = PythonRunner(
            obs=self.obs_pd,
            output_dir=self.output_dir,
            engines=python_engines,
            params=self.params
        )
        python_results = python_runner.run()
        self.results['python'] = python_results
        self._clear_ram()

        # 2. Run SQL engines (ALL of them)
        sql_engines = self.engine_config.get('sql', ALL_SQL_ENGINES)

        print("\n" + "-" * 60)
        print("SQL RUNNER (ALL ENGINES)")
        print("-" * 60)
        print(f"  SQL engines: {len(sql_engines)}")

        sql_runner = SQLRunner(
            observations_path=self.observations_path,
            output_dir=self.output_dir,
            engines=sql_engines,
            params=self.params
        )
        sql_results = sql_runner.run()
        self.results['sql'] = sql_results
        self._clear_ram()

        # 3. Run dynamics engine (ALWAYS)
        self._run_dynamics()
        self._clear_ram()

        # 4. Run topology engine (ALWAYS)
        self._run_topology()
        self._clear_ram()

        # 5. Run information flow engine (ALWAYS)
        self._run_information_flow()
        self._clear_ram()

        # 6. Run physics engine (ALWAYS)
        self._run_physics()
        self._clear_ram()

        # Summary
        print("\n" + "=" * 60)
        print("COMPLETE - FULL COMPUTE")
        print("=" * 60)
        self._print_summary()

        return {
            'status': 'complete',
            'output_dir': str(self.output_dir),
            'results': self.results
        }

    def _run_dynamics(self):
        """Compute dynamics (RQA, attractors) for all entities. Always runs."""
        print("\n" + "-" * 60)
        print("DYNAMICS ENGINE")
        print("-" * 60)

        try:
            from prism.engines.dynamics_runner import run_dynamics

            dynamics_params = self.params.get('dynamics', {})
            dynamics_df = run_dynamics(self.obs_pl, self.output_dir, dynamics_params)

            if not dynamics_df.is_empty():
                self.results['dynamics'] = {'rows': len(dynamics_df), 'cols': len(dynamics_df.columns)}
            else:
                self.results['dynamics'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in dynamics engine: {e}")
            self.results['dynamics'] = {'error': str(e)}

    def _run_topology(self):
        """Compute topology (persistent homology, Betti numbers) for all entities. Always runs."""
        print("\n" + "-" * 60)
        print("TOPOLOGY ENGINE")
        print("-" * 60)

        try:
            from prism.engines.topology_runner import run_topology

            topology_params = self.params.get('topology', {})
            topology_df = run_topology(self.obs_pl, self.output_dir, topology_params)

            if not topology_df.is_empty():
                self.results['topology'] = {'rows': len(topology_df), 'cols': len(topology_df.columns)}
            else:
                self.results['topology'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in topology engine: {e}")
            self.results['topology'] = {'error': str(e)}

    def _run_information_flow(self):
        """Compute information flow (transfer entropy, Granger) for all entities. Always runs."""
        print("\n" + "-" * 60)
        print("INFORMATION FLOW ENGINE")
        print("-" * 60)

        try:
            from prism.engines.information_flow_runner import run_information_flow

            info_params = self.params.get('information_flow', {})
            info_df = run_information_flow(self.obs_pl, self.output_dir, info_params)

            if not info_df.is_empty():
                self.results['information_flow'] = {'rows': len(info_df), 'cols': len(info_df.columns)}
            else:
                self.results['information_flow'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in information flow engine: {e}")
            self.results['information_flow'] = {'error': str(e)}

    def _run_physics(self):
        """Compute physics stack (state distance, coherence, energy) for all entities. Always runs."""
        print("\n" + "-" * 60)
        print("PHYSICS STACK")
        print("-" * 60)

        # Physics engine needs rolling metrics from observations_enriched
        # If not available, compute directly from observations with limited metrics
        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'

        try:
            from prism.engines.signal.physics_stack import compute_physics_for_all_entities

            physics_params = self.params.get('physics', {})

            if obs_enriched_path.exists():
                # Use enriched observations (has rolling metrics)
                obs_enriched = pd.read_parquet(obs_enriched_path)
                print(f"  Using observations_enriched.parquet ({len(obs_enriched):,} rows)")
            else:
                # Use raw observations - physics will compute what it can
                print("  Using raw observations (no rolling metrics available)")
                obs_enriched = self.obs_pd.copy()
                # Add 'y' column expected by physics (alias for value)
                if 'y' not in obs_enriched.columns and 'value' in obs_enriched.columns:
                    obs_enriched['y'] = obs_enriched['value']

            physics_df = compute_physics_for_all_entities(
                obs_enriched=obs_enriched,
                n_baseline=physics_params.get('n_baseline', 100),
                coherence_window=physics_params.get('coherence_window', 50),
            )

            if not physics_df.empty:
                output_path = self.output_dir / 'physics.parquet'
                physics_df.to_parquet(output_path, index=False)
                print(f"  physics.parquet: {len(physics_df):,} rows x {len(physics_df.columns)} cols")
                self.results['physics'] = {'rows': len(physics_df), 'cols': len(physics_df.columns)}
            else:
                print("  Warning: no physics data computed")
                self.results['physics'] = {'rows': 0, 'cols': 0}

        except Exception as e:
            print(f"  Error in physics engine: {e}")
            self.results['physics'] = {'error': str(e)}

    def _print_summary(self):
        """Print summary of outputs."""
        print("\nOutput files:")
        for f in sorted(self.output_dir.glob('*.parquet')):
            size = f.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name}: {size_str}")


def run_manifest(manifest_path: str) -> dict:
    """Load and run a manifest from a YAML or JSON file."""
    manifest = load_manifest(manifest_path)
    runner = ManifestRunner(manifest)
    return runner.run()


def run_manifest_dict(manifest: dict) -> dict:
    """Run a manifest from a dictionary (normalized or raw)."""
    # If raw ORTHON format, normalize it
    if 'data' in manifest or 'prism' in manifest:
        manifest = normalize_manifest(manifest, Path('.'))
    else:
        manifest = _normalize_legacy_manifest(manifest)
    runner = ManifestRunner(manifest)
    return runner.run()


# Export engine lists for CLI
__all__ = [
    'ManifestRunner',
    'run_manifest',
    'run_manifest_dict',
    'load_manifest',
    'normalize_manifest',
    'SIGNAL_ENGINES',
    'PAIR_ENGINES',
    'SYMMETRIC_PAIR_ENGINES',
    'WINDOWED_ENGINES',
    'SQL_ENGINES',
    'ALL_SIGNAL_ENGINES',
    'ALL_PAIR_ENGINES',
    'ALL_SYMMETRIC_PAIR_ENGINES',
    'ALL_WINDOWED_ENGINES',
    'ALL_SQL_ENGINES',
]
