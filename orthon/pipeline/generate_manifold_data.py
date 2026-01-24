#!/usr/bin/env python3
"""
ORTHON Manifold Data Generator
==============================

Converts ORTHON parquet outputs into JSON for the Phase Space Manifold Viewer.

Pipeline:
    signal_typology.parquet + dynamical_systems.parquet + causal_mechanics.parquet
    → Dimensionality reduction (PCA/UMAP)
    → 3D phase space coordinates
    → manifold.json

Usage:
    python -m orthon.pipeline.generate_manifold_data
    python -m orthon.pipeline.generate_manifold_data --method umap
    python -m orthon.pipeline.generate_manifold_data --output viewer/data/manifold.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import polars as pl


def load_orthon_data(data_dir: Path) -> Dict[str, pl.DataFrame]:
    """Load all ORTHON parquet files."""
    files = {
        'typology': 'signal_typology.parquet',
        'geometry': 'manifold_geometry.parquet',
        'dynamics': 'dynamical_systems.parquet',
        'mechanics': 'causal_mechanics.parquet',
    }

    data = {}
    for key, filename in files.items():
        path = data_dir / filename
        if path.exists():
            data[key] = pl.read_parquet(path)
            print(f"  Loaded {filename}: {len(data[key]):,} rows")
        else:
            print(f"  Warning: {filename} not found")

    return data


def build_state_vectors(data: Dict[str, pl.DataFrame]) -> tuple:
    """
    Build state vectors for dimensionality reduction.

    Returns:
        (state_matrix, window_metadata, signal_ids)
    """
    # Primary: use dynamics data (has window_idx)
    if 'dynamics' not in data:
        raise ValueError("dynamical_systems.parquet required")

    dynamics = data['dynamics']

    # Get unique entities and windows
    entities = dynamics['entity_id'].unique().sort().to_list()
    print(f"  Found {len(entities)} entities")

    # Build state vectors per window
    state_vectors = []
    window_metadata = []

    # Numeric columns to include in state vector
    numeric_cols = [
        'stability', 'predictability', 'coupling', 'memory',
        'metric_mean_correlation', 'metric_network_density',
        'metric_n_clusters', 'metric_curvature_forman',
    ]

    # Filter to columns that exist
    available_cols = [c for c in numeric_cols if c in dynamics.columns]
    print(f"  Using {len(available_cols)} numeric features")

    for entity_id in entities:
        entity_df = dynamics.filter(pl.col('entity_id') == entity_id).sort('window_idx')

        for row in entity_df.iter_rows(named=True):
            # Build state vector from available columns
            vector = []
            for col in available_cols:
                val = row.get(col, 0.0)
                if val is None:
                    val = 0.0
                vector.append(float(val))

            state_vectors.append(vector)

            # Metadata for this window
            window_metadata.append({
                'entity_id': entity_id,
                'window_idx': row.get('window_idx', 0),
                'timestamp': str(row.get('timestamp', '')),
                'trajectory': row.get('trajectory', 'stationary'),
                'dynamics_class': row.get('dynamics_class', 'stable_coupled'),
                'stability': row.get('stability', 0.5),
                'coupling': row.get('coupling', 0.5),
            })

    state_matrix = np.array(state_vectors)
    print(f"  Built state matrix: {state_matrix.shape}")

    return state_matrix, window_metadata, available_cols


def reduce_dimensions(state_matrix: np.ndarray, method: str = 'pca') -> np.ndarray:
    """
    Reduce state vectors to 3D.

    Args:
        state_matrix: (n_samples, n_features)
        method: 'pca' or 'umap'

    Returns:
        embedding: (n_samples, 3)
    """
    print(f"  Reducing dimensions using {method}...")

    # Normalize
    mean = np.mean(state_matrix, axis=0)
    std = np.std(state_matrix, axis=0) + 1e-8
    normalized = (state_matrix - mean) / std

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3)
        embedding = reducer.fit_transform(normalized)
        print(f"  Variance explained: {sum(reducer.explained_variance_ratio_):.2%}")

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean'
            )
            embedding = reducer.fit_transform(normalized)
        except ImportError:
            print("  UMAP not installed, falling back to PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3)
            embedding = reducer.fit_transform(normalized)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale to reasonable range
    embedding = embedding * 2

    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Position range: [{embedding.min():.2f}, {embedding.max():.2f}]")

    return embedding


def classify_regime(meta: Dict) -> str:
    """Classify regime based on dynamics metadata."""
    dynamics_class = meta.get('dynamics_class', 'stable_coupled')
    stability = meta.get('stability', 0.5)
    trajectory = meta.get('trajectory', 'stationary')

    if dynamics_class in ['critical', 'chaotic'] or stability < -0.3:
        return 'critical'
    elif dynamics_class in ['unstable', 'evolving'] or trajectory == 'diverging':
        return 'transitional'
    elif trajectory == 'converging' and stability > 0.3:
        return 'recovery'
    else:
        return 'stable'


def compute_coherence(meta: Dict) -> float:
    """Compute coherence from dynamics metadata."""
    coupling = meta.get('coupling', 0.5)
    stability = meta.get('stability', 0.5)

    # Coherence is high when coupling is high and stability is positive
    coherence = (coupling + (stability + 1) / 2) / 2
    return float(np.clip(coherence, 0, 1))


def build_trajectory(
    embedding: np.ndarray,
    window_metadata: List[Dict]
) -> List[Dict]:
    """Build trajectory data for JSON output."""
    trajectory = []

    for i, (pos, meta) in enumerate(zip(embedding, window_metadata)):
        regime = classify_regime(meta)
        coherence = compute_coherence(meta)

        point = {
            't': i,
            'timestamp': meta['timestamp'],
            'position': [float(pos[0]), float(pos[1]), float(pos[2])],
            'regime': regime,
            'coherence': coherence,
            'dominant_signals': [],
            'alerts': []
        }

        # Add alerts for critical events
        if regime == 'critical':
            point['alerts'].append('regime_break')
        if coherence < 0.4:
            point['alerts'].append('coherence_collapse')

        trajectory.append(point)

    return trajectory


def build_signals(
    data: Dict[str, pl.DataFrame],
    embedding: np.ndarray,
    window_metadata: List[Dict]
) -> Dict[str, Any]:
    """Build signal trajectories and metadata."""
    signals = {}

    # If we have mechanics data, use it for causal roles
    roles = {}
    if 'mechanics' in data:
        mechanics = data['mechanics']
        for row in mechanics.iter_rows(named=True):
            signal_id = row.get('signal_id', '')
            energy_class = row.get('energy_class', 'UNDETERMINED')

            if energy_class == 'CONSERVATIVE':
                roles[signal_id] = 'SOURCE'
            elif energy_class == 'DISSIPATIVE':
                roles[signal_id] = 'SINK'
            elif energy_class == 'DRIVEN':
                roles[signal_id] = 'CONDUIT'
            else:
                roles[signal_id] = 'ISOLATED'

    # If we have typology data, get signal list
    if 'typology' in data:
        typology = data['typology']
        signal_ids = typology['signal_id'].unique().to_list()[:10]  # Limit for viz

        for signal_id in signal_ids:
            role = roles.get(signal_id, 'CONDUIT')

            # Color based on role
            if role == 'SOURCE':
                color = '#ff6b6b'
            elif role == 'SINK':
                color = '#4ecdc4'
            else:
                color = '#a0aec0'

            # Generate signal trajectory (offset from main trajectory)
            np.random.seed(hash(signal_id) % 2**32)
            offset_scale = 0.3

            signal_trajectory = []
            for i, pos in enumerate(embedding):
                offset = np.random.randn(3) * offset_scale
                signal_pos = pos + offset
                signal_trajectory.append([
                    float(signal_pos[0]),
                    float(signal_pos[1]),
                    float(signal_pos[2])
                ])

            signals[signal_id] = {
                'role': role,
                'color': color,
                'trajectory': signal_trajectory,
                'typology_shifts': []
            }

    return signals


def build_basins(
    embedding: np.ndarray,
    window_metadata: List[Dict]
) -> List[Dict]:
    """Identify basins from trajectory clustering."""
    # Simple basin detection: find regions of high density
    basins = []

    # Use k-means to find basin centers
    try:
        from sklearn.cluster import KMeans

        # Only cluster if enough points
        if len(embedding) > 10:
            n_clusters = min(3, len(embedding) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embedding)

            for i, center in enumerate(kmeans.cluster_centers_):
                # Determine basin regime from majority of points
                basin_points = [
                    window_metadata[j]
                    for j in range(len(labels))
                    if labels[j] == i
                ]

                if basin_points:
                    regimes = [classify_regime(p) for p in basin_points]
                    majority_regime = max(set(regimes), key=regimes.count)

                    # Compute radius
                    basin_embedding = embedding[labels == i]
                    radius = float(np.std(basin_embedding) * 2)

                    basins.append({
                        'id': f'basin_{i+1}',
                        'centroid': [float(center[0]), float(center[1]), float(center[2])],
                        'radius': max(radius, 0.5),
                        'regime': majority_regime,
                        'member_signals': []
                    })

    except ImportError:
        print("  sklearn not available, skipping basin detection")

    return basins


def generate_manifold_json(
    data_dir: Path,
    output_path: Path,
    method: str = 'pca'
) -> None:
    """
    Generate manifold.json from ORTHON parquet outputs.

    Args:
        data_dir: Directory containing parquet files
        output_path: Output path for manifold.json
        method: Dimensionality reduction method ('pca' or 'umap')
    """
    print("ORTHON Manifold Data Generator")
    print("=" * 50)

    # Load data
    print("\nLoading ORTHON data...")
    data = load_orthon_data(data_dir)

    if not data:
        print("ERROR: No ORTHON data found")
        sys.exit(1)

    # Build state vectors
    print("\nBuilding state vectors...")
    state_matrix, window_metadata, feature_names = build_state_vectors(data)

    # Dimensionality reduction
    print("\nReducing dimensions...")
    embedding = reduce_dimensions(state_matrix, method)

    # Build output structure
    print("\nBuilding manifold structure...")

    trajectory = build_trajectory(embedding, window_metadata)
    signals = build_signals(data, embedding, window_metadata)
    basins = build_basins(embedding, window_metadata)

    # Detect transitions
    transitions = []
    for i in range(1, len(trajectory)):
        prev = trajectory[i-1]
        curr = trajectory[i]

        if prev['regime'] != curr['regime']:
            transitions.append({
                't': i,
                'from_regime': prev['regime'],
                'to_regime': curr['regime'],
                'type': 'regime_shift'
            })

    # Build final JSON
    manifold_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'method': method,
            'n_signals': len(signals),
            'n_timesteps': len(trajectory),
            'n_features': len(feature_names),
            'features': feature_names,
            'time_range': [
                window_metadata[0]['timestamp'] if window_metadata else '',
                window_metadata[-1]['timestamp'] if window_metadata else ''
            ]
        },
        'trajectory': trajectory,
        'signals': signals,
        'basins': basins,
        'transitions': transitions
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifold_data, f, indent=2)

    print(f"\nOutput written to: {output_path}")
    print(f"  Trajectory points: {len(trajectory)}")
    print(f"  Signals: {len(signals)}")
    print(f"  Basins: {len(basins)}")
    print(f"  Transitions: {len(transitions)}")


def main():
    parser = argparse.ArgumentParser(description="Generate ORTHON manifold data")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing ORTHON parquet files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='orthon/viewer/data/manifold.json',
        help='Output path for manifold.json'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['pca', 'umap'],
        default='pca',
        help='Dimensionality reduction method'
    )
    args = parser.parse_args()

    generate_manifold_json(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        method=args.method
    )


if __name__ == '__main__':
    main()
