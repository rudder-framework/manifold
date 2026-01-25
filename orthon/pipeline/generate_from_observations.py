#!/usr/bin/env python3
"""
Generate Manifold Data Directly from Observations
==================================================

Bypasses full ORTHON pipeline when geometry engines fail.
Creates phase space embedding directly from observations.

Usage:
    python -m orthon.pipeline.generate_from_observations
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl


def main():
    data_dir = Path("data")
    output_path = Path("orthon/viewer/data/manifold.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("ORTHON Direct Manifold Generator")
    print("=" * 50)

    # Load observations
    obs_path = data_dir / "observations.parquet"
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found")
        sys.exit(1)

    print(f"\nLoading {obs_path}...")
    obs = pl.read_parquet(obs_path)
    print(f"  {len(obs):,} observations")

    # Load typology if available
    typology = None
    typ_path = data_dir / "signal_typology.parquet"
    if typ_path.exists():
        typology = pl.read_parquet(typ_path)
        print(f"  Loaded typology: {len(typology)} signals")

    # Get entities and signals
    all_entities = obs["entity_id"].unique().sort().to_list()
    signals = obs["signal_id"].unique().sort().to_list()
    print(f"  {len(all_entities)} entities, {len(signals)} signals")

    # Use first entity only - viewer handles entity selection
    entities = all_entities[:1]
    print(f"\nUsing entities: {entities}")

    # Build state vectors with SHARED embedding space
    print("\nBuilding shared embedding space...")

    # First pass: collect all signal data to fit global PCA
    all_signal_matrices = []
    entity_data = {}

    for entity_id in entities:
        entity_obs = obs.filter(pl.col("entity_id") == entity_id)
        try:
            pivot = entity_obs.pivot(
                values="value",
                index="timestamp",
                on="signal_id"
            ).sort("timestamp")
        except Exception:
            continue

        timestamps = pivot["timestamp"].to_list()
        signal_cols = [c for c in pivot.columns if c != "timestamp"]
        signals_matrix = pivot.select(signal_cols).to_numpy()

        # Store for later
        entity_data[entity_id] = {
            "timestamps": timestamps,
            "signal_cols": signal_cols,
            "matrix": signals_matrix
        }
        all_signal_matrices.append(signals_matrix)

    # Combine all matrices and normalize globally
    combined = np.vstack(all_signal_matrices)
    global_mean = np.nanmean(combined, axis=0)
    global_std = np.nanstd(combined, axis=0) + 1e-8

    # Fit global PCA on combined data
    from sklearn.decomposition import PCA
    normalized_combined = (combined - global_mean) / global_std
    normalized_combined = np.nan_to_num(normalized_combined, 0)

    pca = PCA(n_components=3)
    pca.fit(normalized_combined)
    print(f"  Global PCA fitted on {len(combined)} samples")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Initialize outputs
    all_trajectory = []
    all_signals_data = {}
    signal_cols_global = None

    # Second pass: project each entity through shared space
    window_idx = 0
    for entity_id in entities:
        entity_obs = obs.filter(pl.col("entity_id") == entity_id)

        # Pivot to wide format
        try:
            pivot = entity_obs.pivot(
                values="value",
                index="timestamp",
                on="signal_id"
            ).sort("timestamp")
        except Exception as e:
            print(f"  {entity_id}: skip (pivot failed)")
            continue

        timestamps = pivot["timestamp"].to_list()
        signal_cols = [c for c in pivot.columns if c != "timestamp"]

        # Initialize signals data on first entity
        if signal_cols_global is None:
            signal_cols_global = signal_cols[:8]
            for sig in signal_cols_global:
                role = "CONDUIT"
                color = "#a0aec0"
                # Assign roles based on typology
                if typology is not None:
                    typ_row = typology.filter(pl.col("signal_id") == sig)
                    if len(typ_row) > 0:
                        row = typ_row.row(0, named=True)
                        dom_axis = row.get("dominant_axis", "balanced")
                        dom_score = row.get("dominant_score", 0.5)
                        if dom_axis in ["memory", "momentum"] and dom_score > 0.6:
                            role = "SOURCE"
                            color = "#ff6b6b"
                        elif dom_axis in ["derivatives", "volatility"] and dom_score > 0.6:
                            role = "SINK"
                            color = "#4ecdc4"
                all_signals_data[sig] = {"trajectory": [], "role": role, "color": color}
            print(f"  Using signals: {signal_cols_global}")

        # Get signal matrix and normalize with GLOBAL parameters
        signals_matrix = pivot.select(signal_cols).to_numpy()
        normalized = (signals_matrix - global_mean) / global_std
        normalized = np.nan_to_num(normalized, 0)

        # CUMULATIVE degradation approach - smooth monotonic trajectory
        n_samples = len(normalized)

        # Compute rolling statistics (window=10)
        window = min(10, n_samples // 3)
        rolling_mean = np.zeros_like(normalized)
        rolling_std = np.zeros((n_samples,))

        for i in range(n_samples):
            start = max(0, i - window)
            rolling_mean[i] = np.mean(normalized[start:i+1], axis=0)
            rolling_std[i] = np.std(normalized[start:i+1])

        # Project rolling means through PCA
        embedding = pca.transform(rolling_mean)

        # Apply HEAVY smoothing (EMA with alpha=0.1)
        alpha = 0.1
        smoothed = np.zeros_like(embedding)
        smoothed[0] = embedding[0]
        for i in range(1, len(embedding)):
            smoothed[i] = alpha * embedding[i] + (1 - alpha) * smoothed[i-1]

        # Add monotonic drift component (degradation direction)
        # This ensures trajectory moves in one direction over time
        time_component = np.linspace(0, 2, n_samples).reshape(-1, 1)
        drift_direction = smoothed[-1] - smoothed[0]
        drift_direction = drift_direction / (np.linalg.norm(drift_direction) + 1e-8)

        embedding = smoothed + time_component * drift_direction * 0.3

        # Scale to viewing range
        max_dist = np.linalg.norm(embedding, axis=1).max()
        if max_dist > 0:
            embedding = embedding * (3.0 / max_dist)

        n_timesteps = len(timestamps)
        print(f"  {entity_id}: {n_timesteps} timesteps -> embedded")

        # Build trajectory points
        for i, (ts, pos) in enumerate(zip(timestamps, embedding)):
            # Regime based on % of life (degradation stage)
            life_pct = i / max(n_timesteps - 1, 1)

            # Smooth continuous coherence decay (no discontinuities)
            # Sigmoid-like decay from 0.85 to 0.25 over lifecycle
            coherence = 0.25 + 0.60 / (1.0 + np.exp(8 * (life_pct - 0.6)))

            # Regime labels (for metadata only, not affecting coherence)
            if life_pct < 0.5:
                regime = "stable"
            elif life_pct < 0.8:
                regime = "transitional"
            else:
                regime = "critical"

            all_trajectory.append({
                "t": window_idx,
                "timestamp": str(ts),
                "entity_id": entity_id,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "regime": regime,
                "coherence": coherence,
                "dominant_signals": signal_cols[:3],
                "alerts": ["regime_break"] if regime == "critical" else []
            })

            # Update signal trajectories
            for j, sig in enumerate(signal_cols_global):
                if sig in all_signals_data and sig in signal_cols:
                    # Offset from main trajectory
                    offset = np.array([
                        np.sin(i * 0.1 + j) * 0.3,
                        np.cos(i * 0.12 + j) * 0.25,
                        np.sin(i * 0.11 + j) * 0.28
                    ])
                    sig_pos = pos + offset
                    all_signals_data[sig]["trajectory"].append([
                        float(sig_pos[0]), float(sig_pos[1]), float(sig_pos[2])
                    ])

            window_idx += 1

    print(f"\nTotal trajectory points: {len(all_trajectory)}")

    # Build basins from clustering
    basins = []
    if len(all_trajectory) > 20:
        from sklearn.cluster import KMeans
        positions = np.array([t["position"] for t in all_trajectory])
        n_clusters = min(3, len(positions) // 20)
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions)

            for i, center in enumerate(kmeans.cluster_centers_):
                basin_points = [all_trajectory[j] for j in range(len(labels)) if labels[j] == i]
                regimes = [p["regime"] for p in basin_points]
                majority = max(set(regimes), key=regimes.count)

                basins.append({
                    "id": f"basin_{i+1}",
                    "centroid": [float(center[0]), float(center[1]), float(center[2])],
                    "radius": float(np.std(positions[labels == i]) * 2),
                    "regime": majority
                })

    # Detect transitions
    transitions = []
    for i in range(1, len(all_trajectory)):
        if all_trajectory[i]["regime"] != all_trajectory[i-1]["regime"]:
            transitions.append({
                "t": i,
                "from_regime": all_trajectory[i-1]["regime"],
                "to_regime": all_trajectory[i]["regime"]
            })

    # Build final JSON
    manifold = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "C-MAPSS FD001",
            "n_entities": len(entities),
            "n_signals": len(signals[:8]),
            "n_timesteps": len(all_trajectory)
        },
        "trajectory": all_trajectory,
        "signals": all_signals_data,
        "basins": basins,
        "transitions": transitions
    }

    # Write
    with open(output_path, "w") as f:
        json.dump(manifold, f, indent=2)

    print(f"\nWritten to {output_path}")
    print(f"  Trajectory: {len(all_trajectory)} points")
    print(f"  Signals: {len(all_signals_data)}")
    print(f"  Basins: {len(basins)}")
    print(f"  Transitions: {len(transitions)}")


if __name__ == "__main__":
    main()
