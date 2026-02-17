"""
Stage 06: Signal Pairwise Entry Point
=====================================

Orchestration - reads parquets, builds eigenvector gating, calls core engine,
writes output.

Inputs:
    - signal_vector.parquet
    - state_vector.parquet
    - state_geometry.parquet (optional, for eigenvector gating)

Output:
    - signal_pairwise.parquet

Computes pairwise relationships between signals:
    - Correlation
    - Distance
    - Cosine similarity
    - PC co-loading (for Granger gating)
"""

import polars as pl
from pathlib import Path
from typing import Optional

from manifold.core.signal_pairwise import compute_signal_pairwise
from manifold.io.writer import write_output


def _load_eigenvector_gating(state_geometry_path: str, verbose: bool = True) -> dict:
    """
    Build eigenvector gating dict from loadings sidecar (if available).

    Tries narrow loadings sidecar first, then falls back to legacy wide format.
    """
    eigenvector_gating = {}
    try:
        # Try narrow loadings sidecar first (new format)
        loadings_path = str(Path(state_geometry_path).parent / 'state_geometry_loadings.parquet')
        if Path(loadings_path).exists():
            loadings_df = pl.read_parquet(loadings_path)
            if verbose:
                print(f"Eigenvector gating from loadings sidecar: {len(loadings_df)} rows")
            for row in loadings_df.iter_rows(named=True):
                key = (row.get('cohort'), row.get('signal_0_end'), row.get('engine'))
                if key not in eigenvector_gating:
                    eigenvector_gating[key] = {}
                if row.get('pc1_loading') is not None:
                    eigenvector_gating[key][row['signal_id']] = row['pc1_loading']
        else:
            # Backward compat: read wide pc1_signal_* columns from state_geometry
            sg = pl.read_parquet(state_geometry_path)
            pc1_cols = [c for c in sg.columns if c.startswith('pc1_signal_')]
            if pc1_cols and verbose:
                print(f"Eigenvector gating (legacy wide format): {len(pc1_cols)} signal loadings found")
                for row in sg.iter_rows(named=True):
                    key = (row.get('cohort'), row.get('signal_0_end'), row.get('engine'))
                    loadings = {}
                    for col in pc1_cols:
                        sig_id = col.replace('pc1_signal_', '')
                        if row[col] is not None:
                            loadings[sig_id] = row[col]
                    if loadings:
                        eigenvector_gating[key] = loadings
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load eigenvector gating: {e}")

    return eigenvector_gating


def run(
    signal_vector_path: str,
    state_vector_path: str,
    data_path: str = ".",
    state_geometry_path: Optional[str] = None,
    coloading_threshold: float = 0.1,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal pairwise computation with eigenvector gating.

    Args:
        signal_vector_path: Path to signal_vector.parquet
        state_vector_path: Path to state_vector.parquet
        data_path: Root data directory (for write_output)
        state_geometry_path: Path to state_geometry.parquet (for PC gating)
        coloading_threshold: Threshold for PC co-loading to flag Granger
        verbose: Print progress

    Returns:
        Signal pairwise DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 06: SIGNAL PAIRWISE")
        print("Pairwise relationships with eigenvector gating")
        print("=" * 70)

    signal_vector = pl.read_parquet(signal_vector_path)
    state_vector = pl.read_parquet(state_vector_path)

    # Build eigenvector gating dict from sidecar (if available)
    eigenvector_gating = {}
    if state_geometry_path is not None:
        eigenvector_gating = _load_eigenvector_gating(state_geometry_path, verbose=verbose)

    result = compute_signal_pairwise(
        signal_vector,
        state_vector,
        eigenvector_gating=eigenvector_gating,
        coloading_threshold=coloading_threshold,
        verbose=verbose,
    )

    write_output(result, data_path, 'signal_pairwise', verbose=verbose)

    return result
