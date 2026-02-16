"""
Manifest — parse manifest.yaml into engine config.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_manifest(data_path: str) -> Dict[str, Any]:
    """
    Load manifest.yaml from a data directory.

    Tries:
        1. data_path/manifest.yaml
        2. data_path itself (if it's a .yaml file)
    """
    p = Path(data_path)

    if p.is_file() and p.suffix in ('.yaml', '.yml'):
        manifest_path = p
    else:
        manifest_path = p / 'manifest.yaml'

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.yaml in {data_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Stash the manifest path for resolving relative paths
    manifest['_manifest_path'] = str(manifest_path)
    manifest['_data_dir'] = str(manifest_path.parent)

    return manifest


def get_observations_path(manifest: Dict[str, Any]) -> str:
    """Get absolute path to observations.parquet from manifest."""
    obs_rel = manifest.get('paths', {}).get('observations', 'observations.parquet')
    data_dir = Path(manifest.get('_data_dir', '.'))
    obs_path = data_dir / obs_rel
    return str(obs_path)


def get_output_dir(manifest: Dict[str, Any]) -> str:
    """Get absolute path to output directory from manifest."""
    out_rel = manifest.get('paths', {}).get('output_dir', 'output')
    data_dir = Path(manifest.get('_data_dir', '.'))
    out_path = data_dir / out_rel
    out_path.mkdir(parents=True, exist_ok=True)
    return str(out_path)


def get_typology_path(manifest: Dict[str, Any]) -> Optional[str]:
    """Get path to typology.parquet if it exists."""
    typ_rel = manifest.get('paths', {}).get('typology', 'typology.parquet')
    data_dir = Path(manifest.get('_data_dir', '.'))
    typ_path = data_dir / typ_rel
    return str(typ_path) if typ_path.exists() else None


def get_intervention(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get intervention config if present."""
    return manifest.get('intervention')


def get_coordinate_block(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get coordinate config from manifest, or None if absent."""
    return manifest.get('coordinate')


def get_segments(manifest: Dict[str, Any]) -> Optional[list]:
    """Get segments config, deriving from intervention if needed."""
    segments = manifest.get('segments')
    if segments:
        return segments

    intervention = get_intervention(manifest)
    if intervention and intervention.get('enabled'):
        event_idx = intervention.get('event_index', 20)
        return [
            {'name': 'pre', 'range': [0, event_idx - 1]},
            {'name': 'post', 'range': [event_idx, None]},
        ]
    return None
