"""
PRISM Manifest Reader
======================

Reads ORTHON manifest v2.2+ and provides signal configuration.
"""

from typing import Dict, Any, List, Iterator, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class SignalConfig:
    """Configuration for a single signal."""
    signal_id: str
    cohort: str
    engines: List[str]
    rolling_engines: List[str]
    window_size: int
    stride: int
    derivative_depth: int
    eigenvalue_budget: int
    temporal_pattern: str
    spectral: str
    representation: str = 'spectral'  # v2.4+
    state_features: List[str] = None
    bands: List[float] = None
    
    def __post_init__(self):
        if self.state_features is None:
            self.state_features = []
        if self.bands is None:
            self.bands = []


class ManifestReader:
    """
    Read and navigate ORTHON manifest.
    
    Supports:
    - v2.2: cohorts: {cohort: {signal: config}}
    - v2.4: adds system window, representation type
    """
    
    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        with open(self.path) as f:
            self.manifest = yaml.safe_load(f)
        
        self.version = self.manifest.get('version', '2.0')
        self._parse_structure()
    
    def _parse_structure(self):
        """Parse manifest structure."""
        self.cohorts = self.manifest.get('cohorts', {})
        self.skip_signals = set(self.manifest.get('skip_signals', []))
        self.pair_engines = self.manifest.get('pair_engines', [])
        self.symmetric_pair_engines = self.manifest.get('symmetric_pair_engines', [])
        
        # v2.4+ system window
        system = self.manifest.get('system', {})
        self.system_window = system.get('window')
        self.system_stride = system.get('stride')
        
        # Paths
        paths = self.manifest.get('paths', {})
        self.observations_path = paths.get('observations')
        self.typology_path = paths.get('typology')
        self.output_dir = paths.get('output_dir')
    
    def get_signal(self, cohort: str, signal_id: str) -> Optional[SignalConfig]:
        """Get configuration for a specific signal."""
        if cohort not in self.cohorts:
            return None
        if signal_id not in self.cohorts[cohort]:
            return None
        
        cfg = self.cohorts[cohort][signal_id]
        typology = cfg.get('typology', {})
        
        return SignalConfig(
            signal_id=signal_id,
            cohort=cohort,
            engines=cfg.get('engines', []),
            rolling_engines=cfg.get('rolling_engines', []),
            window_size=cfg.get('window_size', cfg.get('signal_window', 128)),
            stride=cfg.get('stride', cfg.get('signal_stride', 64)),
            derivative_depth=cfg.get('derivative_depth', 1),
            eigenvalue_budget=cfg.get('eigenvalue_budget', 5),
            temporal_pattern=typology.get('temporal_pattern', 'STATIONARY'),
            spectral=typology.get('spectral', 'NARROWBAND'),
            representation=cfg.get('representation', 'spectral'),
            state_features=cfg.get('state_features', []),
            bands=cfg.get('bands', []),
        )
    
    def iter_signals(self) -> Iterator[SignalConfig]:
        """Iterate over all active signals."""
        for cohort, signals in self.cohorts.items():
            for signal_id in signals:
                skip_key = f"{cohort}/{signal_id}"
                if skip_key not in self.skip_signals:
                    cfg = self.get_signal(cohort, signal_id)
                    if cfg:
                        yield cfg
    
    def iter_cohort_signals(self, cohort: str) -> Iterator[SignalConfig]:
        """Iterate over signals in a specific cohort."""
        if cohort not in self.cohorts:
            return
        for signal_id in self.cohorts[cohort]:
            skip_key = f"{cohort}/{signal_id}"
            if skip_key not in self.skip_signals:
                cfg = self.get_signal(cohort, signal_id)
                if cfg:
                    yield cfg
    
    def get_all_engines(self) -> List[str]:
        """Get unique list of all engines used."""
        engines = set()
        for cfg in self.iter_signals():
            engines.update(cfg.engines)
        return sorted(engines)
    
    def get_pairs(self, cohort: str = None) -> List[Tuple[str, str, str]]:
        """
        Get signal pairs for pairwise engines.
        
        Returns: List of (cohort, signal_a, signal_b) tuples
        """
        pairs = []
        
        if cohort:
            cohorts_to_check = [cohort] if cohort in self.cohorts else []
        else:
            cohorts_to_check = list(self.cohorts.keys())
        
        for c in cohorts_to_check:
            signals = list(self.cohorts[c].keys())
            for i, sig_a in enumerate(signals):
                for sig_b in signals[i+1:]:
                    skip_a = f"{c}/{sig_a}" in self.skip_signals
                    skip_b = f"{c}/{sig_b}" in self.skip_signals
                    if not skip_a and not skip_b:
                        pairs.append((c, sig_a, sig_b))
        
        return pairs
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get manifest summary."""
        return self.manifest.get('summary', {})
    
    def __repr__(self):
        n_signals = sum(len(s) for s in self.cohorts.values())
        n_skip = len(self.skip_signals)
        return f"ManifestReader(v{self.version}, {n_signals} signals, {n_skip} skipped)"
