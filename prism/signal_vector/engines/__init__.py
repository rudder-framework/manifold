"""
Engine Registry
================

Collects all signal engines into a single registry.

Organization:
- signal/statistics.py: kurtosis, skewness, crest_factor
- signal/spectral.py: spectral, spectral_entropy, band_power, frequency_bands
- signal/memory.py: hurst, acf_decay
- signal/trend.py: rate_of_change, trend_r2, detrend_std, cusum
"""

from typing import Dict, Callable, Any
import numpy as np

# Import engine modules
from .signal import statistics, spectral, memory, trend


def _collect_engines() -> Dict[str, Callable]:
    """Collect all engines from submodules."""
    registry = {}
    
    # Add from each module
    for module in [statistics, spectral, memory, trend]:
        if hasattr(module, 'ENGINES'):
            registry.update(module.ENGINES)
    
    return registry


# Global engine registry
ENGINE_REGISTRY: Dict[str, Callable] = _collect_engines()


def get_engine(name: str) -> Callable:
    """Get engine function by name."""
    if name not in ENGINE_REGISTRY:
        raise KeyError(f"Unknown engine: {name}. Available: {list_engines()}")
    return ENGINE_REGISTRY[name]


def list_engines() -> list:
    """List all available engine names."""
    return sorted(ENGINE_REGISTRY.keys())


def run_engine(name: str, values: np.ndarray, **kwargs) -> Dict[str, float]:
    """Run a single engine on values."""
    engine = get_engine(name)
    return engine(values, **kwargs)


def run_engines(names: list, values: np.ndarray, **kwargs) -> Dict[str, float]:
    """Run multiple engines on values, return flattened results."""
    results = {}
    for name in names:
        try:
            engine_results = run_engine(name, values, **kwargs)
            results.update(engine_results)
        except Exception:
            pass
    return results
