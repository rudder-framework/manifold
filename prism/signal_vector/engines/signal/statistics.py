"""
Statistics Engines
==================

Basic statistical measures per window.

Engines:
- kurtosis: Tail heaviness (excess kurtosis)
- skewness: Distribution asymmetry
- crest_factor: Peak to RMS ratio
"""

import numpy as np
from typing import Dict, Any


def compute_kurtosis(values: np.ndarray) -> Dict[str, float]:
    """
    Compute excess kurtosis.
    
    kurtosis = E[(X-μ)⁴] / σ⁴ - 3
    
    Interpretation:
    - 0: Normal distribution
    - > 0: Heavy tails (leptokurtic)
    - < 0: Light tails (platykurtic)
    """
    if len(values) < 4:
        return {'kurtosis': np.nan}
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if std < 1e-10:
        return {'kurtosis': 0.0}
    
    n = len(values)
    m4 = np.mean((values - mean) ** 4)
    kurtosis = m4 / (std ** 4) - 3
    
    return {'kurtosis': float(kurtosis)}


def compute_skewness(values: np.ndarray) -> Dict[str, float]:
    """
    Compute skewness.
    
    skewness = E[(X-μ)³] / σ³
    
    Interpretation:
    - 0: Symmetric
    - > 0: Right-skewed (tail extends right)
    - < 0: Left-skewed (tail extends left)
    """
    if len(values) < 3:
        return {'skewness': np.nan}
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if std < 1e-10:
        return {'skewness': 0.0}
    
    m3 = np.mean((values - mean) ** 3)
    skewness = m3 / (std ** 3)
    
    return {'skewness': float(skewness)}


def compute_crest_factor(values: np.ndarray) -> Dict[str, float]:
    """
    Compute crest factor (peak to RMS ratio).
    
    crest_factor = max(|x|) / RMS(x)
    
    Interpretation:
    - Sine wave: √2 ≈ 1.414
    - Higher: More impulsive/peaky
    - Lower: More constant
    """
    if len(values) < 1:
        return {'crest_factor': np.nan}
    
    rms = np.sqrt(np.mean(values ** 2))
    
    if rms < 1e-10:
        return {'crest_factor': 1.0}
    
    peak = np.max(np.abs(values))
    crest_factor = peak / rms
    
    return {'crest_factor': float(crest_factor)}


# Engine registry for this module
ENGINES = {
    'kurtosis': compute_kurtosis,
    'skewness': compute_skewness,
    'crest_factor': compute_crest_factor,
}
