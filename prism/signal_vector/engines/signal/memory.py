"""
Memory Engines
===============

Time-series memory and autocorrelation analysis.

Engines:
- hurst: Hurst exponent (long-range dependence)
- acf_decay: Autocorrelation decay rate
"""

import numpy as np
from typing import Dict, Any


def compute_hurst(values: np.ndarray) -> Dict[str, float]:
    """
    Compute Hurst exponent using R/S analysis.
    
    H = 0.5: Random walk (no memory)
    H > 0.5: Persistent (trending)
    H < 0.5: Anti-persistent (mean-reverting)
    """
    if len(values) < 20:
        return {'hurst': np.nan}
    
    n = len(values)
    
    # Use multiple scales
    max_k = min(n // 4, 100)
    if max_k < 8:
        return {'hurst': np.nan}
    
    scales = []
    rs_values = []
    
    for k in range(8, max_k + 1, max(1, max_k // 20)):
        # Number of subseries
        n_subseries = n // k
        if n_subseries < 1:
            continue
        
        rs_list = []
        for i in range(n_subseries):
            subseries = values[i * k:(i + 1) * k]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(subseries)
            cumsum = np.cumsum(subseries - mean)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(subseries, ddof=1)
            
            if S > 1e-10:
                rs_list.append(R / S)
        
        if rs_list:
            scales.append(k)
            rs_values.append(np.mean(rs_list))
    
    if len(scales) < 3:
        return {'hurst': np.nan}
    
    # Linear regression on log-log
    log_scales = np.log(scales)
    log_rs = np.log(rs_values)
    
    # Hurst = slope of log(R/S) vs log(n)
    slope, _ = np.polyfit(log_scales, log_rs, 1)
    
    # Clamp to valid range
    hurst = np.clip(slope, 0.0, 1.0)
    
    return {'hurst': float(hurst)}


def compute_acf_decay(values: np.ndarray, max_lag: int = None) -> Dict[str, float]:
    """
    Compute autocorrelation decay characteristics.
    
    Returns:
    - acf_half_life: Lag at which ACF drops to 0.5
    - acf_decay_rate: Exponential decay rate
    - acf_first_zero: First lag with ACF ≤ 0
    """
    if len(values) < 10:
        return {
            'acf_half_life': np.nan,
            'acf_decay_rate': np.nan,
            'acf_first_zero': np.nan,
        }
    
    n = len(values)
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    # Compute ACF
    mean = np.mean(values)
    var = np.var(values)
    
    if var < 1e-10:
        return {
            'acf_half_life': float(max_lag),
            'acf_decay_rate': 0.0,
            'acf_first_zero': float(max_lag),
        }
    
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            cov = np.mean((values[:-lag] - mean) * (values[lag:] - mean))
            acf.append(cov / var)
    
    acf = np.array(acf)
    
    # Half-life: first lag where ACF < 0.5
    half_life = max_lag
    for i, a in enumerate(acf):
        if a < 0.5:
            half_life = i
            break
    
    # First zero crossing
    first_zero = max_lag
    for i, a in enumerate(acf):
        if a <= 0:
            first_zero = i
            break
    
    # Decay rate: fit exponential to ACF
    # ACF(k) ≈ exp(-λk) → log(ACF) ≈ -λk
    positive_acf = acf[acf > 0.01]
    if len(positive_acf) > 2:
        lags = np.arange(len(positive_acf))
        log_acf = np.log(positive_acf)
        decay_rate, _ = np.polyfit(lags, log_acf, 1)
        decay_rate = -decay_rate  # Make positive
    else:
        decay_rate = 0.0
    
    return {
        'acf_half_life': float(half_life),
        'acf_decay_rate': float(decay_rate),
        'acf_first_zero': float(first_zero),
    }


# Engine registry
ENGINES = {
    'hurst': compute_hurst,
    'acf_decay': compute_acf_decay,
}
