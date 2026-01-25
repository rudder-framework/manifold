#!/usr/bin/env python3
"""
Hurst Exponent Engine
=====================

Computes memory (long-range dependence) via Hurst exponent.

The Hurst exponent H measures persistence in a time series:
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trending, positive autocorrelation)
- H < 0.5: Anti-persistent (mean-reverting, negative autocorrelation)

Methods:
1. Rescaled Range (R/S) Analysis - classic method
2. Detrended Fluctuation Analysis (DFA) - more robust to trends
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False


@dataclass
class HurstResult:
    """Result of Hurst exponent computation."""
    hurst: float              # H value [0, 1]
    memory: float             # Same as hurst, for consistency
    memory_type: str          # "persistent" | "anti_persistent" | "random"
    confidence: float         # R² of fit
    method: str               # Which algorithm was used
    fit_points: int           # Number of points in regression


def compute(signal: np.ndarray,
            method: str = "dfa",
            min_samples: int = 100) -> HurstResult:
    """
    Compute Hurst exponent for a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    method : str
        "dfa" (Detrended Fluctuation Analysis) or "rs" (Rescaled Range)
    min_samples : int
        Minimum samples required
        
    Returns
    -------
    HurstResult
        Contains H, memory type classification, and fit quality
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    
    # Validate
    if n < min_samples:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_data",
            fit_points=0
        )
    
    # Remove NaN/Inf
    signal = signal[np.isfinite(signal)]
    if len(signal) < min_samples:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_valid_data",
            fit_points=0
        )
    
    # Check for constant signal
    if np.std(signal) < 1e-10:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=1.0,
            method="constant_signal",
            fit_points=0
        )
    
    # Compute using requested method
    if NOLDS_AVAILABLE:
        if method == "dfa":
            return _compute_nolds_dfa(signal)
        else:
            return _compute_nolds_rs(signal)
    else:
        if method == "dfa":
            return _compute_dfa_manual(signal)
        else:
            return _compute_rs_manual(signal)


def _classify_memory(H: float) -> str:
    """Classify memory type based on Hurst exponent."""
    if H > 0.55:
        return "persistent"
    elif H < 0.45:
        return "anti_persistent"
    else:
        return "random"


def _compute_nolds_dfa(signal: np.ndarray) -> HurstResult:
    """Compute using nolds DFA."""
    try:
        H = nolds.dfa(signal)
        
        if np.isnan(H) or np.isinf(H):
            return _compute_dfa_manual(signal)  # Fallback
        
        H = float(np.clip(H, 0, 1))
        
        return HurstResult(
            hurst=H,
            memory=H,
            memory_type=_classify_memory(H),
            confidence=0.9,  # nolds doesn't return R², estimate
            method="nolds_dfa",
            fit_points=0
        )
    except Exception:
        return _compute_dfa_manual(signal)


def _compute_nolds_rs(signal: np.ndarray) -> HurstResult:
    """Compute using nolds R/S analysis."""
    try:
        H = nolds.hurst_rs(signal)
        
        if np.isnan(H) or np.isinf(H):
            return _compute_rs_manual(signal)
        
        H = float(np.clip(H, 0, 1))
        
        return HurstResult(
            hurst=H,
            memory=H,
            memory_type=_classify_memory(H),
            confidence=0.85,
            method="nolds_rs",
            fit_points=0
        )
    except Exception:
        return _compute_rs_manual(signal)


def _compute_rs_manual(signal: np.ndarray) -> HurstResult:
    """
    Manual Rescaled Range (R/S) analysis.
    
    R/S = (max(cumsum) - min(cumsum)) / std
    
    For self-similar processes: E[R/S] ~ n^H
    """
    n = len(signal)
    
    # Use various window sizes
    min_window = 10
    max_window = n // 4
    
    if max_window < min_window:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_for_rs",
            fit_points=0
        )
    
    # Window sizes (logarithmically spaced)
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        20
    ).astype(int))
    
    log_n = []
    log_rs = []
    
    for window_size in window_sizes:
        if window_size > n:
            continue
        
        rs_values = []
        
        # Compute R/S for non-overlapping windows
        for start in range(0, n - window_size + 1, window_size):
            window = signal[start:start + window_size]
            
            # Mean and std
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            
            if std < 1e-10:
                continue
            
            # Cumulative deviation from mean
            deviation = window - mean
            cumsum = np.cumsum(deviation)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Rescaled range
            rs = R / std
            rs_values.append(rs)
        
        if rs_values:
            log_n.append(np.log(window_size))
            log_rs.append(np.log(np.mean(rs_values)))
    
    if len(log_n) < 3:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_rs_points",
            fit_points=len(log_n)
        )
    
    # Linear regression: log(R/S) = H * log(n) + c
    log_n = np.array(log_n)
    log_rs = np.array(log_rs)
    
    # Fit
    coeffs = np.polyfit(log_n, log_rs, 1)
    H = coeffs[0]
    
    # Compute R² for confidence
    y_pred = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_rs - y_pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    H = float(np.clip(H, 0, 1))
    
    return HurstResult(
        hurst=H,
        memory=H,
        memory_type=_classify_memory(H),
        confidence=float(max(0, r_squared)),
        method="rs_manual",
        fit_points=len(log_n)
    )


def _compute_dfa_manual(signal: np.ndarray) -> HurstResult:
    """
    Manual Detrended Fluctuation Analysis.
    
    DFA is more robust than R/S for non-stationary signals with trends.
    """
    n = len(signal)
    
    # Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(signal - np.mean(signal))
    
    # Window sizes
    min_window = 10
    max_window = n // 4
    
    if max_window < min_window:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_for_dfa",
            fit_points=0
        )
    
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        20
    ).astype(int))
    
    log_n = []
    log_F = []
    
    for window_size in window_sizes:
        if window_size > n:
            continue
        
        # Number of windows
        n_windows = n // window_size
        if n_windows < 1:
            continue
        
        fluctuations = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            segment = y[start:end]
            
            # Fit linear trend
            x = np.arange(window_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Detrended fluctuation
            F = np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(F)
        
        if fluctuations:
            log_n.append(np.log(window_size))
            log_F.append(np.log(np.mean(fluctuations)))
    
    if len(log_n) < 3:
        return HurstResult(
            hurst=0.5,
            memory=0.5,
            memory_type="random",
            confidence=0.0,
            method="insufficient_dfa_points",
            fit_points=len(log_n)
        )
    
    # Linear regression: log(F) = H * log(n) + c
    log_n = np.array(log_n)
    log_F = np.array(log_F)
    
    coeffs = np.polyfit(log_n, log_F, 1)
    H = coeffs[0]
    
    # R² for confidence
    y_pred = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_F - y_pred) ** 2)
    ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    H = float(np.clip(H, 0, 1))
    
    return HurstResult(
        hurst=H,
        memory=H,
        memory_type=_classify_memory(H),
        confidence=float(max(0, r_squared)),
        method="dfa_manual",
        fit_points=len(log_n)
    )


def compute_mfdfa(signal: np.ndarray,
                  q_range: List[float] = None) -> dict:
    """
    Multifractal Detrended Fluctuation Analysis.
    
    Computes generalized Hurst exponent h(q) for different moment orders q.
    - If h(q) is constant: monofractal
    - If h(q) varies with q: multifractal
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    q_range : List[float]
        Moment orders to compute. Default [-3, -2, -1, 0, 1, 2, 3]
        
    Returns
    -------
    dict with generalized Hurst exponents
    """
    if q_range is None:
        q_range = [-3, -2, -1, 0, 1, 2, 3]
    
    n = len(signal)
    y = np.cumsum(signal - np.mean(signal))
    
    min_window = 10
    max_window = n // 4
    
    if max_window < min_window:
        return {"error": "insufficient_data", "h_q": {}}
    
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        15
    ).astype(int))
    
    h_q = {}
    
    for q in q_range:
        log_n = []
        log_Fq = []
        
        for window_size in window_sizes:
            n_windows = n // window_size
            if n_windows < 1:
                continue
            
            fluctuations = []
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                segment = y[start:end]
                
                x = np.arange(window_size)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                F2 = np.mean((segment - trend) ** 2)
                fluctuations.append(F2)
            
            if not fluctuations:
                continue
            
            # q-th order fluctuation
            if q == 0:
                Fq = np.exp(0.5 * np.mean(np.log(fluctuations)))
            else:
                Fq = np.power(np.mean(np.power(fluctuations, q / 2)), 1 / q)
            
            log_n.append(np.log(window_size))
            log_Fq.append(np.log(Fq))
        
        if len(log_n) >= 3:
            coeffs = np.polyfit(log_n, log_Fq, 1)
            h_q[q] = float(coeffs[0])
    
    # Check for multifractality
    if h_q:
        h_values = list(h_q.values())
        h_range = max(h_values) - min(h_values)
        is_multifractal = h_range > 0.1
    else:
        h_range = 0
        is_multifractal = False
    
    return {
        "h_q": h_q,
        "h_range": h_range,
        "is_multifractal": is_multifractal,
        "method": "mfdfa"
    }
