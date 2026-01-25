#!/usr/bin/env python3
"""
Lyapunov Exponent Engine
========================

Computes stability via largest Lyapunov exponent.

Lyapunov exponent measures rate of separation of infinitesimally close trajectories:
- λ > 0: chaotic (exponential divergence, unstable)
- λ ≈ 0: edge of chaos
- λ < 0: stable (trajectories converge)

Output normalized to [-1, 1] where:
- +1 = highly stable
- 0 = edge of stability
- -1 = highly unstable (chaotic)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False


@dataclass
class LyapunovResult:
    """Result of Lyapunov exponent computation."""
    lyapunov_exponent: float      # Raw λ value
    stability: float              # Normalized to [-1, 1]
    is_chaotic: bool              # λ > 0
    confidence: float             # 0-1, based on data quality
    method: str                   # Which algorithm was used


def compute(signal: np.ndarray,
            emb_dim: int = 10,
            lag: Optional[int] = None,
            min_samples: int = 100) -> LyapunovResult:
    """
    Compute largest Lyapunov exponent for a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    emb_dim : int
        Embedding dimension for phase space reconstruction
    lag : int, optional
        Time delay for embedding (auto-computed if None)
    min_samples : int
        Minimum samples required
        
    Returns
    -------
    LyapunovResult
        Contains raw exponent, normalized stability, and metadata
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    
    # Validate input
    if n < min_samples:
        return LyapunovResult(
            lyapunov_exponent=0.0,
            stability=0.0,
            is_chaotic=False,
            confidence=0.0,
            method="insufficient_data"
        )
    
    # Remove NaN/Inf
    signal = signal[np.isfinite(signal)]
    if len(signal) < min_samples:
        return LyapunovResult(
            lyapunov_exponent=0.0,
            stability=0.0,
            is_chaotic=False,
            confidence=0.0,
            method="insufficient_valid_data"
        )
    
    # Standardize
    std = np.std(signal)
    if std < 1e-10:
        # Constant signal = perfectly stable
        return LyapunovResult(
            lyapunov_exponent=-1.0,
            stability=1.0,
            is_chaotic=False,
            confidence=1.0,
            method="constant_signal"
        )
    
    signal = (signal - np.mean(signal)) / std
    
    # Auto-compute lag if not provided (first minimum of autocorrelation)
    if lag is None:
        lag = _estimate_lag(signal)
    
    # Compute Lyapunov exponent
    if NOLDS_AVAILABLE:
        lyap, confidence, method = _compute_nolds(signal, emb_dim, lag)
    else:
        lyap, confidence, method = _compute_rosenstein(signal, emb_dim, lag)
    
    # Normalize to [-1, 1]
    # Using tanh with appropriate scaling
    # λ = 0.1 → stability ≈ -0.1 (slightly unstable)
    # λ = -0.1 → stability ≈ 0.1 (slightly stable)
    stability = float(-np.tanh(lyap * 5))  # Scale factor 5 for sensitivity
    stability = np.clip(stability, -1, 1)
    
    return LyapunovResult(
        lyapunov_exponent=float(lyap),
        stability=stability,
        is_chaotic=lyap > 0.01,  # Small positive threshold
        confidence=confidence,
        method=method
    )


def _estimate_lag(signal: np.ndarray, max_lag: int = 50) -> int:
    """Estimate optimal time delay via first minimum of autocorrelation."""
    n = len(signal)
    max_lag = min(max_lag, n // 4)
    
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first local minimum
    for i in range(1, min(max_lag, len(autocorr) - 1)):
        if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
            return max(1, i)
    
    # Default to lag = 1 if no minimum found
    return 1


def _compute_nolds(signal: np.ndarray, emb_dim: int, lag: int) -> tuple:
    """Compute using nolds library (Rosenstein's algorithm)."""
    try:
        # nolds.lyap_r implements Rosenstein et al. (1993)
        lyap = nolds.lyap_r(signal, emb_dim=emb_dim, lag=lag)
        
        if np.isnan(lyap) or np.isinf(lyap):
            return 0.0, 0.0, "nolds_failed"
        
        # Estimate confidence based on data length relative to embedding
        min_needed = (emb_dim - 1) * lag + 1
        confidence = min(1.0, len(signal) / (min_needed * 10))
        
        return float(lyap), confidence, "nolds_rosenstein"
        
    except Exception as e:
        return 0.0, 0.0, f"nolds_error: {str(e)[:50]}"


def _compute_rosenstein(signal: np.ndarray, emb_dim: int, lag: int) -> tuple:
    """
    Fallback: Manual implementation of Rosenstein's algorithm.
    
    Rosenstein, Collins, De Luca (1993): A practical method for calculating
    largest Lyapunov exponents from small data sets.
    """
    n = len(signal)
    
    # Phase space reconstruction
    n_vectors = n - (emb_dim - 1) * lag
    if n_vectors < 20:
        return 0.0, 0.0, "insufficient_for_embedding"
    
    # Build embedded vectors
    embedded = np.zeros((n_vectors, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = signal[i * lag : i * lag + n_vectors]
    
    # For each point, find nearest neighbor (excluding temporal neighbors)
    min_temporal_sep = emb_dim * lag  # Theiler window
    
    divergence = []
    max_iter = min(n_vectors // 4, 50)  # How far to track divergence
    
    for i in range(n_vectors - max_iter):
        # Find nearest neighbor
        min_dist = np.inf
        nn_idx = -1
        
        for j in range(n_vectors - max_iter):
            if abs(i - j) <= min_temporal_sep:
                continue
            
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < min_dist and dist > 1e-10:
                min_dist = dist
                nn_idx = j
        
        if nn_idx < 0:
            continue
        
        # Track divergence over time
        for k in range(max_iter):
            if i + k >= n_vectors or nn_idx + k >= n_vectors:
                break
            
            dist_k = np.linalg.norm(embedded[i + k] - embedded[nn_idx + k])
            if dist_k > 1e-10:
                if len(divergence) <= k:
                    divergence.append([])
                divergence[k].append(np.log(dist_k))
    
    if len(divergence) < 5:
        return 0.0, 0.0, "insufficient_divergence_data"
    
    # Average divergence at each time step
    mean_divergence = [np.mean(d) for d in divergence if len(d) > 0]
    
    if len(mean_divergence) < 5:
        return 0.0, 0.0, "insufficient_mean_divergence"
    
    # Fit line to get Lyapunov exponent (slope)
    x = np.arange(len(mean_divergence))
    slope, _ = np.polyfit(x, mean_divergence, 1)
    
    confidence = min(1.0, len(divergence[0]) / 50) if divergence[0] else 0.0
    
    return float(slope), confidence, "rosenstein_manual"


def compute_spectrum(signal: np.ndarray, 
                     emb_dim: int = 10,
                     n_exponents: int = 3) -> dict:
    """
    Compute multiple Lyapunov exponents (full spectrum).
    
    More expensive but provides richer stability picture.
    
    Returns
    -------
    dict with:
        - exponents: list of Lyapunov exponents (descending)
        - sum: sum of all exponents (Kaplan-Yorke dimension related)
        - max: largest exponent (same as compute())
    """
    # This requires more sophisticated algorithms (e.g., Wolf's method)
    # For now, return just the largest
    result = compute(signal, emb_dim=emb_dim)
    
    return {
        "exponents": [result.lyapunov_exponent],
        "sum": result.lyapunov_exponent,
        "max": result.lyapunov_exponent,
        "method": result.method,
        "note": "Full spectrum requires Wolf's algorithm (not implemented)"
    }
