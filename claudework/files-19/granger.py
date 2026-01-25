#!/usr/bin/env python3
"""
Granger Causality Engine
========================

Computes coupling strength between signals via Granger causality.

Granger causality tests whether past values of X help predict Y
beyond Y's own past. It measures information flow / coupling.

For geometry windows, we use the correlation matrix as a proxy
since individual signals aren't available at this layer.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class GrangerResult:
    """Result of Granger causality computation for a signal pair."""
    x_causes_y: bool          # Does X Granger-cause Y?
    y_causes_x: bool          # Does Y Granger-cause X?
    x_to_y_pvalue: float      # P-value for X → Y
    y_to_x_pvalue: float      # P-value for Y → X
    x_to_y_fstat: float       # F-statistic for X → Y
    y_to_x_fstat: float       # F-statistic for Y → X
    bidirectional: bool       # Both directions significant?
    net_direction: str        # "x_to_y" | "y_to_x" | "bidirectional" | "none"
    optimal_lag: int          # Lag with strongest causality
    confidence: float


@dataclass
class CouplingResult:
    """Aggregate coupling metric for a set of signals."""
    coupling: float           # Overall coupling strength [0, 1]
    n_pairs: int              # Number of pairs tested
    n_causal: int             # Number with significant causality
    n_bidirectional: int      # Number with bidirectional causality
    mean_fstat: float         # Average F-statistic
    causal_density: float     # Fraction of pairs with causality
    method: str
    details: Optional[List[Dict]] = None


def compute_pairwise(x: np.ndarray, 
                     y: np.ndarray,
                     max_lag: int = 5,
                     significance: float = 0.05) -> GrangerResult:
    """
    Compute Granger causality between two signals.
    
    Parameters
    ----------
    x, y : np.ndarray
        Time series data (same length)
    max_lag : int
        Maximum lag to test
    significance : float
        P-value threshold for significance
        
    Returns
    -------
    GrangerResult
        Bidirectional causality results
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Validate
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
    
    if len(x) < max_lag * 3:
        return GrangerResult(
            x_causes_y=False,
            y_causes_x=False,
            x_to_y_pvalue=1.0,
            y_to_x_pvalue=1.0,
            x_to_y_fstat=0.0,
            y_to_x_fstat=0.0,
            bidirectional=False,
            net_direction="none",
            optimal_lag=1,
            confidence=0.0
        )
    
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < max_lag * 3:
        return GrangerResult(
            x_causes_y=False,
            y_causes_x=False,
            x_to_y_pvalue=1.0,
            y_to_x_pvalue=1.0,
            x_to_y_fstat=0.0,
            y_to_x_fstat=0.0,
            bidirectional=False,
            net_direction="none",
            optimal_lag=1,
            confidence=0.0
        )
    
    if STATSMODELS_AVAILABLE:
        return _compute_statsmodels(x, y, max_lag, significance)
    else:
        return _compute_manual(x, y, max_lag, significance)


def _compute_statsmodels(x: np.ndarray, y: np.ndarray, 
                         max_lag: int, significance: float) -> GrangerResult:
    """Compute using statsmodels."""
    try:
        # Stack for statsmodels format
        data_xy = np.column_stack([y, x])  # Test X → Y
        data_yx = np.column_stack([x, y])  # Test Y → X
        
        # Test X → Y
        result_xy = grangercausalitytests(data_xy, maxlag=max_lag, verbose=False)
        
        # Test Y → X
        result_yx = grangercausalitytests(data_yx, maxlag=max_lag, verbose=False)
        
        # Find best lag (lowest p-value) for each direction
        best_xy_pval = 1.0
        best_xy_fstat = 0.0
        best_xy_lag = 1
        
        for lag in range(1, max_lag + 1):
            pval = result_xy[lag][0]['ssr_ftest'][1]
            fstat = result_xy[lag][0]['ssr_ftest'][0]
            if pval < best_xy_pval:
                best_xy_pval = pval
                best_xy_fstat = fstat
                best_xy_lag = lag
        
        best_yx_pval = 1.0
        best_yx_fstat = 0.0
        
        for lag in range(1, max_lag + 1):
            pval = result_yx[lag][0]['ssr_ftest'][1]
            fstat = result_yx[lag][0]['ssr_ftest'][0]
            if pval < best_yx_pval:
                best_yx_pval = pval
                best_yx_fstat = fstat
        
        x_causes_y = best_xy_pval < significance
        y_causes_x = best_yx_pval < significance
        
        # Determine net direction
        if x_causes_y and y_causes_x:
            net_direction = "bidirectional"
        elif x_causes_y:
            net_direction = "x_to_y"
        elif y_causes_x:
            net_direction = "y_to_x"
        else:
            net_direction = "none"
        
        # Confidence based on strength of result
        confidence = 1.0 - min(best_xy_pval, best_yx_pval)
        
        return GrangerResult(
            x_causes_y=x_causes_y,
            y_causes_x=y_causes_x,
            x_to_y_pvalue=float(best_xy_pval),
            y_to_x_pvalue=float(best_yx_pval),
            x_to_y_fstat=float(best_xy_fstat),
            y_to_x_fstat=float(best_yx_fstat),
            bidirectional=x_causes_y and y_causes_x,
            net_direction=net_direction,
            optimal_lag=best_xy_lag,
            confidence=float(confidence)
        )
        
    except Exception as e:
        return GrangerResult(
            x_causes_y=False,
            y_causes_x=False,
            x_to_y_pvalue=1.0,
            y_to_x_pvalue=1.0,
            x_to_y_fstat=0.0,
            y_to_x_fstat=0.0,
            bidirectional=False,
            net_direction="none",
            optimal_lag=1,
            confidence=0.0
        )


def _compute_manual(x: np.ndarray, y: np.ndarray,
                    max_lag: int, significance: float) -> GrangerResult:
    """
    Manual Granger causality computation.
    
    Compares two models:
    - Restricted: y_t = a0 + a1*y_{t-1} + ... + an*y_{t-n}
    - Unrestricted: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ...
    
    F-test on whether x lags improve prediction.
    """
    from scipy import stats
    
    n = len(x)
    
    best_xy_pval = 1.0
    best_xy_fstat = 0.0
    best_lag = 1
    
    for lag in range(1, max_lag + 1):
        if n - lag < 2 * lag + 5:
            continue
        
        # Build lagged matrices
        Y = y[lag:]  # Target
        
        # Restricted model: only Y lags
        X_restricted = np.column_stack([
            y[lag - i - 1:n - i - 1] for i in range(lag)
        ])
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
        
        # Unrestricted model: Y lags + X lags
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag - i - 1:n - i - 1] for i in range(lag)]
        ])
        
        try:
            # Fit restricted
            beta_r, residuals_r, rank_r, _ = np.linalg.lstsq(X_restricted, Y, rcond=None)
            if len(residuals_r) == 0:
                RSS_r = np.sum((Y - X_restricted @ beta_r) ** 2)
            else:
                RSS_r = residuals_r[0]
            
            # Fit unrestricted
            beta_u, residuals_u, rank_u, _ = np.linalg.lstsq(X_unrestricted, Y, rcond=None)
            if len(residuals_u) == 0:
                RSS_u = np.sum((Y - X_unrestricted @ beta_u) ** 2)
            else:
                RSS_u = residuals_u[0]
            
            # F-statistic
            df1 = lag  # Number of restrictions
            df2 = len(Y) - 2 * lag - 1  # Degrees of freedom
            
            if RSS_u > 0 and df2 > 0:
                F = ((RSS_r - RSS_u) / df1) / (RSS_u / df2)
                pval = 1 - stats.f.cdf(F, df1, df2)
                
                if pval < best_xy_pval:
                    best_xy_pval = pval
                    best_xy_fstat = F
                    best_lag = lag
                    
        except Exception:
            continue
    
    # Repeat for Y → X
    best_yx_pval = 1.0
    best_yx_fstat = 0.0
    
    for lag in range(1, max_lag + 1):
        if n - lag < 2 * lag + 5:
            continue
        
        Y = x[lag:]
        
        X_restricted = np.column_stack([
            x[lag - i - 1:n - i - 1] for i in range(lag)
        ])
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
        
        X_unrestricted = np.column_stack([
            X_restricted,
            *[y[lag - i - 1:n - i - 1] for i in range(lag)]
        ])
        
        try:
            beta_r, residuals_r, _, _ = np.linalg.lstsq(X_restricted, Y, rcond=None)
            RSS_r = residuals_r[0] if len(residuals_r) > 0 else np.sum((Y - X_restricted @ beta_r) ** 2)
            
            beta_u, residuals_u, _, _ = np.linalg.lstsq(X_unrestricted, Y, rcond=None)
            RSS_u = residuals_u[0] if len(residuals_u) > 0 else np.sum((Y - X_unrestricted @ beta_u) ** 2)
            
            df1 = lag
            df2 = len(Y) - 2 * lag - 1
            
            if RSS_u > 0 and df2 > 0:
                F = ((RSS_r - RSS_u) / df1) / (RSS_u / df2)
                pval = 1 - stats.f.cdf(F, df1, df2)
                
                if pval < best_yx_pval:
                    best_yx_pval = pval
                    best_yx_fstat = F
                    
        except Exception:
            continue
    
    x_causes_y = best_xy_pval < significance
    y_causes_x = best_yx_pval < significance
    
    if x_causes_y and y_causes_x:
        net_direction = "bidirectional"
    elif x_causes_y:
        net_direction = "x_to_y"
    elif y_causes_x:
        net_direction = "y_to_x"
    else:
        net_direction = "none"
    
    return GrangerResult(
        x_causes_y=x_causes_y,
        y_causes_x=y_causes_x,
        x_to_y_pvalue=float(best_xy_pval),
        y_to_x_pvalue=float(best_yx_pval),
        x_to_y_fstat=float(best_xy_fstat),
        y_to_x_fstat=float(best_yx_fstat),
        bidirectional=x_causes_y and y_causes_x,
        net_direction=net_direction,
        optimal_lag=best_lag,
        confidence=float(1.0 - min(best_xy_pval, best_yx_pval))
    )


def compute_from_geometry(geometry: Dict,
                          n_signals: Optional[int] = None) -> CouplingResult:
    """
    Compute coupling from geometry window data.
    
    Since we don't have raw signals at this layer, we use geometry
    metrics as proxies for coupling:
    - mean_correlation: direct measure of linear coupling
    - network_density: fraction of significant connections
    - n_causal_pairs: from Granger tests in geometry (if available)
    
    Parameters
    ----------
    geometry : Dict
        Geometry window data with correlation/network metrics
    n_signals : int, optional
        Number of signals (for normalizing)
        
    Returns
    -------
    CouplingResult
        Aggregate coupling metric
    """
    mean_corr = abs(geometry.get("mean_correlation", 0.5))
    density = geometry.get("network_density", 0.5)
    n_causal = geometry.get("n_causal_pairs", 0)
    n_bidir = geometry.get("n_bidirectional", 0)
    n_sig = geometry.get("n_signals", n_signals or 1)
    
    # Max possible pairs
    n_pairs = n_sig * (n_sig - 1) // 2 if n_sig > 1 else 1
    
    # Causal density
    causal_density = n_causal / n_pairs if n_pairs > 0 else 0
    
    # Combine into coupling metric
    # Weight: 50% correlation, 30% density, 20% causal
    coupling = 0.5 * mean_corr + 0.3 * density + 0.2 * causal_density
    coupling = float(np.clip(coupling, 0, 1))
    
    return CouplingResult(
        coupling=coupling,
        n_pairs=n_pairs,
        n_causal=n_causal,
        n_bidirectional=n_bidir,
        mean_fstat=0.0,  # Not available from geometry
        causal_density=causal_density,
        method="geometry_proxy"
    )


def compute_network(signals: Dict[str, np.ndarray],
                    max_lag: int = 5,
                    significance: float = 0.05) -> CouplingResult:
    """
    Compute coupling across a network of signals.
    
    Parameters
    ----------
    signals : Dict[str, np.ndarray]
        Dictionary of signal_id -> time series
    max_lag : int
        Maximum lag for Granger tests
    significance : float
        P-value threshold
        
    Returns
    -------
    CouplingResult
        Network-level coupling metrics
    """
    signal_ids = list(signals.keys())
    n_signals = len(signal_ids)
    
    if n_signals < 2:
        return CouplingResult(
            coupling=0.0,
            n_pairs=0,
            n_causal=0,
            n_bidirectional=0,
            mean_fstat=0.0,
            causal_density=0.0,
            method="insufficient_signals"
        )
    
    results = []
    n_causal = 0
    n_bidirectional = 0
    fstats = []
    
    # Test all pairs
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            x = signals[signal_ids[i]]
            y = signals[signal_ids[j]]
            
            result = compute_pairwise(x, y, max_lag, significance)
            
            if result.x_causes_y or result.y_causes_x:
                n_causal += 1
            if result.bidirectional:
                n_bidirectional += 1
            
            fstats.append(max(result.x_to_y_fstat, result.y_to_x_fstat))
            
            results.append({
                "signal_x": signal_ids[i],
                "signal_y": signal_ids[j],
                "x_causes_y": result.x_causes_y,
                "y_causes_x": result.y_causes_x,
                "net_direction": result.net_direction
            })
    
    n_pairs = len(results)
    causal_density = n_causal / n_pairs if n_pairs > 0 else 0
    mean_fstat = np.mean(fstats) if fstats else 0
    
    # Coupling = causal density (normalized)
    coupling = float(causal_density)
    
    return CouplingResult(
        coupling=coupling,
        n_pairs=n_pairs,
        n_causal=n_causal,
        n_bidirectional=n_bidirectional,
        mean_fstat=float(mean_fstat),
        causal_density=causal_density,
        method="granger_network",
        details=results
    )
