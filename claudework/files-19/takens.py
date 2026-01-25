#!/usr/bin/env python3
"""
Takens Embedding Engine
=======================

Computes trajectory classification via phase space reconstruction.

Takens' embedding theorem (1981): A time series can reconstruct the
attractor of the underlying dynamical system using delay embedding.

Trajectory classification:
- converging: velocity toward attractor center
- diverging: velocity away from center
- periodic: velocity tangent to closed orbit
- chaotic: velocity direction varies unpredictably
- stationary: near-zero velocity
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class TakensResult:
    """Result of Takens embedding and trajectory analysis."""
    trajectory: str           # converging | diverging | periodic | chaotic | stationary
    
    # Phase space metrics
    embedding_dim: int        # m: embedding dimension used
    delay: int                # τ: time delay used
    
    # Velocity analysis
    mean_velocity: float      # Average speed in phase space
    velocity_variance: float  # Variance of velocity magnitude
    radial_velocity: float    # Component toward/away from center (-1 to 1)
    tangential_velocity: float  # Component perpendicular to radial
    
    # Direction consistency
    direction_entropy: float  # How random is the direction? (0=consistent, 1=random)
    
    confidence: float
    method: str


def compute(signal: np.ndarray,
            emb_dim: Optional[int] = None,
            delay: Optional[int] = None,
            min_samples: int = 100) -> TakensResult:
    """
    Compute trajectory classification via Takens embedding.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series data
    emb_dim : int, optional
        Embedding dimension (auto-computed if None)
    delay : int, optional
        Time delay (auto-computed if None)
    min_samples : int
        Minimum samples required
        
    Returns
    -------
    TakensResult
        Trajectory classification and phase space metrics
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)
    
    # Validate
    if n < min_samples:
        return _empty_result("insufficient_data")
    
    # Remove NaN/Inf
    signal = signal[np.isfinite(signal)]
    if len(signal) < min_samples:
        return _empty_result("insufficient_valid_data")
    
    # Check for constant signal
    std = np.std(signal)
    if std < 1e-10:
        return TakensResult(
            trajectory="stationary",
            embedding_dim=1,
            delay=1,
            mean_velocity=0.0,
            velocity_variance=0.0,
            radial_velocity=0.0,
            tangential_velocity=0.0,
            direction_entropy=0.0,
            confidence=1.0,
            method="constant_signal"
        )
    
    # Standardize
    signal = (signal - np.mean(signal)) / std
    
    # Auto-compute embedding parameters if not provided
    if delay is None:
        delay = _estimate_delay(signal)
    
    if emb_dim is None:
        emb_dim = _estimate_embedding_dim(signal, delay)
    
    # Perform embedding
    embedded = _embed(signal, emb_dim, delay)
    
    if embedded is None or len(embedded) < 20:
        return _empty_result("embedding_failed")
    
    # Analyze trajectory
    return _analyze_trajectory(embedded, emb_dim, delay)


def _empty_result(method: str) -> TakensResult:
    """Return empty result for error cases."""
    return TakensResult(
        trajectory="stationary",
        embedding_dim=0,
        delay=0,
        mean_velocity=0.0,
        velocity_variance=0.0,
        radial_velocity=0.0,
        tangential_velocity=0.0,
        direction_entropy=0.5,
        confidence=0.0,
        method=method
    )


def _estimate_delay(signal: np.ndarray, max_delay: int = 50) -> int:
    """
    Estimate optimal time delay using first minimum of autocorrelation.
    
    Alternative: first minimum of mutual information (more robust but slower).
    """
    n = len(signal)
    max_delay = min(max_delay, n // 4)
    
    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first local minimum
    for i in range(1, min(max_delay, len(autocorr) - 1)):
        if autocorr[i] < autocorr[i-1] and autocorr[i] <= autocorr[i+1]:
            return max(1, i)
    
    # If no minimum, use first zero crossing or default
    for i in range(1, min(max_delay, len(autocorr))):
        if autocorr[i] <= 0:
            return max(1, i)
    
    return max(1, max_delay // 4)


def _estimate_embedding_dim(signal: np.ndarray, delay: int, max_dim: int = 15) -> int:
    """
    Estimate embedding dimension using False Nearest Neighbors (FNN).
    
    FNN: If nearest neighbors in m-dim remain neighbors in (m+1)-dim,
    then m is sufficient.
    """
    n = len(signal)
    threshold = 15.0  # Kennel et al. recommend 10-15
    
    for m in range(1, max_dim + 1):
        n_vectors = n - (m) * delay
        if n_vectors < 20:
            return max(2, m - 1)
        
        # Embed in m dimensions
        embedded_m = _embed(signal, m, delay)
        # Embed in m+1 dimensions
        embedded_m1 = _embed(signal, m + 1, delay)
        
        if embedded_m is None or embedded_m1 is None:
            return max(2, m)
        
        # Truncate to same length
        min_len = min(len(embedded_m), len(embedded_m1))
        embedded_m = embedded_m[:min_len]
        embedded_m1 = embedded_m1[:min_len]
        
        # Count false nearest neighbors
        n_fnn = 0
        n_total = 0
        
        # Sample points for efficiency
        sample_size = min(100, min_len)
        indices = np.random.choice(min_len, sample_size, replace=False)
        
        for i in indices:
            # Find nearest neighbor in m-dim (excluding self)
            distances_m = np.linalg.norm(embedded_m - embedded_m[i], axis=1)
            distances_m[i] = np.inf  # Exclude self
            nn_idx = np.argmin(distances_m)
            
            d_m = distances_m[nn_idx]
            if d_m < 1e-10:
                continue
            
            # Check distance in m+1 dim
            d_m1 = np.linalg.norm(embedded_m1[i] - embedded_m1[nn_idx])
            
            # FNN criterion
            if abs(d_m1 - d_m) / d_m > threshold:
                n_fnn += 1
            
            n_total += 1
        
        # If FNN fraction is low, m is sufficient
        if n_total > 0 and n_fnn / n_total < 0.1:
            return max(2, m)
    
    return max_dim


def _embed(signal: np.ndarray, m: int, tau: int) -> Optional[np.ndarray]:
    """
    Create time-delay embedding of signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series
    m : int
        Embedding dimension
    tau : int
        Time delay
        
    Returns
    -------
    np.ndarray of shape (n_vectors, m)
    """
    n = len(signal)
    n_vectors = n - (m - 1) * tau
    
    if n_vectors < 1:
        return None
    
    embedded = np.zeros((n_vectors, m))
    for i in range(m):
        embedded[:, i] = signal[i * tau : i * tau + n_vectors]
    
    return embedded


def _analyze_trajectory(embedded: np.ndarray, emb_dim: int, delay: int) -> TakensResult:
    """
    Analyze trajectory in embedded phase space.
    """
    n = len(embedded)
    
    # Compute velocities (finite differences)
    velocity = np.diff(embedded, axis=0)
    velocity_magnitudes = np.linalg.norm(velocity, axis=1)
    
    mean_velocity = np.mean(velocity_magnitudes)
    velocity_variance = np.var(velocity_magnitudes)
    
    # Compute center of trajectory
    center = np.mean(embedded, axis=0)
    
    # Radial vectors (from center to each point)
    radial = embedded[:-1] - center  # Exclude last point (no velocity there)
    radial_norms = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norms = np.where(radial_norms < 1e-10, 1e-10, radial_norms)
    radial_unit = radial / radial_norms
    
    # Radial velocity component (positive = moving away from center)
    radial_velocity_components = np.sum(velocity * radial_unit, axis=1)
    mean_radial_velocity = np.mean(radial_velocity_components)
    
    # Tangential velocity (perpendicular to radial)
    tangential = velocity - radial_velocity_components[:, np.newaxis] * radial_unit
    tangential_magnitudes = np.linalg.norm(tangential, axis=1)
    mean_tangential_velocity = np.mean(tangential_magnitudes)
    
    # Direction entropy (how consistent is the direction?)
    direction_entropy = _compute_direction_entropy(velocity)
    
    # Classify trajectory
    trajectory = _classify_trajectory(
        mean_velocity,
        velocity_variance,
        mean_radial_velocity,
        mean_tangential_velocity,
        direction_entropy
    )
    
    # Normalize radial velocity to [-1, 1]
    max_radial = max(abs(mean_radial_velocity), 1e-10)
    normalized_radial = mean_radial_velocity / (mean_velocity + 1e-10)
    normalized_radial = np.clip(normalized_radial, -1, 1)
    
    return TakensResult(
        trajectory=trajectory,
        embedding_dim=emb_dim,
        delay=delay,
        mean_velocity=float(mean_velocity),
        velocity_variance=float(velocity_variance),
        radial_velocity=float(normalized_radial),
        tangential_velocity=float(mean_tangential_velocity),
        direction_entropy=float(direction_entropy),
        confidence=0.8,
        method="takens"
    )


def _compute_direction_entropy(velocity: np.ndarray) -> float:
    """
    Compute entropy of velocity directions.
    
    Discretize directions into bins and compute Shannon entropy.
    """
    n = len(velocity)
    if n < 5:
        return 0.5
    
    # Use first two principal components for direction
    # (handles high-dimensional embeddings)
    if velocity.shape[1] > 2:
        # Simple PCA
        centered = velocity - np.mean(velocity, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Take top 2 components
        idx = np.argsort(eigenvalues)[::-1][:2]
        velocity_2d = centered @ eigenvectors[:, idx]
    else:
        velocity_2d = velocity[:, :2] if velocity.shape[1] >= 2 else velocity
    
    # Compute angles
    angles = np.arctan2(velocity_2d[:, 1] if velocity_2d.shape[1] > 1 else np.zeros(n),
                        velocity_2d[:, 0])
    
    # Discretize into bins (8 directions)
    n_bins = 8
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bins)
    
    # Shannon entropy
    probs = hist / n
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    entropy = -np.sum(probs * np.log(probs))
    
    # Normalize by max entropy
    max_entropy = np.log(n_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(normalized_entropy)


def _classify_trajectory(mean_velocity: float,
                         velocity_variance: float,
                         radial_velocity: float,
                         tangential_velocity: float,
                         direction_entropy: float) -> str:
    """
    Classify trajectory based on phase space velocity analysis.
    """
    # Stationary: very low velocity
    if mean_velocity < 0.05:
        return "stationary"
    
    # Chaotic: high direction entropy and high velocity variance
    if direction_entropy > 0.7 and velocity_variance > 0.1:
        return "chaotic"
    
    # Periodic: consistent tangential motion, low radial motion
    if abs(radial_velocity) < 0.2 * mean_velocity and tangential_velocity > 0.5 * mean_velocity:
        if direction_entropy < 0.5:
            return "periodic"
    
    # Converging: negative radial velocity (toward center)
    if radial_velocity < -0.1:
        return "converging"
    
    # Diverging: positive radial velocity (away from center)
    if radial_velocity > 0.1:
        return "diverging"
    
    # Default based on entropy
    if direction_entropy > 0.6:
        return "chaotic"
    elif direction_entropy < 0.3:
        return "periodic"
    else:
        return "stationary"


def compute_from_geometry(geometry_history: list, 
                          window_idx: int) -> TakensResult:
    """
    Estimate trajectory from geometry window evolution.
    
    When raw signal isn't available, use geometry metric trends.
    
    Parameters
    ----------
    geometry_history : list
        List of geometry dicts over time
    window_idx : int
        Current window index
        
    Returns
    -------
    TakensResult
        Estimated trajectory from geometry trends
    """
    if window_idx < 2:
        return TakensResult(
            trajectory="stationary",
            embedding_dim=0,
            delay=0,
            mean_velocity=0.0,
            velocity_variance=0.0,
            radial_velocity=0.0,
            tangential_velocity=0.0,
            direction_entropy=0.5,
            confidence=0.3,
            method="geometry_proxy_insufficient"
        )
    
    # Look at recent trend (last 3-5 windows)
    lookback = min(5, window_idx + 1)
    recent = geometry_history[window_idx - lookback + 1:window_idx + 1]
    
    if len(recent) < 2:
        return _empty_result("geometry_proxy_insufficient")
    
    # Extract correlation trend as proxy for trajectory
    correlations = [g.get("mean_correlation", 0) for g in recent]
    densities = [g.get("network_density", 0) for g in recent]
    
    # Compute slopes
    x = np.arange(len(correlations))
    
    if len(correlations) > 1:
        corr_slope = np.polyfit(x, correlations, 1)[0]
        density_slope = np.polyfit(x, densities, 1)[0]
    else:
        corr_slope = 0
        density_slope = 0
    
    # Variance = proxy for chaos
    corr_var = np.var(correlations) if len(correlations) > 1 else 0
    
    # Map to trajectory metrics
    mean_velocity = abs(corr_slope) + abs(density_slope)
    velocity_variance = corr_var
    radial_velocity = corr_slope  # Positive = strengthening = diverging from equilibrium
    
    # Classify
    if corr_var > 0.05:
        trajectory = "chaotic"
    elif corr_slope > 0.02 and density_slope > 0.02:
        trajectory = "converging"
    elif corr_slope < -0.02 and density_slope < -0.02:
        trajectory = "diverging"
    elif abs(corr_slope) < 0.01 and corr_var < 0.01:
        # Check for periodicity
        if _detect_periodicity_simple(correlations):
            trajectory = "periodic"
        else:
            trajectory = "stationary"
    else:
        trajectory = "stationary"
    
    return TakensResult(
        trajectory=trajectory,
        embedding_dim=0,
        delay=0,
        mean_velocity=float(mean_velocity),
        velocity_variance=float(velocity_variance),
        radial_velocity=float(radial_velocity),
        tangential_velocity=0.0,
        direction_entropy=0.5,
        confidence=0.4,
        method="geometry_proxy"
    )


def _detect_periodicity_simple(values: list, threshold: float = 0.7) -> bool:
    """Simple periodicity detection via autocorrelation."""
    if len(values) < 6:
        return False
    
    values = np.array(values)
    values = values - np.mean(values)
    
    # Autocorrelation at lag 2-3
    for lag in [2, 3]:
        if len(values) > lag:
            autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(autocorr) and abs(autocorr) > threshold:
                return True
    return False
