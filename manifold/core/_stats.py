"""Inline basic statistics — too simple to warrant a pmtvs dependency."""

from scipy.stats import kurtosis as _scipy_kurtosis, skew as _scipy_skew
import numpy as np


def kurtosis(y, fisher=True):
    """Kurtosis. fisher=True (default) returns excess kurtosis."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 4:
        return np.nan
    return float(_scipy_kurtosis(y, fisher=fisher, nan_policy="omit"))


def skewness(y):
    """Sample skewness."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return np.nan
    return float(_scipy_skew(y, nan_policy="omit"))


def rms(y):
    """Root mean square."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan
    return float(np.sqrt(np.mean(y**2)))


def crest_factor(y):
    """Peak-to-RMS ratio."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan
    r = rms(y)
    if r < 1e-15:
        return np.nan
    return float(np.max(np.abs(y)) / r)


def zero_crossings(y):
    """Count of zero crossings."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(y)) != 0))


def mean_crossings(y):
    """Count of mean crossings."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 0
    centered = y - np.mean(y)
    return int(np.sum(np.diff(np.sign(centered)) != 0))


def peak_to_peak(y):
    """Peak-to-peak amplitude."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan
    return float(np.max(y) - np.min(y))
