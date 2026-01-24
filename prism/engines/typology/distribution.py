"""
Distribution Engine
===================

Analyzes the statistical distribution of time series values.

Metrics:
    - skewness: Asymmetry of distribution
    - kurtosis: Tail heaviness (excess kurtosis)
    - is_normal: Normality test result
    - distribution_type: Classified distribution shape

Usage:
    from prism.engines.typology.distribution import compute_distribution
    result = compute_distribution(values)
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


def compute_distribution(
    values: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute distribution metrics for a time series.

    Args:
        values: 1D array of time series values
        alpha: Significance level for normality tests

    Returns:
        Dictionary with distribution metrics
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 8:
        return _empty_result("Insufficient data (need >= 8 points)")

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    median = np.median(values)

    # Skewness
    skewness = stats.skew(values)

    # Kurtosis (excess kurtosis, normal = 0)
    kurtosis = stats.kurtosis(values)

    # Normality tests
    shapiro_stat, shapiro_pvalue = _shapiro_test(values)
    dagostino_stat, dagostino_pvalue = _dagostino_test(values)
    jb_stat, jb_pvalue = _jarque_bera(values, skewness, kurtosis)

    # Combined normality decision
    # Require at least 2 of 3 tests to pass
    tests_passed = sum([
        shapiro_pvalue > alpha,
        dagostino_pvalue > alpha,
        jb_pvalue > alpha
    ])
    is_normal = tests_passed >= 2

    # Distribution type classification
    distribution_type = _classify_distribution(skewness, kurtosis, is_normal)

    # Percentiles
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    iqr = percentiles[4] - percentiles[2]  # Q3 - Q1

    # Outlier detection (1.5 * IQR rule)
    lower_fence = percentiles[2] - 1.5 * iqr
    upper_fence = percentiles[4] + 1.5 * iqr
    n_outliers = np.sum((values < lower_fence) | (values > upper_fence))
    outlier_fraction = n_outliers / n

    # Coefficient of variation
    cv = std / abs(mean) if mean != 0 else np.inf

    # Mode estimation (using KDE)
    mode = _estimate_mode(values)

    return {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "is_normal": bool(is_normal),
        "distribution_type": distribution_type,
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "mode": float(mode),
        "cv": float(cv),
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_pvalue": float(shapiro_pvalue),
        "dagostino_statistic": float(dagostino_stat),
        "dagostino_pvalue": float(dagostino_pvalue),
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "percentile_1": float(percentiles[0]),
        "percentile_5": float(percentiles[1]),
        "percentile_25": float(percentiles[2]),
        "percentile_50": float(percentiles[3]),
        "percentile_75": float(percentiles[4]),
        "percentile_95": float(percentiles[5]),
        "percentile_99": float(percentiles[6]),
        "iqr": float(iqr),
        "n_outliers": int(n_outliers),
        "outlier_fraction": float(outlier_fraction),
        "n_observations": n,
    }


def _shapiro_test(values: np.ndarray) -> tuple:
    """
    Shapiro-Wilk test for normality.

    H0: Data is normally distributed
    Reject H0 if p-value < alpha
    """
    n = len(values)

    # Shapiro-Wilk has sample size limits
    if n > 5000:
        # Use subsample
        idx = np.random.choice(n, 5000, replace=False)
        values = values[idx]

    if n < 3:
        return np.nan, 1.0

    try:
        stat, pvalue = stats.shapiro(values)
        return stat, pvalue
    except Exception:
        return np.nan, 1.0


def _dagostino_test(values: np.ndarray) -> tuple:
    """
    D'Agostino-Pearson test for normality.

    Combines skewness and kurtosis tests.
    """
    n = len(values)

    if n < 20:
        return np.nan, 1.0

    try:
        stat, pvalue = stats.normaltest(values)
        return stat, pvalue
    except Exception:
        return np.nan, 1.0


def _jarque_bera(values: np.ndarray, skewness: float, kurtosis: float) -> tuple:
    """
    Jarque-Bera test for normality.

    Based on skewness and kurtosis.
    """
    n = len(values)

    if n < 8:
        return np.nan, 1.0

    # JB statistic
    jb = (n / 6) * (skewness ** 2 + (kurtosis ** 2) / 4)

    # p-value from chi-square distribution with 2 df
    pvalue = 1 - stats.chi2.cdf(jb, 2)

    return jb, pvalue


def _classify_distribution(skewness: float, kurtosis: float, is_normal: bool) -> str:
    """
    Classify distribution shape based on moments.

    Returns:
        Distribution type string
    """
    if is_normal and abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "normal"

    # Skewness classification
    if abs(skewness) < 0.5:
        skew_type = "symmetric"
    elif skewness > 0:
        skew_type = "right_skewed"
    else:
        skew_type = "left_skewed"

    # Kurtosis classification
    if abs(kurtosis) < 1:
        kurt_type = "mesokurtic"  # Normal-like tails
    elif kurtosis > 1:
        kurt_type = "leptokurtic"  # Heavy tails
    else:
        kurt_type = "platykurtic"  # Light tails

    # Combined classification
    if skew_type == "symmetric":
        if kurt_type == "leptokurtic":
            return "heavy_tailed"
        elif kurt_type == "platykurtic":
            return "light_tailed"
        else:
            return "symmetric"
    elif skew_type == "right_skewed":
        if kurtosis > 3:
            return "exponential_like"
        else:
            return "right_skewed"
    else:  # left_skewed
        return "left_skewed"


def _estimate_mode(values: np.ndarray) -> float:
    """
    Estimate the mode using kernel density estimation.
    """
    try:
        kde = stats.gaussian_kde(values)
        x = np.linspace(np.min(values), np.max(values), 1000)
        density = kde(x)
        mode = x[np.argmax(density)]
        return mode
    except Exception:
        # Fallback to histogram mode
        hist, bin_edges = np.histogram(values, bins='auto')
        max_bin = np.argmax(hist)
        mode = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
        return mode


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return empty result with reason."""
    return {
        "skewness": np.nan,
        "kurtosis": np.nan,
        "is_normal": False,
        "distribution_type": "unknown",
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "mode": np.nan,
        "cv": np.nan,
        "shapiro_statistic": np.nan,
        "shapiro_pvalue": np.nan,
        "dagostino_statistic": np.nan,
        "dagostino_pvalue": np.nan,
        "jarque_bera_statistic": np.nan,
        "jarque_bera_pvalue": np.nan,
        "percentile_1": np.nan,
        "percentile_5": np.nan,
        "percentile_25": np.nan,
        "percentile_50": np.nan,
        "percentile_75": np.nan,
        "percentile_95": np.nan,
        "percentile_99": np.nan,
        "iqr": np.nan,
        "n_outliers": 0,
        "outlier_fraction": 0.0,
        "n_observations": 0,
        "error": reason,
    }


def classify_tail_behavior(result: Dict[str, Any]) -> str:
    """
    Classify tail behavior for risk assessment.

    Returns:
        'normal_tails': Standard Gaussian-like
        'heavy_tails': Higher probability of extreme events
        'very_heavy_tails': Fat tails, extreme outlier risk
        'light_tails': Lower probability of extremes than normal
    """
    kurtosis = result.get("kurtosis", 0)
    outlier_fraction = result.get("outlier_fraction", 0)

    if kurtosis > 6 or outlier_fraction > 0.1:
        return "very_heavy_tails"
    elif kurtosis > 1:
        return "heavy_tails"
    elif kurtosis < -0.5:
        return "light_tails"
    else:
        return "normal_tails"


def fit_distribution(
    values: np.ndarray,
    candidates: Optional[list] = None
) -> Dict[str, Any]:
    """
    Fit candidate distributions and find best match.

    Args:
        values: 1D array of values
        candidates: List of distribution names to try

    Returns:
        Dictionary with best fit distribution and parameters
    """
    if candidates is None:
        candidates = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min', 't']

    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    if len(values) < 20:
        return {"error": "Insufficient data for distribution fitting"}

    results = []

    for dist_name in candidates:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(values)

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(values, dist_name, params)

            # Log-likelihood
            log_likelihood = np.sum(dist.logpdf(values, *params))

            # AIC
            k = len(params)
            n = len(values)
            aic = 2 * k - 2 * log_likelihood

            results.append({
                "distribution": dist_name,
                "params": params,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "log_likelihood": float(log_likelihood),
                "aic": float(aic),
            })
        except Exception:
            continue

    if not results:
        return {"error": "No distributions could be fitted"}

    # Sort by AIC (lower is better)
    results.sort(key=lambda x: x["aic"])

    return {
        "best_fit": results[0],
        "all_fits": results,
    }
