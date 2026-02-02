"""
Level 1: Bachelor Typology — Stationarity Testing

Core Question: "Is this signal stationary, or not?"

Methods:
    - ADF (Augmented Dickey-Fuller): H0 = unit root exists (non-stationary)
    - KPSS: H0 = series IS stationary (opposite of ADF)
    - ACF decay: how fast does autocorrelation drop?
    - Variance ratio: does local variance change across the signal?

Decision Table (ADF x KPSS):

    | ADF rejects H0?  | KPSS rejects H0?  | Classification         |
    | (stationary)      | (non-stationary)   |                        |
    |-------------------|--------------------|------------------------|
    | Yes               | No                 | STATIONARY             |
    | No                | Yes                | NON_STATIONARY         |
    | Yes               | Yes                | DIFFERENCE_STATIONARY  |
    | No                | No                 | TREND_STATIONARY       |

    When both tests agree, the answer is clear.
    When they conflict, the label describes the recommended transform.

This is the FIRST GATE. Level 2 (Masters) builds on this result to
classify signal type (periodic, chaotic, random, etc).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
import warnings
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss, acf


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StationarityType(Enum):
    """Stationarity classification from ADF + KPSS joint interpretation."""
    STATIONARY = "stationary"
    TREND_STATIONARY = "trend_stationary"
    DIFFERENCE_STATIONARY = "difference_stationary"
    NON_STATIONARY = "non_stationary"
    INSUFFICIENT_DATA = "insufficient_data"


class Confidence(Enum):
    """How much to trust the classification."""
    HIGH = "high"          # Both tests well inside thresholds
    MEDIUM = "medium"      # At least one test near the boundary
    LOW = "low"            # p-values at lookup-table limits
    UNKNOWN = "unknown"    # Test failed or insufficient data


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    """
    Complete Level 1 result.

    Designed so Level 2 can read everything it needs without re-running
    any tests. All raw values are preserved.
    """

    # --- Classification ---
    stationarity_type: StationarityType
    is_stationary: bool                   # True only when STATIONARY
    confidence: Confidence                # How sure are we?

    # --- ADF ---
    adf_statistic: float
    adf_pvalue: float
    adf_rejects: bool                     # True = rejects unit root = stationary evidence

    # --- KPSS (level, regression='c') ---
    kpss_statistic: float
    kpss_pvalue: float
    kpss_rejects: bool                    # True = rejects stationarity = non-stationary evidence
    kpss_pvalue_at_lower_bound: bool      # p reported as 0.01 (true p may be smaller)
    kpss_pvalue_at_upper_bound: bool      # p reported as 0.10 (true p may be larger)

    # --- KPSS trend (regression='ct'), only when needed ---
    kpss_ct_pvalue: Optional[float]       # None if not run
    kpss_ct_rejects: Optional[bool]       # None if not run

    # --- ACF characterisation ---
    acf_decay_lag: int                    # First lag where |ACF| < 1/e
    acf_half_life: int                    # First lag where |ACF| < 0.5
    acf_decayed: bool                     # FIX L1-02: True if ACF crossed threshold, False if persistent
    acf_values: Optional[np.ndarray]      # Raw ACF array for Level 2

    # --- Variance ratio (segmented) ---
    variance_ratio: float                 # var(second_half) / var(first_half)
    variance_stable: bool                 # True if ratio is in [0.5, 2.0]

    # --- Mean stability (structural break detection) ---
    mean_shift_ratio: float               # |mean(half2) - mean(half1)| / std
    mean_stable: bool                     # True if ratio < 0.5

    # --- Heavy-tail detection ---
    kurtosis: float                       # Excess kurtosis (0 = Gaussian)
    heavy_tailed: bool                    # True if kurtosis > 10

    # --- Seasonality detection ---
    seasonal_period: Optional[int]        # Detected period from ACF peaks, None if no seasonality
    is_seasonal: bool                     # True if clear periodicity in ACF

    # --- Metadata ---
    n_samples: int
    recommendation: str

    # --- For serialisation: drop numpy arrays ---
    def to_dict(self) -> dict:
        """Flat dictionary for parquet/JSON. Drops acf_values array."""
        d = {}
        for k, v in self.__dict__.items():
            if k == "acf_values":
                continue
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# ADF test
# ---------------------------------------------------------------------------

def compute_adf(
    y: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """
    Augmented Dickey-Fuller test.

    H0: Series has a unit root (non-stationary).
    Reject H0 (p < 0.05) => evidence of stationarity.

    Returns (statistic, p_value, rejects_H0).
    """
    try:
        result = adfuller(y, maxlag=max_lag, autolag="AIC")
        stat = float(result[0])
        pval = float(result[1])
        return stat, pval, pval < 0.05
    except Exception:
        return np.nan, np.nan, False


# ---------------------------------------------------------------------------
# KPSS test (with warning capture)
# ---------------------------------------------------------------------------

def compute_kpss(
    y: np.ndarray,
    regression: str = "c",
) -> Tuple[float, float, bool, bool, bool]:
    """
    KPSS test.

    H0: Series IS stationary.
    Reject H0 (p < 0.05) => evidence of non-stationarity.

    Returns (statistic, p_value, rejects_H0,
             p_at_lower_bound, p_at_upper_bound).

    The KPSS lookup table only covers p in [0.01, 0.10].
    When the true p is outside that range, statsmodels clamps
    and issues an InterpolationWarning. We capture that.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = kpss(y, regression=regression, nlags="auto")

        stat = float(result[0])
        pval = float(result[1])

        # Detect clamped p-values from warning messages
        at_lower = False
        at_upper = False
        for w in caught:
            msg = str(w.message)
            if "smaller than the p-value returned" in msg:
                at_lower = True          # true p < 0.01
            elif "greater than the p-value returned" in msg:
                at_upper = True          # true p > 0.10

        return stat, pval, pval < 0.05, at_lower, at_upper
    except Exception:
        return np.nan, np.nan, False, False, False


# ---------------------------------------------------------------------------
# ACF decay
# ---------------------------------------------------------------------------

def compute_acf_decay(
    y: np.ndarray,
    max_lag: int = 100,
    return_values: bool = True,
) -> Tuple[int, int, bool, Optional[np.ndarray]]:
    """
    ACF decay characteristics.

    FIX L1-04: Use n/2 lags for short signals (< 200 samples) to catch
    periodicity that n/4 would miss. Statistical validity is lower but
    informative for Level 2 periodicity detection.

    FIX L1-02: Returns explicit acf_decayed flag. When ACF never drops
    below threshold, acf_decayed=False. This distinguishes "never decayed"
    from "decayed at max lag" for Level 2.

    Returns (decay_lag, half_life, acf_decayed, acf_array).
        decay_lag : first lag where |ACF| < 1/e (~0.368), or nlags if never
        half_life : first lag where |ACF| < 0.5, or nlags if never
        acf_decayed : True if ACF crossed threshold, False if persistent
        acf_array : raw ACF values (None if return_values=False)
    """
    try:
        n = len(y)
        # FIX L1-04: More lags for short signals to detect periodicity
        if n < 200:
            nlags = min(max_lag, n // 2)
        else:
            nlags = min(max_lag, n // 4)

        if nlags < 1:
            return -1, -1, False, None

        acf_vals = acf(y, nlags=nlags, fft=True)

        threshold_e = 1.0 / np.e
        decay_lag = nlags
        half_life = nlags
        found_decay = False
        found_half = False

        for i in range(1, len(acf_vals)):
            if not found_decay and abs(acf_vals[i]) < threshold_e:
                decay_lag = i
                found_decay = True
            if not found_half and abs(acf_vals[i]) < 0.5:
                half_life = i
                found_half = True
            if found_decay and found_half:
                break

        # FIX L1-02: Explicit flag for whether ACF actually decayed
        acf_decayed = found_decay or found_half

        return decay_lag, half_life, acf_decayed, (acf_vals if return_values else None)
    except Exception:
        return -1, -1, False, None


# ---------------------------------------------------------------------------
# Variance ratio (segmented stability check)
# ---------------------------------------------------------------------------

def compute_variance_ratio(y: np.ndarray) -> Tuple[float, bool]:
    """
    Compare variance of first half vs second half.

    A stationary signal should have roughly equal variance in both
    halves.  A ratio far from 1.0 suggests the signal's character
    is changing (even if ADF/KPSS don't catch it).

    Returns (ratio, is_stable).
        ratio = var(second_half) / var(first_half)
        is_stable = True if ratio is in [0.5, 2.0]
    """
    n = len(y)
    if n < 20:
        return np.nan, False

    mid = n // 2
    var1 = np.var(y[:mid])
    var2 = np.var(y[mid:])

    if var1 < 1e-15:
        # First half is essentially constant
        return np.inf if var2 > 1e-15 else 1.0, var2 < 1e-15

    ratio = float(var2 / var1)
    return ratio, 0.5 <= ratio <= 2.0


# ---------------------------------------------------------------------------
# Mean shift (structural break detection)
# ---------------------------------------------------------------------------

def compute_mean_shift(y: np.ndarray) -> Tuple[float, bool]:
    """
    Detect structural breaks in mean level.

    Compares mean of first half vs second half, normalized by overall std.
    A stationary signal should have similar means in both halves.

    Returns (ratio, is_stable).
        ratio = |mean(half2) - mean(half1)| / std
        is_stable = True if ratio < 0.5 (less than half a std shift)
    """
    n = len(y)
    if n < 20:
        return np.nan, False

    mid = n // 2
    mean1 = np.mean(y[:mid])
    mean2 = np.mean(y[mid:])
    std = np.std(y)

    if std < 1e-15:
        # Constant signal
        return 0.0, True

    ratio = float(abs(mean2 - mean1) / std)
    return ratio, ratio < 0.5


# ---------------------------------------------------------------------------
# Kurtosis (heavy-tail detection)
# ---------------------------------------------------------------------------

def compute_kurtosis(y: np.ndarray) -> Tuple[float, bool]:
    """
    Compute excess kurtosis to detect heavy-tailed distributions.

    Excess kurtosis = 0 for Gaussian.
    High kurtosis (> 10) indicates heavy tails that may cause
    issues with standard statistical tests.

    Returns (kurtosis, heavy_tailed).
        kurtosis = excess kurtosis (Fisher definition)
        heavy_tailed = True if kurtosis > 10
    """
    n = len(y)
    if n < 20:
        return np.nan, False

    mean = np.mean(y)
    std = np.std(y)

    if std < 1e-15:
        return 0.0, False

    # Fourth central moment / variance^2 - 3
    m4 = np.mean((y - mean) ** 4)
    kurt = float(m4 / (std ** 4) - 3.0)

    return kurt, kurt > 10.0


# ---------------------------------------------------------------------------
# Seasonality detection
# ---------------------------------------------------------------------------

def compute_seasonality(
    acf_vals: Optional[np.ndarray],
    min_period: int = 2,
) -> Tuple[Optional[int], bool]:
    """
    Detect seasonality from ACF peaks.

    Looks for significant secondary peaks in the ACF beyond lag 0.
    A clear peak indicates periodicity.

    Args:
        acf_vals: ACF array from compute_acf_decay
        min_period: Minimum lag to consider as a period

    Returns (period, is_seasonal).
        period = lag of strongest secondary peak, None if no seasonality
        is_seasonal = True if a clear periodic peak exists
    """
    if acf_vals is None or len(acf_vals) < min_period + 2:
        return None, False

    # Skip lag 0 (always 1.0) and very short lags
    acf_search = acf_vals[min_period:]

    if len(acf_search) < 3:
        return None, False

    # Find local maxima: points higher than both neighbors
    peaks = []
    for i in range(1, len(acf_search) - 1):
        if acf_search[i] > acf_search[i - 1] and acf_search[i] > acf_search[i + 1]:
            # Significance threshold: peak should be above noise floor
            # For n samples, 95% CI for white noise ACF is ~1.96/sqrt(n)
            # We use a simpler threshold: peak > 0.2
            if acf_search[i] > 0.2:
                peaks.append((i + min_period, acf_search[i]))

    if not peaks:
        return None, False

    # Return the strongest peak
    best_peak = max(peaks, key=lambda x: x[1])
    period = best_peak[0]

    # Confirm seasonality: peak should be substantially above the decay envelope
    # A strong seasonal signal will have ACF peak > 0.3
    is_seasonal = best_peak[1] > 0.3

    return period, is_seasonal


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_stationarity(
    adf_rejects: bool,
    kpss_rejects: bool,
) -> StationarityType:
    """
    Joint ADF x KPSS classification.

    | ADF rejects? (stat.) | KPSS rejects? (non-stat.) | Result                |
    |----------------------|---------------------------|-----------------------|
    | Yes                  | No                        | STATIONARY            |
    | No                   | Yes                       | NON_STATIONARY        |
    | Yes                  | Yes                       | DIFFERENCE_STATIONARY |
    | No                   | No                        | TREND_STATIONARY      |
    """
    if adf_rejects and not kpss_rejects:
        return StationarityType.STATIONARY
    elif not adf_rejects and kpss_rejects:
        return StationarityType.NON_STATIONARY
    elif adf_rejects and kpss_rejects:
        return StationarityType.DIFFERENCE_STATIONARY
    else:
        return StationarityType.TREND_STATIONARY


def assess_confidence(
    adf_pvalue: float,
    kpss_pvalue: float,
    kpss_at_lower: bool,
    kpss_at_upper: bool,
    var_ratio: float = 1.0,
    var_stable: bool = True,
) -> Confidence:
    """
    Estimate confidence based on how far p-values are from 0.05.

    FIX L1-01: Now incorporates variance ratio. If the variance ratio
    contradicts the stationarity classification (e.g., HIGH confidence
    stationary but var_ratio=5.0), confidence is downgraded.

    HIGH   : both p-values well away from 0.05, not clamped, variance stable
    MEDIUM : one p-value near the boundary OR one clamped OR variance unstable
    LOW    : both near boundary, both clamped, or extreme variance ratio
    """
    if np.isnan(adf_pvalue) or np.isnan(kpss_pvalue):
        return Confidence.UNKNOWN

    # "Near boundary" = within [0.01, 0.10] of the 0.05 threshold
    adf_clear = adf_pvalue < 0.01 or adf_pvalue > 0.10
    kpss_clear = kpss_pvalue < 0.01 or kpss_pvalue > 0.10

    clamped = kpss_at_lower or kpss_at_upper

    # FIX L1-01: Check for extreme variance ratio that contradicts stationarity
    extreme_var = False
    if not np.isnan(var_ratio):
        extreme_var = var_ratio > 4.0 or var_ratio < 0.25

    # Base confidence from p-values
    if adf_clear and kpss_clear and not clamped:
        base_conf = Confidence.HIGH
    elif adf_clear or kpss_clear:
        base_conf = Confidence.MEDIUM
    else:
        base_conf = Confidence.LOW

    # FIX L1-01: Downgrade confidence if variance contradicts
    if extreme_var:
        # Knock down two levels for extreme variance
        if base_conf == Confidence.HIGH:
            return Confidence.LOW
        elif base_conf == Confidence.MEDIUM:
            return Confidence.LOW
        else:
            return Confidence.LOW
    elif not var_stable:
        # Knock down one level for unstable variance
        if base_conf == Confidence.HIGH:
            return Confidence.MEDIUM
        else:
            return base_conf

    return base_conf


def get_recommendation(stype: StationarityType) -> str:
    """Action recommendation for each classification."""
    return {
        StationarityType.STATIONARY:
            "Signal is stationary. Proceed with standard features.",
        StationarityType.TREND_STATIONARY:
            "Deterministic trend detected. Detrend before features.",
        StationarityType.DIFFERENCE_STATIONARY:
            "Stochastic trend / heteroscedastic. Difference or use rolling windows.",
        StationarityType.NON_STATIONARY:
            "Non-stationary. Apply transformations before analysis.",
        StationarityType.INSUFFICIENT_DATA:
            "Fewer than 20 samples. Cannot test stationarity.",
    }.get(stype, "Unknown classification.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def test_stationarity(
    y: np.ndarray,
    verbose: bool = False,
) -> StationarityResult:
    """
    Level 1 typology: test signal stationarity.

    Pipeline:
        1. ADF  (H0: unit root)
        2. KPSS with regression='c'  (H0: level-stationary)
        3. ACF decay characterisation
        4. Variance-ratio stability check
        5. Mean-shift detection (structural breaks)
        6. Kurtosis (heavy-tail detection)
        7. Seasonality detection from ACF peaks
        8. If both ADF & KPSS say non-stationary, try KPSS 'ct'
        9. Classify and assess confidence

    Args:
        y:  1-D array of signal values.
        verbose:  print results to stdout.

    Returns:
        StationarityResult with everything Level 2 needs.
        Includes diagnostic FLAGS (variance_stable, mean_stable,
        heavy_tailed, is_seasonal) for Level 2 to consume.
    """
    # --- Clean input ---
    y = np.asarray(y, dtype=np.float64).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    # --- Guard: constant signal ---
    if np.ptp(y) < 1e-12:
        # All identical values. Trivially stationary, but ADF/KPSS will choke.
        r = StationarityResult(
            stationarity_type=StationarityType.STATIONARY,
            is_stationary=True,
            confidence=Confidence.HIGH,
            adf_statistic=np.nan,
            adf_pvalue=0.0,
            adf_rejects=True,
            kpss_statistic=np.nan,
            kpss_pvalue=1.0,
            kpss_rejects=False,
            kpss_pvalue_at_lower_bound=False,
            kpss_pvalue_at_upper_bound=False,
            kpss_ct_pvalue=None,
            kpss_ct_rejects=None,
            acf_decay_lag=0,
            acf_half_life=0,
            acf_decayed=True,  # FIX L1-02
            acf_values=None,
            variance_ratio=1.0,
            variance_stable=True,
            mean_shift_ratio=0.0,
            mean_stable=True,
            kurtosis=0.0,
            heavy_tailed=False,
            seasonal_period=None,
            is_seasonal=False,
            n_samples=n,
            recommendation="Constant signal. No variation to analyse.",
        )
        if verbose:
            _print_result(r)
        return r

    # --- Guard: too short ---
    if n < 20:
        r = StationarityResult(
            stationarity_type=StationarityType.INSUFFICIENT_DATA,
            is_stationary=False,
            confidence=Confidence.UNKNOWN,
            adf_statistic=np.nan,
            adf_pvalue=np.nan,
            adf_rejects=False,
            kpss_statistic=np.nan,
            kpss_pvalue=np.nan,
            kpss_rejects=False,
            kpss_pvalue_at_lower_bound=False,
            kpss_pvalue_at_upper_bound=False,
            kpss_ct_pvalue=None,
            kpss_ct_rejects=None,
            acf_decay_lag=-1,
            acf_half_life=-1,
            acf_decayed=False,  # FIX L1-02
            acf_values=None,
            variance_ratio=np.nan,
            variance_stable=False,
            mean_shift_ratio=np.nan,
            mean_stable=False,
            kurtosis=np.nan,
            heavy_tailed=False,
            seasonal_period=None,
            is_seasonal=False,
            n_samples=n,
            recommendation=get_recommendation(StationarityType.INSUFFICIENT_DATA),
        )
        if verbose:
            _print_result(r)
        return r

    # --- 1. ADF ---
    adf_stat, adf_p, adf_rejects = compute_adf(y)

    # --- 2. KPSS (level) ---
    kpss_stat, kpss_p, kpss_rejects, kpss_lo, kpss_hi = compute_kpss(y, "c")

    # --- 3. ACF ---
    acf_decay, acf_half, acf_did_decay, acf_vals = compute_acf_decay(y)

    # --- 4. Variance ratio ---
    var_ratio, var_stable = compute_variance_ratio(y)

    # --- 5. Mean shift ---
    mean_shift, mean_stable = compute_mean_shift(y)

    # --- 6. Kurtosis ---
    kurt, heavy_tail = compute_kurtosis(y)

    # --- 7. Seasonality ---
    season_period, is_season = compute_seasonality(acf_vals)

    # --- 8. Classify ---
    stype = classify_stationarity(adf_rejects, kpss_rejects)

    # 8b. When both say non-stationary, check for trend-stationarity
    kpss_ct_p: Optional[float] = None
    kpss_ct_rejects: Optional[bool] = None

    if stype == StationarityType.NON_STATIONARY:
        _, ct_p, ct_rejects, _, _ = compute_kpss(y, "ct")
        kpss_ct_p = ct_p
        kpss_ct_rejects = ct_rejects
        if not ct_rejects:
            # KPSS(ct) fails to reject => trend-stationary
            stype = StationarityType.TREND_STATIONARY

    # --- 9. Confidence (FIX L1-01: now includes variance ratio) ---
    conf = assess_confidence(adf_p, kpss_p, kpss_lo, kpss_hi, var_ratio, var_stable)

    result = StationarityResult(
        stationarity_type=stype,
        is_stationary=(stype == StationarityType.STATIONARY),
        confidence=conf,
        adf_statistic=adf_stat,
        adf_pvalue=adf_p,
        adf_rejects=adf_rejects,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_p,
        kpss_rejects=kpss_rejects,
        kpss_pvalue_at_lower_bound=kpss_lo,
        kpss_pvalue_at_upper_bound=kpss_hi,
        kpss_ct_pvalue=kpss_ct_p,
        kpss_ct_rejects=kpss_ct_rejects,
        acf_decay_lag=acf_decay,
        acf_half_life=acf_half,
        acf_decayed=acf_did_decay,  # FIX L1-02
        acf_values=acf_vals,
        variance_ratio=var_ratio,
        variance_stable=var_stable,
        mean_shift_ratio=mean_shift,
        mean_stable=mean_stable,
        kurtosis=kurt,
        heavy_tailed=heavy_tail,
        seasonal_period=season_period,
        is_seasonal=is_season,
        n_samples=n,
        recommendation=get_recommendation(stype),
    )

    if verbose:
        _print_result(result)

    return result


# ---------------------------------------------------------------------------
# Benchmark validation
# ---------------------------------------------------------------------------

def validate_level1_benchmarks(verbose: bool = True) -> dict:
    """
    Validate Level 1 against known signals.

    Expected:
        white_noise          -> STATIONARY
        random_walk          -> NON_STATIONARY
        sine_wave            -> STATIONARY  (wide-sense stationary)
        sine_noisy           -> STATIONARY  (realistic periodic signal)
        linear_trend_noise   -> TREND_STATIONARY
        ar1_near_unit        -> STATIONARY  (phi=0.9, inside unit circle)

    FIX L1-03: Note on sine_wave benchmark:
        Pure sine waves are wide-sense stationary (constant mean, constant
        autocovariance). However, ADF is a unit-root test designed for
        stochastic processes, not deterministic oscillations. Its behavior
        on pure sinusoids depends on the frequency/length ratio and lag
        selection. With n=1000 and freq=0.05, ADF reliably rejects, but
        this can be fragile at different parameters. The sine_noisy benchmark
        is more realistic and robust. If sine_wave fails on a different seed,
        it's an edge case that Level 2 periodicity detection handles correctly.
    """
    np.random.seed(42)
    n = 1000

    benchmarks = {
        "white_noise": {
            "signal": np.random.randn(n),
            "expected": StationarityType.STATIONARY,
        },
        "random_walk": {
            "signal": np.cumsum(np.random.randn(n)),
            "expected": StationarityType.NON_STATIONARY,
        },
        "sine_wave": {
            "signal": np.sin(2 * np.pi * 0.05 * np.arange(n)),
            "expected": StationarityType.STATIONARY,
            "note": "FIX L1-03: Edge case - see docstring",
        },
        "sine_noisy": {
            # FIX L1-03: Realistic periodic signal - more robust than pure sine
            "signal": np.sin(2 * np.pi * 0.05 * np.arange(n)) + 0.1 * np.random.randn(n),
            "expected": StationarityType.STATIONARY,
        },
        "linear_trend_noise": {
            "signal": 0.01 * np.arange(n) + np.random.randn(n) * 0.5,
            "expected": StationarityType.TREND_STATIONARY,
        },
        "ar1_near_unit": {
            "signal": _generate_ar1(n, phi=0.9),
            "expected": StationarityType.STATIONARY,
        },
    }

    results = {}
    passed = 0

    if verbose:
        print("=" * 60)
        print("Level 1 Typology — Benchmark Validation")
        print("=" * 60)

    for name, spec in benchmarks.items():
        result = test_stationarity(spec["signal"])
        expected = spec["expected"]
        ok = result.stationarity_type == expected
        if ok:
            passed += 1

        results[name] = {
            "result": result,
            "expected": expected,
            "passed": ok,
        }

        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"\n  {name}:")
            print(f"    expected : {expected.value}")
            print(f"    got      : {result.stationarity_type.value}")
            print(f"    ADF p={result.adf_pvalue:.4f}  KPSS p={result.kpss_pvalue:.4f}")
            print(f"    confidence={result.confidence.value}  var_ratio={result.variance_ratio:.2f}")
            print(f"    [{tag}]")

    if verbose:
        print("\n" + "=" * 60)
        print(f"  {passed}/{len(benchmarks)} passed")
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_ar1(n: int, phi: float = 0.9) -> np.ndarray:
    """AR(1) process:  y[t] = phi * y[t-1] + eps."""
    y = np.zeros(n)
    eps = np.random.randn(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _print_result(r: StationarityResult) -> None:
    """Pretty-print a StationarityResult."""
    print(f"=== Level 1 Stationarity (n={r.n_samples}) ===")
    print(f"  ADF  : stat={r.adf_statistic:+.4f}  p={r.adf_pvalue:.4f}  rejects={r.adf_rejects}")

    kpss_note = ""
    if r.kpss_pvalue_at_lower_bound:
        kpss_note = "  (true p < 0.01)"
    elif r.kpss_pvalue_at_upper_bound:
        kpss_note = "  (true p > 0.10)"
    print(f"  KPSS : stat={r.kpss_statistic:.4f}  p={r.kpss_pvalue:.4f}  rejects={r.kpss_rejects}{kpss_note}")

    if r.kpss_ct_pvalue is not None:
        print(f"  KPSS(ct): p={r.kpss_ct_pvalue:.4f}  rejects={r.kpss_ct_rejects}")

    print(f"  ACF  : decay_lag={r.acf_decay_lag}  half_life={r.acf_half_life}  decayed={r.acf_decayed}  seasonal={r.is_seasonal}", end="")
    if r.seasonal_period is not None:
        print(f" (period={r.seasonal_period})")
    else:
        print()
    print(f"  Var  : ratio={r.variance_ratio:.3f}  stable={r.variance_stable}")
    print(f"  Mean : shift_ratio={r.mean_shift_ratio:.3f}  stable={r.mean_stable}")
    print(f"  Kurt : {r.kurtosis:.2f}  heavy_tailed={r.heavy_tailed}")
    print(f"  => {r.stationarity_type.value}  (confidence={r.confidence.value})")
    print(f"  => {r.recommendation}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validate_level1_benchmarks(verbose=True)
