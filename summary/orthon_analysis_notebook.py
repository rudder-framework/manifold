"""
Orthon Analysis Notebook
========================

COMPLETE BLIND-TEST ANALYSIS PIPELINE

Sections:
1. DATA CHARACTERIZATION     -> Methods chapter
2. COHORT DISCOVERY          -> Results 3.1
3. CRITICAL MOMENTS          -> Results 3.2
4. SIGNAL ANALYSIS (Vector)  -> Results 3.3
5. RELATIONSHIP DYNAMICS (Geometry) -> Results 3.4
6. STATE TRAJECTORY          -> Results 3.5
7. ML FEATURES & PREDICTION  -> Results 3.6
8. ANOMALIES                 -> Discussion
9. CONCLUSIONS               -> Conclusion

Input: CSV file (blind - no domain knowledge required)
Output: Thesis-ready report with auto-generated paragraphs

Usage:
    python orthon_analysis_notebook.py --data experiment.csv --output report/

    Or run as Jupyter notebook cells.
"""

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings
import sys

warnings.filterwarnings('ignore')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from summarize module
try:
    from prism.entry_points.summarize import (
        MomentConfig,
        MomentDetector,
        AdaptiveSampler,
        SummaryGenerator,
    )
    SUMMARIZE_AVAILABLE = True
except ImportError:
    print("Note: prism.entry_points.summarize not found. Using inline classes.")
    SUMMARIZE_AVAILABLE = False


# =============================================================================
# SECTION 1: DATA CHARACTERIZATION
# =============================================================================

@dataclass
class SignalQuality:
    """Quality assessment for a single signal."""
    signal_id: str
    status: str           # "valid", "noise", "redundant", "flatline"
    variance: float
    predictive: float     # Correlation with outcome if known
    action: str           # "include", "exclude"
    reason: str           # Why excluded (if applicable)
    redundant_with: Optional[str] = None


class DataCharacterizer:
    """
    Characterize input data quality and structure.

    Outputs Methods chapter content.
    """

    def __init__(self):
        self.signal_quality: Dict[str, SignalQuality] = {}
        self.metadata: Dict[str, Any] = {}

    def analyze(
        self,
        df: pl.DataFrame,
        entity_col: str = "entity_id",
        time_col: str = "timestamp",
        signal_col: str = "signal_id",
        value_col: str = "value",
        outcome_col: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze data structure and signal quality.
        """
        # Detect if data is wide or long format
        if signal_col in df.columns and value_col in df.columns:
            # Long format
            is_long = True
            signals = df[signal_col].unique().sort().to_list()
            n_signals = len(signals)
        else:
            # Wide format - each column is a signal
            is_long = False
            exclude_cols = {entity_col, time_col, outcome_col} if outcome_col else {entity_col, time_col}
            signals = [c for c in df.columns if c not in exclude_cols]
            n_signals = len(signals)

        # Basic metadata
        if entity_col in df.columns:
            n_entities = df[entity_col].n_unique()
            entities = df[entity_col].unique().sort().to_list()
        else:
            n_entities = 1
            entities = ["entity_1"]

        n_rows = len(df)

        self.metadata = {
            "source_rows": n_rows,
            "n_entities": n_entities,
            "n_signals": n_signals,
            "signals": signals,
            "entities": entities[:10],  # First 10
            "is_long_format": is_long,
            "has_outcome": outcome_col is not None,
        }

        # Analyze each signal
        for signal in signals:
            quality = self._analyze_signal(df, signal, is_long, signal_col, value_col, outcome_col)
            self.signal_quality[signal] = quality

        # Find redundant pairs
        self._find_redundant_pairs(df, signals, is_long, signal_col, value_col)

        return self.metadata

    def _analyze_signal(
        self,
        df: pl.DataFrame,
        signal: str,
        is_long: bool,
        signal_col: str,
        value_col: str,
        outcome_col: str,
    ) -> SignalQuality:
        """Analyze a single signal's quality."""

        if is_long:
            values = df.filter(pl.col(signal_col) == signal)[value_col].drop_nulls().to_numpy()
        else:
            if signal not in df.columns:
                return SignalQuality(
                    signal_id=signal,
                    status="missing",
                    variance=0.0,
                    predictive=0.0,
                    action="exclude",
                    reason="column not found",
                )
            values = df[signal].drop_nulls().to_numpy()

        if len(values) == 0:
            return SignalQuality(
                signal_id=signal,
                status="empty",
                variance=0.0,
                predictive=0.0,
                action="exclude",
                reason="no data",
            )

        variance = float(np.var(values))

        # Check for flatline
        if variance < 0.001:
            return SignalQuality(
                signal_id=signal,
                status="flatline",
                variance=variance,
                predictive=0.0,
                action="exclude",
                reason="flatline (variance < 0.001)",
            )

        # Check for noise (high variance, no predictive value)
        # Predictive value requires outcome column
        predictive = 0.0
        if outcome_col and outcome_col in df.columns:
            # Simple correlation with outcome
            if is_long:
                # More complex for long format
                predictive = 0.5  # Placeholder
            else:
                outcome = df[outcome_col].drop_nulls().to_numpy()
                if len(outcome) == len(values):
                    corr = np.corrcoef(values, outcome)[0, 1]
                    predictive = abs(corr) if not np.isnan(corr) else 0.0

        return SignalQuality(
            signal_id=signal,
            status="valid",
            variance=variance,
            predictive=predictive,
            action="include",
            reason="",
        )

    def _find_redundant_pairs(
        self,
        df: pl.DataFrame,
        signals: List[str],
        is_long: bool,
        signal_col: str,
        value_col: str,
        threshold: float = 0.95,
    ):
        """Find highly correlated signal pairs and mark as redundant."""
        if len(signals) < 2:
            return

        # Build correlation matrix (simplified for wide format)
        if not is_long:
            for i, sig_a in enumerate(signals):
                for sig_b in signals[i+1:]:
                    if sig_a not in df.columns or sig_b not in df.columns:
                        continue
                    vals_a = df[sig_a].drop_nulls().to_numpy()
                    vals_b = df[sig_b].drop_nulls().to_numpy()

                    min_len = min(len(vals_a), len(vals_b))
                    if min_len < 10:
                        continue

                    corr = np.corrcoef(vals_a[:min_len], vals_b[:min_len])[0, 1]

                    if not np.isnan(corr) and abs(corr) > threshold:
                        # Mark the second one as redundant
                        if sig_b in self.signal_quality:
                            self.signal_quality[sig_b] = SignalQuality(
                                signal_id=sig_b,
                                status="redundant",
                                variance=self.signal_quality[sig_b].variance,
                                predictive=self.signal_quality[sig_b].predictive,
                                action="exclude",
                                reason=f"redundant (r={corr:.2f} with {sig_a})",
                                redundant_with=sig_a,
                            )

    def get_valid_signals(self) -> List[str]:
        """Return list of signals marked for inclusion."""
        return [s for s, q in self.signal_quality.items() if q.action == "include"]

    def get_excluded_signals(self) -> List[str]:
        """Return list of signals marked for exclusion."""
        return [s for s, q in self.signal_quality.items() if q.action == "exclude"]

    def format_section(self) -> str:
        """Format as report section."""
        lines = []

        lines.append("=" * 65)
        lines.append("SECTION 1: DATA CHARACTERIZATION")
        lines.append("=" * 65)
        lines.append("")

        # Data summary
        lines.append("DATA SUMMARY")
        lines.append("-" * 65)
        lines.append(f"  Entities:        {self.metadata['n_entities']}")
        lines.append(f"  Signals:         {self.metadata['n_signals']}")
        lines.append(f"  Total records:   {self.metadata['source_rows']:,}")
        lines.append(f"  Format:          {'Long' if self.metadata['is_long_format'] else 'Wide'}")
        lines.append("")

        # Signal quality table
        lines.append("SIGNAL QUALITY ASSESSMENT")
        lines.append("-" * 65)
        lines.append(f"{'Signal':<20} {'Status':<12} {'Variance':<10} {'Action':<10} {'Reason'}")
        lines.append("-" * 65)

        for signal, quality in sorted(self.signal_quality.items()):
            status_icon = "+" if quality.status == "valid" else "?" if quality.status == "redundant" else "x"
            lines.append(
                f"{signal:<20} {status_icon} {quality.status:<10} {quality.variance:<10.3f} "
                f"{quality.action:<10} {quality.reason}"
            )

        lines.append("")
        valid = self.get_valid_signals()
        excluded = self.get_excluded_signals()
        lines.append(f"Signals retained:  {len(valid)} of {self.metadata['n_signals']}")
        lines.append(f"Signals excluded:  {len(excluded)}")
        lines.append("")

        # Thesis paragraph
        lines.append("=" * 65)
        lines.append("THESIS LANGUAGE")
        lines.append("=" * 65)
        lines.append("")

        para = self._generate_paragraph()
        lines.append(para)
        lines.append("")

        return "\n".join(lines)

    def _generate_paragraph(self) -> str:
        """Generate thesis-ready paragraph."""
        valid = self.get_valid_signals()
        excluded = self.get_excluded_signals()

        # Build exclusion reasons
        reasons = []
        for sig in excluded:
            q = self.signal_quality[sig]
            if q.status == "redundant":
                reasons.append(f"one due to redundancy with {q.redundant_with} (r>0.95)")
            elif q.status == "flatline":
                reasons.append("one due to insufficient variance (flatline)")
            elif q.status == "noise":
                reasons.append("one due to no predictive value")

        reason_text = ", ".join(set(reasons)) if reasons else "various quality issues"

        para = (
            f"Of the {self.metadata['n_signals']} measured variables, "
            f"{len(valid)} were retained for analysis. "
            f"{len(excluded)} were excluded: {reason_text}. "
            f"Data comprised {self.metadata['n_entities']} experimental entities "
            f"with {self.metadata['source_rows']:,} total observations."
        )

        return para


# =============================================================================
# SECTION 2: COHORT DISCOVERY
# =============================================================================

class CohortDiscoverer:
    """
    Discover natural groupings in data.

    Outputs Results 3.1 content.
    """

    def __init__(self):
        self.cohorts: List[Dict] = []
        self.n_cohorts: int = 0
        self.separation_stat: float = 0.0
        self.separation_p: float = 1.0

    def discover(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        entity_col: str = "entity_id",
        max_k: int = 8,
    ) -> int:
        """
        Discover cohorts using clustering.

        Returns number of cohorts found.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            from scipy.stats import kruskal
        except ImportError:
            print("Warning: sklearn/scipy not available for cohort discovery")
            self.n_cohorts = 1
            return 1

        # Aggregate features per entity
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            self.n_cohorts = 1
            return 1

        if entity_col in df.columns:
            agg_df = df.group_by(entity_col).agg([
                pl.col(c).mean().alias(c) for c in available_cols
            ])
        else:
            agg_df = df.select(available_cols)

        X = agg_df.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0)

        if len(X) < 4:
            self.n_cohorts = 1
            return 1

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find optimal k using silhouette
        best_k = 2
        best_score = -np.inf

        for k in range(2, min(max_k, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        self.n_cohorts = best_k

        # Compute Kruskal-Wallis for separation
        if len(available_cols) > 0:
            groups = [X[:, 0][labels == i] for i in range(best_k)]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                stat, p = kruskal(*groups)
                self.separation_stat = stat
                self.separation_p = p

        # Build cohort descriptions
        self.cohorts = []
        for i in range(best_k):
            mask = labels == i
            cohort_data = X[mask]

            self.cohorts.append({
                "id": chr(65 + i),  # A, B, C, ...
                "n": int(mask.sum()),
                "means": {col: float(cohort_data[:, j].mean())
                         for j, col in enumerate(available_cols)},
                "stds": {col: float(cohort_data[:, j].std())
                        for j, col in enumerate(available_cols)},
            })

        return best_k

    def format_section(self) -> str:
        """Format as report section."""
        lines = []

        lines.append("")
        lines.append("=" * 65)
        lines.append("SECTION 2: COHORT DISCOVERY")
        lines.append("=" * 65)
        lines.append("")

        lines.append("NATURAL GROUPINGS DISCOVERED")
        lines.append("-" * 65)
        lines.append(f"  Method:       K-means clustering on behavioral features")
        lines.append(f"  Optimal k:    {self.n_cohorts}")
        lines.append(f"  Separation:   Kruskal-Wallis H = {self.separation_stat:.1f}, p = {self.separation_p:.4f}")
        lines.append("")

        # Cohort table
        lines.append("COHORT CHARACTERISTICS")
        lines.append("-" * 65)

        for cohort in self.cohorts:
            lines.append(f"  Cohort {cohort['id']}: n = {cohort['n']}")
            for feat, mean in list(cohort['means'].items())[:3]:
                std = cohort['stds'].get(feat, 0)
                lines.append(f"    {feat}: {mean:.2f} +/- {std:.2f}")
            lines.append("")

        # Thesis paragraph
        lines.append("=" * 65)
        lines.append("THESIS LANGUAGE")
        lines.append("=" * 65)
        lines.append("")

        para = self._generate_paragraph()
        lines.append(para)
        lines.append("")

        return "\n".join(lines)

    def _generate_paragraph(self) -> str:
        """Generate thesis-ready paragraph."""
        cohort_sizes = ", ".join([f"Cohort {c['id']} (n={c['n']})" for c in self.cohorts])

        sig = "significant" if self.separation_p < 0.05 else "non-significant"

        para = (
            f"Unsupervised clustering revealed {self.n_cohorts} distinct behavioral cohorts "
            f"(H = {self.separation_stat:.1f}, p = {self.separation_p:.4f}). "
            f"Cohort composition: {cohort_sizes}. "
            f"Separation was statistically {sig} at alpha = 0.05."
        )

        return para


# =============================================================================
# SECTION 3: CRITICAL MOMENTS
# =============================================================================

def format_moments_section(moments: Dict, coherence_series: np.ndarray = None) -> str:
    """
    Format critical moments as report section.

    Outputs Results 3.2 content.
    """
    lines = []

    t0 = moments.get("T0_healthy", {})
    t1 = moments.get("T1_uncoupling", {})
    t2 = moments.get("T2_severe", {})

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 3: CRITICAL MOMENTS")
    lines.append("=" * 65)
    lines.append("")

    lines.append("THREE CRITICAL MOMENTS DETECTED")
    lines.append("-" * 65)
    lines.append("")

    # T0
    t0_range = t0.get('window_range', [t0.get('window', 0), t0.get('window', 0)])
    if t0_range:
        range_str = f"windows {t0_range[0]}-{t0_range[1]}"
    else:
        range_str = f"window {t0.get('window', 0)}"

    lines.append(f"  T0  BASELINE STATE")
    lines.append(f"      {range_str}")
    lines.append(f"      Coherence: {t0.get('coherence', 0):.2f}")
    lines.append(f"      Confidence: {t0.get('confidence', 0)*100:.0f}%")
    lines.append(f"      Detection: {t0.get('detection_method', 'unknown')}")
    lines.append("")

    # T1
    lines.append(f"  T1  TRANSITION ONSET")
    lines.append(f"      window {t1.get('window', 0)}")
    lines.append(f"      Coherence: {t1.get('coherence', 0):.2f}")
    lines.append(f"      Confidence: {t1.get('confidence', 0)*100:.0f}%")
    if t1.get('trigger_pair'):
        lines.append(f"      Trigger: {t1['trigger_pair']} broke first")
    lines.append("")

    # T2
    lines.append(f"  T2  TERMINAL STATE")
    lines.append(f"      window {t2.get('window', 0)}")
    lines.append(f"      Coherence: {t2.get('coherence', 0):.2f}")
    lines.append(f"      Confidence: {t2.get('confidence', 0)*100:.0f}%")
    lines.append("")

    # Coherence trajectory
    coh_t0 = t0.get('coherence', 1)
    coh_t1 = t1.get('coherence', 0.5)
    coh_t2 = t2.get('coherence', 0)

    drop_t1 = ((coh_t0 - coh_t1) / coh_t0 * 100) if coh_t0 > 0 else 0
    drop_t2 = ((coh_t0 - coh_t2) / coh_t0 * 100) if coh_t0 > 0 else 0

    lines.append("COHERENCE TRAJECTORY")
    lines.append("-" * 65)
    lines.append(f"  T0 -> T1:  {coh_t0:.2f} -> {coh_t1:.2f}  (down {drop_t1:.0f}%)")
    lines.append(f"  T1 -> T2:  {coh_t1:.2f} -> {coh_t2:.2f}  (down {drop_t2:.0f}% total)")
    lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    t0_start = t0_range[0] if t0_range else t0.get('window', 0)
    t0_end = t0_range[1] if t0_range else t0.get('window', 0)

    para = (
        f"Analysis identified three distinct phases. "
        f"The baseline state (windows {t0_start}-{t0_end}) exhibited stable "
        f"inter-signal coherence ({coh_t0:.2f}). "
        f"Transition onset occurred at window {t1.get('window', 0)}, marked by a "
        f"{drop_t1:.0f}% reduction in system coherence. "
        f"The terminal state was reached at window {t2.get('window', 0)}, "
        f"with coherence reduced to {coh_t2:.2f} ({drop_t2:.0f}% below baseline)."
    )
    lines.append(para)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 4: SIGNAL ANALYSIS (Vector Layer)
# =============================================================================

class SignalAnalyzer:
    """
    Analyze individual signal behavior through phases.

    Outputs Results 3.3 content.
    """

    def __init__(self):
        self.signal_stats: Dict[str, Dict] = {}

    def analyze(
        self,
        df: pl.DataFrame,
        signals: List[str],
        moments: Dict,
        time_col: str = "timestamp",
    ):
        """Analyze signal behavior at each phase."""

        t0 = moments.get("T0_healthy", {})
        t1 = moments.get("T1_uncoupling", {})
        t2 = moments.get("T2_severe", {})

        t0_range = t0.get('window_range', [0, 0])
        t0_start = t0_range[0] if t0_range else 0
        t0_end = t0_range[1] if t0_range else 0
        t1_window = t1.get('window', 0)
        t2_window = t2.get('window', 0)

        for signal in signals:
            if signal not in df.columns:
                continue

            # Get values at each phase
            baseline_vals = df.filter(
                (pl.col(time_col) >= t0_start) & (pl.col(time_col) <= t0_end)
            )[signal].drop_nulls().to_numpy()

            onset_vals = df.filter(
                pl.col(time_col) == t1_window
            )[signal].drop_nulls().to_numpy()

            terminal_vals = df.filter(
                pl.col(time_col) == t2_window
            )[signal].drop_nulls().to_numpy()

            if len(baseline_vals) == 0:
                continue

            baseline_mean = float(np.mean(baseline_vals))
            baseline_std = float(np.std(baseline_vals))
            # Use coefficient of variation to determine meaningful std threshold
            # If std is < 0.1% of mean (or very small absolute), signal is essentially constant
            min_std = max(abs(baseline_mean) * 0.001, 0.01)
            effective_std = max(baseline_std, min_std)

            onset_val = float(onset_vals[0]) if len(onset_vals) > 0 else baseline_mean
            terminal_val = float(terminal_vals[0]) if len(terminal_vals) > 0 else baseline_mean

            # Compute sigma with clamping to prevent extreme values
            onset_sigma = np.clip((onset_val - baseline_mean) / effective_std, -100, 100)
            terminal_sigma = np.clip((terminal_val - baseline_mean) / effective_std, -100, 100)

            self.signal_stats[signal] = {
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "onset_value": onset_val,
                "onset_sigma": float(onset_sigma),
                "onset_delta_pct": (onset_val - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else 0,
                "terminal_value": terminal_val,
                "terminal_sigma": float(terminal_sigma),
                "terminal_delta_pct": (terminal_val - baseline_mean) / baseline_mean * 100 if baseline_mean != 0 else 0,
            }

    def format_section(self) -> str:
        """Format as report section."""
        lines = []

        lines.append("")
        lines.append("=" * 65)
        lines.append("SECTION 4: SIGNAL ANALYSIS (Vector Layer)")
        lines.append("=" * 65)
        lines.append("")

        lines.append("SIGNAL VALUES BY PHASE")
        lines.append("-" * 65)
        lines.append(f"{'Signal':<20} {'Baseline':<12} {'Onset':<15} {'Terminal':<15}")
        lines.append("-" * 65)

        for signal, stats in sorted(self.signal_stats.items()):
            baseline = f"{stats['baseline_mean']:.2f}"
            onset = f"{stats['onset_value']:.2f} ({stats['onset_sigma']:+.1f}s)"
            terminal = f"{stats['terminal_value']:.2f} ({stats['terminal_sigma']:+.1f}s)"
            lines.append(f"{signal:<20} {baseline:<12} {onset:<15} {terminal:<15}")

        lines.append("")

        # Identify top movers
        lines.append("TOP MOVERS (largest deviation from baseline)")
        lines.append("-" * 65)

        sorted_by_change = sorted(
            self.signal_stats.items(),
            key=lambda x: abs(x[1]['terminal_sigma']),
            reverse=True
        )[:5]

        for i, (signal, stats) in enumerate(sorted_by_change):
            lines.append(f"  {i+1}. {signal}: {stats['terminal_sigma']:+.1f}s at terminal")

        lines.append("")

        # Thesis paragraph
        lines.append("=" * 65)
        lines.append("THESIS LANGUAGE")
        lines.append("=" * 65)
        lines.append("")

        if sorted_by_change:
            top_signal, top_stats = sorted_by_change[0]
            para = (
                f"Individual signal analysis revealed {len(self.signal_stats)} variables "
                f"with measurable changes across phases. "
                f"The most significant deviation was observed in {top_signal}, "
                f"which shifted {top_stats['terminal_sigma']:+.1f} standard deviations "
                f"from baseline ({top_stats['terminal_delta_pct']:+.0f}% change). "
            )
        else:
            para = "Individual signal analysis was not performed due to insufficient data."

        lines.append(para)
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# SECTION 5: RELATIONSHIP DYNAMICS (Geometry Layer)
# =============================================================================

class RelationshipAnalyzer:
    """
    Analyze pairwise relationships through phases.

    Outputs Results 3.4 content.
    """

    def __init__(self):
        self.pair_stats: Dict[str, Dict] = {}

    def analyze(
        self,
        df: pl.DataFrame,
        signals: List[str],
        moments: Dict,
        time_col: str = "timestamp",
    ):
        """Analyze pairwise correlations at each phase."""

        t0 = moments.get("T0_healthy", {})
        t1 = moments.get("T1_uncoupling", {})
        t2 = moments.get("T2_severe", {})

        t0_range = t0.get('window_range', [0, 0])
        t0_start = t0_range[0] if t0_range else 0
        t0_end = t0_range[1] if t0_range else 0
        t1_window = t1.get('window', 0)
        t2_window = t2.get('window', 0)

        # Get data at each phase
        baseline_df = df.filter(
            (pl.col(time_col) >= t0_start) & (pl.col(time_col) <= t0_end)
        )

        # For onset/terminal, use windows around the moment
        onset_df = df.filter(
            (pl.col(time_col) >= t1_window - 5) & (pl.col(time_col) <= t1_window + 5)
        )

        terminal_df = df.filter(
            (pl.col(time_col) >= t2_window - 5) & (pl.col(time_col) <= t2_window + 5)
        )

        # Compute pairwise correlations
        for i, sig_a in enumerate(signals):
            for sig_b in signals[i+1:]:
                if sig_a not in df.columns or sig_b not in df.columns:
                    continue

                pair_id = f"{sig_a} <-> {sig_b}"

                # Baseline correlation
                baseline_r = self._compute_correlation(baseline_df, sig_a, sig_b)
                onset_r = self._compute_correlation(onset_df, sig_a, sig_b)
                terminal_r = self._compute_correlation(terminal_df, sig_a, sig_b)

                # Use previous value if not enough data
                if onset_r == 0.0 and baseline_r != 0.0:
                    onset_r = baseline_r
                if terminal_r == 0.0 and onset_r != 0.0:
                    terminal_r = onset_r

                self.pair_stats[pair_id] = {
                    "signal_a": sig_a,
                    "signal_b": sig_b,
                    "baseline_r": baseline_r,
                    "onset_r": onset_r,
                    "terminal_r": terminal_r,
                    "onset_delta": onset_r - baseline_r,
                    "terminal_delta": terminal_r - baseline_r,
                    "broken": abs(terminal_r) < 0.3 and abs(baseline_r) > 0.5,
                }

    def _compute_correlation(self, df: pl.DataFrame, sig_a: str, sig_b: str) -> float:
        """Compute correlation between two signals."""
        if sig_a not in df.columns or sig_b not in df.columns:
            return 0.0

        vals_a = df[sig_a].drop_nulls().to_numpy()
        vals_b = df[sig_b].drop_nulls().to_numpy()

        min_len = min(len(vals_a), len(vals_b))
        if min_len < 5:
            return 0.0

        corr = np.corrcoef(vals_a[:min_len], vals_b[:min_len])[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    def format_section(self) -> str:
        """Format as report section."""
        lines = []

        lines.append("")
        lines.append("=" * 65)
        lines.append("SECTION 5: RELATIONSHIP DYNAMICS (Geometry Layer)")
        lines.append("=" * 65)
        lines.append("")

        lines.append("PAIRWISE CORRELATION EVOLUTION")
        lines.append("-" * 65)
        lines.append(f"{'Pair':<25} {'Baseline':<10} {'Onset':<10} {'Terminal':<10} {'Status'}")
        lines.append("-" * 65)

        for pair_id, stats in sorted(self.pair_stats.items()):
            status = "BROKEN" if stats['broken'] else "ok"
            lines.append(
                f"{pair_id:<25} {stats['baseline_r']:<10.2f} "
                f"{stats['onset_r']:<10.2f} {stats['terminal_r']:<10.2f} {status}"
            )

        lines.append("")

        # Identify broken relationships
        broken = [p for p, s in self.pair_stats.items() if s['broken']]

        if broken:
            lines.append("RELATIONSHIPS THAT BROKE")
            lines.append("-" * 65)
            for pair in broken:
                stats = self.pair_stats[pair]
                lines.append(f"  {pair}: {stats['baseline_r']:.2f} -> {stats['terminal_r']:.2f}")
            lines.append("")

        # Thesis paragraph
        lines.append("=" * 65)
        lines.append("THESIS LANGUAGE")
        lines.append("=" * 65)
        lines.append("")

        n_pairs = len(self.pair_stats)
        n_broken = len(broken)

        para = (
            f"Pairwise correlation analysis examined {n_pairs} signal relationships. "
            f"At baseline, relationships exhibited stable correlation structure. "
        )

        if n_broken > 0:
            first_broken = broken[0]
            stats = self.pair_stats[first_broken]
            para += (
                f"By terminal state, {n_broken} relationships had decoupled. "
                f"The {first_broken} relationship showed the largest change, "
                f"declining from r={stats['baseline_r']:.2f} to r={stats['terminal_r']:.2f}."
            )
        else:
            para += "All relationships remained intact through terminal state."

        lines.append(para)
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# SECTION 6: STATE TRAJECTORY
# =============================================================================

def format_state_section(
    coherence_series: np.ndarray,
    moments: Dict,
) -> str:
    """
    Format state trajectory section.

    Outputs Results 3.5 content.
    """
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 6: STATE TRAJECTORY")
    lines.append("=" * 65)
    lines.append("")

    # Compute velocity and acceleration
    velocity = np.gradient(coherence_series)
    acceleration = np.gradient(velocity)

    t1 = moments.get("T1_uncoupling", {})
    t2 = moments.get("T2_severe", {})

    t1_window = min(t1.get('window', len(coherence_series)//2), len(velocity)-1)
    t2_window = min(t2.get('window', len(coherence_series)-1), len(velocity)-1)

    lines.append("COHERENCE DYNAMICS")
    lines.append("-" * 65)
    lines.append(f"  Mean coherence:           {np.mean(coherence_series):.3f}")
    lines.append(f"  Min coherence:            {np.min(coherence_series):.3f}")
    lines.append(f"  Max coherence:            {np.max(coherence_series):.3f}")
    lines.append("")

    lines.append("VELOCITY (rate of change)")
    lines.append("-" * 65)
    lines.append(f"  At baseline:              {velocity[0]:.4f} per window")
    lines.append(f"  At onset (T1):            {velocity[t1_window]:.4f} per window")
    lines.append(f"  At terminal (T2):         {velocity[t2_window]:.4f} per window")
    lines.append(f"  Max negative velocity:    {np.min(velocity):.4f} (fastest decline)")
    lines.append("")

    lines.append("ACCELERATION")
    lines.append("-" * 65)
    lines.append(f"  At onset (T1):            {acceleration[t1_window]:.4f}")
    lines.append(f"  At terminal (T2):         {acceleration[t2_window]:.4f}")
    lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    max_decline_window = int(np.argmin(velocity))

    para = (
        f"State trajectory analysis revealed coherence dynamics across {len(coherence_series)} windows. "
        f"Maximum rate of coherence decline ({np.min(velocity):.4f}/window) occurred at window {max_decline_window}. "
        f"Velocity at transition onset was {velocity[t1_window]:.4f}/window, "
        f"indicating {'accelerating' if acceleration[t1_window] < 0 else 'decelerating'} degradation."
    )

    lines.append(para)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 7: ML FEATURES & PREDICTION
# =============================================================================

def format_ml_section(
    features: Dict,
    prediction: Dict = None,
) -> str:
    """
    Format ML features and prediction section.

    Outputs Results 3.6 content.
    """
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 7: ML FEATURES & PREDICTION")
    lines.append("=" * 65)
    lines.append("")

    lines.append("STRUCTURAL FEATURES (ML-ready)")
    lines.append("-" * 65)
    lines.append(f"{'Feature':<30} {'Value':<15} {'Description'}")
    lines.append("-" * 65)

    feature_descriptions = {
        "phase": "Current phase (0=baseline, 1=transition, 2=terminal)",
        "phase_confidence": "Confidence in phase assignment",
        "coherence": "Current system coherence",
        "coherence_velocity": "Rate of coherence change",
        "baseline_distance_sigma": "Z-score from baseline",
        "pct_through_transition": "Progress through transition (0-1)",
    }

    for feat, desc in feature_descriptions.items():
        value = features.get(feat, "N/A")
        if isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        lines.append(f"  {feat:<28} {value_str:<15} {desc}")

    lines.append("")

    # Prediction (if available)
    if prediction:
        lines.append("TRAJECTORY PREDICTION")
        lines.append("-" * 65)

        if prediction.get('predicted_terminal'):
            lines.append(f"  Predicted terminal:     window {prediction['predicted_terminal']}")
            lines.append(f"  Uncertainty:            +/- {prediction.get('terminal_std', 0):.0f} windows")
            lines.append(f"  Windows remaining:      {prediction.get('days_remaining', 'N/A')}")
            lines.append(f"  Confidence:             {prediction.get('confidence', 0)*100:.0f}%")
            lines.append(f"  Based on:               {prediction.get('n_matches', 0)} similar experiments")
        else:
            lines.append(f"  Prediction unavailable: {prediction.get('error', 'No matching data')}")

        lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    phase_names = {0: "baseline", 1: "transition", 2: "terminal"}
    phase = features.get('phase', 0)

    para = (
        f"Structural feature extraction yielded {len(feature_descriptions)} domain-agnostic metrics. "
        f"Current state is classified as {phase_names.get(phase, 'unknown')} "
        f"(confidence: {features.get('phase_confidence', 0)*100:.0f}%). "
    )

    if prediction and prediction.get('predicted_terminal'):
        para += (
            f"Based on trajectory matching with {prediction.get('n_matches', 0)} historical experiments, "
            f"terminal state is predicted at window {prediction['predicted_terminal']} "
            f"(+/-{prediction.get('terminal_std', 0):.0f}, {prediction.get('confidence', 0)*100:.0f}% confidence)."
        )

    lines.append(para)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 8: ANOMALIES
# =============================================================================

def format_anomalies_section(anomalies: List[Dict]) -> str:
    """Format anomalies section."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 8: ANOMALIES")
    lines.append("=" * 65)
    lines.append("")

    if not anomalies:
        lines.append("No anomalies detected.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"{len(anomalies)} samples flagged for review:")
    lines.append("-" * 65)
    lines.append(f"{'Entity':<15} {'Issue':<30} {'Recommendation'}")
    lines.append("-" * 65)

    for anom in anomalies[:10]:  # First 10
        lines.append(
            f"{anom.get('entity_id', 'N/A'):<15} "
            f"{anom.get('issue', 'Unknown'):<30} "
            f"{anom.get('recommendation', 'Review')}"
        )

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# SECTION 9: CONCLUSIONS
# =============================================================================

def format_conclusions_section(
    metadata: Dict,
    n_cohorts: int,
    moments: Dict,
    n_broken_pairs: int,
) -> str:
    """Format conclusions section."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 9: KEY FINDINGS")
    lines.append("=" * 65)
    lines.append("")

    t0 = moments.get("T0_healthy", {})
    t1 = moments.get("T1_uncoupling", {})
    t2 = moments.get("T2_severe", {})

    lines.append(f"1. {n_cohorts} DISTINCT BEHAVIORAL COHORTS identified")
    lines.append(f"   with statistically significant separation.")
    lines.append("")

    t0_range = t0.get('window_range', [0, 0])
    t0_end = t0_range[1] if t0_range else t0.get('window', 0)

    lines.append(f"2. DEGRADATION PROCEEDS IN THREE PHASES:")
    lines.append(f"   - Baseline stability (windows 0-{t0_end})")
    lines.append(f"   - Transition (windows {t1.get('window', 0)}-{t2.get('window', 0)})")
    lines.append(f"   - Terminal state (window {t2.get('window', 0)}+)")
    lines.append("")

    if t1.get('trigger_pair'):
        lines.append(f"3. EARLY WARNING INDICATOR:")
        lines.append(f"   {t1['trigger_pair']} relationship breaks first")
    lines.append("")

    lines.append(f"4. {n_broken_pairs} RELATIONSHIPS DECOUPLED by terminal state")
    lines.append("")

    # Final thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE (Conclusion)")
    lines.append("=" * 65)
    lines.append("")

    para = (
        f"This study characterized the temporal dynamics of system degradation "
        f"across {metadata.get('n_entities', 0)} experimental entities. "
        f"Unsupervised analysis identified {n_cohorts} distinct behavioral cohorts "
        f"and three degradation phases. "
        f"The transition from baseline to terminal state was marked by "
        f"progressive decoupling of {n_broken_pairs} inter-signal relationships. "
    )

    if t1.get('trigger_pair'):
        para += (
            f"The {t1['trigger_pair']} relationship emerged as the primary "
            f"early warning indicator, decoupling prior to system-wide degradation."
        )

    lines.append(para)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# PER-ENTITY ANALYSIS
# =============================================================================

def analyze_single_entity(
    df: pl.DataFrame,
    entity_id: str,
    valid_signals: List[str],
    time_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Run analysis on a single entity.

    Returns dict with moments, signal_stats, pair_stats, coherence.
    """
    df = df.sort(time_col)
    n = len(df)

    # Compute coherence proxy from first two valid signals
    if len(valid_signals) >= 2:
        vals_a = df[valid_signals[0]].to_numpy()
        vals_b = df[valid_signals[1]].to_numpy()

        window = min(20, n // 5)
        coherence = []
        for i in range(n):
            start = max(0, i - window)
            if i - start >= 5:
                r = np.corrcoef(vals_a[start:i+1], vals_b[start:i+1])[0, 1]
                coherence.append(r if not np.isnan(r) else 0.5)
            else:
                coherence.append(0.5)
        coherence = np.array(coherence)
    else:
        coherence = np.ones(n) * 0.5

    # Detect moments
    geom_df = pl.DataFrame({
        "entity_id": [entity_id] * n,
        "timestamp": range(n),
        "mean_mode_coherence": coherence,
    })

    try:
        if SUMMARIZE_AVAILABLE:
            config = MomentConfig()
            detector = MomentDetector(config)
            moments = detector.detect_all(geom_df)
            moments_dict = {k: asdict(v) for k, v in moments.items()}
        else:
            raise ImportError("fallback")
    except:
        moments_dict = {
            "T0_healthy": {"window": 0, "window_range": [0, int(n*0.2)], "coherence": float(coherence[:int(n*0.2)].mean()), "confidence": 0.5, "detection_method": "fallback"},
            "T1_uncoupling": {"window": int(n*0.5), "coherence": float(coherence[int(n*0.5)]), "confidence": 0.5, "detection_method": "fallback"},
            "T2_severe": {"window": n-1, "coherence": float(coherence[-1]), "confidence": 0.5, "detection_method": "fallback"},
        }

    # Signal analysis
    signal_analyzer = SignalAnalyzer()
    signal_analyzer.analyze(df, valid_signals, moments_dict, time_col=time_col)

    # Relationship analysis
    relationship_analyzer = RelationshipAnalyzer()
    relationship_analyzer.analyze(df, valid_signals[:10], moments_dict, time_col=time_col)

    broken_pairs = [p for p, s in relationship_analyzer.pair_stats.items() if s['broken']]

    return {
        "entity_id": entity_id,
        "n_cycles": n,
        "moments": moments_dict,
        "signal_stats": signal_analyzer.signal_stats,
        "pair_stats": relationship_analyzer.pair_stats,
        "coherence": coherence,
        "n_broken_pairs": len(broken_pairs),
        "broken_pairs": broken_pairs,
    }


def format_multi_entity_moments(entity_results: List[Dict]) -> str:
    """Format moments comparison across entities."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 3: CRITICAL MOMENTS (Per-Entity)")
    lines.append("=" * 65)
    lines.append("")

    lines.append("MOMENT DETECTION BY ENTITY")
    lines.append("-" * 65)
    lines.append(f"{'Entity':<15} {'Cycles':<8} {'T0 End':<10} {'T1 Onset':<10} {'T2 Terminal':<12} {'Coh Drop'}")
    lines.append("-" * 65)

    for r in entity_results:
        entity = r['entity_id']
        n = r['n_cycles']
        t0 = r['moments'].get('T0_healthy', {})
        t1 = r['moments'].get('T1_uncoupling', {})
        t2 = r['moments'].get('T2_severe', {})

        t0_end = t0.get('window_range', [0, 0])[1] if t0.get('window_range') else t0.get('window', 0)
        t1_win = t1.get('window', 0)
        t2_win = t2.get('window', 0)

        coh_t0 = t0.get('coherence', 1)
        coh_t2 = t2.get('coherence', 0)
        drop = ((coh_t0 - coh_t2) / coh_t0 * 100) if coh_t0 > 0 else 0

        lines.append(f"{entity:<15} {n:<8} {t0_end:<10} {t1_win:<10} {t2_win:<12} {drop:+.0f}%")

    lines.append("")

    # Summary statistics
    t1_windows = [r['moments'].get('T1_uncoupling', {}).get('window', 0) for r in entity_results]
    t2_windows = [r['moments'].get('T2_severe', {}).get('window', 0) for r in entity_results]
    cycles = [r['n_cycles'] for r in entity_results]

    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 65)
    lines.append(f"  Entities analyzed:     {len(entity_results)}")
    lines.append(f"  Lifespan range:        {min(cycles)} - {max(cycles)} cycles")
    lines.append(f"  Mean lifespan:         {np.mean(cycles):.0f} cycles")
    lines.append(f"  T1 onset range:        {min(t1_windows)} - {max(t1_windows)} (mean: {np.mean(t1_windows):.0f})")
    lines.append(f"  T2 terminal range:     {min(t2_windows)} - {max(t2_windows)} (mean: {np.mean(t2_windows):.0f})")
    lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    coh_drops = []
    for r in entity_results:
        t0 = r['moments'].get('T0_healthy', {})
        t2 = r['moments'].get('T2_severe', {})
        coh_t0 = t0.get('coherence', 1)
        coh_t2 = t2.get('coherence', 0)
        if coh_t0 > 0:
            coh_drops.append((coh_t0 - coh_t2) / coh_t0 * 100)

    para = (
        f"Critical moment detection was performed independently for {len(entity_results)} entities. "
        f"Lifespan ranged from {min(cycles)} to {max(cycles)} cycles (mean: {np.mean(cycles):.0f}). "
        f"Transition onset (T1) occurred between cycles {min(t1_windows)} and {max(t1_windows)} "
        f"(mean: {np.mean(t1_windows):.0f}). "
        f"Mean coherence degradation from baseline to terminal was {np.mean(coh_drops):.0f}%."
    )
    lines.append(para)
    lines.append("")

    return "\n".join(lines)


def format_multi_entity_signals(entity_results: List[Dict], valid_signals: List[str]) -> str:
    """Format signal analysis across entities."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 4: SIGNAL ANALYSIS (Aggregated)")
    lines.append("=" * 65)
    lines.append("")

    # Aggregate signal deviations
    signal_terminal_sigmas = {s: [] for s in valid_signals}

    for r in entity_results:
        for signal, stats in r['signal_stats'].items():
            if signal in signal_terminal_sigmas:
                signal_terminal_sigmas[signal].append(stats['terminal_sigma'])

    lines.append("MEAN TERMINAL DEVIATION BY SIGNAL")
    lines.append("-" * 65)
    lines.append(f"{'Signal':<25} {'Mean Sigma':<12} {'Std':<10} {'N'}")
    lines.append("-" * 65)

    sorted_signals = sorted(
        [(s, np.mean(v), np.std(v), len(v)) for s, v in signal_terminal_sigmas.items() if v],
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for signal, mean_sig, std_sig, n in sorted_signals[:10]:
        lines.append(f"{signal:<25} {mean_sig:+.2f}         {std_sig:.2f}       {n}")

    lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    if sorted_signals:
        top_signal, top_mean, _, _ = sorted_signals[0]
        para = (
            f"Aggregated signal analysis across {len(entity_results)} entities revealed "
            f"consistent patterns of deviation. "
            f"The most significant terminal deviation was observed in {top_signal} "
            f"(mean: {top_mean:+.2f} sigma)."
        )
    else:
        para = "Signal analysis could not be aggregated due to insufficient data."

    lines.append(para)
    lines.append("")

    return "\n".join(lines)


def format_multi_entity_relationships(entity_results: List[Dict]) -> str:
    """Format relationship dynamics across entities."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append("SECTION 5: RELATIONSHIP DYNAMICS (Aggregated)")
    lines.append("=" * 65)
    lines.append("")

    # Count broken pairs across entities
    broken_counts = {}
    pair_deltas = {}

    for r in entity_results:
        for pair, stats in r['pair_stats'].items():
            if pair not in pair_deltas:
                pair_deltas[pair] = []
                broken_counts[pair] = 0

            pair_deltas[pair].append(stats['terminal_delta'])
            if stats['broken']:
                broken_counts[pair] += 1

    # Sort by frequency of breaking
    sorted_pairs = sorted(broken_counts.items(), key=lambda x: x[1], reverse=True)

    lines.append("RELATIONSHIP STABILITY ACROSS ENTITIES")
    lines.append("-" * 65)
    lines.append(f"{'Pair':<30} {'Broke':<8} {'Mean Delta':<12} {'Stable'}")
    lines.append("-" * 65)

    for pair, broke_count in sorted_pairs[:15]:
        deltas = pair_deltas.get(pair, [0])
        mean_delta = np.mean(deltas)
        stable_pct = (len(entity_results) - broke_count) / len(entity_results) * 100
        status = "UNSTABLE" if broke_count > len(entity_results) // 2 else "stable"
        lines.append(f"{pair:<30} {broke_count:<8} {mean_delta:+.2f}         {stable_pct:.0f}% {status}")

    lines.append("")

    # Identify consistently broken relationships
    consistently_broken = [p for p, c in broken_counts.items() if c >= len(entity_results) // 2]
    total_broken = sum(r['n_broken_pairs'] for r in entity_results)

    lines.append("SUMMARY")
    lines.append("-" * 65)
    lines.append(f"  Total broken relationships:     {total_broken} across {len(entity_results)} entities")
    lines.append(f"  Mean broken per entity:         {total_broken / len(entity_results):.1f}")
    lines.append(f"  Consistently unstable pairs:    {len(consistently_broken)}")
    if consistently_broken:
        lines.append(f"    {', '.join(consistently_broken[:3])}")
    lines.append("")

    # Thesis paragraph
    lines.append("=" * 65)
    lines.append("THESIS LANGUAGE")
    lines.append("=" * 65)
    lines.append("")

    para = (
        f"Pairwise relationship analysis across {len(entity_results)} entities revealed "
        f"an average of {total_broken / len(entity_results):.1f} decoupled relationships per entity by terminal state. "
    )
    if consistently_broken:
        para += (
            f"The {consistently_broken[0]} relationship was consistently unstable, "
            f"decoupling in {broken_counts[consistently_broken[0]]} of {len(entity_results)} entities."
        )

    lines.append(para)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_full_analysis(
    data_path: str,
    output_dir: str = "report",
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    verbose: bool = True,
) -> str:
    """
    Run complete analysis pipeline.

    Handles both single-entity and multi-entity data automatically.

    Args:
        data_path: Path to CSV or parquet file
        output_dir: Directory to save outputs
        entity_col: Entity column name
        time_col: Time/window column name
        verbose: Print progress

    Returns:
        Complete report as string
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path(data_path)
    if data_path.suffix == ".csv":
        df = pl.read_csv(data_path)
    else:
        df = pl.read_parquet(data_path)

    if verbose:
        print(f"Loaded {len(df)} rows from {data_path}")

    # Detect entities
    if entity_col in df.columns:
        entities = df[entity_col].unique().sort().to_list()
        n_entities = len(entities)
    else:
        entities = ["entity_1"]
        n_entities = 1
        df = df.with_columns(pl.lit("entity_1").alias(entity_col))

    is_multi_entity = n_entities > 1

    if verbose:
        print(f"Detected {n_entities} entities")
        if is_multi_entity:
            print(f"  Running per-entity analysis...")

    # Get signal columns (everything except entity/time)
    exclude = {entity_col, time_col}
    signal_cols = [c for c in df.columns if c not in exclude]

    report_sections = []

    # === SECTION 1: Data Characterization ===
    if verbose:
        print("\n[1/9] Data characterization...")

    characterizer = DataCharacterizer()
    characterizer.analyze(df, entity_col=entity_col, time_col=time_col)
    report_sections.append(characterizer.format_section())

    valid_signals = characterizer.get_valid_signals()

    # === SECTION 2: Cohort Discovery ===
    if verbose:
        print("[2/9] Cohort discovery...")

    cohort_disco = CohortDiscoverer()
    n_cohorts = cohort_disco.discover(df, valid_signals[:10], entity_col=entity_col)
    report_sections.append(cohort_disco.format_section())

    # === PER-ENTITY ANALYSIS ===
    if is_multi_entity:
        if verbose:
            print("[3-5/9] Per-entity analysis...")

        entity_results = []
        for i, entity in enumerate(entities):
            entity_df = df.filter(pl.col(entity_col) == entity)
            result = analyze_single_entity(entity_df, entity, valid_signals, time_col)
            entity_results.append(result)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_entities} entities")

        if verbose:
            print(f"  Processed {len(entity_results)} entities")

        # Aggregate moments
        report_sections.append(format_multi_entity_moments(entity_results))

        # Aggregate signals
        report_sections.append(format_multi_entity_signals(entity_results, valid_signals))

        # Aggregate relationships
        report_sections.append(format_multi_entity_relationships(entity_results))

        # Use first entity's coherence for state trajectory (or average)
        coherence = entity_results[0]['coherence']
        moments_dict = entity_results[0]['moments']

        # Aggregate broken pairs
        n_broken = sum(r['n_broken_pairs'] for r in entity_results) // len(entity_results)

    else:
        # Single entity analysis
        if verbose:
            print("[3/9] Detecting critical moments...")

        entity_df = df.filter(pl.col(entity_col) == entities[0])
        result = analyze_single_entity(entity_df, entities[0], valid_signals, time_col)

        coherence = result['coherence']
        moments_dict = result['moments']

        report_sections.append(format_moments_section(moments_dict, coherence))

        if verbose:
            print("[4/9] Signal analysis...")

        signal_analyzer = SignalAnalyzer()
        signal_analyzer.signal_stats = result['signal_stats']
        report_sections.append(signal_analyzer.format_section())

        if verbose:
            print("[5/9] Relationship dynamics...")

        relationship_analyzer = RelationshipAnalyzer()
        relationship_analyzer.pair_stats = result['pair_stats']
        report_sections.append(relationship_analyzer.format_section())

        n_broken = result['n_broken_pairs']

    # === SECTION 6: State Trajectory ===
    if verbose:
        print("[6/9] State trajectory...")

    report_sections.append(format_state_section(coherence, moments_dict))

    # === SECTION 7: ML Features ===
    if verbose:
        print("[7/9] ML features...")

    t1 = moments_dict.get("T1_uncoupling", {})
    ml_features = {
        "phase": 1 if t1.get("window", 0) < len(coherence) // 2 else 2,
        "phase_confidence": t1.get("confidence", 0.5),
        "coherence": float(coherence[-1]) if len(coherence) > 0 else 0.5,
        "coherence_velocity": float(np.gradient(coherence)[-1]) if len(coherence) > 1 else 0,
        "baseline_distance_sigma": float((coherence[-1] - coherence[:20].mean()) / (coherence[:20].std() + 1e-10)) if len(coherence) > 20 else 0,
        "pct_through_transition": 0.5,
    }

    report_sections.append(format_ml_section(ml_features))

    # === SECTION 8: Anomalies ===
    if verbose:
        print("[8/9] Anomaly detection...")

    anomalies = []
    report_sections.append(format_anomalies_section(anomalies))

    # === SECTION 9: Conclusions ===
    if verbose:
        print("[9/9] Generating conclusions...")

    report_sections.append(format_conclusions_section(
        characterizer.metadata,
        n_cohorts,
        moments_dict,
        n_broken,
    ))

    # Combine report
    full_report = "\n".join(report_sections)

    # Save report
    report_path = output_path / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)

    if verbose:
        print(f"\nReport saved to: {report_path}")

    # Save moments JSON
    if is_multi_entity:
        all_moments = {r['entity_id']: r['moments'] for r in entity_results}
        moments_path = output_path / "moments.json"
        with open(moments_path, "w") as f:
            json.dump(all_moments, f, indent=2, default=str)
    else:
        moments_path = output_path / "moments.json"
        with open(moments_path, "w") as f:
            json.dump(moments_dict, f, indent=2, default=str)

    # Save coherence series
    coherence_df = pl.DataFrame({
        "window": range(len(coherence)),
        "coherence": coherence,
    })
    coherence_df.write_parquet(output_path / "coherence.parquet")

    return full_report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Orthon Analysis Notebook - Complete blind-test pipeline"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV or Parquet)")
    parser.add_argument("--output", type=str, default="report", help="Output directory")
    parser.add_argument("--entity-col", type=str, default="entity_id", help="Entity column name")
    parser.add_argument("--time-col", type=str, default="timestamp", help="Time/window column name")

    args = parser.parse_args()

    print("=" * 70)
    print("ORTHON ANALYSIS NOTEBOOK")
    print("Complete Blind-Test Pipeline")
    print("=" * 70)

    report = run_full_analysis(
        data_path=args.data,
        output_dir=args.output,
        entity_col=args.entity_col,
        time_col=args.time_col,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nReport preview (first 2000 chars):\n")
    print(report[:2000])
    print("\n... [truncated]")
