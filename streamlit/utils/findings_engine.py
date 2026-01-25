"""
ORTHON Findings Engine

Detects what's interesting BEFORE sending to Claude.
Transforms 410 numbers into 5-10 actual findings.

The difference between:
  - Report: "Hurst = 0.73"
  - Smart Report: "HYD_PS1 has unusually high memory — top 5% of signals.
                   Combined with its high volatility, this is a rare 'elephant'
                   pattern that often precedes system stress."
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class FindingType(Enum):
    OUTLIER = "outlier"              # Statistical anomaly
    PATTERN = "pattern"              # Recognized signature
    RELATIONSHIP = "relationship"    # Unexpected correlation/causality
    TRANSITION = "transition"        # Regime change detected
    HIDDEN = "hidden"                # Not obvious, revealed by analysis
    THRESHOLD = "threshold"          # Critical value crossed
    CONTRADICTION = "contradiction"  # Things that shouldn't go together


class Severity(Enum):
    INFO = "info"           # Interesting but not actionable
    NOTABLE = "notable"     # Worth mentioning
    IMPORTANT = "important" # Should act on this
    CRITICAL = "critical"   # Act now


@dataclass
class Finding:
    type: FindingType
    severity: Severity
    headline: str           # One line: "PS1 is an outlier"
    detail: str             # Explanation: "PS1's memory score of 0.92 is 2.4σ above mean..."
    evidence: dict          # Numbers that support this: {"signal": "PS1", "memory": 0.92, "z_score": 2.4}
    signals_involved: list  # Which signals are relevant
    action: str             # What to do: "Monitor PS1 for early warning"


# =============================================================================
# STATISTICAL BASELINES
# =============================================================================

# Typical ranges from analysis of many datasets (calibrate with real data)
TYPICAL_RANGES = {
    'memory': {'mean': 0.50, 'std': 0.15, 'healthy_min': 0.3, 'healthy_max': 0.7},
    'information': {'mean': 0.50, 'std': 0.18, 'healthy_min': 0.2, 'healthy_max': 0.8},
    'frequency': {'mean': 0.40, 'std': 0.20, 'healthy_min': 0.1, 'healthy_max': 0.9},
    'volatility': {'mean': 0.45, 'std': 0.20, 'healthy_min': 0.2, 'healthy_max': 0.7},
    'dynamics': {'mean': 0.40, 'std': 0.15, 'healthy_min': 0.2, 'healthy_max': 0.6},
    'recurrence': {'mean': 0.50, 'std': 0.18, 'healthy_min': 0.3, 'healthy_max': 0.8},
    'discontinuity': {'mean': 0.25, 'std': 0.15, 'healthy_min': 0.0, 'healthy_max': 0.4},
    'derivatives': {'mean': 0.45, 'std': 0.18, 'healthy_min': 0.2, 'healthy_max': 0.7},
    'momentum': {'mean': 0.50, 'std': 0.15, 'healthy_min': 0.3, 'healthy_max': 0.7},
    'coherence': {'mean': 0.65, 'std': 0.15, 'healthy_min': 0.5, 'danger': 0.3},
}

# Rare/unusual combinations (these don't usually occur together)
RARE_COMBINATIONS = [
    # (axis1, threshold1, axis2, threshold2, name, interpretation)
    ('memory', '>', 0.8, 'volatility', '>', 0.8, 'elephant',
     'High memory + high volatility = "elephant" pattern. Shocks persist AND cluster. Watch closely.'),
    ('memory', '<', 0.3, 'momentum', '>', 0.8, 'whipsaw',
     'Forgetful but trending = whipsaw risk. Direction persists but magnitude reverts.'),
    ('information', '>', 0.8, 'recurrence', '>', 0.8, 'strange_attractor',
     'High entropy + high recurrence = strange attractor dynamics. Complex but bounded.'),
    ('frequency', '>', 0.8, 'discontinuity', '>', 0.5, 'interrupted_cycle',
     'Periodic but discontinuous = interrupted cycles. Process being disrupted.'),
    ('volatility', '<', 0.2, 'dynamics', '>', 0.7, 'quiet_chaos',
     'Low volatility + high chaos = quiet chaos. Looks calm but unpredictable.'),
]

# Known danger patterns (pre-failure signatures)
DANGER_PATTERNS = [
    {
        'name': 'pre_failure',
        'conditions': [
            ('volatility', '>', 0.7),
            ('discontinuity', '>', 0.5),
        ],
        'coherence_trend': 'falling',
        'interpretation': 'Classic pre-failure signature: volatility rising, discontinuities appearing, coherence falling.',
    },
    {
        'name': 'drift',
        'conditions': [
            ('momentum', '>', 0.75),
            ('memory', '>', 0.7),
        ],
        'interpretation': 'Drift pattern: persistent trending. System moving away from baseline.',
    },
    {
        'name': 'oscillation_death',
        'conditions': [
            ('frequency', '<', 0.2),  # Was periodic, now not
        ],
        'requires_history': True,  # Need to compare to baseline
        'interpretation': 'Loss of periodicity. If this signal should oscillate, something stopped it.',
    },
]


# =============================================================================
# MAIN FINDINGS ENGINE
# =============================================================================

class FindingsEngine:
    """
    Analyzes ORTHON results and extracts what's actually interesting.
    """

    def __init__(self, results: dict):
        self.results = results
        self.metadata = results.get('metadata', {})
        self.typology = pd.DataFrame(results.get('typology', []))
        self.groups = results.get('groups', {})
        self.dynamics = results.get('dynamics', {})
        self.mechanics = results.get('mechanics', {})
        self.findings = []

    def analyze(self) -> list:
        """Run all detectors and return prioritized findings."""
        self.findings = []

        # Run all detectors
        self._detect_outlier_signals()
        self._detect_rare_combinations()
        self._detect_coherence_issues()
        self._detect_causal_structure()
        self._detect_group_anomalies()
        self._detect_hidden_relationships()
        self._detect_danger_patterns()
        self._detect_contradictions()

        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.IMPORTANT: 1,
            Severity.NOTABLE: 2,
            Severity.INFO: 3,
        }
        self.findings.sort(key=lambda f: severity_order[f.severity])

        return self.findings

    def get_top_findings(self, n: int = 5) -> list:
        """Get the N most important findings."""
        if not self.findings:
            self.analyze()
        return self.findings[:n]

    def to_prompt(self) -> str:
        """Format findings for Claude prompt."""
        if not self.findings:
            self.analyze()

        lines = ["KEY FINDINGS (pre-analyzed):"]
        for i, f in enumerate(self.findings[:7], 1):
            lines.append(f"\n{i}. [{f.severity.value.upper()}] {f.headline}")
            lines.append(f"   {f.detail}")
            lines.append(f"   → Action: {f.action}")

        return '\n'.join(lines)

    # -------------------------------------------------------------------------
    # DETECTOR: Outlier Signals
    # -------------------------------------------------------------------------

    def _detect_outlier_signals(self):
        """Find signals with unusual typology scores."""

        if self.typology.empty:
            return

        axes = ['memory', 'information', 'frequency', 'volatility',
                'dynamics', 'recurrence', 'discontinuity', 'derivatives', 'momentum']

        for axis in axes:
            if axis not in self.typology.columns:
                continue

            values = self.typology[axis]
            mean = values.mean()
            std = values.std()

            if std < 0.01:  # No variation
                continue

            for _, row in self.typology.iterrows():
                val = row[axis]
                z_score = (val - mean) / std

                # Also check against global baseline
                baseline = TYPICAL_RANGES.get(axis, {})
                global_z = (val - baseline.get('mean', 0.5)) / baseline.get('std', 0.2)

                if abs(z_score) > 2.0 or abs(global_z) > 2.5:
                    direction = 'high' if val > mean else 'low'
                    signal_id = row.get('signal_id', row.get('signal', 'unknown'))

                    self.findings.append(Finding(
                        type=FindingType.OUTLIER,
                        severity=Severity.NOTABLE if abs(z_score) < 2.5 else Severity.IMPORTANT,
                        headline=f"{signal_id} has unusually {direction} {axis}",
                        detail=f"{signal_id}'s {axis} score of {val:.2f} is {abs(z_score):.1f}σ "
                               f"{'above' if direction == 'high' else 'below'} the dataset mean ({mean:.2f}). "
                               f"Compared to typical signals, this is in the {'top' if direction == 'high' else 'bottom'} "
                               f"{100 - min(95, abs(global_z) * 30):.0f}%.",
                        evidence={'signal': signal_id, 'axis': axis, 'value': val,
                                  'z_score': z_score, 'global_z': global_z},
                        signals_involved=[signal_id],
                        action=f"Investigate why {signal_id} has extreme {axis}. "
                               f"This could indicate sensor issues, unique behavior, or a problem.",
                    ))

    # -------------------------------------------------------------------------
    # DETECTOR: Rare Combinations
    # -------------------------------------------------------------------------

    def _detect_rare_combinations(self):
        """Find signals with unusual trait combinations."""

        if self.typology.empty:
            return

        for axis1, op1, thresh1, axis2, op2, thresh2, name, interpretation in RARE_COMBINATIONS:
            if axis1 not in self.typology.columns or axis2 not in self.typology.columns:
                continue

            for _, row in self.typology.iterrows():
                val1, val2 = row[axis1], row[axis2]

                cond1 = val1 > thresh1 if op1 == '>' else val1 < thresh1
                cond2 = val2 > thresh2 if op2 == '>' else val2 < thresh2

                if cond1 and cond2:
                    signal_id = row.get('signal_id', row.get('signal', 'unknown'))

                    self.findings.append(Finding(
                        type=FindingType.PATTERN,
                        severity=Severity.IMPORTANT,
                        headline=f"{signal_id} shows '{name}' pattern",
                        detail=f"{signal_id} has {axis1}={val1:.2f} and {axis2}={val2:.2f}. "
                               f"{interpretation}",
                        evidence={'signal': signal_id, 'pattern': name,
                                  axis1: val1, axis2: val2},
                        signals_involved=[signal_id],
                        action=f"This rare combination in {signal_id} warrants attention. "
                               f"Compare to historical behavior.",
                    ))

    # -------------------------------------------------------------------------
    # DETECTOR: Coherence Issues
    # -------------------------------------------------------------------------

    def _detect_coherence_issues(self):
        """Find coherence problems - the most important system-level metric."""

        mean_coh = self.dynamics.get('mean_coherence', 0.5)
        min_coh = self.dynamics.get('coherence_min', mean_coh)
        max_coh = self.dynamics.get('coherence_max', mean_coh)
        std_coh = self.dynamics.get('coherence_std', 0)
        transitions = self.dynamics.get('transitions', [])

        # Low overall coherence
        if mean_coh < TYPICAL_RANGES['coherence']['healthy_min']:
            self.findings.append(Finding(
                type=FindingType.THRESHOLD,
                severity=Severity.IMPORTANT,
                headline=f"System coherence is low ({mean_coh:.2f})",
                detail=f"Mean coherence of {mean_coh:.2f} is below the healthy threshold of "
                       f"{TYPICAL_RANGES['coherence']['healthy_min']}. This indicates signals "
                       f"are not well-coupled. The system may be fragmented or unstable.",
                evidence={'mean_coherence': mean_coh, 'threshold': TYPICAL_RANGES['coherence']['healthy_min']},
                signals_involved=[],
                action="Investigate what's causing signal decoupling. Check for sensor issues, "
                       "process changes, or control loop problems.",
            ))

        # Danger zone coherence
        if min_coh < TYPICAL_RANGES['coherence']['danger']:
            self.findings.append(Finding(
                type=FindingType.THRESHOLD,
                severity=Severity.CRITICAL,
                headline=f"Coherence dropped to danger zone ({min_coh:.2f})",
                detail=f"Coherence fell to {min_coh:.2f}, below the danger threshold of "
                       f"{TYPICAL_RANGES['coherence']['danger']}. This level of decoupling "
                       f"often precedes failures or major regime changes.",
                evidence={'min_coherence': min_coh, 'danger_threshold': TYPICAL_RANGES['coherence']['danger']},
                signals_involved=[],
                action="URGENT: Determine when and why coherence collapsed. Identify which signal diverged first.",
            ))

        # High coherence volatility
        if std_coh > 0.15:
            self.findings.append(Finding(
                type=FindingType.PATTERN,
                severity=Severity.NOTABLE,
                headline=f"Coherence is unstable (std={std_coh:.2f})",
                detail=f"Coherence varies significantly over time (range: {min_coh:.2f} to {max_coh:.2f}). "
                       f"Unstable coupling often indicates a system under stress or transitioning between states.",
                evidence={'coherence_std': std_coh, 'min': min_coh, 'max': max_coh},
                signals_involved=[],
                action="Look at when coherence is high vs low. What's different about those periods?",
            ))

        # Detected transitions
        for t in transitions:
            severity = Severity.CRITICAL if t.get('to_coherence', 1) < 0.3 else Severity.IMPORTANT

            self.findings.append(Finding(
                type=FindingType.TRANSITION,
                severity=severity,
                headline=f"System transition at t={t.get('time', '?')}",
                detail=f"Coherence {'dropped' if t.get('type') == 'drop' else 'rose'} from "
                       f"{t.get('from_coherence', 0):.2f} to {t.get('to_coherence', 0):.2f} at sample {t.get('time', '?')}. "
                       f"This lasted {t.get('duration', 'unknown')} samples. "
                       f"Confidence: {t.get('confidence', 0):.0%}",
                evidence=t,
                signals_involved=[],
                action=f"Investigate what happened at t={t.get('time', '?')}. Check logs, events, interventions. "
                       f"Identify which signal changed first.",
            ))

    # -------------------------------------------------------------------------
    # DETECTOR: Causal Structure
    # -------------------------------------------------------------------------

    def _detect_causal_structure(self):
        """Identify important causal relationships."""

        drivers = self.mechanics.get('drivers', [])
        followers = self.mechanics.get('followers', [])
        top_links = self.mechanics.get('top_links', [])
        causal_density = self.mechanics.get('causal_density', 0)

        # Clear driver identified
        if len(drivers) == 1:
            driver = drivers[0]
            n_caused = sum(1 for link in top_links if link.get('source') == driver)

            self.findings.append(Finding(
                type=FindingType.RELATIONSHIP,
                severity=Severity.IMPORTANT,
                headline=f"{driver} is the primary system driver",
                detail=f"{driver} Granger-causes {n_caused} other signals. It's a net exporter of "
                       f"information, meaning changes here propagate outward. This is your "
                       f"leading indicator.",
                evidence={'driver': driver, 'n_caused': n_caused},
                signals_involved=[driver],
                action=f"Monitor {driver} for early warning. Changes here will affect the system "
                       f"with some lead time.",
            ))

        elif len(drivers) > 1:
            self.findings.append(Finding(
                type=FindingType.RELATIONSHIP,
                severity=Severity.NOTABLE,
                headline=f"Multiple drivers: {', '.join(drivers)}",
                detail=f"The system has {len(drivers)} driver signals. This indicates either "
                       f"multiple independent processes or a distributed causal structure.",
                evidence={'drivers': drivers},
                signals_involved=drivers,
                action="Monitor all drivers. Consider whether they represent different subsystems.",
            ))

        # Strongest causal link
        if top_links:
            link = top_links[0]
            if link.get('granger_f', 0) > 10 or link.get('transfer_entropy', 0) > 0.15:
                self.findings.append(Finding(
                    type=FindingType.RELATIONSHIP,
                    severity=Severity.NOTABLE,
                    headline=f"Strong causality: {link.get('source', '?')} → {link.get('target', '?')}",
                    detail=f"{link.get('source', '?')} strongly predicts {link.get('target', '?')} "
                           f"(Granger F={link.get('granger_f', 0):.1f}, p={link.get('granger_p', 1):.4f}, "
                           f"TE={link.get('transfer_entropy', 0):.3f} bits). "
                           f"Lead time: ~{link.get('lag', '?')} samples.",
                    evidence=link,
                    signals_involved=[link.get('source', ''), link.get('target', '')],
                    action=f"Use {link.get('source', '?')} to forecast {link.get('target', '?')}. "
                           f"Check if this relationship is stable over time.",
                ))

        # Sparse causal structure
        if causal_density < 0.15 and len(top_links) > 0:
            self.findings.append(Finding(
                type=FindingType.PATTERN,
                severity=Severity.INFO,
                headline="Sparse causal structure",
                detail=f"Only {causal_density:.0%} of possible causal links are significant. "
                       f"Signals are relatively independent.",
                evidence={'causal_density': causal_density},
                signals_involved=[],
                action="Independent signals may indicate parallel processes or lack of coupling.",
            ))

    # -------------------------------------------------------------------------
    # DETECTOR: Group Anomalies
    # -------------------------------------------------------------------------

    def _detect_group_anomalies(self):
        """Find interesting patterns in signal groupings."""

        n_clusters = self.groups.get('n_clusters', 1)
        silhouette = self.groups.get('silhouette', 0)
        clusters = self.groups.get('clusters', [])

        # Very strong clustering
        if silhouette > 0.75 and n_clusters > 1:
            self.findings.append(Finding(
                type=FindingType.PATTERN,
                severity=Severity.NOTABLE,
                headline=f"Strong signal grouping ({n_clusters} distinct clusters)",
                detail=f"Signals form {n_clusters} well-separated groups (silhouette={silhouette:.2f}). "
                       f"This indicates clear behavioral differences between groups.",
                evidence={'n_clusters': n_clusters, 'silhouette': silhouette},
                signals_involved=[],
                action="Analyze groups separately. They may represent different subsystems or processes.",
            ))

        # Weak clustering (everything similar)
        if silhouette < 0.3 and n_clusters > 1:
            self.findings.append(Finding(
                type=FindingType.INFO,
                severity=Severity.INFO,
                headline="Weak signal grouping",
                detail=f"Cluster structure is weak (silhouette={silhouette:.2f}). "
                       f"Signals are relatively similar to each other.",
                evidence={'silhouette': silhouette},
                signals_involved=[],
                action="Groups may not be meaningful. Consider treating all signals as one cohort.",
            ))

        # Imbalanced clusters
        if clusters:
            sizes = [len(c.get('members', [])) for c in clusters]
            if sizes and max(sizes) > 3 * min(sizes) and min(sizes) <= 2:
                small_cluster = next((c for c in clusters if len(c.get('members', [])) == min(sizes)), None)

                if small_cluster:
                    members = small_cluster.get('members', [])
                    self.findings.append(Finding(
                        type=FindingType.OUTLIER,
                        severity=Severity.NOTABLE,
                        headline=f"Isolated signals: {', '.join(members)}",
                        detail=f"These {len(members)} signal(s) don't fit with others. "
                               f"Dominant trait: {small_cluster.get('dominant_trait', 'unknown')}.",
                        evidence={'isolated': members, 'sizes': sizes},
                        signals_involved=members,
                        action="Investigate why these signals are different. Possible sensor issues or unique processes.",
                    ))

    # -------------------------------------------------------------------------
    # DETECTOR: Hidden Relationships
    # -------------------------------------------------------------------------

    def _detect_hidden_relationships(self):
        """Find non-obvious relationships revealed by analysis."""

        top_links = self.mechanics.get('top_links', [])

        # Indirect causality (A→B→C but no A→C)
        sources = set(l.get('source', '') for l in top_links)
        targets = set(l.get('target', '') for l in top_links)

        # Find chains
        for link1 in top_links:
            for link2 in top_links:
                if link1.get('target') == link2.get('source'):  # Chain found
                    a, b, c = link1.get('source'), link1.get('target'), link2.get('target')

                    # Check if direct A→C link exists
                    direct = any(l.get('source') == a and l.get('target') == c for l in top_links)

                    if not direct:
                        self.findings.append(Finding(
                            type=FindingType.HIDDEN,
                            severity=Severity.NOTABLE,
                            headline=f"Hidden causal chain: {a} → {b} → {c}",
                            detail=f"Information flows from {a} to {c} through {b}. "
                                   f"There's no direct link from {a} to {c}, but {b} mediates "
                                   f"the relationship. {b} is a bottleneck.",
                            evidence={'chain': [a, b, c]},
                            signals_involved=[a, b, c],
                            action=f"If you need to affect {c}, monitor or control {b}. "
                                   f"Blocking {b} may isolate {a} from {c}.",
                        ))
                        break

    # -------------------------------------------------------------------------
    # DETECTOR: Danger Patterns
    # -------------------------------------------------------------------------

    def _detect_danger_patterns(self):
        """Match known pre-failure signatures."""

        if self.typology.empty:
            return

        # Check coherence trend
        transitions = self.dynamics.get('transitions', [])
        coherence_falling = any(t.get('type') == 'drop' for t in transitions)

        for pattern in DANGER_PATTERNS:
            if pattern.get('requires_history'):
                continue  # Skip patterns that need baseline comparison

            if pattern.get('coherence_trend') == 'falling' and not coherence_falling:
                continue

            # Check if any signal matches the conditions
            for _, row in self.typology.iterrows():
                matches_all = True
                for axis, op, thresh in pattern['conditions']:
                    if axis not in row:
                        matches_all = False
                        break
                    val = row[axis]
                    if op == '>' and val <= thresh:
                        matches_all = False
                        break
                    if op == '<' and val >= thresh:
                        matches_all = False
                        break

                if matches_all:
                    signal_id = row.get('signal_id', row.get('signal', 'unknown'))
                    self.findings.append(Finding(
                        type=FindingType.PATTERN,
                        severity=Severity.CRITICAL,
                        headline=f"⚠️ {signal_id} shows {pattern['name']} pattern",
                        detail=f"{pattern['interpretation']}",
                        evidence={'signal': signal_id, 'pattern': pattern['name']},
                        signals_involved=[signal_id],
                        action="Review this signal immediately. Compare to historical failures if available.",
                    ))

    # -------------------------------------------------------------------------
    # DETECTOR: Contradictions
    # -------------------------------------------------------------------------

    def _detect_contradictions(self):
        """Find things that don't make sense together."""

        # Driver is also a follower? (bidirectional causality)
        drivers = set(self.mechanics.get('drivers', []))
        followers = set(self.mechanics.get('followers', []))

        overlap = drivers & followers
        if overlap:
            self.findings.append(Finding(
                type=FindingType.CONTRADICTION,
                severity=Severity.NOTABLE,
                headline=f"Feedback loop detected: {', '.join(overlap)}",
                detail=f"These signal(s) both drive and are driven by others. "
                       f"This indicates feedback loops or bidirectional causality.",
                evidence={'signals': list(overlap)},
                signals_involved=list(overlap),
                action="Analyze these signals carefully. Feedback loops can amplify disturbances.",
            ))

        # High coherence but sparse causality? (correlated but not causal)
        mean_coh = self.dynamics.get('mean_coherence', 0.5)
        causal_density = self.mechanics.get('causal_density', 0)

        if mean_coh > 0.7 and causal_density < 0.15:
            self.findings.append(Finding(
                type=FindingType.CONTRADICTION,
                severity=Severity.NOTABLE,
                headline="High coherence but sparse causality",
                detail=f"Signals move together (coherence={mean_coh:.2f}) but don't predict each other "
                       f"(causal density={causal_density:.0%}). They may share a common driver "
                       f"that isn't in your dataset.",
                evidence={'coherence': mean_coh, 'causal_density': causal_density},
                signals_involved=[],
                action="Look for an external driver not captured in your signals (temperature, time, input variable).",
            ))


# =============================================================================
# SMART REPORT GENERATOR
# =============================================================================

def generate_smart_report(results: dict) -> dict:
    """
    Generate a smart report from ORTHON results.

    Returns dict with:
    - findings: List of Finding objects
    - summary: One paragraph executive summary
    - prompt_context: Formatted string for Claude
    """

    engine = FindingsEngine(results)
    findings = engine.analyze()

    # Generate executive summary
    critical = [f for f in findings if f.severity == Severity.CRITICAL]
    important = [f for f in findings if f.severity == Severity.IMPORTANT]

    if critical:
        urgency = "URGENT: "
        top_issue = critical[0].headline
    elif important:
        urgency = ""
        top_issue = important[0].headline
    else:
        urgency = ""
        top_issue = "No critical issues detected"

    n_signals = results.get('metadata', {}).get('n_signals', 0)
    n_groups = results.get('groups', {}).get('n_clusters', 1)
    coherence = results.get('dynamics', {}).get('mean_coherence', 0)
    drivers = results.get('mechanics', {}).get('drivers', [])

    summary = (
        f"{urgency}{top_issue}. "
        f"Analyzed {n_signals} signals forming {n_groups} behavioral group(s). "
        f"System coherence: {coherence:.2f}. "
        f"{'Primary driver: ' + drivers[0] if drivers else 'No clear driver identified'}. "
        f"Found {len(critical)} critical and {len(important)} important findings."
    )

    return {
        'findings': findings,
        'summary': summary,
        'prompt_context': engine.to_prompt(),
        'engine': engine,
    }


# =============================================================================
# CLAUDE INTEGRATION
# =============================================================================

def build_smart_prompt(results: dict) -> str:
    """
    Build a Claude prompt that includes pre-analyzed findings.

    This is what makes Claude sound like an expert —
    we've already done the detective work.
    """

    report = generate_smart_report(results)

    prompt = f"""
Analyze these ORTHON signal processing results. I've pre-identified the key findings —
explain them and add any additional insights.

DATASET: {results.get('metadata', {}).get('name', 'Unknown')}
- {results.get('metadata', {}).get('n_signals', 0)} signals
- {results.get('metadata', {}).get('n_samples', 0)} samples

EXECUTIVE SUMMARY:
{report['summary']}

{report['prompt_context']}

---

RAW METRICS (for reference):
- Mean coherence: {results.get('dynamics', {}).get('mean_coherence', 0):.2f}
- Groups: {results.get('groups', {}).get('n_clusters', 'N/A')}
- Causal density: {results.get('mechanics', {}).get('causal_density', 0):.2f}
- Drivers: {', '.join(results.get('mechanics', {}).get('drivers', [])) or 'None'}

Based on these findings, provide:
1. A clear explanation of the system state
2. What the findings mean practically
3. Specific recommendations for action
4. Any additional patterns you notice in the raw metrics

Be specific. Use signal names. Reference the findings by number.
"""

    return prompt
