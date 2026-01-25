"""
Signal Typology HTML Report Generator
=====================================

Generates standalone HTML reports for signal typology analysis.

Features:
- 6-axis radar chart visualization
- Archetype classification with confidence badge
- Axis detail cards with classifications
- Discontinuity detection summary
- Regime transition timeline (for windowed analysis)

Usage:
    from prism.typology import analyze_signal, typology_to_html, typologies_to_html

    # Single typology report
    typology = analyze_signal(series)
    html = typology_to_html(typology)

    # Save to file
    typology_to_html_file(typology, "report.html")

    # Windowed analysis report
    typologies = analyze_windowed(series, window_size=50, step_size=10)
    html = typologies_to_html(typologies)
"""

import math
from typing import List, Optional
from datetime import datetime

from .models import SignalTypology, TransitionType


# =============================================================================
# STYLES
# =============================================================================

CSS_STYLES = """
:root {
    --bg-primary: #0a1628;
    --bg-secondary: #0f1d32;
    --bg-card: #152238;
    --text-primary: #f0f4f8;
    --text-secondary: #d1d9e6;
    --text-muted: #94a3b8;
    --accent-cyan: #22d3ee;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-yellow: #eab308;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --border-color: #1e3a5f;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 1.5rem;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
}

h3 {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.header {
    margin-bottom: 2rem;
}

.header-meta {
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* Archetype Badge */
.archetype-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}

.archetype-badge {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
    padding: 1rem 2rem;
    border-radius: 6px;
    text-align: center;
}

.archetype-name {
    font-size: 1.25rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.archetype-confidence {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.25rem;
}

.archetype-details {
    flex: 1;
}

.archetype-meta {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}

.meta-item {
    font-size: 0.8rem;
}

.meta-label {
    color: var(--text-muted);
}

.meta-value {
    color: var(--text-primary);
    font-weight: 500;
}

/* Radar Chart */
.radar-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.radar-container {
    display: flex;
    justify-content: center;
    align-items: center;
}

.radar-chart {
    width: 350px;
    height: 350px;
}

/* Axis Cards */
.axes-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.axis-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1rem;
}

.axis-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.axis-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--accent-cyan);
}

.axis-class {
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: var(--bg-secondary);
    color: var(--text-secondary);
}

.axis-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.axis-bar {
    height: 4px;
    background: var(--bg-secondary);
    border-radius: 2px;
    overflow: hidden;
}

.axis-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

.axis-metrics {
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Discontinuity Section */
.discontinuity-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.discontinuity-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

.discontinuity-card {
    background: var(--bg-secondary);
    border-radius: 6px;
    padding: 1rem;
}

.discontinuity-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.discontinuity-icon {
    font-size: 1.25rem;
}

.discontinuity-name {
    font-weight: 600;
}

.discontinuity-status {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.detected {
    color: var(--accent-yellow);
}

.not-detected {
    color: var(--text-muted);
}

/* Timeline */
.timeline-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.timeline {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.timeline-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem;
    background: var(--bg-secondary);
    border-radius: 4px;
    font-size: 0.85rem;
}

.timeline-window {
    width: 60px;
    color: var(--text-muted);
}

.timeline-archetype {
    flex: 1;
    font-weight: 500;
}

.timeline-confidence {
    width: 50px;
    text-align: right;
    color: var(--text-secondary);
}

.timeline-transition {
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
}

.transition-none {
    background: transparent;
}

.transition-approaching {
    background: var(--accent-yellow);
    color: black;
}

.transition-in_progress {
    background: var(--accent-red);
    color: white;
}

.transition-completed {
    background: var(--accent-green);
    color: black;
}

/* Summary */
.summary-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
}

.summary-text {
    font-size: 0.9rem;
    color: var(--text-secondary);
    white-space: pre-wrap;
}

/* Fingerprint */
.fingerprint {
    font-family: monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 1rem;
    padding: 0.5rem;
    background: var(--bg-secondary);
    border-radius: 4px;
}

@media (max-width: 768px) {
    .axes-grid {
        grid-template-columns: 1fr;
    }

    .archetype-section {
        flex-direction: column;
        text-align: center;
    }

    .archetype-meta {
        grid-template-columns: 1fr;
    }
}
"""


# =============================================================================
# RADAR CHART SVG GENERATION
# =============================================================================

def generate_radar_svg(fingerprint: List[float], labels: List[str] = None) -> str:
    """Generate SVG radar chart for 6D fingerprint."""

    if labels is None:
        labels = ["Memory", "Information", "Recurrence", "Volatility", "Frequency", "Dynamics"]

    # SVG dimensions
    width, height = 350, 350
    cx, cy = width // 2, height // 2
    max_radius = 120

    n = len(fingerprint)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    # Generate grid lines
    grid_lines = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        r = max_radius * level
        points = []
        for angle in angles:
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")
        grid_lines.append(f'<polygon points="{" ".join(points)}" fill="none" stroke="#27272a" stroke-width="1"/>')

    # Generate axis lines
    axis_lines = []
    for angle in angles:
        x = cx + max_radius * math.cos(angle)
        y = cy + max_radius * math.sin(angle)
        axis_lines.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#27272a" stroke-width="1"/>')

    # Generate data polygon
    data_points = []
    for i, val in enumerate(fingerprint):
        r = max_radius * min(max(val, 0), 1)  # Clamp to [0, 1]
        x = cx + r * math.cos(angles[i])
        y = cy + r * math.sin(angles[i])
        data_points.append(f"{x:.1f},{y:.1f}")

    data_polygon = f'<polygon points="{" ".join(data_points)}" fill="rgba(34, 211, 238, 0.2)" stroke="#22d3ee" stroke-width="2"/>'

    # Generate data points
    data_circles = []
    for i, val in enumerate(fingerprint):
        r = max_radius * min(max(val, 0), 1)
        x = cx + r * math.cos(angles[i])
        y = cy + r * math.sin(angles[i])
        data_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#22d3ee"/>')

    # Generate labels
    label_elements = []
    label_radius = max_radius + 25
    for i, label in enumerate(labels):
        x = cx + label_radius * math.cos(angles[i])
        y = cy + label_radius * math.sin(angles[i])

        # Adjust text anchor based on position
        if angles[i] > -0.1 and angles[i] < 0.1:
            anchor = "middle"
        elif angles[i] > 0:
            anchor = "start" if math.cos(angles[i]) > 0 else "end"
        else:
            anchor = "start" if math.cos(angles[i]) > 0 else "end"

        # Adjust for top position
        if i == 0:
            anchor = "middle"
            y -= 5

        label_elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'fill="#a1a1aa" font-size="11" font-family="SF Mono, monospace">{label}</text>'
        )

        # Add value label
        val_str = f"{fingerprint[i]:.2f}"
        label_elements.append(
            f'<text x="{x:.1f}" y="{y + 12:.1f}" text-anchor="{anchor}" '
            f'fill="#71717a" font-size="9" font-family="SF Mono, monospace">{val_str}</text>'
        )

    svg = f'''<svg viewBox="0 0 {width} {height}" class="radar-chart">
        {"".join(grid_lines)}
        {"".join(axis_lines)}
        {data_polygon}
        {"".join(data_circles)}
        {"".join(label_elements)}
    </svg>'''

    return svg


# =============================================================================
# AXIS COLOR MAPPING
# =============================================================================

def get_axis_color(axis_name: str) -> str:
    """Get color for axis visualization."""
    colors = {
        "memory": "#22d3ee",      # cyan
        "information": "#a855f7",  # purple
        "recurrence": "#22c55e",   # green
        "volatility": "#eab308",   # yellow
        "frequency": "#3b82f6",    # blue
        "dynamics": "#ef4444",     # red
    }
    return colors.get(axis_name.lower(), "#71717a")


# =============================================================================
# SINGLE TYPOLOGY HTML
# =============================================================================

def typology_to_html(typology: SignalTypology, title: str = None) -> str:
    """
    Generate standalone HTML report for a single SignalTypology.

    Args:
        typology: SignalTypology object to render
        title: Optional title override

    Returns:
        Complete HTML document as string
    """

    if title is None:
        title = f"Signal Typology: {typology.entity_id} / {typology.signal_id}"

    # Generate radar chart
    fingerprint = list(typology.fingerprint) if typology.fingerprint is not None else [0.5] * 6
    radar_svg = generate_radar_svg(fingerprint)

    # Build axis cards
    axes_html = ""
    axes_data = [
        ("Memory", typology.memory, typology.memory.hurst_exponent,
         f"ACF decay: {typology.memory.acf_decay_type.value}", get_axis_color("memory")),
        ("Information", typology.information, typology.information.entropy_permutation,
         f"Sample entropy: {typology.information.entropy_sample:.3f}", get_axis_color("information")),
        ("Recurrence", typology.recurrence, typology.recurrence.determinism,
         f"Laminarity: {typology.recurrence.laminarity:.3f}", get_axis_color("recurrence")),
        ("Volatility", typology.volatility, typology.volatility.garch_persistence,
         f"Hilbert amp std: {typology.volatility.hilbert_amplitude_std:.3f}", get_axis_color("volatility")),
        ("Frequency", typology.frequency, typology.frequency.spectral_bandwidth,
         f"Centroid: {typology.frequency.spectral_centroid:.3f}", get_axis_color("frequency")),
        ("Dynamics", typology.dynamics, typology.dynamics.lyapunov_exponent,
         f"Embed dim: {typology.dynamics.embedding_dimension}", get_axis_color("dynamics")),
    ]

    for name, axis, value, extra_metric, color in axes_data:
        # Get class name from axis
        class_attr = getattr(axis, f"{name.lower()}_class", None)
        class_name = class_attr.value if class_attr else "unknown"

        # Normalize value for bar (handle negative Lyapunov)
        bar_value = min(max((value + 1) / 2 if value < 0 else value, 0), 1) * 100

        axes_html += f'''
        <div class="axis-card">
            <div class="axis-header">
                <span class="axis-name" style="color: {color}">{name}</span>
                <span class="axis-class">{class_name}</span>
            </div>
            <div class="axis-value">{value:.3f}</div>
            <div class="axis-bar">
                <div class="axis-bar-fill" style="width: {bar_value}%; background: {color}"></div>
            </div>
            <div class="axis-metrics">{extra_metric}</div>
        </div>
        '''

    # Discontinuity section
    dirac_status = "detected" if typology.discontinuity.dirac.detected else "not-detected"
    heaviside_status = "detected" if typology.discontinuity.heaviside.detected else "not-detected"

    discontinuity_html = f'''
    <div class="discontinuity-grid">
        <div class="discontinuity-card">
            <div class="discontinuity-header">
                <span class="discontinuity-icon">&#x3B4;</span>
                <span class="discontinuity-name">Dirac (Impulse)</span>
            </div>
            <div class="discontinuity-status {dirac_status}">
                {"Detected" if typology.discontinuity.dirac.detected else "Not detected"}
                | Count: {typology.discontinuity.dirac.count}
                | Max magnitude: {typology.discontinuity.dirac.max_magnitude:.2f}
            </div>
        </div>
        <div class="discontinuity-card">
            <div class="discontinuity-header">
                <span class="discontinuity-icon">H</span>
                <span class="discontinuity-name">Heaviside (Step)</span>
            </div>
            <div class="discontinuity-status {heaviside_status}">
                {"Detected" if typology.discontinuity.heaviside.detected else "Not detected"}
                | Count: {typology.discontinuity.heaviside.count}
                | Max magnitude: {typology.discontinuity.heaviside.max_magnitude:.2f}
            </div>
        </div>
    </div>
    '''

    # Fingerprint display
    fp_str = ", ".join([f"{v:.3f}" for v in fingerprint])

    # Build full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Signal Typology Report</h1>
            <div class="header-meta">
                Entity: {typology.entity_id} | Signal: {typology.signal_id} |
                Observations: {typology.n_observations} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>

        <div class="archetype-section">
            <div class="archetype-badge">
                <div class="archetype-name">{typology.archetype}</div>
                <div class="archetype-confidence">{typology.confidence:.0%} confidence</div>
            </div>
            <div class="archetype-details">
                <div class="archetype-meta">
                    <div class="meta-item">
                        <div class="meta-label">Secondary</div>
                        <div class="meta-value">{typology.secondary_archetype}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Boundary Proximity</div>
                        <div class="meta-value">{typology.boundary_proximity:.0%}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Transition</div>
                        <div class="meta-value">{typology.regime_transition.value}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="radar-section">
            <h2>6D Fingerprint</h2>
            <div class="radar-container">
                {radar_svg}
            </div>
            <div class="fingerprint">Fingerprint: [{fp_str}]</div>
        </div>

        <h2>Axis Measurements</h2>
        <div class="axes-grid">
            {axes_html}
        </div>

        <div class="discontinuity-section">
            <h2>Structural Discontinuity Detection</h2>
            {discontinuity_html}
        </div>

        <div class="summary-section">
            <h2>Summary</h2>
            <div class="summary-text">{typology.summary}</div>
        </div>
    </div>
</body>
</html>'''

    return html


# =============================================================================
# WINDOWED TYPOLOGIES HTML
# =============================================================================

def typologies_to_html(
    typologies: List[SignalTypology],
    title: str = None,
    show_all_windows: bool = False
) -> str:
    """
    Generate HTML report for windowed analysis with timeline.

    Args:
        typologies: List of SignalTypology objects from windowed analysis
        title: Optional title override
        show_all_windows: If True, show all windows in timeline; if False, only show transitions

    Returns:
        Complete HTML document as string
    """

    if not typologies:
        return "<html><body>No typologies to display</body></html>"

    first = typologies[0]
    last = typologies[-1]

    if title is None:
        title = f"Windowed Typology: {first.entity_id} / {first.signal_id}"

    # Generate radar chart for last window
    fingerprint = list(last.fingerprint) if last.fingerprint is not None else [0.5] * 6
    radar_svg = generate_radar_svg(fingerprint)

    # Build timeline
    timeline_html = ""
    prev_archetype = None

    for i, t in enumerate(typologies):
        # Determine if we should show this window
        is_transition = (t.archetype != prev_archetype) or (t.regime_transition != TransitionType.NONE)

        if show_all_windows or is_transition or i == 0 or i == len(typologies) - 1:
            transition_class = f"transition-{t.regime_transition.value.replace(' ', '_')}"
            transition_text = t.regime_transition.value if t.regime_transition != TransitionType.NONE else ""

            change_marker = ""
            if prev_archetype and t.archetype != prev_archetype:
                change_marker = f' <span style="color: var(--accent-yellow)">&larr; changed</span>'

            timeline_html += f'''
            <div class="timeline-item">
                <span class="timeline-window">Win {i}</span>
                <span class="timeline-archetype">{t.archetype}{change_marker}</span>
                <span class="timeline-confidence">{t.confidence:.0%}</span>
                <span class="timeline-transition {transition_class}">{transition_text}</span>
            </div>
            '''

        prev_archetype = t.archetype

    # Count regime changes
    regime_changes = sum(1 for i in range(1, len(typologies))
                        if typologies[i].archetype != typologies[i-1].archetype)

    # Archetype distribution
    from collections import Counter
    archetype_counts = Counter(t.archetype for t in typologies)
    dominant_archetype = archetype_counts.most_common(1)[0][0]

    # Build axis cards for final window
    axes_html = ""
    axes_data = [
        ("Memory", last.memory, last.memory.hurst_exponent, get_axis_color("memory")),
        ("Information", last.information, last.information.entropy_permutation, get_axis_color("information")),
        ("Recurrence", last.recurrence, last.recurrence.determinism, get_axis_color("recurrence")),
        ("Volatility", last.volatility, last.volatility.garch_persistence, get_axis_color("volatility")),
        ("Frequency", last.frequency, last.frequency.spectral_bandwidth, get_axis_color("frequency")),
        ("Dynamics", last.dynamics, last.dynamics.lyapunov_exponent, get_axis_color("dynamics")),
    ]

    for name, axis, value, color in axes_data:
        class_attr = getattr(axis, f"{name.lower()}_class", None)
        class_name = class_attr.value if class_attr else "unknown"
        bar_value = min(max((value + 1) / 2 if value < 0 else value, 0), 1) * 100

        axes_html += f'''
        <div class="axis-card">
            <div class="axis-header">
                <span class="axis-name" style="color: {color}">{name}</span>
                <span class="axis-class">{class_name}</span>
            </div>
            <div class="axis-value">{value:.3f}</div>
            <div class="axis-bar">
                <div class="axis-bar-fill" style="width: {bar_value}%; background: {color}"></div>
            </div>
        </div>
        '''

    fp_str = ", ".join([f"{v:.3f}" for v in fingerprint])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Windowed Typology Report</h1>
            <div class="header-meta">
                Entity: {first.entity_id} | Signal: {first.signal_id} |
                Windows: {len(typologies)} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>

        <div class="archetype-section">
            <div class="archetype-badge">
                <div class="archetype-name">{dominant_archetype}</div>
                <div class="archetype-confidence">Dominant ({archetype_counts[dominant_archetype]}/{len(typologies)} windows)</div>
            </div>
            <div class="archetype-details">
                <div class="archetype-meta">
                    <div class="meta-item">
                        <div class="meta-label">Final Archetype</div>
                        <div class="meta-value">{last.archetype}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Regime Changes</div>
                        <div class="meta-value">{regime_changes}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Unique Archetypes</div>
                        <div class="meta-value">{len(archetype_counts)}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="timeline-section">
            <h2>Regime Timeline</h2>
            <div class="timeline">
                {timeline_html}
            </div>
        </div>

        <div class="radar-section">
            <h2>Final Window Fingerprint</h2>
            <div class="radar-container">
                {radar_svg}
            </div>
            <div class="fingerprint">Fingerprint: [{fp_str}]</div>
        </div>

        <h2>Final Window Axes</h2>
        <div class="axes-grid">
            {axes_html}
        </div>

        <div class="summary-section">
            <h2>Final Window Summary</h2>
            <div class="summary-text">{last.summary}</div>
        </div>
    </div>
</body>
</html>'''

    return html


# =============================================================================
# FILE OUTPUT
# =============================================================================

def typology_to_html_file(typology: SignalTypology, filepath: str, title: str = None):
    """
    Save single typology HTML report to file.

    Args:
        typology: SignalTypology object
        filepath: Output file path
        title: Optional title override
    """
    html = typology_to_html(typology, title)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)


def typologies_to_html_file(
    typologies: List[SignalTypology],
    filepath: str,
    title: str = None,
    show_all_windows: bool = False
):
    """
    Save windowed typologies HTML report to file.

    Args:
        typologies: List of SignalTypology objects
        filepath: Output file path
        title: Optional title override
        show_all_windows: If True, show all windows in timeline
    """
    html = typologies_to_html(typologies, title, show_all_windows)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
