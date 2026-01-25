#!/usr/bin/env python3
"""
ORTHON Signal Analysis Platform
===============================

Streamlit-based interface for the ORTHON four-framework analytical system.

Frameworks:
    1. Signal Typology     → What IS this signal?
    2. Structural Geometry → What is its STRUCTURE?
    3. Dynamical Systems   → How does the SYSTEM evolve?
    4. Causal Mechanics    → What DRIVES the system?

Usage:
    streamlit run orthon/app.py
"""

import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="ORTHON Signal Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4aa;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .axis-label {
        font-size: 0.8rem;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


def load_data(data_dir: Path):
    """Load all available data files."""
    data = {}

    # Observations
    obs_path = data_dir / "observations.parquet"
    if obs_path.exists():
        data['observations'] = pl.read_parquet(obs_path)

    # Signal Typology
    metrics_path = data_dir / "signal_typology_metrics.parquet"
    profile_path = data_dir / "signal_typology_profile.parquet"
    if metrics_path.exists():
        data['typology_metrics'] = pl.read_parquet(metrics_path)
    if profile_path.exists():
        data['typology_profile'] = pl.read_parquet(profile_path)

    return data


def render_radar_chart(profile_row: dict, signal_id: str):
    """Render a plotly radar chart for the signal profile."""
    import plotly.graph_objects as go

    axes = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
            'derivatives', 'recurrence', 'discontinuity', 'momentum']

    values = [profile_row.get(ax, 0.5) for ax in axes]
    values.append(values[0])  # Close the polygon

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=axes + [axes[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.3)',
        line=dict(color='#00d4aa', width=2),
        name=signal_id
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#888'),
                gridcolor='#333'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#ccc'),
                gridcolor='#333'
            ),
            bgcolor='#0e1117'
        ),
        showlegend=False,
        paper_bgcolor='#0e1117',
        margin=dict(l=60, r=60, t=40, b=40),
        height=350
    )

    return fig


def render_comparison_radar(profiles: list, signal_ids: list):
    """Render comparison radar chart for multiple signals."""
    import plotly.graph_objects as go

    axes = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
            'derivatives', 'recurrence', 'discontinuity', 'momentum']

    colors = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#ffd93d', '#6bcb77', '#9b5de5']

    fig = go.Figure()

    for i, (profile, signal_id) in enumerate(zip(profiles, signal_ids)):
        values = [profile.get(ax, 0.5) for ax in axes]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=axes + [axes[0]],
            fill='toself',
            fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, '
                      f'{int(colors[i % len(colors)][3:5], 16)}, '
                      f'{int(colors[i % len(colors)][5:7], 16)}, 0.2)',
            line=dict(color=colors[i % len(colors)], width=2),
            name=signal_id
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#888'),
                gridcolor='#333'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#ccc'),
                gridcolor='#333'
            ),
            bgcolor='#0e1117'
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#ccc'),
            bgcolor='#0e1117'
        ),
        paper_bgcolor='#0e1117',
        margin=dict(l=60, r=60, t=40, b=40),
        height=400
    )

    return fig


def render_axis_bars(profile: dict):
    """Render horizontal axis bars with pole labels."""
    from prism.signal_typology import AXIS_POLES

    axes = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
            'derivatives', 'recurrence', 'discontinuity', 'momentum']

    for axis in axes:
        score = profile.get(axis, 0.5)
        low, high = AXIS_POLES.get(axis, ('Low', 'High'))

        col1, col2, col3 = st.columns([1.5, 4, 1.5])

        with col1:
            st.markdown(f"<span style='color: #888; font-size: 0.85rem;'>{low}</span>",
                       unsafe_allow_html=True)

        with col2:
            st.progress(score)
            st.caption(f"{axis.title()}: {score:.3f}")

        with col3:
            st.markdown(f"<span style='color: #888; font-size: 0.85rem;'>{high}</span>",
                       unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">ORTHON</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Signal Typology Analysis Platform</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Data Source")

    data_dir = Path("data")
    data = load_data(data_dir)

    if 'observations' not in data:
        st.error("No observations.parquet found in data/ directory")
        st.info("Run: python -m fetchers.hydraulic_fetcher")
        return

    df_obs = data['observations']
    st.sidebar.success(f"Loaded {len(df_obs):,} observations")

    # Signal selector
    signals = sorted(df_obs['signal_id'].unique().to_list())

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🧬 Signal Typology",
        "📈 Time Series",
        "🔍 Compare"
    ])

    # =========================================================================
    # TAB 1: Overview
    # =========================================================================
    with tab1:
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Observations", f"{len(df_obs):,}")
        with col2:
            st.metric("Signals", df_obs['signal_id'].n_unique())
        with col3:
            st.metric("Entities", df_obs['entity_id'].n_unique())
        with col4:
            min_date = df_obs['timestamp'].min()
            max_date = df_obs['timestamp'].max()
            n_days = (max_date - min_date).days
            st.metric("Time Span", f"{n_days:,} days")

        st.markdown("---")

        # Signal summary
        st.subheader("Signal Summary")
        summary = (
            df_obs.group_by('signal_id')
            .agg([
                pl.len().alias('count'),
                pl.col('value').mean().alias('mean'),
                pl.col('value').std().alias('std'),
                pl.col('value').min().alias('min'),
                pl.col('value').max().alias('max'),
            ])
            .sort('signal_id')
        )
        st.dataframe(summary.to_pandas(), use_container_width=True, hide_index=True)

        # Typology status
        if 'typology_profile' in data:
            st.success(f"✓ Signal Typology computed for {len(data['typology_profile'])} signals")
        else:
            st.warning("⚠ Signal Typology not computed yet")
            st.code("python -m prism.entry_points.signal_typology", language="bash")

    # =========================================================================
    # TAB 2: Signal Typology
    # =========================================================================
    with tab2:
        st.subheader("Signal Typology Analysis")
        st.markdown("**Principle:** Data = math. Labels = rendering.")

        if 'typology_profile' not in data:
            st.error("Signal Typology not computed. Run the entry point first.")
            st.code("python -m prism.entry_points.signal_typology", language="bash")
            return

        profile_df = data['typology_profile']
        metrics_df = data.get('typology_metrics', None)

        # Signal selector
        typology_signal = st.selectbox(
            "Select Signal",
            profile_df['signal_id'].unique().to_list(),
            key="typology_signal"
        )

        # Get profile for selected signal
        profile_row = profile_df.filter(
            pl.col('signal_id') == typology_signal
        ).to_dicts()[0]

        # Two columns: Radar and Bars
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Radar Profile")
            fig = render_radar_chart(profile_row, typology_signal)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Axis Scores")
            render_axis_bars(profile_row)

        # Classification
        st.markdown("---")
        st.markdown("#### Classification")

        from prism.signal_typology import classify_profile, summarize_profile

        classification = classify_profile(profile_row)
        summary = summarize_profile(profile_row)

        st.info(f"**Summary:** {summary}")

        # Show classification in columns
        cols = st.columns(3)
        class_items = list(classification.items())
        for i, (key, label) in enumerate(class_items):
            axis = key.replace('_class', '')
            score = profile_row.get(axis, 0.5)

            # Color based on extremity
            if score < 0.25 or score > 0.75:
                color = "#00d4aa"  # Strong
            elif score < 0.4 or score > 0.6:
                color = "#aaa"     # Weak
            else:
                color = "#666"     # Indeterminate

            with cols[i % 3]:
                st.markdown(
                    f"<span style='color: {color};'>{axis}: **{label}**</span>",
                    unsafe_allow_html=True
                )

        # Raw metrics expander
        if metrics_df is not None:
            with st.expander("Raw Metrics"):
                metrics_row = metrics_df.filter(
                    pl.col('signal_id') == typology_signal
                ).to_dicts()[0]

                # Filter display columns
                display_metrics = {
                    k: v for k, v in metrics_row.items()
                    if k not in ['signal_id', 'timestamp', 'entity_id']
                }

                cols = st.columns(2)
                items = list(display_metrics.items())
                mid = len(items) // 2

                with cols[0]:
                    for k, v in items[:mid]:
                        if isinstance(v, float):
                            st.text(f"{k}: {v:.6f}")
                        else:
                            st.text(f"{k}: {v}")

                with cols[1]:
                    for k, v in items[mid:]:
                        if isinstance(v, float):
                            st.text(f"{k}: {v:.6f}")
                        else:
                            st.text(f"{k}: {v}")

    # =========================================================================
    # TAB 3: Time Series
    # =========================================================================
    with tab3:
        st.subheader("Time Series Visualization")

        selected_signals = st.multiselect(
            "Select Signals to Visualize",
            signals,
            default=signals[:3] if len(signals) >= 3 else signals
        )

        if not selected_signals:
            st.warning("Select at least one signal")
        else:
            for signal_id in selected_signals:
                signal_data = df_obs.filter(
                    pl.col('signal_id') == signal_id
                ).sort('timestamp')

                if len(signal_data) > 0:
                    st.markdown(f"**{signal_id}**")

                    chart_data = signal_data.select(['timestamp', 'value']).to_pandas()
                    chart_data = chart_data.set_index('timestamp')
                    st.line_chart(chart_data, use_container_width=True, height=200)

                    col1, col2, col3, col4 = st.columns(4)
                    vals = signal_data['value']
                    col1.metric("Mean", f"{vals.mean():.2f}")
                    col2.metric("Std", f"{vals.std():.2f}")
                    col3.metric("Min", f"{vals.min():.2f}")
                    col4.metric("Max", f"{vals.max():.2f}")
                    st.markdown("---")

    # =========================================================================
    # TAB 4: Compare
    # =========================================================================
    with tab4:
        st.subheader("Signal Comparison")

        if 'typology_profile' not in data:
            st.error("Signal Typology not computed")
            return

        profile_df = data['typology_profile']
        available_signals = profile_df['signal_id'].unique().to_list()

        compare_signals = st.multiselect(
            "Select Signals to Compare",
            available_signals,
            default=available_signals[:3] if len(available_signals) >= 3 else available_signals,
            key="compare_signals"
        )

        if len(compare_signals) < 2:
            st.info("Select at least 2 signals to compare")
        else:
            # Get profiles
            profiles = []
            for sig in compare_signals:
                row = profile_df.filter(pl.col('signal_id') == sig).to_dicts()[0]
                profiles.append(row)

            # Comparison radar
            st.markdown("#### Comparison Radar")
            fig = render_comparison_radar(profiles, compare_signals)
            st.plotly_chart(fig, use_container_width=True)

            # Comparison table
            st.markdown("#### Axis Comparison")

            axes = ['memory', 'information', 'frequency', 'volatility', 'wavelet',
                    'derivatives', 'recurrence', 'discontinuity', 'momentum']

            table_data = []
            for axis in axes:
                row = {'Axis': axis.title()}
                for sig, profile in zip(compare_signals, profiles):
                    row[sig] = f"{profile.get(axis, 0.5):.3f}"
                table_data.append(row)

            st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ORTHON Framework**")
    st.sidebar.markdown("""
    1. Signal Typology
    2. Structural Geometry
    3. Dynamical Systems
    4. Causal Mechanics
    """)


if __name__ == "__main__":
    main()
