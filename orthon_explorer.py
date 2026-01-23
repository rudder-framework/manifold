"""
Ørthon Explorer
===============

A Streamlit app for exploring ORTHON four-layer analysis results.

Answers the fundamental questions:
    - WHAT is it? (Signal Typology)
    - HOW does it behave? (Behavioral Geometry)  
    - WHEN/HOW does it change? (Dynamical Systems)
    - WHY does it change? (Causal Mechanics)

Features:
    - Load and explore parquet files from each layer
    - View LaTeX equations for mathematical foundations
    - Ask Claude to explain analysis in plain language

Usage:
    streamlit run orthon_explorer.py
"""

import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Claude API (optional - graceful fallback if not installed)
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Ørthon Explorer",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .tagline {
        font-size: 1.1rem;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    /* Question cards */
    .question-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .what-card { border-left-color: #4CAF50; }
    .how-card { border-left-color: #2196F3; }
    .when-card { border-left-color: #FF9800; }
    .why-card { border-left-color: #9C27B0; }
    
    .question-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Equation boxes */
    .equation-box {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Computer Modern', serif;
    }
    
    /* Interpretation text */
    .interpretation {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    /* Metric display */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MATHEMATICAL EQUATIONS (LaTeX)
# =============================================================================

EQUATIONS = {
    "signal_typology": {
        "title": "Signal Typology — WHAT is it?",
        "description": "Characterizes signals along six orthogonal behavioral axes.",
        "equations": {
            "Memory (Hurst Exponent)": r"H = \frac{\log(R/S)}{\log(n)} \quad \text{where } R/S = \frac{\max(Y_t) - \min(Y_t)}{\sigma}",
            "Periodicity (FFT)": r"\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt",
            "Volatility (GARCH)": r"\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2",
            "Discontinuity (Heaviside)": r"H(t-t_0) = \begin{cases} 0 & t < t_0 \\ 1 & t \geq t_0 \end{cases}",
            "Impulsivity (Dirac)": r"\delta(t-t_0) = \lim_{\epsilon \to 0} \frac{1}{\epsilon} \Pi\left(\frac{t-t_0}{\epsilon}\right)",
            "Complexity (Entropy)": r"S = -\sum_{i} p_i \log(p_i)",
        },
    },
    "behavioral_geometry": {
        "title": "Behavioral Geometry — HOW does it behave?",
        "description": "Analyzes pairwise relationships and network structure.",
        "equations": {
            "Gradient (Velocity)": r"\nabla E(t) = \frac{E(t+1) - E(t-1)}{2}",
            "Laplacian (Acceleration)": r"\nabla^2 E(t) = E(t+1) - 2E(t) + E(t-1)",
            "Divergence": r"\text{div}(E) = \sum_i \frac{\partial^2 E_i}{\partial t^2} \quad \begin{cases} >0 & \text{SOURCE} \\ <0 & \text{SINK} \end{cases}",
            "Field Potential": r"\phi = \int \|\nabla E\| \, dt = \sum_t |\nabla E(t)|",
            "Correlation": r"\rho_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}",
            "Network Density": r"D = \frac{2|E|}{|V|(|V|-1)} \quad \text{where } |E| = \text{edges}, |V| = \text{nodes}",
        },
    },
    "dynamical_systems": {
        "title": "Dynamical Systems — WHEN/HOW does it change?",
        "description": "Tracks regime evolution, stability, and trajectory.",
        "equations": {
            "Hamilton's Equations": r"\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}",
            "Lyapunov Exponent": r"\lambda = \lim_{t \to \infty} \frac{1}{t} \ln\left(\frac{|\delta x(t)|}{|\delta x(0)|}\right)",
            "Recurrence Rate": r"RR = \frac{1}{N^2} \sum_{i,j} R_{i,j} \quad \text{where } R_{i,j} = \Theta(\varepsilon - \|X_i - X_j\|)",
            "Determinism": r"DET = \frac{\sum_{l=l_{min}}^{N} l \cdot P(l)}{\sum_{i,j} R_{i,j}}",
            "Stability Index": r"\sigma = 1 - \frac{|\Delta \rho|}{\max(|\Delta \rho|)}",
            "Trajectory Curvature": r"\kappa = \frac{|x'y'' - y'x''|}{(x'^2 + y'^2)^{3/2}}",
        },
    },
    "causal_mechanics": {
        "title": "Causal Mechanics — WHY does it change?",
        "description": "Physics-inspired analysis of energy, equilibrium, and flow.",
        "equations": {
            "Hamiltonian (Energy)": r"H = T + V = \frac{p^2}{2m} + V(q) \quad \text{(Total Energy)}",
            "Lagrangian (Motion)": r"L = T - V \quad \text{Action: } S = \int L \, dt",
            "Gibbs Free Energy": r"G = H - TS \quad \text{Spontaneous if } \Delta G < 0",
            "Angular Momentum": r"L = r \times p = I\omega",
            "Momentum Flux": r"\Pi_{ij} = \rho v_i v_j + p\delta_{ij} - \tau_{ij}",
            "Conservation": r"\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0",
        },
    },
}


# =============================================================================
# INTERPRETATION TEMPLATES
# =============================================================================

def interpret_signal_typology(row: Dict[str, Any]) -> str:
    """Generate plain-language interpretation of signal typology."""
    parts = []
    
    # Memory interpretation
    memory = row.get("memory", 0)
    if memory > 0.65:
        parts.append("exhibits **persistent behavior** (tends to continue in the same direction)")
    elif memory < 0.35:
        parts.append("shows **anti-persistent behavior** (tends to reverse direction)")
    else:
        parts.append("displays **random walk characteristics** (no directional bias)")
    
    # Periodicity interpretation
    periodicity = row.get("periodicity", 0)
    if periodicity > 0.7:
        parts.append("contains **strong cyclical patterns**")
    elif periodicity > 0.3:
        parts.append("has **moderate periodic components**")
    else:
        parts.append("lacks significant periodicity")
    
    # Volatility interpretation
    volatility = row.get("volatility", 0)
    if volatility > 0.7:
        parts.append("experiences **high variance clustering** (volatility begets volatility)")
    elif volatility < 0.3:
        parts.append("maintains **stable variance** over time")
    
    # Complexity interpretation
    complexity = row.get("complexity", 0)
    if complexity > 0.7:
        parts.append("exhibits **high unpredictability**")
    elif complexity < 0.3:
        parts.append("follows **highly predictable patterns**")
    
    # Classification
    classification = row.get("classification", "UNKNOWN")
    
    return f"This signal is classified as **{classification}**. It {', '.join(parts[:3])}."


def interpret_behavioral_geometry(row: Dict[str, Any]) -> str:
    """Generate plain-language interpretation of behavioral geometry."""
    parts = []
    
    mean_corr = row.get("mean_correlation", 0)
    n_clusters = row.get("n_clusters", 1)
    density = row.get("network_density", 0)
    
    # Correlation interpretation
    if mean_corr > 0.7:
        parts.append("signals are **strongly coupled** (move together)")
    elif mean_corr > 0.3:
        parts.append("signals show **moderate interdependence**")
    elif mean_corr < -0.3:
        parts.append("signals exhibit **inverse relationships** (move oppositely)")
    else:
        parts.append("signals operate **relatively independently**")
    
    # Clustering interpretation
    if n_clusters > 3:
        parts.append(f"with **{n_clusters} distinct behavioral groups**")
    elif n_clusters > 1:
        parts.append(f"organized into **{n_clusters} main clusters**")
    else:
        parts.append("behaving as a **unified system**")
    
    # Network interpretation
    if density > 0.7:
        parts.append("The network is **densely connected** — changes propagate quickly.")
    elif density < 0.3:
        parts.append("The network is **sparsely connected** — components are isolated.")
    
    return " ".join(parts)


def interpret_dynamical_systems(row: Dict[str, Any]) -> str:
    """Generate plain-language interpretation of dynamical systems."""
    regime = row.get("regime", "UNKNOWN")
    stability = row.get("stability", "UNKNOWN")
    
    regime_text = {
        "COUPLED": "The system is in a **coupled regime** — components are synchronized.",
        "DECOUPLED": "The system is **decoupled** — components operate independently.",
        "TRANSITIONING": "The system is **transitioning** between regimes — a critical period.",
        "MODERATE": "The system shows **moderate coupling** — partial synchronization.",
    }.get(regime, f"The system is in a **{regime}** regime.")
    
    stability_text = {
        "STABLE": "The current state is **stable** — low probability of sudden change.",
        "EVOLVING": "The system is **gradually evolving** — monitor for acceleration.",
        "UNSTABLE": "The system is **unstable** — high probability of regime shift.",
        "CRITICAL": "⚠️ **Critical state** — system is at a tipping point.",
    }.get(stability, f"Stability is classified as **{stability}**.")
    
    return f"{regime_text} {stability_text}"


def interpret_causal_mechanics(row: Dict[str, Any]) -> str:
    """Generate plain-language interpretation of causal mechanics."""
    energy_class = row.get("energy_class", "UNKNOWN")
    equilibrium_class = row.get("equilibrium_class", "UNKNOWN")
    
    energy_text = {
        "conservative": "Energy is **conserved** — a closed system with no external forcing.",
        "driven": "Energy is **increasing** — external forces are injecting energy.",
        "dissipative": "Energy is **dissipating** — the system is losing energy.",
        "fluctuating": "Energy **fluctuates irregularly** — complex forcing dynamics.",
    }.get(energy_class.lower() if energy_class else "", f"Energy classification: **{energy_class}**.")
    
    equilibrium_text = {
        "approaching": "The system is **approaching equilibrium** spontaneously.",
        "at_equilibrium": "The system is **at equilibrium** — a stable state.",
        "departing": "The system is **departing from equilibrium** — investigate causes.",
        "forced": "The system is being **externally forced** — non-spontaneous changes.",
    }.get(equilibrium_class.lower() if equilibrium_class else "", f"Equilibrium: **{equilibrium_class}**.")
    
    return f"{energy_text} {equilibrium_text}"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_parquet(file) -> Optional[pl.DataFrame]:
    """Load a parquet file from upload."""
    try:
        return pl.read_parquet(file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None


def get_unique_values(df: pl.DataFrame, column: str) -> List:
    """Get unique values from a column."""
    if column in df.columns:
        return sorted(df[column].unique().to_list())
    return []


def filter_data(df: pl.DataFrame, entity_id: str, timestamp: float) -> pl.DataFrame:
    """Filter dataframe by entity and timestamp."""
    filtered = df
    if entity_id and "entity_id" in df.columns:
        filtered = filtered.filter(pl.col("entity_id") == entity_id)
    if timestamp is not None and "timestamp" in df.columns:
        filtered = filtered.filter(pl.col("timestamp") == timestamp)
    return filtered


# =============================================================================
# CLAUDE INTEGRATION
# =============================================================================

SYSTEM_PROMPT = """You are an expert analyst helping users understand time series analysis results from the Ørthon framework.

Ørthon analyzes how relationships between system components evolve over time. The core insight is: "Systems lose coherence before they fail."

The framework has four layers:
1. **Signal Typology** (WHAT): Characterizes signals along 6 axes - memory, periodicity, volatility, discontinuity, impulsivity, complexity
2. **Behavioral Geometry** (HOW): Analyzes relationships between signals - correlation, clustering, network density
3. **Dynamical Systems** (WHEN/HOW): Tracks regime changes and stability over time
4. **Causal Mechanics** (WHY): Physics-inspired analysis of energy, equilibrium, and flow

Guidelines for your responses:
- Use domain-agnostic scientific language (works for turbines, chemistry, bearings, any system)
- Explain what the numbers mean in practical terms
- Highlight concerning patterns or anomalies
- Be concise but thorough
- Use analogies when helpful
- If you see signs of degradation or instability, say so clearly
- Avoid jargon unless you explain it"""


def get_data_context(dataframes: Dict[str, pl.DataFrame], entity_id: str, timestamp: float) -> str:
    """Build context string from current data state."""
    context_parts = []
    
    if entity_id:
        context_parts.append(f"Entity: {entity_id}")
    if timestamp is not None:
        context_parts.append(f"Time: t = {timestamp}")
    
    for layer_name, df in dataframes.items():
        if df is None:
            continue
            
        # Filter to current selection
        filtered = df
        if entity_id and "entity_id" in df.columns:
            filtered = filtered.filter(pl.col("entity_id") == entity_id)
        if timestamp is not None and "timestamp" in df.columns:
            filtered = filtered.filter(pl.col("timestamp") == timestamp)
        
        if len(filtered) == 0:
            continue
        
        # Get first row as dict
        row = filtered.to_dicts()[0] if len(filtered) > 0 else {}
        
        # Format layer data
        layer_info = f"\n**{layer_name.replace('_', ' ').title()}:**\n"
        for key, value in row.items():
            if key in ['entity_id', 'timestamp']:
                continue
            if isinstance(value, float):
                layer_info += f"  - {key}: {value:.4f}\n"
            else:
                layer_info += f"  - {key}: {value}\n"
        
        context_parts.append(layer_info)
    
    return "\n".join(context_parts)


def ask_claude(question: str, data_context: str, api_key: str) -> str:
    """Send question to Claude with data context."""
    if not HAS_ANTHROPIC:
        return "Error: anthropic package not installed. Run `pip install anthropic`"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        user_message = f"""Here is the current analysis data:

{data_context}

User question: {question}"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        
        return response.content[0].text
        
    except anthropic.AuthenticationError:
        return "Error: Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "Error: Rate limit exceeded. Please wait a moment and try again."
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_equation_section(layer_key: str, expanded: bool = False):
    """Render the equations for a layer."""
    layer = EQUATIONS[layer_key]
    
    with st.expander(f"📐 Mathematical Foundations", expanded=expanded):
        st.markdown(f"*{layer['description']}*")
        
        cols = st.columns(2)
        for i, (name, eq) in enumerate(layer["equations"].items()):
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                st.latex(eq)


def render_data_at_time(df: pl.DataFrame, layer_key: str, interpretation_fn):
    """Render data and interpretation for a specific time point."""
    if df is None or len(df) == 0:
        st.info("No data available for the selected filters.")
        return
    
    # Show metrics
    row = df.to_dicts()[0] if len(df) > 0 else {}
    
    # Display key metrics
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    if numeric_cols:
        cols = st.columns(min(4, len(numeric_cols)))
        for i, col in enumerate(numeric_cols[:8]):
            value = row.get(col, None)
            if value is not None and isinstance(value, (int, float)):
                with cols[i % 4]:
                    st.metric(
                        label=col.replace("_", " ").title(),
                        value=f"{value:.4f}" if isinstance(value, float) else value
                    )
    
    # Interpretation
    st.markdown("---")
    st.markdown("### 💡 Interpretation")
    interpretation = interpretation_fn(row)
    st.markdown(f'<div class="interpretation">{interpretation}</div>', unsafe_allow_html=True)


def render_full_data(df: pl.DataFrame, layer_key: str):
    """Render full data table with optional filtering."""
    if df is None:
        return
    
    st.markdown("### 📊 Full Dataset")
    st.dataframe(
        df.to_pandas(),
        use_container_width=True,
        height=300
    )
    
    # Summary statistics
    with st.expander("📈 Summary Statistics"):
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64]]
        if numeric_cols:
            stats = df.select(numeric_cols).describe()
            st.dataframe(stats.to_pandas(), use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">◎ Ørthon Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">geometry leads — systems lose coherence before they fail</p>', unsafe_allow_html=True)
    
    # Sidebar: File uploads
    st.sidebar.header("📁 Load Data")
    
    uploaded_files = {}
    file_configs = [
        ("observations", "Observations", "Optional: raw sensor data"),
        ("signal_typology", "Signal Typology", "Layer 1: WHAT is it?"),
        ("behavioral_geometry", "Behavioral Geometry", "Layer 2: HOW does it behave?"),
        ("dynamical_systems", "Dynamical Systems", "Layer 3: WHEN/HOW does it change?"),
        ("causal_mechanics", "Causal Mechanics", "Layer 4: WHY does it change?"),
    ]
    
    for key, label, help_text in file_configs:
        uploaded_files[key] = st.sidebar.file_uploader(
            f"{label}",
            type=["parquet"],
            key=f"upload_{key}",
            help=help_text
        )
    
    # Load dataframes
    dataframes = {}
    for key, file in uploaded_files.items():
        if file is not None:
            dataframes[key] = load_parquet(file)
    
    # Check if any data loaded
    if not any(dataframes.values()):
        st.info("👆 Upload parquet files using the sidebar to begin exploring your ORTHON analysis.")
        
        # Show example structure
        with st.expander("📋 Expected File Structure"):
            st.markdown("""
            **observations.parquet** (optional)
            - `entity_id`: Entity identifier (e.g., machine ID)
            - `signal_id`: Signal identifier (e.g., sensor name)
            - `timestamp`: Time value
            - `value`: Observation value
            
            **signal_typology.parquet**
            - `entity_id`, `signal_id`, `timestamp`
            - `memory`, `periodicity`, `volatility`, `discontinuity`, `impulsivity`, `complexity`
            - `classification`
            
            **behavioral_geometry.parquet**
            - `entity_id`, `timestamp`
            - `mean_correlation`, `n_clusters`, `network_density`
            
            **dynamical_systems.parquet**
            - `entity_id`, `timestamp`
            - `regime`, `stability`
            
            **causal_mechanics.parquet**
            - `entity_id`, `timestamp`
            - `energy_class`, `equilibrium_class`
            """)
        return
    
    # Sidebar: Filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    # Collect all unique entities and timestamps
    all_entities = set()
    all_timestamps = set()
    
    for df in dataframes.values():
        if df is not None:
            if "entity_id" in df.columns:
                all_entities.update(df["entity_id"].unique().to_list())
            if "timestamp" in df.columns:
                all_timestamps.update(df["timestamp"].unique().to_list())
    
    all_entities = sorted(list(all_entities))
    all_timestamps = sorted(list(all_timestamps))
    
    # Entity selector
    selected_entity = st.sidebar.selectbox(
        "Entity",
        options=["All"] + all_entities,
        index=0
    )
    if selected_entity == "All":
        selected_entity = None
    
    # Timestamp selector
    if all_timestamps:
        timestamp_options = ["All"] + [str(t) for t in all_timestamps]
        selected_ts_str = st.sidebar.selectbox(
            "Time (t)",
            options=timestamp_options,
            index=0
        )
        if selected_ts_str == "All":
            selected_timestamp = None
        else:
            selected_timestamp = float(selected_ts_str)
    else:
        selected_timestamp = None
    
    # Time slider for specific time point
    if all_timestamps and len(all_timestamps) > 1:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Time Slider**")
        time_idx = st.sidebar.slider(
            "Select Time Point",
            min_value=0,
            max_value=len(all_timestamps) - 1,
            value=0,
            format=f"t = %d"
        )
        slider_timestamp = all_timestamps[time_idx]
        st.sidebar.caption(f"t = {slider_timestamp}")
    else:
        slider_timestamp = selected_timestamp
    
    # Main content tabs
    tab_labels = ["🔬 Analysis", "🤖 Ask Claude", "📐 Equations", "📊 Data Tables"]
    tabs = st.tabs(tab_labels)
    
    # TAB 1: Analysis
    with tabs[0]:
        st.markdown("## Analysis at Time t")
        
        if slider_timestamp is not None:
            st.info(f"📍 Showing analysis at **t = {slider_timestamp}**" + 
                   (f" for entity **{selected_entity}**" if selected_entity else ""))
        
        # Four columns for the four questions
        col1, col2 = st.columns(2)
        
        # WHAT - Signal Typology
        with col1:
            st.markdown('<div class="question-card what-card">', unsafe_allow_html=True)
            st.markdown("### 🟢 WHAT is it?")
            st.caption("Signal Typology — Layer 1")
            
            if "signal_typology" in dataframes and dataframes["signal_typology"] is not None:
                df = dataframes["signal_typology"]
                filtered = filter_data(df, selected_entity, slider_timestamp)
                
                if len(filtered) > 0:
                    render_data_at_time(filtered, "signal_typology", interpret_signal_typology)
                else:
                    st.warning("No signal typology data for selected filters.")
            else:
                st.info("Upload signal_typology.parquet to see WHAT analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # HOW - Behavioral Geometry
        with col2:
            st.markdown('<div class="question-card how-card">', unsafe_allow_html=True)
            st.markdown("### 🔵 HOW does it behave?")
            st.caption("Behavioral Geometry — Layer 2")
            
            if "behavioral_geometry" in dataframes and dataframes["behavioral_geometry"] is not None:
                df = dataframes["behavioral_geometry"]
                filtered = filter_data(df, selected_entity, slider_timestamp)
                
                if len(filtered) > 0:
                    render_data_at_time(filtered, "behavioral_geometry", interpret_behavioral_geometry)
                else:
                    st.warning("No geometry data for selected filters.")
            else:
                st.info("Upload behavioral_geometry.parquet to see HOW analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        # WHEN/HOW - Dynamical Systems
        with col3:
            st.markdown('<div class="question-card when-card">', unsafe_allow_html=True)
            st.markdown("### 🟠 WHEN/HOW does it change?")
            st.caption("Dynamical Systems — Layer 3")
            
            if "dynamical_systems" in dataframes and dataframes["dynamical_systems"] is not None:
                df = dataframes["dynamical_systems"]
                filtered = filter_data(df, selected_entity, slider_timestamp)
                
                if len(filtered) > 0:
                    render_data_at_time(filtered, "dynamical_systems", interpret_dynamical_systems)
                else:
                    st.warning("No dynamics data for selected filters.")
            else:
                st.info("Upload dynamical_systems.parquet to see WHEN/HOW analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # WHY - Causal Mechanics
        with col4:
            st.markdown('<div class="question-card why-card">', unsafe_allow_html=True)
            st.markdown("### 🟣 WHY does it change?")
            st.caption("Causal Mechanics — Layer 4")
            
            if "causal_mechanics" in dataframes and dataframes["causal_mechanics"] is not None:
                df = dataframes["causal_mechanics"]
                filtered = filter_data(df, selected_entity, slider_timestamp)
                
                if len(filtered) > 0:
                    render_data_at_time(filtered, "causal_mechanics", interpret_causal_mechanics)
                else:
                    st.warning("No mechanics data for selected filters.")
            else:
                st.info("Upload causal_mechanics.parquet to see WHY analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: Ask Claude
    with tabs[1]:
        st.markdown("## 🤖 Ask Claude About Your Data")
        st.markdown("*Get plain-language explanations of your analysis results.*")
        
        # API Key input
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key. Get one at console.anthropic.com",
            key="anthropic_api_key"
        )
        
        if not HAS_ANTHROPIC:
            st.warning("⚠️ The `anthropic` package is not installed. Run `pip install anthropic` to enable this feature.")
        
        # Build current data context
        data_context = get_data_context(dataframes, selected_entity, slider_timestamp)
        
        # Show current context
        with st.expander("📋 Current Data Context", expanded=False):
            st.markdown(data_context if data_context else "*No data loaded*")
        
        # Preset questions
        st.markdown("### Quick Questions")
        preset_cols = st.columns(2)
        
        preset_questions = [
            "What does this data tell me about system health?",
            "Are there any warning signs I should be concerned about?",
            "Explain the current regime and stability in plain terms.",
            "What's causing the changes I'm seeing?",
            "How do the signals relate to each other?",
            "What should I monitor going forward?",
        ]
        
        selected_preset = None
        for i, q in enumerate(preset_questions):
            with preset_cols[i % 2]:
                if st.button(q, key=f"preset_{i}", use_container_width=True):
                    selected_preset = q
        
        # Custom question input
        st.markdown("### Or Ask Your Own Question")
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., Why is the correlation dropping? What does high volatility mean for my system?",
            height=100,
            key="user_question"
        )
        
        # Use preset if selected
        question_to_ask = selected_preset if selected_preset else user_question
        
        # Ask button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            ask_button = st.button("🔍 Ask Claude", type="primary", use_container_width=True)
        
        # Handle question
        if ask_button and question_to_ask:
            if not api_key:
                st.error("Please enter your Anthropic API key above.")
            elif not data_context or data_context.strip() == "":
                st.error("No data loaded. Please upload parquet files first.")
            else:
                with st.spinner("Claude is analyzing your data..."):
                    response = ask_claude(question_to_ask, data_context, api_key)
                
                st.markdown("### Claude's Analysis")
                st.markdown(response)
                
                # Add to chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    "question": question_to_ask,
                    "answer": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
        # Show chat history
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
    
    # TAB 3: Equations
    with tabs[2]:
        st.markdown("## Mathematical Foundations")
        st.markdown("*The equations underlying each analytical layer.*")
        
        for layer_key, layer_data in EQUATIONS.items():
            st.markdown(f"### {layer_data['title']}")
            st.markdown(f"*{layer_data['description']}*")
            
            cols = st.columns(2)
            for i, (name, eq) in enumerate(layer_data["equations"].items()):
                with cols[i % 2]:
                    st.markdown(f"**{name}**")
                    st.latex(eq)
            
            st.markdown("---")
    
    # TAB 4: Data Tables
    with tabs[3]:
        st.markdown("## Data Tables")
        
        for key, label, _ in file_configs:
            if key in dataframes and dataframes[key] is not None:
                with st.expander(f"📄 {label}", expanded=False):
                    df = dataframes[key]
                    
                    # Apply filters
                    filtered = df
                    if selected_entity and "entity_id" in df.columns:
                        filtered = filtered.filter(pl.col("entity_id") == selected_entity)
                    
                    st.dataframe(filtered.to_pandas(), use_container_width=True, height=300)
                    
                    # Download button
                    csv = filtered.to_pandas().to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {label} CSV",
                        data=csv,
                        file_name=f"{key}_filtered.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">geometry leads — ørthon | '
        'Systems lose coherence before they fail.</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
