"""
Ørthon Explorer
geometry leads — ørthon

Standalone parquet explorer. No dependencies on orthon package.
Just upload parquet, query with SQL, visualize.
"""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Ørthon",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE
# ============================================================

if 'db' not in st.session_state:
    st.session_state.db = duckdb.connect(":memory:")
if 'tables' not in st.session_state:
    st.session_state.tables = {}

conn = st.session_state.db

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_parquet(name: str, file_path: str):
    """Load parquet into DuckDB as a named table"""
    conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{file_path}')")
    st.session_state.tables[name] = file_path

def get_table_info(table_name: str) -> pd.DataFrame:
    """Get column info for a table"""
    return conn.execute(f"DESCRIBE {table_name}").df()

def run_query(sql: str) -> pd.DataFrame:
    """Run SQL and return DataFrame"""
    try:
        return conn.execute(sql).df()
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()

def get_unique_values(table: str, column: str) -> list:
    """Get unique values for dropdown"""
    result = conn.execute(f"SELECT DISTINCT {column} FROM {table} ORDER BY {column} LIMIT 1000").df()
    return result[column].tolist()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 Ørthon")
st.sidebar.caption("geometry leads")

st.sidebar.markdown("---")

# File upload section
st.sidebar.subheader("📁 Load Data")

upload_method = st.sidebar.radio("Method:", ["Upload File", "File Path"], horizontal=True)

if upload_method == "Upload File":
    uploaded = st.sidebar.file_uploader("Upload Parquet", type=['parquet'], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            temp_path = Path(f"/tmp/{f.name}")
            temp_path.write_bytes(f.read())
            table_name = f.name.replace('.parquet', '').replace('-', '_').replace('.', '_')
            load_parquet(table_name, str(temp_path))
            st.sidebar.success(f"Loaded: {table_name}")
else:
    file_path = st.sidebar.text_input("Parquet path:")
    table_name = st.sidebar.text_input("Table name:", value="data")
    if st.sidebar.button("Load") and file_path:
        if Path(file_path).exists():
            load_parquet(table_name, file_path)
            st.sidebar.success(f"Loaded: {table_name}")
        else:
            st.sidebar.error("File not found")

st.sidebar.markdown("---")

# Show loaded tables
if st.session_state.tables:
    st.sidebar.subheader("📊 Loaded Tables")
    for t in st.session_state.tables:
        st.sidebar.write(f"• `{t}`")
    
    st.sidebar.markdown("---")
    
    # Filters (if tables loaded)
    st.sidebar.subheader("🔍 Quick Filters")
    
    active_table = st.sidebar.selectbox("Table:", list(st.session_state.tables.keys()))
    
    if active_table:
        cols = get_table_info(active_table)['column_name'].tolist()
        filter_col = st.sidebar.selectbox("Filter column:", ["None"] + cols)
        
        if filter_col != "None":
            unique_vals = get_unique_values(active_table, filter_col)
            selected_vals = st.sidebar.multiselect(f"Select {filter_col}:", unique_vals)

# ============================================================
# MAIN CONTENT - TABS
# ============================================================

tabs = st.tabs([
    "📋 Data Summary",
    "📈 Signal Typology",
    "🔷 Behavioral Geometry",
    "🔄 State",
    "⚡ Derivatives",
    "🎯 Advanced Analysis",
    "💻 SQL Console"
])

# ------------------------------------------------------------
# TAB 1: Data Summary
# ------------------------------------------------------------
with tabs[0]:
    st.header("Data Summary")
    
    if not st.session_state.tables:
        st.info("👈 Load a parquet file from the sidebar to begin.")
    else:
        for table_name in st.session_state.tables:
            st.subheader(f"Table: `{table_name}`")
            
            # Row count
            count = run_query(f"SELECT COUNT(*) as rows FROM {table_name}")
            st.metric("Total Rows", f"{count['rows'][0]:,}")
            
            # Schema
            st.markdown("**Schema:**")
            schema = get_table_info(table_name)
            st.dataframe(schema, use_container_width=True, hide_index=True)
            
            # Sample
            st.markdown("**Sample (first 100 rows):**")
            sample = run_query(f"SELECT * FROM {table_name} LIMIT 100")
            st.dataframe(sample, use_container_width=True)
            
            # Stats
            if st.checkbox(f"Show statistics for {table_name}"):
                st.markdown("**Numeric column stats:**")
                stats = run_query(f"SUMMARIZE {table_name}")
                st.dataframe(stats, use_container_width=True)
            
            st.markdown("---")

# ------------------------------------------------------------
# TAB 2: Signal Typology
# ------------------------------------------------------------
with tabs[1]:
    st.header("Signal Typology")
    st.caption("Explore individual signals")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        table = st.selectbox("Table:", list(st.session_state.tables.keys()), key="sig_table")
        cols = get_table_info(table)['column_name'].tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X axis (time/index):", cols, key="sig_x")
        with col2:
            y_col = st.selectbox("Y axis (signal):", cols, key="sig_y")
        with col3:
            chart_type = st.selectbox("Chart:", ["Line", "Scatter", "Histogram"], key="sig_chart")
        
        # Optional filter
        group_col = st.selectbox("Group by (optional):", ["None"] + cols, key="sig_group")
        
        limit = st.slider("Row limit:", 100, 10000, 1000, key="sig_limit")
        
        # Build query
        if group_col != "None":
            unique_groups = get_unique_values(table, group_col)
            selected_group = st.selectbox(f"Select {group_col}:", unique_groups)
            query = f"SELECT {x_col}, {y_col} FROM {table} WHERE {group_col} = '{selected_group}' ORDER BY {x_col} LIMIT {limit}"
        else:
            query = f"SELECT {x_col}, {y_col} FROM {table} ORDER BY {x_col} LIMIT {limit}"
        
        st.code(query, language="sql")
        
        data = run_query(query)
        
        if not data.empty:
            if chart_type == "Line":
                st.line_chart(data, x=x_col, y=y_col)
            elif chart_type == "Scatter":
                st.scatter_chart(data, x=x_col, y=y_col)
            else:
                st.bar_chart(data[y_col].value_counts())

# ------------------------------------------------------------
# TAB 3: Behavioral Geometry
# ------------------------------------------------------------
with tabs[2]:
    st.header("Behavioral Geometry")
    st.caption("Pairwise relationships between signals")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        table = st.selectbox("Table:", list(st.session_state.tables.keys()), key="geo_table")
        cols = get_table_info(table)['column_name'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_signal = st.selectbox("Signal X:", cols, key="geo_x")
        with col2:
            y_signal = st.selectbox("Signal Y:", cols, index=min(1, len(cols)-1), key="geo_y")
        
        limit = st.slider("Row limit:", 100, 10000, 1000, key="geo_limit")
        
        query = f"SELECT {x_signal}, {y_signal} FROM {table} LIMIT {limit}"
        st.code(query, language="sql")
        
        data = run_query(query)
        
        if not data.empty:
            st.scatter_chart(data, x=x_signal, y=y_signal)
            
            # Correlation
            if st.checkbox("Compute correlation"):
                corr_query = f"SELECT CORR({x_signal}, {y_signal}) as correlation FROM {table}"
                corr = run_query(corr_query)
                st.metric("Correlation", f"{corr['correlation'][0]:.4f}")

# ------------------------------------------------------------
# TAB 4: State
# ------------------------------------------------------------
with tabs[3]:
    st.header("State Analysis")
    st.caption("Regime detection, coherence tracking")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        table = st.selectbox("Table:", list(st.session_state.tables.keys()), key="state_table")
        cols = get_table_info(table)['column_name'].tolist()
        
        st.markdown("**Custom SQL for state analysis:**")
        
        default_query = f"SELECT * FROM {table} LIMIT 100"
        user_query = st.text_area("SQL:", value=default_query, height=100, key="state_sql")
        
        if st.button("Run", key="state_run"):
            result = run_query(user_query)
            st.dataframe(result, use_container_width=True)

# ------------------------------------------------------------
# TAB 5: Derivatives
# ------------------------------------------------------------
with tabs[4]:
    st.header("Derivatives")
    st.caption("Rate of change analysis")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        table = st.selectbox("Table:", list(st.session_state.tables.keys()), key="deriv_table")
        cols = get_table_info(table)['column_name'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("Time column:", cols, key="deriv_time")
        with col2:
            value_col = st.selectbox("Value column:", cols, key="deriv_val")
        
        # Compute derivative via SQL window function
        deriv_query = f"""
        SELECT 
            {time_col},
            {value_col},
            {value_col} - LAG({value_col}) OVER (ORDER BY {time_col}) as derivative
        FROM {table}
        LIMIT 1000
        """
        
        st.code(deriv_query, language="sql")
        
        if st.button("Compute", key="deriv_run"):
            result = run_query(deriv_query)
            st.line_chart(result, x=time_col, y=['derivative'])
            st.dataframe(result, use_container_width=True)

# ------------------------------------------------------------
# TAB 6: Advanced Analysis
# ------------------------------------------------------------
with tabs[5]:
    st.header("Advanced Analysis")
    st.caption("Cohort discovery, aggregations")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        table = st.selectbox("Table:", list(st.session_state.tables.keys()), key="adv_table")
        cols = get_table_info(table)['column_name'].tolist()
        
        st.markdown("**Group by analysis:**")
        
        group_col = st.selectbox("Group by:", cols, key="adv_group")
        agg_col = st.selectbox("Aggregate column:", cols, key="adv_agg")
        agg_func = st.selectbox("Function:", ["AVG", "SUM", "MIN", "MAX", "COUNT", "STDDEV"], key="adv_func")
        
        agg_query = f"""
        SELECT 
            {group_col},
            {agg_func}({agg_col}) as {agg_func.lower()}_{agg_col}
        FROM {table}
        GROUP BY {group_col}
        ORDER BY {agg_func.lower()}_{agg_col} DESC
        """
        
        st.code(agg_query, language="sql")
        
        if st.button("Run", key="adv_run"):
            result = run_query(agg_query)
            st.bar_chart(result, x=group_col, y=f"{agg_func.lower()}_{agg_col}")
            st.dataframe(result, use_container_width=True)

# ------------------------------------------------------------
# TAB 7: SQL Console
# ------------------------------------------------------------
with tabs[6]:
    st.header("SQL Console")
    st.caption("Run any SQL against loaded tables")
    
    if not st.session_state.tables:
        st.info("👈 Load data first.")
    else:
        st.markdown("**Available tables:**")
        for t in st.session_state.tables:
            cols = get_table_info(t)['column_name'].tolist()
            st.code(f"{t}: {', '.join(cols)}")
        
        user_sql = st.text_area("SQL Query:", height=150, key="console_sql", 
                                placeholder="SELECT * FROM your_table LIMIT 100")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            run = st.button("▶ Run", key="console_run", type="primary")
        
        if run and user_sql:
            result = run_query(user_sql)
            st.dataframe(result, use_container_width=True)
            
            # Export option
            if not result.empty:
                csv = result.to_csv(index=False)
                st.download_button("Download CSV", csv, "query_result.csv", "text/csv")

# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption("Ørthon Explorer v0.1")
st.sidebar.caption("geometry leads — ørthon")
