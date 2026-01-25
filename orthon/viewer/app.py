#!/usr/bin/env python3
"""
ORTHON Phase Space Manifold - Streamlit App
============================================

Embeds the 3D manifold viewer in a Streamlit application.

Usage:
    streamlit run orthon/viewer/app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="ORTHON Phase Space Manifold",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit chrome for immersive experience
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Read the HTML file
viewer_path = Path(__file__).parent / "manifold.html"

if viewer_path.exists():
    html_content = viewer_path.read_text()

    # Embed the viewer
    st.components.v1.html(html_content, height=900, scrolling=False)
else:
    st.error(f"Viewer not found at {viewer_path}")
    st.info("Run from the orthon/viewer directory or check the file path.")
