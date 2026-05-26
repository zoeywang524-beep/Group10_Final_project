import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import time
import io
import scipy.io.wavfile as wav
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EcoPulse AI – Customer Sentiment Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] { background: #0f2027; }
section[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
section[data-testid="stSidebar"] .stTextArea textarea {
    background:#1a3a2a; color:#e8f5e9; border:1px solid #2e7d52;
}

div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.neg-card {
    background:#fff3f3; border-left:4px solid #d32f2f;
    border-radius:8px; padding:10px 14px; margin-bottom:10px;
}
.pos-card {
    background:#f3fff6; border-left:4px solid #388e3c;
    border-radius:8px; padding:10px 14px; margin-bottom:10px;
}
.neu-card {
    background:#f8f9fa; border-left:4px solid #757575;
    border-radius:8px; padding:10px 14px; margin-bottom:10px;
}
.escalate-box {
    background:#fff8e1; border:2px solid #f57c00;
    border-radius:10px; padding:14px 18px; margin-top:10px;
    font-weight:600; font-size:15px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# !! Replace with your actual HuggingFace mode
