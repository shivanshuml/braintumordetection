# styles.py
import streamlit as st

def inject_css():
    st.markdown("""
    <style>

    /* Base Theme */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    /* Typography */
    .main-title {
        text-align: center;
        color: #00D4FF;
        font-size: 48px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #B0B0B0;
        font-size: 20px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background-color: #1A1F2B;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #00D4FF;
    }

    .info-card {
        background-color: #1A1F2B;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00D4FF;
    }

    .reject-card {
        background-color: #2A151B;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #FF4B4B;
        border-left: 6px solid #FF4B4B;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .verified-badge {
        display: inline-block;
        background-color: #0d3b2e;
        color: #00FFB2;
        border: 1px solid #00FFB2;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 12px;
    }

    .stage-badge {
        display: inline-block;
        background-color: #162438;
        color: #00D4FF;
        border: 1px solid #00D4FF;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        background-color: #1A1F2B;
        color: white;
    }
    
    [data-testid="stDataFrame"] * {
        color: white !important;
    }

    /* Tables */
    table {
        color: white !important;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: #00D4FF !important;
        color: black !important;
        border-radius: 10px;
        font-weight: bold;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1A1F2B;
        border-radius: 10px;
        padding: 10px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1A1F2B;
        border: 1px solid #00D4FF;
        padding: 15px;
        border-radius: 10px;
    }

    /* Success message */
    .stSuccess {
        background-color: #1A1F2B !important;
        color: white !important;
    }

    /* Warning message */
    .stWarning {
        background-color: #332B00 !important;
        color: #FFD700 !important;
        border-left: 5px solid #FFD700 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: gray;
        margin-top: 40px;
    }

    </style>
    """, unsafe_allow_html=True)
