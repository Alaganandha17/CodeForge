import streamlit as st

def apply_glass_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #E8D5F2 0%, #D4C5E8 50%, #C8D5F0 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }

    hr {
        border: none !important;
        border-top: 2px solid rgba(91, 75, 126, 0.15) !important;
        margin: 3rem 0 !important;
    }

    .footer {
        text-align: center;
        color: #9B8DB3;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 3rem;
        padding: 2rem 0;
    }

    </style>
    """, unsafe_allow_html=True)