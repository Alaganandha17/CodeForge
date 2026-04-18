import streamlit as st
import html as html_module
from utils.auth import require_auth, get_current_user

st.set_page_config(page_title="About", layout="wide")

require_auth()
user = get_current_user()

# ---------------- STYLING ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

h1 {
    font-weight: 600;
    font-size: 24px;
}

h2 {
    font-weight: 600;
    font-size: 16px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 30px;
}

.section {
    background: #18181F;
    border: 1px solid #26262E;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

ul {
    margin-top: 8px;
}

hr {
    border: none;
    border-top: 1px solid #26262E;
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("## About CodeForge")
st.markdown(
    f"<div style='color:#9CA3AF;font-size:14px;'>Signed in as {html_module.escape(user['username'])}</div>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- OVERVIEW ----------------

st.markdown("### Overview")

st.markdown("""
<div class='section'>
CodeForge is a performance-focused Python optimization platform that transforms
inefficient code into faster, leaner versions using rule-based analysis
and AI-driven rewriting.

The system measures runtime and memory improvements, validates safety,
and generates structured explanations of applied optimizations.
</div>
""", unsafe_allow_html=True)

# ---------------- CORE CAPABILITIES ----------------

st.markdown("### Core Capabilities")

st.markdown("""
<div class='section'>
<ul>
<li>Hybrid rule-based and AI optimization</li>
<li>Runtime and memory benchmarking</li>
<li>Safety validation and semantic checks</li>
<li>Confidence scoring</li>
<li>Structured optimization explanations</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- TECHNOLOGY ----------------

st.markdown("### Technology")

st.markdown("""
<div class='section'>
Backend:
<ul>
<li>Python</li>
<li>FastAPI</li>
<li>AST transformations</li>
<li>Google Gemini</li>
</ul>

Frontend:
<ul>
<li>Streamlit</li>
<li>Plotly</li>
<li>Pandas</li>
</ul>
</div>
""", unsafe_allow_html=True)