import streamlit as st
import html as html_module
from utils.auth import require_auth, get_current_user

st.set_page_config(page_title="API Docs", layout="wide")



# ---------------- STYLING ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 1000px;
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
}

.section {
    background: #18181F;
    border: 1px solid #26262E;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

code {
    background: #0F0F14;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 13px;
}

pre {
    background: #0F0F14;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #26262E;
}

hr {
    border: none;
    border-top: 1px solid #26262E;
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("## API Documentation")
st.markdown(
    f"<div style='color:#9CA3AF;font-size:14px;'>Signed in as {html_module.escape(user['username'])}</div>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- BASE URL ----------------

st.markdown("### Base URL")
st.markdown("<div class='section'><code>http://localhost:8000</code></div>", unsafe_allow_html=True)

# ---------------- ENDPOINTS ----------------

st.markdown("### Endpoints")

st.markdown("""
<div class='section'>
<strong>GET /</strong><br>
Health check endpoint.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='section'>
<strong>POST /optimize</strong><br>
Hybrid optimization using rules and AI.<br><br>
Request body:
<pre>{
  "code": "your python code"
}</pre>
Returns optimized code, benchmarks, explanation, safety analysis, and confidence.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='section'>
<strong>POST /optimize-rules-only</strong><br>
Rule-based optimization with benchmarking.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='section'>
<strong>POST /optimize-rules-only/simple</strong><br>
Rule-based optimization without benchmarking.
</div>
""", unsafe_allow_html=True)