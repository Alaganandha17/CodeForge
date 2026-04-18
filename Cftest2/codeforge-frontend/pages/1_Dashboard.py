import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html as html_module


from utils import api

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CodeForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)



# -------------------- SESSION STATE --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "result" not in st.session_state:
    st.session_state.result = None

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- EXAMPLE SNIPPETS --------------------
EXAMPLE_SNIPPETS = {
    "Select an example...": "",
    "List append in loop": """data = list(range(1000))
result = []
for x in data:
    result.append(x * 2)
print(len(result))""",
    "range(len()) anti-pattern": """data = list(range(1000))
result = []
for i in range(len(data)):
    result.append(data[i] * 2)
print(len(result))""",
    "Nested loops (quadratic)": """data = list(range(200))
pairs = []
for i in range(len(data)):
    for j in range(len(data)):
        if data[i] + data[j] == 100:
            pairs.append((data[i], data[j]))
print(len(pairs))""",
    "String concatenation in loop": """items = list(range(500))
result = ""
for item in items:
    result += str(item)
print(len(result))""",
    "Inefficient duplicate finder": """lst = list(range(500)) + list(range(250))
duplicates = []
for i in range(len(lst)):
    count = 0
    for j in range(len(lst)):
        if lst[i] == lst[j]:
            count += 1
    if count > 1 and lst[i] not in duplicates:
        duplicates.append(lst[i])
print(len(duplicates))""",
    "Sum of even numbers": """data_list = list(range(10000))
total_even = 0
for number in data_list:
    if number % 2 == 0:
        total_even += number
print(total_even)""",
    "Count primes (Sieve opportunity)": """def count_primes_below(limit):
    count = 0
    for number in range(2, limit):
        is_prime = True
        for divisor in range(2, number):
            if number % divisor == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

print(count_primes_below(500))""",
}

# -------------------- THEME TOGGLE --------------------
def switch_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# -------------------- THEME COLORS --------------------
if st.session_state.theme == "dark":
    BG = "#0B0F19"
    CARD = "rgba(17,24,39,0.8)"
    TEXT = "#E6E8EC"
    ACCENT = "#00F5FF"
    GRAPH_FONT = "#E6E8EC"
    button_label = "Light Mode"
else:
    BG = "#F5F7FA"
    CARD = "rgba(255,255,255,0.95)"
    TEXT = "#111827"
    ACCENT = "#2563EB"
    GRAPH_FONT = "#111827"
    button_label = "Dark Mode"

# -------------------- GLOBAL STYLING --------------------
st.markdown(f"""
<style>

/* Keep sidebar toggle visible */
#MainMenu, footer {{visibility:hidden;}}

/* Proper centering behaviour */
.block-container {{
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 1.5rem;
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

.codeforge-title {{
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(90deg, #00F5FF, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.subtitle {{
    text-align: center;
    font-size: 16px;
    opacity: 0.7;
    margin-bottom: 20px;
}}

.stButton>button {{
    background: linear-gradient(90deg, #2563EB, #00F5FF);
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: 600;
}}

.metric-card {{
    background: {CARD};
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(59,130,246,0.3);
}}

.metric-value {{
    font-size: 30px;
    font-weight: 700;
    color: {ACCENT};
}}

.ai-panel {{
    background: {CARD};
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(0,245,255,0.3);
}}

.rule-tag {{
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 4px;
    font-size: 13px;
}}

.safety-ok {{
    color: #22C55E;
    font-weight: 600;
}}

.safety-warn {{
    color: #F59E0B;
    font-weight: 600;
}}

.complexity-card {{
    background: {CARD};
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(59,130,246,0.2);
    margin-bottom: 8px;
}}

</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR CONTENT --------------------
with st.sidebar:
    st.title("Navigation")
    
    st.write("---")

    # Backend health check
    health = api.api_client.health_check()
    if health:
        st.success(f"Backend: Online (v{health.get('version', '?')})")
        if health.get("gemini_available"):
            st.info("Gemini AI: Available")
        else:
            st.warning("Gemini AI: Not configured")
    else:
        st.error("Backend: Offline")

    st.write("---")

    # Optimization history
    if st.session_state.history:
        st.markdown("### Recent Optimizations")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            speedup_txt = f"{h['speedup']}x" if h.get('speedup') else "N/A"
            st.caption(f"{i+1}. {h['mode']} | Speedup: {speedup_txt}")
    else:
        st.caption("No optimization history yet.")

# -------------------- HEADER --------------------
header_left, header_right = st.columns([8,2])

with header_left:
    st.markdown("<div class='codeforge-title'>CODEFORGE</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Shaping Faster. Smarter. Leaner Code.</div>", unsafe_allow_html=True)

with header_right:
    st.button(button_label, on_click=switch_theme)
    if st.button("Logout"):
        from utils.auth import logout
        logout()

st.divider()

# -------------------- MODE SELECTOR --------------------
st.markdown("### Select Optimization Mode")
opt_mode = st.radio(
    "Optimization Mode",
    ["AI-Powered (Hybrid)", "Rules-Only (Offline)"],
    horizontal=True,
    help="AI-Powered uses Gemini API with automatic fallback. Rules-Only works fully offline."
)
st.markdown("")

# Example snippet selector
example_choice = st.selectbox("Load an example snippet:", list(EXAMPLE_SNIPPETS.keys()))

uploaded_file = st.file_uploader("Or upload a .py file", type=["py"])

# -------------------- LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input Code")

    # Pre-fill from uploaded file or example
    default_code = ""
    if uploaded_file is not None:
        default_code = uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)
    elif example_choice and example_choice != "Select an example...":
        default_code = EXAMPLE_SNIPPETS[example_choice]

    code_input = st.text_area(
        "Code Input",
        value=default_code,
        height=300,
        placeholder="Paste your Python code here...",
        label_visibility="collapsed"
    )

    b1, b2 = st.columns(2)
    with b1:
        run = st.button("Run Optimization", use_container_width=True)
    with b2:
        bench = st.button("Benchmark Only", use_container_width=True)

with col2:
    st.markdown("### Optimized Code")
    output_placeholder = st.empty()

# -------------------- RUN LOGIC --------------------
if run and code_input.strip():
    with st.spinner("Optimizing..."):
        if opt_mode == "AI-Powered (Hybrid)":
            result = api.api_client.optimize(code_input)
        else:
            result = api.api_client.optimize_rules_only(code_input)
        st.session_state.result = result

        # Track history
        if result and not result.get("error"):
            benchmarks = result.get("benchmarks", {})
            st.session_state.history.append({
                "mode": result.get("mode", "?"),
                "speedup": benchmarks.get("speedup_factor"),
                "rules": len(result.get("rules_detected", [])),
                "timestamp": result.get("timestamp", ""),
            })

elif bench and code_input.strip():
    with st.spinner("Benchmarking (Rules-Only)..."):
        result = api.api_client.optimize_rules_only(code_input)
        st.session_state.result = result

        if result and not result.get("error"):
            benchmarks = result.get("benchmarks", {})
            st.session_state.history.append({
                "mode": "BENCHMARK",
                "speedup": benchmarks.get("speedup_factor"),
                "rules": len(result.get("rules_detected", [])),
                "timestamp": result.get("timestamp", ""),
            })

result = st.session_state.result

# -------------------- ERROR HANDLING --------------------
if result and result.get("error"):
    st.error(result["error"])
    st.stop()

# -------------------- OUTPUT --------------------
if result and result.get("optimized_code"):
    optimized_code = result["optimized_code"]
    output_placeholder.code(optimized_code, language="python")

    if result.get("ai_explanation"):
        st.markdown("### AI Analysis")
        st.markdown(result["ai_explanation"])

    # -------------------- TABS --------------------
    tab_code, tab_perf, tab_complexity, tab_rules, tab_safety, tab_ai = st.tabs([
        "Code Diff", "Performance", "Complexity", "Rules Detected", "Safety & Confidence", "AI Explanation"
    ])

    # ---- TAB: Code Diff ----
    with tab_code:
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.markdown("**Original**")
            st.code(result.get("original_code", ""), language="python")
        with diff_col2:
            st.markdown("**Optimized**")
            st.code(optimized_code, language="python")

        # Show explainability diff if available
        explainability = result.get("explainability")
        if explainability:
            transform = explainability.get("transformation", {})
            st.markdown(f"""
            **Lines:** {transform.get('original_lines', '?')} -> {transform.get('optimized_lines', '?')}
            (net change: {transform.get('net_change', '?')})
            """)

    # ---- TAB: Performance ----
    with tab_perf:
        metrics = result.get("benchmarks") or {}

        if metrics:
            orig = metrics.get("original") or {}
            opt = metrics.get("optimized") or {}

            orig_ms = orig.get("runtime_ms")
            opt_ms = opt.get("runtime_ms")
            orig_mem = orig.get("memory_mb")
            opt_mem = opt.get("memory_mb")
            speed = metrics.get("speedup_factor")

            st.markdown("### Performance Metrics")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                orig_runtime_text = f"{orig_ms:.3f} ms" if orig_ms is not None else "N/A"
                st.markdown(f"<div class='metric-card'><div>Original Runtime</div><div class='metric-value'>{orig_runtime_text}</div></div>", unsafe_allow_html=True)

            with m2:
                opt_runtime_text = f"{opt_ms:.3f} ms" if opt_ms is not None else "N/A"
                st.markdown(f"<div class='metric-card'><div>Optimized Runtime</div><div class='metric-value'>{opt_runtime_text}</div></div>", unsafe_allow_html=True)

            with m3:
                if orig_ms is not None and opt_ms is not None:
                    runtime_saved = f"{round(orig_ms - opt_ms, 3)} ms"
                else:
                    runtime_saved = "N/A"
                st.markdown(f"<div class='metric-card'><div>Runtime Saved</div><div class='metric-value'>{runtime_saved}</div></div>", unsafe_allow_html=True)

            with m4:
                speedup_text = f"{speed}x" if speed is not None else "N/A"
                color = ACCENT
                if speed is not None:
                    if speed >= 1.5:
                        color = "#22C55E"  # green
                    elif speed < 1.0:
                        color = "#EF4444"  # red - slower
                st.markdown(f"<div class='metric-card'><div>Speedup Factor</div><div class='metric-value' style='color:{color}'>{speedup_text}</div></div>", unsafe_allow_html=True)

            # Memory metrics
            mem1, mem2 = st.columns(2)
            with mem1:
                orig_mem_text = f"{orig_mem:.2f} MB" if orig_mem is not None else "N/A"
                st.markdown(f"<div class='metric-card'><div>Original Memory</div><div class='metric-value'>{orig_mem_text}</div></div>", unsafe_allow_html=True)
            with mem2:
                opt_mem_text = f"{opt_mem:.2f} MB" if opt_mem is not None else "N/A"
                st.markdown(f"<div class='metric-card'><div>Optimized Memory</div><div class='metric-value'>{opt_mem_text}</div></div>", unsafe_allow_html=True)

            # Variance info
            orig_var = orig.get("variance_pct", 0)
            opt_var = opt.get("variance_pct", 0)
            if orig_var or opt_var:
                st.caption(f"Benchmark variance: Original {orig_var}% | Optimized {opt_var}%")

            # Gauge
            if speed is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=speed,
                    title={'text': "Speedup Factor"},
                    gauge={
                        'axis': {'range': [0, max(5, speed + 1)]},
                        'bar': {'color': "#00F5FF"},
                        'steps': [
                            {'range': [0, 1], 'color': "rgba(239,68,68,0.2)"},
                            {'range': [1, 2], 'color': "rgba(59,130,246,0.2)"},
                            {'range': [2, max(5, speed + 1)], 'color': "rgba(34,197,94,0.2)"},
                        ],
                    }
                ))

                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=GRAPH_FONT,
                    height=350
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

            # Bar Charts
            st.markdown("### Performance Comparison")

            c1, c2 = st.columns(2)

            if orig_ms is not None and opt_ms is not None:
                df_runtime = pd.DataFrame({
                    "Version": ["Before", "After"],
                    "Runtime (ms)": [orig_ms, opt_ms]
                })
                fig_runtime = px.bar(
                    df_runtime, x="Version", y="Runtime (ms)",
                    color="Version",
                    color_discrete_map={"Before": "#EF4444", "After": "#22C55E"}
                )
                fig_runtime.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=GRAPH_FONT,
                    showlegend=False
                )
                c1.plotly_chart(fig_runtime, use_container_width=True)

            if orig_mem is not None and opt_mem is not None:
                df_mem = pd.DataFrame({
                    "Version": ["Before", "After"],
                    "Memory (MB)": [orig_mem, opt_mem]
                })
                fig_mem = px.bar(
                    df_mem, x="Version", y="Memory (MB)",
                    color="Version",
                    color_discrete_map={"Before": "#EF4444", "After": "#22C55E"}
                )
                fig_mem.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=GRAPH_FONT,
                    showlegend=False
                )
                c2.plotly_chart(fig_mem, use_container_width=True)
        else:
            st.info("No benchmark data available for this optimization.")

    # ---- TAB: Complexity ----
    with tab_complexity:
        complexity_data = result.get("complexity")
        if complexity_data:
            orig_cx = complexity_data.get("original", {})
            opt_cx = complexity_data.get("optimized", {})

            st.markdown("### Code Complexity Analysis")

            cx1, cx2 = st.columns(2)

            with cx1:
                st.markdown("**Original Code**")
                st.markdown(f"""<div class='complexity-card'>
                    <b>Cyclomatic Complexity:</b> {orig_cx.get('cyclomatic_complexity', '?')}<br>
                    <b>Max Nesting Depth:</b> {orig_cx.get('max_nesting_depth', '?')}<br>
                    <b>Loops:</b> {orig_cx.get('num_loops', '?')} | <b>Branches:</b> {orig_cx.get('num_branches', '?')}<br>
                    <b>Functions:</b> {orig_cx.get('num_functions', '?')}<br>
                    <b>Lines of Code:</b> {orig_cx.get('lines_of_code', '?')}<br>
                    <b>Big-O Estimate:</b> {html_module.escape(str(orig_cx.get('big_o_estimate', '?')))}
                </div>""", unsafe_allow_html=True)

            with cx2:
                st.markdown("**Optimized Code**")
                st.markdown(f"""<div class='complexity-card'>
                    <b>Cyclomatic Complexity:</b> {opt_cx.get('cyclomatic_complexity', '?')}<br>
                    <b>Max Nesting Depth:</b> {opt_cx.get('max_nesting_depth', '?')}<br>
                    <b>Loops:</b> {opt_cx.get('num_loops', '?')} | <b>Branches:</b> {opt_cx.get('num_branches', '?')}<br>
                    <b>Functions:</b> {opt_cx.get('num_functions', '?')}<br>
                    <b>Lines of Code:</b> {opt_cx.get('lines_of_code', '?')}<br>
                    <b>Big-O Estimate:</b> {html_module.escape(str(opt_cx.get('big_o_estimate', '?')))}
                </div>""", unsafe_allow_html=True)

            # Complexity comparison chart
            if orig_cx and opt_cx:
                metrics_list = ["cyclomatic_complexity", "max_nesting_depth", "num_loops", "lines_of_code"]
                labels = ["Cyclomatic", "Max Depth", "Loops", "LOC"]
                orig_vals = [orig_cx.get(m, 0) for m in metrics_list]
                opt_vals = [opt_cx.get(m, 0) for m in metrics_list]

                fig_cx = go.Figure(data=[
                    go.Bar(name='Original', x=labels, y=orig_vals, marker_color='#EF4444'),
                    go.Bar(name='Optimized', x=labels, y=opt_vals, marker_color='#22C55E')
                ])
                fig_cx.update_layout(
                    barmode='group',
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=GRAPH_FONT,
                    height=350,
                    title="Complexity Comparison"
                )
                st.plotly_chart(fig_cx, use_container_width=True)
        else:
            st.info("Complexity analysis not available.")

    # ---- TAB: Rules Detected ----
    with tab_rules:
        rules_detected = result.get("rules_detected", [])
        if rules_detected:
            st.markdown(f"**{len(rules_detected)} optimization pattern(s) detected:**")
            for rule in rules_detected:
                rule_name = html_module.escape(rule.get("rule", "unknown"))
                message = html_module.escape(rule.get("message", ""))
                suggestion = html_module.escape(rule.get("suggestion", ""))
                line = rule.get("line", "?")
                st.markdown(f"""
                <div class='rule-tag'>
                    <strong>{rule_name}</strong> (line {line})<br>
                    {message}<br>
                    <em>Suggestion: {suggestion}</em>
                </div>
                """, unsafe_allow_html=True)

            # Show transformations applied
            transformations = result.get("transformations", [])
            if transformations:
                st.markdown("---")
                st.markdown(f"**{len(transformations)} transformation(s) applied:**")
                for t in transformations:
                    st.success(f"Applied: {t.get('rule', '?')} - {t.get('suggestion', '')}")
        else:
            st.info("No rule-based patterns detected in this code.")

    # ---- TAB: Safety & Confidence ----
    with tab_safety:
        safety = result.get("safety_analysis")
        confidence = result.get("confidence")

        if safety:
            verdict = safety.get("verdict", "Unknown")
            is_safe = safety.get("is_safe", False)
            css_class = "safety-ok" if is_safe else "safety-warn"
            st.markdown(f"<div class='{css_class}'>Verdict: {html_module.escape(verdict)}</div>", unsafe_allow_html=True)

            warnings = safety.get("warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(f"**{w.get('type', 'Warning')}** ({w.get('severity', '?')}): {w.get('message', '')}")
            else:
                st.success("No safety warnings. Optimization is safe to apply.")
        else:
            st.info("Safety analysis not available for this mode.")

        if confidence:
            st.markdown("---")
            overall = confidence.get("overall", 0)
            level = confidence.get("confidence_level", "?")
            recommendation = confidence.get("recommendation", "?")

            st.markdown(f"**Confidence Score: {overall}/100** ({level})")
            st.progress(min(overall / 100.0, 1.0))
            st.markdown(f"**Recommendation:** {recommendation}")

            breakdown = confidence.get("breakdown", {})
            if breakdown:
                bd1, bd2, bd3 = st.columns(3)
                with bd1:
                    st.metric("Rule Certainty", f"{breakdown.get('rule_certainty', 0)}/40")
                with bd2:
                    st.metric("Speedup Gain", f"{breakdown.get('speedup_gain', 0)}/40")
                with bd3:
                    st.metric("Benchmark Stability", f"{breakdown.get('benchmark_stability', 0)}/20")
        else:
            st.info("Confidence scoring not available for this mode.")

    # ---- TAB: AI Explanation ----
    with tab_ai:
        ai_explanation = result.get("ai_explanation")
        if ai_explanation:
            st.markdown("#### AI Analysis")
            st.markdown(ai_explanation)
        else:
            st.info("AI explanation not available. Configure Gemini API key for detailed analysis.")

    # ---- Mode indicator ----
    mode = result.get("mode", "UNKNOWN")
    st.caption(f"Mode: {mode} | Timestamp: {result.get('timestamp', 'N/A')}")
