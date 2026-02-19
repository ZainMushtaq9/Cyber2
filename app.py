"""
app.py
======
Streamlit web application for the Agentic AI Smart Grid framework.

Aesthetic: Dark industrial / control-room — deep charcoal backgrounds,
amber accent, monospaced data readouts, clean grid lines.
"""

import math
import random
import time

import numpy as np
import streamlit as st

# ── Page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grid · Agentic AI Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Framework imports ────────────────────────────────────────────────────────
from agentic_framework import AgenticFramework
from adaptive_controller import AdaptiveController
from evaluation_pipeline import _default_data_generator

# ============================================================================
# Custom CSS — industrial control-room dark theme
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@400;700&display=swap');

/* ── Global reset ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0f13;
    color: #c8cdd6;
}

/* ── Main container ─────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #13161c;
    border-right: 1px solid #1f2430;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Header / brand ─────────────────────────────────────────────────── */
.brand-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #f0a500;
    text-transform: uppercase;
    border-bottom: 1px solid #1f2430;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
}
.brand-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 0.15em;
    margin-top: -0.8rem;
    margin-bottom: 1.2rem;
}

/* ── Section labels ─────────────────────────────────────────────────── */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #f0a500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* ── Cards ──────────────────────────────────────────────────────────── */
.sg-card {
    background: #13161c;
    border: 1px solid #1f2430;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.sg-card-accent {
    border-left: 3px solid #f0a500;
}
.sg-card-green { border-left: 3px solid #22c55e; }
.sg-card-blue  { border-left: 3px solid #38bdf8; }
.sg-card-red   { border-left: 3px solid #f43f5e; }
.sg-card-purple{ border-left: 3px solid #a78bfa; }

/* ── Metric overrides ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #13161c;
    border: 1px solid #1f2430;
    border-radius: 4px;
    padding: 0.7rem 1rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #6b7280 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #f0a500 !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0d0f13;
    border-bottom: 1px solid #1f2430;
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4a5568;
    border-bottom: 2px solid transparent;
    padding: 0.5rem 1.2rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #f0a500 !important;
    border-bottom: 2px solid #f0a500 !important;
    background: transparent;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #f0a500;
    color: #0d0f13;
    border: none;
    border-radius: 2px;
    padding: 0.45rem 1.4rem;
    font-weight: 700;
    transition: background 0.15s;
}
.stButton > button:hover { background: #fbbf24; color: #0d0f13; }

/* ── Code / mono text ───────────────────────────────────────────────── */
code, .mono {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
}

/* ── Status badges ──────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    padding: 0.15rem 0.6rem;
    border-radius: 2px;
    text-transform: uppercase;
    font-weight: 600;
}
.badge-ok      { background: #14532d; color: #4ade80; }
.badge-warn    { background: #78350f; color: #fcd34d; }
.badge-alert   { background: #4c0519; color: #fb7185; }
.badge-explore { background: #1e1b4b; color: #a5b4fc; }
.badge-exploit { background: #064e3b; color: #6ee7b7; }

/* ── Progress / separator ───────────────────────────────────────────── */
hr { border-color: #1f2430; }

/* ── Expander ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1f2430 !important;
    border-radius: 4px;
    background: #0d0f13;
}

/* ── JSON viewer ────────────────────────────────────────────────────── */
[data-testid="stJson"] { font-size: 0.78rem; }

/* ── Sidebar widgets ────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session state initialisation
# ============================================================================

def init_state():
    if "framework" not in st.session_state:
        st.session_state.framework = AgenticFramework()
    if "controller" not in st.session_state:
        st.session_state.controller = None
    if "adaptive_enabled" not in st.session_state:
        st.session_state.adaptive_enabled = False
    if "cycle_outputs" not in st.session_state:
        st.session_state.cycle_outputs = []
    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "data_gen" not in st.session_state:
        st.session_state.data_gen = _default_data_generator()
    if "adaptation_count" not in st.session_state:
        st.session_state.adaptation_count = 0
    if "prev_config" not in st.session_state:
        st.session_state.prev_config = {}


def get_or_create_controller():
    if st.session_state.controller is None:
        st.session_state.controller = AdaptiveController(
            framework=st.session_state.framework
        )
    return st.session_state.controller


init_state()
fw: AgenticFramework = st.session_state.framework


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown('<div class="brand-header">⚡ Smart Grid</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AGENTIC AI MONITOR · v2.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
    use_synthetic = st.toggle("Synthetic Data Generator", value=True)

    if not use_synthetic:
        st.markdown('<div class="section-label">Manual Sensor Values</div>', unsafe_allow_html=True)
        manual_vals_str = st.text_area(
            "Values (comma-separated floats)",
            value="1.02, 0.98, 1.01, 1.05, 0.99, 1.03, 0.97, 1.00",
            height=70,
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Execution</div>', unsafe_allow_html=True)
    n_cycles = st.slider("Cycles to run", 1, 100, 10)

    st.markdown("---")
    st.markdown('<div class="section-label">Adaptive Learning</div>', unsafe_allow_html=True)
    adaptive_on = st.toggle("Enable Adaptive Controller", value=st.session_state.adaptive_enabled)
    st.session_state.adaptive_enabled = adaptive_on
    if adaptive_on:
        get_or_create_controller()
        st.markdown(
            f'<span class="badge badge-ok">ACTIVE</span> &nbsp; '
            f'<span class="mono">adaptations: {st.session_state.adaptation_count}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state.controller = None
        st.markdown('<span class="badge badge-warn">DISABLED</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Runtime Config Override</div>', unsafe_allow_html=True)
    with st.expander("Manually set parameters"):
        new_sens = st.number_input(
            "behavioral_sensitivity",
            0.5, 8.0,
            float(fw.runtime_config.behavioral_sensitivity), 0.1,
        )
        new_cont = st.number_input(
            "anomaly_contamination",
            0.01, 0.30,
            float(fw.runtime_config.anomaly_contamination), 0.005,
            format="%.3f",
        )
        new_casc = st.number_input(
            "cascade_propagation_threshold",
            0.1, 0.95,
            float(fw.runtime_config.cascade_propagation_threshold), 0.05,
        )
        new_expl = st.number_input(
            "strategy_exploration_rate",
            0.02, 0.50,
            float(fw.runtime_config.strategy_exploration_rate), 0.01,
        )
        if st.button("Apply Config"):
            fw.update_runtime_config({
                "behavioral_sensitivity": new_sens,
                "anomaly_contamination": new_cont,
                "cascade_propagation_threshold": new_casc,
                "strategy_exploration_rate": new_expl,
            })
            st.success("Config updated!")

    st.markdown("---")
    if st.button("🔄  Reset Framework"):
        for k in ["framework", "controller", "cycle_outputs", "last_output",
                   "data_gen", "adaptation_count", "prev_config"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown(
        f'<div class="mono" style="color:#2d3748;font-size:0.65rem;margin-top:1rem;">'
        f'CYCLES RUN: {fw.cycle_count}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Header
# ============================================================================

col_h1, col_h2, col_h3, col_h4 = st.columns([3, 1, 1, 1])
with col_h1:
    st.markdown(
        '<h1 style="font-family:\'Barlow Condensed\',sans-serif;font-size:2rem;'
        'font-weight:700;letter-spacing:0.1em;color:#f0a500;margin:0;">'
        '⚡ SMART GRID AGENTIC MONITOR</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mono" style="color:#4a5568;margin-top:0.1rem;">'
        'Real-time multi-agent observability · adaptive runtime configuration</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.metric("TOTAL CYCLES", fw.cycle_count)
with col_h3:
    adapt_cnt = st.session_state.adaptation_count
    st.metric("ADAPTATIONS", adapt_cnt)
with col_h4:
    cfg_changes = fw.runtime_config.change_log_length()
    st.metric("CONFIG CHANGES", cfg_changes)

st.markdown("---")

# ============================================================================
# Run buttons
# ============================================================================

def make_sensor_data():
    if use_synthetic:
        return st.session_state.data_gen()
    else:
        try:
            vals = [float(x.strip()) for x in manual_vals_str.split(",")]
        except Exception:
            vals = [1.0] * 8
        return {
            "values": vals,
            "features": [[v, v**2, math.sin(v)] for v in vals],
            "timestamp": fw.cycle_count,
            "source_nodes": [],
        }


def run_cycles(n: int):
    prev_cfg = fw.runtime_config.snapshot()
    st.session_state.prev_config = prev_cfg

    bar = st.progress(0, text="Running cycles…")
    for i in range(n):
        data = make_sensor_data()
        out = fw.run_cycle(data)
        st.session_state.cycle_outputs.append(out)
        st.session_state.last_output = out

        if st.session_state.adaptive_enabled and st.session_state.controller:
            st.session_state.controller.adapt(out)
            st.session_state.adaptation_count += 1

        bar.progress((i + 1) / n, text=f"Cycle {fw.cycle_count} / {fw.cycle_count + n - i - 1}")
        time.sleep(0.02)  # slight pause so progress is visible

    bar.empty()


col_run1, col_run2, col_spacer = st.columns([1, 1, 5])
with col_run1:
    if st.button("▶  RUN CYCLE", use_container_width=True):
        run_cycles(1)
        st.rerun()
with col_run2:
    if st.button(f"▶▶  RUN {n_cycles} CYCLES", use_container_width=True):
        run_cycles(n_cycles)
        st.rerun()

# ============================================================================
# Guard: nothing run yet
# ============================================================================

if st.session_state.last_output is None:
    st.markdown(
        '<div class="sg-card sg-card-accent" style="text-align:center;padding:2.5rem;">'
        '<span class="mono" style="color:#4a5568;">NO DATA YET — RUN A CYCLE TO BEGIN</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

out = st.session_state.last_output
prev_cfg = st.session_state.prev_config
cur_cfg = fw.runtime_config.to_dict()


# ============================================================================
# Helper renderers
# ============================================================================

def latency_ms(agent_id: str) -> str:
    hist = fw.get_agents()
    for a in hist:
        if a.agent_id == agent_id:
            h = a.get_history(last_n=1)
            if h:
                return f"{h[-1]['elapsed_ms']:.2f} ms"
    return "—"


def config_changed(key: str) -> bool:
    return prev_cfg.get(key) != cur_cfg.get(key)


def delta_str(key: str) -> str:
    if prev_cfg.get(key) is not None and config_changed(key):
        diff = cur_cfg[key] - prev_cfg[key]
        return f"{diff:+.4f}"
    return None


# ============================================================================
# TABS
# ============================================================================

tabs = st.tabs([
    "🔬 Envelope Agent",
    "🧠 Anomaly Detection",
    "🌊 Cascade Predictor",
    "🛡️ Mitigation",
    "⚙️ Runtime Config",
    "📊 Performance",
])


# ── TAB 1: BehavioralEnvelopeAgent ──────────────────────────────────────────
with tabs[0]:
    env = out.get("envelope", {})
    st.markdown('<div class="section-label">Behavioral Envelope Agent</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANOMALY COUNT", env.get("anomaly_count", 0))
    c2.metric("VALUES PROCESSED", env.get("values_processed", 0))
    c3.metric("THRESHOLD σ", f"{env.get('threshold_sigma', 0):.2f}")
    c4.metric("LATENCY", latency_ms("behavioral_envelope"))

    anomalies = env.get("anomalies", [])
    if anomalies:
        st.markdown(
            '<span class="badge badge-alert">ANOMALIES DETECTED</span>',
            unsafe_allow_html=True,
        )
        for a in anomalies:
            st.markdown(
                f'<div class="sg-card sg-card-red">'
                f'<span class="mono">Index: <b>{a["index"]}</b> &nbsp;|&nbsp; '
                f'Value: <b>{a["value"]:.4f}</b> &nbsp;|&nbsp; '
                f'Z-Score: <b>{a["z_score"]:.4f}</b> &nbsp;|&nbsp; '
                f'Mean: {a["mean"]:.4f} &nbsp;|&nbsp; '
                f'Std: {a["std"]:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="badge badge-ok">ALL VALUES WITHIN ENVELOPE</span>',
            unsafe_allow_html=True,
        )

    with st.expander("Raw agent output"):
        st.json(env)


# ── TAB 2: AnomalyDetectionAgent ────────────────────────────────────────────
with tabs[1]:
    ano = out.get("anomaly", {})
    st.markdown('<div class="section-label">Anomaly Detection Agent (IsolationForest)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ANOMALY COUNT", ano.get("anomaly_count", 0))
    c2.metric("CONTAMINATION", f"{ano.get('contamination_used', 0):.4f}")
    c3.metric("SAMPLES IN BUFFER", ano.get("samples_in_buffer", 0))
    c4.metric("MODEL", ano.get("model_type", "—"))
    c5.metric("LATENCY", latency_ms("anomaly_detection"))

    preds = ano.get("predictions", [])
    scores = ano.get("scores", [])
    if preds:
        st.markdown("**Prediction map** (−1 = anomaly, +1 = normal)")
        pred_html = "".join([
            f'<span class="badge {'badge-alert' if p == -1 else 'badge-ok'}" '
            f'style="margin:2px;">{i}:{p}</span>'
            for i, p in enumerate(preds)
        ])
        st.markdown(pred_html, unsafe_allow_html=True)

        if scores:
            st.markdown("**Anomaly scores**")
            score_html = " ".join([
                f'<span class="mono" style="color:{"#f43f5e" if p==-1 else "#4ade80"};">'
                f'{s:.4f}</span>'
                for s, p in zip(scores, preds)
            ])
            st.markdown(score_html, unsafe_allow_html=True)

    with st.expander("Raw agent output"):
        st.json(ano)


# ── TAB 3: CascadePredictorAgent ─────────────────────────────────────────────
with tabs[2]:
    cas = out.get("cascade", {})
    st.markdown('<div class="section-label">Cascade Predictor Agent</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AFFECTED NODES", cas.get("affected_count", 0))
    c2.metric("CASCADE DEPTH", cas.get("cascade_depth", 0))
    c3.metric("THRESHOLD", f"{cas.get('threshold_used', 0):.3f}")
    c4.metric("LATENCY", latency_ms("cascade_predictor"))

    src = cas.get("source_nodes", [])
    aff = cas.get("affected_nodes", [])
    paths = cas.get("propagation_paths", [])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Source Nodes</div>', unsafe_allow_html=True)
        if src:
            st.markdown(
                " ".join([f'<span class="badge badge-alert">{n}</span>' for n in src]),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="badge badge-ok">NONE</span>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-label">Affected Nodes</div>', unsafe_allow_html=True)
        if aff:
            st.markdown(
                " ".join([f'<span class="badge badge-warn">{n}</span>' for n in aff]),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="badge badge-ok">NONE</span>', unsafe_allow_html=True)

    if paths:
        st.markdown('<div class="section-label" style="margin-top:1rem;">Propagation Paths</div>', unsafe_allow_html=True)
        for i, path in enumerate(paths):
            chain = " → ".join([f'<b>{n}</b>' for n in path])
            st.markdown(
                f'<div class="sg-card sg-card-blue"><span class="mono">Path {i+1}: {chain}</span></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Raw agent output"):
        st.json(cas)


# ── TAB 4: MitigationGeneratorAgent ──────────────────────────────────────────
with tabs[3]:
    mit = out.get("mitigation", {})
    st.markdown('<div class="section-label">Mitigation Generator Agent (ε-greedy)</div>', unsafe_allow_html=True)

    mode = mit.get("selection_mode", "")
    badge_cls = "badge-explore" if mode == "exploration" else "badge-exploit"
    badge_txt = "EXPLORATION" if mode == "exploration" else "EXPLOITATION"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STRATEGY", mit.get("selected_strategy", "—"))
    c2.metric("EPSILON ε", f"{mit.get('exploration_rate', 0):.3f}")
    c3.metric("THREAT PROFILE", mit.get("threat_profile", "—"))
    c4.metric("LATENCY", latency_ms("mitigation_generator"))

    st.markdown(
        f'<span class="badge {badge_cls}">{badge_txt}</span>',
        unsafe_allow_html=True,
    )

    top = mit.get("top_strategies", [])
    if top:
        st.markdown('<div class="section-label" style="margin-top:1rem;">Top Strategies (Q-Values)</div>', unsafe_allow_html=True)
        for rank, item in enumerate(top, 1):
            bar_w = max(0, min(100, int((item["q_value"] + 1) * 50)))
            st.markdown(
                f'<div class="sg-card sg-card-purple" style="padding:0.6rem 1rem;">'
                f'<span class="mono" style="color:#a78bfa;">#{rank}</span> &nbsp;'
                f'<span class="mono">{item["strategy"]}</span>'
                f'<span style="float:right;color:#a78bfa;" class="mono">Q={item["q_value"]:.4f}</span>'
                f'<div style="background:#1f2430;border-radius:2px;margin-top:0.3rem;height:4px;">'
                f'<div style="background:#a78bfa;width:{bar_w}%;height:4px;border-radius:2px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    actions = mit.get("actions", [])
    if actions:
        st.markdown('<div class="section-label" style="margin-top:1rem;">Action Plan</div>', unsafe_allow_html=True)
        st.json(actions)

    with st.expander("Raw agent output"):
        st.json(mit)


# ── TAB 5: Runtime Configuration ────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-label">Runtime Configuration — Single Source of Truth</div>', unsafe_allow_html=True)

    params = [
        ("behavioral_sensitivity",        "σ SENSITIVITY",    "#f0a500"),
        ("anomaly_contamination",         "CONTAMINATION",    "#38bdf8"),
        ("cascade_propagation_threshold", "CASCADE THRESH.",  "#fb923c"),
        ("strategy_exploration_rate",     "EXPLORATION ε",    "#a78bfa"),
    ]

    cols = st.columns(4)
    for col, (key, label, color) in zip(cols, params):
        val = cur_cfg.get(key, 0)
        changed = config_changed(key)
        delta = delta_str(key)
        with col:
            st.metric(label, f"{val:.4f}", delta=delta)
            if changed:
                st.markdown(
                    '<span class="badge badge-warn">UPDATED</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-label">Change Log</div>', unsafe_allow_html=True)
        change_log = fw.runtime_config.get_change_log()
        if change_log:
            for entry in change_log[-8:]:
                st.markdown(
                    f'<div class="sg-card sg-card-accent">'
                    f'<span class="mono" style="color:#f0a500;">Cycle #{entry["cycle"]}</span><br/>'
                    + "".join([
                        f'<span class="mono" style="font-size:0.78rem;">'
                        f'{k}: {v["old"]} → {v["new"]}</span><br/>'
                        for k, v in entry["changes"].items()
                    ])
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span class="mono" style="color:#4a5568;">No changes yet.</span>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Current Config (JSON)</div>', unsafe_allow_html=True)
        st.json(cur_cfg)
        st.metric("CHANGE LOG ENTRIES", fw.runtime_config.change_log_length())
        if st.session_state.adaptive_enabled:
            st.metric("ADAPTATION COUNT", st.session_state.adaptation_count)


# ── TAB 6: Performance Monitor ───────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-label">Agent Performance Monitor</div>', unsafe_allow_html=True)

    report = fw.performance_monitor.generate_report()

    if "error" in report:
        st.warning(report["error"])
    else:
        # Per-agent stats
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Per-Agent Latency</div>', unsafe_allow_html=True)
        agent_data = report.get("agents", {})
        agent_cols = st.columns(len(agent_data)) if agent_data else []
        for col, (agent_id, stats) in zip(agent_cols, agent_data.items()):
            with col:
                short = agent_id.replace("_", " ").title()
                mean_lat = stats.get("mean_latency_ms")
                st.metric(
                    short,
                    f"{mean_lat:.3f} ms" if mean_lat is not None else "—",
                    help=f"Calls: {stats.get('total_calls', 0)}",
                )

        st.markdown("---")

        # Detection stats + config convergence side by side
        col_d, col_c = st.columns(2)
        with col_d:
            st.markdown('<div class="section-label">Detection Statistics</div>', unsafe_allow_html=True)
            det = report.get("detection", {})
            st.metric("Mean Anomalies / Cycle", det.get("mean_anomalies_per_cycle", "—"))
            st.metric("Mean Cascade Depth", det.get("mean_cascade_depth", "—"))

        with col_c:
            st.markdown('<div class="section-label">Config Convergence (StdDev)</div>', unsafe_allow_html=True)
            drift = report.get("config_drift", {})
            st.metric("Config Adjustments", drift.get("adjustment_count", 0))
            stdevs = drift.get("param_stdev", {})
            for k, v in stdevs.items():
                short_k = k.replace("_", " ").title()
                st.metric(short_k, f"{v:.6f}" if v is not None else "—")

        st.markdown("---")

        # Mitigation distribution
        st.markdown('<div class="section-label">Mitigation Strategy Distribution</div>', unsafe_allow_html=True)
        dist = report.get("mitigation_distribution", {})
        if dist:
            total_mit = sum(dist.values())
            for strategy, count in sorted(dist.items(), key=lambda x: -x[1]):
                pct = count / total_mit * 100
                bar_w = int(pct)
                st.markdown(
                    f'<div class="sg-card" style="padding:0.5rem 1rem;">'
                    f'<span class="mono">{strategy}</span>'
                    f'<span style="float:right;" class="mono">{count}x &nbsp; {pct:.1f}%</span>'
                    f'<div style="background:#1f2430;border-radius:2px;margin-top:0.35rem;height:5px;">'
                    f'<div style="background:#f0a500;width:{bar_w}%;height:5px;border-radius:2px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Full monitor report (JSON)"):
            st.json(report)


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    '<div class="mono" style="color:#2d3748;font-size:0.65rem;text-align:center;">'
    'AGENTIC AI SMART GRID · MULTI-AGENT MONITORING SYSTEM · '
    'RUNTIME ADAPTIVE ARCHITECTURE</div>',
    unsafe_allow_html=True,
)
