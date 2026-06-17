"""
V2G-IRL Interactive Demo
========================
Run from the project root:
    streamlit run demo/app.py
"""

import os
import sys

# ── Ensure project src/ is importable ─────────────────────────────────────
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.config import (
    SCENARIOS,
    SEGMENT,
    DISCRETE_SEGMENTS,
    METHOD_COLOURS,
    BATTERY_OPTIONS,
    HOME_CHARGER_OPTIONS,
    WORK_CHARGER_OPTIONS,
    ENERGY_PRICE_PROFILE,
)
from core.data_loader import (
    get_canonical_episodes,
    load_expert_trajectories,
    format_episode_label,
    _ts_to_hhmm,
)
from core.model_loader import get_model
from core.trajectory_runner import run_rl_trajectory, compute_metrics

# ══════════════════════════════════════════════════════════════════════════════
#  Page config & CSS injection
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="V2G-IRL Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(_CSS_PATH) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ts_to_minutes(ts: int) -> int:
    return int(ts) * 15


def _minutes_to_hhmm(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _hhmm_to_ts(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m) // 15


def _soc_to_pct(soc_list: list) -> list:
    return [v * 100 for v in soc_list]


def _build_time_axis(n: int = 96) -> list[str]:
    return [_minutes_to_hhmm(i * 15) for i in range(n)]


TIME_AXIS = _build_time_axis(96)


def _badge(method_key: str, label: str) -> str:
    return f'<span class="method-badge badge-{method_key}">{label}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ V2G-IRL Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Inverse Reinforcement Learning for EV Charging</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">Scenario</div>', unsafe_allow_html=True)
    scenario_name = st.radio(
        "scenario",
        options=list(SCENARIOS.keys()),
        label_visibility="collapsed",
        key="scenario_radio",
    )

    st.divider()

    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        options=["Expert Comparison", "Free"],
        label_visibility="collapsed",
        key="mode_radio",
    )

    st.divider()

    scenario_cfg = SCENARIOS[scenario_name]
    methods_cfg = scenario_cfg["methods"]

    st.markdown('<div class="section-label">Models to Display</div>', unsafe_allow_html=True)

    selected_methods: dict[str, bool] = {}
    for method_key, mcfg in methods_cfg.items():
        available = mcfg["available"]
        label_text = mcfg["label"]
        if available:
            selected_methods[method_key] = st.checkbox(
                label_text,
                value=True,
                key=f"chk_{scenario_name}_{method_key}",
            )
        else:
            st.markdown(
                f'<div class="method-disabled">☐ {label_text} <span style="font-size:0.7rem;color:#94A3B8">(not available)</span></div>',
                unsafe_allow_html=True,
            )
            selected_methods[method_key] = False

    # Segment selector — only for Discrete scenario
    is_discrete = scenario_name == "Discrete (Profit)"
    if is_discrete:
        st.divider()
        st.markdown('<div class="section-label">Population Segment</div>', unsafe_allow_html=True)
        active_segment = st.radio(
            "segment",
            options=DISCRETE_SEGMENTS,
            label_visibility="collapsed",
            key="segment_radio",
        )
    else:
        active_segment = SEGMENT

    st.divider()

    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            f"""
**Segment:** {active_segment}

Each IRL method learns a reward function from expert EV-charging
trajectories and trains an RL policy that mimics human behaviour.

- **MaxEnt**: Linear reward  $R = w^\\top \\phi(s,a)$
- **Deep MaxEnt**: Neural reward  $R = f_\\theta(s,a)$
- **AIRL**: Adversarial reward via discriminator
"""
        )

# ══════════════════════════════════════════════════════════════════════════════
#  Main content
# ══════════════════════════════════════════════════════════════════════════════

# ── Page header ───────────────────────────────────────────────────────────
st.markdown(f'<div class="page-title">{scenario_name}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">{scenario_cfg["description"]}  ·  Segment: <strong>{active_segment}</strong></div>',
    unsafe_allow_html=True,
)

# ── Mode panels ───────────────────────────────────────────────────────────
initial_values: dict | None = None
expert_soc: list | None = None
expert_journey_info: dict | None = None

if mode == "Expert Comparison":
    episodes = get_canonical_episodes(scenario_cfg, active_segment)

    if not episodes:
        st.error("No expert episodes found for this scenario / segment.")
        st.stop()

    col_pick, col_info = st.columns([1, 2], gap="large")

    with col_pick:
        ep_labels = [format_episode_label(ep) for ep in episodes]
        chosen_idx = st.selectbox(
            "Select Expert Episode",
            options=range(len(episodes)),
            format_func=lambda i: ep_labels[i],
            key="episode_select",
        )
        chosen_ep = episodes[chosen_idx]

    with col_info:
        iv = chosen_ep["initial_values"]
        bat_kw = BATTERY_OPTIONS.get(iv.get("battery_capacity", 0), "—")
        home_kw = HOME_CHARGER_OPTIONS.get(iv.get("home_charge_power", 0), "—")
        work_kw = WORK_CHARGER_OPTIONS.get(iv.get("work_charge_power", 0), "—")
        soc_pct = round(iv.get("soc", 0) * 100, 1)
        dep_hhmm = _ts_to_hhmm(iv.get("out_start_timestep", 0))
        ret_hhmm = _ts_to_hhmm(iv.get("return_start_timestep", 0))
        dist_mi = round(iv.get("journey_distance", 0), 1)
        out_spd = round(iv.get("out_journey_speed", 0), 1)
        ret_spd = round(iv.get("return_journey_speed", 0), 1)

        st.markdown(
            f"""
<div class="ic-card">
  <div class="ic-card-grid">
    <div><div class="ic-item-label">Initial SoC</div><div class="ic-item-value">{soc_pct}%</div></div>
    <div><div class="ic-item-label">Battery</div><div class="ic-item-value">{bat_kw}</div></div>
    <div><div class="ic-item-label">Home Charger</div><div class="ic-item-value">{home_kw}</div></div>
    <div><div class="ic-item-label">Work Charger</div><div class="ic-item-value">{work_kw}</div></div>
    <div><div class="ic-item-label">Departure</div><div class="ic-item-value">{dep_hhmm}</div></div>
    <div><div class="ic-item-label">Return</div><div class="ic-item-value">{ret_hhmm}</div></div>
    <div><div class="ic-item-label">Distance</div><div class="ic-item-value">{dist_mi} mi</div></div>
    <div><div class="ic-item-label">Out Speed</div><div class="ic-item-value">{out_spd} mph</div></div>
    <div><div class="ic-item-label">Ret Speed</div><div class="ic-item-value">{ret_spd} mph</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    initial_values = chosen_ep["initial_values"]
    expert_soc = chosen_ep["soc_history"]
    expert_journey_info = {
        "out_start_timestep":    iv.get("out_start_timestep", 0),
        "return_start_timestep": iv.get("return_start_timestep", 0),
    }

else:  # Free mode
    st.markdown("#### Initial Conditions")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        soc_input = st.slider("Initial SoC (%)", 0, 100, 50, step=5, key="free_soc")
        battery_cap_idx = st.selectbox(
            "Battery Capacity",
            options=[0, 1, 2],
            format_func=lambda i: BATTERY_OPTIONS[i],
            index=1,
            key="free_bat",
        )

    with col2:
        home_charger_idx = st.selectbox(
            "Home Charger Power",
            options=[0, 1, 2],
            format_func=lambda i: HOME_CHARGER_OPTIONS[i],
            index=1,
            key="free_home_chg",
        )
        work_charger_idx = st.selectbox(
            "Work Charger Power",
            options=[0, 1, 2],
            format_func=lambda i: WORK_CHARGER_OPTIONS[i],
            index=1,
            key="free_work_chg",
        )

    with col3:
        # Departure / return as HH:MM string selectors (15-min grid)
        time_options = [_minutes_to_hhmm(i * 15) for i in range(96)]
        dep_hhmm = st.selectbox(
            "Departure Time",
            options=time_options,
            index=28,    # ~07:00
            key="free_dep",
        )
        ret_hhmm = st.selectbox(
            "Return Time",
            options=time_options,
            index=68,    # ~17:00
            key="free_ret",
        )

    col4, col5 = st.columns(2, gap="large")
    with col4:
        journey_dist = st.slider("Journey Distance (miles)", 1.0, 80.0, 15.0, step=0.5, key="free_dist")
    with col5:
        out_speed = st.slider("Outbound Speed (mph)", 10.0, 70.0, 35.0, step=1.0, key="free_out_spd")
        ret_speed = st.slider("Return Speed (mph)", 10.0, 70.0, 35.0, step=1.0, key="free_ret_spd")

    out_ts = _hhmm_to_ts(dep_hhmm)
    ret_ts = _hhmm_to_ts(ret_hhmm)

    initial_values = {
        "soc":                  soc_input / 100.0,
        "battery_capacity":     battery_cap_idx,
        "home_charge_power":    home_charger_idx,
        "work_charge_power":    work_charger_idx,
        "journey_distance":     journey_dist,
        "out_journey_speed":    out_speed,
        "return_journey_speed": ret_speed,
        "out_start_timestep":   out_ts,
        "return_start_timestep": ret_ts,
    }
    expert_journey_info = {
        "out_start_timestep":    out_ts,
        "return_start_timestep": ret_ts,
    }

# ── Run button ────────────────────────────────────────────────────────────
active_methods = [m for m, sel in selected_methods.items() if sel]

run_col, _ = st.columns([1, 5])
with run_col:
    run_clicked = st.button("▶  Run Simulation", type="primary", use_container_width=True)
if "trajectories" not in st.session_state:
    st.session_state["trajectories"] = {}
if "last_scenario" not in st.session_state:
    st.session_state["last_scenario"] = None
if "last_segment" not in st.session_state:
    st.session_state["last_segment"] = None

# Clear results when scenario or segment changes
if (st.session_state["last_scenario"] != scenario_name
        or st.session_state["last_segment"] != active_segment):
    st.session_state["trajectories"] = {}
    st.session_state["last_scenario"] = scenario_name
    st.session_state["last_segment"] = active_segment

if run_clicked:
    if not active_methods:
        st.warning("Please select at least one model to run.")
    else:
        results: dict[str, dict] = {}
        errors: list[str] = []

        for method_key in active_methods:
            mcfg = methods_cfg[method_key]
            method_label = mcfg["label"]

            with st.spinner(f"Loading {method_label}…"):
                try:
                    model, env_factory, _, reward_net, shaping_net = get_model(
                        scenario_name, method_key, active_segment
                    )
                except Exception as exc:
                    errors.append(f"**{method_label}**: {exc}")
                    continue

            with st.spinner(f"Running {method_label}…"):
                try:
                    traj = run_rl_trajectory(
                        model,
                        env_factory,
                        initial_values,
                        reward_net=reward_net,
                        shaping_net=shaping_net,
                    )
                    results[method_key] = traj
                except Exception as exc:
                    errors.append(f"**{method_label}** inference: {exc}")

        st.session_state["trajectories"] = results
        if errors:
            for e in errors:
                st.error(e)

trajectories: dict[str, dict] = st.session_state.get("trajectories", {})

# ══════════════════════════════════════════════════════════════════════════════
#  Plotly chart
# ══════════════════════════════════════════════════════════════════════════════

if trajectories or (mode == "Expert Comparison" and expert_soc):

    fig = go.Figure()

    # ── Determine journey markers ──────────────────────────────────────────
    # Use info from the first successful RL trajectory, fall back to initial_values
    first_traj = next(iter(trajectories.values()), None) if trajectories else None
    out_ts_plot = int(expert_journey_info["out_start_timestep"])
    ret_ts_plot = int(expert_journey_info["return_start_timestep"])
    out_dur = int(first_traj["out_duration"]) if first_traj else 4
    ret_dur = int(first_traj["return_duration"]) if first_traj else 4

    out_end_ts = min(out_ts_plot + out_dur, 95)
    ret_end_ts = min(ret_ts_plot + ret_dur, 95)

    # ── Shaded journey bands ───────────────────────────────────────────────
    fig.add_vrect(
        x0=TIME_AXIS[out_ts_plot], x1=TIME_AXIS[out_end_ts],
        fillcolor="rgba(239,68,68,0.10)", line_width=0,
        annotation_text="Outbound", annotation_position="top left",
        annotation=dict(font=dict(size=10, color="#EF4444"), xanchor="left"),
    )
    fig.add_vrect(
        x0=TIME_AXIS[ret_ts_plot], x1=TIME_AXIS[ret_end_ts],
        fillcolor="rgba(59,130,246,0.10)", line_width=0,
        annotation_text="Return", annotation_position="top left",
        annotation=dict(font=dict(size=10, color="#3B82F6"), xanchor="left"),
    )

    # ── SoC operating band (20%–80%) ──────────────────────────────────────
    fig.add_hrect(
        y0=20, y1=80,
        fillcolor="rgba(16,185,129,0.04)", line_width=0,
    )
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(16,185,129,0.4)", line_width=1)
    fig.add_hline(y=80, line_dash="dot", line_color="rgba(16,185,129,0.4)", line_width=1)

    # ── Expert trajectory ──────────────────────────────────────────────────
    if mode == "Expert Comparison" and expert_soc:
        soc_pct_exp = _soc_to_pct(expert_soc)
        fig.add_trace(go.Scatter(
            x=TIME_AXIS[:len(soc_pct_exp)],
            y=soc_pct_exp,
            name="Expert",
            line=dict(color=METHOD_COLOURS["expert"], width=2.5, dash="dash"),
            mode="lines",
        ))

    # ── RL model trajectories ──────────────────────────────────────────────
    for method_key, traj_data in trajectories.items():
        label = methods_cfg[method_key]["label"]
        colour = METHOD_COLOURS.get(method_key, "#888888")
        soc_hist = traj_data["soc_history"]
        soc_pct_rl = _soc_to_pct(soc_hist)
        fig.add_trace(go.Scatter(
            x=TIME_AXIS[:len(soc_pct_rl)],
            y=soc_pct_rl,
            name=label,
            line=dict(color=colour, width=2.5),
            mode="lines",
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=420,
        margin=dict(l=48, r=16, t=32, b=48),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
        xaxis=dict(
            title="Time of Day",
            tickangle=-30,
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            tickvals=TIME_AXIS[::8],   # every 2 hours
            ticktext=TIME_AXIS[::8],
        ),
        yaxis=dict(
            title="State of Charge (%)",
            range=[-2, 102],
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Metrics (Expert Comparison mode only, after run)
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Expert Comparison" and expert_soc and trajectories:
    st.markdown("#### Model Comparison")
    badges_html = " ".join(_badge(mk, methods_cfg[mk]["label"]) for mk in trajectories)
    st.markdown(badges_html, unsafe_allow_html=True)
    st.write("")

    metric_cols = st.columns(len(trajectories), gap="medium")
    for col, (method_key, traj_data) in zip(metric_cols, trajectories.items()):
        label = methods_cfg[method_key]["label"]
        colour = METHOD_COLOURS.get(method_key, "#888888")
        try:
            metrics = compute_metrics(traj_data["soc_history"], expert_soc)
            with col:
                st.markdown(
                    f"""
<div class="metric-card">
  <div class="metric-card-title" style="color:{colour}">{label}</div>
  <div class="ic-card-grid" style="grid-template-columns:1fr 1fr;gap:0.5rem 1rem;margin-top:0.4rem">
    <div>
      <div class="ic-item-label">MAE</div>
      <div class="metric-card-value" style="font-size:1.2rem">{metrics['mae']:.4f}</div>
      <div class="metric-card-sub">mean abs error</div>
    </div>
    <div>
      <div class="ic-item-label">DTW</div>
      <div class="metric-card-value" style="font-size:1.2rem">{metrics['dtw']:.3f}</div>
      <div class="metric-card-sub">dyn. time warp</div>
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            with col:
                st.warning(f"Metric error for {label}: {e}")
