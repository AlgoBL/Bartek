"""11_Regime_Clock.py — Zegar Biznesowy (Merrill Lynch Investment Clock)"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from modules.styling import apply_styling
from modules.macro_regime_clock import (
    CLOCK_PHASES, classify_clock_phase, compute_regime_from_macro,
    historical_performance_table, clock_position_coords,
)

st.set_page_config(page_title="Macro Regime Clock", page_icon="🕐", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🕐 Macro Regime Clock")
st.markdown("*Merrill Lynch Investment Clock — 4 fazy cyklu biznesowego i optymalna alokacja aktywów*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Sygnały Makro")
    st.markdown("*Dostosuj na podstawie aktualnych danych z Control Center*")
    yield_curve = st.slider("Yield Curve (10Y-2Y, %)", -2.0, 3.0, 0.3, 0.1)
    copper_trend = st.slider("Trend Miedzi (mom %)", -10.0, 10.0, 2.0, 0.5)
    real_yield = st.slider("Real Yield (TIPS 10Y, %)", -3.0, 4.0, 1.5, 0.1)
    hy_spread = st.slider("HY Spread (bps)", 200, 1000, 380, 10)
    manual_override = st.checkbox("Ręczny override fazy", value=False)
    if manual_override:
        manual_phase = st.selectbox("Faza", list(CLOCK_PHASES.keys()))

# Compute phase
macro_snap = {
    "yield_curve_10_2": yield_curve,
    "copper_trend": copper_trend / 100,
    "real_yield": real_yield,
    "hy_oas": hy_spread,
}
result = compute_regime_from_macro(macro_snap)
phase = manual_phase if manual_override else result.get("phase", "Recovery")
phase_info = CLOCK_PHASES[phase]
gdp_sig = result.get("gdp_signal", 0)
infl_sig = result.get("inflation_signal", 0)
confidence = result.get("confidence", 0.5)

# Header
c1, c2, c3, c4 = st.columns(4)
pc = phase_info["color"]
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">FASE CYKLU</div>
        <div style="font-size:28px;">{phase_info['emoji']}</div>
        <div class="metric-value" style="color:{pc};font-size:18px;">{phase}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    cc = "#00e676" if gdp_sig > 0.1 else "#ff1744"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SYGNAŁ WZROSTU</div>
        <div class="metric-value" style="color:{cc}">{'Przyspieszanie ↑' if gdp_sig > 0 else 'Spowolnienie ↓'}</div>
        <div style="font-size:12px;color:#6b7280">score: {gdp_sig:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    ic = "#ff1744" if infl_sig > 0.1 else "#00e676"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SYGNAŁ INFLACJI</div>
        <div class="metric-value" style="color:{ic}">{'Rosnąca ↑' if infl_sig > 0 else 'Malejąca ↓'}</div>
        <div style="font-size:12px;color:#6b7280">score: {infl_sig:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    sig = phase_info["signal"]
    sc = "#00e676" if sig == "RISK_ON" else "#ff1744" if sig == "RISK_OFF" else "#ffea00"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SIGNAL</div>
        <div class="metric-value" style="color:{sc}">{sig}</div>
        <div style="font-size:12px;color:#6b7280">pewność: {confidence:.0%}</div>
    </div>""", unsafe_allow_html=True)

st.divider()
col_chart, col_info = st.columns([1, 1])

with col_chart:
    # Draw the investment clock
    theta_map = {"Recovery": 45, "Overheat": 135, "Stagflation": 225, "Reflation": 315}
    fig = go.Figure()

    # 4 sectors
    sector_colors = {"Recovery": "rgba(0,230,118,0.12)", "Overheat": "rgba(255,234,0,0.12)",
                     "Stagflation": "rgba(255,23,68,0.12)", "Reflation": "rgba(0,204,255,0.12)"}
    for i, (pname, pinfo) in enumerate(CLOCK_PHASES.items()):
        angle_start = i * 90
        angles = np.linspace(np.radians(angle_start), np.radians(angle_start + 90), 30)
        x_arc = np.concatenate([[0], np.cos(angles), [0]])
        y_arc = np.concatenate([[0], np.sin(angles), [0]])
        fig.add_trace(go.Scatter(
            x=x_arc, y=y_arc, fill="toself",
            fillcolor=sector_colors.get(pname, "rgba(100,100,100,0.1)"),
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            name=pname, showlegend=True,
            mode="lines",
        ))
        # Phase label
        langle = angle_start + 45
        lx = 0.65 * np.cos(np.radians(langle))
        ly = 0.65 * np.sin(np.radians(langle))
        fig.add_annotation(x=lx, y=ly, text=f"{pinfo['emoji']}<br><b>{pname}</b>",
                           showarrow=False, font=dict(size=12, color=pinfo["color"]))

    # Clock axes
    for angle, label in [(0, "GDP+"), (90, "INF+"), (180, "GDP−"), (270, "INF−")]:
        x_end = np.cos(np.radians(angle)) * 1.05
        y_end = np.sin(np.radians(angle)) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=x_end * 0.95, y1=y_end * 0.95,
                      line=dict(color="rgba(255,255,255,0.2)", width=1))
        fig.add_annotation(x=x_end, y=y_end, text=label,
                           showarrow=False, font=dict(size=11, color="#9ca3af"))

    # Current position (clock hand)
    coords = clock_position_coords(gdp_sig, infl_sig)
    fx, fy = coords["x"] * 0.75, coords["y"] * 0.75
    fig.add_trace(go.Scatter(
        x=[0, fx], y=[0, fy],
        mode="lines+markers",
        line=dict(color=pc, width=4),
        marker=dict(size=[6, 16], color=[pc, pc], symbol=["circle", "circle"]),
        name="Bieżąca pozycja", showlegend=True,
    ))

    fig.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown(f"### {phase_info['emoji']} Faza: {phase}")
    st.markdown(f"*{phase_info['description']}*")
    st.markdown("**✅ Rekomendowane aktywa:**")
    for a in phase_info["recommended"]:
        st.markdown(f"  • {a}")
    st.markdown("**❌ Unikaj:**")
    for a in phase_info["avoid"]:
        st.markdown(f"  • {a}")
    st.markdown(f"**Sygnał:** `{phase_info['signal']}`")

st.divider()
st.markdown("### 📊 Historyczne Wyniki Aktywów per Faza (1973-2023)")
perf_df = historical_performance_table()
st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.markdown("### 💡 Jak używać Zegara?")
with st.expander("📚 Metodologia Investment Clock"):
    st.markdown("""
    **Merrill Lynch Investment Clock** (2004) klasyfikuje fazę cyklu na podstawie 2 osi:
    - **Oś X (GDP)**: Wzrost PKB przyspiesza (+) lub zwalnia (−)
    - **Oś Y (Inflacja)**: Inflacja rośnie (+) lub spada (−)

    **4 fazy:**
    | Faza | Wzrost | Inflacja | Premia |
    |------|--------|----------|--------|
    | 🌅 Recovery | ↑ | ↓ | Akcje |
    | ☀️ Overheat | ↑ | ↑ | Surowce |
    | 🌪️ Stagflation | ↓ | ↑ | Złoto/Cash |
    | 🌙 Reflation | ↓ | ↓ | Obligacje |

    *Źródło: Merrill Lynch "The Investment Clock" (2004); 50 lat danych 1973-2023*
    """)
