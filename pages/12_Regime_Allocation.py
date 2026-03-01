"""12_Regime_Allocation.py — Dynamiczna Alokacja wg Reżimu Rynkowego"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.regime_adaptive_allocation import (
    detect_regime_rule_based, fit_gaussian_mixture_regimes,
    regime_conditional_weights, REGIMES,
)

st.set_page_config(page_title="Regime Adaptive Allocation", page_icon="🔀", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker="SPY", period="5y"):
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        elif "Close" in raw.columns:
            closes = raw["Close"]
        else:
            closes = raw.iloc[:, 0]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        return closes.squeeze().dropna()
    except Exception:
        return None

st.markdown("# 🔀 Regime Adaptive Allocation")
st.markdown("*HMM detekcja reżimów — automatyczne dostosowanie wag Barbella do reżimu rynkowego*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    ticker = st.text_input("Ticker (benchmark)", "SPY")
    period = st.selectbox("Okres historyczny", ["3y", "5y", "10y"], index=1)
    vix_level = st.slider("VIX (bieżący)", 10.0, 60.0, 18.0, 0.5)
    yield_curve_val = st.slider("Yield Curve (10Y-2Y)", -2.0, 3.0, 0.5, 0.1)
    hy_spread_val = st.slider("HY Spread (bps)", 200, 900, 380, 10)
    smooth_alpha = st.slider("EMA Smoothing (1=stały, 0=natychmiastowy)", 0.0, 0.95, 0.70, 0.05)

with st.spinner("Ładowanie danych..."):
    prices = load_data(ticker, period)

if prices is None or (hasattr(prices, "empty") and prices.empty):
    st.error("Brak danych.")
    st.stop()

# Guarantee 1-D Series
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]
prices = prices.squeeze()

returns = prices.pct_change().dropna()
if isinstance(returns, pd.DataFrame):
    returns = returns.iloc[:, 0]
returns = returns.squeeze()

# Regime detection
rule_res = detect_regime_rule_based(
    returns, vix=vix_level, yield_curve=yield_curve_val, credit_spread_hy=hy_spread_val
)
gmm_res = fit_gaussian_mixture_regimes(returns)


regime_id = rule_res.get("regime_id", 0)
regime_info = REGIMES[regime_id]
emoji = rule_res.get("emoji", "🟢")
name = rule_res.get("regime_name", "Risk-On")
confidence = rule_res.get("confidence", 0.5)
color = regime_info["color"]

# GMM probs
gmm_probs = gmm_res.get("current_probs", np.array([0.6, 0.3, 0.1]))
weights_result = regime_conditional_weights(gmm_probs, smooth_alpha=smooth_alpha)

safe_w = weights_result.get("safe_weight", 0.5)
risky_w = weights_result.get("risky_weight", 0.5)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">REŻIM RYNKOWY</div>
        <div style="font-size:26px;">{emoji}</div>
        <div class="metric-value" style="color:{color};font-size:16px;">{name}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">SAFE (Obligacje/Złoto)</div>
        <div class="metric-value" style="color:#00ccff">{safe_w:.0%}</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">RISKY (Akcje)</div>
        <div class="metric-value" style="color:#a855f7">{risky_w:.0%}</div></div>""", unsafe_allow_html=True)
with c4:
    cc = "#00e676" if confidence > 0.6 else "#ffea00" if confidence > 0.3 else "#ff1744"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">PEWNOŚĆ</div>
        <div class="metric-value" style="color:{cc}">{confidence:.0%}</div></div>""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Probs reżimów", "📈 Reżim historyczny", "🎯 Wagi Barbella"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        labels_r = [REGIMES[i]["name"] for i in range(3)]
        colors_r = [REGIMES[i]["color"] for i in range(3)]
        probs = gmm_probs.tolist()
        fig_probs = go.Figure(go.Bar(
            x=labels_r, y=[p * 100 for p in probs],
            marker_color=colors_r,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        ))
        fig_probs.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Prawdopodobieństwo (%)", range=[0, 120]),
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    with col_b:
        st.markdown("**Sygnały użyte do klasyfikacji:**")
        for k, v in rule_res.get("signals", {}).items():
            st.markdown(f"• **{k}**: {v:.3f}" if isinstance(v, float) else f"• **{k}**: {v}")
        st.markdown(f"**Raw score:** {rule_res.get('raw_score', 0):.1f}")
        st.markdown(f"**Normalized:** {rule_res.get('normalized_score', 0):.2f}")
        st.markdown("---")
        st.markdown("**Docelowe wagi per reżim:**")
        for rid, rinfo in REGIMES.items():
            st.markdown(f"{rinfo['emoji']} **{rinfo['name']}**: Safe={rinfo['barbell_safe']:.0%}, Risky={rinfo['barbell_risky']:.0%}")

with tab2:
    if "state_probs" in gmm_res:
        sp = gmm_res["state_probs"]
        fig_sp = go.Figure()
        cols_reg = ["#00e676", "#ffea00", "#ff1744"]
        for i, col_name in enumerate(sp.columns):
            fig_sp.add_trace(go.Scatter(
                x=sp.index, y=sp[col_name] * 100,
                name=col_name, stackgroup="one",
                line=dict(color=cols_reg[i]),
                fillcolor=cols_reg[i].replace(")", ",0.4)").replace("rgb", "rgba") if "rgb" in cols_reg[i] else cols_reg[i],
            ))
        fig_sp.update_layout(
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="P(reżim | dane) %",
            title="Probabilistyczne stany rynku (GMM)",
        )
        st.plotly_chart(fig_sp, use_container_width=True)

with tab3:
    fig_w = go.Figure(go.Pie(
        labels=["SAFE (Obligacje/Złoto)", "RISKY (Akcje/ETF)"],
        values=[safe_w * 100, risky_w * 100],
        hole=0.5,
        marker_colors=["#00ccff", "#a855f7"],
        texttemplate="%{label}<br><b>%{percent:.0%}</b>",
    ))
    fig_w.update_layout(
        template="plotly_dark", height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=f"Docelowe wagi Barbella — {name}",
    )
    st.plotly_chart(fig_w, use_container_width=True)
    st.info(f"💡 **EMA Smoothing {smooth_alpha:.0%}**: wagi nie zmieniają się gwałtownie. Im wyższy smooth_alpha, tym wolniejsze dostosowanie.")
    st.markdown("**Interpretacja:** Regime-adaptive allocation automatycznie przesuwa ciężar portfela między bezpiecznymi aktywami (obligacje, złoto) a ryzykownymi (akcje, ETF) w zależności od fazy rynkowej detektowanej przez GMM.")
