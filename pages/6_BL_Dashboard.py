"""
6_BL_Dashboard.py — Black-Litterman AI Views Dashboard

Zawiera:
  - CAPM Prior vs BL Posterior comparison (bar chart)
  - Tabelka views: które views AI wygenerowało, confidence, kierunek
  - Suwak tau — siła przeniesienia AI views na alokację
  - Optymalne wagi portfela BL vs market-cap
  - Math explainery
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer
from modules.black_litterman import BlackLittermanEngine

st.set_page_config(page_title="Black-Litterman Dashboard", page_icon="🎯", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🎯 Black-Litterman AI Dashboard")
st.markdown(
    "<p style='color:#6b7280;'>Model Black-Litterman (1992) z AI views — "
    "bayesowski prior CAPM + views od agentów LocalCIO/LocalEconomist</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry BL")
    tau = st.slider(
        "τ (Tau) — siła przeniesienia AI views",
        min_value=0.01, max_value=0.50, value=0.05, step=0.01,
        help="Im większe τ, tym bardziej AI views 'nadpisują' CAPM equilibrium. "
             "Idzorek (2005): τ = 1/T lub 0.01–0.05."
    )
    risk_aversion = st.slider("λ — awersja do ryzyka", 1.0, 5.0, 2.5, 0.1)
    regime_choice = st.selectbox(
        "Symulacja reżimu rynkowego",
        ["risk_off", "neutral", "risk_on"],
        index=1,
    )
    vix_signal = st.slider("Sjgnał VIX (0=low, 1=high)", 0.0, 1.0, 0.30, 0.05)
    yc_signal  = st.slider("Sygnał Yield Curve (0=normal, 1=inverted)", 0.0, 1.0, 0.25, 0.05)
    cs_signal  = st.slider("Sygnał Credit Spread", 0.0, 1.0, 0.20, 0.05)

# ─── DEMO PORTFOLIO ──────────────────────────────────────────────────────────
ASSETS = ["Akcje (SPY)", "Obligacje (TLT)", "BTC/Krypto", "Złoto (GLD)"]
n = len(ASSETS)

# Symulated returns (annualized %)
DEMO_RETURNS_MEAN = np.array([0.12, 0.04, 0.25, 0.08])  # annual
DEMO_VOL          = np.array([0.16, 0.08, 0.65, 0.15])
DEMO_CORR = np.array([
    [1.00,  0.00,  0.30,  0.05],
    [0.00,  1.00, -0.10,  0.20],
    [0.30, -0.10,  1.00,  0.10],
    [0.05,  0.20,  0.10,  1.00],
])
sigma = np.outer(DEMO_VOL, DEMO_VOL) * DEMO_CORR  # annual covariance

# Market cap weights (heuristic: equities-heavy)
w_mkt = np.array([0.55, 0.30, 0.05, 0.10])

# ─── ENGINE ──────────────────────────────────────────────────────────────────
engine = BlackLittermanEngine(risk_aversion=risk_aversion, tau=tau)

pi = engine.compute_implied_returns(sigma, w_mkt)

# Synthetic CIO/Economist output based on sidebar signals
cio_thesis = {
    "regime": regime_choice,
    "raw_signals": {
        "vix_level": vix_signal,
        "yield_curve": yc_signal,
        "credit_spread": cs_signal,
        "dxy_strength": 0.3,
    },
}
econ_analysis = {
    "score": 5.5 if vix_signal > 0.5 else 2.0,
    "raw_signals": cio_thesis["raw_signals"],
}

P, Q, omega = engine.build_views_from_agents_with_sigma(
    cio_thesis, econ_analysis, ASSETS, sigma
)
mu_bl, sigma_bl = engine.posterior_returns(sigma, pi, P, Q, omega)
w_bl    = engine.optimize_portfolio(mu_bl, sigma_bl)
w_capm  = engine.optimize_portfolio(pi, sigma)

# ─── ROW 1: KEY METRICS ──────────────────────────────────────────────────────
_c1, _c2, _c3 = st.columns(3)
with _c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Liczba AI Views</div>
        <div class='metric-value' style='color:#00ccff;'>{P.shape[0]}</div>
        <div style='font-size:10px;color:#6b7280;'>Reżim: <b>{regime_choice}</b></div>
    </div>""", unsafe_allow_html=True)
with _c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>τ (Tau)</div>
        <div class='metric-value' style='color:#00e676;'>{tau:.2f}</div>
        <div style='font-size:10px;color:#6b7280;'>Siła przeniesienia views</div>
    </div>""", unsafe_allow_html=True)
with _c3:
    bl_sharpe = (mu_bl @ w_bl) / max(np.sqrt(w_bl @ sigma_bl @ w_bl), 1e-6)
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>BL Sharpe (annualized)</div>
        <div class='metric-value' style='color:#00e676;'>{bl_sharpe:.2f}</div>
        <div style='font-size:10px;color:#6b7280;'>Portfel BL-optimal</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ROW 2: BL vs CAPM CHART + VIEWS TABLE ───────────────────────────────────
col_chart, col_views = st.columns([2, 1])

with col_chart:
    st.markdown("#### CAPM Prior (π) vs BL Posterior (μ_BL)")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=ASSETS, y=pi * 100, name="Market-Implied CAPM (π)",
        marker_color="#00ccff", opacity=0.8,
    ))
    fig_comp.add_trace(go.Bar(
        x=ASSETS, y=mu_bl * 100, name="BL Posterior (μ_BL)",
        marker_color="#00e676", opacity=0.9,
    ))
    fig_comp.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        barmode="group",
        height=340,
        yaxis_title="Oczekiwany Zwrot (%/rok)",
        legend=dict(orientation="h", y=-0.15),
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("🧮 Jak BL łączy prior i views?"):
        st.markdown(math_explainer(
            "Black-Litterman Posterior",
            "μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]",
            "Prior CAPM π = 'co rynek implikuje'. AI views (P,Q) = 'co agent sądzi'. "
            "Ω diagonal = niepewność views (Idzorek: (1-conf)/conf × PΣP'). "
            "τ → 0: wynik ≈ CAPM. τ → ∞: wynik ≈ pure views.",
            "Black & Litterman (1992); Idzorek (2005); He & Litterman (1999)",
        ), unsafe_allow_html=True)

with col_views:
    st.markdown("#### AI Views (P·Q)")
    if P.shape[0] == 0:
        st.info("Brak views w tym reżimie. Posterior = Prior (CAPM).")
    else:
        rows_v = []
        for i in range(P.shape[0]):
            long_assets  = [ASSETS[j] for j in range(n) if P[i,j] > 0]
            short_assets = [ASSETS[j] for j in range(n) if P[i,j] < 0]
            direction = ""
            if long_assets:  direction += f"↑ {', '.join(long_assets)}"
            if short_assets: direction += f" ↓ {', '.join(short_assets)}"
            omega_ii = omega[i, i]
            conf = 1.0 / (1.0 + omega_ii / max(float(P[i] @ P[i]), 1e-8))
            rows_v.append({
                "View #": i+1,
                "Kierunek": direction,
                "Q (%/yr)": f"{Q[i]*100:+.1f}%",
                "Conf.": f"{conf*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows_v), use_container_width=True, hide_index=True)

    st.markdown("#### Wagi portfela")
    df_w = pd.DataFrame({
        "Aktywo":       ASSETS,
        "BL Optimal":  [f"{w*100:.1f}%" for w in w_bl],
        "CAPM/MKT":    [f"{w*100:.1f}%" for w in w_capm],
        "Δ":           [f"{(b-c)*100:+.1f}%" for b, c in zip(w_bl, w_capm)],
    })
    st.dataframe(df_w, use_container_width=True, hide_index=True)

st.divider()

# ─── ROW 3: EFFICIENT FRONTIER (BL vs CAPM) ───────────────────────────────────
st.markdown("#### Efficient Frontier: BL (green) vs CAPM (cyan)")

try:
    def _frontier(mu_vec, sig_mat, n_pts=60):
        from scipy.optimize import minimize
        rets, vols = [], []
        target_rets = np.linspace(mu_vec.min(), mu_vec.max(), n_pts)
        for r_target in target_rets:
            res = minimize(
                lambda w: w @ sig_mat @ w,
                x0=np.ones(len(mu_vec)) / len(mu_vec),
                method="SLSQP",
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w: w @ mu_vec - r_target},
                ],
                bounds=[(0, 1)] * len(mu_vec),
                options={"ftol": 1e-9},
            )
            if res.success:
                rets.append(r_target)
                vols.append(np.sqrt(res.fun))
        return np.array(vols) * 100, np.array(rets) * 100

    bl_vols, bl_rets = _frontier(mu_bl, sigma_bl)
    capm_vols, capm_rets = _frontier(pi, sigma)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=bl_vols, y=bl_rets, mode="lines",
        line=dict(color="#00e676", width=2.5), name="BL Frontier",
    ))
    fig_ef.add_trace(go.Scatter(
        x=capm_vols, y=capm_rets, mode="lines",
        line=dict(color="#00ccff", width=2, dash="dot"), name="CAPM Frontier",
    ))
    # Optimal portfolios
    fig_ef.add_trace(go.Scatter(
        x=[np.sqrt(w_bl @ sigma_bl @ w_bl) * 100],
        y=[mu_bl @ w_bl * 100],
        mode="markers", marker=dict(size=12, color="#00e676", symbol="star"),
        name="BL Optimal",
    ))
    fig_ef.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        xaxis_title="Ryzyko — Volatility (%/rok)",
        yaxis_title="Oczekiwany Zwrot (%/rok)",
        height=350, legend=dict(orientation="h", y=-0.15),
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_ef, use_container_width=True)
except Exception as e:
    st.info(f"Efficient frontier: {e}")
