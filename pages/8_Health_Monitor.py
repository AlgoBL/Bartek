"""8_Health_Monitor.py — Portfolio Health Monitor Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

from modules.styling import apply_styling
from modules.portfolio_health_monitor import (
    drawdown_alert, volatility_spike_detector, correlation_breakdown_alert,
    kelly_fraction_monitor, portfolio_health_score, get_active_alerts,
)

st.set_page_config(page_title="Portfolio Health Monitor", page_icon="🏥", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_demo_data(tickers, period="2y"):
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        prices = prices.dropna(how="all")
        returns = prices.pct_change().dropna(how="all")
        return prices, returns
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return None, None


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Portfolio Health Monitor")
st.markdown("*Ciągłe monitorowanie kondycji portfela — alerty, drawdown, zmienność, korelacje*")
st.divider()

# ── Sidebar: Portfolio Input ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfiguracja Portfela")

    tickers_input = st.text_area(
        "Tickery aktywów (każdy w nowej linii)",
        value="SPY\nTLT\nGLD\nQQQ\nEEM",
        height=120,
    )
    tickers = [t.strip().upper() for t in tickers_input.strip().split("\n") if t.strip()]

    weights_input = st.text_area(
        "Wagi (suma = 1, kolejność jak wyżej)",
        value="0.40\n0.30\n0.10\n0.10\n0.10",
        height=120,
    )
    try:
        weights = [float(w.strip()) for w in weights_input.strip().split("\n") if w.strip()]
        if abs(sum(weights) - 1.0) > 0.05:
            st.warning(f"Wagi sumują się do {sum(weights):.2f} ≠ 1.0 — normalizuję")
        weights = np.array(weights) / sum(weights)
    except Exception:
        weights = np.ones(len(tickers)) / len(tickers)

    dd_thresholds = st.multiselect(
        "Progi Drawdown Alert (%)",
        [3, 5, 8, 10, 15, 20, 25, 30],
        default=[5, 10, 15, 20],
    )
    dd_thresholds = [t / 100 for t in dd_thresholds]

    vol_threshold = st.slider("Próg spike zmienności (Z-score)", 1.0, 4.0, 2.0, 0.5)

# ── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Pobieranie danych..."):
    prices_df, returns_df = load_demo_data(tickers)

if prices_df is None or prices_df.empty:
    st.error("Nie udało się pobrać danych. Sprawdź tickery.")
    st.stop()

# ── Build Portfolio Equity Curve ─────────────────────────────────────────────
available = [t for t in tickers if t in prices_df.columns]
if not available:
    st.error("Żaden ticker nie ma danych.")
    st.stop()

prices_used = prices_df[available].dropna()
w_used = np.array(weights[:len(available)])
w_used = w_used / w_used.sum()

prices_norm = prices_used / prices_used.iloc[0]
equity_curve = (prices_norm * w_used).sum(axis=1) * 100_000  # PLN

daily_r = equity_curve.pct_change().dropna()
returns_used = returns_df[available].dropna() if available else None

# ── HEALTH SCORE ─────────────────────────────────────────────────────────────
health = portfolio_health_score(equity_curve, returns_used)
total_score = health.get("total_score", 50)
grade = health.get("grade", "B")
status = health.get("status", "")

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

score_color = "#00e676" if total_score >= 75 else "#ffea00" if total_score >= 45 else "#ff1744"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">HEALTH SCORE</div>
        <div class="metric-value" style="color:{score_color};font-size:40px;">{total_score:.0f}<span style="font-size:16px;">/100</span></div>
        <div style="color:{score_color};font-size:22px;font-weight:700;">{grade}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    dd_detail = health.get("drawdown_detail", {})
    dd_val = dd_detail.get("current_drawdown", 0)
    dd_color = "#00e676" if dd_val > -0.05 else "#ffea00" if dd_val > -0.10 else "#ff1744"
    dd_days = dd_detail.get("days_in_dd", 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BIEŻĄCY DRAWDOWN</div>
        <div class="metric-value" style="color:{dd_color}">{dd_val:.1%}</div>
        <div style="color:#6b7280;font-size:12px;">{dd_days} dni pod ATH</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    vol_d = health.get("vol_detail", {})
    vol5 = vol_d.get("current_vol_5d", 0)
    vol_color = "#00e676" if vol5 < 0.15 else "#ffea00" if vol5 < 0.25 else "#ff1744"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">VOL (5D ANNUALIZED)</div>
        <div class="metric-value" style="color:{vol_color}">{vol5:.1%}</div>
        <div style="color:#6b7280;font-size:12px;">{vol_d.get('vol_regime','')}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"<div style='padding:14px;'><b>{status}</b></div>", unsafe_allow_html=True)

st.divider()

# ── ACTIVE ALERTS ─────────────────────────────────────────────────────────────
alerts = get_active_alerts(equity_curve, returns_used)
st.markdown("### 🚨 Aktywne Alerty")
for alert in alerts:
    lvl = alert["level"]
    icon = alert["icon"]
    msg = alert["message"]
    metric = alert["metric"]
    if lvl == "critical":
        st.error(f"{icon} **{metric}**: {msg}")
    elif lvl == "warning":
        st.warning(f"{icon} **{metric}**: {msg}")
    else:
        st.success(f"{icon} **{metric}**: {msg}")

st.divider()

# ── CHARTS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity & Drawdown", "📊 Zmienność", "🔗 Korelacje", "🎯 Health Breakdown"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        # Equity curve
        dd_series = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index, y=equity_curve,
            name="Portfel (PLN)", line=dict(color="#00e676", width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.05)",
        ))
        fig.update_layout(
            template="plotly_dark", height=280, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series * 100,
            name="Drawdown (%)", line=dict(color="#ff1744", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,23,68,0.10)",
        ))
        for thr in dd_thresholds:
            fig2.add_hline(y=-thr * 100, line_dash="dash", line_color="#ffea00",
                           annotation_text=f"{-thr:.0%}", line_width=1)
        fig2.update_layout(
            template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        # Drawdown alert details
        dd_a = drawdown_alert(equity_curve, thresholds=dd_thresholds)
        st.markdown("**Drawdown Szczegóły**")
        st.metric("ATH", f"{dd_a.get('ath', 0):,.0f} PLN")
        st.metric("Bieżąca wartość", f"{dd_a.get('current_value', 0):,.0f} PLN")
        st.metric("Recovery needed", f"{dd_a.get('recovery_needed', 0):.1%}")
        alert_l = dd_a.get("alert_label", "")
        st.markdown(f"**Status:** {alert_l}")

with tab2:
    vol_r = volatility_spike_detector(daily_r, spike_threshold=vol_threshold)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rolling Volatility**")
        roll_vol = daily_r.rolling(21).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol,
            name="Vol 21D", line=dict(color="#00ccff", width=1.5),
        ))
        fig_vol.update_layout(
            template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Annualized Vol (%)",
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    with c2:
        st.markdown("**Spike Detector Wyniki**")
        for k, v in {
            "Vol 5D (ann.)": f"{vol_r.get('current_vol_5d', 0):.1%}",
            "Vol 63D baseline": f"{vol_r.get('baseline_vol_63d', 0):.1%}",
            "Vol ratio": f"{vol_r.get('vol_ratio', 1):.2f}×",
            "Z-score": f"{vol_r.get('z_score', 0):.2f}σ",
            "Spike?": vol_r.get("alert_label", ""),
            "Reżim": vol_r.get("vol_regime", ""),
        }.items():
            st.markdown(f"**{k}:** {v}")

with tab3:
    if returns_used is not None and returns_used.shape[1] >= 2:
        corr_r = correlation_breakdown_alert(returns_used)
        cm = corr_r.get("corr_matrix")
        if cm is not None:
            fig_corr = go.Figure(go.Heatmap(
                z=cm.values,
                x=cm.columns.tolist(),
                y=cm.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(cm.values, 2),
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_corr.update_layout(
                template="plotly_dark", height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown(f"**{corr_r.get('alert_label', '')}**")
            st.markdown(f"Średnia korelacja (21D): **{corr_r.get('avg_corr_current', 0):.2f}** | baseline: {corr_r.get('avg_corr_baseline', 0):.2f}")
    else:
        st.info("Potrzeba min. 2 aktywów do analizy korelacji.")

with tab4:
    comps = health.get("components", {})
    labels = list(comps.keys())
    scores = [c["score"] for c in comps.values()]
    maxes = [c["max"] for c in comps.values()]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=labels,
        y=scores,
        marker_color=[
            "#00e676" if s / m > 0.7 else "#ffea00" if s / m > 0.4 else "#ff1744"
            for s, m in zip(scores, maxes)
        ],
        text=[f"{s:.0f}/{m}" for s, m in zip(scores, maxes)],
        textposition="outside",
        name="Score",
    ))
    fig_bar.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Score", range=[0, max(maxes) * 1.3]),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(f"**Total: {total_score:.0f}/100 — Grade: {grade}** | {status}")
