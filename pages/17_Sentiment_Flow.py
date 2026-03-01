"""17_Sentiment_Flow.py — Sentiment & Fund Flow Tracker"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.sentiment_flow_tracker import (
    compute_etf_flow_proxy, composite_fear_greed,
    short_squeeze_risk, ETF_UNIVERSE,
)

st.set_page_config(page_title="Sentiment Flow Tracker", page_icon="🌊", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_etf_data(ticker, period="1y"):
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        def _col(name):
            if isinstance(raw.columns, pd.MultiIndex):
                col = raw[name]
            elif name in raw.columns:
                col = raw[name]
            else:
                return None
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col.squeeze().dropna()
        return _col("Close"), _col("Volume")
    except Exception:
        return None, None

st.markdown("# 🌊 Sentiment & Fund Flow Tracker")
st.markdown("*ETF flows, Fear & Greed, put/call ratio — Smart Money vs Market Sentiment*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Parametry Sentymentu")
    vix_input = st.slider("VIX (bieżący)", 10.0, 60.0, 18.0, 0.5)
    ad_ratio = st.slider("Advance/Decline Ratio", 0.3, 3.0, 1.4, 0.1)
    pcr_input = st.slider("Put/Call Ratio", 0.4, 2.0, 0.85, 0.05)
    hy_bps = st.slider("HY Spread (bps)", 200, 900, 380, 10)
    breadth_pct = st.slider("Breadth (% powyżej MA200)", 20, 90, 55) / 100

# Fear & Greed
fng = composite_fear_greed(
    vix=vix_input,
    advance_decline_ratio=ad_ratio,
    put_call_ratio=pcr_input,
    hy_spread_bps=hy_bps,
    breadth_pct=breadth_pct,
)
score = fng.get("fng_score", 50)
label = fng.get("label", "Neutral")
signal = fng.get("signal", "")

# Header
c1, c2, c3 = st.columns(3)
score_color = "#ff1744" if score < 25 else "#ff6d00" if score < 40 else "#ffea00" if score < 60 else "#76ff03" if score < 80 else "#00e676"
c1.markdown(f"""<div class="metric-card">
    <div class="metric-label">FEAR & GREED INDEX</div>
    <div class="metric-value" style="color:{score_color};font-size:42px;">{score:.0f}</div>
    <div style="color:{score_color};font-weight:700">{label}</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card">
    <div class="metric-label">SYGNAŁ CONTRARIAN</div>
    <div class="metric-value" style="font-size:16px;">{signal}</div>
    <div style="color:#6b7280;font-size:11px">contrarian = rób odwrotnie niż tłum</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card">
    <div class="metric-label">PUT/CALL RATIO</div>
    <div class="metric-value" style="color:{'#00e676' if pcr_input > 1.0 else '#ff1744' if pcr_input < 0.7 else '#ffea00'}">{pcr_input:.2f}</div>
    <div style="font-size:11px;color:#6b7280">{'>1.0 = Kontrar. Bullish' if pcr_input > 1.0 else '<0.7 = Kontrar. Bearish' if pcr_input < 0.7 else 'Neutral'}</div>
</div>""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["🌡️ Fear & Greed Gauge", "📊 ETF Flow Monitor", "🔥 Short Squeeze"])

with tab1:
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": "Fear & Greed Index", "font": {"size": 20, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": score_color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(255,23,68,0.25)"},
                {"range": [20, 40], "color": "rgba(255,100,0,0.18)"},
                {"range": [40, 60], "color": "rgba(255,234,0,0.15)"},
                {"range": [60, 80], "color": "rgba(100,255,0,0.15)"},
                {"range": [80, 100], "color": "rgba(0,230,118,0.25)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": score},
        },
        number={"font": {"color": score_color, "size": 56}},
    ))
    for x_val, label_txt in [(10, "EXT\nFEAR"), (30, "FEAR"), (50, "NEUTRAL"), (70, "GREED"), (90, "EXT\nGREED")]:
        angle = np.radians(180 - x_val * 1.8)
        r = 0.80
        cx = 0.5 + r * 0.36 * np.cos(angle)
        cy = 0.32 + r * 0.35 * np.sin(angle)
        fig_gauge.add_annotation(x=cx, y=cy, text=label_txt, showarrow=False,
                                 font=dict(size=9, color="#9ca3af"),
                                 xref="paper", yref="paper")

    fig_gauge.update_layout(
        template="plotly_dark", height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Components breakdown
    comps = fng.get("components", {})
    if comps:
        labels_c = list(comps.keys())
        vals_c = [comps[k] for k in labels_c]
        fig_comp = go.Figure(go.Bar(
            x=labels_c, y=vals_c,
            marker_color=[
                "#00e676" if v > 60 else "#ffea00" if v > 40 else "#ff1744"
                for v in vals_c
            ],
            text=[f"{v:.0f}" for v in vals_c],
            textposition="outside",
        ))
        fig_comp.update_layout(
            template="plotly_dark", height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Score (0-100)", range=[0, 120]),
            title="Składowe Fear & Greed",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    etf_select = st.multiselect("Wybierz ETF do analizy flow", list(ETF_UNIVERSE.keys()), default=["SPY", "TLT", "GLD"])
    if etf_select:
        flow_results = {}
        for etf in etf_select:
            with st.spinner(f"Ładowanie {etf}..."):
                p, v = load_etf_data(etf, "1y")
            if p is not None and v is not None:
                flow_res = compute_etf_flow_proxy(p, v)
                if "error" not in flow_res:
                    flow_results[etf] = flow_res

        if flow_results:
            rows = []
            for etf, res in flow_results.items():
                etf_info = ETF_UNIVERSE.get(etf, {})
                rows.append({
                    "ETF": etf,
                    "Nazwa": etf_info.get("name", etf),
                    "Flow 5D": f"{res.get('flow_5d_normalized', 0):.3f}",
                    "Flow 20D": f"{res.get('flow_20d_normalized', 0):.3f}",
                    "Trend": res.get("trend", ""),
                    "Przyspieszenie?": "📈" if res.get("is_accelerating") else "📉",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Wybierz co najmniej 1 ETF.")

with tab3:
    st.markdown("### 🔥 Short Squeeze Risk Calculator")
    col1, col2 = st.columns(2)
    with col1:
        si = st.number_input("Short Interest (akcji)", value=50_000_000, step=1_000_000)
        float_s = st.number_input("Float (akcji w obiegu)", value=500_000_000, step=10_000_000)
    with col2:
        adv = st.number_input("Śr. Dzienny Obrót (akcji)", value=20_000_000, step=1_000_000)
        price = st.number_input("Cena Akcji (USD)", value=50.0, step=1.0)

    sq_res = short_squeeze_risk(si, float_s, adv, price)

    c1, c2, c3 = st.columns(3)
    dtc = sq_res.get("days_to_cover", 0)
    dtc_c = "#ff1744" if dtc > 5 else "#ffea00" if dtc > 2 else "#00e676"
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">DAYS TO COVER</div>
        <div class="metric-value" style="color:{dtc_c}">{dtc:.1f}</div></div>""", unsafe_allow_html=True)
    pf = sq_res.get("short_pct_float", 0)
    fc = "#ff1744" if pf > 0.2 else "#ffea00" if pf > 0.1 else "#00e676"
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">SHORT % FLOAT</div>
        <div class="metric-value" style="color:{fc}">{pf:.1%}</div></div>""", unsafe_allow_html=True)
    sq = sq_res.get("squeeze_risk_score", 0)
    sc2 = "#ff1744" if sq > 75 else "#ffea00" if sq > 40 else "#00e676"
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">SQUEEZE RISK SCORE</div>
        <div class="metric-value" style="color:{sc2}">{sq:.0f}/100</div></div>""", unsafe_allow_html=True)

    sq_label = sq_res.get("label", "")
    if "Bardzo wysokie" in sq_label:
        st.error(sq_label)
    elif "Podwyższone" in sq_label:
        st.warning(sq_label)
    else:
        st.success(sq_label)
