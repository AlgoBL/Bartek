"""9_Concentration_Risk.py — Monitor Ryzyka Koncentracji"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from modules.styling import apply_styling
from modules.concentration_risk_monitor import (
    compute_hhi, pca_concentration, concentration_risk_score, diversification_ratio,
)

st.set_page_config(page_title="Concentration Risk Monitor", page_icon="🎯", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(tickers, period="2y"):
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw
        return prices.dropna(how="all")
    except Exception:
        return None

st.markdown("# 🎯 Concentration Risk Monitor")
st.markdown("*HHI, PCA—analiza skrytej koncentracji i ryzyka jednoczynnikowego*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Portfel")
    tickers_input = st.text_area("Tickery", value="SPY\nQQQ\nTLT\nGLD\nBTC-USD\nEEM\nIWM", height=150)
    tickers = [t.strip().upper() for t in tickers_input.strip().split("\n") if t.strip()]
    weights_input = st.text_area("Wagi (suma=1)", value="\n".join(["0.20", "0.15", "0.20", "0.10", "0.10", "0.15", "0.10"]), height=150)
    try:
        weights = np.array([float(w.strip()) for w in weights_input.strip().split("\n") if w.strip()])
        weights = weights / weights.sum()
    except Exception:
        weights = np.ones(len(tickers)) / len(tickers)

with st.spinner("Pobieranie danych..."):
    prices = load_data(tickers)

if prices is None or prices.empty:
    st.error("Brak danych. Sprawdź tickery.")
    st.stop()

available = [t for t in tickers if t in prices.columns]
prices = prices[available].dropna()
w = weights[:len(available)]
w = w / w.sum()

returns_df = prices.pct_change().dropna()

# Scores
hhi_res = compute_hhi(w)
pca_res = pca_concentration(returns_df)
score_res = concentration_risk_score(w, returns_df, available)

# Header metrics
eff_n = hhi_res.get("effective_n", 1)
n = len(available)
pc1 = pca_res.get("pc1_variance", 0.5) if "error" not in pca_res else 0.5
total_score = score_res.get("total_score", 50)
grade = score_res.get("grade", "B")

col1, col2, col3, col4 = st.columns(4)
with col1:
    sc = "#00e676" if total_score >= 75 else "#ffea00" if total_score >= 50 else "#ff1744"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">DIVERSIFICATION SCORE</div>
        <div class="metric-value" style="color:{sc}">{total_score:.0f}/100</div>
        <div style="color:{sc}">{grade}</div></div>""", unsafe_allow_html=True)
with col2:
    ec = "#00e676" if eff_n >= n * 0.6 else "#ffea00" if eff_n >= n * 0.35 else "#ff1744"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">EFFECTIVE N</div>
        <div class="metric-value" style="color:{ec}">{eff_n:.1f}</div>
        <div style="color:#6b7280;font-size:12px;">z {n} aktywów</div></div>""", unsafe_allow_html=True)
with col3:
    hc = "#ff1744" if pc1 > 0.6 else "#ffea00" if pc1 > 0.4 else "#00e676"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">PC1 VARIANCE</div>
        <div class="metric-value" style="color:{hc}">{pc1:.0%}</div>
        <div style="color:#6b7280;font-size:12px;">wariancja 1. składowej</div></div>""", unsafe_allow_html=True)
with col4:
    hhi_v = hhi_res.get("hhi", 0.5)
    hc2 = "#ff1744" if hhi_v > 0.2 else "#ffea00" if hhi_v > 0.1 else "#00e676"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">HHI INDEX</div>
        <div class="metric-value" style="color:{hc2}">{hhi_v:.3f}</div>
        <div style="color:#6b7280;font-size:12px;">1/N={1/n:.3f} (idealne)</div></div>""", unsafe_allow_html=True)

st.divider()

# Recommendations
st.markdown("### 💡 Rekomendacje")
for rec in score_res.get("recommendations", []):
    if "🔴" in rec:
        st.error(rec)
    elif "🟠" in rec:
        st.warning(rec)
    else:
        st.success(rec)

st.divider()

tab1, tab2, tab3 = st.tabs(["🥧 Alokacja & HHI", "📊 PCA Variance", "🔗 Korelacje"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = go.Figure(go.Pie(
            labels=available, values=w,
            hole=0.45,
            marker_colors=px.colors.qualitative.Set3[:len(available)],
            texttemplate="%{label}<br>%{percent:.1%}",
        ))
        fig_pie.update_layout(
            template="plotly_dark", height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Alokacja portfela",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        # HHI vs Idealne
        hhi_scores = [hhi_v * 100, (1.0 / n) * 100]
        labels_hhi = ["Twoje HHI×100", "Idealne (1/N)×100"]
        fig_hhi = go.Figure(go.Bar(
            x=labels_hhi, y=hhi_scores,
            marker_color=["#ff1744" if hhi_v > 1.5 / n else "#ffea00", "#00e676"],
        ))
        fig_hhi.update_layout(
            template="plotly_dark", height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="HHI: im niższy tym lepszy",
        )
        st.plotly_chart(fig_hhi, use_container_width=True)
        st.markdown(f"**{hhi_res.get('label', '')}**")
        st.markdown(f"Effective N = **{eff_n:.1f}** | Max = **{n}** aktywów")

with tab2:
    if "error" not in pca_res:
        var_exp = pca_res.get("variance_explained", [])
        n_dom = pca_res.get("n_dominant_factors", 1)
        labels_pc = [f"PC{i+1}" for i in range(len(var_exp))]
        fig_pca = go.Figure(go.Bar(
            x=labels_pc, y=[v * 100 for v in var_exp],
            marker_color=[
                "#ff1744" if i == 0 and var_exp[0] > 0.6 else "#ffea00" if i == 0 else "#00ccff"
                for i in range(len(var_exp))
            ],
            text=[f"{v:.1%}" for v in var_exp],
            textposition="outside",
        ))
        cumulative = np.cumsum(var_exp) * 100
        fig_pca.add_trace(go.Scatter(
            x=labels_pc, y=cumulative.tolist(),
            name="Kumulatywnie", line=dict(color="#00e676", width=2), yaxis="y2"
        ))
        fig_pca.update_layout(
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Wariancja (%)"),
            yaxis2=dict(title="Kum. (%)", overlaying="y", side="right", range=[0, 110]),
            title=f"PCA: PC1={pc1:.0%} wariancji | {n_dom} czynniki = 80%",
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown(f"**{pca_res.get('label', '')}**")
        st.markdown(f"Diversity ratio (entropia): **{pca_res.get('diversification_ratio', 0):.2f}**")
        if pca_res.get("loadings") is not None:
            with st.expander("📐 Ładowanie czynników (Factor Loadings)"):
                st.dataframe(pca_res["loadings"].style.format("{:.3f}").background_gradient(cmap="RdYlGn", axis=None))
    else:
        st.warning(f"PCA: {pca_res.get('error', 'błąd')}")

with tab3:
    corr_m = returns_df.corr()
    fig_c = go.Figure(go.Heatmap(
        z=corr_m.values, x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
        colorscale="RdYlGn", zmid=0,
        text=np.round(corr_m.values, 2), texttemplate="%{text}", showscale=True,
    ))
    fig_c.update_layout(
        template="plotly_dark", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Macierz korelacji portfela",
    )
    st.plotly_chart(fig_c, use_container_width=True)

    avg_corr = corr_m.values[np.triu_indices(len(corr_m), k=1)].mean()
    st.markdown(f"Średnia korelacja: **{avg_corr:.3f}** | Pary > 0.8: **{int((corr_m.values[np.triu_indices(len(corr_m), k=1)] > 0.8).sum())}**")
