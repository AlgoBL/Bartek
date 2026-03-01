"""13_Liquidity_Risk.py — Analiza Ryzyka Płynności"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.liquidity_risk_analyzer import (
    amihud_ratio, liquidity_adjusted_var, liquidity_ladder,
)

st.set_page_config(page_title="Liquidity Risk Analyzer", page_icon="💧", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(tickers, period="2y"):
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
            vols = raw["Volume"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
            vols = raw[["Volume"]] if "Volume" in raw.columns else pd.DataFrame()
        return closes.dropna(how="all"), vols.dropna(how="all")
    except Exception as e:
        return None, None

st.markdown("# 💧 Liquidity Risk Analyzer")
st.markdown("*Amihud Illiquidity, LVaR, Liquidity Ladder — ile możesz sprzedać bez market impact?*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Portfel")
    tickers_in = st.text_area("Tickery", "SPY\nQQQ\nIWM\nEEM\nBTC-USD", height=120)
    tickers = [t.strip().upper() for t in tickers_in.strip().split("\n") if t.strip()]
    weights_in = st.text_area("Wagi (suma=1)", "0.35\n0.25\n0.15\n0.15\n0.10", height=120)
    try:
        weights = np.array([float(w.strip()) for w in weights_in.strip().split("\n") if w.strip()])
        weights = weights / weights.sum()
    except Exception:
        weights = np.ones(len(tickers)) / len(tickers)
    portfolio_value = st.number_input("Wartość portfela (USD)", value=500_000, step=50_000)
    st.divider()
    st.markdown("### LVaR Kalkulator")
    conf_lvar = st.slider("Confidence LVaR (%)", 90, 99, 99)
    holding_lvar = st.slider("Holding Period (dni)", 1, 30, 10)
    max_adv = st.slider("Max % ADV do handlu dziennie", 5, 50, 20)

with st.spinner("Pobieranie danych..."):
    prices_df, volumes_df = load_data(tickers)

if prices_df is None:
    st.error("Brak danych.")
    st.stop()

available = [t for t in tickers if t in prices_df.columns]
prices_a = prices_df[available].dropna()
w_a = weights[:len(available)]
w_a = w_a / w_a.sum()
returns_a = prices_a.pct_change().dropna()

# Portfolio equity curve & Amihud per asset
illiq_scores = {}
for t in available:
    if volumes_df is not None and t in volumes_df.columns:
        res = amihud_ratio(prices_a[t], volumes_df[t])
        illiq_scores[t] = res.get("illiq", 3.0)
    else:
        illiq_scores[t] = 2.0  # fallback

weighted_illiq = float(sum(w_a[i] * illiq_scores.get(t, 2.0) for i, t in enumerate(available)))

# LVaR for portfolio
port_returns = (returns_a * w_a[:len(available)]).sum(axis=1)
avg_volume_usd_approx = portfolio_value * 0.10  # rough proxy
lvar_res = liquidity_adjusted_var(
    port_returns, portfolio_value,
    avg_daily_volume_usd=avg_volume_usd_approx,
    confidence=conf_lvar / 100,
    holding_period_days=holding_lvar,
)

# Header metrics
c1, c2, c3, c4 = st.columns(4)
lc = "#00e676" if weighted_illiq < 0.5 else "#ffea00" if weighted_illiq < 2.0 else "#ff1744"
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">AMIHUD (WEIGHTED)</div>
        <div class="metric-value" style="color:{lc}">{weighted_illiq:.3f}</div>
        <div style="font-size:11px;color:#6b7280">× 10⁻⁶ (niższy = płynniejszy)</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">STANDARD VaR {conf_lvar}%</div>
        <div class="metric-value" style="color:#ffea00">{lvar_res.get('var_standard_pct', 0):.1%}</div>
        <div style="font-size:11px;color:#6b7280">{lvar_res.get('var_standard_usd', 0):,.0f} USD</div></div>""", unsafe_allow_html=True)
with c3:
    lvar_pct = lvar_res.get("var_liq_pct", 0)
    lc3 = "#ff1744" if lvar_pct > 0.05 else "#ffea00"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">LIQUIDITY VaR ({holding_lvar}D)</div>
        <div class="metric-value" style="color:{lc3}">{lvar_pct:.1%}</div>
        <div style="font-size:11px;color:#6b7280">{lvar_res.get('var_liq_usd', 0):,.0f} USD</div></div>""", unsafe_allow_html=True)
with c4:
    dtl = lvar_res.get("days_to_liquidate", 1)
    lc4 = "#00e676" if dtl <= 3 else "#ffea00" if dtl <= 10 else "#ff1744"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">DNI DO LIKWIDACJI</div>
        <div class="metric-value" style="color:{lc4}">{dtl}</div>
        <div style="font-size:11px;color:#6b7280">przy max {max_adv}% ADV</div></div>""", unsafe_allow_html=True)

liq_cost = lvar_res.get("liquidity_cost_usd", 0)
if liq_cost > 1000:
    st.warning(f"⚠️ Szacowany koszt płynności likwidacji: **{liq_cost:,.0f} USD** ({lvar_res.get('liquidity_cost_pct', 0):.2%} NAV)")

st.divider()

tab1, tab2 = st.tabs(["📊 Amihud per Aktywo", "🪜 Liquidity Ladder"])

with tab1:
    illiq_df = pd.DataFrame({
        "Asset": list(illiq_scores.keys()),
        "Amihud Illiq.": list(illiq_scores.values()),
        "Waga": [f"{w:.1%}" for w in w_a[:len(available)]],
    }).sort_values("Amihud Illiq.", ascending=False)

    colors_bar = ["#ff1744" if v > 2 else "#ffea00" if v > 0.5 else "#00e676" for v in illiq_df["Amihud Illiq."]]
    fig = go.Figure(go.Bar(
        x=illiq_df["Asset"], y=illiq_df["Amihud Illiq."],
        marker_color=colors_bar,
        text=[f"{v:.3f}" for v in illiq_df["Amihud Illiq."]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Amihud Ratio (×10⁻⁶)", title="Wskaźnik Illiquidity per Aktywo",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(illiq_df, use_container_width=True, hide_index=True)

with tab2:
    portfolio_data = [
        {
            "name": t,
            "value": float(portfolio_value * w_a[i]),
            "avg_daily_volume_usd": float(portfolio_value * w_a[i] * (10 / max(illiq_scores.get(t, 1.0), 0.01))),
            "asset_class": "etf" if t not in ["BTC-USD", "ETH-USD"] else "crypto",
        }
        for i, t in enumerate(available)
    ]
    ladder_df = liquidity_ladder(portfolio_data, max_adv_fraction=max_adv / 100)
    if not ladder_df.empty:
        fig_lad = go.Figure()
        for col, color, name_l in [
            ("Płynność 1D", "#00e676", "1 Dzień"),
            ("Płynność 1W", "#00ccff", "1 Tydzień"),
            ("Płynność 1M", "#a855f7", "1 Miesiąc"),
        ]:
            if col in ladder_df.columns:
                fig_lad.add_trace(go.Bar(x=ladder_df["Aktywo"], y=ladder_df[col], name=name_l, marker_color=color))
        fig_lad.update_layout(
            template="plotly_dark", height=340, barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Płynna kwota (USD)", title="Liquidity Ladder — Co możesz spieniężyć?",
        )
        st.plotly_chart(fig_lad, use_container_width=True)
        display_cols = ["Aktywo", "Wartość (USD)", "Płynność 1D", "Płynność 1W", "Płynność 1M", "Dni do pełnej likwidacji"]
        st.dataframe(ladder_df[[c for c in display_cols if c in ladder_df.columns]], use_container_width=True, hide_index=True)
