"""16_Rebalancing.py — Smart Rebalancing Engine"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.smart_rebalancing_engine import (
    compute_drift, minimum_trade_rebalance, rebalancing_cost_benefit,
)

st.set_page_config(page_title="Smart Rebalancing", page_icon="⚖️", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(tickers, period="3y"):
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].dropna(how="all")
        return raw.dropna(how="all")
    except Exception:
        return None

st.markdown("# ⚖️ Smart Rebalancing Engine")
st.markdown("*Threshold-based, tax-aware rebalancing — kiedy i jak rebalansować portfel?*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Portfel")
    tickers_in = st.text_area("Tickery", "SPY\nTLT\nGLD\nQQQ\nEEM", height=110)
    tickers = [t.strip().upper() for t in tickers_in.strip().split("\n") if t.strip()]
    target_w_in = st.text_area("Docelowe wagi (suma=1)", "0.40\n0.30\n0.10\n0.10\n0.10", height=110)
    try:
        target_w = np.array([float(w.strip()) for w in target_w_in.strip().split("\n") if w.strip()])
        target_w = target_w / target_w.sum()
    except Exception:
        target_w = np.ones(len(tickers)) / len(tickers)
    total_nav = st.number_input("NAV portfela (PLN)", 100_000, step=50_000)
    new_cash = st.number_input("Nowa gotówka do zainwestowania (PLN)", 0, step=5_000)
    band_pct = st.slider("Próg drift-band (%)", 1, 15, 5) / 100
    period_bt = st.selectbox("Okres backtest", ["3y", "5y", "10y"], index=0)

with st.spinner("Ładowanie..."):
    prices = load_data(tickers, period_bt)

if prices is None or prices.empty:
    st.error("Brak danych.")
    st.stop()

available = [t for t in tickers if t in prices.columns]
prices_a = prices[available].dropna()
tw = target_w[:len(available)]
tw = tw / tw.sum()

# Simulate current drift (random for demo)
rng = np.random.default_rng(42)
drift_noise = rng.normal(0, 0.04, len(available))
current_w = np.clip(tw + drift_noise, 0.01, 0.99)
current_w = current_w / current_w.sum()
current_values = current_w * total_nav

# Compute drift
drift_res = compute_drift(current_w, tw, band_pct)
needs_rebal = drift_res.get("needs_rebalance", False)
max_drift = drift_res.get("max_drift", 0)

c1, c2, c3, c4 = st.columns(4)
rc = "#ff1744" if needs_rebal else "#00e676"
c1.markdown(f"""<div class="metric-card"><div class="metric-label">REBALANCING WYMAGANY?</div>
    <div class="metric-value" style="color:{rc}">{'✅ TAK' if needs_rebal else '⚪ NIE'}</div></div>""", unsafe_allow_html=True)
dc = "#ff1744" if max_drift > band_pct else "#ffea00" if max_drift > band_pct * 0.7 else "#00e676"
c2.markdown(f"""<div class="metric-card"><div class="metric-label">MAX DRIFT</div>
    <div class="metric-value" style="color:{dc}">{max_drift:.1%}</div>
    <div style="font-size:11px;color:#6b7280">Próg: {band_pct:.0%}</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="metric-card"><div class="metric-label">NAV</div>
    <div class="metric-value" style="color:#a855f7">{total_nav:,.0f} PLN</div></div>""", unsafe_allow_html=True)
c4.markdown(f"""<div class="metric-card"><div class="metric-label">NOWA GOTÓWKA</div>
    <div class="metric-value" style="color:#00ccff">{new_cash:,.0f} PLN</div></div>""", unsafe_allow_html=True)

st.divider()
tab1, tab2, tab3 = st.tabs(["📐 Trade Plan", "📊 Drift Visualization", "📈 Strategy Backtest"])

with tab1:
    acs = ["etf"] * len(available)
    trade_res = minimum_trade_rebalance(current_values, tw, new_cash=new_cash, asset_classes=acs)
    trades_df = trade_res.get("trades", pd.DataFrame())

    c1, c2, c3 = st.columns(3)
    c1.metric("Łączny Turnover", f"{trade_res.get('total_turnover', 0):.1%}")
    c2.metric("Koszty transakcyjne", f"{trade_res.get('total_tc_pln', 0):,.0f} PLN")
    c3.metric("Szac. podatek Belka", f"{trade_res.get('tax_impact_pln', 0):,.0f} PLN")

    if not trades_df.empty:
        display_trades = trades_df[["Aktywo", "Obecna wartość (PLN)", "Docelowa wartość (PLN)", "Trade (PLN)", "Trade (%)", "Akcja"]].copy()
        display_trades["Trade (PLN)"] = display_trades["Trade (PLN)"].apply(lambda x: f"{x:+,.0f}")
        display_trades["Trade (%)"] = display_trades["Trade (%)"].apply(lambda x: f"{x:+.1%}")
        display_trades["Obecna wartość (PLN)"] = display_trades["Obecna wartość (PLN)"].apply(lambda x: f"{x:,.0f}")
        display_trades["Docelowa wartość (PLN)"] = display_trades["Docelowa wartość (PLN)"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_trades, use_container_width=True, hide_index=True)

with tab2:
    abs_drift = drift_res.get("abs_drift", np.zeros(len(available)))
    colors = ["#ff1744" if d > band_pct else "#ffea00" if d > band_pct * 0.7 else "#00e676" for d in abs_drift[:len(available)]]
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Bar(
        x=available, y=abs_drift[:len(available)] * 100,
        marker_color=colors,
        text=[f"{d:.1%}" for d in abs_drift[:len(available)]],
        textposition="outside",
        name="Drift",
    ))
    fig_drift.add_hline(y=band_pct * 100, line_dash="dash", line_color="#ffea00", annotation_text=f"Band {band_pct:.0%}")
    fig_drift.update_layout(
        template="plotly_dark", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Absolutny Drift (%)", title="Drift od Docelowej Alokacji",
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    # Current vs Target comparison
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=available, y=current_w * 100, name="Bieżące %", marker_color="#00ccff", opacity=0.7))
    fig_comp.add_trace(go.Bar(x=available, y=tw * 100, name="Docelowe %", marker_color="#a855f7", opacity=0.7))
    fig_comp.update_layout(
        template="plotly_dark", height=280, barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="%", title="Aktualny Portfel vs Cel",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    returns_df = prices_a.pct_change().dropna()
    if len(returns_df) >= 63 and returns_df.shape[1] >= 1:
        with st.spinner("Backtest strategii rebalansowania..."):
            try:
                bt_res = rebalancing_cost_benefit(returns_df, tw, initial_capital=total_nav)
                bt_df = bt_res.get("results", pd.DataFrame())
                best = bt_res.get("best_strategy", "")
                if not bt_df.empty:
                    st.success(f"🏆 Najlepsza strategia: **{best}** (wg Sharpe ratio)")
                    display_bt = bt_df.copy()
                    display_bt["CAGR"] = display_bt["CAGR"].apply(lambda x: f"{x:.1%}")
                    display_bt["Sharpe"] = display_bt["Sharpe"].apply(lambda x: f"{x:.2f}")
                    display_bt["Max DD"] = display_bt["Max DD"].apply(lambda x: f"{x:.1%}")
                    display_bt["Wartość końcowa"] = display_bt["Wartość końcowa"].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(display_bt, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Backtest: {e}")
    else:
        st.info("Za mało danych do backtestu.")
