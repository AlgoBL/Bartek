"""18_Alt_Risk_Premia.py — Alternative Risk Premia (ARP)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.alternative_risk_premia import (
    time_series_momentum, cross_sectional_momentum,
    low_volatility_factor, bond_carry_signal, arp_portfolio_suggestion,
)

st.set_page_config(page_title="Alternative Risk Premia", page_icon="⚡", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(tickers, period="5y"):
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].pct_change().dropna(how="all")
        return raw.pct_change().dropna(how="all")
    except Exception:
        return None

st.markdown("# ⚡ Alternative Risk Premia")
st.markdown("*Momentum, Low-Vol, Carry — niezależne źródła zwrotu poza beta rynkową*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Universe")
    tickers_in = st.text_area("Tickery (ARP universe)", "SPY\nQQQ\nIWM\nTLT\nGLD\nEEM\nGSG", height=150)
    tickers = [t.strip().upper() for t in tickers_in.strip().split("\n") if t.strip()]
    period_h = st.selectbox("Okres historyczny", ["3y", "5y", "10y"], index=1)
    vol_target = st.slider("Vol Target TSMOM (%)", 10, 60, 40) / 100
    top_n_mom = st.slider("Momentum: Top N aktywów", 1, 5, 2)
    st.divider()
    st.markdown("### 📊 Bond Carry Inputs")
    y10 = st.slider("Yield 10Y (%)", 1.0, 8.0, 4.2, 0.1)
    y2 = st.slider("Yield 2Y (%)", 0.5, 8.0, 4.8, 0.1)
    y3m = st.slider("Yield 3M (%)", 0.1, 7.0, 5.3, 0.1)

with st.spinner("Ładowanie danych..."):
    returns_df = load_data(tickers, period_h)

if returns_df is None or returns_df.empty:
    st.error("Brak danych.")
    st.stop()

available = [t for t in tickers if t in returns_df.columns]
r = returns_df[available].dropna()

tab1, tab2, tab3, tab4 = st.tabs(["📈 TSMOM (CTA Sim.)", "🏆 Cross-Sect. Momentum", "💤 Low-Vol Factor", "📐 Bond Carry & ARP Mix"])

with tab1:
    if len(available) > 0:
        first_asset = available[0]
        with st.spinner("Backtest TSMOM..."):
            ts_res = time_series_momentum(r[first_asset], lookback=252, vol_target=vol_target)
        if "error" not in ts_res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TSMOM CAGR", f"{ts_res.get('cagr', 0):.1%}")
            c2.metric("Sharpe", f"{ts_res.get('sharpe', 0):.2f}")
            c3.metric("B&H Sharpe", f"{ts_res.get('bh_sharpe', 0):.2f}")
            c4.metric("Korelacja z B&H", f"{ts_res.get('correlation_to_buy_hold', 0):.2f}")

            strat_r = ts_res.get("strategy_returns", pd.Series())
            bh_cagr = ts_res.get("bh_cagr", 0)
            if len(strat_r) > 0:
                strat_cum = (1 + strat_r).cumprod() * 100
                bh_cum = (1 + r[first_asset].loc[strat_r.index]).cumprod() * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum, name=f"TSMOM ({first_asset})", line=dict(color="#00e676")))
                fig.add_trace(go.Scatter(x=bh_cum.index, y=bh_cum, name="Buy & Hold", line=dict(color="#ff1744", dash="dash")))
                fig.update_layout(
                    template="plotly_dark", height=320,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Wartość (baza=100)", title=f"TSMOM Strategy vs Buy & Hold — {first_asset}",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"🎯 TSMOM to strategia **CTA/Managed Futures**. Korelacja {ts_res.get('correlation_to_buy_hold', 0):.2f} z B&H oznacza {'dobrą dywersyfikację' if abs(ts_res.get('correlation_to_buy_hold', 1)) < 0.4 else 'podobne ryzyko'}.")
        else:
            st.warning(ts_res.get("error", "Błąd TSMOM"))

with tab2:
    if r.shape[1] >= 3:
        with st.spinner("Cross-Sectional Momentum..."):
            cs_res = cross_sectional_momentum(r, lookback=252, top_n=top_n_mom)
        if "error" not in cs_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("CAGR", f"{cs_res.get('cagr', 0):.1%}")
            c2.metric("Sharpe", f"{cs_res.get('sharpe', 0):.2f}")
            c3.metric("Top N", cs_res.get("top_n", top_n_mom))
            ranks_df = cs_res.get("current_rankings", pd.DataFrame())
            if not ranks_df.empty:
                st.markdown("**Bieżące rankingi momentum:**")
                st.dataframe(ranks_df[["Asset", "12-1 Momentum", "Rank", "Signal"]].rename(
                    columns={"Asset": "Aktywo", "12-1 Momentum": "Mom 12-1", "Rank": "Ranking"}
                ), use_container_width=True, hide_index=True)
        else:
            st.warning(cs_res.get("error", ""))
    else:
        st.info("Potrzeba min. 3 aktywów do cross-sectional momentum.")

with tab3:
    if r.shape[1] >= 2:
        with st.spinner("Low-Vol Factor..."):
            lv_res = low_volatility_factor(r)
        if "error" not in lv_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Low-Vol CAGR", f"{lv_res.get('cagr', 0):.1%}")
            sharpe_lv = lv_res.get("sharpe", 0)
            sharpe_imp = lv_res.get("sharpe_improvement", 0)
            c2.metric("Sharpe", f"{sharpe_lv:.2f}", delta=f"{sharpe_imp:+.2f} vs B&H")
            c3.metric("Wybrane aktywa (Low-Vol)", ", ".join(lv_res.get("current_low_vol_picks", [])[:3]))

            recent_vols = lv_res.get("recent_vols", {})
            if recent_vols:
                vol_df = pd.DataFrame(list(recent_vols.items()), columns=["Asset", "Vol (ann.)"]).sort_values("Vol (ann.)")
                vol_df["Vol (ann.)"] = vol_df["Vol (ann.)"].apply(lambda x: f"{x:.1%}")
                vol_df["Status"] = ["✅ LOW-VOL" if a in lv_res.get("current_low_vol_picks", []) else "⚪" for a in pd.DataFrame(list(recent_vols.items()), columns=["Asset", "Vol"]).sort_values("Vol")["Asset"]]
                st.dataframe(vol_df, use_container_width=True, hide_index=True)
        else:
            st.warning(lv_res.get("error", ""))

with tab4:
    carry_res = bond_carry_signal(y10, y2, y3m)
    c1, c2, c3, c4 = st.columns(4)
    carry_val = carry_res.get("carry", 0)
    cc = "#00e676" if carry_val > 0.02 else "#ffea00" if carry_val > 0 else "#ff1744"
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">BOND CARRY</div>
        <div class="metric-value" style="color:{cc}">{carry_val:+.2%}</div></div>""", unsafe_allow_html=True)
    slope = carry_res.get("slope", 0)
    sc = "#ff1744" if slope < 0 else "#00e676"
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">YIELD SLOPE</div>
        <div class="metric-value" style="color:{sc}">{slope:+.2%}</div>
        <div>{' ⚠️ INVERTED' if slope < 0 else '✅ Normal'}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">REKOMENDACJA</div>
        <div style="font-size:13px;color:#e5e7eb">{carry_res.get('recommended', '')}</div></div>""", unsafe_allow_html=True)
    st.markdown(f"**{carry_res.get('signal', '')}**")

    st.divider()
    st.markdown("### 🧩 ARP Portfolio Mix Suggestion")
    arp_sug = arp_portfolio_suggestion(r)
    total_arp = arp_sug.get("total_recommended_arp_allocation", 0)
    st.info(f"Rekomendowana alokacja do ARP: **{total_arp:.0%}** portfela\n\n*{arp_sug.get('note', '')}*")
    for sug in arp_sug.get("suggestions", []):
        st.markdown(f"""
        <div class="glassmorphism-card" style="margin-bottom:8px;padding:12px">
            <b>{sug.get('strategy', '')}</b> — Alokacja: <b>{sug.get('allocation_suggestion', 0):.0%}</b><br>
            <small>Est. Sharpe: {sug.get('estimated_sharpe', 0):.2f} | Korelacja z portfelem: {sug.get('estimated_corr', 0):.2f} | {sug.get('reason', '')}</small>
        </div>
        """, unsafe_allow_html=True)
