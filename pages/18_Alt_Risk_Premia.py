"""18_Alt_Risk_Premia.py — Alternative Risk Premia (ARP)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.data_provider import fetch_data
from modules.styling import apply_styling
from modules.ui.widgets import tickers_area
from modules.alternative_risk_premia import (
    time_series_momentum, cross_sectional_momentum,
    low_volatility_factor, bond_carry_signal, arp_portfolio_suggestion,
)
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(tickers, period="5y"):
    # ISINResolver jest wywoływany automatycznie wewnątrz fetch_data
    try:
        from modules.isin_resolver import ISINResolver
        resolved = [ISINResolver.resolve(t) for t in tickers]
        rev_map = {r: o for o, r in zip(tickers, resolved)}
        
        raw = fetch_data(resolved, period=period)
        if raw is None or raw.empty:
            return None
        
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique()
            if "Close" in lvl0:
                prices = raw["Close"].copy()
            elif "Adj Close" in lvl0:
                prices = raw["Adj Close"].copy()
            else:
                prices = raw.iloc[:, 0].to_frame()
        else:
            prices = raw.copy()
        
        prices.columns = [rev_map.get(c, c) for c in prices.columns]
        rets = prices.pct_change().dropna(how="all")
        return rets
    except Exception as e:
        from modules.logger import setup_logger
        setup_logger(__name__).error(f"load_data error: {e}")
        return None

st.markdown("# ⚡ Alternative Risk Premia")
st.markdown("*Momentum, Low-Vol, Carry — niezależne źródła zwrotu poza beta rynkową*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Universe")
    tickers_in = tickers_area("Tickery (ARP universe)", "SPY\nQQQ\nIWM\nTLT\nGLD\nEEM\nGSG", height=150)
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 TSMOM (CTA Sim.)", "🏆 Cross-Sect. Momentum", "💤 Low-Vol Factor", "📐 Bond Carry Strategy", "🌿 ESG Risk Premium (Nowość)"])

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

    st.markdown("---")
    st.markdown("### 🧮 Backtest Strategii Carry (Symulacja Historyczna)")
    st.markdown("W klasycznym ujęciu, strategia Carry kupuje waluty/aktywa o wysokich stopach zwrotu (np. EM) i finansuje się walutami o niskich (np. JPY, CHF). Symulacja poniżej wykorzystuje proste założenie stałego zysku z różnicy rentowności (Carry) z uwzględnieniem szumu/zmienności.")
    
    if st.button("Uruchom Symulację Carry (Monte Carlo)"):
        with st.spinner("Symulowanie Carry..."):
            np.random.seed(42)
            days = 1000
            # Zakładamy codzienny zysk ze spreadu (część carry) i losowy szok cenowy (FX risk)
            daily_carry = (y10 - y3m) / 100 / 252 
            
            # W warunkach kryzysowych waluta finansujaca gwałtownie drozeje, tworzac Carry Crash
            fx_shocks = np.random.normal(0, 0.005, days)
            # Gruby ogon (Crash carry trade'u 2 razy w historii)
            fx_shocks[200] = -0.15 
            fx_shocks[800] = -0.12
            
            pnl = np.cumsum(daily_carry + fx_shocks)
            
            fig_carry = go.Figure()
            fig_carry.add_trace(go.Scatter(y=pnl * 100, mode='lines', name="P&L Carry Strategy", line=dict(color="#00ccff")))
            fig_carry.add_trace(go.Scatter(y=np.cumsum([daily_carry]*days) * 100, mode='lines', name="Yield (Stały dochód bez ryzyka cenowego)", line=dict(color="#ffea00", dash="dash")))
            
            fig_carry.update_layout(title="Symulacja Carry P&L (z uwzględnieniem 'Rozwijania Pozycji' w kryzysach)", yaxis_title="Zysk (%)", xaxis_title="Dni", template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_carry, use_container_width=True)
            st.info("💡 **Ostrzeżenie Hyman Minsky'ego**: Carry Trades to 'zbieranie groszy przed walcem'. Zyski na Carry powoli rosną (linia przerywana), ale ryzyko gwałtownego rozwijania długu w panice gwałtownie ścina P&L w kilka dni.")

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

with tab5:
    st.markdown("### 🌿 Environmental, Social, and Governance (ESG) Risk Premium")
    st.markdown("Czy spółki nisko-emisyjne faktycznie zapewniają ekstra zwrot? Akademickie dowody sugerują 'Sin Premium' (spółki węglowe, zbrojeniowe dają wyższy zwrot przez brak popytu ze strony instytucji pro-ESG). Zbadajmy portfel syntetyczny ESG.")
    
    esg_ratio = st.slider("Zaangażowanie w ESG (Screening, % wykluczenia najbrudniejszych)", 0, 100, 50, 5)
    
    # Symulacja ESG vs Non-ESG
    try:
        if not r.empty:
            mean_ret = r.mean().mean() * 252
            vol = r.std().mean() * np.sqrt(252)
            
            # Sin Premium (Tough sectors out-perform purely for cost of capital reasons)
            esg_drag = (esg_ratio / 100.0) * 0.015 # do 1.5% kary za max ESG (czyli odcięcie od paliw)
            
            esg_cagr = mean_ret + (vol * 0.1) - esg_drag 
            non_esg_cagr = mean_ret + (vol * 0.15) 
            
            esg_var = vol * (1.0 - (esg_ratio / 100.0)*0.1) # Lekki spadek zmienności w ESG
            non_esg_var = vol * 1.1
            
            c1, c2 = st.columns(2)
            c1.metric("ESG Portfolio Expected CAGR", f"{esg_cagr*100:.2f}%", help="Teoretyczny zannualizowany zwrot spółek proekologicznych.")
            c2.metric("Sin Stocks (Non-ESG) CAGR", f"{non_esg_cagr*100:.2f}%", delta=f"{(non_esg_cagr - esg_cagr)*100:+.2f}% Sin Premium", delta_color="normal")
            
            st.markdown("""
            > [!TIP]
            > Akademickie pomiary (np. *Fama & French ESG Studies*) wskazują, że **czysto fundamentalnie aktywa ESG powinny dawać MNIEJSZY zwrot na kapitale**. 
            > Ze względu na masowe wykluczanie z portfeli instytucjonalnych firm "brudnych" (tytoń, broń, węgiel), te spółki stają się tanie, gwarantując **wysoką stopę dywidendy**. Jednocześnie spółki ESG wskutek masowych doważeń pasywnych cierpią z powodu "drożyzny" mnożnikowej.
            """)
            
            # Porownanie wykresu ESG vs Brudne
            x_ax = np.arange(10)
            esg_wealth = 100 * (1 + esg_cagr) ** x_ax
            sin_wealth = 100 * (1 + non_esg_cagr) ** x_ax
            
            fig_esg = go.Figure()
            fig_esg.add_trace(go.Bar(x=[f"Year {i}" for i in x_ax], y=esg_wealth, name="Krzywa kapitału ESG", marker_color="#00e676"))
            fig_esg.add_trace(go.Bar(x=[f"Year {i}" for i in x_ax], y=sin_wealth, name="Krzywa kapitału 'Sin Stocks'", marker_color="#ff1744"))
            
            fig_esg.update_layout(title="Symulacja dywergencji przez następne 10 Lat (Teoretyczna)", barmode='group', template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_esg, use_container_width=True)
            
    except Exception as e:
        st.error(f"Nie udało się wygenerować scenariusza ESG: {e}")
