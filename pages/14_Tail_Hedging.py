"""14_Tail_Hedging.py — Tail Risk Hedging Calculator"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.tail_risk_hedging import (
    bs_put_price, put_hedge_calculator, collar_strategy,
    hedge_recommendation, ETF_HEDGES,
)

st.set_page_config(page_title="Tail Risk Hedging", page_icon="🛡️", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🛡️ Tail Risk Hedging")
st.markdown("*Kalkulator zabezpieczeń ogonowych — opcje put, collar, ETF hedges*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Parametry Portfela")
    portfolio_value = st.number_input("Wartość portfela (PLN)", value=500_000, step=50_000)
    beta = st.slider("Beta portfela do rynku", 0.3, 2.0, 1.0, 0.1)
    max_dd_target = st.slider("Max akceptowany drawdown (%)", 5, 40, 10) / 100
    risk_score = st.slider("Risk Score (z Control Center)", 0, 100, 55)
    st.divider()
    st.markdown("### 📐 Opcje Put")
    spot = st.number_input("Cena spot benchmarku (USD)", value=500.0, step=10.0)
    iv = st.slider("Implied Volatility (IV %)", 10, 60, 20) / 100
    expiry = st.selectbox("Termin wygaśnięcia (miesiące)", [1, 3, 6, 12], index=1)
    otm_pct = st.slider("OTM % (głębokość put)", 2, 20, 5) / 100

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Put Hedge Calculator", "🔄 Collar Strategy", "🏦 ETF Hedges"])

with tab1:
    hedge = put_hedge_calculator(
        portfolio_value=portfolio_value,
        beta_to_market=beta,
        max_drawdown_target=max_dd_target,
        spot_price=spot, iv=iv,
        expiry_months=expiry, otm_pct=otm_pct,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liczba kontraktów", hedge.get("n_contracts", 0))
    c2.metric("Strike", f"${hedge.get('strike', 0):.2f}")
    c3.metric("Cena 1 puta", f"${hedge.get('put_price_usd', 0):.2f}")
    c4.metric("Łączny koszt", f"${hedge.get('total_cost_usd', 0):,.0f}")

    annual_cost = hedge.get("annual_cost_pct", 0)
    ac = "#00e676" if annual_cost < 0.01 else "#ffea00" if annual_cost < 0.02 else "#ff1744"
    st.metric("Roczny koszt zabezpieczenia", f"{annual_cost:.2%}", help="Jako % wartości portfela")
    st.progress(min(1.0, annual_cost / 0.03), text=f"Koszt roczny: {annual_cost:.2%} NAV (max rekomend.: 1.5%)")

    st.divider()
    st.markdown("**Grekie opcji put:**")
    pd_detail = hedge.get("put_details", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Delta", f"{pd_detail.get('delta', 0):.3f}")
    col2.metric("Gamma", f"{pd_detail.get('gamma', 0):.4f}")
    col3.metric("Theta (dzienny)", f"${hedge.get('theta_daily', 0):.2f}")
    col4.metric("Vega (per 1% IV)", f"${pd_detail.get('vega', 0) * hedge.get('n_contracts', 1) * 100:.0f}")

    # Payoff diagram
    prices_range = np.linspace(spot * 0.7, spot * 1.15, 60)
    K = hedge.get("strike", spot * 0.95)
    n_c = hedge.get("n_contracts", 1)
    put_pnl = np.maximum(K - prices_range, 0) * n_c * 100 - hedge.get("total_cost_usd", 0)
    unhedged = (prices_range - spot) / spot * portfolio_value
    hedged = unhedged + put_pnl

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_range, y=unhedged, name="Bez zabezpieczenia", line=dict(color="#ff1744", dash="dash")))
    fig.add_trace(go.Scatter(x=prices_range, y=hedged, name="Z potami (put)", line=dict(color="#00e676")))
    fig.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.8)
    fig.add_vline(x=spot, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Cena benchmarku (USD)", yaxis_title="P&L (USD)",
        title="Payoff Diagram: Portfel vs Portfel + Put Hedge"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        put_strike_pct = st.slider("Put Strike (% of spot)", 85, 99, 95) / 100
    with col2:
        call_strike_pct = st.slider("Call Strike (% of spot)", 101, 120, 105) / 100

    collar = collar_strategy(spot, put_strike_pct, call_strike_pct, expiry / 12, iv, max(0.05, iv - 0.02))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Put Strike", f"${collar['put_strike']:.2f}")
    c2.metric("Call Strike", f"${collar['call_strike']:.2f}")
    c3.metric("Koszt Netto", f"${collar['net_cost']:.2f} ({collar['net_cost_pct']:.2%})")
    c4.metric("Zero-cost?", "✅ Tak" if collar["is_zero_cost"] else "❌ Nie")

    payoff_t = collar.get("payoff_table", pd.DataFrame())
    if not payoff_t.empty:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=payoff_t["Cena Końcowa"], y=payoff_t["Zwrot Akcji"] * 100, name="Akcja bez collar", line=dict(color="#ff1744", dash="dash")))
        fig_c.add_trace(go.Scatter(x=payoff_t["Cena Końcowa"], y=payoff_t["Zwrot Collar"] * 100, name="Collar", line=dict(color="#00e676"), fill="tozeroy", fillcolor="rgba(0,230,118,0.05)"))
        fig_c.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.8)
        fig_c.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Cena końcowa", yaxis_title="Zwrot (%)",
            title=f"Collar: floor={put_strike_pct:.0%}, cap={call_strike_pct:.0%}",
        )
        st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    rec = hedge_recommendation(portfolio_beta=beta, current_vol=0.18, risk_score=risk_score)
    st.markdown(f"### Rekomendacje dla Risk Score = {risk_score}/100")
    recs = rec.get("recommendations", [])
    total_alloc = rec.get("total_allocation", 0)
    est_cost = rec.get("estimated_annual_cost_pct", 0)

    col1, col2 = st.columns(2)
    col1.metric("Łączna alokacja hedge", f"{total_alloc:.0%}")
    col2.metric("Est. roczny koszt carry", f"{est_cost:.2%}")

    for r in recs:
        etf = r.get("etf", {})
        alloc = r.get("allocation", 0)
        reason = r.get("reason", "")
        name_h = r.get("instrument", r.get("name", ""))
        corr = etf.get("correlation_spy", 0)
        carry = etf.get("annual_cost_carry", 0)
        st.markdown(f"""
        <div class="glassmorphism-card" style="margin-bottom:8px;padding:12px">
            <b>{name_h}</b> — Alokacja: <b>{alloc:.0%}</b><br>
            <small>Korelacja z SPY: {corr:.2f} | Carry: {carry:+.1%}/rok | {reason}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Katalog ETF Hedges:**")
    rows = []
    for emoji_name, etf_data in ETF_HEDGES.items():
        rows.append({
            "Instrument": emoji_name,
            "Ticker": etf_data["ticker"],
            "Korelacja SPY": f"{etf_data['correlation_spy']:.2f}",
            "Carry roczny": f"{etf_data['annual_cost_carry']:+.1%}",
            "Rekomend. alok.": f"{etf_data['recommended_allocation']:.1%}",
            "Typ": etf_data["type"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
