"""37_Crypto_Quant.py — Zaawansowana Analiza Ilościowa Kryptowalut"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Crypto Quant",
    subtitle="On-chain Metrics · Funding Rate Arb · Stock-to-Flow · DEX Mathematics",
    icon="₿", badge="Digital Assets"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🔗 On-chain Alpha", "🔄 Funding Arb", 
    "📈 Stock-to-Flow", "💧 DEX / LP Math"
])

# --- TAB 1: ON-CHAIN ---
with tabs[0]:
    st.markdown("### 🔗 Wskaźniki On-chain (MVRV & NVT)")
    st.caption("Wykorzystanie danych z blockchaina do wyceny fundamentalnej sieci.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div style='{CARD}'>
        <b>MVRV Ratio:</b> Market Value / Realized Value.<br>
        • > 3.0: Rynek przegrzany (Overbought).<br>
        • < 1.0: Kapitulacja / Akumulacja.
        </div>""", unsafe_allow_html=True)
        mvrv_val = st.slider("Symulowany MVRV Ratio", 0.5, 4.0, 1.8)
        
    with col2:
        # Mock data
        days = np.arange(100)
        price = 20000 * np.exp(0.01 * days) + np.random.normal(0, 1000, 100)
        realized_val = 15000 + 50 * days
        mvrv_line = price / realized_val
        
        fig_on = go.Figure()
        fig_on.add_trace(go.Scatter(x=days, y=mvrv_line, mode='lines', line=dict(color="#f39c12", width=3), name="MVRV Ratio"))
        fig_on.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Sell Zone")
        fig_on.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Buy Zone")
        fig_on.update_layout(title="Wycena On-chain (MVRV)", template="plotly_dark", height=350)
        st.plotly_chart(fig_on, use_container_width=True)

# --- TAB 2: FUNDING ARB ---
with tabs[1]:
    st.markdown("### 🔄 Arbitraż Funding Rate (Cash and Carry)")
    st.caption("Zarabianie na różnicy między ceną Spot a Perpetual Futures.")
    
    funding_rate_8h = st.slider("8h Funding Rate (%)", -0.05, 0.1, 0.01, format="%.3f")
    annualized_yield = ((1 + funding_rate_8h/100)**(3 * 365) - 1) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Stopa 8-godzinna", f"{funding_rate_8h:.4f}%")
    col2.metric("Rentowność Roczna (APY)", f"{annualized_yield:.2f}%")
    
    st.info("Strategia: Kupujesz Spot + Otwierasz Short Perpetual. Jesteś Delta-Neutral. Pobierasz funding od longów (gdy funding > 0).")

# --- TAB 3: STOCK-TO-FLOW ---
with tabs[2]:
    st.markdown("### 📈 Model Stock-to-Flow (S2F)")
    st.caption("Wycena aktywów rzadkich (Złoto, BTC) na podstawie podaży.")
    
    s2f_ratio = st.slider("Stock-to-Flow Ratio", 10, 120, 56)
    # Price = exp(a * log(S2F) + b)
    est_price = np.exp(3.3 * np.log(s2f_ratio) - 1.5)
    
    st.markdown(f"""<div style='text-align:center; padding:20px; {CARD}'>
    Przy S2F = {s2f_ratio}, teoretyczna cena BTC wynosi:<br>
    <b style='font-size:32px; color:#f39c12'>${est_price:,.0f}</b>
    </div>""", unsafe_allow_html=True)

# --- TAB 4: DEX MATH ---
with tabs[3]:
    st.markdown("### 💧 Matematyka Płynności (Uniswap v2/v3)")
    st.caption("Constant Product Market Maker: X * Y = K")
    
    p0 = st.number_input("Cena wejścia (Asset/Stable)", value=2000)
    p1 = st.number_input("Cena obecna", value=2500)
    
    price_ratio = p1 / p0
    # Impermanent Loss = (2 * sqrt(r) / (1+r)) - 1
    il = (2 * np.sqrt(price_ratio) / (1 + price_ratio)) - 1
    
    st.warning(f"Impermanent Loss (Nietrwała strata): **{il*100:.2f}%**")
    st.caption("To strata względem zwykłego trzymania (HODL) aktywów poza pulą płynności.")
