"""38_Market_Impact.py — Modele Wpływu Rynkowego i Optymalna Egzekucja"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Market Impact",
    subtitle="Almgren-Chriss · Kyle's Lambda · Optimal Execution · Slippage",
    icon="📉", badge="Mikrostruktura"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🎯 Almgren-Chriss", "🧠 Kyle's Lambda",
    "💥 Temporary vs Permanent", "📐 Slippage Calculator"
])

# --- TAB 1: ALMGREN-CHRISS ---
with tabs[0]:
    st.markdown("### 🎯 Model Almgren-Chriss (Optimal Execution)")
    st.caption("Jak sprzedać/kupić duży blok akcji minimalizując koszty wpływu rynkowego i ryzyka zmienności.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        total_shares = st.number_input("Całkowita liczba akcji (X)", 1000, 100000, 10000)
        t_horiz = st.slider("Horyzont czasowy (T)", 1, 20, 10)
        risk_aversion = st.slider("Awersja do ryzyka (κ)", 0.0, 1.0, 0.1)
        sigma = 0.02
        eta = 0.005 # temp impact
        gamma = 0.002 # perm impact
        
    with col2:
        # Almgren-Chriss Solution (Simplified)
        # n_t = X / T (TWAP) if risk_aversion = 0
        # More risk aversion -> execute faster
        t_steps = np.arange(0, t_horiz + 1)
        
        if risk_aversion == 0:
            inventory = total_shares * (1 - t_steps/t_horiz)
        else:
            # Optimal trajectory: sinh(kappa * (T-t)) / sinh(kappa * T)
            # kappa depends on risk aversion, sigma, eta
            kappa = np.sqrt(risk_aversion * sigma**2 / eta) if risk_aversion > 0 else 0
            if kappa > 0:
                inventory = total_shares * np.sinh(kappa * (t_horiz - t_steps)) / np.sinh(kappa * t_horiz)
            else:
                inventory = total_shares * (1 - t_steps/t_horiz)

        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(x=t_steps, y=inventory, mode='lines+markers', 
                                   name="Inventory (Pozostałe akcje)", line=dict(color="#3498db", width=3)))
        fig_ac.update_layout(title="Trajektoria Likwidacji (Optimal Liquidation Path)", template="plotly_dark", height=400,
                           xaxis_title="Kroki czasowe", yaxis_title="Liczba akcji w portfelu")
        st.plotly_chart(fig_ac, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <b>Kluczowy kompromis (Efficient Frontier of Execution):</b><br>
    • <b>Szybka egzekucja:</b> Minimalizujesz ryzyko zmiany ceny (zmienność), ale maksymalizujesz <b>temporary impact</b> (płacisz spread i uderzasz w order book).<br>
    • <b>Powolna egzekucja:</b> Minimalizujesz wpływ rynkowy (slippage), ale wystawiasz się na <b>ryzyko rynkowe</b> (cena może uciec w dół zanim sprzedasz).
    </div>""", unsafe_allow_html=True)

# --- TAB 2: KYLE'S LAMBDA ---
with tabs[1]:
    st.markdown("### 🧠 Kyle's Lambda (λ) — Głębokość Rynku")
    st.caption("Zmiana ceny wywołana przez jednostkę wolumenu: ΔP = λ * (Buy - Sell)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div style='{CARD}'>
        <b>Parametry:</b><br>
        λ = √ (Σ_v / Σ_u) / 2<br><br>
        Σ_v: Zmienność fundamentalna<br>
        Σ_u: Wolumen szumu (Noise Traders)<br><br>
        Wysokie λ = niskie <b>Market Depth</b>.
        </div>""", unsafe_allow_html=True)
        
    with col2:
        order_imbalance = np.linspace(-5000, 5000, 100)
        lambdas = [0.0001, 0.0005, 0.001]
        fig_l = go.Figure()
        for l in lambdas:
            fig_l.add_trace(go.Scatter(x=order_imbalance, y=l * order_imbalance, mode='lines', name=f"λ = {l}"))
            
        fig_l.update_layout(title="Wpływ Order Flow na Cenę (Kyle 1985)", template="plotly_dark", height=400,
                          xaxis_title="Order Flow Imbalance", yaxis_title="Price Change (ΔP)")
        st.plotly_chart(fig_l, use_container_width=True)

# --- TAB 3: IMPACT TYPES ---
with tabs[2]:
    st.markdown("### 💥 Temporary vs Permanent Market Impact")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Visualizing the impact of a single large buy
        t_imp = np.linspace(0, 50, 200)
        price_base = 100 * np.ones_like(t_imp)
        
        # Large buy at t=10
        buy_t = 10
        dur = 5
        impact = np.zeros_like(t_imp)
        mask_during = (t_imp >= buy_t) & (t_imp <= buy_t + dur)
        mask_after = t_imp > buy_t + dur
        
        # Permanent
        perm = 2.0
        # Temporary (reverts)
        temp = 4.0
        
        impact[mask_during] = (t_imp[mask_during] - buy_t)/dur * (perm + temp)
        impact[mask_after] = perm + temp * np.exp(-0.2 * (t_imp[mask_after] - (buy_t+dur)))
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Scatter(x=t_imp, y=price_base + impact, mode='lines', line=dict(color="#00e676", width=3), name="Cena z wpływem"))
        fig_imp.add_trace(go.Scatter(x=t_imp, y=price_base + perm, mode='lines', line=dict(color="white", dash="dash"), name="Nowy poziom permanentny"))
        fig_imp.update_layout(title="Anatomia dużego zlecenia (The Impact Profile)", template="plotly_dark", height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🔍 Definicje</div>
        <b>Permanent Impact:</b> Zmiana ceny wynikająca z <i>informacji</i> niesionej przez zlecenie. Trwała zmiana equilibrium.<br><br>
        <b>Temporary Impact:</b> Zmiana ceny wynikająca z chwilowego braku <i>płynności</i> (slippage). Cena wraca gdy inni traderzy (arbitrażyści) wypełnią lukę.
        </div>""", unsafe_allow_html=True)

# --- TAB 4: SLIPPAGE ---
with tabs[3]:
    st.markdown("### 📐 Kalkulator Poślizgu (Slippage)")
    
    col1, col2, col3 = st.columns(3)
    adv = col1.number_input("Średni Wolumen Dzienny (ADV)", 100000, 10000000, 1000000)
    order_size = col2.number_input("Twój rozmiar zlecenia", 1000, 1000000, 50000)
    vol_annual = col3.slider("Roczna zmienność (%)", 10, 100, 30)
    
    participation = order_size / adv
    # Simplified impact rule: I = sigma_daily * (Order / ADV)^0.5
    sigma_daily = (vol_annual / 100) / np.sqrt(252)
    est_impact_bps = 10000 * sigma_daily * np.sqrt(participation)
    
    st.markdown(f"""<div style='{CARD}; text-align:center'>
    Twoja partycypacja w ADV: <b style='color:#3498db'>{participation:.2%}</b><br>
    Szacowany wpływ rynkowy: <b style='color:#ff1744; font-size:24px'>{est_impact_bps:.2f} BPS</b><br>
    Koszty ukryte: <b style='color:#ff1744'>{(est_impact_bps/10000 * order_size):.2f} jednostek waluty</b>
    </div>""", unsafe_allow_html=True)
