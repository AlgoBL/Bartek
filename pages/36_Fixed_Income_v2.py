"""36_Fixed_Income_v2.py — Zaawansowane Modele Obligacji i Krzywej Dochodowości"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Fixed Income v2.0",
    subtitle="Nelson-Siegel-Svensson · Duration Convexity PnL · Yield Curve Scenarios",
    icon="🏦", badge="Rynek Długu"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "📈 Nelson-Siegel-Svensson", "📊 Duration & Convexity",
    "🔮 Curve Scenarios", "🏠 MBS Prepayment"
])

# --- TAB 1: NSS MODEL ---
with tabs[0]:
    st.markdown("### 📈 Model Nelson-Siegel-Svensson (NSS)")
    st.caption("Standardowe narzędzie banków centralnych do estymacji zerokuponowej krzywej dochodowości.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("Parametry krzywej:")
        beta0 = st.slider("Beta 0 (Poziom długookresowy)", 1.0, 10.0, 5.0)
        beta1 = st.slider("Beta 1 (Nachylenie/Short-term)", -5.0, 5.0, -2.0)
        beta2 = st.slider("Beta 2 (Garb/Medium-term)", -5.0, 10.0, 3.0)
        beta3 = st.slider("Beta 3 (Drugi Garb)", -5.0, 5.0, 0.0)
        tau1 = st.slider("Tau 1 (Położenie garbu 1)", 0.1, 10.0, 2.0)
        tau2 = st.slider("Tau 2 (Położenie garbu 2)", 0.1, 10.0, 5.0)
        
    with col2:
        def nss_model(t, b0, b1, b2, b3, t1, t2):
            term1 = b1 * (1 - np.exp(-t/t1)) / (t/t1)
            term2 = b2 * ((1 - np.exp(-t/t1)) / (t/t1) - np.exp(-t/t1))
            term3 = b3 * ((1 - np.exp(-t/t2)) / (t/t2) - np.exp(-t/t2))
            return b0 + term1 + term2 + term3

        maturities = np.linspace(0.1, 30, 300)
        yields = nss_model(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
        
        fig_nss = go.Figure()
        fig_nss.add_trace(go.Scatter(x=maturities, y=yields, mode='lines', line=dict(color="#3498db", width=4)))
        fig_nss.update_layout(title="Krzywa Dochodowości (NSS Model)", template="plotly_dark", height=400,
                            xaxis_title="Tenor (Lata)", yaxis_title="Yield (%)")
        st.plotly_chart(fig_nss, use_container_width=True)

# --- TAB 2: DURATION & CONVEXITY ---
with tabs[1]:
    st.markdown("### 📊 PnL Attribution: Duration & Convexity")
    
    col1, col2 = st.columns(2)
    with col1:
        yield_change_bps = st.slider("Zmiana Yield (BPS)", -200, 200, 50)
        dy = yield_change_bps / 10000
        
        duration = st.slider("Modified Duration (D)", 1, 20, 7)
        convexity = st.slider("Convexity (C)", 10, 500, 80)
        
        # PnL = -D * dy + 0.5 * C * dy^2
        pnl_dur = -duration * dy
        pnl_conv = 0.5 * convexity * dy**2
        total_pnl = pnl_dur + pnl_conv
        
        st.markdown(f"""<div style='{CARD}'>
        Kontrybucja Duration: <b style='color:#ff1744'>{pnl_dur:.2%}</b><br>
        Kontrybucja Convexity: <b style='color:#00e676'>{pnl_conv:.2%}</b><br>
        <b>Całkowita zmiana ceny: {total_pnl:.2%}</b>
        </div>""", unsafe_allow_html=True)
        
    with col2:
        dy_range = np.linspace(-0.03, 0.03, 100)
        pnl_line = -duration * dy_range + 0.5 * convexity * dy_range**2
        
        fig_dc = go.Figure()
        fig_dc.add_trace(go.Scatter(x=dy_range*10000, y=pnl_line*100, mode='lines', line=dict(color="#f39c12", width=3)))
        fig_dc.add_vline(x=yield_change_bps, line_dash="dash")
        fig_dc.update_layout(title="Wrażliwość Ceny Obligacji na Yield", template="plotly_dark", height=350,
                           xaxis_title="Zmiana Yield (BPS)", yaxis_title="Zmiana Ceny (%)")
        st.plotly_chart(fig_dc, use_container_width=True)

# --- TAB 3: CURVE SCENARIOS ---
with tabs[2]:
    st.markdown("### 🔮 Scenariusze Krzywej (Bear Flattener vs Bull Steepener)")
    
    scen = st.selectbox("Wybierz scenariusz makro:", ["Parallel Shift", "Bear Flattener", "Bull Steepener", "Butterfly Twist"])
    
    mats = np.array([1, 2, 5, 10, 30])
    base_curve = np.array([3.0, 3.2, 3.5, 4.0, 4.5])
    
    if scen == "Parallel Shift":
        new_curve = base_curve + 1.0
    elif scen == "Bear Flattener":
        new_curve = base_curve + np.array([1.5, 1.2, 0.8, 0.4, 0.1])
    elif scen == "Bull Steepener":
        new_curve = base_curve + np.array([-1.0, -0.8, -0.5, -0.2, 0.1])
    else: # Butterfly
        new_curve = base_curve + np.array([0.5, 0.1, -0.3, 0.1, 0.5])
        
    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=mats, y=base_curve, mode='lines+markers', name="Base Curve", line=dict(color="gray")))
    fig_scen.add_trace(go.Scatter(x=mats, y=new_curve, mode='lines+markers', name=scen, line=dict(color="#ff1744", width=3)))
    fig_scen.update_layout(title="Prognoza Przesunięcia Krzywej", template="plotly_dark", height=400)
    st.plotly_chart(fig_scen, use_container_width=True)

# --- TAB 4: MBS PREPAYMENT ---
with tabs[3]:
    st.markdown("### 🏠 MBS: Mortgage-Backed Securities & Prepayment Risk")
    st.caption("Unikalne ryzyko instrumentów hipotecznych: kiedy stopy spadają, ludzie spłacają kredyty wcześniej (Refinancing).")
    
    rate_current = st.slider("Obecne stopy rynkowe (%)", 2.0, 10.0, 6.0)
    rate_mortgage = 7.5 # Initial mortgage rate
    
    incentive = rate_mortgage - rate_current
    # CPR = Conditional Prepayment Rate
    cpr = 100 * (1 / (1 + np.exp(-1.5 * (incentive - 1.0)))) # S-curve model
    
    col1, col2 = st.columns(2)
    col1.metric("Incentive to Refinance", f"{incentive:.2f}%")
    col2.metric("Estymowany CPR (Roczny % spłat)", f"{cpr:.1f}%")
    
    st.info("Wysoki CPR skraca czas trwania (duration) MBS, co jest niekorzystne dla inwestora przy spadających stopach (Negative Convexity).")
