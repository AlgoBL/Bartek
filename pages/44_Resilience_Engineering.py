"""44_Resilience_Engineering.py — Inżynieria Odporności i Punkty Krytyczne"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import odeint
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Resilience Engineering",
    subtitle="Tipping Points · Regime Shifts · Critical Slowing Down · Panarchy",
    icon="🛡️", badge="Złożoność"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🌋 Tipping Points", "⏳ Critical Slowing Down", 
    "🔄 Adaptive Cycle", "📐 Robustness vs Resilience"
])

# --- TAB 1: TIPPING POINTS ---
with tabs[0]:
    st.markdown("### 🌋 Punkty Krytyczne i Przesunięcia Reżimowe")
    st.caption("Modelowanie systemów z wieloma stanami stabilnymi (Histereza).")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Potential well simulation
        # dX/dt = r*X*(1 - X/K) - c*X^2 / (X^2 + h^2)
        harv = st.slider("Presja zewnętrzna (Harvesting / Short Selling)", 0.0, 1.0, 0.5)
        
    with col2:
        x_vals = np.linspace(0, 10, 200)
        # Potential function: V(x) = - integral(dX/dt)
        # For viz, we show the landscape
        pot = - (0.5 * x_vals**2 - (1/30) * x_vals**3) + harv * x_vals
        
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=x_vals, y=pot, mode='lines', fill='tozeroy', line=dict(color="#3498db", width=4)))
        # Ball in the well
        ball_x = 2.0 if harv < 0.6 else 8.0 # simplified logic
        fig_tp.add_trace(go.Scatter(x=[ball_x], y=[pot[int(ball_x*20)]], mode='markers', 
                                   marker=dict(size=20, color="#ff1744"), name="Stan Systemu"))
        
        fig_pt_layout = {
            "title": "Krajobraz Stabilności (Stability Landscape)",
            "xaxis": {"title": "Stan Systemu (np. Płynność)", "showticklabels": False},
            "yaxis": {"title": "Energia / Potencjał", "showticklabels": False},
            "template": "plotly_dark", "height": 400
        }
        fig_tp.update_layout(**fig_pt_layout)
        st.plotly_chart(fig_tp, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <b>Tipping Point:</b> Moment, w którym mała zmiana parametru (np. wzrost stóp o 25bps) powoduje gwałtowny przeskok systemu do zupełnie innego stanu (np. krach/kryzys), z którego nie da się łatwo wrócić (Histereza).
    </div>""", unsafe_allow_html=True)

# --- TAB 2: CRITICAL SLOWING DOWN ---
with tabs[1]:
    st.markdown("### ⏳ Critical Slowing Down (CSD)")
    st.caption("Wczesne ostrzeganie: system przed katastrofą potrzebuje więcej czasu, by wrócić do równowagi po małym wstrząsie.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Wzrost autokorelacji i wariancji w danych szeregów czasowych często poprzedza gwałtowną zmianę reżimu (Tipping Point).")
        st.markdown(f"""<div style='{CARD}'>
        <b>Metryki CSD:</b><br>
        1. <b>Autokorelacja (AR1):</b> Rośnie blisko 1.0.<br>
        2. <b>Wariancja:</b> Eksploduje przed zmianą stanu.<br>
        3. <b>Skewness:</b> System 'przechyla się' w stronę nowego stanu.
        </div>""", unsafe_allow_html=True)
        
    with col2:
        # Simulate a system approaching a bifurcation
        t = np.arange(500)
        noise = np.random.normal(0, 1, 500)
        # Recovery rate decreases over time
        recovery = np.linspace(0.9, 0.99, 500)
        signal = np.zeros(500)
        for i in range(1, 500):
            signal[i] = recovery[i] * signal[i-1] + noise[i]
            
        # Rolling AR(1)
        ar1 = [pd.Series(signal).iloc[max(0, i-50):i].autocorr(lag=1) for i in range(500)]
        
        fig_csd = go.Figure()
        fig_csd.add_trace(go.Scatter(y=ar1, mode='lines', line=dict(color="#f39c12", width=3), name="AR(1) Indicator"))
        fig_csd.update_layout(title="Wskaźnik Wczesnego Ostrzegania (Critical Slowing Down)", template="plotly_dark", height=350,
                            yaxis=dict(title="Autokorelacja"))
        st.plotly_chart(fig_csd, use_container_width=True)

# --- TAB 3: ADAPTIVE CYCLE ---
with tabs[2]:
    st.markdown("### 🔄 Adaptive Cycle (Gunderson & Holling)")
    st.caption("Dynamika zmian w ekosystemach i rynkach: r, K, Ω, α.")
    
    st.markdown(f"""
    <div style='display: flex; justify-content: space-around; padding: 20px;'>
        <div style='text-align:center'><b>r (Wzrost)</b><br>Eksploatacja, szybka akumulacja</div>
        <div style='text-align:center'><b>K (Konsolidacja)</b><br>Stabilizacja, sztywność, efektywność</div>
        <div style='text-align:center'><b>Ω (Uwolnienie)</b><br>Kryzys, kreatywna destrukcja</div>
        <div style='text-align:center'><b>α (Reorganizacja)</b><br>Innowacja, nowe nisze</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("Systemy odporne (resilient) potrafią szybko przechodzić przez fazę Ω do α. Systemy kruche pękają w fazie K i zostają zniszczone w Ω.")

# --- TAB 4: ROBUSTNESS VS RESILIENCE ---
with tabs[3]:
    st.markdown("### 📐 Robustness vs Resilience")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>Solidność (Robustness)</div>
        Zdolność do wytrzymania wstrząsu bez zmiany. <br>
        <i>Przykład:</i> Mur betonowy.<br>
        <b>Ryzyko:</b> Jeśli wstrząs przekroczy limit, mur pęka całkowicie (Fragility).
        </div>""", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>Odporność (Resilience)</div>
        Zdolność do ugięcia się, transformacji i powrotu (lub adaptacji do nowego stanu).<br>
        <i>Przykład:</i> Las po pożarze.<br>
        <b>Zaleta:</b> Akceptuje błąd i ewoluuje.
        </div>""", unsafe_allow_html=True)
