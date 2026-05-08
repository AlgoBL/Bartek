"""35_Stochastic_Calculus.py — Rachunek Stochastyczny w Inżynierii Finansowej"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Rachunek Stochastyczny",
    subtitle="Itô vs Stratonovich · Procesy Dyfuzji · Lemat Itô · Black-Scholes SDE",
    icon="🌊", badge="Inżynieria Kwantowa"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "📈 Proces Wienera (Brown)", "🧪 Itô vs Stratonovich",
    "🔄 Proces Ornsteina-Uhlenbecka", "📐 Lemat Itô (Black-Scholes)"
])

# --- TAB 1: WIENER PROCESS ---
with tabs[0]:
    st.markdown("### 📈 Proces Wienera (Ruch Browna)")
    st.caption("Podstawowy budulec nowoczesnych finansów. dW_t ~ N(0, dt)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        n_paths = st.slider("Liczba ścieżek", 1, 50, 10)
        t_final = st.slider("Horyzont czasowy (T)", 1, 10, 1)
        dt = 0.001
        n_steps = int(t_final / dt)
        
    with col2:
        times = np.linspace(0, t_final, n_steps)
        fig_w = go.Figure()
        for i in range(n_paths):
            dw = np.random.normal(0, np.sqrt(dt), n_steps)
            w = np.cumsum(dw)
            w[0] = 0
            fig_w.add_trace(go.Scatter(x=times, y=w, mode='lines', line=dict(width=1), showlegend=False))
            
        fig_w.update_layout(title="Standardowy Proces Wienera W(t)", template="plotly_dark", height=400,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <b>Właściwości matematyczne:</b><br>
    1. W(0) = 0<br>
    2. Przyrosty są niezależne i mają rozkład normalny: W(t) - W(s) ~ N(0, t-s)<br>
    3. Trajektorie są ciągłe wszędzie, ale <b>nigdzie nieróżniczkowalne</b> w sensie klasycznym.<br>
    4. Kwadratowa wariacja: [W, W]_t = t.
    </div>""", unsafe_allow_html=True)

# --- TAB 2: ITO VS STRATONOVICH ---
with tabs[1]:
    st.markdown("### 🧪 Całka Stochastyczna: Itô vs Stratonovich")
    st.caption("Różnica w punkcie ewaluacji całki Riemann-Stieltjes prowadzi do różnych wyników.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div style='{CARD}'>
        <b>Interpretacja Itô:</b><br>
        Ewaluacja w lewym końcu interwału. Modeluje 'brak przewidywania przyszłości' (non-anticipating). Standard w finansach.<br><br>
        <b>Interpretacja Stratonovicha:</b><br>
        Ewaluacja w środku interwału. Zachowuje klasyczne reguły rachunku (np. chain rule). Używana w fizyce.
        </div>""", unsafe_allow_html=True)
        
    with col2:
        # Example: Integrate W dW
        # Ito: (1/2)(W_T^2 - T)
        # Stratonovich: (1/2)W_T^2
        dw = np.random.normal(0, np.sqrt(dt), n_steps)
        w = np.cumsum(dw)
        
        ito_int = np.sum(w[:-1] * dw[1:])
        strat_int = np.sum(0.5 * (w[:-1] + w[1:]) * dw[1:])
        
        fig_is = go.Figure()
        fig_is.add_trace(go.Bar(x=["Całka Itô", "Całka Stratonovicha"], y=[ito_int, strat_int], marker_color=["#3498db", "#ff1744"]))
        fig_is.update_layout(title="Wynik całki ∫ W dW", template="plotly_dark", height=350,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_is, use_container_width=True)
        st.caption("Różnica między nimi wynosi dokładnie (1/2) * [W, W]_T = T/2.")

# --- TAB 3: ORNSTEIN-UHLENBECK ---
with tabs[2]:
    st.markdown("### 🔄 Proces Ornsteina-Uhlenbecka (Mean Reversion)")
    st.caption("dX_t = θ(μ - X_t)dt + σdW_t")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        theta = st.slider("Szybkość powrotu (θ)", 0.1, 10.0, 1.0)
        mu = st.slider("Średnia długookresowa (μ)", -2.0, 2.0, 0.0)
        sigma = st.slider("Zmienność (σ)", 0.01, 0.5, 0.1)
        
    with col2:
        x = np.zeros(n_steps)
        x[0] = 1.0 # start from 1.0
        for i in range(1, n_steps):
            dx = theta * (mu - x[i-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            x[i] = x[i-1] + dx
            
        fig_ou = go.Figure()
        fig_ou.add_trace(go.Scatter(x=times, y=x, mode='lines', line=dict(color="#00e676", width=2)))
        fig_ou.add_hline(y=mu, line_dash="dash", line_color="white", annotation_text="Target μ")
        fig_ou.update_layout(title="Symulacja Procesu O-U", template="plotly_dark", height=400,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ou, use_container_width=True)

# --- TAB 4: BLACK-SCHOLES SDE ---
with tabs[3]:
    st.markdown("### 📐 Lemat Itô i Black-Scholes SDE")
    st.markdown("Geometryczny Ruch Browna (GBM): dS_t = μS_t dt + σS_t dW_t")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        mu_gbm = st.slider("Dryf (μ)", -0.2, 0.5, 0.05)
        vol_gbm = st.slider("Volatility (σ)", 0.05, 0.8, 0.2)
        s0 = st.number_input("Cena początkowa S0", 10, 1000, 100)
        
    with col2:
        # Solution: S_t = S_0 * exp((μ - σ²/2)t + σW_t)
        dw = np.random.normal(0, np.sqrt(dt), n_steps)
        w = np.cumsum(dw)
        s_t = s0 * np.exp((mu_gbm - 0.5 * vol_gbm**2) * times + vol_gbm * w)
        
        fig_gbm = go.Figure()
        fig_gbm.add_trace(go.Scatter(x=times, y=s_t, mode='lines', line=dict(color="#3498db", width=2.5)))
        fig_gbm.update_layout(title="Log-Normalna ścieżka cen (GBM)", template="plotly_dark", height=400,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gbm, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Lemat Itô (Klucz do wyceny)</div>
    Dla funkcji f(S,t), zmiana df wynosi:<br>
    <b>df = (∂f/∂t + μS ∂f/∂S + ½σ²S² ∂f²/∂S²)dt + σS ∂f/∂S dW_t</b><br><br>
    Ten dodatkowy człon (½σ²S² ∂f²/∂S²) to <b>Itô Correction</b>. Bez niego (w klasycznym rachunku) Black-Scholes nie działałby. 
    Reprezentuje on wpływ zmienności na średnią wartość procesu log-normalnego.
    </div>""", unsafe_allow_html=True)
