"""43_Entropy_Finance.py — Teoria Informacji i Entropia w Finansach"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import entropy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Entropy & Info-Dynamics",
    subtitle="Transfer Entropy · Permutation Entropy · Sample Entropy · Shannon Info",
    icon="📉", badge="Teoria Informacji"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🌌 Transfer Entropy", "🎲 Permutation Entropy", 
    "📈 Shannon Information", "📐 Complexity Analysis"
])

# --- TAB 1: TRANSFER ENTROPY ---
with tabs[0]:
    st.markdown("### 🌌 Transfer Entropy (TE) — Kierunkowy Przepływ Informacji")
    st.caption("TE mierzy ile informacji o przyszłości szeregu Y otrzymujemy z przeszłości szeregu X, po uwzględnieniu przeszłości samego Y.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("W przeciwieństwie do korelacji, TE jest asymetryczna i wykrywa nieliniowe relacje przyczynowe.")
        st.markdown(f"""<div style='{CARD}'>
        <b>Zastosowanie:</b><br>
        • Czy BTC steruje rynkiem Altcoinów?<br>
        • Przepływ info między 10Y Yield a S&P500.<br>
        • Lead-lag detection w HFT.
        </div>""", unsafe_allow_html=True)
        
    with col2:
        # Mock TE matrix
        assets = ["S&P500", "Gold", "BTC", "USD"]
        te_matrix = np.array([
            [0.0, 0.1, 0.4, 0.2],
            [0.1, 0.0, 0.05, 0.3],
            [0.15, 0.05, 0.0, 0.1],
            [0.3, 0.25, 0.1, 0.0]
        ])
        
        fig_te = go.Figure(data=go.Heatmap(
            z=te_matrix, x=assets, y=assets, colorscale="Viridis",
            hovertemplate="Źródło: %{y}<br>Cel: %{x}<br>Transfer: %{z}<extra></extra>"
        ))
        fig_te.update_layout(title="Macierz Transferu Entropii (Informational Causality)", template="plotly_dark", height=400)
        st.plotly_chart(fig_te, use_container_width=True)

# --- TAB 2: PERMUTATION ENTROPY ---
with tabs[1]:
    st.markdown("### 🎲 Permutation Entropy (PE)")
    st.caption("Miarą złożoności szeregu czasowego oparta na kolejności (permutacjach) sąsiednich wartości.")
    
    noise_level = st.slider("Poziom szumu rynkowego", 0.0, 1.0, 0.5)
    
    # Generate signal
    t = np.linspace(0, 10, 500)
    signal = np.sin(t) + noise_level * np.random.normal(0, 1, 500)
    
    # Very simplified PE: H = -sum(p*log(p)) where p is distribution of up/down moves
    moves = np.diff(signal) > 0
    p_up = np.mean(moves)
    p_down = 1 - p_up
    pe_val = - (p_up * np.log2(p_up) + p_down * np.log2(p_down)) if 0 < p_up < 1 else 0
    
    fig_pe = go.Figure()
    fig_pe.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color="#3498db")))
    fig_pe.update_layout(title=f"Sygnał rynkowy (Permutation Entropy H = {pe_val:.3f})", template="plotly_dark", height=350)
    st.plotly_chart(fig_pe, use_container_width=True)
    
    st.success("H = 1.0 oznacza całkowity chaos (Random Walk). H bliskie 0 oznacza silną strukturę i przewidywalność.")

# --- TAB 3: SHANNON INFO ---
with tabs[2]:
    st.markdown("### 📈 Shannon Information & Surprise")
    st.caption("I(x) = -log2(P(x)). Informacja to negacja prawdopodobieństwa.")
    
    prob = st.slider("Prawdopodobieństwo zdarzenia (np. podwyżka stóp)", 0.01, 0.99, 0.5)
    surprise = -np.log2(prob)
    
    col1, col2 = st.columns(2)
    col1.metric("Prawdopodobieństwo (P)", f"{prob:.2f}")
    col2.metric("Wartość Informacyjna (Surprise)", f"{surprise:.2f} bits")
    
    st.info("Wydarzenia o niskim prawdopodobieństwie (Black Swans) niosą najwięcej informacji i najmocniej ruszają rynkiem.")

# --- TAB 4: COMPLEXITY ---
with tabs[3]:
    st.markdown("### 📐 Complexity vs Regularity (Sample Entropy)")
    
    st.markdown(f"""<div style='{CARD}'>
    <b>Sample Entropy (SampEn):</b><br>
    Mierzy regularność i powtarzalność wzorców. <br><br>
    • <b>Niskie SampEn:</b> Rynek jest w reżimie 'limit cycle' lub trendu. Łatwy do prognozowania algorytmicznego.<br>
    • <b>Wysokie SampEn:</b> Rynek jest w stanie szumu lub 'turbulencji'. Modele ML tracą skuteczność.
    </div>""", unsafe_allow_html=True)
    
    st.write("Wskaźnik ten jest używany do wykrywania momentów, w których należy wyłączyć automatyczne systemy tradingowe.")
