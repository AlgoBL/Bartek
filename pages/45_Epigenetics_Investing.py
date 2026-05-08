"""45_Epigenetics_Investing.py — Epigenetyka Strategii i Adaptacja Meta-Poziomu"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Epigenetic Investing",
    subtitle="Meta-Strategy Expression · Environmental Adaptation · Baldwin Effect",
    icon="🧬", badge="Ewolucja"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🧬 Ekspresja Genów", "🌍 Context Switching", 
    "🐣 Baldwin Effect", "🛠️ Meta-Adaptive Layer"
])

# --- TAB 1: GENE EXPRESSION ---
with tabs[0]:
    st.markdown("### 🧬 Epigenetyka: Ekspresja Strategii bez zmiany kodu")
    st.caption("W biologii środowisko 'włącza' lub 'wyłącza' geny. W tradingu, ten sam kod może zachowywać się inaczej zależnie od kontekstu rynkowego.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>Kod Genetyczny (Algorytm)</div>
        Twój podstawowy zestaw reguł (np. Crossover MA). Nie zmienia się w czasie rzeczywistym.
        </div>""", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>Warstwa Epigenetyczna</div>
        Mechanizm decydujący, które parametry są aktywne (np. stop-loss dynamiczny, wielkość pozycji). Reaguje na 'stres' rynkowy.
        </div>""", unsafe_allow_html=True)

# --- TAB 2: CONTEXT SWITCHING ---
with tabs[1]:
    st.markdown("### 🌍 Context Switching (Środowisko)")
    
    env_stress = st.slider("Poziom stresu rynkowego (Volatility)", 0, 100, 20)
    
    # Simulate strategy behavior based on "stress"
    if env_stress < 30:
        mode = "Aggressive Growth"
        color = "#00e676"
    elif env_stress < 70:
        mode = "Balanced Defense"
        color = "#3498db"
    else:
        mode = "Survival / Cash"
        color = "#ff1744"
        
    st.markdown(f"""<div style='text-align:center; padding:30px; {CARD}; border: 2px solid {color}'>
    Obecna Ekspresja Strategii:<br>
    <b style='font-size:28px; color:{color}'>{mode}</b>
    </div>""", unsafe_allow_html=True)

# --- TAB 3: BALDWIN EFFECT ---
with tabs[2]:
    st.markdown("### 🐣 Efekt Baldwina w Tradingu")
    st.caption("Proces, w którym wyuczone zachowania stają się z czasem 'wrodzonymi' (zakodowanymi na stałe).")
    
    st.markdown(f"""<div style='{CARD}'>
    1. <b>Uczenie się:</b> Trader manualnie odkrywa, że w czasie inflacji złoto działa lepiej.<br><br>
    2. <b>Epigenetyka:</b> Trader dodaje regułę 'jeśli CPI > 5%, zwiększ wagę złota'.<br><br>
    3. <b>Baldwin Effect:</b> Reguła ta zostaje zautomatyzowana i staje się integralną częścią 'kodu genetycznego' systemu (Hardcoded Alpha).
    </div>""", unsafe_allow_html=True)

# --- TAB 4: META-ADAPTIVE ---
with tabs[3]:
    st.markdown("### 🛠️ Meta-Adaptive Layer")
    
    st.info("To warstwa 'nadzorcy' (Superviser), która monitoruje zdrowie strategii i decyduje o jej 'metylacji' (wyciszeniu) gdy przestaje pasować do obecnej epoki rynkowej.")
    
    # Chart: Performance of 2 genes in different environments
    envs = ["Hossa", "Bessa", "Chop", "Crisis"]
    gene_a = [10, -5, -2, 5]
    gene_b = [-2, 8, 1, 12]
    
    fig_epi = go.Figure()
    fig_epi.add_trace(go.Bar(x=envs, y=gene_a, name="Strategia Momentum (Gen A)", marker_color="#3498db"))
    fig_epi.add_trace(go.Bar(x=envs, y=gene_b, name="Strategia Mean-Reversion (Gen B)", marker_color="#f39c12"))
    
    fig_epi.update_layout(title="Wydajność 'Genów' Strategicznych w różnych reżimach", template="plotly_dark", barmode='group')
    st.plotly_chart(fig_epi, use_container_width=True)
