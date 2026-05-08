"""46_4E_Cognition.py — Poznanie Rozszerzone i 4E Cognition"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="4E Cognition",
    subtitle="Embodied · Enacted · Embedded · Extended Mind · Andy Clark",
    icon="🧠", badge="Kognitywistyka"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🧩 Filary 4E", "💻 Extended Mind", 
    "🏹 Affordances", "🛠️ Cognitive Offloading"
])

# --- TAB 1: 4E PILLARS ---
with tabs[0]:
    st.markdown("### 🧩 Cztery Filary Poznania (4E)")
    st.caption("Poznanie to nie tylko proces wewnątrz mózgu. To interakcja całego organizmu ze środowiskiem.")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>1. Embodied (Ucieleśnione)</div>
        Poznanie zależy od posiadania ciała i jego fizycznych właściwości. Stan emocjonalny (np. głód, stres) bezpośrednio wpływa na abstrakcyjne decyzje finansowe.
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>3. Embedded (Osadzone)</div>
        Poznanie zachodzi w konkretnym środowisku fizycznym i społecznym. Twój pokój tradingowy to część Twojego systemu myślowego.
        </div>""", unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>2. Enacted (Uczynione)</div>
        Poznanie polega na działaniu. Myślimy poprzez interakcję (np. manipulowanie wykresem), a nie tylko pasywną obserwację.
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>4. Extended (Rozszerzone)</div>
        Narzędzia zewnętrzne (notatniki, kalkulatory, algorytmy) są dosłownymi częściami Twojego umysłu (Extended Mind Hypothesis).
        </div>""", unsafe_allow_html=True)

# --- TAB 2: EXTENDED MIND ---
with tabs[1]:
    st.markdown("### 💻 Hipoteza Rozszerzonego Umysłu (Andy Clark & David Chalmers)")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("**Kryterium Parzystości:** Jeśli proces zachodzący poza głową działałby identycznie jak proces wewnątrz, gdybyśmy go zinternalizowali, to należy go uznać za część umysłu.")
        st.markdown("""
        **System Intelligent Barbell jako Twoje Exocortex:**
        - Pamięć długotrwała: Baza danych i Dashboard.
        - Przetwarzanie: Moduły AI i Quant.
        - Intuicja: Wizualizacje danych.
        """)
        
    with col2:
        # Visual representation of extended mind
        fig_ext = go.Figure()
        # Brain node
        fig_ext.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["MÓZG"], textposition="top center", 
                                   marker=dict(size=50, color="#ff1744")))
        # Tool nodes
        tools_x = [1, 0.8, -0.8, -1]
        tools_y = [0.5, -0.8, -0.8, 0.5]
        tool_names = ["Kalkulator", "Algorytm", "Smartphone", "Notatnik"]
        
        fig_ext.add_trace(go.Scatter(x=tools_x, y=tools_y, mode="markers+text", text=tool_names, textposition="bottom center", 
                                   marker=dict(size=30, color="#3498db")))
        
        # Connections
        for tx, ty in zip(tools_x, tools_y):
            fig_ext.add_trace(go.Scatter(x=[0, tx], y=[0, ty], mode="lines", line=dict(color="gray", dash="dash"), showlegend=False))
            
        fig_ext.update_layout(title="Topologia Umysłu Rozszerzonego", xaxis=dict(visible=False), yaxis=dict(visible=False), 
                            template="plotly_dark", height=400)
        st.plotly_chart(fig_ext, use_container_width=True)

# --- TAB 3: AFFORDANCES ---
with tabs[2]:
    st.markdown("### 🏹 Affordances (James Gibson)")
    st.caption("Środowisko oferuje nam możliwości działania. Widzimy świat nie jako obiekty, ale jako zaproszenia do akcji.")
    
    st.markdown(f"""<div style='{CARD}'>
    W tradingu, wykres 'prowokuje' do działania. <br><br>
    • <b>Wsparcie (Support):</b> Affordance do kupna.<br>
    • <b>Opór (Resistance):</b> Affordance do sprzedaży.<br><br>
    Mistrz tradingu widzi affordances tam, gdzie inni widzą tylko szum informacyjny.
    </div>""", unsafe_allow_html=True)

# --- TAB 4: OFF-LOADING ---
with tabs[3]:
    st.markdown("### 🛠️ Cognitive Off-loading")
    st.caption("Zrzucanie ciężaru obliczeniowego na środowisko.")
    
    task_complexity = st.slider("Złożoność zadania (np. wycena opcji)", 1, 100, 50)
    mental_effort_internal = task_complexity
    mental_effort_with_tool = task_complexity * 0.1
    
    fig_off = go.Figure()
    fig_off.add_trace(go.Bar(x=["Wysiłek Czysto Mentalny", "Wysiłek z Barbell OS"], 
                            y=[mental_effort_internal, mental_effort_with_tool], 
                            marker_color=["#ff1744", "#00e676"]))
    fig_off.update_layout(title="Oszczędność zasobów poznawczych (Energy Efficiency)", template="plotly_dark", height=350)
    st.plotly_chart(fig_off, use_container_width=True)
    
    st.success("Zwolnione zasoby poznawcze możesz przeznaczyć na Meta-poznanie (nadzór nad strategią) zamiast na proste obliczenia.")
