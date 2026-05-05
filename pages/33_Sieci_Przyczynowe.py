import streamlit as st
import numpy as np
import plotly.graph_objects as go
from modules.causal_risk import get_default_financial_dag
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Sieci Przyczynowe",
    subtitle="Analiza Ryzyka Systemowego: Sieci Bayesowskie (Causal DAGs) i Propagacja Wstrząsów.",
    icon="🕸️",
    badge="Causal AI"
), unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(0,204,255,0.06);border:1px solid rgba(0,204,255,0.15);
            border-radius:12px;padding:16px;margin-bottom:20px">
Ten moduł przenosi zarządzanie ryzykiem z <b>korelacji</b> (która bywa złudna) na <b>przyczynowość</b> (Causality). 
Używając Skierowanych Grafów Acyklicznych (DAG) oraz wnioskowania opartego na twierdzeniu Bayesa (Interwencje <i>Do-Calculus</i>), modelujemy jak wstrząsy w jednym punkcie systemu propagują się i wywołują reakcje łańcuchowe.
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Metodologia: Jak działają Causal DAGs?"):
    st.markdown("""
    **Zasada działania:**
    System składa się z węzłów (wydarzeń) połączonych strzałkami przyczynowo-skutkowymi.
    Zamiast używać zwykłej macierzy korelacji, używamy **Tabel Warunkowego Prawdopodobieństwa (CPT)**.
    *   Jeśli wydarzy się Recesja, jakie jest prawdopodobieństwo Kryzysu Płynności?
    *   Co jeśli wystąpi JEDNOCZEŚNIE Recesja i Wojna Handlowa? Prawdopodobieństwo Szoku Technologicznego rośnie nieliniowo!
    
    **Tryb 'What-If' (Interwencja):**
    Możesz "wymusić" zdarzenie. Zaznacz "Wymuś", aby z symulacji wykluczyć losowość tego czynnika i ustawić go na 100% pewności. Algorytm wyliczy na nowo Prawdopodobieństwo Bankructwa (Crash Portfela) metodą Monte Carlo.
    """)

# Load DAG
dag = get_default_financial_dag()

col_controls, col_graph = st.columns([1, 2])

with col_controls:
    st.subheader("Symulator What-If (Dowody)")
    st.markdown("Wymuś wystąpienie lub brak wystąpienia konkretnych zdarzeń. Pozostawienie 'Auto', oznacza naturalne losowanie z bazowym prawdopodobieństwem.")
    
    evidence = {}
    
    st.markdown("**Warstwa Makro**")
    recesja_state = st.radio("Recesja USA:", ["Auto", "Wymuś TAK", "Wymuś NIE"], horizontal=True)
    wojna_state = st.radio("Wojna Handlowa:", ["Auto", "Wymuś TAK", "Wymuś NIE"], horizontal=True)
    
    st.markdown("**Warstwa Sektorowa**")
    szok_state = st.radio("Szok Technologiczny:", ["Auto", "Wymuś TAK", "Wymuś NIE"], horizontal=True)
    plynnosc_state = st.radio("Kryzys Płynności:", ["Auto", "Wymuś TAK", "Wymuś NIE"], horizontal=True)
    
    # Przetwarzanie Evidence
    if recesja_state == "Wymuś TAK": evidence["Recesja_USA"] = True
    elif recesja_state == "Wymuś NIE": evidence["Recesja_USA"] = False
        
    if wojna_state == "Wymuś TAK": evidence["Wojna_Handlowa"] = True
    elif wojna_state == "Wymuś NIE": evidence["Wojna_Handlowa"] = False
        
    if szok_state == "Wymuś TAK": evidence["Szok_Technologiczny"] = True
    elif szok_state == "Wymuś NIE": evidence["Szok_Technologiczny"] = False
        
    if plynnosc_state == "Wymuś TAK": evidence["Kryzys_Plynnosci"] = True
    elif plynnosc_state == "Wymuś NIE": evidence["Kryzys_Plynnosci"] = False

    if st.button("Uruchom Symulację Sieci (Monte Carlo 10k)", type="primary"):
        with st.spinner("Przeprowadzam wnioskowanie na grafie..."):
            results = dag.simulate_inference(num_samples=10000, evidence=evidence)
            st.session_state["dag_results"] = results
            
with col_graph:
    st.subheader("Graf Ryzyka (Prawdopodobieństwo Węzłów)")
    
    if "dag_results" not in st.session_state:
        # Run baseline
        st.session_state["dag_results"] = dag.simulate_inference(num_samples=10000, evidence={})
        
    res = st.session_state["dag_results"]
    
    # Hardcoded Layout for the specific 5 node DAG
    pos = {
        "Recesja_USA": (1, 3),
        "Wojna_Handlowa": (3, 3),
        "Szok_Technologiczny": (2.5, 2),
        "Kryzys_Plynnosci": (1.5, 2),
        "Crash_Portfela": (2, 1)
    }
    
    edges = [
        ("Recesja_USA", "Szok_Technologiczny"),
        ("Wojna_Handlowa", "Szok_Technologiczny"),
        ("Recesja_USA", "Kryzys_Plynnosci"),
        ("Szok_Technologiczny", "Crash_Portfela"),
        ("Kryzys_Plynnosci", "Crash_Portfela")
    ]
    
    fig = go.Figure()
    
    # Edges
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none',
            showlegend=False
        ))
        # Add arrow manually (middle point)
        fig.add_annotation(
            x=(x0+x1)/2, y=(y0+y1)/2,
            ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='gray'
        )

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node, xy in pos.items():
        node_x.append(xy[0])
        node_y.append(xy[1])
        prob = res[node]
        
        # Color gradient based on probability
        if node in evidence:
            color = "#9b59b6" # Purple for forced evidence
            label = f"<b>{node}</b><br>(WYMUSZONE)<br>P = {prob*100:.1f}%"
        else:
            color = f"rgba({int(255*prob)}, {int(200*(1-prob))}, 50, 0.9)"
            label = f"<b>{node}</b><br>P = {prob*100:.1f}%"
            
        node_text.append(label)
        node_color.append(color)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=45, color=node_color, line=dict(width=2, color='white')),
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=500,
        margin=dict(b=0, l=0, r=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Crash Result Highlighting
    crash_p = res["Crash_Portfela"]
    
    st.markdown("---")
    c_res1, c_res2 = st.columns([2, 1])
    
    with c_res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PRAWDOPODOBIEŃSTWO CRASHU PORTFELA</div>
            <div class="metric-value" style="color:{'#ff1744' if crash_p > 0.5 else '#ffea00' if crash_p > 0.2 else '#00e676'}">
                {crash_p*100:.1f}%
            </div>
            <div style="font-size:11px;color:var(--text-dim)">
                {f"{crash_p*100 - 3.2:+.1f} p.p. vs Bazowe" if "dag_results" in st.session_state else "Monte Carlo Inference"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c_res2:
        if crash_p > 0.5:
            st.error("⚠️ SYSTEMIC RISK ALERT: Prawdopodobieństwo krytyczne!")
        elif crash_p > 0.2:
            st.warning("⚠️ Ostrzeżenie: Podwyższone ryzyko.")
        else:
            st.success("✅ System stabilny.")

