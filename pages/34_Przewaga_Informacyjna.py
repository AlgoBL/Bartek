import streamlit as st
from modules.styling import apply_styling, module_header, scicard
from modules.information_theory_edge import (
    calculate_information_edge,
    calculate_bet_sizing,
    max_entropy_fusion,
    demo_information_theory
)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# 3. Main Navigation Header
st.markdown(module_header(
    title="Przewaga Informacyjna",
    subtitle="Teoria Informacji Shannona, Kryterium Kelly'ego i detekcja Smart Money (Entropia Wolumenu).",
    icon="🧠",
    badge="Meta-Decyzje"
), unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Parametry Decyzji (KL & Kelly)")
p_market = st.sidebar.slider("Prawdopodobieństwo Rynku / Bazy (%)", 1, 99, 50, 1) / 100.0
p_mine = st.sidebar.slider("Twoje Prawdopodobieństwo (Model) (%)", 1, 99, 65, 1) / 100.0
is_symmetric = st.sidebar.checkbox("Symetryczne Ryzyko/Zysk?", value=True, help="Zaznacz, jeśli obstawiasz kierunek (np. ETF) bez opcji. Odznacz i podaj ratio niżej, jeśli grasz opcjami/lewarem.")
rr_ratio = 1.0
if not is_symmetric:
    rr_ratio = st.sidebar.number_input("Risk/Reward Ratio (np. ryzykuj 1 by zyskać 3 = 3.0)", min_value=0.1, max_value=100.0, value=2.0)

st.markdown("""
Ten moduł służy jako nakładka decyzyjna (Meta-Layer) na inne systemy. Oblicza Twój obiektywny informacyjny "Edge" 
(krawędź) względem rynku oraz wyznacza optymalny procent kapitału do zainwestowania, zapobiegając nadmiernemu ryzyku.
""")

st.markdown("### 1. Rozbieżność Kullbacka-Leiblera (Edge) & Kryterium Kelly'ego")

col1, col2, col3 = st.columns(3)

if is_symmetric:
    decision = calculate_bet_sizing(p_mine, p_market, is_symmetric=True)
else:
    from modules.information_theory_edge import calculate_information_edge, kelly_fraction
    edge = calculate_information_edge(p_mine, p_market)
    full_kelly = kelly_fraction(p_mine, rr_ratio)
    decision = {
        "edge_bits": edge,
        "full_kelly": full_kelly,
        "half_kelly": full_kelly / 2.0,
        "status": "Działaj" if edge >= 0.10 and full_kelly > 0 else "Odpuść"
    }

edge_color = "normal" if decision["edge_bits"] < 0.10 else "inverse"
if decision["edge_bits"] < 0.05:
    st_color = "red"
elif decision["edge_bits"] < 0.10:
    st_color = "orange"
else:
    st_color = "green"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">INFORMACYJNY EDGE (KL)</div>
        <div class="metric-value" style="color:{st_color}">{decision['edge_bits']:.3f} bits</div>
        <div style="font-size:11px;color:var(--text-dim)">{"Masz przewagę!" if decision['edge_bits'] >= 0.10 else "Niska przewaga (<0.10)"}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ALOKACJA KELLY'EGO</div>
        <div class="metric-value" style="color:var(--cyan)">{decision['full_kelly']*100:.1f}%</div>
        <div style="font-size:11px;color:var(--text-dim)">Half-Kelly: {decision['half_kelly']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {st_color}">
        <div class="metric-label">STATUS DECYZJI</div>
        <div class="metric-value" style="color:{st_color}">{decision['status'].upper()}</div>
        <div style="font-size:11px;color:var(--text-dim)">Model p={p_mine:.0%} vs Market p={p_market:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("### 2. Detekcja Anomalii Wolumenowych (Smart Money Entropy)")
st.markdown("""
Gdy instytucje ("Smart Money") masowo akumulują pozycje starając się nie ruszać ceny, doprowadzają do skupienia wolumenu na wybranych poziomach lub dniach. 
Klasyczna zmienność (odchylenie standardowe) tego nie wychwyci, ale **Entropia Wolumenu** ulega natychmiastowemu "zapadnięciu", generując alert.
""")

# Generowanie i wyświetlanie wykresu Smart Money z dema
demo_data = demo_information_theory()
st.plotly_chart(demo_data['fig_smart_money'], use_container_width=True)

st.divider()

st.markdown("### 3. Fuzja Sygnałów (Max-Entropy)")
st.markdown("""
Naiwne uśrednianie sygnałów z różnych modułów (np. Makro + Sentyment + Przepływy) wprowadza ukryte założenia. 
Rozwiązaniem jest fuzja **Max-Entropy** (Zasada Jaynesa z 1957 r.), która matematycznie spłaszcza rozkład w miejscach, w których nic nie wiesz, bazując tylko na twardych danych i ich pewności.
""")

col_s1, col_s2, col_s3, col_res = st.columns(4)

with col_s1:
    sig1 = st.slider("Model Makro (%)", 0, 100, 40) / 100.0
    conf1 = st.slider("Pewność Makro", 0.0, 5.0, 1.0, 0.1)
with col_s2:
    sig2 = st.slider("Przepływy (Flows) (%)", 0, 100, 60) / 100.0
    conf2 = st.slider("Pewność Flows", 0.0, 5.0, 2.5, 0.1)
with col_s3:
    sig3 = st.slider("Pairs Trading (%)", 0, 100, 50) / 100.0
    conf3 = st.slider("Pewność Pairs", 0.0, 5.0, 1.5, 0.1)
    
fused_prob = max_entropy_fusion([sig1, sig2, sig3], [conf1, conf2, conf3])

with col_res:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Skonsolidowany Sygnał", f"{fused_prob*100:.1f}%")
    st.caption("Użyj tego prawdopodobieństwa jako `p_mine` w panelu wyżej do wyliczenia Kelly'ego.")
