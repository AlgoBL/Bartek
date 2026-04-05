import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer
from modules.recession_models import calculate_sahm_rule, load_simulated_sahm_data, batch_recession_estrella_mishkin, load_simulated_yield_spread
from modules.global_settings import get_gs, apply_gs_to_session

st.markdown(apply_styling(), unsafe_allow_html=True)

_gs = get_gs()
apply_gs_to_session(_gs)

st.markdown("# 🚨 Recession Nowcasting")
st.markdown("*Modele wczesnego ostrzegania przed recesją oparte o bezrobocie (Reguła Sahm) i krzywą dochodowości (Estrella-Mishkin).*")
st.divider()

tab1, tab2 = st.tabs(["📉 Reguła Sahm (Bezrobocie)", "💸 Estrella-Mishkin (Krzywa Dochodowości)"])

with tab1:
    st.markdown("### Wskaźnik Recesji Sahm (Real-Time)")
    st.markdown("Reguła ta wskazuje początek recesji, gdy trzymiesięczna średnia ruchoma krajowej stopy bezrobocia (U3) wzrośnie o **0.50 p.p.** lub więcej względem swojego minimum z poprzednich 12 miesięcy.")
    
    st.info("Pobieranie historii makroekonomicznej (MOCK)...", icon="📡")
    unrate_series = load_simulated_sahm_data()
    sahm_res = calculate_sahm_rule(unrate_series)
    
    current_sahm_val = sahm_res['sahm_indicator'].iloc[-1]
    is_recession = sahm_res['is_recession_signal'].iloc[-1]
    
    color_s = "#ff1744" if is_recession else "#00e676"
    status_text = "RECESJA ZASYGNALIZOWANA" if is_recession else "Klarowne niebo"
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>AKTUALNY WSKAŹNIK SAHM</div>
        <div class='metric-value' style='color:{color_s}'>{current_sahm_val:.2f} p.p.</div>
        <div style='font-size:10px;color:#6b7280;'>Próg: 0.50 p.p.</div>
    </div>""", unsafe_allow_html=True)
    
    c2.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>STATUS EKONOMICZNY</div>
        <div class='metric-value' style='color:{color_s};font-size:18px;'>{status_text}</div>
    </div>""", unsafe_allow_html=True)
    
    c3.metric("Bieżące Bezrobocie (3M SMA)", f"{sahm_res['3m_sma'].iloc[-1]:.1f}%")
    
    fig_sahm = go.Figure()
    fig_sahm.add_trace(go.Scatter(x=sahm_res.index, y=sahm_res['sahm_indicator'], line=dict(color="#00ccff", width=2), name="Sahm Indicator"))
    fig_sahm.add_hline(y=0.50, line_dash="dash", line_color="#ff1744", annotation_text="Krytyczny Próg Recesji (0.50)")
    
    # Dodajemy kolor kiedy było >= 0.50
    rec_periods = sahm_res[sahm_res['is_recession_signal']]
    if not rec_periods.empty:
        fig_sahm.add_trace(go.Bar(x=rec_periods.index, y=rec_periods['sahm_indicator'], marker_color="rgba(255,23,68,0.5)", name="Sygnał Recesji (czas)"))
    
    fig_sahm.update_layout(
        template="plotly_dark", height=380,
        title="Reguła Sahm vs Próg Recesyjny w czasie",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Sahm Indicator (p.p.)"
    )
    st.plotly_chart(fig_sahm, use_container_width=True)
    
    with st.expander("🧮 Dlaczego Reguła Sahm działa tak dobrze?"):
         st.markdown(math_explainer(
            "Reguła Sahm",
            "S_t = U3_{3m} - \\min(U3_{3m, (t-12 to t)}) \\ge 0.50\\%",
            "Zamiast czekać na rewizje PKB z 6 miesięcznym opóźnieniem (ostateczne ogłoszenie NBER), Claudia Sahm zaproponowała metrykę behawioralną z rynku pracy. Kiedy bezrobocie zaczyna rosnąć, ludzie boją się o pracę, redukcję popytu -> błędne koło Keynesa zaczyna się. 0.50 to próg braku powrotu.",
            "Claudia Sahm (2019) / FRED"
         ), unsafe_allow_html=True)


with tab2:
    st.markdown("### Model Estrella-Mishkin (Krzywa Dochodowości)")
    st.markdown("Prawdopodobieństwo nadejścia recesji **za 12 miesięcy** na podstawie Spreadu 10Y-3M Treasuries. Klasyczny model probitowy New York Fed.")
    
    st.info("Pobieranie historii spreadu T10Y3M (MOCK)...", icon="📡")
    spread_series = load_simulated_yield_spread()
    prob_series = batch_recession_estrella_mishkin(spread_series)
    
    current_spread = spread_series.iloc[-1]
    current_prob = prob_series.iloc[-1]
    
    c_color = "#ff1744" if current_prob > 0.30 else "#ffea00" if current_prob > 0.15 else "#00e676"
    
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Bieżący Spread (10Y-3M)", f"{current_spread:.2f}%")
    cm2.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>PRAWDOP. RECESJI (ZA 12M)</div>
        <div class='metric-value' style='color:{c_color}'>{current_prob:.1%}</div>
    </div>""", unsafe_allow_html=True)
    
    # Rekomendowane z NBER: 30% z tego probita to próg dla recesji.
    cm3.metric("Interpretacja", "Rynki bezpieczne" if current_prob < 0.15 else "Podwyższona czujność" if current_prob < 0.3 else "Recesja wyceniana!")
    
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=prob_series.index, y=prob_series.values * 100, fill="tozeroy", line=dict(color="#f39c12", width=2), fillcolor="rgba(243, 156, 18, 0.2)", name="Prawdopodobieństwo t+12"))
    fig_prob.add_hline(y=30, line_dash="dash", line_color="#ff1744", annotation_text="Krytyczny Próg NY Fed (30%)")
    
    fig_prob.update_layout(
        template="plotly_dark", height=380,
        title="Prawdopodobieństwo Recesji wg Estrella-Mishkin w czasie (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Prawdopodobieństwo (%)", yaxis_range=[0, 100]
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    
    with st.expander("🧮 Model Probitowy NY Fed"):
         st.markdown(math_explainer(
            "Estrella-Mishkin Recession Probability",
            "P(\\text{Recesja}_{t+12}) = \\Phi(-2.17 - 0.76 \\cdot \\text{Spread}_{10Y-3M})",
            "Gdzie Φ to dystrybuanta standardowego rozkładu normalnego. Odwrócenie krzywej dochodowości (Spread ujemny) dramatycznie zwiększa argument wewnątrz funkcji, dając P > 30%, co historycznie niemal zawsze zwiastowało recesję w ciągu 12 miesięcy.",
            "Arturo Estrella, Frederic S. Mishkin (1998)"
         ), unsafe_allow_html=True)
