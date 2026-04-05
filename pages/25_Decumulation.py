"""25_Decumulation.py — Strategie Bezpiecznej Wypłaty (Decumulation)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer
from modules.decumulation_engine import simulate_bengen_swr, vpw_simulation
from modules.global_settings import get_gs, apply_gs_to_session
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)

# Ladowanie globalnych parametrow
_gs = get_gs()
apply_gs_to_session(_gs)

st.markdown("# 💸 Decumulation & Retirement Income")
st.markdown("*Jak bezpiecznie wypłacać kapitał, unikać bankructwa (Sequence of Returns Risk) i radzić sobie z inflacją na emeryturze.*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Parametry Portfela")
    initial_cap = st.number_input("Kapitał startowy (PLN)", value=1000000, step=100000)
    mu_port = st.slider("Oczekiwana (nominalna) stopa zwrotu", 0.0, 0.15, 0.07)
    vol_port = st.slider("Zmienność portfela (%)", 0.0, 0.30, 0.12)
    inflation = st.slider("Inflacja bazowa (%)", 0.0, 0.10, 0.035)

tab1, tab2, tab3 = st.tabs(["⚠️ Safe Withdrawal Rate (SWR)", "🔄 Variable Percentage Withdrawal (VPW)", "📉 Sequence of Returns Risk"])

with tab1:
    st.markdown("### Klasyczna Reguła 4% (Bengen's SWR)")
    st.markdown("Określa, ile można stale wypłacać co roku (korygując o inflację) bez wyczerpania portfela przez X lat.")
    
    col1, col2 = st.columns(2)
    with col1:
        withdrawal_rate = st.slider("Początkowa stopa wypłaty (SWR)", 0.02, 0.08, 0.04, 0.005, format="%.3f")
    with col2:
        retirement_years = st.slider("Długość trwania emerytury (Lata)", 10, 50, 30, 5)
        
    res_swr = simulate_bengen_swr(
        initial_capital=initial_cap,
        withdrawal_rate=withdrawal_rate,
        years=retirement_years,
        mu=mu_port,
        vol=vol_port,
        inflation_rate=inflation,
        num_simulations=1000
    )
    
    c1, c2, c3 = st.columns(3)
    sr = res_swr['success_rate']
    
    clr = "#ff1744" if sr < 0.8 else "#ffea00" if sr < 0.95 else "#00e676"
    c1.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>PRAWDOPODOBIEŃSTWO SUKCESU</div>
        <div class='metric-value' style='color:{clr}'>{sr:.1%}</div>
        <div style='font-size:10px;color:#6b7280;'>brak bankructwa</div>
    </div>""", unsafe_allow_html=True)
    
    first_year_withdrawal = initial_cap * withdrawal_rate
    c2.metric("Wypłata (Rok 1)", f"{first_year_withdrawal:,.0f} PLN")
    
    med = res_swr['median_final']
    c3.metric("Mediana Kapitału Koniec", f"{med:,.0f} PLN")

    # Wykres pajączek dla ścieżek
    fig_swr = go.Figure()
    
    paths = res_swr["paths"]
    years_arr = np.arange(retirement_years + 1)
    
    # percentyle
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.median(paths, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    fig_swr.add_trace(go.Scatter(x=years_arr, y=p95, line=dict(color="rgba(0,230,118,0.1)"), showlegend=False))
    fig_swr.add_trace(go.Scatter(x=years_arr, y=p5, fill="tonexty", fillcolor="rgba(0,230,118,0.1)", line=dict(color="rgba(0,230,118,0.1)"), name="5-95 percentyl"))
    fig_swr.add_trace(go.Scatter(x=years_arr, y=p50, line=dict(color="#00e676", width=2.5), name="Mediana Portfela"))
    
    fig_swr.update_layout(
        template="plotly_dark", height=380,
        title="Symulacja Wartości Portfela (Stała Wypłata)",
        xaxis_title="Rok emerytury", yaxis_title="Wartość Portfela (PLN)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_swr, use_container_width=True)
    
    with st.expander("🧮 SWR The 4% Rule Explained"):
        st.markdown(math_explainer(
            "Bengen Safe Withdrawal Rate",
            "W_t = W_0 \\cdot (1 + \\pi)^t",
            "Stała poczatkowa stopa wypłaty podnoszona o inflację. W okresie posuchy portfel szybko się zjada. Bardzo narażony na Sequence of Returns Risk.",
            "William Bengen (1994), The Trinity Study"
        ), unsafe_allow_html=True)


with tab2:
    st.markdown("### Variable Percentage Withdrawal (VPW - Vanguard/Bogleheads)")
    st.markdown("Wypłacasz stały *procent* swojego pozostałego portfela, a nie stałą *kwotę*. Portfel nigdy nie upadnie z matematycznego punktu widzenia, ale siła nabywcza spada w okresie kryzysu.")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        ret_age = st.slider("Wiek przejścia na emeryturę", 50, 75, 65)
    with v_col2:
        max_age_val = st.slider("Oczekiwana długość życia (Max Age)", 85, 110, 100)
        
    res_vpw = vpw_simulation(
        initial_capital=initial_cap,
        retire_age=ret_age,
        max_age=max_age_val,
        mu=mu_port,
        vol=vol_port,
        inflation_rate=inflation,
        num_simulations=1000
    )
    
    vpw_years = np.arange(max_age_val - ret_age + 1)
    vpw_ages = vpw_years + ret_age
    
    med_real_drw = res_vpw["median_real_withdrawal"]
    p5_real_drw = np.percentile(res_vpw["real_withdrawals"], 5, axis=0)
    p95_real_drw = np.percentile(res_vpw["real_withdrawals"], 95, axis=0)
    
    fig_vpw_drw = go.Figure()
    fig_vpw_drw.add_trace(go.Scatter(x=vpw_ages, y=p95_real_drw, line=dict(color="rgba(0,204,255,0.1)"), showlegend=False))
    fig_vpw_drw.add_trace(go.Scatter(x=vpw_ages, y=p5_real_drw, fill="tonexty", fillcolor="rgba(0,204,255,0.1)", line=dict(color="rgba(0,204,255,0.1)"), name="5-95 percentyl (Realnie)"))
    fig_vpw_drw.add_trace(go.Scatter(x=vpw_ages, y=med_real_drw, line=dict(color="#00ccff", width=2.5), name="Mediana Wypłaty (skowrygowana o inflację)"))
    
    fig_vpw_drw.update_layout(
         template="plotly_dark", height=380,
         title="Ile faktycznie będziesz mógł kupić? (Realne Wypłaty w VPW)",
         xaxis_title="Wiek", yaxis_title="Siła Nabywcza Wypłaty (Wartość Dzisiejsza PLN)",
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_vpw_drw, use_container_width=True)

    with st.expander("🧮 Jak działa VPW?"):
        st.markdown(math_explainer(
            "Variable Percentage Withdrawal",
            "P_t = \\frac{R}{1 - (1+R)^{-n}}",
            "Opiera się na amortyzacji portfela, stąd naturalnie zwiększa się % wypłaty z wiekiem (bo maleje horyzont). Utrzymuje portfel we wszystkich warunkach kosztem stabilności cashflowu.",
            "Bogleheads Retirement Planning"
        ), unsafe_allow_html=True)


with tab3:
    st.markdown("### 📉 Sequence of Returns Risk")
    st.markdown("Wyobraź sobie dwa portfele z **dokładnie taką samą stopą CAGR** przez 30 lat. W jednym krach jest w pierwszym roku formowania wypłat, w drugim — 20 lat później. Sytuacja diametralnie inna!")
    
    st.info("Symulacja: Portfel o stałym średnim zwrocie 5%, stała zmienność 15%. Bierzemy 1 pechową i 1 szczęśliwą dekadę układając w chronologii lub od tyłu.")
    
    # Hardcode bad seq vs good seq
    # bad first: -20, -10, 0, 10, 20
    # good first: 20, 10, 0, -10, -20
    returns_seq = np.array([-0.25, -0.10, -0.05, 0.15, 0.20, 0.08, 0.12, 0.18, 0.10, 0.05])
    returns_good = returns_seq[::-1]
    
    years_seq = 10
    path_bad = [initial_cap]
    path_good = [initial_cap]
    
    const_w = initial_cap * 0.08 # dosc agresywne wyplaty zeby unaocznic wyczerpanie
    
    for y in range(years_seq):
        # bad
        pb = path_bad[-1] - const_w
        if pb > 0:
            pb = pb * (1 + returns_seq[y])
        else: pb = 0
        path_bad.append(pb)
        
        # good
        pg = path_good[-1] - const_w
        if pg > 0:
            pg = pg * (1 + returns_good[y])
        else: pg = 0
        path_good.append(pg)

    fig_seq = go.Figure()
    fig_seq.add_trace(go.Scatter(y=path_bad, mode="lines+markers", name="Pechowy Początek (Krachy w 1-3 r.)", line=dict(color="#ff1744", width=3)))
    fig_seq.add_trace(go.Scatter(y=path_good, mode="lines+markers", name="Szczęśliwy Początek (Hossa w 1-3 r.)", line=dict(color="#00e676", width=3)))
    
    fig_seq.update_layout(
         template="plotly_dark", height=380,
         title="Wykres: Różnica Sequence Risk dla tego samego CAGR i Kapitału Pocz.",
         xaxis_title="Lata Emerytury", yaxis_title="Kapitał z ujemnymi rzędowymi",
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_seq, use_container_width=True)
