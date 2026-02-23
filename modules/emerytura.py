import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.analysis_content import display_chart_guide


# ---------------------------------------------------------------------------
# Pomocniki do persystencji widżetów między modułami
# Streamlit usuwa klucze widżetów z session_state, gdy widżet nie jest renderowany.
# Rozwiązanie: on_change zapisuje do klucza "_s.<key>", a value= czyta z niego.
# ---------------------------------------------------------------------------
def _save(widget_key):
    """Callback on_change: kopiuje wartość widżetu do trwałego klucza."""
    st.session_state[f"_s.{widget_key}"] = st.session_state[widget_key]

def _saved(widget_key, default):
    """Zwraca ostatnio zapisaną wartość lub domyślną."""
    return st.session_state.get(f"_s.{widget_key}", default)


def render_emerytura_module():
    st.header("🏖️ Analiza Emerytury i Przyszłości Finansowej")
    st.markdown("""
    Ten moduł pozwala zaplanować Twoją wolność finansową. Analizuje bezpieczeństwo portfela w horyzoncie długoterminowym, 
    wykorzystując dane z Twoich symulacji oraz standardy naukowe (Trinity Study 2024).
    """)

    # --- Sidebar Configuration ---
    st.sidebar.title("🛠️ Parametry Emerytury")
    
    # 1. Import Data
    st.sidebar.markdown("### 📥 Dane z Portfela")
    if st.sidebar.button("🔄 Wczytaj z Symulatora"):
        if 'mc_results' in st.session_state:
            res = st.session_state['mc_results']
            st.session_state['rem_initial_capital'] = float(res['wealth_paths'][0, 0])
            st.session_state['rem_expected_return'] = float(res['metrics']['mean_cagr'])
            st.session_state['rem_volatility'] = float(res['metrics']['median_volatility'])
            st.sidebar.success("Wczytano wyniki Monte Carlo!")
        elif 'backtest_results' in st.session_state:
            res = st.session_state['backtest_results']
            st.session_state['rem_initial_capital'] = float(res['results']['PortfolioValue'].iloc[-1])
            st.session_state['rem_expected_return'] = float(res['metrics']['mean_cagr'])
            st.session_state['rem_volatility'] = float(res['metrics']['median_volatility'])
            st.sidebar.success("Wczytano wyniki Backtestu AI!")
        else:
            st.sidebar.warning("Brak aktywnych symulacji w pamięci.")

    # 2. Manual Inputs
    init_cap = st.sidebar.number_input(
        "Kapitał Dzisiaj (PLN)",
        value=_saved("rem_cap", st.session_state.get('rem_initial_capital', 1000000.0)),
        step=100000.0,
        key="rem_cap",
        on_change=_save, args=("rem_cap",)
    )
    monthly_expat = st.sidebar.number_input(
        "Wydatki Miesięczne (PLN)",
        value=_saved("rem_monthly_expat", 5000),
        step=500,
        key="rem_monthly_expat",
        on_change=_save, args=("rem_monthly_expat",)
    )
    inflation = st.sidebar.slider(
        "Oczekiwana Inflacja (%)", 0.0, 15.0,
        value=_saved("rem_inflation", 3.0),
        step=0.5,
        key="rem_inflation",
        on_change=_save, args=("rem_inflation",)
    ) / 100.0
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Horyzont i Wiek")
    current_age = st.sidebar.slider(
        "Obecny Wiek", 18, 100,
        value=_saved("rem_current_age", 53),
        key="rem_current_age",
        on_change=_save, args=("rem_current_age",)
    )
    retirement_age = st.sidebar.slider(
        "Wiek przejścia na emeryturę", 18, 100,
        value=_saved("rem_retirement_age", 60),
        key="rem_retirement_age",
        on_change=_save, args=("rem_retirement_age",)
    )
    life_expectancy = st.sidebar.slider(
        "Oczekiwana długość życia", 50, 110,
        value=_saved("rem_life_expectancy", 90),
        key="rem_life_expectancy",
        on_change=_save, args=("rem_life_expectancy",)
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Dodatkowy Dochód i Wpłaty")
    enable_contributions = st.sidebar.checkbox(
        "Aktywuj dodatkowy dochód/wpłaty",
        value=_saved("rem_contrib_enabled", False),
        key="rem_contrib_enabled",
        on_change=_save, args=("rem_contrib_enabled",),
        help="Regularne wpłaty zwiększają kapitał. Możesz wybrać czy trwają tylko do emerytury, czy przez całe życie."
    )
    monthly_contribution = st.sidebar.slider(
        "Kwota Dodatkowa (PLN/mies)",
        0, 30000,
        value=_saved("rem_monthly_contrib", 5000),
        step=500,
        key="rem_monthly_contrib",
        on_change=_save, args=("rem_monthly_contrib",),
        disabled=not enable_contributions,
        help="Kwota dokładana co miesiąc do portfela (faza akumulacji) lub dostępna jako dodatkowy dochód (faza emerytury)."
    )
    contrib_during_retirement = st.sidebar.checkbox(
        "Dochód dostępny również na emeryturze",
        value=_saved("rem_contrib_ret", False),
        key="rem_contrib_ret",
        on_change=_save, args=("rem_contrib_ret",),
        disabled=not enable_contributions,
        help="Jeśli zaznaczone, ten dochód będzie sumowany z wypłatami z portfela po osiągnięciu wieku emerytalnego."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Parametry Rynku")
    ret_return = st.sidebar.slider(
        "Oczekiwany Zwrot Portfela (%)", -5.0, 20.0,
        value=_saved("rem_ret_return", 7.0),
        step=0.5,
        key="rem_ret_return",
        on_change=_save, args=("rem_ret_return",)
    ) / 100.0
    ret_vol = st.sidebar.slider(
        "Zmienność Portfela (Vol %)", 1.0, 40.0,
        value=_saved("rem_ret_vol", 15.0),
        step=0.5,
        key="rem_ret_vol",
        on_change=_save, args=("rem_ret_vol",)
    ) / 100.0
    
    # --- Calculations ---
    years_to_retirement = max(0, retirement_age - current_age)
    years_in_retirement = max(1, life_expectancy - retirement_age)
    total_years = years_to_retirement + years_in_retirement
    
    annual_contribution = monthly_contribution * 12 if enable_contributions else 0
    total_contributions = annual_contribution * years_to_retirement

    # SWR Analysis
    fire_number = monthly_expat * 12 / 0.035

    # 1. FIRE Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FIRE Number (SWR 3.5%)", f"{fire_number:,.0f} PLN",
                  help="Kapitał potrzebny do utrzymania wydatków przy wypłacie 3.5% rocznie.")
    with col2:
        current_swr = (monthly_expat * 12) / init_cap if init_cap > 0 else 0
        st.metric("Twoje obecne SWR", f"{current_swr:.2%}")
    with col3:
        st.metric("Lata do emerytury", f"{years_to_retirement}")
    with col4:
        st.metric("Okres emerytury", f"{years_in_retirement} lat")

    # Income Info Row
    st.markdown("### 💸 Budżet i Przepływy")
    inc_col1, inc_col2, inc_col3 = st.columns(3)
    with inc_col1:
        total_monthly_retirement = monthly_expat + (monthly_contribution if contrib_during_retirement else 0)
        st.metric("Budżet na Emeryturze", f"{total_monthly_retirement:,.0f} PLN", 
                  help="Suma wypłaty z portfela + dodatkowy dochód (jeśli zaznaczono).")
    with inc_col2:
        st.metric("Dodatkowy Dochód", f"{monthly_contribution:,.0f} PLN", delta=f"{monthly_contribution*12:,.0f} PLN/rok", delta_color="normal")
    with inc_col3:
        st.metric("Portfel pokrywa", f"{(monthly_expat / total_monthly_retirement):.1%}" if total_monthly_retirement > 0 else "0%", help="Jaki procent Twojego budżetu pochodzi z wypłat z inwestycji.")

    st.markdown("---")
    
    # --- Main Analysis Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Symulacja Przyszłości", "🛡️ Strategia Bezpieczna", "🧪 Scenariusze i Ryzyko", "💰 Analiza Dochodu"])
    
    with tab1:
        st.subheader("🔮 Projekcja Majątku (Monte Carlo Retirement)")
        
        # Monte Carlo Settings
        n_sims = 500
        horizon = total_years
        
        # Prepare simulation data
        mu = ret_return
        sigma = ret_vol
        daily_mu = (mu - 0.5 * sigma**2) / 1.0
        daily_sigma = sigma / 1.0

        # We simulate two scenarios: With Contributions vs Without
        wealth_matrix = np.zeros((n_sims, horizon + 1))
        wealth_matrix_no_contrib = np.zeros((n_sims, horizon + 1))
        
        wealth_matrix[:, 0] = init_cap
        wealth_matrix_no_contrib[:, 0] = init_cap
        
        shocks = np.random.normal(0, 1, (n_sims, horizon))
        annual_returns = np.exp(daily_mu + daily_sigma * shocks) - 1
        
        expenses = monthly_expat * 12
        
        for y in range(horizon):
            # GROWTH - Apply 19% Belka Tax on annual gains
            ret = annual_returns[:, y]
            taxed_ret = np.where(ret > 0, ret * 0.81, ret)
            
            wealth_matrix[:, y+1] = wealth_matrix[:, y] * (1 + taxed_ret)
            wealth_matrix_no_contrib[:, y+1] = wealth_matrix_no_contrib[:, y] * (1 + taxed_ret)
            
            # INFLATION ADJUSTMENT
            inf_factor = (1 + inflation) ** y
            curr_expenses = expenses * inf_factor
            curr_contrib = annual_contribution * inf_factor
            
            # PHASE LOGIC
            if y < years_to_retirement:
                # Accumulation
                if enable_contributions:
                    wealth_matrix[:, y+1] += curr_contrib
            else:
                # Retirement - apply withdrawals
                wealth_matrix[:, y+1] -= curr_expenses
                wealth_matrix_no_contrib[:, y+1] -= curr_expenses
                # In scenario with contributions, if active during retirement, we could arguably subtract less from portfolio
                # But typically 'expenses' IS what we want to spend. 
                # If we have extra income, we withdraw LESS from portfolio to meet the same expense goal.
                if enable_contributions and contrib_during_retirement:
                    # We only withdraw (expenses - additional_income) from portfolio
                    # So we ADD back the additional income to the wealth matrix
                    wealth_matrix[:, y+1] += curr_contrib
            
            # Floor at 0
            wealth_matrix[:, y+1] = np.maximum(wealth_matrix[:, y+1], 0)
            wealth_matrix_no_contrib[:, y+1] = np.maximum(wealth_matrix_no_contrib[:, y+1], 0)
            
        years_arr = np.arange(current_age, current_age + horizon + 1)
        
        # Comparison Toggle
        show_comparison = st.checkbox("Pokaż wpływ dodatkowego dochodu (vs brak wpłat)", value=True)
        
        fig_mc = go.Figure()
        
        # Scenario: WITH CONTRIBS
        p_with = np.percentile(wealth_matrix, [5, 50, 95], axis=0)
        fig_mc.add_trace(go.Scatter(x=years_arr, y=p_with[2], mode='lines', line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=years_arr, y=p_with[0], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)', name='90% Przedział (Z dochodem)'))
        fig_mc.add_trace(go.Scatter(x=years_arr, y=p_with[1], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana (Z dochodem)'))
        
        if show_comparison:
            # Scenario: WITHOUT CONTRIBS
            p_no = np.percentile(wealth_matrix_no_contrib, [50], axis=0)
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p_no[0], mode='lines', line=dict(color='white', width=2, dash='dash'), name='Mediana (Bez dochodu)'))
            
            # Area between Medians
            fig_mc.add_trace(go.Scatter(
                x=years_arr, 
                y=p_with[1],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig_mc.add_trace(go.Scatter(
                x=years_arr, 
                y=p_no[0],
                fill='tonexty',
                fillcolor='rgba(255, 200, 0, 0.2)',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Zysk z dodatkowych wpłat'
            ))

        fig_mc.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
        
        fig_mc.update_layout(
            title="Wpływ Dochodu na Kapitał Trwały",
            template="plotly_dark",
            xaxis_title="Wiek",
            yaxis_title="Kapitał (PLN)",
            height=500
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        success_prob = np.mean(wealth_matrix[:, -1] > 0)
        success_prob_no = np.mean(wealth_matrix_no_contrib[:, -1] > 0)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Szansa Sukcesu", f"{success_prob:.1%}", delta=f"{(success_prob - success_prob_no):.1%}" if show_comparison else None)
        m2.metric("Finalny Majątek (Mediana)", f"{np.median(wealth_matrix[:, -1]):,.0f} PLN")
        m3.metric("Wartość dodana dochodu", f"{(np.median(wealth_matrix[:, -1]) - np.median(wealth_matrix_no_contrib[:, -1])):,.0f} PLN")

    with tab2:
        st.subheader("🛡️ Trinity Study i Bezpieczna Stopa Wypłat (SWR)")
        # SWR visualization updated for "Residual Net SWR"
        # If we have extra income, our 'needed' SWR from portfolio is lower.
        effective_annual_spending = (monthly_expat * 12) - (annual_contribution if contrib_during_retirement else 0)
        effective_annual_spending = max(0, effective_annual_spending)
        net_swr = effective_annual_spending / init_cap if init_cap > 0 else 0
        
        st.info(f"Twoje zapotrzebowanie na wypłaty z portfela wynosi **{effective_annual_spending:,.0f} PLN rocznie** (po uwzględnieniu dodatkowego dochodu). To daje realne SWR na poziomie **{net_swr:.2%}**.")
        
        col_swr1, col_swr2 = st.columns(2)
        with col_swr1:
            st.markdown("### 🔥 Mapa Wrażliwości SWR x Inflacja")
            # Sensitivity Map: Success probability using net-of-tax returns
            swr_range = np.linspace(0.02, 0.06, 9)
            inf_range = np.linspace(0.0, 0.08, 9)
            net_ret_base = ret_return * 0.81
            z_success = [[max(0, min(1, (1.0 if (net_ret_base - inf) > s else (1.0 - (s - (net_ret_base - inf)) * 10)))) for inf in inf_range] for s in swr_range]
            fig_heat = px.imshow(
                z_success,
                x=inf_range,
                y=swr_range,
                color_continuous_scale='RdYlGn',
                labels=dict(x="Inflacja", y="Stopa Wypłaty (SWR)"),
                title="Prawdopodobieństwo Sukcesu"
            )
            # Używamy wartości numerycznych dla osi, ale formatujemy jako procenty
            fig_heat.update_xaxes(tickvals=inf_range, ticktext=[f"{i:.1%}" for i in inf_range])
            fig_heat.update_yaxes(tickvals=swr_range, ticktext=[f"{s:.1%}" for s in swr_range])
            
            fig_heat.add_hline(y=min(0.06, net_swr), line_dash="dash", line_color="white", annotation_text="Twoje Netto SWR")
            fig_heat.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_swr2:
            st.markdown("### 📊 Radar: Bezpieczeństwo Planu")
            safety = success_prob * 10
            flex = (1.0 - net_swr) * 10 if net_swr < 1 else 0
            inf_prot = (1.0 - inflation) * 10 
            legacy = (np.median(wealth_matrix[:, -1]) / init_cap) * 5 if init_cap > 0 else 0
            comfort = 9 if success_prob > 0.95 else (7 if success_prob > 0.8 else 4)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[safety, flex, inf_prot, legacy, comfort], theta=['Bezpieczeństwo', 'Elastyczność', 'Ochrona Inflacji', 'Dziedziczenie', 'Komfort'], fill='toself', line_color='#00ff88'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), template="plotly_dark", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.subheader("🧪 Scenariusze i Ryzyko Sekwencji")
        # Update sequence risk to account for contributions
        bad_years_ret = -0.10
        w_bad = np.zeros(horizon + 1)
        w_bad_no = np.zeros(horizon + 1)
        w_bad[0] = init_cap
        w_bad_no[0] = init_cap
        
        for y in range(horizon):
            r = bad_years_ret if y < 5 else ret_return
            # Apply 19% tax to positive returns
            r_taxed = r * 0.81 if r > 0 else r
            inf_f = (1 + inflation) ** y
            
            w_bad[y+1] = w_bad[y] * (1 + r_taxed)
            w_bad_no[y+1] = w_bad_no[y] * (1 + r_taxed)
            
            if y < years_to_retirement:
                if enable_contributions: w_bad[y+1] += annual_contribution * inf_f
            else:
                w_bad[y+1] -= expenses * inf_f
                w_bad_no[y+1] -= expenses * inf_f
                if enable_contributions and contrib_during_retirement: w_bad[y+1] += annual_contribution * inf_f
            
            w_bad[y+1] = max(0, w_bad[y+1])
            w_bad_no[y+1] = max(0, w_bad_no[y+1])

        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(x=years_arr, y=w_bad, name="Z dodatkowym dochodem", line=dict(color='#00ff88', width=3)))
        fig_seq.add_trace(go.Scatter(x=years_arr, y=w_bad_no, name="Bez dodatkowego dochodu", line=dict(color='red', width=2, dash='dot')))
        fig_seq.update_layout(title="Scenariusz: Fatalny Start (Bessa co roku przez pierwsze 5 lat)", template="plotly_dark", height=400, hovermode="x unified")
        fig_seq.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        fig_seq.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        st.plotly_chart(fig_seq, use_container_width=True)

    with tab4:
        st.subheader("💰 Analiza Przepływów Pieniężnych (Cash Flow)")
        st.markdown("Poniższy wykres pokazuje realną kwotę dostępną do życia co miesiąc (skorygowaną o inflację).")
        
        income_years = np.arange(current_age, current_age + horizon)
        port_withdrawals = []
        extra_incomes = []
        
        for y in range(horizon):
            inf_f = (1 + inflation) ** y
            # Extra income
            if enable_contributions:
                if y < years_to_retirement or contrib_during_retirement:
                    extra_incomes.append(monthly_contribution * inf_f)
                else:
                    extra_incomes.append(0)
            else:
                extra_incomes.append(0)
            
            # Portfolio income
            if y >= years_to_retirement:
                port_withdrawals.append(monthly_expat * inf_f)
            else:
                port_withdrawals.append(0)
        
        income_df = pd.DataFrame({
            "Wiek": income_years,
            "Wypłata z Portfela": port_withdrawals,
            "Dodatkowy Dochód": extra_incomes,
            "Suma do dyspozycji": [a+b for a,b in zip(port_withdrawals, extra_incomes)]
        })
        
        fig_income = go.Figure()
        fig_income.add_trace(go.Bar(x=income_df["Wiek"], y=income_df["Wypłata z Portfela"], name="Wypłata z Portfela (SWR)", marker_color="#00ccff"))
        fig_income.add_trace(go.Bar(x=income_df["Wiek"], y=income_df["Dodatkowy Dochód"], name="Dodatkowy Dochód", marker_color="#ffaa00"))
        
        fig_income.update_layout(
            barmode='stack', 
            title="Realna miesięczna kwota do dyspozycji (z inflacją)", 
            template="plotly_dark",
            xaxis_title="Wiek",
            yaxis_title="PLN / Miesiąc",
            showlegend=True,
            hovermode="x unified"
        )
        fig_income.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        fig_income.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Current Real-Time Disposable Income Metric Area
        st.success(f"### 🎯 Twoja kwota do życia (Wiek {retirement_age}): **{total_monthly_retirement:,.0f} PLN**")
        st.markdown(f"""
        W momencie przejścia na emeryturę będziesz dysponować kwotą:
        - **{monthly_expat:,.0f} PLN** z Twojego portfela (SWR).
        - **{monthly_contribution if contrib_during_retirement else 0:,.0f} PLN** z dodatkowego dochodu.
        
        *Wszystkie kwoty powyżej są podane w dzisiejszej sile nabywczej. System uwzględnia inflację w symulacji kapitału.*
        """)

    st.divider()
    st.subheader("💡 Wnioski i Rekomendacje")
    if success_prob < 0.9:
        st.warning(f"⚠️ Twoja szansa na sukces ({success_prob:.1%}) jest poniżej progu bezpieczeństwa (90%).")
        if not contrib_during_retirement and enable_contributions:
            st.info("💡 **Tip**: Jeśli możesz utrzymać dodatkowy dochód również na emeryturze, zwiększysz bezpieczeństwo portfela (zmniejszysz wypłaty).")
    else:
        st.success("✅ Twój plan jest antykruchy. Dodatkowy dochód zapewnia Ci ogromny margines błędu.")

    st.caption("Analiza uwzględnia podatki i inflację w sposób uproszczony (realne stopy zwrotu). Dane oparte na modelu Monte Carlo (500 ścieżek).")
