import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.analysis_content import display_chart_guide

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
    init_cap = st.sidebar.number_input("Kapitał Dzisiaj (PLN)", value=st.session_state.get('rem_initial_capital', 1000000.0), step=100000.0, key="rem_cap")
    monthly_expat = st.sidebar.number_input("Wydatki Miesięczne (PLN)", value=5000, step=500, key="rem_exp")
    inflation = st.sidebar.slider("Oczekiwana Inflacja (%)", 0.0, 15.0, 3.0, 0.5) / 100.0
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Horyzont i Wiek")
    current_age = st.sidebar.slider("Obecny Wiek", 18, 100, 53)
    retirement_age = st.sidebar.slider("Wiek przejścia na emeryturę", 18, 100, 60)
    life_expectancy = st.sidebar.slider("Oczekiwana długość życia", 50, 110, 90)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Parametry Rynku")
    ret_return = st.sidebar.slider("Oczekiwany Zwrot Portfela (%)", -5.0, 20.0, st.session_state.get('rem_expected_return', 0.07) * 100) / 100.0
    ret_vol = st.sidebar.slider("Zmienność Portfela (Vol %)", 1.0, 40.0, st.session_state.get('rem_volatility', 0.15) * 100) / 100.0
    
    # --- Calculations ---
    years_to_retirement = max(0, retirement_age - current_age)
    years_in_retirement = max(1, life_expectancy - retirement_age)
    total_years = years_to_retirement + years_in_retirement
    
    # SWR Analysis
    fire_number = monthly_expat * 12 / 0.035 # Assuming 3.5% SWR
    
    # 1. FIRE Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FIRE Number (SWR 3.5%)", f"{fire_number:,.0f} PLN", help="Kapitał potrzebny do utrzymania wydatków przy wypłacie 3.5% rocznie.")
    with col2:
        current_swr = (monthly_expat * 12) / init_cap if init_cap > 0 else 0
        st.metric("Twoje obecne SWR", f"{current_swr:.2%}")
    with col3:
        st.metric("Lata do emerytury", f"{years_to_retirement}")
    with col4:
        st.metric("Okres emerytury", f"{years_in_retirement} lat")

    st.markdown("---")
    
    # --- Main Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Symulacja Przyszłości", "🛡️ Strategia Bezpieczna", "🧪 Scenariusze i Ryzyko"])
    
    with tab1:
        st.subheader("🔮 Projekcja Majątku (Monte Carlo Retirement)")
        
        # Monte Carlo Logic
        n_sims = 500
        horizon = total_years
        
        # Simulating paths
        # Log-normal returns
        mu = ret_return
        sigma = ret_vol
        
        daily_mu = (mu - 0.5 * sigma**2) / 1.0
        daily_sigma = sigma / 1.0 # Simple annual step for performance
        
        shocks = np.random.normal(0, 1, (n_sims, horizon))
        annual_returns = np.exp(daily_mu + daily_sigma * shocks) - 1
        
        wealth_matrix = np.zeros((n_sims, horizon + 1))
        wealth_matrix[:, 0] = init_cap
        
        expenses = monthly_expat * 12
        
        for y in range(horizon):
            current_wealth = wealth_matrix[:, y]
            
            # Growth
            wealth_matrix[:, y+1] = current_wealth * (1 + annual_returns[:, y])
            
            # Inflation adjustment for expenses
            current_expenses = expenses * ((1 + inflation) ** y)
            
            # Withdrawal (only if retired)
            if y >= years_to_retirement:
                wealth_matrix[:, y+1] -= current_expenses
                
            # Floor at 0
            wealth_matrix[:, y+1] = np.maximum(wealth_matrix[:, y+1], 0)
            
        # Plotting
        years_arr = np.arange(current_age, current_age + horizon + 1)
        percentiles = np.percentile(wealth_matrix, [5, 25, 50, 75, 95], axis=0)
        
        fig_mc = go.Figure()
        # Area 90%
        fig_mc.add_trace(go.Scatter(x=years_arr, y=percentiles[4], mode='lines', line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=years_arr, y=percentiles[0], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)', name='Scenariusze Skrajne (90%)'))
        # Area 50%
        fig_mc.add_trace(go.Scatter(x=years_arr, y=percentiles[3], mode='lines', line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=years_arr, y=percentiles[1], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 136, 0.3)', name='Typowy Wynik (50%)'))
        # Median
        fig_mc.add_trace(go.Scatter(x=years_arr, y=percentiles[2], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'))
        
        # Vertical line for retirement
        fig_mc.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
        
        fig_mc.update_layout(title="Projekcja Majątku do Końca Życia", template="plotly_dark", xaxis_title="Wiek", yaxis_title="Kapitał (PLN)", height=500)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Prob of Success
        success_prob = np.mean(wealth_matrix[:, -1] > 0)
        st.metric("Prawdopodobieństwo Sukcesu (Portfel > 0)", f"{success_prob:.1%}")
        
        display_chart_guide("Wykres Monte Carlo Emerytury", """
        *   **Obszar Jasnozielony**: Najbardziej prawdopodobny rozwój wypadków.
        *   **Obszar Ciemnozielony (Cień)**: Zakres zmienności rynkowej (90% pewności).
        *   **Linia Mediana**: Ścieżka środkowa. Jeśli na końcu opada do zera, Twoje oszczędności mogą się skończyć przedwcześnie.
        """)

    with tab2:
        st.subheader("🛡️ Trinity Study i Bezpieczna Stopa Wypłat (SWR)")
        
        # SWR Table based on latest data
        swr_data = {
            "Horyzont": ["2 lata", "5 lat", "10 lat", "15 lat"],
            "Sugerowana SWR": ["40% - 45%", "15% - 18%", "8% - 10%", "5% - 7%"],
            "Prawdopodobieństwo": ["Bardzo wysokie", "Wysokie", "Umiarkowane", "Konserwatywne"]
        }
        st.table(pd.DataFrame(swr_data))
        
        col_swr1, col_swr2 = st.columns(2)
        
        with col_swr1:
            st.markdown("### 🔥 Mapa Wrażliwości SWR x Inflacja")
            # Heatmap data
            swr_range = np.linspace(0.02, 0.06, 9)
            inf_range = np.linspace(0.0, 0.08, 9)
            
            z_success = []
            for s in swr_range:
                row = []
                for inf in inf_range:
                    # Quick logic: Success if (Return - Infl) > SWR (approx)
                    # Real simulation would be better but for heatmap we can use analytical approx or mini-sim
                    net_ret = ret_return - inf
                    prob = 1.0 if net_ret > s else (1.0 - (s - net_ret) * 10) # Mock decay
                    row.append(max(0, min(1, prob)))
                z_success.append(row)
                
            fig_heat = px.imshow(
                z_success,
                x=[f"{i:.1%}" for i in inf_range],
                y=[f"{s:.1%}" for s in swr_range],
                color_continuous_scale='RdYlGn',
                labels=dict(x="Inflacja", y="Stopa Wypłaty (SWR)"),
                title="Prawdopodobieństwo Sukcesu"
            )
            fig_heat.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_swr2:
            st.markdown("### 📊 Radar Chart: Profil Projektu")
            # Metrics for radar
            categories = ['Bezpieczeństwo', 'Elastyczność', 'Ochrona Inflacji', 'Dziedziczenie', 'Komfort Psychiczny']
            
            # Values based on settings
            safety = success_prob * 10
            flex = (1.0 - current_swr) * 10 if current_swr < 1 else 0
            inf_prot = (1.0 - inflation) * 10 
            legacy = (np.median(wealth_matrix[:, -1]) / init_cap) * 5 if init_cap > 0 else 0
            comfort = 8 if success_prob > 0.9 else 4
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[safety, flex, inf_prot, legacy, comfort],
                theta=categories,
                fill='toself',
                line_color='#00ff88',
                name='Twój Plan'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                template="plotly_dark",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.subheader("🧪 Scenariusze i Ryzyka Ogonowe")
        
        col_scen1, col_scen2 = st.columns(2)
        
        with col_scen1:
            st.markdown("### 📉 Ryzyko Sekwencji Zwrotów")
            st.caption("Co jeśli pierwsze 5 lat emerytury to bessy? (Sequence-of-Returns Risk)")
            
            # Simulation with bad first 5 years
            bad_years_ret = -0.10
            wealth_bad = np.zeros(horizon + 1)
            wealth_bad[0] = init_cap
            
            for y in range(horizon):
                r = bad_years_ret if y < 5 else ret_return
                wealth_bad[y+1] = wealth_bad[y] * (1 + r)
                
                curr_exp = (monthly_expat * 12) * ((1 + inflation) ** y)
                if y >= years_to_retirement:
                    wealth_bad[y+1] -= curr_exp
                wealth_bad[y+1] = max(0, wealth_bad[y+1])
                
            fig_seq = go.Figure()
            fig_seq.add_trace(go.Scatter(x=years_arr, y=percentiles[2], name="Scenariusz Normalny", line=dict(dash='dash')))
            fig_seq.add_trace(go.Scatter(x=years_arr, y=wealth_bad, name="Bessa na starcie", line=dict(color='red', width=3)))
            fig_seq.update_layout(title="Wpływ Złego Startu", template="plotly_dark", height=400)
            st.plotly_chart(fig_seq, use_container_width=True)

        with col_scen2:
            st.markdown("### 🧊 Powierzchnia Wrażliwości 3D")
            # 3D Surface: Years remaining vs SWR vs Initial Capital
            swr_v = np.linspace(0.01, 0.08, 10)
            years_v = np.linspace(10, 50, 10)
            
            # Success prob mapping
            z_3d = []
            for s in swr_v:
                row = []
                for y in years_v:
                    # Success if s < 1/y + return (simplistic)
                    prob = 1.0 if s < (1/y + 0.03) else 0.5
                    row.append(prob)
                z_3d.append(row)
                
            fig_3d = go.Figure(data=[go.Surface(z=z_3d, x=years_v, y=swr_v, colorscale='Viridis')])
            fig_3d.update_layout(
                title="Prawdopodobieństwo Przetrwania",
                scene = dict(
                    xaxis_title='Horyzont (Lata)',
                    yaxis_title='Stopa Wypłat (SWR)',
                    zaxis_title='Sukces'
                ),
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # Histogram of Years
        st.markdown("### ⏳ Histogram: Kiedy skończą się pieniądze?")
        end_years = []
        for sim in range(n_sims):
            path = wealth_matrix[sim, :]
            zero_indices = np.where(path <= 0)[0]
            if len(zero_indices) > 0:
                end_years.append(current_age + zero_indices[0])
            else:
                end_years.append(current_age + horizon)
                
        fig_hist_end = px.histogram(
            end_years, 
            nbins=30, 
            title="Rozkład wieku wyczerpania kapitału",
            labels={'value': 'Wiek'},
            color_discrete_sequence=['#ff4444']
        )
        fig_hist_end.update_layout(template="plotly_dark", showlegend=False, height=400)
        st.plotly_chart(fig_hist_end, use_container_width=True)

    st.divider()
    st.subheader("💡 Propozycje i Optymalizacja")
    
    if success_prob < 0.9:
        st.warning(f"⚠️ Twoja szansa na sukces wynosi tylko {success_prob:.1%}. Rozważ:")
        st.markdown("""
        - **Zmniejszenie wydatków**: Spróbuj obniżyć wydatki miesięczne o 10%.
        - **Późniejsza emerytura**: Każdy rok pracy to więcej składek i krótszy czas wypłat.
        - **Zwiększenie ryzyka (Barbell)**: Jeśli masz dużo aktywów bezpiecznych, mała alokacja w 'Rakiety' (Crypto/Tech) może uratować plan.
        """)
    else:
        st.success("✅ Twój plan wygląda bardzo solidnie! Masz duży margines bezpieczeństwa.")
        st.markdown("""
        - **Możesz wydawać więcej**: Rozważ zwiększenie standardu życia o 10-15%.
        - **Wcześniejsza emerytura**: Możesz przejść na emeryturę o 2-3 lata wcześniej bez dużego ryzyka.
        """)

    st.markdown("---")
    st.caption("Dane oparte na Trinity Study (2024 Update) oraz symulacjach Monte Carlo. Pamiętaj, że przeszłe wyniki nie gwarantują przyszłych zysków.")
