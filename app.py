
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling
from modules.simulation import simulate_barbell_strategy, calculate_metrics, run_ai_backtest, calculate_individual_metrics
from modules.ai.data_loader import load_data
from modules.analysis_content import display_analysis_report, display_scanner_methodology, display_chart_guide
from modules.scanner import calculate_convecity_metrics, score_asset

# ... existsing code ...

# 1. Page Configuration
st.set_page_config(
    page_title="Barbell Strategy Simulator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# Navigation State Handler
if "force_navigate" in st.session_state:
    st.session_state["module_nav"] = st.session_state.pop("force_navigate")

# 3. Main Navigation
module_selection = st.radio("Wybierz Moduł:", ["📉 Symulator Portfela", "🔍 Skaner Wypukłości (BCS)"], horizontal=True, label_visibility="collapsed", key="module_nav")
st.markdown("---")

if module_selection == "📉 Symulator Portfela":
    st.sidebar.title("🛠️ Konfiguracja Strategii")
    
    mode = st.sidebar.radio("Tryb Symulacji", ["Monte Carlo (Teoretyczny)", "Intelligent Barbell (Backtest AI)"], key="sim_mode")

    if mode == "Monte Carlo (Teoretyczny)":
        st.sidebar.markdown("### 1. Kapitał i Czas")
        initial_capital = st.sidebar.number_input("Kapitał Początkowy (PLN)", value=100000, step=10000, key="mc_cap")
        years = st.sidebar.slider("Horyzont Inwestycyjny (Lata)", 1, 30, 10, key="mc_years")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 2. Część Bezpieczna (Safe Sleeve)")
        st.sidebar.info("🔒 Obligacje Skarbowe RP 3-letnie (Stałe 5.51%)")
        safe_rate = 0.0551 

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 3. Część Ryzykowna (Risky Sleeve)")
        risky_mean = st.sidebar.slider("Oczekiwany Zwrot Roczny (Średnia)", -0.20, 0.50, 0.08, 0.01)
        risky_vol = st.sidebar.slider("Zmienność Roczna (Volatility)", 0.10, 1.50, 0.50, 0.05)
        risky_kurtosis = st.sidebar.slider("Grubość Ogonów (Kurtosis)", 2.1, 30.0, 4.0, 0.1)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 4. Optymalizacja Kelly'ego")
        use_kelly = st.sidebar.checkbox("Użyj Kryterium Kelly'ego", key="mc_kelly")
        
        kelly_fraction = 1.0
        kelly_shrinkage = 0.0
        
        if use_kelly:
            kelly_fraction = st.sidebar.slider("Ułamek Kelly'ego (Fraction)", 0.1, 1.0, 0.25, 0.05)
            kelly_shrinkage = st.sidebar.slider("Czynnik Kurczenia (Shrinkage)", 0.0, 0.9, 0.1, 0.05)
            
            if risky_vol > 0:
                kelly_full = (risky_mean - safe_rate) / (risky_vol ** 2)
            else:
                kelly_full = 0
                
            kelly_optimal = kelly_full * kelly_fraction * (1 - kelly_shrinkage)
            kelly_optimal = max(0.0, min(1.0, kelly_optimal)) 
            
            st.sidebar.markdown(f"**Wynik Kelly'ego:** `{kelly_optimal:.2%}`")
            alloc_safe = 1.0 - kelly_optimal
        else:
            st.sidebar.markdown("### 5. Alokacja Manualna")
            alloc_safe = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, 85, key="mc_alloc_safe") / 100.0

        rebalance_strategy = st.sidebar.selectbox(
            "Strategia Rebalansowania",
            ["None (Buy & Hold)", "Yearly", "Monthly", "Threshold (Shannon's Demon)"],
            key="mc_rebalance"
        )
        
        threshold_percent = 0.0
        if rebalance_strategy == "Threshold (Shannon's Demon)":
            threshold_percent = st.sidebar.slider("Próg Rebalansowania (%)", 5, 50, 20, 5, key="mc_threshold") / 100.0

        # MAIN CONTENT FOR MONTE CARLO
        st.title("⚖️ Barbell Strategy - Monte Carlo")
        
        if st.button("🚀 Symuluj Wyniki", type="primary", key="mc_run"):
            with st.spinner("Symulacja..."):
                wealth_paths = simulate_barbell_strategy(
                    n_years=years,
                    n_simulations=1000,
                    initial_captial=initial_capital,
                    safe_rate=safe_rate,
                    risky_mean=risky_mean,
                    risky_vol=risky_vol,
                    risky_kurtosis=risky_kurtosis,
                    alloc_safe=alloc_safe,
                    rebalance_strategy=rebalance_strategy.split(" ")[0],
                    threshold_percent=threshold_percent
                )
                metrics = calculate_metrics(wealth_paths, years)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Średni Kapitał", f"{metrics['mean_final_wealth']:,.0f} PLN")
            col2.metric("CAGR", f"{metrics['mean_cagr']:.2%}")
            col3.metric("Mediana CAGR", f"{metrics['median_cagr']:.2%}")
            col4.metric("Szansa Straty", f"{metrics['prob_loss']:.1%}")
            
            display_chart_guide("Kluczowe Wskaźniki (KPI)", """
            *   **Średni Kapitał**: Oczekiwana wartość końcowa (średnia arytmetyczna ze wszystkich symulacji).
            *   **CAGR**: Średnioroczna stopa zwrotu (procent składany).
            *   **Mediana CAGR**: Bardziej "realistyczny" zwrot (połowa scenariuszy jest lepsza, połowa gorsza).
            *   **Szansa Straty**: Prawdopodobieństwo, że po X latach będziesz miał mniej pieniędzy niż na początku.
            """)

            days = np.arange(wealth_paths.shape[1])
            percentiles = np.percentile(wealth_paths, [5, 50, 95], axis=0)
            
            fig_paths = go.Figure()
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[2], mode='lines', line=dict(width=0), showlegend=False))
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[0], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 136, 0.2)', name='95% CI'))
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[1], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'))
            fig_paths.update_layout(title="Projekcja Bogactwa", template="plotly_dark", height=500)
            st.plotly_chart(fig_paths, use_container_width=True)
            
            display_chart_guide("Projekcja Bogactwa (Fan Chart)", """
            *   **Ciemnozielona Linia (Mediana)**: Najbardziej prawdopodobna ścieżka Twojego portfela.
            *   **Obszar Cieniowany (90% CI)**: "Stożek niepewności". Z 90% prawdopodobieństwem Twój wynik zmieści się w tym tunelu.
            *   **Szerokość Tunelu**: Im szerszy, tym większa niepewność (ryzyko) strategii.
            """)

            # --- Professional Metrics Table ---
            with st.expander("📊 Tabela Profesjonalna (Risk & Performance)", expanded=True):
                # Organize in 3 categories
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.markdown("**Efektywność (Risk-Adjusted)**")
                    st.metric("Sharpe Ratio", f"{metrics['median_sharpe']:.2f}")
                    st.metric("Sortino Ratio", f"{metrics.get('median_sortino', 0):.2f}") # Placeholder
                    st.metric("Calmar Ratio", f"{metrics['median_calmar']:.2f}")

                with m_col2:
                    st.markdown("**Ryzyko (Risk Mgt)**")
                    st.metric("Max Drawdown (Avg)", f"{metrics['mean_max_drawdown']:.1%}")
                    st.metric("VaR 95% (Wynik)", f"{metrics['var_95']:,.0f} PLN")
                    st.metric("CVaR 95% (Krach)", f"{metrics['cvar_95']:,.0f} PLN", help="Średnia wartość kapitału w 5% najgorszych scenariuszy.")

                with m_col3:
                    st.markdown("**Statystyka**")
                    st.metric("Median Volatility", f"{metrics['median_volatility']:.1%}")
                    st.metric("Szansa Bankructwa", f"{metrics['prob_loss']:.1%}")
                    st.metric("Worst Case Drawdown", f"{metrics['worst_case_drawdown']:.1%}")
            
            display_chart_guide("Tabela Profesjonalna (Hedge Fund Grade)", """
            *   **Sharpe Ratio**: Zysk za każdą jednostkę ryzyka. > 1.0 = Dobrze, > 2.0 = Wybitnie.
            *   **Sortino Ratio**: Jak Sharpe, ale liczy tylko "złą" zmienność (spadki). Ważniejsze dla inwestora indywidualnego.
            *   **Calmar Ratio**: CAGR / Max Drawdown. Mówi, jak szybko strategia "odkopuje się" z dołka.
            *   **VaR 95%**: "Value at Risk". Kwota, której NIE stracisz z 95% pewnością. (Ale z 5% pewnością stracisz więcej!).
            *   **CVaR 95%**: "Expected Shortfall". Jeśli już nastąpi te 5% najgorszych dni (krach), tyle średnio stracisz. To jest prawdziwy wymiar ryzyka ogona.
            """)
            
            # --- New Visualization Section ---
            st.divider()
            st.subheader("📊 Zaawansowane Wizualizacje")
            
            # A. 3D Risk-Reward Cloud (Scatter)
            st.markdown("### ☁️ Chmura Ryzyka i Zysku (Hedge Fund View)")
            
            # Calculate metrics for every simulation
            sim_metrics_df = calculate_individual_metrics(wealth_paths, years)
            
            fig_cloud = px.scatter_3d(
                sim_metrics_df,
                x='MaxDrawdown',
                y='FinalWealth',
                z='Volatility',
                color='Sharpe',
                hover_data=['CAGR'],
                color_continuous_scale='RdYlGn',
                opacity=0.6,
                title="Każda kropka to inna symulowana przyszłość"
            )
            fig_cloud.update_layout(
                scene=dict(
                    xaxis_title='Max Drawdown (Ból)',
                    yaxis_title='Kapitał (Zysk)',
                    zaxis_title='Zmienność (Emocje)'
                ),
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_cloud, use_container_width=True)
            
            display_chart_guide("Chmura Ryzyka i Zysku", """
            *   **Cel**: Pokazuje relację między "Bólem" (Max Drawdown - oś X) a "Zyskiem" (Kapitał - oś Y).
            *   **Oś Z (Pionowa)**: Zmienność. Im wyżej, tym bardziej "szarpie" portfelem.
            *   **Kolor (Sharpe)**: Zielone kropki to "Dobre Ryzyko" (duży zysk przy małym ryzyku). Czerwone to "Złe Ryzyko".
            *   **Gdzie patrzeć?**: Szukamy skupisk kropek w **lewym, górnym rogu** (Mały Drawdown, Duży Zysk). Jeśli chmura jest płaska i szeroka, wynik jest loterią.
            """)

            st.divider()
            
            # B. Histogram of Final Wealth
            final_wealths = wealth_paths[:, -1]
            fig_hist = px.histogram(
                final_wealths, 
                nbins=50, 
                title="Rozkład Kapitału Końcowego",
                labels={'value': 'Kapitał (PLN)'},
                color_discrete_sequence=['#00ff88']
            )
            
            # Add VaR lines
            var_95 = np.percentile(final_wealths, 5)
            fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
            fig_hist.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            display_chart_guide("Histogram i VaR", """
            *   **VaR 95% (Value at Risk)**: Czerwona linia oznacza "Pesymistyczny Scenariusz". Z 95% pewnością Twój wynik będzie lepszy niż ta linia.
            *   **Gruby Ogon**: Jeśli histogram ma "długi ogon" w prawo, masz szansę na ogromne zyski (Black Swan).
            """)

            # C. 3D Sensitivity Analysis (On Demand)
            st.subheader("🧊 Mapa Wrażliwości 3D")
            st.caption("Sprawdź jak wynik zależy od Volatility (Ryzyka) i % Alokacji.")
            
            if st.button("Generuj Mapę 3D (Może potrwać chwilę)", key="mc_3d_btn"):
                st.session_state['mc_3d_data'] = None # Clear old
                
                with st.status("Symulowanie wariantów (Grid 10x10)...", expanded=True) as status:
                    # Define grid
                    vol_range = np.linspace(0.10, 0.80, 10) # 10 steps
                    alloc_range = np.linspace(0.10, 1.0, 10) # 10 steps
                    
                    z_data = []
                    
                    total_steps = len(vol_range) * len(alloc_range)
                    step_count = 0
                    
                    # Create a placeholder for progress
                    progress_text = st.empty()
                    
                    for v in vol_range:
                        row = []
                        for a in alloc_range:
                            step_count += 1
                            if step_count % 10 == 0:
                                progress_text.text(f"Symulacja: {step_count}/{total_steps}")
                                
                            w_paths = simulate_barbell_strategy(
                                n_years=years,
                                n_simulations=100, 
                                initial_captial=initial_capital,
                                safe_rate=safe_rate,
                                risky_mean=risky_mean,
                                risky_vol=v, 
                                risky_kurtosis=risky_kurtosis,
                                alloc_safe=1.0 - a, 
                                rebalance_strategy=rebalance_strategy.split(" ")[0],
                                threshold_percent=threshold_percent
                            )
                            metric = np.median(w_paths[:, -1])
                            row.append(metric)
                            
                        z_data.append(row)
                    
                    st.session_state['mc_3d_data'] = {
                        'z': z_data,
                        'x': alloc_range,
                        'y': vol_range
                    }
                    progress_text.empty()
                    status.update(label="Mapa wygenerowana!", state="complete", expanded=False)

            # Render if data exists
            if 'mc_3d_data' in st.session_state and st.session_state['mc_3d_data'] is not None:
                data_3d = st.session_state['mc_3d_data']
                fig_3d = go.Figure(data=[go.Surface(
                    z=data_3d['z'], 
                    x=data_3d['x'], 
                    y=data_3d['y']
                )])
                fig_3d.update_layout(
                    title="Mediana Kapitału Końcowego",
                    scene = dict(
                        xaxis_title='Alokacja w Ryzyko (%)',
                        yaxis_title='Zmienność (Vol)',
                        zaxis_title='Kapitał (PLN)'
                    ),
                    template="plotly_dark",
                    height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
             
            display_chart_guide("Mapa Wrażliwości 3D", """
            *   **Płaskowyż**: Szukamy "płaskiego szczytu" (stabilne zyski). Jeśli mapa przypomina "iglicę", strategia jest niestabilna.
            *   **Oś Alokacji**: Zobacz, przy jakim % wkładzie w ryzyko, zyski zaczynają spadać (nadmierne ryzyko niszczy portfel - Variance Drag).
            """)
            
            display_analysis_report()

    elif mode == "Intelligent Barbell (Backtest AI)":
        st.sidebar.markdown("### 1. Konfiguracja Podstawowa")
        initial_capital = st.sidebar.number_input("Kapitał Początkowy (USD)", value=100000, step=10000, key="ai_cap")
        start_date = st.sidebar.date_input("Data Początkowa", value=pd.to_datetime("2020-01-01"), key="ai_start")
    
        st.sidebar.markdown("### 2. Aktywa")
        
        # Safe Asset Selection
        safe_type = st.sidebar.radio("Rodzaj Bezpiecznego Aktywa", ["Tickers (Yahoo)", "Holistyczne Obligacje Skarbowe (TOS 5.51%)"], key="ai_safe_type")
        safe_tickers_str = ""
        safe_fixed_rate = 0.0551
        
        if safe_type == "Tickers (Yahoo)":
            safe_tickers_str = st.sidebar.text_area("Koszyk Bezpieczny (Safe)", "TLT, IEF, GLD", help="Obligacje, Złoto", key="ai_safe_tickers")
        else:
            st.sidebar.info("Generowanie syntetycznego aktywa o stałym wzroście 5.51% rocznie.")
            safe_fixed_rate = st.sidebar.number_input("Oprocentowanie Obligacji (%)", value=5.51, step=0.1, key="ai_safe_rate") / 100.0
    
        risky_asset_mode = st.sidebar.radio("Tryb Wyboru Aktywów Ryzykownych", ["Lista (Auto Wagi)", "Manualne Wagi"], key="ai_risky_mode")
        risky_tickers_str = "SPY, QQQ, NVDA, BTC-USD" # Default for logic
        risky_weights_manual = None
        
        if risky_asset_mode == "Lista (Auto Wagi)":
             risky_tickers_str = st.sidebar.text_area("Koszyk Ryzykowny (Risky)", "SPY, QQQ, NVDA, BTC-USD", help="Akcje, Krypto", key="ai_risky_tickers")
             # Logic uses this string later
        else:
            st.sidebar.markdown("**Manualne Wagi Aktywów**")
            # Initialize session state for table if needed, or just use default
            default_data = pd.DataFrame([
                {"Ticker": "SPY", "Waga (%)": 100.0}
            ])
            
            # Check for data transferred from Scanner
            if 'transfer_data' in st.session_state and not st.session_state['transfer_data'].empty:
                default_data = st.session_state['transfer_data']
                # Clean up session state to avoid persistence if not desired, 
                # or keep it. Let's keep it until user changes it.
            
            edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="ai_manual_table")
            
            # Validation
            total_weight = edited_df["Waga (%)"].sum()
            if abs(total_weight - 100.0) > 0.01:
                st.sidebar.error(f"Suma wag musi wynosić 100%! Obecnie: {total_weight:.1f}%")
            
            # Prepare data for simulation
            # Create a dictionary {Ticker: Fraction}
            # And also update risky_tickers_str just in case or use it to load data
            risky_weights_manual = {}
            valid_tickers = []
            for index, row in edited_df.iterrows():
                t = str(row["Ticker"]).strip().upper()
                try:
                    w = float(row["Waga (%)"]) / 100.0
                except:
                    w = 0.0
                    
                if t:
                    risky_weights_manual[t] = w
                    valid_tickers.append(t)
            
            risky_tickers_str = ", ".join(valid_tickers) # Mock string to reuse existing download logic
        
        st.sidebar.markdown("### 3. Strategia Alokacji")
        allocation_mode = st.sidebar.selectbox("Tryb Alokacji", ["AI Dynamic (Regime + RL)", "Manual Fixed", "Rolling Kelly"], key="ai_alloc_mode")
        
        alloc_safe_fixed = 0.85
        kelly_params = {}
        
        if allocation_mode == "Manual Fixed":
            alloc_safe_fixed = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, 85, key="ai_alloc_safe_slider") / 100.0
            
        elif allocation_mode == "Rolling Kelly":
            kelly_fraction = st.sidebar.slider("Ułamek Kelly'ego (Fraction)", 0.1, 1.5, 0.5, 0.1, key="ai_kelly_frac")
            kelly_shrinkage = st.sidebar.slider("Czynnik Kurczenia (Shrinkage)", 0.0, 0.9, 0.1, 0.1, key="ai_kelly_shrink")
            kelly_window = st.sidebar.slider("Okno Analizy (dni)", 30, 500, 252, 10, key="ai_kelly_win")
            kelly_params = {"fraction": kelly_fraction, "shrinkage": kelly_shrinkage, "window": kelly_window}
    
        st.sidebar.markdown("### 4. Zarządzanie")
        rebalance_strategy = st.sidebar.selectbox(
            "Strategia Rebalansowania",
            ["None (Buy & Hold)", "Yearly", "Monthly", "Threshold (Shannon's Demon)"],
            index=2, # Default Monthly
            key="ai_rebal"
        )
        
        threshold_percent = 0.0
        if rebalance_strategy == "Threshold (Shannon's Demon)":
            threshold_percent = st.sidebar.slider("Próg Rebalansowania (%)", 5, 50, 20, 5, key="ai_thresh") / 100.0
            
    
        
        st.title("🧠 Intelligent Barbell - AI Backtest")
        st.markdown("""
        **Moduły AI:**
        - **Observer (HMM)**: Wykrywa reżimy rynkowe (Risk-On / Risk-Off).
        - **Architect (HRP)**: Buduje zdywersyfikowany portfel wewnątrz koszyków.
        - **Trader (RL Agent)**: Dynamicznie zarządza lewarem (Kelly).
        """)
        
        if st.button("🧠 Uruchom AI Backtest", type="primary"):
            safe_tickers = []
            if safe_type == "Tickers (Yahoo)":
                 safe_tickers = [x.strip() for x in safe_tickers_str.split(",") if x.strip()]
            
            # Handle Risky Tickers
            if risky_asset_mode == "Manualne Wagi":
                # risky_tickers_str was constructed from valid keys in the loop above
                risky_tickers = [x.strip() for x in risky_tickers_str.split(",") if x.strip()]
                if not risky_tickers:
                     st.error("Błąd: Lista manualnych tickerów jest pusta! Dodaj przynajmniej jeden ticker w tabeli.")
                     st.stop()
            else:
                risky_tickers = [x.strip() for x in risky_tickers_str.split(",") if x.strip()]
                
            with st.container(): # Progress Container
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                     progress_bar.progress(pct)
                     status_text.markdown(f"**{msg}**")
                 
            # with st.spinner("Pobieranie danych i trenowanie modeli..."): # Removed spinner to rely on progress bar
            safe_data = pd.DataFrame()
            if safe_tickers:
                    safe_data = load_data(safe_tickers, start_date=start_date)
                    
            risky_data = load_data(risky_tickers, start_date=start_date)
            
            # Check if fetch success
            if risky_data.empty:
                st.error("Błąd: Brak danych dla ryzykownych aktywów.")
            else:
                # Prepare args
                safe_type_arg = "Ticker" if safe_type == "Tickers (Yahoo)" else "Fixed"
                rebalance_strat_arg = rebalance_strategy.split(" ")[0]
                
                results, weight_history, regimes = run_ai_backtest(
                    safe_data, 
                    risky_data, 
                    initial_capital=initial_capital,
                    safe_type=safe_type_arg,
                    safe_fixed_rate=safe_fixed_rate,
                    allocation_mode=allocation_mode,
                    alloc_safe_fixed=alloc_safe_fixed,
                    kelly_params=kelly_params,
                    rebalance_strategy=rebalance_strat_arg,
                    threshold_percent=threshold_percent,
                    progress_callback=update_progress,
                    risky_weights_dict=risky_weights_manual
                )
                
                progress_bar.empty()
                status_text.empty()
                
                # Metrics
                years = (results.index[-1] - results.index[0]).days / 365.25
                metrics = calculate_metrics(results['PortfolioValue'].values, years)
                
                # Display Results
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Kapitał Końcowy", f"${metrics['mean_final_wealth']:,.0f}")
                col2.metric("CAGR", f"{metrics['mean_cagr']:.2%}")
                col3.metric("Max Drawdown", f"{metrics['worst_case_drawdown']:.2%}")
                col4.metric("Regime Risk-Off", f"{np.mean(regimes):.1%} czasu")
                
                display_chart_guide("Wyniki Backtestu AI", """
                *   **Kapitał Końcowy**: Ile zarobiłeś na koniec testu.
                *   **Max Drawdown**: Najgłębszy spadek wartości portfela w historii.
                *   **Regime Risk-Off**: Jak często AI "bało się" rynku i uciekało do bezpiecznych aktywów (Obligacje/Gotówka).
                """)

                # --- Algo / Professional Metrics Table ---
                from modules.metrics import calculate_trade_stats

                # Calculate Trade Stats approximation
                trade_stats = calculate_trade_stats(results['PortfolioValue'])
                
                with st.expander("📊 Raport Funduszu (Algo Stats & Risk)", expanded=True):
                    a_col1, a_col2, a_col3 = st.columns(3)
                    with a_col1:
                        st.markdown("**Efektywność Algo**")
                        st.metric("Profit Factor", f"{trade_stats['profit_factor']:.2f}")
                        st.metric("Win Rate (Dni)", f"{trade_stats['win_rate']:.1%}")
                        st.metric("Risk/Reward", f"{trade_stats['risk_reward']:.2f}")

                    with a_col2:
                         st.markdown("**Risk-Adjusted**")
                         st.metric("Sharpe Ratio", f"{metrics['median_sharpe']:.2f}")
                         st.metric("Sortino Ratio", f"{metrics.get('median_sortino', 0):.2f}") 
                         st.metric("Calmar Ratio", f"{metrics['median_calmar']:.2f}")
                    
                    with a_col3:
                        st.markdown("**Ryzyko**")
                        st.metric("VaR 95%", f"{metrics['var_95']:,.0f} PLN")
                        st.metric("CVaR 95%", f"{metrics['cvar_95']:,.0f} PLN")
                        st.metric("Max Drawdown", f"{metrics['worst_case_drawdown']:.2%}")

                display_chart_guide("Tabela Algo & Risk", """
                *   **Profit Factor**: Suma zysków / Suma strat. > 1.5 oznacza solidną strategię. < 1.0 to strata.
                *   **Win Rate**: Procent dni zyskownych. Wysoki Win Rate nie gwarantuje sukcesu (można mieć 90% małych zysków i jedną stratę bankruta).
                *   **Risk/Reward**: Średni Zysk / Średnia Strata. Strategie Trend-Following często mają niski Win Rate, ale wysoki R/R (tnij straty, pozwól zyskom rosnąć).
                """)
                
                # Plot Portfolio vs Regime
                fig = go.Figure()
                
                # Add Portfolio Line
                fig.add_trace(go.Scatter(
                    x=results.index, 
                    y=results['PortfolioValue'], 
                    mode='lines', 
                    name='Intelligent Barbell',
                    line=dict(color='#00ff88', width=2)
                ))
                
                fig.update_layout(
                    title="Wyniki Backtestu AI",
                    xaxis_title="Data",
                    yaxis_title="Wartość Portfela",
                    template="plotly_dark",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                display_chart_guide("Wykres Kapitału (Equity Curve)", """
                *   **Cel**: Chcesz widzieć stabilny wzrost (nachylenie w górę) z jak najmniejszymi "zębami" (drawdowns).
                *   **Porównanie**: Jeśli linia jest gładsza niż "Kup i Trzymaj" na S&P 500, to strategia działa.
                *   **Zielona Linia**: Wartość Twojego portfela w czasie.
                """)
                
                # Plot Regimes
                st.subheader("🕵️ Detekcja Reżimów Rynkowych (HMM)")
                st.caption("Czerwony = Wysoka Zmienność (Trader ucieka do bezpiecznych aktywów), Zielony = Niska Zmienność (Trader atakuje).")
                
                st.caption("Czerwony = Wysoka Zmienność (Trader ucieka do bezpiecznych aktywów), Zielony = Niska Zmienność (Trader atakuje).")
                
                # Create Colored Segments (by plotting markers on top of a gray line)
                fig_regime = go.Figure()
                
                # 1. Base Line (Gray)
                fig_regime.add_trace(go.Scatter(
                    x=results.index,
                    y=risky_data.mean(axis=1),
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

                # 2. Colored Markers (Larger)
                regime_colors = np.where(regimes == 1, '#ff4444', '#00ff88') # Bright Red / Bright Green
                fig_regime.add_trace(go.Scatter(
                    x=results.index,
                    y=risky_data.mean(axis=1), # Proxy for market
                    mode='markers',
                    marker=dict(color=regime_colors, size=6, line=dict(width=1, color='black')), # Size 2 -> 6
                    name='Market Regime'
                ))
                
                fig_regime.update_layout(
                    title="Cykle Rynkowe (HMM)",
                    xaxis_title="Data",
                    yaxis_title="Średnia Cena Koszyka (Proxy)",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_regime, use_container_width=True)
                
                display_chart_guide("Detekcja Reżimów (HMM)", """
                *   **Kropki Zielone (Risk-On)**: AI uznaje rynek za bezpieczny (niska/średnia zmienność). Strategia agresywnie inwestuje w ryzykowne aktywa.
                *   **Kropki Czerwone (Risk-Off)**: AI wykrywa turbulencje (wysoka zmienność/krach). Strategia ucieka do bezpiecznej przystani (Obligacje).
                *   **Cel**: Unikanie czerwonych kropek w trakcie największych krachów (np. 2020, 2022).
                """)

                # --- New AI Visualizations ---
                st.divider()
                st.divider()
                st.subheader("🔮 Zaawansowana Analityka (Hedge Fund View)")
                
                # --- 1. Monthly Returns Heatmap ---
                st.markdown("### 🗓️ Mapa Zwrotów Miesięcznych (Monthly Heatmap)")
                
                # Calculate monthly returns
                res_monthly = results['PortfolioValue'].resample('M').last().pct_change()
                res_monthly_df = pd.DataFrame(res_monthly)
                res_monthly_df['Year'] = res_monthly_df.index.year
                res_monthly_df['Month'] = res_monthly_df.index.month_name()
                
                # Pivot
                heatmap_data = res_monthly_df.pivot(index='Year', columns='Month', values='PortfolioValue')
                # Sort months correctly
                months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                heatmap_data = heatmap_data.reindex(columns=months_order)
                
                # Calculate centered range for Heatmap
                max_val = heatmap_data.abs().max().max() if not heatmap_data.empty else 0.1
                
                fig_heat = px.imshow(
                    heatmap_data, 
                    text_auto=".1%", 
                    color_continuous_scale='RdYlGn',
                    range_color=[-max_val, max_val],
                    title="Miesięczne Stopy Zwrotu"
                )
                fig_heat.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_heat, use_container_width=True)
                
                display_chart_guide("Mapa Ciepła (Heatmap)", """
                *   **Cel**: Szybka ocena sezonowości i spójności wyników.
                *   **Kolory**: Czerwień to strata, Zieleń to zysk.
                *   **Co jest dobre?**: Dużo zieleni, brak długich "czerwonych pasów" (serii strat).
                """)

                col_viz_1, col_viz_2 = st.columns(2)
                
                with col_viz_1:
                    # 2. 3D Phase Space Trajectory
                    st.markdown("**Trajektoria Fazowa Portfela (Phase Space)**")
                    ret_roll = results['PortfolioValue'].pct_change().rolling(21).mean() * 252
                    vol_roll = results['PortfolioValue'].pct_change().rolling(21).std() * np.sqrt(252)
                    
                    fig_3d_phase = go.Figure(data=go.Scatter3d(
                        x=ret_roll,
                        y=vol_roll,
                        z=np.arange(len(results)),
                        mode='lines',
                        line=dict(
                            color=np.where(regimes==1, 1.0, 0.0), # Map to colorscale
                            colorscale='RdYlGn_r',
                            width=4
                        ),
                        name='Trajektoria'
                    ))
                    fig_3d_phase.update_layout(
                        scene=dict(
                            xaxis_title='Zwrot (Rolling)',
                            yaxis_title='Ryzyko (Vol)',
                            zaxis_title='Czas'
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_3d_phase, use_container_width=True)
                    
                    display_chart_guide("Trajektoria Fazowa", """
                    *   **Spirala**: Portfel "oddycha". Zwróć uwagę, czy w okresach wysokiego ryzyka (oś Y) zwroty (oś X) są dodatnie.
                    *   **Kolor**: Czerwony = Reżim Wysokiej Zmienności (Risk-Off). Zielony = Hossa.
                    """)

                with col_viz_2:
                    # 3. Rolling Sharpe Ratio
                    st.markdown("**Stabilność Wyników (Rolling Sharpe)**")
                    
                    # Rolling 6-month Sharpe
                    window = 126
                    rolling_sharpe = (ret_roll.rolling(window).mean() / vol_roll.rolling(window).mean()) * np.sqrt(252) # Approximation
                    
                    fig_sharpe = go.Figure()
                    fig_sharpe.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#00d4ff', width=2),
                        name='Rolling Sharpe'
                    ))
                    fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Dobre (1.0)")
                    fig_sharpe.add_hline(y=0.0, line_dash="dot", line_color="red", annotation_text="Krytyczne (0.0)")
                    
                    fig_sharpe.update_layout(
                        yaxis_title='Sharpe Ratio (6M)',
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)
                    
                    display_chart_guide("Rolling Sharpe", """
                    *   **Powyżej 1.0**: Strategia generuje zysk nieproporcjonalnie duży do ryzyka.
                    *   **Poniżej 0**: Portfel nie zarabia nawet na pokrycie ryzyka.
                    """)
                
                # 4. Underwater Plot dedicated
                st.markdown("### ⚓ Wykres Obsunięć (Underwater Plot)")
                wealth = results['PortfolioValue']
                peaks = wealth.cummax()
                drawdowns = (wealth - peaks) / peaks
                
                fig_underwater = go.Figure()
                fig_underwater.add_trace(go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#ff4444', width=1),
                    name='Drawdown'
                ))
                fig_underwater.update_layout(
                    yaxis_title='Obsunięcie (%)',
                    template="plotly_dark",
                    height=300,
                    yaxis=dict(tickformat=".1%")
                )
                st.plotly_chart(fig_underwater, use_container_width=True)

                display_chart_guide("Underwater Plot", """
                *   **Głębokość**: Jak mocno bolało (oś Y).
                *   **Szerokość**: Jak długo trwało odrabianie strat (oś X). Długie płaskie dna to "Zombie Markets".
                """)
                
                # 5. Volatility Cone (Future implementation requires distinct windows logic but we add Rolling Vol here)
                # Let's add Rolling Volatility vs Market Proxy if feasible, or just standalone Rolling Vol
                st.divider()
                
                # 6. Rolling Correlation Heatmap (If multiple risky assets)
                if len(risky_data.columns) > 1:
                    st.subheader("🔥 Mapa Korelacji (Rolling)")
                    
                    corr_matrix = risky_data.tail(60).corr()
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        color_continuous_scale='RdBu_r', 
                        zmin=-1, zmax=1,
                        title="Macierz Korelacji (Ostatnie 60 dni)"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    display_chart_guide("Korelacja", """
                    *   **Czerwień (Blisko 1.0)**: Aktywa chodzą razem. Niebezpieczne w krachu.
                    *   **Niebieski (Blisko -1.0)**: Aktywa chodzą przeciwnie. Idealne do hedgingu.
                    *   **Biel (0.0)**: Brak korelacji. Święty Graal dywersyfikacji.
                    """)
                
                # --- AI Insights Visualizations ---
                st.subheader("🧠 Analityka AI: Architect & Trader")
                
                # Process Weights Data
                weights_df = pd.DataFrame(weight_history, index=results.index)
                
                # Identify risky and safe columns present in weights_df
                present_risky = [c for c in risky_tickers if c in weights_df.columns]
                present_safe = [c for c in safe_tickers if c in weights_df.columns]
                if not present_safe and "FIXED_SAFE" in weights_df.columns:
                     present_safe = ["FIXED_SAFE"]
    
                # 1. TRADER: Risky Exposure over Time
                if present_risky:
                    risky_exposure = weights_df[present_risky].sum(axis=1)
                    
                    fig_trader = go.Figure()
                    fig_trader.add_trace(go.Scatter(
                        x=risky_exposure.index, 
                        y=risky_exposure, 
                        mode='lines', 
                        name='Ekspozycja Ryzykowna (Trader)',
                        fill='tozeroy',
                        line=dict(color='#ffaa00', width=2)
                    ))
                    fig_trader.update_layout(
                        title="🎮 Trader (RL Agent): Dynamiczne Zarządzanie Lewarem (Kelly)",
                        yaxis_title=" % Portfela w Ryzyku",
                        xaxis_title="Data",
                        template="plotly_dark",
                        height=400,
                        yaxis=dict(tickformat=".0%")
                    )
                    st.plotly_chart(fig_trader, use_container_width=True)
                    
                    display_chart_guide("Decyzje Tradera (Risk Exposure)", """
                    *   **Cel**: Zarządzanie wielkością pozycji (Bet Sizing) w oparciu o Kryterium Kelly'ego.
                    *   **Wykres Wysoko**: Trader jest pewny siebie i zwiększa ekspozycję na ryzyko (lewaruje wynik).
                    *   **Wykres Nisko (lub 0)**: Trader ucina ryzyko (de-lewarowanie) w obliczu zagrożenia. To mechanizm obronny.
                    """)
    
                # 2. ARCHITECT: Internal Composition of Risky Basket
                if present_risky:
                    # Normalize risky weights to sum to 100% relative to the risky basket only
                    risky_internal = weights_df[present_risky].div(weights_df[present_risky].sum(axis=1), axis=0).fillna(0)
                    
                    fig_architect = go.Figure()
                    for col in risky_internal.columns:
                        fig_architect.add_trace(go.Scatter(
                            x=risky_internal.index,
                            y=risky_internal[col],
                            mode='lines',
                            stackgroup='one', # Stacked Area
                            name=col
                        ))
                    fig_architect.update_layout(
                        title="🏗️ Architect (HRP): Dywersyfikacja Wewnątrz Koszyka Ryzykownego",
                        yaxis_title="Waga Wewnętrzna",
                        xaxis_title="Data",
                        template="plotly_dark",
                        height=400,
                         yaxis=dict(tickformat=".0%")
                    )
                    st.plotly_chart(fig_architect, use_container_width=True)
                    
                    display_chart_guide("Decyzje Architekta (HRP)", """
                    *   **Cel**: Minimalizacja ryzyka wewnątrz koszyka spekulacyjnego poprzez inteligentną dywersyfikację (Hierarchical Risk Parity).
                    *   **Kolorowe Pola**: Pokazują, ile % portfela spekulacyjnego jest w danym aktywie.
                    *   **Zmiany**: Jeśli jedno pole rośnie kosztem innych, Architekt wykrył, że to aktywo stało się bezpieczniejsze lub mniej skorelowane z resztą.
                    """)
    
                display_analysis_report()

elif module_selection == "🔍 Skaner Wypukłości (BCS)":
    st.header("🔍 Barbell Convexity Scanner (BCS)")
    st.markdown("""
    **Cel**: Znajdowanie aktywów o asymetrycznym profilu zysku (Wypukłych/Antykruchych) do ryzykownej części portfela.
    **Kryteria EVT**: Szukamy "Grubych Ogonów" (czyli szansy na ogromne wzrosty) przy zdefiniowanym ryzyku.
    """)
    
    # Scanner Inputs
    col_scan1, col_scan2 = st.columns([3, 1])
    with col_scan1:
        default_tickers = "TQQQ, SOXL, UPRO, TMF, SPY, QQQ, BTC-USD, ETH-USD, ARKK, UVXY, COIN, NVDA, TSLA, MSTR"
        scan_tickers_str = st.text_area("Lista do przeskanowania (Tickery oddzielone przecinkami)", default_tickers)
    with col_scan2:
        scan_years = st.number_input("Historia (Lat)", value=3, step=1)
        scan_btn = st.button("🔎 Skanuj Wypukłość", type="primary")
        
    if scan_btn:
        tickers = [x.strip().upper() for x in scan_tickers_str.split(",") if x.strip()]
        
        if not tickers:
            st.error("Podaj przynajmniej jeden ticker.")
        else:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=scan_years)
            
            with st.status(f"Analiza EVT dla {len(tickers)} aktywów...", expanded=True) as status:
                final_metrics = []
                
                # Use load_data
                data = load_data(tickers, start_date=start_date.strftime("%Y-%m-%d"))
                
                if data.empty:
                    st.error("Brak danych.")
                    status.update(label="Błąd pobierania danych", state="error")
                else:
                    # Creating progress
                    progress_scan = st.progress(0)
                    
                    for i, t in enumerate(tickers):
                        if t in data.columns:
                            series = data[t]
                        elif len(tickers) == 1 and isinstance(data, pd.DataFrame): 
                             series = data[t] if t in data.columns else data.iloc[:, 0]
                        else:
                            continue
                            
                        # Calculate Metrics
                        m = calculate_convecity_metrics(t, series)
                        if m:
                            m["Score"] = score_asset(m)
                            final_metrics.append(m)
                            
                        progress_scan.progress((i + 1) / len(tickers))
                        
                    progress_scan.empty()
                    status.update(label="Skanowanie zakończone!", state="complete", expanded=False)
                    
                    if not final_metrics:
                        st.warning("Nie udało się obliczyć metryk dla żadnego aktywa (zbyt krótka historia?).")
                    else:
                        df_res = pd.DataFrame(final_metrics)
                        
                        # Sort by Score descending
                        df_res = df_res.sort_values("Score", ascending=False)
                        st.session_state['scanner_results'] = df_res
                        st.session_state['scanner_data'] = data # store raw data for charts
                        
    # Display results if they exist in session state
    if 'scanner_results' in st.session_state:
        df_res = st.session_state['scanner_results']
        data = st.session_state.get('scanner_data', pd.DataFrame()) # Retrieve data for charts
                        
        # Formatting
        st.subheader("🏆 Wyniki Rankingu Antykruchości")
        
        # Apply coloring style
        def highlight_score(val):
            color = '#2ecc71' if val > 50 else '#e74c3c' if val < 0 else ''
            return f'color: {color}; font-weight: bold'
            
        # Add selection column
        df_display = df_res.copy()
        df_display.insert(0, "Wybierz", False)

        # Dynamic height calculation: (Rows + Header) * Height per row
        # Approx 35px per row + 38px header + buffer
        dynamic_height = (len(df_res) + 1) * 35 + 3

        edited_df_scan = st.data_editor(
            df_display.style.format({
                "Annual Return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Skewness": "{:.2f}",
                "Kurtosis": "{:.2f}",
                "Hill Alpha (Tail)": "{:.2f}",
                "Kelly Safe (50%)": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max Drawdown": "{:.1%}"
            }).applymap(highlight_score, subset=['Score']),
            use_container_width=True,
            height=dynamic_height,
            column_config={
                "Wybierz": st.column_config.CheckboxColumn(
                    "Wybierz",
                    help="Zaznacz, aby przenieść do Symulatora",
                    default=False,
                )
            },
            disabled=list(df_res.columns) # Disable editing for metrics, enable only for checkbox
        )
        
        # Selection Logic
        selected_rows = edited_df_scan[edited_df_scan["Wybierz"]]
        
        if not selected_rows.empty:
            if st.button(f"➡️ Przenieś zaznaczone ({len(selected_rows)}) do Symulatora", type="primary"):
                tickers_to_transfer = selected_rows["Ticker"].tolist()
                
                # Calculate Equal Weights
                weight = 100.0 / len(tickers_to_transfer)
                transfer_data = [{"Ticker": t, "Waga (%)": weight} for t in tickers_to_transfer]
                
                # Store in Session State
                st.session_state['transfer_data'] = pd.DataFrame(transfer_data)
                
                # Switch Tabs/Mode
                st.session_state["force_navigate"] = "📉 Symulator Portfela"
                st.session_state["sim_mode"] = "Intelligent Barbell (Backtest AI)"
                st.session_state["ai_risky_mode"] = "Manualne Wagi"
                
                st.rerun()

        # --- New Visualization: 3D Antifragile Scatter ---
        st.divider()
        st.subheader("🧊 Mapa Antykruchości 3D")
        st.caption("Szukamy aktywów w prawym górnym rogu (Wysoki Skew, Wysoka Kurtoza, Niskie Hill Alpha jako duży bąbel).")
        
        # Prepare Data for 3D Plot
        # X: Skewness, Y: Kurtosis, Z: Annual Return
        # Color: Score, Size: Inverse Hill Alpha (or just fixed if nan)
        
        plot_df = df_res.copy()
        # Handle NaNs for plot
        plot_df['Hill Alpha (Tail)'] = plot_df['Hill Alpha (Tail)'].fillna(4.0) 
        # Create Size dimension: Inverse related to Hill Alpha (Lower Alpha = Bigger Bubble)
        # Avoid division by zero close to 1
        plot_df['Size'] = 10 / np.log(plot_df['Hill Alpha (Tail)'] + 0.1)
        plot_df['Size'] = plot_df['Size'].clip(upper=30, lower=5)
        
        fig_3d_scan = px.scatter_3d(
            plot_df,
            x='Skewness',
            y='Kurtosis',
            z='Annual Return',
            color='Score',
            size='Size', # Dynamic size
            hover_name='Ticker',
            hover_data=['Hill Alpha (Tail)', 'Kelly Safe (50%)'],
            color_continuous_scale='Viridis',
            title='Przestrzeń Wypukłości (Convexity Space)'
        )
        fig_3d_scan.update_layout(
             scene=dict(
                xaxis_title='Skośność (Skew)',
                yaxis_title='Kurtoza (Kurt)',
                zaxis_title='Zwrot Roczny'
            ),
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig_3d_scan, use_container_width=True)
        
        display_chart_guide("Mapa Antykruchości 3D", """
        *   **Szukaj Baniek**: Szukamy aktywów w prawym górnym rogu (Wysoka Skośność, Wysoka Kurtoza).
        *   **Rozmiar Bańki**: Większa bańka = Bardziej "Gruby Ogon" (Mniejsze Hill Alpha). To są potencjalne "rakiety".
        """)
        
        st.markdown("""
        ### 📖 Legenda Metryk (Słownik)
        
        *   **Annual Return**: Średni roczny zwrot geometryczny.
        *   **Volatility (Zmienność)**: Zmienność roczna. W strategii sztangi traktujemy ją jako **zasób**.
        *   **Skewness (Skośność)**: Mierzy asymetrię. >0 to nasz cel (częste małe straty, rzadkie wielkie zyski).
        *   **Kurtosis (Kurtoza)**: Mierzy "grubość" ogonów. Im wyższa, tym więcej ekstremalnych zdarzeń.
        *   **Hill Alpha**: Kluczowa metryka EVT. < 3.0 oznacza Gruby Ogon (szansa na wykładniczy wzrost).
        *   **Sharpe Ratio**: Wynik > 1.0 jest dobry. Mierzy zysk na jednostkę całkowitego ryzyka (zmienności).
        *   **Sortino Ratio**: Lepsza wersja Sharpe'a. Mierzy zysk na jednostkę "złej zmienności" (tylko spadki).
        *   **Max Drawdown**: Maksymalne obsunięcie kapitału. Mówi o tym, jak bardzo zaboli w najgorszym momencie.
        """)
        
        # Best Asset Charts
        best_asset = df_res.iloc[0]
        st.subheader(f"💎 Najlepsze Aktywo: {best_asset['Ticker']}")
        
        col_chart1, col_chart2 = st.columns(2)
        
        # Returns Histogram
        asset_data = data[best_asset['Ticker']].pct_change().dropna()
        fig_hist = px.histogram(asset_data, nbins=100, title=f"Rozkład Zwrotów {best_asset['Ticker']}")
        col_chart1.plotly_chart(fig_hist, use_container_width=True)
        
        # Cumulative Return (Log scale)
        cum_ret = (1 + asset_data).cumprod()
        fig_line = px.line(cum_ret, log_y=True, title=f"Wzrost Kapitału (Skala Log) {best_asset['Ticker']}")
        col_chart2.plotly_chart(fig_line, use_container_width=True)
        
        display_chart_guide("Wykresy Najlepszego Aktywa", """
        *   **Histogram**: Szukamy prawego "długiego ogona" (wiele słupków po prawej stronie zera).
        *   **Skala Logarytmiczna**: Linia prosta oznacza stabilne tempo wzrostu procentowego ($CAGR$).
        """)

        # 2a. Correlation Heatmap (Scanner)
        if len(data.columns) > 1:
            st.divider()
            st.subheader("🔥 Mapa Korelacji (Skaner)")
            st.caption("Sprawdź, czy wybrane aktywa są ze sobą powiązane.")
            
            # Calculate correlation matrix
            corr_matrix_scan = data.pct_change().corr()
            
            fig_corr_scan = px.imshow(
                corr_matrix_scan, 
                text_auto=".2f", 
                color_continuous_scale='RdBu_r', 
                zmin=-1, zmax=1,
                title="Macierz Korelacji"
            )
            fig_corr_scan.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_corr_scan, use_container_width=True)
            
            display_chart_guide("Mapa Korelacji (Skaner)", """
            *   **Cel**: Budowa portfela wymaga niskiej korelacji. Jeśli wszystko jest czerwone (kor > 0.8), dywersyfikacja nie działa.
            *   **Szukaj Błękitu**: Wartości bliskie 0 lub ujemne to idealni kandydaci do pary w strategii Barbell.
            """)

        # 3. Log-Log Tail Plot (Power Law Visualizer)
        st.markdown("**Analiza Ogonów (Log-Log Plot)**")
        st.caption("Jeśli linia jest prosta (opada liniowo na skali log-log), mamy do czynienia z Rozkładem Potęgowym (Power Law) i Grubymi Ogonami. Krzywa opadająca szybko (jak parabola) to rozkład Normalny (Gaussa).")
        
        # Calculate Tail Survival Function
        # We look at right tail (positive returns)
        pos_rets = asset_data[asset_data > 0].sort_values(ascending=False)
        if len(pos_rets) > 10:
            rank = np.arange(1, len(pos_rets) + 1)
            prob = rank / len(pos_rets)
            
            fig_loglog = go.Figure()
            fig_loglog.add_trace(go.Scatter(
                x=pos_rets,
                y=prob,
                mode='markers',
                name='Empiryczne Dane',
                marker=dict(color='#00ff88', size=5)
            ))
            fig_loglog.update_layout(
                title=f"Ogon Prawy: {best_asset['Ticker']} (Log-Log)",
                xaxis_title="Zwrot Dzienny (Log)",
                yaxis_title="P(X > x) (Log)",
                xaxis_type="log",
                yaxis_type="log",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_loglog, use_container_width=True)
            
            display_chart_guide("Log-Log Tail Plot", """
            *   **Test Potęgowy**: To najważniejszy test "Antykruchości".
            *   **Linia Prosta**: Oznacza, że ryzyko nie maleje wykładniczo. Krachy i Rakiety są bardziej prawdopodobne niż sądzisz (Mandlebrot/Taleb).
            *   **Parabola w dół**: Oznacza "Bezpieczny" rozkład normalny (mało niespodzianek).
            """)
        else:
            st.info("Za mało danych do wygenerowania wykresu Log-Log.")

    st.markdown("---")
    display_scanner_methodology()




