
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling
from modules.simulation import simulate_barbell_strategy, calculate_metrics, run_ai_backtest
from modules.ai.data_loader import load_data
from modules.analysis_content import display_analysis_report, display_scanner_methodology
from modules.scanner import calculate_convecity_metrics, score_asset

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

            days = np.arange(wealth_paths.shape[1])
            percentiles = np.percentile(wealth_paths, [5, 50, 95], axis=0)
            
            fig_paths = go.Figure()
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[2], mode='lines', line=dict(width=0), showlegend=False))
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[0], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 136, 0.2)', name='95% CI'))
            fig_paths.add_trace(go.Scatter(x=days, y=percentiles[1], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'))
            fig_paths.update_layout(title="Projekcja Bogactwa", template="plotly_dark", height=500)
            st.plotly_chart(fig_paths, use_container_width=True)
            
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
                
                # Plot Regimes
                st.subheader("🕵️ Detekcja Reżimów Rynkowych (HMM)")
                st.caption("Czerwony = Wysoka Zmienność (Trader ucieka do bezpiecznych aktywów), Zielony = Niska Zmienność (Trader atakuje).")
                
                regime_colors = np.where(regimes == 1, 'red', 'green')
                fig_regime = go.Figure()
                fig_regime.add_trace(go.Scatter(
                    x=results.index,
                    y=risky_data.mean(axis=1), # Proxy for market
                    mode='markers',
                    marker=dict(color=regime_colors, size=2),
                    name='Market Regime'
                ))
                st.plotly_chart(fig_regime, use_container_width=True)
                
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
            
            with st.spinner(f"Analiza EVT dla {len(tickers)} aktywów..."):
                final_metrics = []
                
                # Use load_data but maybe iteratively or batch? Load data handles batches.
                data = load_data(tickers, start_date=start_date.strftime("%Y-%m-%d"))
                
                if data.empty:
                    st.error("Brak danych.")
                else:
                    # Creating progress
                    progress_scan = st.progress(0)
                    
                    for i, t in enumerate(tickers):
                        if t in data.columns:
                            series = data[t]
                        elif len(tickers) == 1 and isinstance(data, pd.DataFrame): # Single ticker case handled in load_data usually returns df with ticker col
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
                "Kelly Safe (50%)": "{:.1%}"
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
        
        st.markdown("""
        ### 📖 Legenda Metryk (Słownik)

        *   **Annual Return**: Średni roczny zwrot geometryczny.
        *   **Volatility (Zmienność)**: Zmienność roczna. W strategii sztangi traktujemy ją jako **zasób** (paliwo dla Demona Shannona), a nie tylko jako ryzyko. Wysoka zmienność przy braku korelacji umożliwia generowanie "premii z rebalansowania".
        *   **Skewness (Skośność)**: Mierzy asymetrię rozkładu zwrotów.
            *   **> 0 (Pozytywna)**: Rozkład ma "długi prawy ogon". Oznacza częste małe straty i rzadkie, ale ogromne zyski (Profil Antykruchy). **To jest nasz cel.**
            *   **< 0 (Negatywna)**: Rozkład ma "długi lewy ogon". Oznacza częste małe zyski i rzadkie katastrofalne straty (Profil Kruchy - np. sprzedaż opcji). Unikaj tego.
        *   **Kurtosis (Kurtoza)**: Mierzy "grubość" ogonów. Wysoka kurtoza oznacza, że ekstremalne zdarzenia (krachy lub rakiety) zdarzają się częściej niż przewiduje rozkład normalny Gaussa.
        *   **Hill Alpha (Indeks Ogonowy)**: Kluczowa metryka EVT (Extreme Value Theory).
            *   **< 3.0**: Gruby ogon (Fat Tail). Aktywo ma potencjał do wykładniczych wzrostów.
            *   **< 2.0**: Ekstremalna wypukłość (Infinite Variance). Najbardziej pożądane aktywa w części ryzykownej (np. Krypto, Opcje).
            *   **> 4.0**: Rozkład zbliżony do normalnego (Brak potencjału Black Swan).
        *   **Kelly Safe (50%)**: Sugerowana wielkość alokacji kapitału w dane aktywo wg kryterium Kelly'ego, zredukowana o 50% (tzw. Half-Kelly) dla bezpieczeństwa. Uwzględnia relację zysku do ryzyka. Ujemna wartość oznacza, że nie należy inwestować.
        *   **Score**: Syntetyczna ocena algorytmu, który promuje aktywa o niskim Hill Alpha i wysokiej dodatniej skośności.
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

    st.markdown("---")
    display_scanner_methodology()




