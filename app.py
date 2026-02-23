
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling
from modules.simulation import simulate_barbell_strategy, calculate_metrics, run_ai_backtest, calculate_individual_metrics
from modules.metrics import (
    calculate_trade_stats, calculate_omega, calculate_ulcer_index,
    calculate_pain_index, calculate_drawdown_analytics, calculate_max_drawdown
)
from modules.ai.data_loader import load_data
from modules.analysis_content import display_analysis_report, display_scanner_methodology, display_chart_guide
from modules.scanner import calculate_convecity_metrics, score_asset, compute_hierarchical_dendrogram
from modules.ai.scanner_engine import ScannerEngine
from modules.ai.asset_universe import get_sp500_tickers, get_global_etfs
from modules.ui.status_manager import StatusManager
from modules.stress_test import run_stress_test, CRISIS_SCENARIOS
from modules.frontier import compute_efficient_frontier
from modules.emerytura import render_emerytura_module
from modules.ai.observer import REGIME_BULL, REGIME_BEAR, REGIME_CRISIS

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

# ---------------------------------------------------------------------------
# Persystencja ustawień między modułami
# Streamlit USUWA klucze widżetów z session_state gdy widżet nie jest renderowany.
# Rozwiązanie: on_change zapisuje do "_s.<key>", a value= czyta z tego klucza.
# ---------------------------------------------------------------------------
def _save(wk):
    """Callback on_change: kopiuje wartość widżetu do trwałego klucza."""
    st.session_state[f"_s.{wk}"] = st.session_state[wk]

def _saved(wk, default):
    """Zwraca ostatnio zapisaną wartość lub domyślną."""
    return st.session_state.get(f"_s.{wk}", default)

# Klucze pomocnicze (nie-widżetowe) dla modułu Emerytura
if "rem_initial_capital" not in st.session_state:
    st.session_state["rem_initial_capital"] = 1000000.0
if "rem_expected_return" not in st.session_state:
    st.session_state["rem_expected_return"] = 0.07
if "rem_volatility" not in st.session_state:
    st.session_state["rem_volatility"] = 0.15


# Navigation State Handler
if "force_navigate" in st.session_state:
    st.session_state["module_nav"] = st.session_state.pop("force_navigate")

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

# 3. Main Navigation
module_selection = st.radio("Wybierz Moduł:", ["📉 Symulator Portfela", "🔍 Skaner Wypukłości (BCS)", "⚡ Stress Test", "🏖️ Emerytura"], horizontal=True, label_visibility="collapsed", key="module_nav")
st.markdown("---")

if module_selection == "📉 Symulator Portfela":
    st.sidebar.title("🛠️ Konfiguracja Strategii")
    
    mode = st.sidebar.radio("Tryb Symulacji", ["Monte Carlo (Teoretyczny)", "Intelligent Barbell (Backtest AI)"], index=["Monte Carlo (Teoretyczny)", "Intelligent Barbell (Backtest AI)"].index(_saved("sim_mode", "Monte Carlo (Teoretyczny)")), key="sim_mode", on_change=_save, args=("sim_mode",))

    if mode == "Monte Carlo (Teoretyczny)":
        st.sidebar.markdown("### 1. Kapitał i Czas")
        initial_capital = st.sidebar.number_input("Kapitał Początkowy (PLN)", value=_saved("mc_cap", 100000), step=10000, key="mc_cap", on_change=_save, args=("mc_cap",))
        years = st.sidebar.slider("Horyzont Inwestycyjny (Lata)", 1, 30, value=_saved("mc_years", 10), key="mc_years", on_change=_save, args=("mc_years",))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 2. Część Bezpieczna (Safe Sleeve)")
        st.sidebar.info("🔒 Obligacje Skarbowe RP 3-letnie (Stałe 5.51%)")
        safe_rate = 0.0551 

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 3. Część Ryzykowna (Risky Sleeve)")
        risky_mean = st.sidebar.slider("Oczekiwany Zwrot Roczny (Średnia)", -0.20, 0.50, value=_saved("mc_risky_mean", 0.08), step=0.01, key="mc_risky_mean", on_change=_save, args=("mc_risky_mean",))
        risky_vol = st.sidebar.slider("Zmienność Roczna (Volatility)", 0.10, 1.50, value=_saved("mc_risky_vol", 0.50), step=0.05, key="mc_risky_vol", on_change=_save, args=("mc_risky_vol",))
        risky_kurtosis = st.sidebar.slider("Grubość Ogonów (Kurtosis)", 2.1, 30.0, value=_saved("mc_risky_kurtosis", 4.0), step=0.1, key="mc_risky_kurtosis", on_change=_save, args=("mc_risky_kurtosis",))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 4. Optymalizacja Kelly'ego")
        use_kelly = st.sidebar.checkbox("Użyj Kryterium Kelly'ego", value=_saved("mc_kelly", False), key="mc_kelly", on_change=_save, args=("mc_kelly",))
        
        kelly_fraction = 1.0
        kelly_shrinkage = 0.0
        
        if use_kelly:
            kelly_fraction = st.sidebar.slider("Ułamek Kelly'ego (Fraction)", 0.1, 1.0, value=_saved("mc_kelly_frac", 0.25), step=0.05, key="mc_kelly_frac", on_change=_save, args=("mc_kelly_frac",))
            kelly_shrinkage = st.sidebar.slider("Czynnik Kurczenia (Shrinkage)", 0.0, 0.9, value=_saved("mc_kelly_shrink", 0.1), step=0.05, key="mc_kelly_shrink", on_change=_save, args=("mc_kelly_shrink",))
            
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
            alloc_safe = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, value=_saved("mc_alloc_safe", 85), key="mc_alloc_safe", on_change=_save, args=("mc_alloc_safe",)) / 100.0

        _mc_rebal_opts = ["None (Buy & Hold)", "Yearly", "Monthly", "Threshold (Shannon's Demon)"]
        rebalance_strategy = st.sidebar.selectbox(
            "Strategia Rebalansowania",
            _mc_rebal_opts,
            index=_mc_rebal_opts.index(_saved("mc_rebalance", "None (Buy & Hold)")),
            key="mc_rebalance",
            on_change=_save, args=("mc_rebalance",)
        )
        
        threshold_percent = 0.0
        if rebalance_strategy == "Threshold (Shannon's Demon)":
            threshold_percent = st.sidebar.slider("Próg Rebalansowania (%)", 5, 50, value=_saved("mc_threshold", 20), step=5, key="mc_threshold", on_change=_save, args=("mc_threshold",)) / 100.0

        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Zaawansowane (Naukowe)")
        use_qmc = st.sidebar.checkbox(
            "Użyj Quasi-Monte Carlo (Sobol)",
            value=_saved("mc_use_qmc", False), key="mc_use_qmc",
            on_change=_save, args=("mc_use_qmc",),
            help="Sekwencje Sobola dają 10x szybszą zbieżność niż losowanie pseudolosowe. Joe & Kuo (2010)."
        )
        use_garch = st.sidebar.checkbox(
            "Symuluj GARCH(1,1) Zmienność",
            value=_saved("mc_use_garch", False), key="mc_use_garch",
            on_change=_save, args=("mc_use_garch",),
            help="Modeluje klastrowanie zmienności (volatility clustering). Bollerslev (1986). Wolniejsze, ale bardziej realistyczne."
        )
        use_jump_diffusion = st.sidebar.checkbox(
            "Merton Jump-Diffusion (Skoki Cen)",
            value=_saved("mc_use_jump", True), key="mc_use_jump",
            on_change=_save, args=("mc_use_jump",),
            help="Symuluje nagłe luki cenowe (Czarne Łabędzie) poprzez proces Poissona. Merton (1976)."
        )


        # MAIN CONTENT FOR MONTE CARLO
        st.title("⚖️ Barbell Strategy - Monte Carlo")
        
        if st.button("🚀 Symuluj Wyniki", type="primary", key="mc_run"):
            status_mc = StatusManager("Symulacja Monte Carlo...")
            
            status_mc.info_math(f"Generowanie {1000} ścieżek dla horyzontu {years} lat...")
            
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
                threshold_percent=threshold_percent,
                use_qmc=use_qmc,
                use_garch=use_garch,
                use_jump_diffusion=use_jump_diffusion
            )
            
            status_mc.info_math("Obliczanie zaawansowanych metryk (Sharpe, VaR, CVaR)...")
            metrics = calculate_metrics(wealth_paths, years)

            status_mc.success("Symulacja zakończona!")
            
            # Save to session state
            st.session_state['mc_results'] = {
                "wealth_paths": wealth_paths,
                "metrics": metrics,
                "years": years
            }
            
        # Check if results exist and display
        if 'mc_results' in st.session_state:
            res = st.session_state['mc_results']
            wealth_paths = res['wealth_paths']
            metrics = res['metrics']
            years = res['years']


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
            
            # ─── Ridge Plot / Joyplot of Wealth Distribution ───────────────────
            st.markdown("### ⛰️ Ewoluujący Rozkład Majątku (Ridge Plot)")
            st.caption("Płynna zmiana rozkładu prawdopodobieństwa w czasie. Ukazuje asymetrię zysków i ryzyko ogona.")
            
            fig_ridge = go.Figure()
            
            # Select 5 key milestones to plot distributions for (e.g. year 1, quarter-way, half-way, 3-quarter, final)
            milestones = np.linspace(252, wealth_paths.shape[1]-1, min(5, years)).astype(int)
            colors = ['#00ff88', '#00ccff', '#ffaa00', '#ff4444', '#aa88ff']
            
            for i, day_idx in enumerate(milestones[::-1]): # Reverse to draw back-to-front
                year_mark = int(day_idx / 252)
                data_slice = wealth_paths[:, day_idx]
                
                # Use violin plot with horizontal orientation for ridge effect
                fig_ridge.add_trace(go.Violin(
                    x=data_slice,
                    name=f"Rok {year_mark}",
                    side='positive',
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)].replace(')', ', 0.3)').replace('rgb', 'rgba') if 'rgb' in colors[i] else 'rgba(0, 255, 136, 0.4)',
                    meanline_visible=True
                ))
                
            fig_ridge.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Kapitał (PLN)",
                yaxis_title="Oś Czasu (Horyzont)",
                violinmode='overlay',
                violingap=0,
                violingroupgap=0,
                showlegend=False
            )
            fig_ridge.update_traces(orientation='h', width=2.5, points=False)
            fig_ridge.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig_ridge.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig_ridge.add_vline(x=initial_capital, line_dash="dash", line_color="orange", annotation_text="Start")
            st.plotly_chart(fig_ridge, use_container_width=True)
            
            display_chart_guide("Ridge Plot (Ewoluujący Kapitał)", """
            *   Zamiast płaskich linii, widzisz **pełny rozkład prawdopodobieństwa kapitału** w kluczowych latach symulacji.
            *   **Asymetria**: Zauważ, jak wraz z upływem czasu rozkład staje się prawoskośny (długi ogon bogactwa) dzięki procentowi składanemu, podczas gdy lewa strona (straty) jest węższa, obrazując asymetrię ryzyka/zysku Barbell'a.
            """)

            # --- Professional Metrics Table ---
            with st.expander("📊 Tabela Profesjonalna (Risk & Performance)", expanded=True):
                # Organize in 3 categories
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.markdown("**Efektywność (Risk-Adjusted)**")
                    st.metric("Sharpe Ratio", f"{metrics['median_sharpe']:.2f}")
                    st.metric("Sortino Ratio", f"{metrics.get('median_sortino', 0):.2f}")
                    st.metric("Calmar Ratio", f"{metrics['median_calmar']:.2f}")

                with m_col2:
                    st.markdown("**Ryzyko (Risk Mgt)**")
                    st.metric("Max Drawdown (Avg)", f"{metrics['mean_max_drawdown']:.1%}")
                    st.metric("VaR 95% (Wynik)", f"{metrics['var_95']:,.0f} PLN")
                    st.metric("CVaR 95% (Krach)", f"{metrics['cvar_95']:,.0f} PLN", help="Średnia wartość kapitału w 5% najgorszych scenariuszy.")

                with m_col3:
                    st.markdown("**Statystyka + NOWE 🆕**")
                    st.metric("Median Volatility", f"{metrics['median_volatility']:.1%}")
                    # Omega Ratio (new)
                    final_returns = np.diff(wealth_paths[:, :], axis=1).flatten() / wealth_paths[:, :-1].flatten()
                    omega = calculate_omega(final_returns)
                    st.metric("Omega Ratio 🆕", f"{omega:.2f}", help="Omega > 1 = więcej zysku niż straty. Idealny wskaźnik dla strategii Barbell (Shadwick & Keating 2002).")
                    # Ulcer Index (new)
                    median_path = np.median(wealth_paths, axis=0)
                    ulcer = calculate_ulcer_index(median_path)
                    st.metric("Ulcer Index 🆕", f"{ulcer:.2f}", help="Mierzy 'ból inwestora' przez cały okres. Niższy = lepszy (Martin & McCann 1989).")

            
            display_chart_guide("Tabela Profesjonalna (Hedge Fund Grade)", """
            *   **Sharpe Ratio**: Zysk za każdą jednostkę ryzyka. > 1.0 = Dobrze, > 2.0 = Wybitnie.
            *   **Sortino Ratio**: Jak Sharpe, ale liczy tylko "złą" zmienność (spadki). Ważniejsze dla inwestora indywidualnego.
            *   **Calmar Ratio**: CAGR / Max Drawdown. Mówi, jak szybko strategia "odkopuje się" z dołka.
            *   **VaR 95%**: "Value at Risk". Kwota, której NIE stracisz z 95% pewnością. (Ale z 5% pewnością stracisz więcej!).
            *   **CVaR 95%**: "Expected Shortfall". Jeśli już nastąpi te 5% najgorszych dni (krach), tyle średnio stracisz. To jest prawdziwy wymiar ryzyka ogona.
            """)
            
            # --- TRANSFER BUTTON (MC) ---
            if st.button("⚡ Przenieś 'Worst Case' do Stress Testów", key="mc_to_stress"):
                # Find the 5th percentile path (the one representing VaR 95%)
                final_wealths = wealth_paths[:, -1]
                # Index of the path closest to the 5% percentile
                target_val = np.percentile(final_wealths, 5)
                idx_worst = np.abs(final_wealths - target_val).argmin()
                worst_path = wealth_paths[idx_worst]
                
                # Create DataFrame with fake dates for Stress Test
                dates = pd.date_range(start=pd.Timestamp.now(), periods=len(worst_path), freq='D')
                df_custom = pd.DataFrame({
                    "Portfolio (Barbell)": worst_path,
                    "Benchmark": np.full(len(worst_path), initial_capital) # Mock benchmark for MC
                }, index=dates)
                
                st.session_state["custom_stress_scenarios"]["🔥 MC: Worst Case (5%)"] = {
                    "df": df_custom,
                    "initial_capital": initial_capital,
                    "description": f"Symulowany najgorszy scenariusz (5. percentyl) z Monte Carlo ({years} lat)."
                }
                st.session_state["force_navigate"] = "⚡ Stress Test"
                st.rerun()
            
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
        initial_capital = st.sidebar.number_input("Kapitał Początkowy (USD)", value=_saved("ai_cap", 100000), step=10000, key="ai_cap", on_change=_save, args=("ai_cap",))
        start_date = st.sidebar.date_input("Data Początkowa", value=_saved("ai_start", pd.to_datetime("2020-01-01")), key="ai_start", on_change=_save, args=("ai_start",))
    
        st.sidebar.markdown("### 2. Aktywa")
        
        # Safe Asset Selection
        _ai_safe_opts = ["Tickers (Yahoo)", "Holistyczne Obligacje Skarbowe (TOS 5.51%)"]
        safe_type = st.sidebar.radio("Rodzaj Bezpiecznego Aktywa", _ai_safe_opts, index=_ai_safe_opts.index(_saved("ai_safe_type", "Holistyczne Obligacje Skarbowe (TOS 5.51%)")), key="ai_safe_type", on_change=_save, args=("ai_safe_type",))
        safe_tickers_str = ""
        safe_fixed_rate = 0.0551
        
        if safe_type == "Tickers (Yahoo)":
            safe_tickers_str = st.sidebar.text_area("Koszyk Bezpieczny (Safe)", value=_saved("ai_safe_tickers", "TLT, IEF, GLD"), help="Obligacje, Złoto", key="ai_safe_tickers", on_change=_save, args=("ai_safe_tickers",))
        else:
            st.sidebar.info("Generowanie syntetycznego aktywa o stałym wzroście 5.51% rocznie.")
            safe_fixed_rate = st.sidebar.number_input("Oprocentowanie Obligacji (%)", value=_saved("ai_safe_rate", 5.51), step=0.1, key="ai_safe_rate", on_change=_save, args=("ai_safe_rate",)) / 100.0
    
        _ai_risky_opts = ["Lista (Auto Wagi)", "Manualne Wagi"]
        risky_asset_mode = st.sidebar.radio("Tryb Wyboru Aktywów Ryzykownych", _ai_risky_opts, index=_ai_risky_opts.index(_saved("ai_risky_mode", "Lista (Auto Wagi)")), key="ai_risky_mode", on_change=_save, args=("ai_risky_mode",))
        risky_tickers_str = "SPY, QQQ, NVDA, BTC-USD" # Default for logic
        risky_weights_manual = None
        
        if risky_asset_mode == "Lista (Auto Wagi)":
             risky_tickers_str = st.sidebar.text_area("Koszyk Ryzykowny (Risky)", value=_saved("ai_risky_tickers", "SPY, QQQ, NVDA, BTC-USD"), help="Akcje, Krypto", key="ai_risky_tickers", on_change=_save, args=("ai_risky_tickers",))
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
        _ai_alloc_opts = ["AI Dynamic (Regime + RL)", "Manual Fixed", "Rolling Kelly"]
        allocation_mode = st.sidebar.selectbox("Tryb Alokacji", _ai_alloc_opts, index=_ai_alloc_opts.index(_saved("ai_alloc_mode", "AI Dynamic (Regime + RL)")), key="ai_alloc_mode", on_change=_save, args=("ai_alloc_mode",))
        
        alloc_safe_fixed = 0.85
        kelly_params = {}
        
        if allocation_mode == "Manual Fixed":
            alloc_safe_fixed = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, value=_saved("ai_alloc_safe_slider", 85), key="ai_alloc_safe_slider", on_change=_save, args=("ai_alloc_safe_slider",)) / 100.0
            
        elif allocation_mode == "Rolling Kelly":
            kelly_fraction = st.sidebar.slider("Ułamek Kelly'ego (Fraction)", 0.1, 1.5, value=_saved("ai_kelly_frac", 0.5), step=0.1, key="ai_kelly_frac", on_change=_save, args=("ai_kelly_frac",))
            kelly_shrinkage = st.sidebar.slider("Czynnik Kurczenia (Shrinkage)", 0.0, 0.9, value=_saved("ai_kelly_shrink", 0.1), step=0.1, key="ai_kelly_shrink", on_change=_save, args=("ai_kelly_shrink",))
            kelly_window = st.sidebar.slider("Okno Analizy (dni)", 30, 500, value=_saved("ai_kelly_win", 252), step=10, key="ai_kelly_win", on_change=_save, args=("ai_kelly_win",))
            kelly_params = {"fraction": kelly_fraction, "shrinkage": kelly_shrinkage, "window": kelly_window}
    
        st.sidebar.markdown("### 4. Zarządzanie")
        _ai_rebal_opts = ["None (Buy & Hold)", "Yearly", "Monthly", "Threshold (Shannon's Demon)"]
        rebalance_strategy = st.sidebar.selectbox(
            "Strategia Rebalansowania",
            _ai_rebal_opts,
            index=_ai_rebal_opts.index(_saved("ai_rebal", "Monthly")),
            key="ai_rebal",
            on_change=_save, args=("ai_rebal",)
        )
        
        threshold_percent = 0.0
        if rebalance_strategy == "Threshold (Shannon's Demon)":
            threshold_percent = st.sidebar.slider("Próg Rebalansowania (%)", 5, 50, value=_saved("ai_thresh", 20), step=5, key="ai_thresh", on_change=_save, args=("ai_thresh",)) / 100.0
            
    
        
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
                     # Strip percentage from message to act on user feedback
                     clean_msg = msg.split("(")[0].strip() if "(" in msg else msg
                     status_text.markdown(f"**{clean_msg}**")
                 
            # with st.spinner("Pobieranie danych i trenowanie modeli..."): # Removed spinner to rely on progress bar
            status_ai = StatusManager("Przygotowanie Backtestu AI...", expanded=True)
            
            status_ai.info_data("Pobieranie danych historycznych...")
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
                
                status_ai.info_ai("Obliczanie Reżimów Rynkowych i Symulacja Tradera RL...")
                
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
                
                status_ai.info_math("Finalizacja metryk...")
                
                # Metrics
                years = (results.index[-1] - results.index[0]).days / 365.25
                metrics = calculate_metrics(results['PortfolioValue'].values, years)
                
                # Calculate Trade Stats approximation
                trade_stats = calculate_trade_stats(results['PortfolioValue'])

                status_ai.success("Backtest zakończony sukcesem!")
                
                # Save results and Rerun
                st.session_state['backtest_results'] = {
                    "results": results,
                    "metrics": metrics,
                    "trade_stats": trade_stats,
                    "risky_mean": risky_data.mean(axis=1),
                    "regimes": regimes
                }
                # ★ Save raw price data for Efficient Frontier (persists across reruns)
                st.session_state['backtest_safe_data']  = safe_data
                st.session_state['backtest_risky_data'] = risky_data
                st.rerun()

        # RENDER RESULTS (Only if state exists)
        if 'backtest_results' in st.session_state:
            # Just display what is in state.
            res = st.session_state['backtest_results']
            metrics = res['metrics']
            
            # --- 1. TABLES SECTION (Top) ---
            st.subheader("📊 Wyniki Strategii (Scorecard)")
            
            # KPI Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Kapitał Końcowy", f"${metrics['mean_final_wealth']:,.0f}")
            col2.metric("CAGR", f"{metrics['mean_cagr']:.2%}")
            col3.metric("Max Drawdown", f"{metrics['mean_max_drawdown']:.2%}")
            
            # Regime Info
            regime_disp = "N/A"
            if 'results' in res:
                df_res = res['results']
                if 'Regime' in df_res.columns:
                    risk_off_cnt = df_res['Regime'].str.contains("Risk-Off").sum()
                    regime_disp = f"{risk_off_cnt / len(df_res):.1%} czasu"
            
            col4.metric("Regime Risk-Off", regime_disp)
            
            st.divider()
            
            # Professional Metrics Table
            trade_stats = res['trade_stats']
            
            possible_sortino = metrics.get('median_sortino', 0)
            if possible_sortino is None: possible_sortino = 0

            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.markdown("**Efektywność (Return)**")
                st.metric("Total Return", f"{(metrics['mean_final_wealth'] - initial_capital)/initial_capital:.1%}")
                st.metric("CAGR", f"{metrics['mean_cagr']:.2%}")
                st.metric("Profit Factor", f"{trade_stats.get('profit_factor', 0):.2f}")

            with m_col2:
                st.markdown("**Ryzyko (Risk)**")
                st.metric("Max Drawdown", f"{metrics['mean_max_drawdown']:.2%}")
                st.metric("Volatility (Ann.)", f"{metrics['median_volatility']:.1%}")
                st.metric("CVaR 95% (Tail Risk)", f"{metrics['cvar_95']:,.0f}")
                
            with m_col3:
                st.markdown("**Jakość (Ratios)**")
                st.metric("Sharpe Ratio", f"{metrics['median_sharpe']:.2f}")
                st.metric("Sortino Ratio", f"{possible_sortino:.2f}")
                st.metric("Risk/Reward", f"{trade_stats.get('risk_reward', 0):.2f}")
                
            # --- TRANSFER BUTTON (AI) ---
            if st.button("⚡ Przenieś Backtest AI do Stress Testów", key="ai_to_stress"):
                df_results = res['results']
                df_custom = pd.DataFrame({
                    "Portfolio (Barbell)": df_results["PortfolioValue"],
                    "Benchmark": df_results.get("Benchmark", df_results["PortfolioValue"])
                }, index=df_results.index)
                
                st.session_state["custom_stress_scenarios"]["🧠 AI: Backtest Strategy"] = {
                    "df": df_custom,
                    "initial_capital": initial_capital,
                    "description": f"Pełna historia portfela wypracowana przez AI od {df_custom.index.min().date()}."
                }
                st.session_state["force_navigate"] = "⚡ Stress Test"
                st.rerun()

            st.divider()

            # --- 2. CHARTS SECTION (Bottom) ---
            st.subheader("📈 Wykresy Analityczne")
            
            # A. Equity Curve
            st.markdown("#### 1. Krzywa Kapitału (Equity Curve)")
            
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=res['results'].index, y=res['results']['PortfolioValue'], mode='lines', name='Smart Barbell', line=dict(color='#00ff88', width=2)))
            
            risky_series = res.get('risky_mean', pd.Series())
            regimes_arr = res.get('regimes', [])

            fig.update_layout(title="Wzrost Wartości Portfela", template="plotly_dark", height=500, hovermode="x unified")
            fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            st.plotly_chart(fig, use_container_width=True, key="chart_equity_main")
            
            display_chart_guide("Wykres Kapitału", """
            *   **Cel**: Chcesz widzieć stabilny wzrost (nachylenie w górę).
            *   **Stabilność**: Im mniej poszarpana linia, tym lepiej śpisz.
            """)
            
            st.divider()
            
            # B. Underwater Plot
            st.markdown("#### 2. Obsunięcia Kapitału (Drawdowns)")
            wealth = res['results']['PortfolioValue']
            peaks = wealth.cummax()
            drawdowns = (wealth - peaks) / peaks
            
            fig_underwater = go.Figure()
            fig_underwater.add_trace(go.Scattergl(
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
                yaxis=dict(tickformat=".1%"),
                hovermode="x unified"
            )
            fig_underwater.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig_underwater.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            st.plotly_chart(fig_underwater, use_container_width=True, key="chart_underwater_plot")
            
            display_chart_guide("Underwater Plot", """
            *   **Interpretacja**: Pokazuje ile % tracisz względem "szczytu" portfela.
            *   **Cel**: Jak najpłytsze (krótkie słupki) i jak najwęższe (szybki powrót) "dołki".
            """)
            
            st.divider()

            # C. Regime Plot
            st.markdown("#### 3. Detekcja Reżimów (AI Context)")
            
            fig_regime = go.Figure()
            
            # Base Line (Gray)
            if not risky_series.empty:
                fig_regime.add_trace(go.Scattergl(
                    x=res['results'].index,
                    y=risky_series,
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

                # Colored Markers
            # Colored Markers — now 3-color for Bull/Bear/Crisis
            if len(regimes_arr) > 0:
                # Map regime integers to colors
                if hasattr(res.get('observer'), 'high_vol_state') and res.get('observer'):
                    obs = res['observer']
                    crisis_idx = obs.high_vol_state
                    low_idx = obs.low_vol_state
                    mid_idx = getattr(obs, 'mid_vol_state', -1)
                    def _regime_color(r):
                        if r == crisis_idx: return '#ff2222'
                        if r == mid_idx: return '#ffaa00'
                        return '#00ff88'
                    regime_colors = [_regime_color(r) for r in regimes_arr]
                    regime_label_map = {crisis_idx: 'Crisis 🔴', mid_idx: 'Bear 🟠', low_idx: 'Bull 🟢'}
                else:
                    # Fallback: 2-state
                    regime_colors = ['#ff4444' if r == 1 else '#00ff88' for r in regimes_arr]
                    regime_label_map = {1: 'Risk-Off 🔴', 0: 'Risk-On 🟢'}
                fig_regime.add_trace(go.Scattergl(
                    x=res['results'].index,
                    y=risky_series,
                    mode='markers',
                    marker=dict(color=regime_colors, size=4, opacity=0.6),
                    name='Regime State'
                ))

            
            fig_regime.update_layout(title="Reżimy Rynkowe (HMM) na tle rynku", template="plotly_dark", height=400, hovermode="closest")
            fig_regime.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig_regime.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            st.plotly_chart(fig_regime, use_container_width=True, key="chart_regime_dots")
            
            display_chart_guide("Detekcja Reżimów (HMM)", """
            *   **Kropki Zielone (Risk-On)**: AI uznaje rynek za bezpieczny.
            *   **Kropki Czerwone (Risk-Off)**: AI wykrywa turbulencje.
            """)
            
            # --- Advanced Charts (Restored) ---
            st.subheader("🔮 Zaawansowana Analityka (Hedge Fund View)")
            
            # 1. Monthly Returns Heatmap
            st.markdown("### 🗓️ Mapa Zwrotów Miesięcznych (Monthly Heatmap)")
            
            res_monthly = res['results']['PortfolioValue'].resample('M').last().pct_change()
            res_monthly_df = pd.DataFrame(res_monthly)
            res_monthly_df['Year'] = res_monthly_df.index.year
            res_monthly_df['Month'] = res_monthly_df.index.month_name()
            
            heatmap_data = res_monthly_df.pivot(index='Year', columns='Month', values='PortfolioValue')
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            heatmap_data = heatmap_data.reindex(columns=months_order)
            
            max_val = heatmap_data.abs().max().max() if not heatmap_data.empty else 0.1
            
            fig_heat = px.imshow(
                heatmap_data, 
                text_auto=".1%", 
                color_continuous_scale='RdYlGn',
                range_color=[-max_val, max_val],
                title="Miesięczne Stopy Zwrotu"
            )
            fig_heat.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_heat, use_container_width=True, key="chart_heatmap_monthly")
            
            display_chart_guide("Mapa Ciepła (Heatmap)", """
            *   **Cel**: Szybka ocena sezonowości i spójności wyników.
            *   **Kolory**: Czerwień to strata, Zieleń to zysk.
            """)
            
            # 2. Phase Space & Sharpe
            col_viz_1, col_viz_2 = st.columns(2)
            
            with col_viz_1:
                st.markdown("**Trajektoria Fazowa Portfela (Phase Space)**")
                results_pv = res['results']['PortfolioValue']
                ret_roll = results_pv.pct_change().rolling(21).mean() * 252
                vol_roll = results_pv.pct_change().rolling(21).std() * np.sqrt(252)
                
                # Using known regimes length
                regimes_to_plot = regimes_arr if len(regimes_arr) == len(results_pv) else np.zeros(len(results_pv))

                fig_3d_phase = go.Figure(data=go.Scatter3d(
                    x=ret_roll,
                    y=vol_roll,
                    z=np.arange(len(results_pv)),
                    mode='lines',
                    line=dict(
                        color=np.where(regimes_to_plot==1, 1.0, 0.0), 
                        colorscale='RdYlGn_r',
                        width=4
                    ),
                    name='Trajektoria'
                ))
                fig_3d_phase.update_layout(
                    scene=dict(xaxis_title='Zwrot', yaxis_title='Ryzyko', zaxis_title='Czas'),
                    margin=dict(l=0, r=0, b=0, t=0),
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_3d_phase, use_container_width=True, key="chart_phase_space")
                
                display_chart_guide("Trajektoria Fazowa", """
                *   **Spirala**: Portfel "oddycha". Zwróć uwagę, czy w okresach wysokiego ryzyka (oś Y) zwroty (oś X) są dodatnie.
                *   **Kolor**: Czerwony = Reżim Wysokiej Zmienności (Risk-Off). Zielony = Hossa.
                """)

            with col_viz_2:
                st.markdown("**Stabilność Wyników (Rolling Sharpe)**")
                window = 126
                # Avoid division by zero
                vol_safe = vol_roll.replace(0, 0.001)
                rolling_sharpe = (ret_roll.rolling(window).mean() / vol_safe.rolling(window).mean()) * np.sqrt(252)
                
                fig_sharpe = go.Figure()
                fig_sharpe.add_trace(go.Scattergl(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#00d4ff', width=2),
                    name='Rolling Sharpe'
                ))
                fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Dobre (1.0)")
                
                fig_sharpe.update_layout(
                    yaxis_title='Sharpe Ratio (6M)',
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_sharpe, use_container_width=True, key="chart_rolling_sharpe")
                
                display_chart_guide("Rolling Sharpe", """
                *   **Powyżej 1.0**: Strategia generuje zysk nieproporcjonalnie duży do ryzyka.
                *   **Poniżej 0**: Portfel nie zarabia nawet na pokrycie ryzyka.
                """)
                
            st.divider()
            




            
             


            
            st.divider()
            
            # Methodology Report
            display_analysis_report()

            # ─────────────────────────────────────────────────────────────────
            # 🆕 DRAWDOWN ANALYTICS (Naukowe)
            # ─────────────────────────────────────────────────────────────────
            st.divider()
            st.subheader("🩺 Zaawansowana Analiza Obsunięć 🆕")
            st.caption("Reference: Magdon-Ismail & Atiya (2004), Chekhlov et al. (2005), Martin & McCann (1989)")

            pv_series = res['results']['PortfolioValue']
            dd_analytics = calculate_drawdown_analytics(pv_series.values)

            dd_col1, dd_col2, dd_col3, dd_col4 = st.columns(4)
            dd_col1.metric("Max Drawdown", f"{dd_analytics['max_drawdown']:.1%}")
            dd_col2.metric("Ulcer Index", f"{dd_analytics['ulcer_index']:.2f}",
                          help="sqrt(mean(DD²)). Mierzy ból i długość spadków. Niższy = lepszy.")
            dd_col3.metric("Pain Index", f"{dd_analytics['pain_index']:.2%}",
                          help="Średnia głębokość obsunięcia przez cały okres.")
            dd_col4.metric("DD-at-Risk 95%", f"{dd_analytics['drawdown_at_risk_95']:.1%}",
                          help="Analogia CVaR dla drawdownów. Najgorsze 5% obsunięć.")

            dd_col5, dd_col6 = st.columns(2)
            dd_col5.metric("Śr. Czas Obsunięcia", f"{dd_analytics['avg_drawdown_duration_days']:.0f} sesji",
                          help="Średnia liczba dni trwania jednego obsunięcia.")
            dd_col6.metric("Max Czas Obsunięcia", f"{dd_analytics['max_drawdown_duration_days']} sesji",
                          help="Najdłuższe obsunięcie (ile dni od szczytu do powrotu).")

            # Omega Ratio
            port_returns_pct = pv_series.pct_change().dropna().values
            omega_val = calculate_omega(port_returns_pct)
            st.metric("Omega Ratio 🆕", f"{min(omega_val, 99):.2f}",
                     help="Omega > 1 = portfel generuje więcej zysku niż straty. Idealny dla Barbell.")

            # ─────────────────────────────────────────────────────────────────
            # 🆕 EFFICIENT FRONTIER
            # ─────────────────────────────────────────────────────────────────
            st.divider()
            st.subheader("📐 Granica Efektywna — Gdzie jest Twój Barbell? 🆕")
            st.caption("Reference: Markowitz (1952), Shadwick & Keating (2002)")

            with st.expander("🔍 Generuj Granicę Efektywną (może chwilę potrwać)", expanded=False):
                if st.button("📐 Oblicz Granicę Efektywną", key="frontier_btn"):
                    try:
                        # Load from session_state (saved when backtest ran)
                        _safe_data  = st.session_state.get('backtest_safe_data',  pd.DataFrame())
                        _risky_data = st.session_state.get('backtest_risky_data', pd.DataFrame())

                        if _risky_data.empty:
                            st.warning("⚠️ Uruchom najpierw AI Backtest żeby załadować dane aktywów.")
                        else:
                            # Build returns DataFrame — safe may be empty (TOS fixed rate mode)
                            if not _safe_data.empty:
                                ef_prices = pd.concat([_safe_data, _risky_data], axis=1)
                            else:
                                ef_prices = _risky_data.copy()

                            # Forward-fill then drop rows still all-NaN
                            ef_prices = ef_prices.ffill().dropna(how="all")
                            ef_returns = ef_prices.pct_change().dropna(how="all")

                            # Remove columns with too many NaNs
                            ef_returns = ef_returns.dropna(axis=1, thresh=int(len(ef_returns) * 0.8))

                            n_cols = len(ef_returns.columns)
                            st.caption(f"📊 Obliczam granicę dla {n_cols} aktywów: {', '.join(ef_returns.columns.tolist())}")

                            if n_cols < 2:
                                st.warning(f"⚠️ Za mało aktywów ({n_cols}). Potrzeba ≥2. Dodaj więcej tickerów w sekcji Koszyk Ryzykowny.")
                            else:
                                frontier_result = compute_efficient_frontier(
                                    ef_returns,
                                    n_portfolios=2000,
                                    risk_free_rate=0.04,
                                )
                                if frontier_result.get("error"):
                                    st.error(frontier_result["error"])
                                else:
                                    st.session_state["frontier_fig"] = frontier_result["fig"]
                                    st.session_state["frontier_data"] = {
                                        "max_sharpe": frontier_result["max_sharpe"],
                                        "min_vol":    frontier_result["min_vol"],
                                        "max_omega":  frontier_result["max_omega"],
                                    }
                    except Exception as e:
                        st.error(f"Błąd obliczenia granicy: {e}")


                if "frontier_fig" in st.session_state:
                    st.plotly_chart(st.session_state["frontier_fig"], use_container_width=True)
                    fd = st.session_state["frontier_data"]
                    ef_c1, ef_c2, ef_c3 = st.columns(3)
                    ef_c1.metric("⭐ Max Sharpe", f"{fd['max_sharpe']['sharpe']:.2f}",
                                 f"Return: {fd['max_sharpe']['return']:.1%}")
                    ef_c2.metric("🔵 Min Vol", f"{fd['min_vol']['volatility']:.1%}",
                                 f"Return: {fd['min_vol']['return']:.1%}")
                    ef_c3.metric("🟢 Max Omega", f"{fd['max_omega']['omega']:.2f}",
                                 f"Return: {fd['max_omega']['return']:.1%}")


elif module_selection == "🔍 Skaner Wypukłości (BCS)":
    st.header("🔍 Barbell Convexity Scanner (BCS)")
    st.markdown("""
    **Cel**: Znajdowanie aktywów o asymetrycznym profilu zysku (Wypukłych/Antykruchych) do ryzykownej części portfela.
    **Kryteria EVT**: Szukamy "Grubych Ogonów" (czyli szansy na ogromne wzrosty) przy zdefiniowanym ryzyku.
    """)
    
    # Scanner Inputs
    col_scan1, col_scan2 = st.columns([3, 1])
    
    with col_scan1:
        scan_mode = st.radio("Tryb Skanowania", ["Manualny (Lista)", "Auto-Select (Math Score)"], horizontal=True)
        
        scan_tickers_str = ""
        max_ai_tickers = 10
        
        if scan_mode == "Manualny (Lista)":
            default_tickers = "TQQQ, SOXL, UPRO, TMF, SPY, QQQ, BTC-USD, ETH-USD, ARKK, UVXY, COIN, NVDA, TSLA, MSTR"
            scan_tickers_str = st.text_area("Lista do przeskanowania (Tickery oddzielone przecinkami)", default_tickers)
        else:
            st.info("🤖 System przeszuka S&P 500 oraz wybrane Europejskie ETFy i wybierze najlepsze aktywa na podstawie wyniku matematycznego (Score).")
            
            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                max_ai_tickers = st.slider("Ile tickerów wybrać?", 3, 20, 10)
            with col_ai2:
                # Spacer
                st.write("")

    
    with col_scan2:
        scan_years = st.number_input("Historia (Lat)", value=3, step=1)
        st.markdown("###") # spacer
        scan_btn = st.button("🔎 Skanuj Wypukłość", type="primary")
        
    if scan_btn:
        final_tickers = []
        
        # 1. Determine Tickers Universe
        if scan_mode == "Manualny (Lista)":
            final_tickers = [x.strip().upper() for x in scan_tickers_str.split(",") if x.strip()]
        else:
            with st.spinner("Pobieranie listy aktywów (S&P 500 + Top 50 Global ETF)..."):
                sp500 = get_sp500_tickers()
                etfs = get_global_etfs() 
                final_tickers = sp500 + etfs
                st.toast(f"Znaleziono {len(final_tickers)} aktywów do analizy.")
        
        # Create Source Map
        source_map = {}
        if scan_mode == "Manualny (Lista)":
            for t in final_tickers:
                source_map[t] = "Manualne"
        else:
            # Re-fetch lists to map correctly
            s_sp500 = get_sp500_tickers()
            s_etfs = get_global_etfs()
            for t in s_sp500:
                source_map[t] = "S&P 500"
            for t in s_etfs:
                source_map[t] = "Top 50 Global"
        
        if not final_tickers:
            st.error("Podaj przynajmniej jeden ticker.")
        else:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=scan_years)
            
            # 2. Run Scan Engine
            engine = ScannerEngine()
            
            # Use StatusManager instead of st.status context
            status_scan = StatusManager(f"Analiza EVT dla {len(final_tickers)} aktywów...", expanded=True)
            status_scan.info_math(f"Obliczanie metryk (Hill Alpha, Skewness, Kelly) dla {len(final_tickers)} rynków...")
                
            # Progress bar inside (optional, handled by update?)
            # Manager logic doesn't expose inner container easily, so we just use spinner style
            # OR pass a callback if we refactor Engine. 
            # Engine takes progress bar object.
            # Let's use a placeholder progress bar or just rely on text updates if engine allows.
            # Engine expects a streamlit progress object.
            
            progress_scan = st.progress(0)
            
            # Let's adapt engine call
            df_candidates = engine.scan_markets(final_tickers, progress_bar=progress_scan)
            
            # Add Source Column
            if not df_candidates.empty:
                df_candidates["Source"] = df_candidates["Ticker"].map(source_map).fillna("Unknown")
            
            if df_candidates.empty:
                st.error("Nie udało się obliczyć metryk (brak danych lub błędne tickery).")
                status_scan.error("Błąd podczas analizy.")
            else:
                progress_scan.progress(1.0, "Analiza matematyczna zakończona.")
                
                # 3. Auto Selection (Math Based)
                if scan_mode == "Auto-Select (Math Score)":

                    status_scan.update(label="🤖 Auto-Selekcja (Math Score)...", state="running")
                    
                    # Filter top 50 first by Math Score
                    top_candidates = df_candidates.sort_values("Score", ascending=False)
                    
                    status_scan.info_ai(f"Wybieranie {max_ai_tickers} najlepszych aktywów wg wyniku...")
                    selected_tickers = engine.select_best_candidates(top_candidates, max_count=max_ai_tickers)
                    
                    status_scan.info_ai("Selekcja zakończona.")
                    
                    # Filter results to show only selected
                    df_res = df_candidates[df_candidates['Ticker'].isin(selected_tickers)]
                    
                    df_res = df_res.sort_values("Score", ascending=False)
                    st.toast(f"Wybrano {len(df_res)} najlepszych kandydatów!")
                    
                else:
                    df_res = df_candidates.sort_values("Score", ascending=False)

                st.session_state['scanner_results'] = df_res
                
                # Fetch full data for charts
                final_result_tickers = df_res['Ticker'].tolist()
                status_scan.info_data("Pobieranie pełnej historii cen dla wykresów...")
                chart_data = load_data(final_result_tickers, start_date=start_date.strftime("%Y-%m-%d"))
                st.session_state['scanner_data'] = chart_data
                
                status_scan.success("Skanowanie zakończone!")
                progress_scan.empty()
                st.rerun()

                        
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
                "EVT Shape (Tail)": "{:.2f}",
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
        plot_df['EVT Shape (Tail)'] = plot_df['EVT Shape (Tail)'].fillna(0.0) 
        # Create Size dimension: Higher EVT Shape = Bigger Bubble (Fatter tail)
        plot_df['Size'] = (plot_df['EVT Shape (Tail)'] * 50).clip(upper=30, lower=5)
        
        fig_3d_scan = px.scatter_3d(
            plot_df,
            x='Skewness',
            y='Kurtosis',
            z='Annual Return',
            color='Score',
            size='Size', # Dynamic size
            hover_name='Ticker',
            hover_data=['EVT Shape (Tail)', 'Kelly Safe (50%)'],
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
        *   **Rozmiar Bańki**: Większa bańka = Bardzo "Gruby Ogon" Mierzony Teorią EVT (Wyższe EVT Shape). To są potencjalne "rakiety".
        """)
        
        st.markdown("""
        ### 📖 Legenda Metryk (Słownik)
        
        *   **Annual Return**: Średni roczny zwrot geometryczny.
        *   **Volatility (Zmienność)**: Zmienność roczna. W strategii sztangi traktujemy ją jako **zasób**.
        *   **Skewness (Skośność)**: Mierzy asymetrię. >0 to nasz cel (częste małe straty, rzadkie wielkie zyski).
        *   **Kurtosis (Kurtoza)**: Mierzy "grubość" ogonów. Im wyższa, tym więcej ekstremalnych zdarzeń.
        *   **EVT Shape (Tail)**: Kluczowa metryka Teorii EVT (Peaks Over Threshold). Im wyższa, tym większa szansa na wybitne zdarzenia (Black Swans).
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
            fig_loglog.add_trace(go.Scattergl(
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

    # ─────────────────────────────────────────────────────────────────
    # 🆕 HIERARCHICAL DENDROGRAM (Zamiast płaskiego MST)
    # ─────────────────────────────────────────────────────────────────
    if 'bcs_returns' in st.session_state:
        st.divider()
        st.subheader("🌳 Dendrogram Klastrowy (Hierarchical Risk Parity) 🆕")
        st.caption("Maszyny uczące budują zagnieżdżoną strukturę ryzyka. Szukaj aktywów, które odłączają się najwcześniej na dole (najdłuższe pionowe gałęzie) — to prawdziwe, nieskorelowane dywersyfikatory wg Teorematu Lopeza de Prado.")
        mst_fig = compute_hierarchical_dendrogram(st.session_state['bcs_returns'])
        
        if mst_fig:
            st.plotly_chart(mst_fig, use_container_width=True)
            st.caption("🟢 Odkryj prawdziwą architekturę rynku: Drzewo pokazuje które rynki zachowują się jak jedno aktywo, a które faktycznie różnią się od reszty koszyka.")



# ─────────────────────────────────────────────────────────────────
# ⚡ STRESS TEST TAB
# ─────────────────────────────────────────────────────────────────
elif module_selection == "⚡ Stress Test":
    st.title("⚡ Stress Testing — Historyczne Kryzysy")
    st.markdown("""
    Testuje jak Twoja strategia Barbell zachowałaby się w 5 historycznych kryzysach.
    Wczytuje prawdziwe dane historyczne z Yahoo Finance i porównuje z benchmarkiem (SPY/QQQ).
    """)

    st.sidebar.title("⚡ Konfiguracja Stress Testu")
    st.sidebar.markdown("### Aktywa do Testu")

    from modules.stress_test import run_stress_test, CRISIS_SCENARIOS, run_reverse_stress_test
    
    st_safe_str  = st.sidebar.text_input("Koszyk Bezpieczny", "TLT, GLD", key="st_safe")
    st_risky_str = st.sidebar.text_input("Koszyk Ryzykowny", "SPY, QQQ, BTC-USD", key="st_risky")
    st_safe_w    = st.sidebar.slider("Waga Bezpieczna (%)", 10, 95, 85, key="st_sw") / 100.0
    st_capital   = st.sidebar.number_input("Kapitał Początkowy", value=100000, step=10000, key="st_cap")

    # ─────────────────────────────────────────────────────────────────
    # 🆕 REVERSE STRESS TESTING
    # ─────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### 🧨 Reverse Stress Test")
    st.sidebar.caption("Odwrócony test stresu (Standard Basel III). Podaj cel spadku portfela, a system wyliczy jak duży musi być krach rynkowy, by to mencapai.")
    rst_target_loss = st.sidebar.slider("Zakładana strata portfela (-%)", 5, 80, 20, step=5) / 100.0
    
    if st.sidebar.button("🧨 Szukaj Punktu Pęknięcia"):
        st.subheader("🧨 Reverse Stress Test (Punkt Pęknięcia)")
        rst_res = run_reverse_stress_test(safe_weight=st_safe_w, target_loss=rst_target_loss)
        
        if rst_res.get("error"):
            st.error(rst_res["error"])
        elif rst_res["is_possible"]:
            st.warning(rst_res["message"])
            col_rst1, col_rst2 = st.columns(2)
            col_rst1.metric("Szok Bezpieczny (Założenie)", f"{rst_res['safe_shock']:.1%}")
            col_rst2.metric("Wymagany Krach Ryzykowny", f"{rst_res['risky_shock']:.1%}", delta="Punkt krytyczny", delta_color="inverse")
            st.caption(f"Przy wadze bezpiecznej {st_safe_w:.0%} i ryzykownej {1-st_safe_w:.0%}, portfel straci {rst_target_loss:.0%} tylko wtedy, gdy ryzykowna część załamie się o {abs(rst_res['risky_shock']):.1%}.")
        else:
            st.success(rst_res["message"])
            st.metric("Ostrzeżenie", f"{rst_res['max_loss']:.1%}", "Maksymalna teoretyczna strata portfela")

    st.sidebar.divider()
    crisis_options = list(CRISIS_SCENARIOS.keys())
    custom_options = list(st.session_state["custom_stress_scenarios"].keys())
    
    selected_crises = st.multiselect(
        "Wybierz Scenariusze Kryzysu",
        crisis_options + custom_options,
        default=crisis_options[:3],
        key="st_crises"
    )

    if st.button("🚀 Uruchom Stress Test", type="primary", key="st_run"):
        st_safe_tickers  = [x.strip() for x in st_safe_str.split(",")  if x.strip()]
        st_risky_tickers = [x.strip() for x in st_risky_str.split(",") if x.strip()]

        st_results = {}
        with st.spinner("Pobieranie danych historycznych i symulacja..."):
            # 1. Historical Scenarios
            for crisis in selected_crises:
                if crisis in CRISIS_SCENARIOS:
                    result = run_stress_test(
                        safe_tickers=st_safe_tickers,
                        risky_tickers=st_risky_tickers,
                        safe_weight=st_safe_w,
                        crisis_name=crisis,
                        initial_capital=float(st_capital),
                    )
                    st_results[crisis] = result
                elif crisis in st.session_state["custom_stress_scenarios"]:
                    # 2. Custom Scenarios from Simulator
                    custom_data = st.session_state["custom_stress_scenarios"][crisis]
                    # We need to scale to current capital if different
                    scale = float(st_capital) / custom_data["initial_capital"]
                    df_scaled = custom_data["df"].copy()
                    df_scaled["Portfolio (Barbell)"] *= scale
                    df_scaled["Benchmark"] *= scale
                    
                    # Calculate basic metrics for the custom scenario
                    # (This is a simplification, we could use calculate_metrics for more depth)
                    port_vals = df_scaled["Portfolio (Barbell)"].values
                    bench_vals = df_scaled["Benchmark"].values
                    port_dd = calculate_drawdown_analytics(port_vals)
                    bench_dd = calculate_max_drawdown(bench_vals)
                    
                    st_results[crisis] = {
                        "results_df": df_scaled,
                        "metrics": {
                            "crash_portfolio_max_dd": port_dd["max_drawdown"],
                            "crash_benchmark_max_dd": bench_dd,
                            "dd_protection": bench_dd - port_dd["max_drawdown"],
                            "recovery_days": port_dd["max_drawdown_duration_days"],
                            "ulcer_index": port_dd["ulcer_index"],
                            "scenario": {"description": custom_data["description"], "end": str(df_scaled.index.max())}
                        },
                        "error": None
                    }

        st.session_state['stress_results'] = st_results

    if 'stress_results' in st.session_state:
        st_results = st.session_state['stress_results']

        for crisis_name, result in st_results.items():
            if result.get("error"):
                st.error(f"{crisis_name}: {result['error']}")
                continue

            scenario = result['metrics']['scenario']
            st.divider()
            st.subheader(f"{crisis_name}")
            st.caption(scenario['description'])

            m = result['metrics']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Barbell MaxDD w Krachu",
                f"{m['crash_portfolio_max_dd']:.1%}",
                delta=f"{m['dd_protection']:.1%} lepsza niż benchmark",
                delta_color="inverse"
            )
            c2.metric("Benchmark MaxDD", f"{m['crash_benchmark_max_dd']:.1%}")
            c3.metric(
                "Czas Odrabiania Strat",
                f"{m['recovery_days']} sesji" if isinstance(m['recovery_days'], int) else str(m['recovery_days'])
            )
            c4.metric("Ulcer Index", f"{m['ulcer_index']:.2f}")

            # Chart: Portfolio vs Benchmark during crisis
            df_chart = result['results_df']
            fig_st = go.Figure()
            for col, color in zip(df_chart.columns, ['#00ff88', '#ff8800']):
                fig_st.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart[col],
                    mode='lines', name=col,
                    line=dict(color=color, width=2)
                ))

            # Mark crash end — use add_shape+add_annotation to avoid Plotly bug
            # where add_vline(annotation_text=...) crashes on date strings
            crash_end_dt = pd.to_datetime(scenario['end'])
            if crash_end_dt in df_chart.index or (df_chart.index.min() < crash_end_dt < df_chart.index.max()):
                x_str = crash_end_dt.strftime("%Y-%m-%d")
                fig_st.add_shape(
                    type="line",
                    x0=x_str, x1=x_str, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="red", dash="dash", width=1.5),
                )
                fig_st.add_annotation(
                    x=x_str, y=1, xref="x", yref="paper",
                    text="Dno krachu", showarrow=False,
                    yanchor="bottom", font=dict(color="red", size=11),
                )

            fig_st.update_layout(
                title=f"{crisis_name} — Barbell vs Benchmark",
                template="plotly_dark", height=400,
                yaxis_title="Wartość Portfela (PLN)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,25,0.9)",
                hovermode="x unified"
            )
            fig_st.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            fig_st.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
            st.plotly_chart(fig_st, use_container_width=True)

            # ─────────────────────────────────────────────────────────────────
            # 🆕 ROLLING CORRELATION HEATMAP (Dywersyfikacja w kryzysie)
            # ─────────────────────────────────────────────────────────────────
            # Ensure we have Safe_Val and Risky_Val
            if "Safe_Val" in df_chart.columns and "Risky_Val" in df_chart.columns:
                safe_rets = df_chart["Safe_Val"].pct_change().dropna()
                risky_rets = df_chart["Risky_Val"].pct_change().dropna()
                
                # Rolling 21-day correlation
                roll_corr = safe_rets.rolling(21).corr(risky_rets).dropna()
                
                if not roll_corr.empty:
                    fig_corr_time = go.Figure()
                    
                    # Fill logic based on correlation > 0 (bad) vs < 0 (good)
                    fig_corr_time.add_trace(go.Scatter(
                        x=roll_corr.index, 
                        y=roll_corr,
                        mode='lines',
                        name='Korelacja (21-dniowa)',
                        line=dict(color='#00ccff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 204, 255, 0.2)'
                    ))
                    
                    fig_corr_time.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
                    
                    # Also mark crash end
                    if crash_end_dt in df_chart.index or (df_chart.index.min() < crash_end_dt < df_chart.index.max()):
                        fig_corr_time.add_shape(
                            type="line", x0=x_str, x1=x_str, y0=-1, y1=1,
                            xref="x", yref="y", line=dict(color="red", dash="dash", width=1.5),
                        )
                        
                    fig_corr_time.update_layout(
                        title=f"⏳ Płynna Korelacja (Safe vs Risky) w czasie {crisis_name}",
                        template="plotly_dark", height=250,
                        yaxis=dict(range=[-1.05, 1.05], title="Korelacja (Pearson)"),
                        margin=dict(t=40, b=10)
                    )
                    st.plotly_chart(fig_corr_time, use_container_width=True)
                    st.caption("🟢 **Wartości ujemne** (< 0): Idealna dywersyfikacja (gdy ryzykowny spada, bezpieczny rośnie). 🔴 **Wartości dodatnie** (> 0): Załamanie dywersyfikacji (wszystko spada naraz).")


elif module_selection == "🏖️ Emerytura":
    render_emerytura_module()
