
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling, module_header
from modules.chart_annotations import add_market_annotations
from modules.simulation import (
    simulate_barbell_strategy, calculate_metrics, run_ai_backtest, 
    calculate_individual_metrics, fit_best_copula
)
from modules.metrics import (
    calculate_trade_stats, calculate_omega, calculate_ulcer_index,
    calculate_pain_index, calculate_drawdown_analytics, calculate_max_drawdown
)
from modules.ai.data_loader import load_data
from modules.analysis_content import display_analysis_report, display_scanner_methodology, display_chart_guide
from modules.scanner import calculate_convecity_metrics, score_asset, compute_hierarchical_dendrogram
from modules.ai.scanner_engine import ScannerEngine
from config import TAX_BELKA, RISK_FREE_RATE_PL
from modules.global_settings import get_gs, apply_gs_to_session, force_apply_gs_to_session, gs_sidebar_badge, should_show_explainer
from modules.i18n import t
from modules.ai.asset_universe import get_sp500_tickers, get_global_etfs
from modules.ui.status_manager import StatusManager
from modules.ui.widgets import tickers_area
from modules.stress_test import run_stress_test, CRISIS_SCENARIOS
from modules.frontier import compute_efficient_frontier
from modules.emerytura import render_emerytura_module
from modules.ai.observer import REGIME_BULL_QUIET, REGIME_BULL_VOL, REGIME_BEAR, REGIME_CRISIS

# ... existsing code ...

# 1. Page Configuration (handled by app.py)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Globalne ustawienia portfela — wczytaj i wstrzyknij jako domyślne
# ---------------------------------------------------------------------------
_gs = get_gs()
apply_gs_to_session(_gs)  # ustawia _s.* klucze tylko jeśli jeszcze nie istnieją

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
    st.session_state["rem_initial_capital"] = _gs.initial_capital
if "rem_expected_return" not in st.session_state:
    st.session_state["rem_expected_return"] = 0.07
if "rem_volatility" not in st.session_state:
    st.session_state["rem_volatility"] = 0.15


# Navigation handled by Streamlit natively.

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = {}

if "baseline_metrics" not in st.session_state:
    st.session_state["baseline_metrics"] = None

# 3. Main Navigation

st.sidebar.title(t("sim_title"))
st.sidebar.markdown(f"### {t('settings')}")

# Przycisk przywracania ustawień globalnych
if st.sidebar.button(t("restore_global"), key="sim_restore_gs", help="Przywróć domyślne wartości z Globalnych Ustawień Portfela", use_container_width=True):
    force_apply_gs_to_session(get_gs())
    st.rerun()

MC_MODE = "Monte Carlo (Teoretyczny)"
AI_MODE = "Intelligent Barbell (Backtest Algorytmiczny)"
mode = st.sidebar.radio(t("sim_mode_label"), [MC_MODE, AI_MODE], index=[MC_MODE, AI_MODE].index(_saved("sim_mode", MC_MODE)), key="sim_mode", on_change=_save, args=("sim_mode",))

if mode == MC_MODE:
    st.sidebar.markdown(t("sim_cap_time"))
    initial_capital = st.sidebar.number_input(t("sim_initial_cap"), value=_saved("mc_cap", 100000), step=10000, key="mc_cap", on_change=_save, args=("mc_cap",))
    years = st.sidebar.slider("Horyzont Inwestycyjny (Lata)", 1, 30, value=_saved("mc_years", 10), key="mc_years", on_change=_save, args=("mc_years",))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2. Część Bezpieczna (Safe Sleeve)")
    _gs_now = get_gs()
    st.sidebar.info(f"🔒 Obligacje Skarbowe RP 3-letnie (Stałe {_gs_now.safe_rate*100:.2f}%) — z Globalnych Ustawień")
    safe_rate = _gs_now.safe_rate

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
        _gs_alloc_default = int(round(get_gs().alloc_safe_pct * 100))
        alloc_safe = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, value=_saved("mc_alloc_safe", _gs_alloc_default), key="mc_alloc_safe", on_change=_save, args=("mc_alloc_safe",)) / 100.0

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
    use_neural_sde = st.sidebar.checkbox(
        "Neural SDEs (Latent Dynamics)",
        value=_saved("mc_use_neural_sde", False), key="mc_use_neural_sde",
        on_change=_save, args=("mc_use_neural_sde",),
        help="Modeluje zmienność za pomocą dynamicznych wag pseudo-sieci uderzając drift i latents. Kidger et al (2022)."
    )
    use_alpha_stable = st.sidebar.checkbox(
        "Lévy-Stable Processes (Heavy Tails)",
        value=_saved("mc_use_alpha_stable", False), key="mc_use_alpha_stable",
        on_change=_save, args=("mc_use_alpha_stable",),
        help="Symuluj rynki finansowe za pomocą procesów o nieskończonej wariancji "
             "z parametrem α wg algorytmu CMS (Chambers-Mallows-Stuck 1976)."
    )
    alpha_stable_alpha = 1.7
    if use_alpha_stable:
        alpha_stable_alpha = st.sidebar.slider(
            "Wykładnik ogona α (Tail Index)", 1.05, 2.0, value=_saved("mc_alpha_stable_alpha", 1.7), step=0.05,
            key="mc_alpha_stable_alpha", on_change=_save, args=("mc_alpha_stable_alpha",),
            help="Dla α=2.0 mamy rozkład normalny. α<2 modelują rynki z grubymi ogonami (krypto to często α≈1.6, S&P500 α≈1.8)."
        )
    use_fbm = st.sidebar.checkbox(
        "Rynki Fraktalne (fBM)",
        value=_saved("mc_use_fbm", False), key="mc_use_fbm",
        on_change=_save, args=("mc_use_fbm",),
        help="Użyj Fractional Brownian Motion (Mandelbrot) by symulować długoterminową pamięć rynków zamiast błądzenia losowego."
    )
    fbm_hurst = 0.5
    if use_fbm:
        fbm_hurst = st.sidebar.slider(
            "Wykładnik Hursta (H)", 0.05, 0.95, value=_saved("mc_fbm_hurst", 0.65), step=0.05,
            key="mc_fbm_hurst", on_change=_save, args=("mc_fbm_hurst",),
            help="H > 0.5 to trend (Momentum), H < 0.5 to Mean-Reversion, H = 0.5 to Random Walk."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 5.5 🕸️ Zależność Ogonowa (Kopuła)")
    copula_family_opt = st.sidebar.selectbox(
        "Model kopuły",
        ["student_t", "clayton", "gumbel", "frank", "Auto-Fit (MLE)"],
        index=["student_t", "clayton", "gumbel", "frank", "Auto-Fit (MLE)"].index(_saved("mc_copula_family", "student_t")),
        key="mc_copula_family",
        on_change=_save, args=("mc_copula_family",),
        help="Zależność ogonowa. Clayton dla wspólnych spadków, Gumbel dla wspólnych wzrostów, Auto-Fit szuka optymalnych MLE parametrów."
    )
    
    copula_theta_val = 2.0
    if copula_family_opt not in ["student_t", "Auto-Fit (MLE)"]:
        copula_theta_val = st.sidebar.slider(
            "Siła zależności (θ)", 0.1, 20.0, value=_saved("mc_copula_theta", 2.0), step=0.1,
            key="mc_copula_theta", on_change=_save, args=("mc_copula_theta",)
        )
    elif copula_family_opt == "Auto-Fit (MLE)":
        st.sidebar.caption("💡 Parametry zostaną skalibrowane podczas symulacji (Max Likelihood na S&P500 i TLT).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 6. 🌩️ Scenario Builder")
    with st.sidebar.expander("Stwórz własny kryzys (Szok)", expanded=False):
        st.markdown("Zdefiniuj ręcznie krach na rynku akcji w konkretnym roku w przyszłości.")
        use_custom_shock = st.checkbox("Włącz własny kryzys", value=False, key="mc_c_shock")
        shock_year = st.slider("W którym roku wystąpi?", 1, years, value=min(3, years), key="mc_c_year")
        shock_drop = st.slider("Głębokość krachu na ryzyku (%)", 10, 80, value=30, step=5, key="mc_c_drop") / 100.0
        
        custom_scenarios = []
        if use_custom_shock:
            custom_scenarios.append({"year": shock_year, "drop_pct": shock_drop})
            st.warning(f"Symulacja zrzuci rynek o **{shock_drop*100:.0f}%** w {shock_year}. roku.")

    # ─── FX Risk Configuration (USD/PLN) ────────────────────────────────────
    st.sidebar.markdown(t("sim_currency_header"))
    use_fx = st.sidebar.toggle(
        t("sim_use_fx"), 
        value=_saved("mc_use_fx", get_gs().currency_risk_enabled), 
        key="mc_use_fx", on_change=_save, args=("mc_use_fx",)
    )
    fx_vol = st.sidebar.slider(
        t("sim_fx_vol"), 0.01, 0.40, 
        value=_saved("mc_fx_vol", get_gs().usd_pln_vol), 
        step=0.01, key="mc_fx_vol", on_change=_save, args=("mc_fx_vol",),
        disabled=not use_fx
    )
    fx_corr = st.sidebar.slider(
        t("sim_fx_corr"), -1.0, 1.0, 
        value=_saved("mc_fx_corr", get_gs().usd_pln_corr), 
        step=0.05, key="mc_fx_corr", on_change=_save, args=("mc_fx_corr",),
        disabled=not use_fx
    )

    # MAIN CONTENT FOR MONTE CARLO
    st.markdown(module_header(
        title="Symulator Monte Carlo",
        subtitle="Analiza teoretyczna Barbell Strategy za pomocą stochastycznych ścieżek cenowych. Porównuj warianty alokacji i optymalizuj ryzyko tail-risk.",
        icon="⚖️",
        badge="Wariant Teoretyczny"
    ), unsafe_allow_html=True)
    
    @st.cache_data(ttl="1h", show_spinner="Pobieranie danych rynkowych dla Kopuły...")
    def get_auto_copula_params():
        import yfinance as yf
        from scipy.stats import rankdata
        try:
            proxy_df = yf.download(["SPY", "TLT"], period="5y", progress=False)["Close"].pct_change().dropna()
            if isinstance(proxy_df.columns, pd.MultiIndex):
                proxy_df.columns = proxy_df.columns.get_level_values(0)
            if not proxy_df.empty and "SPY" in proxy_df.columns and "TLT" in proxy_df.columns:
                u = rankdata(proxy_df["SPY"]) / (len(proxy_df) + 1)
                v = rankdata(proxy_df["TLT"]) / (len(proxy_df) + 1)
                fit_res = fit_best_copula(u, v)
                return fit_res["best_family"], fit_res["best_theta"]
        except Exception as e:
            st.sidebar.warning(f"Auto-Fit MLE failed: {e}")
        return "clayton", 2.0

    # ─── Sequential Fallback Execution ──────────────────────────────────────
    if st.session_state.get('mc_sequential_retry', False):
        st.session_state['mc_sequential_retry'] = False
        with st.status("🏗️ Uruchamiam Symulację w trybie bezpiecznym (sekwencyjnym)...", expanded=True):
            try:
                # Resolve Auto-Fit in Main Process if needed
                final_copula_family = copula_family_opt
                final_copula_theta = copula_theta_val
                if copula_family_opt == "Auto-Fit (MLE)":
                    final_copula_family, final_copula_theta = get_auto_copula_params()

                sim_args_retry = {
                    "n_years": years,
                    "n_simulations": 1000,
                    "initial_captial": initial_capital,
                    "safe_rate": safe_rate,
                    "risky_mean": risky_mean,
                    "risky_vol": risky_vol,
                    "risky_kurtosis": risky_kurtosis,
                    "alloc_safe": alloc_safe,
                    "rebalance_strategy": rebalance_strategy.split(" ")[0],
                    "threshold_percent": threshold_percent,
                    "use_qmc": use_qmc,
                    "use_garch": use_garch,
                    "use_jump_diffusion": use_jump_diffusion,
                    "custom_scenarios": custom_scenarios,
                    "use_fbm": use_fbm,
                    "fbm_hurst": fbm_hurst,
                    "use_alpha_stable": use_alpha_stable,
                    "alpha_stable_alpha": alpha_stable_alpha,
                    "copula_family": final_copula_family,
                    "copula_theta": final_copula_theta,
                    "use_neural_sde": use_neural_sde,
                    "use_currency_risk": use_fx,
                    "usd_pln_vol": fx_vol,
                    "usd_pln_corr": fx_corr
                }
                wealth_paths = simulate_barbell_strategy(**sim_args_retry)
                metrics = calculate_metrics(wealth_paths, years)
                st.session_state['mc_results'] = {
                    "wealth_paths": wealth_paths,
                    "metrics": metrics,
                    "years": years
                }
                st.success("Symulacja zakończona sukcesem (Tryb Bezpieczny)!")
                st.rerun()
            except Exception as e:
                st.error(f"Krytyczny błąd trybu bezpiecznego: {e}")

    if st.button("🚀 Symuluj Wyniki (Ctrl+Enter)", type="primary", key="mc_run"):
        # 1. Resolve Auto-Fit (MLE) in Main Process if needed
        final_copula_family = copula_family_opt
        final_copula_theta = copula_theta_val
        
        if copula_family_opt == "Auto-Fit (MLE)":
            final_copula_family, final_copula_theta = get_auto_copula_params()

        # Prepare arguments for execution
        sim_args = {
            "n_years": years,
            "n_simulations": 1000,
            "initial_captial": initial_capital,
            "safe_rate": safe_rate,
            "risky_mean": risky_mean,
            "risky_vol": risky_vol,
            "risky_kurtosis": risky_kurtosis,
            "alloc_safe": alloc_safe,
            "rebalance_strategy": rebalance_strategy.split(" ")[0],
            "threshold_percent": threshold_percent,
            "use_qmc": use_qmc,
            "use_garch": use_garch,
            "use_jump_diffusion": use_jump_diffusion,
            "custom_scenarios": custom_scenarios,
            "use_fbm": use_fbm,
            "fbm_hurst": fbm_hurst,
            "use_alpha_stable": use_alpha_stable,
            "alpha_stable_alpha": alpha_stable_alpha,
            "copula_family": final_copula_family,
            "copula_theta": final_copula_theta,
            "use_neural_sde": use_neural_sde,
            "use_currency_risk": use_fx,
            "usd_pln_vol": fx_vol,
            "usd_pln_corr": fx_corr
        }
        
        # Submit to thread pool directly (avoids ProcessPool Windows/Streamlit conflicts causing BrokenProcessPool)
        from concurrent.futures import ThreadPoolExecutor

        def get_executor():
            if 'mc_executor' not in st.session_state:
                st.session_state['mc_executor'] = ThreadPoolExecutor(max_workers=2)
            return st.session_state['mc_executor']

        try:
            executor = get_executor()
            future = executor.submit(simulate_barbell_strategy, **sim_args)
            st.session_state['mc_future'] = future
            st.session_state['mc_task_years'] = years
            st.session_state.pop('mc_results', None) # Clear previous results
        except RuntimeError as e:
            st.warning(f"⚠️ Problem z pulą wątków: {e}. Przechodzę na tryb bezpieczny (sekwencyjny)...")
            # Fallback: Run sequentially in main thread
            try:
                wealth_paths = simulate_barbell_strategy(**sim_args)
                metrics = calculate_metrics(wealth_paths, years)
                st.session_state['mc_results'] = {
                    "wealth_paths": wealth_paths,
                    "metrics": metrics,
                    "years": years
                }
                st.success("Symulacja zakończona w trybie bezpiecznym!")
                st.rerun()
            except Exception as e2:
                st.error(f"Krytyczny błąd symulacji: {e2}")
    
    # Async polling fragment
    if 'mc_future' in st.session_state and 'mc_results' not in st.session_state:
        future = st.session_state['mc_future']
        
        @st.fragment(run_every="2s")
        def poll_monte_carlo():
            if future.done():
                try:
                    wealth_paths = future.result()
                    # Calculate metrics synchronously since it's fast
                    metrics = calculate_metrics(wealth_paths, st.session_state['mc_task_years'])
                    st.session_state['mc_results'] = {
                        "wealth_paths": wealth_paths,
                        "metrics": metrics,
                        "years": st.session_state['mc_task_years']
                    }
                    st.success("Symulacja zakończona!")
                    st.rerun()
                except Exception as e:
                    if "terminated abruptly" in str(e).lower() or "broken" in str(e).lower():
                        st.warning("⚠️ Proces przerwał pracę. Ponawiam symulację w trybie bezpiecznym...")
                        # Automatic retry in main thread
                        try:
                            # Re-fetch args from session if possible or just re-simulate if context allows
                            # Here we assume sim_args is available or we use a fallback flag
                            st.session_state['mc_sequential_retry'] = True
                            st.rerun()
                        except: pass
                    st.error(f"Błąd symulacji: {e}")
                    st.session_state.pop('mc_future', None)
            else:
                with st.spinner("⏳ Symulacja Monte Carlo działa w tle... Możesz korzystać z innych opcji."):
                    st.info("Obliczanie ścieżek...")
        
        poll_monte_carlo()

    # Check if results exist and display
    if 'mc_results' in st.session_state:
        res = st.session_state['mc_results']
        wealth_paths = res['wealth_paths']
        metrics = res['metrics']
        years = res['years']


        col1, col2, col3, col4 = st.columns(4)
        
        # Helper for Baseline delta
        def _delta(key, val, fmt, invert=False):
            if st.session_state["baseline_metrics"] is None: return None
            base_val = st.session_state["baseline_metrics"].get(key, val)
            diff = val - base_val
            if abs(diff) < 1e-6: return None
            # Formatting
            if "%" in fmt: d_str = f"{diff:+.2%}"
            else: d_str = f"{diff:+.2f}"
            # Color logic
            if invert: color = "normal" if diff <= 0 else "inverse"
            else: color = "normal" if diff >= 0 else "inverse"
            return d_str, color

        # Metrics display with delta
        d1 = _delta('mean_final_wealth', metrics['mean_final_wealth'], ".0f")
        col1.metric("Średni Kapitał", f"{metrics['mean_final_wealth']:,.0f} PLN", 
                    delta=d1[0] if d1 else None, delta_color=d1[1] if d1 else "normal")
                    
        d2 = _delta('mean_cagr', metrics['mean_cagr'], "%")
        col2.metric("CAGR", f"{metrics['mean_cagr']:.2%}", 
                    delta=d2[0] if d2 else None, delta_color=d2[1] if d2 else "normal")
                    
        d3 = _delta('median_cagr', metrics['median_cagr'], "%")
        col3.metric("Mediana CAGR", f"{metrics['median_cagr']:.2%}",
                    delta=d3[0] if d3 else None, delta_color=d3[1] if d3 else "normal")
                    
        d4 = _delta('prob_loss', metrics['prob_loss'], "%", invert=True)
        col4.metric("Szansa Straty", f"{metrics['prob_loss']:.1%}",
                    delta=d4[0] if d4 else None, delta_color=d4[1] if d4 else "normal")
        
        # ── Buttons for Scenario and Baseline ──
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
        with c_btn1:
            if st.button("📌 Ustaw jako Baseline", key="set_baseline_btn", help="Kolejne symulacje będą pokazywać różnicę względem tego wyniku."):
                st.session_state["baseline_metrics"] = metrics.copy()
                st.rerun()
        with c_btn2:
            if st.button("🗑️ Reset Baseline", key="clear_baseline_btn"):
                st.session_state["baseline_metrics"] = None
                st.rerun()
                
        with st.expander("💾 Zapisz Scenariusz do Porównania", expanded=False):
            scn_name = st.text_input("Nazwa scenariusza", value=f"Scenariusz {len(st.session_state['saved_scenarios'])+1}")
            if st.button("Zapisz", key="save_scen_btn"):
                st.session_state["saved_scenarios"][scn_name] = {
                    "wealth_paths": wealth_paths,
                    "metrics": metrics,
                    "years": years
                }
                st.success(f"Zapisano '{scn_name}'")
        
        # ── COMPARISON VIEW ──
        if len(st.session_state["saved_scenarios"]) > 0:
            with st.expander("📊 Porównanie Scenariuszy A/B/C", expanded=False):
                st.markdown("Porównanie zapisanych wyników symulacji.")
                
                # Checkboxes to select scenarios to plot
                sel_scens = []
                st.write("Wybierz do nałożenia na wykres:")
                sc_cols = st.columns(min(4, max(1, len(st.session_state["saved_scenarios"]))))
                for i, sname in enumerate(st.session_state["saved_scenarios"].keys()):
                    with sc_cols[i % len(sc_cols)]:
                        if st.checkbox(sname, value=True, key=f"chk_{sname}"):
                            sel_scens.append(sname)
                
                if st.button("Wyczyść zapisane scenariusze"):
                    st.session_state["saved_scenarios"] = {}
                    st.rerun()
                
                if sel_scens:
                    fig_comp = go.Figure()
                    colors = ['#00e676', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
                    
                    rows_comp = []
                    for i, sname in enumerate(sel_scens):
                        sdata = st.session_state["saved_scenarios"][sname]
                        smet = sdata["metrics"]
                        
                        # Add to table
                        rows_comp.append({
                            "Scenariusz": sname,
                            "Mediana CAGR": f"{smet['median_cagr']:.2%}",
                            "Max Drawdown": f"{smet['mean_max_drawdown']:.1%}",
                            "Sharpe": f"{smet['median_sharpe']:.2f}",
                            "Ulcer Index": f"{calculate_ulcer_index(np.median(sdata['wealth_paths'], axis=0)):.2f}"
                        })
                        
                        # Add to plot (median path)
                        med_path = np.median(sdata["wealth_paths"], axis=0)
                        days_c = np.arange(len(med_path))
                        fig_comp.add_trace(go.Scatter(
                            x=days_c, y=med_path, mode='lines',
                            name=sname, line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    
                    fig_comp.update_layout(
                        template="plotly_dark", title="Mediana Kapitału — Porównanie",
                        xaxis_title="Dni", yaxis_title="PLN", height=350,
                        hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    st.dataframe(pd.DataFrame(rows_comp), use_container_width=True, hide_index=True)

        exp_mode = should_show_explainer()
        with st.expander("📖 Kluczowe Wskaźniki (KPI)", expanded=exp_mode):
            st.markdown("""
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
        if "results_df" in st.session_state and not st.session_state["results_df"].empty:
            with st.expander("📊 Tabela Profesjonalna (Risk & Performance)", expanded=False):
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
            st.switch_page("pages/3_Stress_Test.py")
        
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
        
        # B. Histogram of Wealth (Animated via Slider)
        st.markdown("### 📊 Animowany Rozkład Kapitału")
        st.caption("Przesuń suwak, aby zobaczyć jak rozkład majątku i ryzyko ogona (VaR) ewoluują w czasie.")
        selected_year = st.slider("Wybierz rok symulacji do analizy:", min_value=1, max_value=int(years), value=int(years), key="mc_hist_year")
        day_idx = min(selected_year * 252, wealth_paths.shape[1] - 1)
        current_wealths = wealth_paths[:, day_idx]
        
        fig_hist = px.histogram(
            current_wealths, 
            nbins=50, 
            title=f"Rozkład Kapitału w {selected_year}. roku",
            labels={'value': 'Kapitał (PLN)'},
            color_discrete_sequence=['#00ff88']
        )
        
        # Add VaR lines
        var_95 = np.percentile(current_wealths, 5)
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text=f"VaR 95% = {var_95:,.0f} PLN")
        fig_hist.add_vline(x=initial_capital, line_dash="dash", line_color="white", annotation_text="Start")
        fig_hist.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        display_chart_guide("Animowany Histogram i VaR", """
        *   **Ewolucja w czasie**: Zobaczysz, jak z czasem rozkład się spłaszcza (rośnie niepewność), ale jednocześnie przesuwa w prawo (rośnie zysk dzięki procentowi składanemu).
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

elif mode == "Intelligent Barbell (Backtest Algorytmiczny)":
    st.sidebar.markdown("### 1. Konfiguracja Podstawowa")
    initial_capital = st.sidebar.number_input("Kapitał Początkowy (USD)", value=_saved("ai_cap", 100000), step=10000, key="ai_cap", on_change=_save, args=("ai_cap",))
    start_date = st.sidebar.date_input("Data Początkowa", value=_saved("ai_start", pd.to_datetime("2020-01-01")), key="ai_start", on_change=_save, args=("ai_start",))

    st.sidebar.markdown("### 2. Aktywa")
    
    # Safe Asset Selection
    _ai_safe_opts = ["Tickers (Yahoo)", "Holistyczne Obligacje Skarbowe (TOS 5.51%)"]
    safe_type = st.sidebar.radio("Rodzaj Bezpiecznego Aktywa", _ai_safe_opts, index=_ai_safe_opts.index(_saved("ai_safe_type", "Holistyczne Obligacje Skarbowe (TOS 5.51%)")), key="ai_safe_type", on_change=_save, args=("ai_safe_type",))
    safe_tickers_str = ""
    safe_fixed_rate = RISK_FREE_RATE_PL
    
    if safe_type == "Tickers (Yahoo)":
        safe_tickers_str = tickers_area("Koszyk Bezpieczny (Safe)", value=_saved("ai_safe_tickers", "TLT, IEF, GLD"), help="Obligacje, Złoto", key="ai_safe_tickers", on_change=_save, args=("ai_safe_tickers",), parent=st.sidebar)
        cap_freq = 1  # bez znaczenia dla tickerów
    else:
        st.sidebar.info("Generowanie syntetycznego aktywa o stałym wzroście 5.51% rocznie.")
        safe_fixed_rate = st.sidebar.number_input("Oprocentowanie Obligacji (%)", value=_saved("ai_safe_rate", 5.51), step=0.1, key="ai_safe_rate", on_change=_save, args=("ai_safe_rate",)) / 100.0

        _cap_opts = {
            "Roczna (TOS — domyślna)": 1,
            "Półroczna": 2,
            "Kwartalna": 4,
            "Miesięczna": 12,
        }
        _cap_label = st.sidebar.selectbox(
            "💰 Kapitalizacja odsetek",
            list(_cap_opts.keys()),
            index=0,
            key="ai_cap_freq",
            help="Jak często odsetki są dopisywane do kapitału.\n"
                 "Roczna = standard TOS (PKO BP).\n"
                 "Częstsza kapitalizacja → wyższy efektywny zysk (procent składany)."
        )
        cap_freq = _cap_opts[_cap_label]
        # Info: efektywna stopa roczna po podatku Belki
        rate_net = safe_fixed_rate * (1 - 0.19)
        ear = (1 + rate_net / cap_freq) ** cap_freq - 1
        st.sidebar.caption(
            f"📈 EAR po Belce: **{ear:.4%}** "
            f"(nominalna {safe_fixed_rate:.2%} → netto {rate_net:.4%} → "
            f"kapitalizowana {cap_freq}×/rok)"
        )

    _ai_risky_opts = ["Lista (Auto Wagi)", "Manualne Wagi"]
    risky_asset_mode = st.sidebar.radio("Tryb Wyboru Aktywów Ryzykownych", _ai_risky_opts, index=_ai_risky_opts.index(_saved("ai_risky_mode", "Lista (Auto Wagi)")), key="ai_risky_mode", on_change=_save, args=("ai_risky_mode",))
    _gs_risky_default = get_gs().risky_tickers_str or "SPY, QQQ, NVDA, BTC-USD"
    risky_tickers_str = _gs_risky_default  # Default for logic
    risky_weights_manual = None

    if risky_asset_mode == "Lista (Auto Wagi)":
         risky_tickers_str = tickers_area("Koszyk Ryzykowny (Risky)", value=_saved("ai_risky_tickers", _gs_risky_default), help="Akcje, Krypto", key="ai_risky_tickers", on_change=_save, args=("ai_risky_tickers",), parent=st.sidebar)
         # Logic uses this string later
    else:
        st.sidebar.markdown("**Manualne Wagi Aktywów**")
        # Initialize session state for table if it doesn't exist
        if "ai_manual_df" not in st.session_state:
            st.session_state["ai_manual_df"] = pd.DataFrame([{"Ticker": "SPY", "Waga (%)": 100.0}])
            
            # Check for data transferred from Scanner
            if 'transfer_data' in st.session_state and not st.session_state['transfer_data'].empty:
                st.session_state["ai_manual_df"] = st.session_state['transfer_data']

        edited_df = st.sidebar.data_editor(
            st.session_state["ai_manual_df"], 
            num_rows="dynamic", 
            use_container_width=True, 
            key="ai_manual_table_editor"
        )
        
        # Translation of ISINs in the dataframe
        from modules.isin_resolver import ISINResolver
        needs_rerun = False
        for i, row in edited_df.iterrows():
            tkr = str(row["Ticker"]).strip()
            if tkr and ISINResolver.is_isin(tkr):
                resolved = ISINResolver.resolve(tkr)
                if resolved != tkr:
                    edited_df.at[i, "Ticker"] = resolved
                    needs_rerun = True

        if needs_rerun:
            st.session_state["ai_manual_df"] = edited_df
            st.rerun()
        else:
            # Update base state in case user added/deleted row normally so we persist it
            st.session_state["ai_manual_df"] = edited_df

        # Validation
        total_weight = edited_df["Waga (%)"].sum()
        if abs(total_weight - 100.0) > 0.01:
            st.sidebar.error(f"Suma wag musi wynosić 100%! Obecnie: {total_weight:.1f}%")
        
        # Prepare data for simulation
        risky_weights_manual = {}
        valid_tickers = []
        for index, row in edited_df.iterrows():
            tkr = str(row["Ticker"]).strip().upper()
            try:
                w = float(row["Waga (%)"]) / 100.0
            except:
                w = 0.0
                
            if tkr:
                risky_weights_manual[tkr] = w
                valid_tickers.append(tkr)
        
        risky_tickers_str = ", ".join(valid_tickers) # Mock string to reuse existing download logic
    
    st.sidebar.markdown("### 3. Strategia Alokacji")
    _ai_alloc_opts = ["Dynamiczna Alokacja (Regime + RL)", "Manual Fixed", "Rolling Kelly"]
    allocation_mode = st.sidebar.selectbox("Tryb Alokacji", _ai_alloc_opts, index=_ai_alloc_opts.index(_saved("ai_alloc_mode", "Dynamiczna Alokacja (Regime + RL)")), key="ai_alloc_mode", on_change=_save, args=("ai_alloc_mode",))
    
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
        
    st.sidebar.markdown("### 5. Koszty i Ryzyko (NOWE 🆕)")
    with st.sidebar.expander("🛠️ Parametry Zaawansowane", expanded=False):
        st.markdown("**Koszty Transakcyjne (Ratio)**")
        cost_equity = st.slider("Akcje PL/US (%)", 0.0, 1.0, value=TAX_BELKA, step=0.01, key="cost_eq") / 100.0
        cost_crypto = st.slider("Kryptowaluty (%)", 0.0, 2.0, value=0.60, step=0.1, key="cost_crypto") / 100.0
        cost_etf    = st.slider("ETF broker (%)", 0.0, 0.5, value=0.05, step=0.01, key="cost_etf") / 100.0
        cost_bonds  = st.slider(
            "Obligacje / TOS (%)", 0.0, 0.5, value=0.0, step=0.01, key="cost_bonds",
            help="Koszty transakcyjne zakupu i sprzedaży obligacji (dom maklerski, spread). "
                 "Dla TOS (Obligacji Skarbowych) typowo 0% — brak prowizji w PKO BP."
        ) / 100.0

        st.markdown("**Zarządzanie Ryzykiem**")
        stop_loss = st.slider("Hard Stop-Loss (%)", 0, 50, value=0, key="sl") / 100.0
        trailing_stop = st.slider("Trailing Stop (%)", 0, 30, value=0, key="ts") / 100.0
        vol_target = st.slider("Volatility Target (%)", 0, 100, value=0, key="vt") / 100.0

    st.sidebar.markdown(t("sim_currency_header"))
    use_fx_ai = st.sidebar.toggle(
        t("sim_use_fx"), 
        value=_saved("ai_use_fx", get_gs().currency_risk_enabled), 
        key="ai_use_fx", on_change=_save, args=("ai_use_fx",)
    )

    trans_costs = {
        "equity_pl": cost_equity,
        "etf": cost_etf,
        "crypto": cost_crypto,
        "bonds": cost_bonds,
        "bid_ask": 0.0002
    }
    risk_params = {
        "stop_loss": stop_loss,
        "trailing_stop": trailing_stop,
        "vol_target": vol_target
    }
        

    
    st.title("🧠 Intelligent Barbell - Backtest Algorytmiczny")
    st.markdown("""
    **Moduły Algorytmiczne:**
    - **Observer (HMM)**: Wykrywa reżimy rynkowe (Risk-On / Risk-Off).
    - **Architect (HRP)**: Buduje zdywersyfikowany portfel wewnątrz koszyków.
    - **Trader (RL Agent)**: Dynamicznie zarządza lewarem (Kelly).
    """)
    
    if st.button("🧠 Uruchom Backtest", type="primary"):
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
        status_ai = StatusManager("Przygotowanie Backtestu...", expanded=False)
        
        status_ai.info_data("Pobieranie danych historycznych...")
        safe_data = pd.DataFrame()
        base_curr_arg = "PLN" if use_fx_ai else None
        
        if safe_tickers:
            safe_data = load_data(safe_tickers, start_date=start_date, base_currency=base_curr_arg)
                
        risky_data = load_data(risky_tickers, start_date=start_date, base_currency=base_curr_arg)
        
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
                risky_weights_dict=risky_weights_manual,
                transaction_costs=trans_costs,
                risk_params=risk_params,
                cap_freq=cap_freq
            )
            
            status_ai.info_math("Finalizacja metryk...")
            
            # Metrics
            years = (results.index[-1] - results.index[0]).days / 365.25
            metrics = calculate_metrics(results['PortfolioValue'].values, years)
            
            # Calculate Trade Stats approximation
            trade_stats = calculate_trade_stats(results['PortfolioValue'])

            # Fetch Benchmark (S&P 500 & 60/40)
            status_ai.info_data("Pobieranie benchmarku (S&P 500)...")
            bench_data = load_data(["^GSPC"], start_date=start_date)
            bench_spy = None
            bench_6040 = None
            if not bench_data.empty:
                bench_prices = bench_data["^GSPC"]
                # Align indices
                bench_prices.index = bench_prices.index.tz_localize(None) if bench_prices.index.tz is None else bench_prices.index.tz_convert(None)
                results_idx = results.index.tz_localize(None) if results.index.tz is None else results.index.tz_convert(None)
                bench_prices = bench_prices.reindex(results_idx, method='ffill').ffill().bfill()
                bench_spy = (bench_prices / bench_prices.iloc[0]) * initial_capital
                
                # 60/40 Portfolio (60% SPY, 40% Bonds at safe_fixed_rate)
                spy_rets = bench_prices.pct_change().fillna(0)
                daily_safe = (1 + safe_fixed_rate)**(1/252) - 1
                b6040_rets = 0.6 * spy_rets + 0.4 * daily_safe
                bench_6040 = initial_capital * (1 + b6040_rets).cumprod()

            status_ai.success("Backtest zakończony sukcesem!")
            
            # Save results and Rerun
            st.session_state['backtest_results'] = {
                "results": results,
                "metrics": metrics,
                "trade_stats": trade_stats,
                "risky_mean": risky_data.mean(axis=1),
                "regimes": regimes,
                "bench_spy": bench_spy,
                "bench_6040": bench_6040
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
        if st.button("⚡ Przenieś Backtest Algorytmiczny do Stress Testów", key="ai_to_stress"):
            df_results = res['results']
            df_custom = pd.DataFrame({
                "Portfolio (Barbell)": df_results["PortfolioValue"],
                "Benchmark": df_results.get("Benchmark", df_results["PortfolioValue"])
            }, index=df_results.index)
            
            st.session_state["custom_stress_scenarios"]["🧠 Algorytm: Backtest Strategy"] = {
                "df": df_custom,
                "initial_capital": initial_capital,
                "description": f"Pełna historia portfela wypracowana przez system algorytmiczny od {df_custom.index.min().date()}."
            }
            st.switch_page("pages/3_Stress_Test.py")

        st.divider()

        # --- 2. CHARTS SECTION (Bottom) ---
        st.subheader("📈 Wykresy Analityczne")
        
        # A. Equity Curve
        st.markdown("#### 1. Krzywa Kapitału (Equity Curve)")
        
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=res['results'].index, y=res['results']['PortfolioValue'], mode='lines', name='Smart Barbell', line=dict(color='#00ff88', width=2)))
        
        # --- Benchmarks ---
        show_benchmarks = st.checkbox("Pokaż benchmarki (S&P500, 60/40)", value=True, key="ai_show_bench")
        show_crisis = st.checkbox("🏛️ Pokaż Historyczne Kryzysy", value=False, key="ai_show_crisis", help="Zaznacza na wykresie okresy takie jak COVID-19 czy bessa z 2022.")
        
        if show_benchmarks and res.get('bench_spy') is not None:
            fig.add_trace(go.Scattergl(x=res['results'].index, y=res['bench_spy'], mode='lines', name='S&P 500 (100% Akcje)', line=dict(color='#ff4444', width=1, dash='dash')))
            fig.add_trace(go.Scattergl(x=res['results'].index, y=res['bench_6040'], mode='lines', name='Klasyczne 60/40', line=dict(color='#3498db', width=1, dash='dash')))

        if show_crisis:
            add_market_annotations(fig, res['results'].index.min(), res['results'].index.max())

        risky_series = res.get('risky_mean', pd.Series())
        regimes_arr = res.get('regimes', [])

        fig.update_layout(title="Wzrost Wartości Portfela vs Benchmark", template="plotly_dark", height=500, hovermode="x unified")
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
        if show_crisis:
            add_market_annotations(fig_underwater, drawdowns.index.min(), drawdowns.index.max())
        fig_underwater.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        fig_underwater.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
        st.plotly_chart(fig_underwater, use_container_width=True, key="chart_underwater_plot")
        
        display_chart_guide("Underwater Plot", """
        *   **Interpretacja**: Pokazuje ile % tracisz względem "szczytu" portfela.
        *   **Cel**: Jak najpłytsze (krótkie słupki) i jak najwęższe (szybki powrót) "dołki".
        """)
        
        st.divider()

        # C. Regime Plot
        st.markdown("#### 3. Detekcja Reżimów (Kontekst Architektury)")
        
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
            obs = res.get('observer')
            if obs and hasattr(obs, 'n_regimes') and obs.n_regimes == 4:
                # 4-state HMM colors
                def _regime_color(r):
                    desc = obs.get_regime_desc(r)
                    if "Crisis" in desc: return '#ff2222' # 🔴
                    if "Bear" in desc:   return '#ffaa00' # 🟠
                    if "Volatile" in desc: return '#3498db' # 🔵 (Bull Vol)
                    return '#00ff88' # 🟢 (Bull Quiet)
                
                regime_colors = [_regime_color(r) for r in regimes_arr]
                regime_label_map = {r: obs.get_regime_desc(r) for r in range(4)}
            elif obs and hasattr(obs, 'high_vol_state'):
                # Legacy 3-state
                crisis_idx = obs.high_vol_state
                low_idx = getattr(obs, 'low_vol_state', 0)
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
        # 🆕 ANTI-FRAGILITY SCORE (Taleb)
        # ─────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🛡️ Anti-Fragility Score (Odporność na Kryzysy) 🆕")
        st.caption("Reference: Nassim Nicholas Taleb (2012) — 'Antifragile: Things That Gain from Disorder'")
        
        # Obliczenie AF Score - załóżmy, że aktywo ryzykowne (SPY/EQ) to nasz benchmark
        try:
            # Sprobuj pobrac benchmark. Jak nie ma, fallback = proxy benchmark
            if 'backtest_risky_data' in st.session_state and not st.session_state['backtest_risky_data'].empty:
                bench_series = st.session_state['backtest_risky_data'].iloc[:, 0]
                bench_rets = bench_series.pct_change().dropna()
            else:
                bench_rets = pv_series.pct_change().dropna() # Fallback same returns

            # Zrownywanie indeksow jesli trzeba byloby
            min_len = min(len(port_returns_pct), len(bench_rets))
            port_arr = port_returns_pct[-min_len:]
            bench_arr = bench_rets.values[-min_len:]

            from modules.metrics import calculate_antifragility_score
            af_score = calculate_antifragility_score(port_arr, bench_arr, crisis_threshold_pct=-0.10)
            
            af_col1, af_col2 = st.columns([1, 2])
            with af_col1:
                st.metric("Anti-Fragility Score", f"{af_score:.2f}", help="Score > 0 oznacza że portfel ZYSKUJE podczas krachów. Score = -1.0 oznacza że podąża za rynkiem.")
            with af_col2:
                if af_score > 0:
                    st.success("**Klasyfikacja: ANTY-KRUCHY (Antifragile)**. Twój portfel faktycznie zarabia gdy inni tracą. Gratulacje! Jesteś gotowy na Czarnego Łabędzia.")
                elif af_score > -0.5:
                    st.info("**Klasyfikacja: ODPORNY (Robust/Resilient)**. Twój portfel traci znacznie mniej niż rynek podczas krachów. Dobra ochrona kapitału.")
                else:
                    st.warning("**Klasyfikacja: KRUCHY (Fragile)**. Twój portfel silnie traci pod presją. Rozważ zakup opcji OTM (przy użyciu suwaka Tail Hedging) aby odwrócić asymetrię wypłaty.")
                    
        except Exception as e:
            st.error(f"Nie udało się wyliczyć AF Score: {e}")

        # ─────────────────────────────────────────────────────────────────
        # 🆕 TAIL INSURANCE (Prawdziwy Barbell Taleba)
        # ─────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🛡️ Opcje OTM — Prawdziwy Barbell Taleba 🆕")
        st.markdown("Barbell nie polega tylko na posiadaniu 85% w bezpiecznych aktywach. Najważniejszym elementem jest opcjonalność dla prawego ogona — regularne 'krwawienie' z małej składki, które eksploduje zyskiem przy Czarnym Łabędziu.")
        
        with st.expander("🦅 Symulator Tail Hedging (Kalkulator Premii Opcji)"):
            st.markdown("Jak stałe wydawanie % kapitału na głęboko OTM (Out of The Money) opcje sprzedaży / kupna zachowa się w kryzysie?")
            c_tail1, c_tail2 = st.columns(2)
            
            with c_tail1:
                annual_premium = st.slider("Roczny budżet na opcje (% kapitału)", 0.0, 5.0, 2.0, 0.5) / 100.0
                crash_prob = st.slider("Szansa na kryzys każdego roku (%)", 1.0, 30.0, 10.0, 1.0) / 100.0
                
            with c_tail2:
                option_payout = st.slider("Wypłata z opcji przy krachu (Mnożnik)", 5, 50, 15)
                sim_years_tail = st.slider("Lata symulacji opcjonalności", 5, 30, 15)
                
            if st.button("Uruchom Symulację Opcjonalności"):
                paths_qty = 1000
                rng_tail = np.random.default_rng(42)
                crashes = rng_tail.binomial(1, crash_prob, size=(paths_qty, sim_years_tail))
                
                # Zwykly portfel SP500 uproszczony
                port_normal = np.ones((paths_qty, sim_years_tail+1))
                port_hedged = np.ones((paths_qty, sim_years_tail+1))
                
                for y in range(sim_years_tail):
                    market_ret = rng_tail.normal(0.08, 0.15, size=paths_qty)
                    # nadpisz zwrot w roku kryzysu
                    is_crash = crashes[:, y] == 1
                    market_ret[is_crash] = rng_tail.normal(-0.35, 0.10, size=np.sum(is_crash))
                    
                    # Portfel normalny rośnie/traci z rynkiem
                    port_normal[:, y+1] = port_normal[:, y] * (1 + market_ret)
                    
                    # Portfel ze strata premii, ale z wypłatą przy crashu
                    hedge_payout = np.zeros(paths_qty)
                    hedge_payout[is_crash] = annual_premium * option_payout
                    hedge_return = market_ret - annual_premium + hedge_payout
                    
                    port_hedged[:, y+1] = port_hedged[:, y] * (1 + hedge_return)
                
                fig_tail = go.Figure()
                fig_tail.add_trace(go.Scatter(y=np.median(port_normal, axis=0), name="Zwykły Maklerski (Mediana)", line=dict(color="#ff1744")))
                fig_tail.add_trace(go.Scatter(y=np.median(port_hedged, axis=0), name="Portfel z Tail Hedge (Mediana)", line=dict(color="#00e676", width=3)))
                
                fig_tail.update_layout(template="plotly_dark", height=300, title="Wpływ ubezpieczenia od ogona (Symulacja 1000 ścieżek)", yaxis_title="Wielokrotność Kapitału")
                st.plotly_chart(fig_tail, use_container_width=True)
                
                st.info(f"Opcje kosztują Cię **{annual_premium*100}%** każdego roku. W scenariuszach bez kryzysu obniżają CAGR. Gdy jednak kryzys się zmaterializuje, zapewniają kapitał na zakupy w samym dołku.")

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


