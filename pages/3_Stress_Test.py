
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.vanguard_math import compute_tail_dependence_matrix
from modules.styling import apply_styling, module_header, add_crisis_annotations
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
from modules.market_conditions import detect_current_regime
from modules.ui.widgets import ticker_input
from modules.frontier import compute_efficient_frontier
from modules.emerytura import render_emerytura_module
from modules.ai.observer import REGIME_BULL_QUIET, REGIME_BULL_VOL, REGIME_BEAR, REGIME_CRISIS
from modules.global_settings import get_gs, apply_gs_to_session, force_apply_gs_to_session, gs_sidebar_badge
from modules.i18n import t

# ... existsing code ...

# 1. Page Configuration (handled by app.py)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# Globalne ustawienia portfela — wczytaj i wstrzyknij jako domyślne
_gs = get_gs()
apply_gs_to_session(_gs)

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


# Navigation handled by Streamlit natively.

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

# 3. Main Navigation

st.markdown(module_header(
    title=t("st_title"),
    subtitle=t("st_subtitle"),
    icon="🛡️",
    badge="Ochrona Kapitału"
), unsafe_allow_html=True)


st.sidebar.title(t("st_sidebar_title"))
st.sidebar.markdown(t("settings"))
st.sidebar.markdown(t("st_assets"))

from modules.stress_test import run_stress_test, CRISIS_SCENARIOS, run_reverse_stress_test

# Przywracanie z globalnych
if st.sidebar.button(t("restore_global"), key="st_restore_gs", use_container_width=True):
    force_apply_gs_to_session(get_gs())
    st.rerun()

_gs_now = get_gs()
st_safe_str  = ticker_input(t("st_safe_basket"), value=st.session_state.get("_s.st_safe", _gs_now.safe_tickers_str or "TLT, GLD"), key="st_safe", parent=st.sidebar)

default_risky = st.session_state.pop('st_risky_transfer', "SPY, QQQ, BTC-USD")
if 'st_risky' not in st.session_state:
    st.session_state['st_risky'] = default_risky
else:
    # If a transfer happened, overwrite the widget block
    if default_risky != "SPY, QQQ, BTC-USD" and default_risky != st.session_state['st_risky']:
        st.session_state['st_risky'] = default_risky

st_risky_str = ticker_input(t("st_risky_basket"), key="st_risky", parent=st.sidebar)
_gs_sw_default = int(round(get_gs().alloc_safe_pct * 100))
st_safe_w    = st.sidebar.slider(t("st_safe_weight"), 10, 95, st.session_state.get("_s.st_sw", _gs_sw_default), key="st_sw") / 100.0
st_capital   = st.sidebar.number_input(t("st_capital"), value=int(st.session_state.get("_s.st_cap", int(get_gs().initial_capital))), step=10000, key="st_cap")

# ─────────────────────────────────────────────────────────────────
# 🆕 WŁASNY SZOK (CUSTOM SHOCK)
# ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(t("st_custom_shock"))
st.sidebar.caption(t("st_shock_cap"))
c_safe = st.sidebar.slider(t("st_safe_shock").replace("%%", "%"), 0, 50, 0, step=5) / 100.0
c_risky = st.sidebar.slider(t("st_risky_shock").replace("%%", "%"), 0, 90, 40, step=5) / 100.0

if st.sidebar.button(t("st_shock_btn")):
    from modules.stress_test import run_custom_shock
    st.subheader(t("st_custom_title"))
    c_res = run_custom_shock(st_safe_w, c_risky, c_safe, st_capital)
    st.warning(c_res['message'])
    c1, c2, c3 = st.columns(3)
    c1.metric(t("st_safe_val"), f"{c_res['safe_value']:,.0f}")
    c2.metric(t("st_risky_val"), f"{c_res['risky_value']:,.0f}")
    c3.metric(t("st_final"), f"{c_res['final']:,.0f}", f"-{c_res['loss_pct']*100:.1f}%", delta_color="inverse")

# ─────────────────────────────────────────────────────────────────
# 🆕 REVERSE STRESS TESTING
# ─────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(t("st_reverse"))
st.sidebar.caption(t("st_reverse_cap"))
rst_target_loss = st.sidebar.slider(t("st_target_loss").replace("%%", "%"), 5, 80, 20, step=5) / 100.0

if st.sidebar.button(t("st_break_btn")):
    st.subheader(t("st_break_title"))
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
    t("st_select"),
    crisis_options + custom_options,
    default=crisis_options[:3],
    key="st_crises"
)

if st.button(t("st_run_btn"), type="primary", key="st_run"):
    st_safe_tickers  = [x.strip() for x in st_safe_str.split(",")  if x.strip()]
    st_risky_tickers = [x.strip() for x in st_risky_str.split(",") if x.strip()]

    st_results = {}
    with st.spinner(t("st_loading")):
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
            t("st_barbell_dd"),
            f"{m['crash_portfolio_max_dd']:.1%}",
            delta=f"{m['dd_protection']:.1%} {'lepsza niż' if t('lang_label') == '🌐 Język interfejsu' else 'better than'} benchmark",
            delta_color="inverse"
        )
        c2.metric(t("st_bench_dd"), f"{m['crash_benchmark_max_dd']:.1%}")
        c3.metric(
            t("st_recovery"),
            f"{m['recovery_days']} {'sesji' if st.session_state.get('_lang','pl')=='pl' else 'sessions'}" if isinstance(m['recovery_days'], int) else str(m['recovery_days'])
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
        crash_end_dt = pd.to_datetime(scenario['end']) if scenario.get('end') else None
        if crash_end_dt is not None and not pd.isnull(crash_end_dt) and \
                (crash_end_dt in df_chart.index or (df_chart.index.min() < crash_end_dt < df_chart.index.max())):
            x_str = crash_end_dt.strftime("%Y-%m-%d")
            fig_st.add_shape(
                type="line",
                x0=x_str, x1=x_str, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="red", dash="dash", width=1.5),
            )
            fig_st.add_annotation(
                x=x_str, y=1, xref="x", yref="paper",
                text=t("st_crash_bottom"), showarrow=False,
                yanchor="bottom", font=dict(color="red", size=11),
            )

        add_crisis_annotations(fig_st, show=True, opacity=0.15)
        
        fig_st.update_layout(
            title=f"{crisis_name} — Barbell vs Benchmark",
            template="plotly_dark", height=400,
            yaxis_title=t("st_portfolio_val"),
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
                
                if crash_end_dt is not None and not pd.isnull(crash_end_dt) and \
                        (crash_end_dt in df_chart.index or (df_chart.index.min() < crash_end_dt < df_chart.index.max())):
                    fig_corr_time.add_shape(
                        type="line", x0=x_str, x1=x_str, y0=-1, y1=1,
                        xref="x", yref="y", line=dict(color="red", dash="dash", width=1.5),
                    )
                    
                add_crisis_annotations(fig_corr_time, show=True, opacity=0.1)

                fig_corr_time.update_layout(
                    title=f"⏳ Płynna Korelacja (Safe vs Risky) w czasie {crisis_name}",
                    template="plotly_dark", height=250,
                    yaxis=dict(range=[-1.05, 1.05], title="Korelacja (Pearson)"),
                    margin=dict(t=40, b=10)
                )
                st.plotly_chart(fig_corr_time, use_container_width=True)
                st.caption("🟢 **Wartości ujemne** (< 0): Idealna dywersyfikacja (gdy ryzykowny spada, bezpieczny rośnie). 🔴 **Wartości dodatnie** (> 0): Załamanie dywersyfikacji (wszystko spada naraz).")

        # ─────────────────────────────────────────────────────────────────
        # 🆕 DYNAMIC COPULAS: Wskaźnik "Contagion" - Macierz zależności lewego ogona (TDC)
        # ─────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🕸️ Analiza Copula: Efekt Zarazy (Tail Dependence)")
        st.caption("W czasie krachu klasyczna korelacja zawsze dąży do 1. Zależność ogonów (TDC) ukazuje, które dokładnie aktywa pociągają się na dno (Contagion Effect).")

        # Zbierzmy wszystkie testowane aktywa (bez kolumn portfolio) i policzmy TDC
        original_cols = [c for c in df_chart.columns if c not in ["Portfolio (Barbell)", "Benchmark", "Safe_Val", "Risky_Val"]]
        if len(original_cols) > 1:
            asset_rets = df_chart[original_cols].pct_change().dropna()
            if not asset_rets.empty and len(asset_rets) > 20:
                td_matrix = compute_tail_dependence_matrix(asset_rets, q=0.15)
                fig_cop = px.imshow(
                    td_matrix,
                    text_auto=".2f",
                    color_continuous_scale="Reds",
                    zmin=0, zmax=1,
                    title="Macierz Zależności Dolnego Ogona (P(X < q | Y < q))"
                )
                fig_cop.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_cop, use_container_width=True)
                st.caption("🔴 **Czerwone wartości (bliskie 1.0)** oznaczają ekstremalny efekt zarazy: gdy ubezpieczenie nie działa. 🟢 **Białe/Jasne wartości (bliskie 0.0)** oznaczają asymetryczność w piekle rynkowym (Idealny Barbell).")
            else:
                st.info("Kryzys trwał za krótko na wyliczenie Gumbel/Clayton Copula TDC.")


