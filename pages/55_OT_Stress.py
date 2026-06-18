"""
55_OT_Stress.py — Optimal Transport Stress Testing
Strona Streamlit dla stress testów przez Wasserstein interpolację.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="OT Stress Testing",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .main { background: #0a0b14; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #12131f, #1a1b2e);
        border: 1px solid #2a2b3d;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .stress-high  { border-left: 4px solid #e74c3c; }
    .stress-med   { border-left: 4px solid #f39c12; }
    .stress-low   { border-left: 4px solid #00e676; }
    h1 { color: #e2e4f0; font-family: 'Inter', sans-serif; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚡ Optimal Transport Stress Testing")
st.markdown("""
Generuje realistyczne scenariusze stresowe przez **interpolację rozkładów** (Wasserstein barycentrum).
Zamiast prostego skalowania parametrów, OT morphuje rozkład normalny w rozkład kryzysowy.

*Villani (2009) "Optimal Transport", Peyré & Cuturi (2019), Blanchet & Murthy (2019)*
""")
st.divider()

from modules.optimal_transport_stress import (
    OptimalTransportStressTester,
    CRISIS_SCENARIOS,
    plot_ot_stress_paths,
    plot_stress_sweep,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry Testu")
    ticker = st.text_input("Ticker portfela", value="SPY")
    period = st.selectbox("Okres historyczny", ["1y", "2y", "3y", "5y"], index=2)

    scenario_options = list(CRISIS_SCENARIOS.keys())
    scenario = st.selectbox("Scenariusz kryzysu", scenario_options, index=0)
    stress_level = st.slider(
        "Poziom stresu", 0.0, 1.0, 0.5, 0.05,
        help="0.0 = normalny rynek | 1.0 = pełny kryzys historyczny",
        format="%.2f",
    )
    n_sims = st.slider("Liczba symulacji", 200, 2000, 500, 100)
    initial_capital = st.number_input("Kapitał startowy (PLN)", 10000, 10000000, 100000, 10000)

    st.divider()
    run_sweep = st.checkbox("Uruchom sweep 0→1 stress level", value=False,
                            help="Oblicza metryki dla każdego poziomu stresu (wolniejsze)")

    st.divider()
    scen_info = CRISIS_SCENARIOS.get(scenario, {})
    st.markdown(f"**{scen_info.get('label', scenario)}**")
    st.markdown(f"*{scen_info.get('description', '')}*")
    st.markdown(f"Max Drawdown historyczny: **{scen_info.get('max_drawdown', 0):.1%}**")

# ─── Pobierz dane ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_returns(ticker: str, period: str) -> pd.Series:
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        closes = data["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        return closes.pct_change().dropna()
    except Exception as e:
        st.error(f"Błąd: {e}")
        return pd.Series(dtype=float)

with st.spinner("Pobieranie danych..."):
    returns = load_returns(ticker, period)

if returns.empty or len(returns) < 30:
    st.error("Niewystarczające dane.")
    st.stop()

# ─── Główna analiza ───────────────────────────────────────────────────────────
with st.spinner(f"Generuję scenariusz stresowy (stress_level={stress_level:.2f})..."):
    try:
        tester = OptimalTransportStressTester(n_simulations=n_sims, seed=42)
        stress_result = tester.generate_stress_returns(
            returns, crisis_scenario=scenario, stress_level=stress_level,
        )
        metrics = tester.compute_stress_metrics(stress_result, initial_capital=initial_capital)
    except Exception as e:
        st.error(f"Błąd stress test: {e}")
        st.stop()

# ─── Metryki ─────────────────────────────────────────────────────────────────
st.markdown("## 📊 Metryki Ryzyka")

cols = st.columns(4)
stress_m = metrics["stress"]
normal_m = metrics["normal"]
comp = metrics.get("comparison", {})

with cols[0]:
    var95 = stress_m.get("var_95", 0)
    norm95 = normal_m.get("var_95", 0)
    ratio = comp.get("var_95", {}).get("ratio", 1)
    border = "#e74c3c" if ratio > 2 else ("#f39c12" if ratio > 1.3 else "#00e676")
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {border};">
        <div style="color:#aaa;font-size:0.75rem;">VaR 95% (stress)</div>
        <div style="color:{border};font-size:1.6rem;font-weight:700;">{var95:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">Normal: {norm95:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">Amplifikacja: {ratio:.1f}×</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    cvar95 = stress_m.get("cvar_95", 0)
    norm_c95 = normal_m.get("cvar_95", 0)
    ratio_c = comp.get("cvar_95", {}).get("ratio", 1)
    border = "#e74c3c" if ratio_c > 2 else ("#f39c12" if ratio_c > 1.3 else "#00e676")
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {border};">
        <div style="color:#aaa;font-size:0.75rem;">CVaR 95% (stress)</div>
        <div style="color:{border};font-size:1.6rem;font-weight:700;">{cvar95:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">Normal: {norm_c95:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">Amplifikacja: {ratio_c:.1f}×</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    mdd = stress_m.get("max_drawdown_median", 0)
    border = "#e74c3c" if abs(mdd) > 0.3 else ("#f39c12" if abs(mdd) > 0.15 else "#00e676")
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {border};">
        <div style="color:#aaa;font-size:0.75rem;">Max Drawdown (mediana)</div>
        <div style="color:{border};font-size:1.6rem;font-weight:700;">{mdd:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">5.percentyl: {stress_m.get('max_drawdown_5pct', 0):.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    surv = stress_m.get("survival_rate", 0)
    border = "#00e676" if surv > 0.7 else ("#f39c12" if surv > 0.5 else "#e74c3c")
    shortfall = stress_m.get("shortfall_5pct", 0)
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {border};">
        <div style="color:#aaa;font-size:0.75rem;">Stopa Przeżycia (> kapitał)</div>
        <div style="color:{border};font-size:1.6rem;font-weight:700;">{surv:.1%}</div>
        <div style="color:#888;font-size:0.8rem;">5.pct wartość: {shortfall:,.0f} PLN</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Wizualizacja ścieżek ─────────────────────────────────────────────────────
st.markdown("## 📈 Ścieżki Portfela — Normal vs Stress")
try:
    fig_paths = plot_ot_stress_paths(
        stress_result,
        n_paths_to_show=min(100, n_sims),
        initial_capital=initial_capital,
    )
    st.plotly_chart(fig_paths, use_container_width=True)
except Exception as e:
    st.warning(f"Błąd wizualizacji: {e}")

# ─── Porównanie VaR99 szczegółowe ────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📋 Pełna Tabela Ryzyka")
    rows = []
    for cl_name, cl_key_s, cl_key_n in [
        ("VaR 95%", "var_95", "var_95"),
        ("VaR 99%", "var_99", "var_99"),
        ("CVaR 95%", "cvar_95", "cvar_95"),
        ("CVaR 99%", "cvar_99", "cvar_99"),
        ("Max DD (mediana)", "max_drawdown_median", "max_drawdown_median"),
        ("Stopa przeżycia", "survival_rate", "survival_rate"),
    ]:
        s_val = stress_m.get(cl_key_s, 0)
        n_val = normal_m.get(cl_key_n, 0)
        amp = abs(s_val) / (abs(n_val) + 1e-10) if abs(n_val) > 1e-6 else 1.0
        rows.append({
            "Metryka": cl_name,
            "Normal": f"{n_val:.2%}",
            "Stress": f"{s_val:.2%}",
            "Amplifikacja": f"{amp:.1f}×",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🎯 Interpretacja OT")
    scen = CRISIS_SCENARIOS.get(scenario, {})
    st.markdown(f"""
    **Scenariusz:** {scen.get('label', scenario)}
    **Poziom stresu:** {stress_level:.0%} → {"normalny" if stress_level < 0.2 else ("pół-kryzys" if stress_level < 0.6 else "pełny kryzys")}

    **Jak działa OT Stress Test:**

    1. Pobiera historyczne zwroty portfela → **rozkład normalny**
    2. Generuje zwroty z parametrów kryzysu → **rozkład kryzysowy**
    3. **Wasserstein barycentrum:** kwantylowa interpolacja:
       `q_stress(p) = (1-α)·q_normal(p) + α·q_crisis(p)`
    4. Wynik: realistyczny scenariusz stresowy z zachowaną strukturą ogonów

    **Przewaga nad klasycznym stress testem:**
    - Zachowuje asymetrię i grube ogony
    - Płynne przejście 0→1 (dial stress level)
    - Matematycznie udowodniona optymalność (Wasserstein)

    *Villani (2009), Blanchet & Murthy (2019)*
    """)

# ─── Sweep stress_level ───────────────────────────────────────────────────────
if run_sweep:
    st.divider()
    st.markdown("## 🔄 Sweep Poziomu Stresu 0% → 100%")
    with st.spinner("Obliczam metryki dla każdego poziomu stresu..."):
        try:
            sweep_df = tester.stress_level_sweep(
                returns, crisis_scenario=scenario, initial_capital=initial_capital,
            )
            fig_sweep = plot_stress_sweep(sweep_df, title=f"Metryki Ryzyka vs Stress Level — {scenario}")
            st.plotly_chart(fig_sweep, use_container_width=True)
            st.dataframe(
                sweep_df.round(4),
                use_container_width=True, hide_index=True
            )
        except Exception as e:
            st.warning(f"Błąd sweep: {e}")
