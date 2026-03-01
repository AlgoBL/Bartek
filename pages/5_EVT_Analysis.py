"""
5_EVT_Analysis.py — Strona Extreme Value Theory & Tail Risk Diagnostics

Zawiera:
  - QQ-Plot GPD (zweryfikowanie dopasowania ogona)
  - Mean Excess Function (MEF) plot — diagnostyka progue
  - Tabela EVT-VaR vs Historical-VaR vs Parametric-VaR
  - Math explainery dla każdego elementu
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer
from modules.risk_manager import RiskManager

st.set_page_config(page_title="EVT Tail Risk", page_icon="📐", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 📐 EVT Tail Risk Analysis")
st.markdown(
    "<p style='color:#6b7280;'>Peaks-Over-Threshold (POT) — Generalized Pareto Distribution "
    "| Balkema-de Haan (1974) + McNeil & Frey (2000)</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── SIDEBAR CONTROLS ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    ticker = st.text_input("Ticker (zwroty z Symulatora lub wpisz ręcznie)", value="SPY")
    threshold_pct = st.slider("Próg POT (percentyl strat)", 0.85, 0.99, 0.95, 0.01,
                              help="Im wyższy próg, tym mniej danych — EVT działa na skrajnym ogonie")
    confidence_levels = st.multiselect(
        "Poziomy ufności VaR/CVaR",
        [0.90, 0.95, 0.99, 0.999],
        default=[0.95, 0.99, 0.999],
    )
    n_days = st.number_input("Dni historycznych (symulowane)", 500, 5000, 1500, 100)
    seed = st.number_input("Seed symulacji", 0, 9999, 42)

# ─── DATA GENERATION (fallback: simulated t-distribution) ─────────────────
@st.cache_data(ttl=1800)
def load_returns(ticker_sym: str, n: int, seed_val: int) -> pd.Series:
    try:
        import yfinance as yf
        data = yf.download(ticker_sym, period=f"{n//252+3}y", progress=False)["Adj Close"]
        if len(data) > 0:
            rets = data.pct_change().dropna()
            return rets
    except Exception:
        pass
    rng = np.random.default_rng(seed_val)
    # t(5) — heavy tail proxy
    return pd.Series(rng.standard_t(5, n) * 0.012, name=ticker_sym)

with st.spinner("Ładowanie danych..."):
    returns = load_returns(ticker, n_days, seed)

st.markdown(
    f"<div style='color:#6b7280;font-size:12px;margin-bottom:8px;'>"
    f"Dataset: <b style='color:#00e676'>{ticker}</b> — {len(returns):,} obserwacji "
    f"| Std: {returns.std()*100:.2f}%/dzień | Skewness: {returns.skew():.2f}"
    f"</div>",
    unsafe_allow_html=True,
)

rm = RiskManager()

# ─── FIT EVT ─────────────────────────────────────────────────────────────────
evt = rm.fit_evt_pot(returns, threshold_pct=threshold_pct)

if "error" in evt:
    st.error(f"EVT Fit Error: {evt['error']}")
    st.stop()

xi = evt["xi"]
sigma = evt["sigma"]
u = evt["threshold"]
excesses = evt["excesses"]

# ─── ROW 1: KEY METRICS ──────────────────────────────────────────────────────
tail_info = rm.evt_full_metrics(returns, threshold_pct)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Shape ξ (Tail Index)</div>
        <div class='metric-value' style='color:{"#ff1744" if xi > 0.3 else "#00e676"}'>{xi:.4f}</div>
        <div style='font-size:10px;color:#6b7280;'>{"Heavy tail" if xi > 0 else "Thin tail"}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Scale σ</div>
        <div class='metric-value'>{sigma:.4f}</div>
        <div style='font-size:10px;color:#6b7280;'>Skala ogona GPD</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Próg u ({threshold_pct*100:.0f}%)</div>
        <div class='metric-value'>{u*100:.2f}%</div>
        <div style='font-size:10px;color:#6b7280;'>N_u = {evt["N_u"]} / {evt["N_total"]}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    tail_type = tail_info.get("tail_type", "N/A")
    tail_short = tail_type.split("—")[0].strip()
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Typ Ogona</div>
        <div class='metric-value' style='font-size:14px;'>{tail_short}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ROW 2: QQ-PLOT + MEF ────────────────────────────────────────────────────
col_qq, col_mef = st.columns(2)

with col_qq:
    st.markdown("#### GPD QQ-Plot — weryfikacja dopasowania ogona")
    try:
        from scipy.stats import genpareto
        n_exc = len(excesses)
        quantiles_th = (np.arange(1, n_exc+1) - 0.5) / n_exc
        theoretical = genpareto.ppf(quantiles_th, c=xi, scale=sigma)
        emp_sorted = np.sort(excesses)
        lim = max(theoretical.max(), emp_sorted.max()) * 1.05

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical, y=emp_sorted, mode="markers",
            marker=dict(color="#00e676", size=5, opacity=0.7),
            name="Dane empiryczne",
        ))
        fig_qq.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color="#ff1744", dash="dash", width=1.5),
            name="Idealne dopasowanie (45°)",
        ))
        fig_qq.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.6)",
            xaxis_title="Kwantyl GPD (teoretyczny)",
            yaxis_title="Kwantyl empiryczny",
            height=360, font=dict(color="white", family="Inter"),
            legend=dict(font=dict(size=10)),
            margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    except Exception as e:
        st.error(f"QQ-Plot error: {e}")

    with st.expander("🧮 Co oznacza QQ-Plot?"):
        st.markdown(math_explainer(
            "GPD QQ-Plot",
            "Q_emp(p) vs Q_GPD(p; ξ, σ) — kwantyle empiryczne vs teoretyczne",
            "Jeśli punkty leżą blisko linii 45°, GPD dobrze opisuje ogon. "
            "Odchylenia w prawym górnym rogu = ekstremalnie grube ogony (ξ niedoszacowane).",
            "Coles (2001) — An Introduction to Statistical Modeling of Extreme Values",
        ), unsafe_allow_html=True)

with col_mef:
    st.markdown("#### Mean Excess Function — optymalny próg")
    try:
        thresholds, mean_exc = rm.mean_excess_plot_data(returns, n_thresholds=30)
        fig_mef = go.Figure()
        fig_mef.add_trace(go.Scatter(
            x=thresholds * 100, y=mean_exc * 100,
            mode="lines+markers",
            line=dict(color="#00ccff", width=2),
            marker=dict(size=5, color="#00ccff"),
            name="Mean Excess",
        ))
        # Mark chosen threshold
        fig_mef.add_vline(
            x=u * 100, line_dash="dash", line_color="#ffea00",
            annotation_text=f"Wybrany próg: {u*100:.2f}%",
            annotation_font_color="#ffea00",
        )
        fig_mef.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.6)",
            xaxis_title="Próg u (% strata)",
            yaxis_title="Średnia nadwyżka E[X-u | X>u] (%)",
            height=360, font=dict(color="white", family="Inter"),
            margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_mef, use_container_width=True)
    except Exception as e:
        st.error(f"MEF error: {e}")

    with st.expander("🧮 Jak wybrać optymalny próg?"):
        st.markdown(math_explainer(
            "Mean Residual Life Plot",
            "e(u) = E[X - u | X > u] — powinno być liniowe dla GPD",
            "Wybierz próg u gdzie MEF staje się w przybliżeniu liniowa i rosnąca. "
            "Wzrastająca e(u) potwierdza heavy tail (ξ > 0). Wybór za niskiego progu "
            "= zbyt mało danych do asymptotycznego przybliżenia GPD.",
            "Embrechts, Klüppelberg & Mikosch (1997)",
        ), unsafe_allow_html=True)

st.divider()

# ─── ROW 3: VaR / CVaR COMPARISON TABLE ────────────────────────────────────
st.markdown("#### 📊 Porównanie metod — EVT vs Historical vs Parametric Normal")

rows = []
losses = (-returns.dropna()).values
for conf in confidence_levels:
    # EVT
    var_evt   = rm.evt_var(evt, conf)
    cvar_evt  = rm.evt_cvar(evt, conf)
    # Historical
    var_hist  = float(np.percentile(losses, conf * 100))
    exc_hist  = losses[losses > var_hist]
    cvar_hist = float(exc_hist.mean()) if len(exc_hist) > 0 else var_hist
    # Parametric (Normal)
    from scipy.stats import norm
    mu_l, sigma_l = losses.mean(), losses.std()
    var_norm  = float(norm.ppf(conf, loc=mu_l, scale=sigma_l))
    cvar_norm = float(mu_l + sigma_l * norm.pdf(norm.ppf(conf)) / (1 - conf))

    rows.append({
        "Confd.": f"{conf*100:.1f}%",
        "EVT VaR":     f"{var_evt*100:.2f}%",
        "Hist VaR":    f"{var_hist*100:.2f}%",
        "Norm VaR":    f"{var_norm*100:.2f}%",
        "EVT CVaR":    f"{cvar_evt*100:.2f}%" if cvar_evt != float('inf') else "∞",
        "Hist CVaR":   f"{cvar_hist*100:.2f}%",
        "EVT vs Hist": f"{(var_evt-var_hist)*100:+.2f}%",
    })

df_comp = pd.DataFrame(rows)
st.dataframe(
    df_comp,
    use_container_width=True,
    hide_index=True,
)

with st.expander("🧮 Dlaczego EVT daje wyższe wartości niż Normal?"):
    st.markdown(math_explainer(
        "EVT vs Normal VaR",
        "VaR_EVT = u + (σ/ξ)·[(N/N·(1-p))^{-ξ} - 1]",
        "Model Normalny zakłada 'thin tails' (Gaussian). EVT modeluje empicznie zmierzone "
        "grube ogony. Dla aktywów finansowych ξ > 0 (Pareto), stąd EVT VaR > Normal VaR. "
        "Niedoszacowanie przez Normal to 'Model Risk' (główna przyczyna krachów 2008).",
        "McNeil & Frey (2000); Danielsson & de Vries (1997)",
    ), unsafe_allow_html=True)
