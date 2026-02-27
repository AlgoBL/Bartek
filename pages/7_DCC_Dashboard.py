"""
7_DCC_Dashboard.py — DCC-GARCH Dynamic Correlations Dashboard

Zawiera:
  - Rolling correlation heatmap (ewolucja w czasie)
  - Alert gdy korelacja > 0.80 (kontagion)
  - Pary asset correlation evolution chart
  - Math explainery DCC-GARCH
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer

st.set_page_config(page_title="DCC Correlacje", page_icon="🔗", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🔗 DCC-GARCH Dynamic Correlations")
st.markdown(
    "<p style='color:#6b7280;'>Dynamic Conditional Correlations — Engle (2002). "
    "Korelacje zmieniają się w czasie — klasyczna static corr matrix mija się z celem "
    "podczas krachów (kontagion).</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry DCC")
    tickers_input = st.text_area(
        "Tickery (po średniku)", value="SPY;TLT;GLD;BTC-USD", height=80
    )
    rolling_window = st.slider("Okno rolling correlation (dni)", 20, 120, 60)
    a_dcc  = st.slider("a_dcc — reakcja na szoki", 0.01, 0.20, 0.04, 0.01)
    b_dcc  = st.slider("b_dcc — persistence korelacji", 0.70, 0.99, 0.92, 0.01)
    n_days_sim = st.number_input("Dni symulacji (fallback)", 500, 2000, 750, 50)
    contagion_threshold = st.slider("Próg kontagion alert", 0.50, 0.95, 0.80, 0.05)
    crash_simul = st.checkbox("Symuluj crash (a_dcc ×3 przez 10 dni)", value=False)

tickers = [t.strip() for t in tickers_input.split(";") if t.strip()]
n_assets = len(tickers)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_multi_returns(ticker_list: list, n: int) -> pd.DataFrame:
    dfs = []
    for t in ticker_list:
        try:
            import yfinance as yf
            data = yf.download(t, period=f"{n//252+2}y", progress=False)["Adj Close"]
            if len(data) > 30:
                dfs.append(data.pct_change().dropna().rename(t))
        except Exception:
            pass
    if not dfs:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        data = {t: rng.standard_normal(n) * 0.012 for t in ticker_list}
        return pd.DataFrame(data, index=idx)
    df = pd.concat(dfs, axis=1).dropna()
    return df

with st.spinner("Ładowanie danych..."):
    returns_df = load_multi_returns(tickers, n_days_sim)

st.markdown(
    f"<div style='color:#6b7280;font-size:12px;'>"
    f"Dataset: {len(returns_df):,} dni | {len(returns_df.columns)} aktywów: "
    f"<b style='color:#00e676'>{', '.join(returns_df.columns.tolist())}</b></div>",
    unsafe_allow_html=True,
)

# ─── DCC-GARCH FIT ───────────────────────────────────────────────────────────
try:
    from modules.dcc_garch import DCCGARCHModel
    dcc = DCCGARCHModel(a_dcc=a_dcc, b_dcc=b_dcc)
    with st.spinner("DCC-GARCH fitting..."):
        dcc.fit(returns_df)
    R_series = dcc.get_conditional_correlations(returns_df)
    dcc_available = True
except Exception as e:
    st.warning(f"DCC-GARCH fit failed: {e} — używam rolling correlation")
    dcc_available = False

# ─── ROLLING CORRELATION (dla heatmap) ────────────────────────────────────
def rolling_mean_corr(df: pd.DataFrame, window: int) -> pd.Series:
    """Średnia korelacja portfela w rolling window."""
    daily_corrs = []
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        corr = sub.corr().values
        # Średnia off-diagonal
        n = corr.shape[0]
        if n < 2:
            daily_corrs.append(np.nan)
            continue
        mask = ~np.eye(n, dtype=bool)
        daily_corrs.append(corr[mask].mean())
    return pd.Series(daily_corrs, index=df.index[window:])

mean_corr_series = rolling_mean_corr(returns_df, rolling_window)

# ─── ROW 1: CONTAGION ALERT ──────────────────────────────────────────────────
current_corr = float(mean_corr_series.iloc[-1]) if len(mean_corr_series) > 0 else 0.0
is_contagion = current_corr > contagion_threshold

if is_contagion:
    st.markdown(f"""
    <div class='alert-badge-red' style='font-size:16px;padding:10px 24px;border-radius:12px;margin-bottom:12px;'>
        ⚠️ KONTAGION ALERT — Srednia korelacja portfela = {current_corr:.2f}
        (próg = {contagion_threshold:.2f}) — DYWERSYFIKACJA ZAWODZI!
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='alert-badge-green' style='font-size:14px;padding:8px 20px;border-radius:10px;margin-bottom:12px;'>
        ✓ Korelacja OK — {current_corr:.2f} (próg = {contagion_threshold:.2f})
    </div>""", unsafe_allow_html=True)

# ─── ROW 2: METRICS ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cols_metrics = [c1, c2, c3, c4]
metric_data = [
    ("Śr. korelacja (bieżąca)", f"{current_corr:.3f}", "#ff1744" if is_contagion else "#00e676"),
    ("Max korelacja (hist.)", f"{float(mean_corr_series.max()):.3f}", "#ffea00"),
    ("Min korelacja (hist.)", f"{float(mean_corr_series.min()):.3f}", "#00e676"),
    ("Liczba aktywów", f"{n_assets}", "#00ccff"),
]
for col, (lbl, val, col_v) in zip(cols_metrics, metric_data):
    with col:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-value' style='color:{col_v};'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ROW 3: ROLLING CORR TIMELINE ────────────────────────────────────────────
col_timeline, col_heatmap = st.columns([2, 1])

with col_timeline:
    st.markdown(f"#### Ewolucja średniej korelacji portfela (okno {rolling_window}d)")
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=mean_corr_series.index,
        y=mean_corr_series.values,
        fill="tozeroy",
        fillcolor="rgba(0,204,255,0.10)",
        line=dict(color="#00ccff", width=1.8),
        name="Śr. korelacja",
    ))
    fig_mc.add_hline(
        y=contagion_threshold, line_dash="dash", line_color="#ff1744",
        line_width=1.5,
        annotation_text=f"Próg kontagion ({contagion_threshold:.2f})",
        annotation_font_color="#ff1744",
    )
    fig_mc.add_hline(y=0, line_color="#333", line_width=1)
    fig_mc.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=340, yaxis_title="Korelacja",
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("🧮 Jak działa DCC-GARCH?"):
        st.markdown(math_explainer(
            "DCC-GARCH (Engle 2002)",
            "Q_t = (1-a-b)Q̄ + a·z_{t-1}z'_{t-1} + b·Q_{t-1}; R_t = D(Q_t)Q_tD(Q_t)",
            "Q̄ = długookresowa macierz korelacji. Parametr 'a' (reakcja) + 'b' (persistence). "
            "R_t = macierz korelacji warunkowych. W krachu: z_{t-1}z'_{t-1} rośnie → Q_t rośnie → "
            "korelacje rosną (kontagion). Standardowy model nie widzi tej zmiany.",
            "Engle (2002) — Dynamic Conditional Correlations; Engle & Sheppard (2001)",
        ), unsafe_allow_html=True)

with col_heatmap:
    st.markdown("#### Aktualny snapshot korelacji")
    try:
        if dcc_available and len(R_series) > 0:
            R_current = R_series[-1]
        else:
            R_current = returns_df.tail(rolling_window).corr().values

        col_names = returns_df.columns.tolist()
        fig_heat = go.Figure(go.Heatmap(
            z=np.round(R_current, 2),
            x=col_names,
            y=col_names,
            colorscale="RdYlGn_r",
            zmin=-1, zmax=1,
            text=np.round(R_current, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
        ))
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            font=dict(color="white", family="Inter", size=10),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")

st.divider()

# ─── ROW 4: PAIR CORRELATION EVOLUTION (DCC) ─────────────────────────────────
if dcc_available and n_assets >= 2 and len(R_series) > 0:
    st.markdown("#### DCC Korelacja par aktywów w czasie")
    pair_options = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
    pair_labels  = [f"{returns_df.columns[i]} / {returns_df.columns[j]}" for i, j in pair_options]
    selected_pairs = st.multiselect("Wybierz pary:", pair_labels, default=pair_labels[:min(2, len(pair_labels))])

    fig_pairs = go.Figure()
    colors = ["#00e676", "#00ccff", "#ffea00", "#ff1744", "#a855f7"]
    for idx, (pair_label) in enumerate(selected_pairs):
        pi_idx = pair_labels.index(pair_label)
        i, j = pair_options[pi_idx]
        corr_vals = [R[i, j] for R in R_series]
        fig_pairs.add_trace(go.Scatter(
            x=returns_df.index[:len(corr_vals)],
            y=corr_vals,
            mode="lines",
            line=dict(color=colors[idx % len(colors)], width=1.8),
            name=pair_label,
        ))
    fig_pairs.add_hline(
        y=contagion_threshold, line_dash="dash", line_color="#ff1744",
        line_width=1, annotation_text="Próg kontagion",
    )
    fig_pairs.add_hline(y=0, line_color="#333", line_width=1)
    fig_pairs.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=320, yaxis_title="Korelacja warunkowa R_t",
        legend=dict(orientation="h", y=-0.15),
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_pairs, use_container_width=True)

# ─── ROW 5: DCC SIMULATION (crash scenario) ──────────────────────────────────
if crash_simul and dcc_available:
    st.divider()
    st.markdown("#### 💥 Symulacja krachu — jak DCC wpływa na portfel?")
    crash_days = list(range(50, 65))  # crash po 50 dniach
    try:
        sim_paths = dcc.simulate_paths(
            n_sims=200, n_days=120,
            crash_regime_days=crash_days,
            crash_dcc_multiplier=3.0, seed=42
        )
        # Średni zwrot portfela equal-weight
        w_eq = np.ones(n_assets) / n_assets
        port_rets = sim_paths @ w_eq  # (n_sims, n_days)
        cum_rets  = np.cumprod(1 + port_rets, axis=1)
        p50 = np.percentile(cum_rets, 50, axis=0)
        p05 = np.percentile(cum_rets, 5,  axis=0)
        p95 = np.percentile(cum_rets, 95, axis=0)
        x = np.arange(120)

        fig_crash = go.Figure()
        fig_crash.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself", fillcolor="rgba(0,230,118,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="5-95 percentyl",
        ))
        fig_crash.add_trace(go.Scatter(x=x, y=p50, line=dict(color="#00e676", width=2), name="Mediana"))
        fig_crash.add_trace(go.Scatter(x=x, y=p05, line=dict(color="#ff1744", width=1, dash="dot"), name="P5 pesymistyczny"))
        for cd in crash_days[:1]:
            fig_crash.add_vline(x=cd, line_color="#ffea00", line_dash="dash",
                                annotation_text="Crash start", annotation_font_color="#ffea00")
        fig_crash.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.6)",
            height=320, yaxis_title="Skumulowany zwrot portfela",
            xaxis_title="Dzień",
            font=dict(color="white", family="Inter"),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_crash, use_container_width=True)
        st.caption("DCC amplifikuje korelacje podczas krachu → portfel traci bardziej niż zakładałby statyczny model")
    except Exception as e:
        st.error(f"Simualcja crash error: {e}")
