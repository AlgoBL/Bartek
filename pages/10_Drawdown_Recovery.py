"""10_Drawdown_Recovery.py — Analiza Czasu Odrobienia Strat"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from modules.styling import apply_styling
from modules.drawdown_recovery_analyzer import (
    underwater_analysis, sequence_of_returns_risk,
    breakeven_calculator, recovery_probability_mc,
)

st.set_page_config(page_title="Drawdown Recovery Analyzer", page_icon="📉", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker, period="10y"):
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        elif "Close" in raw.columns:
            closes = raw["Close"]
        else:
            closes = raw.iloc[:, 0]
        # Always return 1-D Series
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        return closes.dropna()
    except Exception:
        return None

st.markdown("# 📉 Drawdown Recovery Analyzer")
st.markdown("*Underwater periods, sequence-of-returns risk i prawdopodobieństwo odrobienia strat*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    ticker = st.text_input("Ticker (benchmark/portfel)", value="SPY")
    period = st.selectbox("Okres historyczny", ["5y", "10y", "15y", "20y"], index=1)
    initial_capital = st.number_input("Kapitał startowy (PLN)", value=100_000, step=10_000)
    withdrawal = st.number_input("Roczna wypłata (PLN)", value=4_000, step=1_000)
    horizon_mc = st.slider("Horyzont Monte Carlo (lata)", 3, 20, 7)
    current_dd = st.slider("Bieżący drawdown (%)", -60, 0, -15) / 100

with st.spinner("Pobieranie danych..."):
    prices = load_data(ticker, period)

if prices is None or (hasattr(prices, "empty") and prices.empty):
    st.error("Brak danych.")
    st.stop()

# Guarantee 1-D Series
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]
prices = prices.squeeze()

returns = prices.pct_change().dropna()
if isinstance(returns, pd.DataFrame):
    returns = returns.iloc[:, 0]
returns = returns.squeeze()


tab1, tab2, tab3, tab4 = st.tabs(["🌊 Underwater Periods", "🎲 Sequence Risk", "⚖️ Break-Even", "🔮 MC Recovery"])

with tab1:
    uw = underwater_analysis(prices)
    dd_s = uw.get("dd_series", pd.Series())
    periods_df = uw.get("drawdown_periods", pd.DataFrame())
    summary = uw.get("summary", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("N Drawdownów", summary.get("n_drawdowns", 0))
    with c2:
        st.metric("Max DD", f"{summary.get('max_depth', 0):.1%}")
    with c3:
        st.metric("Avg DD Duration", f"{summary.get('avg_duration', 0):.0f} dni")
    with c4:
        st.metric("Avg Recovery", f"{summary.get('avg_recovery', 0):.0f} dni")

    # Underwater chart
    if not dd_s.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd_s.index, y=dd_s * 100,
            fill="tozeroy", fillcolor="rgba(255,23,68,0.12)",
            line=dict(color="#ff1744", width=1.5),
        ))
        fig.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Drawdown (%)", title=f"Underwater Chart — {ticker}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not periods_df.empty:
        display_df = periods_df[["depth", "duration_days", "recovery_days", "total_days", "recovered"]].copy()
        display_df.columns = ["Głębokość", "Czas DD (dni)", "Czas recovery (dni)", "Łącznie (dni)", "Odrobione?"]
        display_df["Głębokość"] = display_df["Głębokość"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df.head(15), use_container_width=True)

with tab2:
    # Annual returns from period
    annual_r = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1).dropna()
    # Squeeze in case resample returns DataFrame with single col
    if isinstance(annual_r, pd.DataFrame):
        annual_r = annual_r.iloc[:, 0]
    annual_r = annual_r.squeeze()
    ann_list = list(annual_r)

    if len(ann_list) >= 3:
        seq_res = sequence_of_returns_risk(ann_list, initial_capital, withdrawal)
        c1, c2, c3 = st.columns(3)
        c1.metric("Najlepszy scenariusz", f"{seq_res.get('best_final', 0):,.0f} PLN")
        c2.metric("Najgorszy scenariusz", f"{seq_res.get('worst_final', 0):,.0f} PLN")
        c3.metric("CAGR (wspólny)", f"{seq_res.get('cagr', 0):.1%}")
        impact = seq_res.get("sequence_impact", 1)
        st.warning(f"⚡ **Sequence Impact Ratio: {impact:.1f}×** — ten sam CAGR, różna kolejność → {impact:.1f}× różnica w wyniku końcowym!")

        # Chart: original vs reversed paths
        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(y=seq_res.get("original_path", []), name="Oryginalna kolejność", line=dict(color="#00e676")))
        fig_seq.add_trace(go.Scatter(y=seq_res.get("reversed_path", []), name="Odwrócona kolejność", line=dict(color="#ff1744")))
        for path in seq_res.get("sample_paths", [])[:10]:
            fig_seq.add_trace(go.Scatter(y=path, line=dict(color="rgba(100,100,200,0.15)", width=0.5), showlegend=False))
        fig_seq.add_hline(y=initial_capital, line_dash="dash", line_color="white", annotation_text="Kapitał startowy")
        fig_seq.update_layout(
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Wartość portfela (PLN)", title="Ryzyko kolejności stóp zwrotu",
        )
        st.plotly_chart(fig_seq, use_container_width=True)
    else:
        st.info("Za mało danych rocznych do analizy sekwencji.")

with tab3:
    be = breakeven_calculator()
    df_be = be.get("table", pd.DataFrame())
    st.markdown("### ⚖️ Ile potrzeba zarobić żeby odrobić stratę?")

    fig_be = go.Figure()
    fig_be.add_trace(go.Bar(
        x=df_be["Strata"], y=[v * 100 for v in df_be["required_gain"].tolist()],
        marker_color=[
            "#00e676" if v < 0.15 else "#ffea00" if v < 0.40 else "#ff1744"
            for v in df_be["required_gain"].tolist()
        ],
        text=[f"{v:.1%}" for v in df_be["required_gain"].tolist()],
        textposition="outside",
    ))
    fig_be.update_layout(
        template="plotly_dark", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Wymagany zysk (%)", title="Break-Even Return po Stracie",
    )
    st.plotly_chart(fig_be, use_container_width=True)
    st.dataframe(
        df_be[["Strata", "Zwrot do BE", "Lata (CAGR=10%)", "Lata (CAGR=7%)", "Effort Ratio"]],
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    if current_dd < 0:
        mc_res = recovery_probability_mc(current_dd, returns, horizon_years=horizon_mc, n_sims=3000)
        if "error" not in mc_res:
            p_full = mc_res.get("prob_recovery_full", 0)
            p_half = mc_res.get("prob_recovery_50pct", 0)
            median_t = mc_res.get("median_years_to_recovery", np.inf)

            c1, c2, c3 = st.columns(3)
            pc = "#00e676" if p_full > 0.7 else "#ffea00" if p_full > 0.4 else "#ff1744"
            c1.markdown(f"""<div class="metric-card"><div class="metric-label">P(pełne odrobienie)</div>
                <div class="metric-value" style="color:{pc}">{p_full:.1%}</div>
                <div style="color:#6b7280">w {horizon_mc} latach</div></div>""", unsafe_allow_html=True)
            c2.metric(f"P(odrobienie 50%)", f"{p_half:.1%}")
            c3.metric("Mediana czasu", f"{median_t:.1f} lat" if median_t < 50 else "Brak danych")

            if p_full < 0.5:
                st.error(f"🔴 Przy drawdown {current_dd:.0%} istnieje mniej niż 50% szans na pełne odrobienie w {horizon_mc} lat.")
            elif p_full < 0.75:
                st.warning(f"⚠️ Przy drawdown {current_dd:.0%}: {p_full:.0%} szans na pełne odrobienie w {horizon_mc} lat.")
            else:
                st.success(f"✅ Wysoka szansa ({p_full:.0%}) na odrobienie drawdown {current_dd:.0%} w {horizon_mc} lat.")
        else:
            st.warning(mc_res.get("error", "Błąd"))
    else:
        st.info("Ustaw bieżący drawdown < 0% w bocznym panelu.")
