"""
53_HMM_Regime.py — Hidden Markov Model Regime Detection
Strona Streamlit dla analizy reżimów rynkowych przez HMM.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="HMM Regime Detection",
    page_icon="🎲",
    layout="wide",
)

# CSS
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
    .regime-bull   { border-left: 4px solid #00e676; }
    .regime-normal { border-left: 4px solid #3498db; }
    .regime-caution{ border-left: 4px solid #f39c12; }
    .regime-bear   { border-left: 4px solid #e74c3c; }
    h1 { color: #e2e4f0; font-family: 'Inter', sans-serif; font-size: 1.6rem; }
    h2, h3 { color: #b0b4cc; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎲 Hidden Markov Model — Detekcja Reżimów")
st.markdown("""
Gaussian HMM identyfikuje **ukryte stany** rynku (Bull/Normal/Caution/Bear) z szeregu zwrotów.
W odróżnieniu od prostych progów na VIX, HMM zwraca **probabilistyczne** przypisanie do reżimu.

*Hamilton (1989), Hamilton (1994)*
""")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry HMM")
    ticker = st.text_input("Ticker", value="SPY", help="Ticker z Yahoo Finance")
    period = st.selectbox("Okres danych", ["2y", "3y", "5y", "10y"], index=2)
    n_regimes = st.slider("Liczba reżimów (K)", 2, 4, 3,
                          help="K=2: Bull/Bear | K=3: Bull/Normal/Bear | K=4: +Caution")
    n_iter = st.slider("Max. iteracje EM", 50, 300, 100)
    use_hmmlearn = st.checkbox("Użyj hmmlearn (jeśli dostępny)", value=True)

    st.divider()
    st.markdown("### 📖 Teoria")
    with st.expander("Hamilton (1989) — HMM"):
        st.markdown("""
**Obserwacje:** r_t = μ_{S_t} + σ_{S_t}·ε_t

**Stany ukryte:** S_t ∈ {0,..,K-1}

**Macierz przejść:** A_{ij} = P(S_t=j | S_{t-1}=i)

**Estymacja:** Baum-Welch EM
(Forward-Backward Algorithm)

**Dekodowanie:** Algorytm Viterbiego
→ najbardziej prawdopodobna sekwencja
        """)

# ─── Główna logika ────────────────────────────────────────────────────────────
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
        st.error(f"Błąd pobierania danych: {e}")
        return pd.Series(dtype=float)


def run_hmm(returns: pd.Series, n_regimes: int, n_iter: int) -> tuple:
    from modules.hmm_regime import HMMRegimeDetector, plot_hmm_regimes, plot_hmm_transition_matrix
    hmm = HMMRegimeDetector(n_regimes=n_regimes, n_iter=n_iter)
    hmm.fit(returns)
    return hmm


with st.spinner("Pobieranie danych..."):
    returns = load_returns(ticker, period)

if returns.empty or len(returns) < 60:
    st.error("Niewystarczające dane. Zmień ticker lub okres.")
    st.stop()

st.success(f"✅ Załadowano {len(returns)} obserwacji dla **{ticker}**")

with st.spinner(f"Trenuję HMM z K={n_regimes} reżimami..."):
    try:
        from modules.hmm_regime import HMMRegimeDetector, plot_hmm_regimes, plot_hmm_transition_matrix
        hmm = run_hmm(returns, n_regimes, n_iter)
    except Exception as e:
        st.error(f"Błąd HMM: {e}")
        st.stop()

# ─── Bieżący reżim ────────────────────────────────────────────────────────────
current = hmm.get_current_regime(returns)
summary = hmm.regime_summary()

st.markdown("## 🎯 Aktualny Reżim Rynkowy")
cols = st.columns([2, 1, 1, 1])
with cols[0]:
    color = current["color"]
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div style="color:#aaa;font-size:0.8rem;">AKTUALNY REŻIM HMM</div>
        <div style="color:{color};font-size:2rem;font-weight:700;">{current['label']}</div>
        <div style="color:#888;font-size:0.9rem;">Pewność: <b>{current['confidence']:.1%}</b></div>
        <div style="color:#888;font-size:0.85rem;">μ_ann = {current['mu_annual']:.1%} | σ_ann = {current['sigma_annual']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

for i, reg in enumerate(summary["regimes"]):
    with cols[i + 1]:
        col = reg["color"]
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {col};">
            <div style="color:#aaa;font-size:0.75rem;">{reg['label']}</div>
            <div style="color:{col};font-size:1.4rem;font-weight:600;">
                {current['probabilities'].get(reg['label'], 0):.1%}
            </div>
            <div style="color:#888;font-size:0.75rem;">μ={reg['mu_annual']:.1%} σ={reg['sigma_annual']:.1%}</div>
            <div style="color:#888;font-size:0.75rem;">Trwałość: {reg['expected_duration_days']:.0f}d</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── Główny wykres ────────────────────────────────────────────────────────────
st.markdown("## 📊 Wizualizacja Reżimów HMM")
try:
    fig_main = plot_hmm_regimes(returns, hmm, title=f"HMM Regime Detection — {ticker} (K={n_regimes})")
    st.plotly_chart(fig_main, use_container_width=True)
except Exception as e:
    st.warning(f"Błąd wizualizacji: {e}")

# ─── Macierz przejść ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 🔄 Macierz Przejść A")
    try:
        fig_trans = plot_hmm_transition_matrix(hmm)
        st.plotly_chart(fig_trans, use_container_width=True)
    except Exception as e:
        st.warning(f"Błąd: {e}")

with col2:
    st.markdown("### 📋 Parametry Reżimów")
    rows = []
    for reg in summary["regimes"]:
        rows.append({
            "Reżim": reg["label"],
            "μ_roczny": f"{reg['mu_annual']:.1%}",
            "σ_roczny": f"{reg['sigma_annual']:.1%}",
            "Sharpe": f"{reg['sharpe_proxy']:.2f}",
            "Trwałość (dni)": f"{reg['expected_duration_days']:.0f}",
            "P(persistencja)": f"{reg['persistence']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 📐 Interpretacja")
    st.markdown(f"""
    **Log-Likelihood modelu:** `{summary['log_likelihood']:.2f}`

    **Jak interpretować:**
    - **Wysoka trwałość** (>0.97) = reżim utrzymuje się tygodniami
    - **Niska trwałość** (<0.90) = częste przejścia między reżimami
    - **Sharpe > 1.0** w reżimie = historycznie korzystny dla długich pozycji
    - **σ > 25%** roczna = reżim podwyższonej zmienności

    *Hamilton (1989) Econometrica 57(2)*
    """)

# ─── Porównanie z prostymi progami ────────────────────────────────────────────
st.divider()
st.markdown("## 🔬 HMM vs. Proste Progi VIX")
with st.expander("Pokaż porównanie metodologiczne"):
    cols_cmp = st.columns(2)
    with cols_cmp[0]:
        st.markdown("""
        **Proste progi (dotychczasowe):**
        - VIX > 20 → "Stres"
        - VIX > 30 → "Kryzys"
        - Binarne (0/1), brak niepewności
        - Podatne na fałszywe sygnały
        - Parametry arbitralne (hardcodowane)
        """)
    with cols_cmp[1]:
        st.markdown("""
        **HMM (nowe):**
        - Automatycznie wykrywa K ukrytych stanów
        - Zwraca P(reżim | dane) ∈ [0, 1]
        - Uwzględnia dynamikę przejść (persistence)
        - Parametry estymowane z danych (EM)
        - Matematycznie rygorystyczne (Baum-Welch)
        """)
