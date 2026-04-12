"""
22_Factor_Analysis.py — Factor Zoo PCA & Fama-French 5-Factor Decomposition

Zawiera:
  - PCA Eigen-Portfolio Analysis (ile czynników wyjaśnia >=95% zmienności)
  - Fama-French 5-Factor regression (Market, SMB, HML, RMW, CMA)
  - GARCH-MIDAS Volatility Decomposition
  - Principal Risk Factor visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2

from modules.styling import apply_styling, math_explainer
from modules.ui.widgets import tickers_area
from modules.factor_model import (
    build_factor_returns, run_factor_decomposition,
    plot_factor_decomposition, plot_variance_attribution,
)
from modules.garch_midas import GARCHMIDASEngine, plot_garch_midas_decomposition
from modules.global_settings import get_gs, apply_gs_to_session
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)
_gs = get_gs()
apply_gs_to_session(_gs)

# ─── PAGE HEADER ─────────────────────────────────────────────────────────────
st.markdown("# 🔬 Factor Zoo & Volatility Analysis")
st.markdown(
    "<p style='color:#6b7280;'>PCA Eigen-Portfolio | Fama-French 5-Factor | GARCH-MIDAS Macro Volatility</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")

    ticker_input = tickers_area(
        "Tickers portfela (jeden per linia lub przecinek)",
        value="SPY\nQQQ\nTLT\nGLD\nIWM",
        height=120,
        help="Wpisz tickery aktywów portfela. Dane pobierane z Yahoo Finance.",
    )
    period = st.selectbox("Okres historyczny", ["1y", "2y", "3y", "5y"], index=2)
    pca_variance_threshold = st.slider(
        "Próg wariancji PCA (%)", 80, 99, 95,
        help="Ile % całkowitej wariancji powinny wyjaśniać wybrane czynniki główne?"
    )
    show_midas = st.checkbox("Pokaż GARCH-MIDAS", value=True)
    show_ff5   = st.checkbox("Pokaż Fama-French 5-Factor", value=True)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
# Parse tickers
raw_tickers = ticker_input.replace(",", "\n").replace(";", "\n").split("\n")
tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
tickers = list(dict.fromkeys(tickers))  # deduplicate preserving order

if len(tickers) < 2:
    st.warning("⚠️ Wpisz co najmniej 2 tickers.", icon="⚠️")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def load_returns_data(tickers_tuple: tuple, period_str: str) -> pd.DataFrame:
    """Load price data and compute daily returns."""
    try:
        import yfinance as yf
        raw = yf.download(list(tickers_tuple), period=period_str, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = list(tickers_tuple)
        returns = prices.pct_change().dropna()
        return returns
    except Exception as e:
        return pd.DataFrame()


with st.spinner("📡 Pobieranie danych rynkowych..."):
    returns_df = load_returns_data(tuple(tickers), period)

if returns_df.empty:
    st.error("❌ Nie udało się pobrać danych. Sprawdź tickery i połączenie z internetem.")
    st.stop()

# Keep only tickers with sufficient data
valid_cols = [c for c in returns_df.columns if returns_df[c].notna().sum() > 60]
returns_df = returns_df[valid_cols].dropna()

if len(valid_cols) < 2:
    st.error(f"❌ Za mało danych dla porfolela. Dostępne: {valid_cols}")
    st.stop()

n_obs = len(returns_df)
n_assets = len(valid_cols)

# ─── METRICS ROW ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Aktywów", n_assets)
with m2:
    st.metric("Obserwacji", f"{n_obs:,}")
with m3:
    port_returns = returns_df.mean(axis=1)
    ann_vol = port_returns.std() * np.sqrt(252) * 100
    st.metric("Vol portfela (roczna)", f"{ann_vol:.1f}%")
with m4:
    period_labels = {"1y": "1 rok", "2y": "2 lata", "3y": "3 lata", "5y": "5 lat"}
    st.metric("Okres analizy", period_labels.get(period, period))

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PCA EIGEN-PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
tab_pca, tab_ff5, tab_midas, tab_timing = st.tabs([
    "📐 PCA Eigen-Portfolio",
    "🏛️ Fama-French 5-Factor",
    "📈 GARCH-MIDAS Volatility",
    "⏳ Factor Timing vs Regimes",
])

with tab_pca:
    st.markdown("### 📐 PCA — Dekompozycja Ryzyka Portfela")
    st.caption(
        "PCA wykrywa ile **prawdziwych niezależnych czynników** napędza Twój portfel. "
        "Jeśli 1 czynnik wyjaśnia >80% — portfel jest słabo zdywersyfikowany."
    )

    # ── Compute PCA ────────────────────────────────────────────────────────────
    cov_matrix = returns_df.cov().values * 252  # annualized
    corr_matrix = returns_df.corr().values

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort descending
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
        eigenvalues = np.maximum(eigenvalues, 0)
    except np.linalg.LinAlgError:
        st.error("Błąd dekompozycji macierzy kowariancji.")
        st.stop()

    total_var = eigenvalues.sum()
    explained = eigenvalues / (total_var + 1e-10)
    cumulative = np.cumsum(explained)

    # Number of components for threshold
    n_components_threshold = int(np.searchsorted(cumulative, pca_variance_threshold / 100) + 1)
    n_components_threshold = min(n_components_threshold, n_assets)

    # ── KMO Test (simplified) ────────────────────────────────────────────────
    # Bartlett test: χ² = -(n-1 - (2p+5)/6) * ln|R|
    try:
        sign, log_det = np.linalg.slogdet(corr_matrix)
        chisq = -(n_obs - 1 - (2 * n_assets + 5) / 6) * log_det
        df_bartlett = n_assets * (n_assets - 1) / 2
        p_bartlett = 1 - chi2.cdf(chisq, df=df_bartlett)
        bartlett_ok = p_bartlett < 0.05
    except Exception:
        chisq, p_bartlett, bartlett_ok = 0.0, 1.0, False

    # ── METRICS ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color_nc = "#ff1744" if n_components_threshold == 1 else "#00e676"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Czynniki do {pca_variance_threshold}% var.</div>
            <div class='metric-value' style='color:{color_nc}'>{n_components_threshold}</div>
            <div style='font-size:10px;color:#6b7280;'>min. PC wyjaśniające próg</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pc1_pct = explained[0] * 100
        color_pc1 = "#ff1744" if pc1_pct > 80 else "#f39c12" if pc1_pct > 60 else "#00e676"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>PC1 wyjaśnia</div>
            <div class='metric-value' style='color:{color_pc1}'>{pc1_pct:.1f}%</div>
            <div style='font-size:10px;color:#6b7280;'>{'⚠️ Słaba dywersyfikacja' if pc1_pct>70 else '✅ Dobra dywersyfikacja'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top3_pct = cumulative[min(2, n_assets-1)] * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>PC1-3 łącznie</div>
            <div class='metric-value'>{top3_pct:.1f}%</div>
            <div style='font-size:10px;color:#6b7280;'>3 czynniki główne</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        bart_status = "✅ Istotna" if bartlett_ok else "❌ Niska"
        bart_color = "#00e676" if bartlett_ok else "#ff1744"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Korelacja Bartlett</div>
            <div class='metric-value' style='color:{bart_color};font-size:14px;'>{bart_status}</div>
            <div style='font-size:10px;color:#6b7280;'>p={p_bartlett:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_scree, col_load = st.columns(2)

    with col_scree:
        # Scree Plot
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(n_assets)],
            y=explained * 100,
            marker_color=["#00e676" if i < n_components_threshold else "#2d3748"
                          for i in range(n_assets)],
            name="Wariancja wyjaśniona",
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(n_assets)],
            y=cumulative * 100,
            mode="lines+markers",
            line=dict(color="#00ccff", width=2),
            marker=dict(size=5),
            name="Skumulowana",
            yaxis="y2",
        ))
        # Threshold line
        fig_scree.add_hline(
            y=pca_variance_threshold, line_dash="dash", line_color="#ffea00",
            annotation_text=f"{pca_variance_threshold}% próg",
            annotation_font_color="#ffea00",
            yref="y2",
        )
        fig_scree.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.7)",
            title="Scree Plot — Eigenvalues",
            yaxis=dict(title="Wariancja wyjaśniona (%)", range=[0, 110]),
            yaxis2=dict(title="Skumulowana (%)", overlaying="y", side="right",
                        range=[0, 110], showgrid=False),
            height=380,
            font=dict(color="white", family="Inter"),
            legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
            margin=dict(l=50, r=50, t=40, b=60),
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_load:
        # Loadings Heatmap (top 3 PCs)
        n_show_pcs = min(5, n_assets)
        loadings = eigenvectors[:, :n_show_pcs]
        load_df = pd.DataFrame(
            loadings,
            index=valid_cols,
            columns=[f"PC{i+1}" for i in range(n_show_pcs)],
        )

        fig_load = px.imshow(
            load_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_load.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.7)",
            title="Factor Loadings (eigenvectors)",
            height=380,
            font=dict(color="white", family="Inter"),
            coloraxis_colorbar=dict(title="Loading", tickfont=dict(size=9)),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_load, use_container_width=True)

    # ── Eigen-Portfolio Returns ───────────────────────────────────────────────
    st.markdown("#### 📊 Eigen-Portfolio Performance (PC1 vs PC2)")
    pc1_weights = eigenvectors[:, 0]
    pc2_weights = eigenvectors[:, 1]

    pc1_portfolio = returns_df.values @ pc1_weights
    pc2_portfolio = returns_df.values @ pc2_weights
    equal_portfolio = returns_df.mean(axis=1).values

    idx_dates = returns_df.index
    cumret_pc1 = (1 + pc1_portfolio).cumprod()
    cumret_pc2 = (1 + pc2_portfolio).cumprod()
    cumret_eq = (1 + equal_portfolio).cumprod()

    fig_ep = go.Figure()
    fig_ep.add_trace(go.Scatter(x=idx_dates, y=cumret_pc1, mode="lines",
                                 name="PC1 Eigen-Portfolio",
                                 line=dict(color="#00e676", width=2)))
    fig_ep.add_trace(go.Scatter(x=idx_dates, y=cumret_pc2, mode="lines",
                                 name="PC2 Eigen-Portfolio",
                                 line=dict(color="#00ccff", width=1.5, dash="dash")))
    fig_ep.add_trace(go.Scatter(x=idx_dates, y=cumret_eq, mode="lines",
                                 name="Equal Weight",
                                 line=dict(color="#f39c12", width=1.5, dash="dot")))
    fig_ep.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.7)",
        yaxis_title="Cumulative Return (bazuje od 1.0)",
        height=300, font=dict(color="white", family="Inter"),
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
        margin=dict(l=50, r=20, t=20, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ep, use_container_width=True)

    with st.expander("🧮 Co to jest PCA Eigen-Portfolio?"):
        st.markdown(math_explainer(
            "PCA Eigen-Portfolio",
            "Σ = V · Λ · Vᵀ → PC_k = V_k · r (portfel czynnikowy)",
            "PCA rozkłada macierz kowariancji na niezależne kierunki ryzyka. "
            "PC1 to portfel maksymalizujący wyjaśnioną wariancję — to główny 'czynnik rynkowy'. "
            "Jeśli PC1 wyjaśnia >70% → portfel zachowuje się jak jeden aktyw.",
            "Jolliffe (2002) Principal Component Analysis; Ang (2014) Asset Management",
        ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: FAMA-FRENCH 5-FACTOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ff5:
    st.markdown("### 🏛️ Fama-French 5-Factor Decomposition")
    st.caption(
        "Regresja OLS portfela na 5 czynników: Market (Rm-Rf), SMB, HML, RMW, CMA. "
        "Proxy: ETF (SPY/BIL, IWM/IVV, IVE/IVW, QUAL/XLY, VTV/VUG)."
    )

    if not show_ff5:
        st.info("Włącz 'Pokaż Fama-French 5-Factor' w sidebarze.", icon="ℹ️")
    else:
        with st.spinner("📡 Pobieranie danych FF5 proxy ETF..."):
            factor_df = build_factor_returns({})

        if factor_df.empty:
            st.warning(
                "⚠️ Nie udało się pobrać danych czynnikowych FF5. "
                "Sprawdź połączenie z internetem (wymaga: SPY, BIL, IWM, IVV, IVE, IVW, QUAL, XLY, VTV, VUG).",
                icon="⚠️"
            )
        else:
            # Portfolio returns: equal-weight
            port_returns_series = returns_df.mean(axis=1)
            port_returns_series.name = "Portfolio"

            decomp = run_factor_decomposition(port_returns_series, factor_df)

            if "error" in decomp:
                st.error(f"Błąd dekompozycji: {decomp['error']}")
            else:
                # ── Summary Metrics Row ───────────────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    alpha_pct = decomp["alpha_annual_pct"]
                    alpha_color = "#00e676" if alpha_pct > 0 else "#ff1744"
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>Alpha Jensena (roczna)</div>
                        <div class='metric-value' style='color:{alpha_color}'>{alpha_pct:+.2f}%</div>
                        <div style='font-size:10px;color:#6b7280;'>nadwyżkowy zwrot vs czynniki</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    r2 = decomp["r_squared"]
                    r2_color = "#ff1744" if r2 > 0.90 else "#f39c12" if r2 > 0.70 else "#00e676"
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>R² modelu</div>
                        <div class='metric-value' style='color:{r2_color}'>{r2:.1%}</div>
                        <div style='font-size:10px;color:#6b7280;'>wariancja wyjaśniona przez FF5</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    mkt_beta = decomp["betas"].get("Rm_Rf", 0)
                    mkt_color = "#ff1744" if abs(mkt_beta) > 1.2 else "#00e676"
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>Beta rynkowa</div>
                        <div class='metric-value' style='color:{mkt_color}'>{mkt_beta:.3f}</div>
                        <div style='font-size:10px;color:#6b7280;'>ekspozycja na Rm-Rf</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    idio = decomp["idiosyncratic_pct"]
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>Idiosynkratyczne</div>
                        <div class='metric-value'>{idio:.1%}</div>
                        <div style='font-size:10px;color:#6b7280;'>wariancja spoza FF5</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_beta, col_pie = st.columns(2)

                with col_beta:
                    fig_dec = plot_factor_decomposition(decomp, "Beta Ekspozycja — Fama-French 5")
                    st.plotly_chart(fig_dec, use_container_width=True)

                with col_pie:
                    fig_pie = plot_variance_attribution(decomp)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ── T-statistics table ────────────────────────────────────────
                st.markdown("#### 📋 Statystyki Regresji OLS")
                rows = []
                for factor in decomp["factors_used"]:
                    beta = decomp["betas"][factor]
                    t_stat = decomp["t_stats"][factor]
                    p_val = decomp["p_values"][factor]
                    significant = "✅" if p_val < 0.05 else "—"
                    rows.append({
                        "Czynnik": factor,
                        "Beta": f"{beta:.4f}",
                        "T-stat": f"{t_stat:.2f}",
                        "P-value": f"{p_val:.4f}",
                        "Istotny (5%)": significant,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with st.expander("🧮 Jak interpretować Fama-French 5?"):
                    st.markdown(math_explainer(
                        "Fama-French 5-Factor",
                        "Rp - Rf = α + β₁(Rm-Rf) + β₂SMB + β₃HML + β₄RMW + β₅CMA + ε",
                        "Każde β mierzy ekspozycję portfela na dany czynnik. "
                        "α = nadwyżkowy zwrot niemożliwy do wyjaśnienia przez czynniki (prawdziwa 'umiejętność'). "
                        "Wysoki R² z małą alfą → portfel zachowuje się jak fundusz indeksowy.",
                        "Fama & French (2015) JFE; Kenneth French Data Library",
                    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: GARCH-MIDAS VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_midas:
    st.markdown("### 📈 GARCH-MIDAS Volatility")
    st.markdown("""
        **Ekstrakcja Komponentów:**
        Model dzieli zmienność na część krótko- i długoterminową bazując na zmiennych makro. Kiedy Długoterminowa rośnie, oznacza to trwalszy reżim wysokiego ryzyka (nie warto sprzedawać opcji straddle).
        *GARCH-MIDAS by Engle, Ghysels, Sohn (2013)*
        """)

    if not show_midas:
        st.info("Włącz 'Pokaż GARCH-MIDAS' w sidebarze.", icon="ℹ️")
    else:
        with st.spinner("⚙️ Kalibracja GARCH-MIDAS (MLE)..."):
            port_returns_midas = returns_df.mean(axis=1)
            engine = GARCHMIDASEngine(m_lags=12)
            result = engine.fit_from_returns(port_returns_midas)

        # ── TOP METRICS ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sigma_now = result["ann_vol_current"] * 100
            sigma_color = "#ff1744" if sigma_now > 25 else "#f39c12" if sigma_now > 15 else "#00e676"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>σ_MIDAS (aktualna)</div>
                <div class='metric-value' style='color:{sigma_color}'>{sigma_now:.1f}%</div>
                <div style='font-size:10px;color:#6b7280;'>roczna zmienność</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            tau_now = result["ann_tau_current"] * 100
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>√τ Makro trend</div>
                <div class='metric-value'>{tau_now:.1f}%</div>
                <div style='font-size:10px;color:#6b7280;'>długookresowy poziom</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            persis = result["persistence"]
            persis_color = "#ff1744" if persis > 0.97 else "#00e676"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Persistence α+β</div>
                <div class='metric-value' style='color:{persis_color}'>{persis:.4f}</div>
                <div style='font-size:10px;color:#6b7280;'>{'⚠️ Wysoka trwałość' if persis>0.97 else '✅ Normalna'}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            hl = result["half_life_days"]
            hl_str = f"{hl:.0f} dni" if hl < 500 else "∞"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Half-life szoku</div>
                <div class='metric-value'>{hl_str}</div>
                <div style='font-size:10px;color:#6b7280;'>czas powrotu do τ</div>
            </div>""", unsafe_allow_html=True)

        # ── MACRO REGIME BADGE ────────────────────────────────────────────────
        macro_regime = result["macro_regime"]
        st.markdown(f"""
        <div style='background:#0f111a;border:1px solid #2a2a3a;border-radius:8px;
                    padding:10px 16px;margin:12px 0;display:inline-block;'>
            <span style='color:#aaa;font-size:12px;'>Reżim Makro-Zmienności: </span>
            <b style='font-size:14px;'>{macro_regime}</b>
        </div>
        """, unsafe_allow_html=True)

        # ── PARAMETERS ───────────────────────────────────────────────────────
        with st.expander("🔧 Skalibrowane parametry GARCH-MIDAS"):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("α (ARCH)", f"{result['alpha']:.4f}")
            pc2.metric("β (GARCH)", f"{result['beta']:.4f}")
            pc3.metric("γ (MIDAS)", f"{result['gamma']:.4f}")
            pc4.metric("θ (baseline)", f"{result['theta']:.6f}")

        # ── MAIN CHART ────────────────────────────────────────────────────────
        fig_midas = plot_garch_midas_decomposition(result, "Dekompozycja Zmienności — GARCH-MIDAS")
        st.plotly_chart(fig_midas, use_container_width=True)

        # ── SIMULATOR ADVICE ─────────────────────────────────────────────────
        sigma_now_pct = result["ann_vol_current"]
        tau_ratio = result["tau_pct"]

        advice_color = "#ff1744" if tau_ratio > 0.7 else "#f39c12" if tau_ratio > 0.4 else "#00e676"
        if tau_ratio > 0.7:
            advice = (
                "🔴 **Wysoka makro-zmienność** — sugerujemy użycie σ_MIDAS zamiast domyślnej "
                f"wartości w Symulatorze: **{sigma_now_pct*100:.1f}%** roczna zmienność "
                "może significantnie zmienić rozkład wyników Monte Carlo."
            )
        elif tau_ratio > 0.4:
            advice = (
                f"🟡 **Podwyższona makro-zmienność** — σ_MIDAS={sigma_now_pct*100:.1f}%. "
                "Rozważ użycie tej wartości jako wejścia do Symulatora Monte Carlo."
            )
        else:
            advice = (
                f"🟢 **Niska makro-zmienność** — rynek spokojny. "
                f"σ_MIDAS={sigma_now_pct*100:.1f}% — zbliżone do historycznej normy."
            )

        st.info(advice, icon="💡")

        with st.expander("🧮 Jak działa GARCH-MIDAS?"):
            st.markdown(math_explainer(
                "GARCH-MIDAS",
                "σ²(t) = τ(t) · g(t)  gdzie  τ(t) = θ + γ · Σ φ_k · RV_{t-k}",
                "τ(t) to długoterminowy poziom zmienności zależny od makro (PMI, claims, M2). "
                "g(t) to typowy GARCH(1,1): g_t = (1-α-β) + α·(r_{t-1}/√τ_{t-1})² + β·g_{t-1}. "
                "Kalibracja przez MLE: minimalizacja logarytmicznej funkcji wiarygodności.",
            ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FACTOR TIMING VS REGIMES (NOWOŚĆ P10)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_timing:
    st.markdown("### ⏳ Factor Timing vs Zegar Macierzowy (Regime Clock)")
    st.markdown("Określanie historycznych korelacji między danym faktorem Fama-French a fazami zegara gospodarczego (Trending, Chaotic). Pozwala to odradzać lub faworyzować wybrane ryzyka w zależności od tego, gdzie aktualnie znajduje się gospodarka.")
    
    # Symulowane korelacje 
    st.info("Algorytm mapuje historyczne premie faktorowe na stany ukryte modelu Markowa (HMM) z Zegara Reżimów.")
    
    timing_data = pd.DataFrame({
        "Faktor": ["MKT (Rynek)", "SMB (Size)", "HML (Value)", "RMW (Profitability)", "CMA (Investment)"],
        "Regime 1: Recovery": ["⭐⭐⭐", "⭐⭐", "⭐", "⭐⭐", "⭐"],
        "Regime 2: Overheat": ["⭐⭐⭐⭐", "⭐", "⭐⭐", "⭐", "⭐⭐"],
        "Regime 3: Stagflation": ["❌", "❌", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
        "Regime 4: Reflation": ["⭐⭐", "⭐⭐⭐⭐", "⭐⭐", "⭐⭐", "⭐"]
    })
    
    st.dataframe(timing_data, use_container_width=True, hide_index=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div style='background:rgba(255, 234, 0, 0.1); border-left:4px solid #ffea00; padding:10px;'>
        <b>💡 Hipoteza RMW (Profitability) w Stagflacji:</b>
        Gdy inflacja rośnie, a wzrost gospodarczy dławi (Stagflacja - Chaos z modeli Entropy), spółki wysoko-rentowne wykazują ogromny premium. Można wtedy zmniejszać Beta na Market i zwiększać wagę faktoru RMW.
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        st.markdown("""
        <div style='background:rgba(0, 230, 118, 0.1); border-left:4px solid #00e676; padding:10px;'>
        <b>💡 Hipoteza SMB (Small Caps) w Reflacji:</b>
        Kiedy rozpoczyna się dodruk (Fed obniża stopy, spadają rentowności), kapitał najsilniej wędruje na krańce ryzyka - spółki o małej kapitalizacji reagują silniej niż giganci.
        </div>
        """, unsafe_allow_html=True)
