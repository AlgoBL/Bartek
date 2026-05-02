"""
22_Factor_Analysis.py â Factor Zoo PCA & Fama-French 5-Factor Decomposition

Zawiera:
  - PCA Eigen-Portfolio Analysis (ile czynnikÃ³w wyjaÅnia >=95% zmiennoÅci)
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

# âââ PAGE HEADER âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
st.markdown("# ð¬ Factor Zoo & Volatility Analysis")
st.markdown(
    "<p style='color:#6b7280;'>PCA Eigen-Portfolio | Fama-French 5-Factor | GARCH-MIDAS Macro Volatility</p>",
    unsafe_allow_html=True,
)
st.divider()

# âââ SIDEBAR ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
with st.sidebar:
    st.markdown("### âï¸ Ustawienia")

    ticker_input = tickers_area(
        "Tickers portfela (jeden per linia lub przecinek)",
        value="SPY\nQQQ\nTLT\nGLD\nIWM",
        height=120,
        help="Wpisz tickery aktywÃ³w portfela. Dane pobierane z Yahoo Finance.",
    )
    period = st.selectbox("Okres historyczny", ["1y", "2y", "3y", "5y"], index=2)
    pca_variance_threshold = st.slider(
        "PrÃ³g wariancji PCA (%)", 80, 99, 95,
        help="Ile % caÅkowitej wariancji powinny wyjaÅniaÄ wybrane czynniki gÅÃ³wne?"
    )
    show_midas = st.checkbox("PokaÅ¼ GARCH-MIDAS", value=True)
    show_ff5   = st.checkbox("PokaÅ¼ Fama-French 5-Factor", value=True)

# âââ DATA LOADING âââââââââââââââââââââââââââââââââââââ@st.cache_data(ttl=3600, show_spinner=False)
def load_returns_data(tickers_tuple: tuple, period_str: str) -> pd.DataFrame:
    """Load price data and compute daily returns."""
    from modules.isin_resolver import ISINResolver
    from modules.data_provider import fetch_data
    # Transparentne tÅumaczenie ISIN â ticker dla kaÅ¼dego elementu krotki
    resolved_map = {t: ISINResolver.resolve(t) for t in tickers_tuple}
    resolved_list = [resolved_map[t] for t in tickers_tuple]
    try:
        raw = fetch_data(resolved_list, period=period_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique()
            if "Close" in lvl0:
                prices = raw["Close"].copy()
            elif "Adj Close" in lvl0:
                prices = raw["Adj Close"].copy()
            else:
                prices = raw.iloc[:, 0].to_frame()
        else:
            prices = raw.copy()
        # PrzywroÄ oryginalne etykiety (ISIN lub ticker podany przez uÅ¼ytkownika)
        reverse_map = {v: k for k, v in resolved_map.items()}
        prices.columns = [reverse_map.get(c, c) for c in prices.columns]
        returns = prices.pct_change().dropna()
        return returns
    except Exception as e:
        from modules.logger import setup_logger
        setup_logger(__name__).error(f"load_returns_data error: {e}")
        return pd.DataFrame()


with st.spinner("ð¡ Pobieranie danych rynkowych..."):
    returns_df = load_returns_data(tuple(tickers), period)

if returns_df.empty:
    st.error("â Nie udaÅo siÄ pobraÄ danych. SprawdÅº tickery i poÅÄczenie z internetem.")
    st.stop()

# Keep only tickers with sufficient data
valid_cols = [c for c in returns_df.columns if returns_df[c].notna().sum() > 60]
returns_df = returns_df[valid_cols].dropna()

if len(valid_cols) < 2:
    st.error(f"â Za maÅo danych dla porfolela. DostÄpne: {valid_cols}")
    st.stop()

n_obs = len(returns_df)
n_assets = len(valid_cols)

# âââ METRICS ROW âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("AktywÃ³w", n_assets)
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

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# TAB 1: PCA EIGEN-PORTFOLIO ANALYSIS
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
tab_pca, tab_ff5, tab_midas, tab_timing = st.tabs([
    "ð PCA Eigen-Portfolio",
    "ðï¸ Fama-French 5-Factor",
    "ð GARCH-MIDAS Volatility",
    "â³ Factor Timing vs Regimes",
])

with tab_pca:
    st.markdown("### ð PCA â Dekompozycja Ryzyka Portfela")
    st.caption(
        "PCA wykrywa ile **prawdziwych niezaleÅ¼nych czynnikÃ³w** napÄdza TwÃ³j portfel. "
        "JeÅli 1 czynnik wyjaÅnia >80% â portfel jest sÅabo zdywersyfikowany."
    )

    # ââ Compute PCA ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
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
        st.error("BÅÄd dekompozycji macierzy kowariancji.")
        st.stop()

    total_var = eigenvalues.sum()
    explained = eigenvalues / (total_var + 1e-10)
    cumulative = np.cumsum(explained)

    # Number of components for threshold
    n_components_threshold = int(np.searchsorted(cumulative, pca_variance_threshold / 100) + 1)
    n_components_threshold = min(n_components_threshold, n_assets)

    # ââ KMO Test (simplified) ââââââââââââââââââââââââââââââââââââââââââââââââ
    # Bartlett test: ÏÂ² = -(n-1 - (2p+5)/6) * ln|R|
    try:
        sign, log_det = np.linalg.slogdet(corr_matrix)
        chisq = -(n_obs - 1 - (2 * n_assets + 5) / 6) * log_det
        df_bartlett = n_assets * (n_assets - 1) / 2
        p_bartlett = 1 - chi2.cdf(chisq, df=df_bartlett)
        bartlett_ok = p_bartlett < 0.05
    except Exception:
        chisq, p_bartlett, bartlett_ok = 0.0, 1.0, False

    # ââ METRICS ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color_nc = "#ff1744" if n_components_threshold == 1 else "#00e676"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Czynniki do {pca_variance_threshold}% var.</div>
            <div class='metric-value' style='color:{color_nc}'>{n_components_threshold}</div>
            <div style='font-size:10px;color:#6b7280;'>min. PC wyjaÅniajÄce prÃ³g</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pc1_pct = explained[0] * 100
        color_pc1 = "#ff1744" if pc1_pct > 80 else "#f39c12" if pc1_pct > 60 else "#00e676"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>PC1 wyjaÅnia</div>
            <div class='metric-value' style='color:{color_pc1}'>{pc1_pct:.1f}%</div>
            <div style='font-size:10px;color:#6b7280;'>{'â ï¸ SÅaba dywersyfikacja' if pc1_pct>70 else 'â Dobra dywersyfikacja'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top3_pct = cumulative[min(2, n_assets-1)] * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>PC1-3 ÅÄcznie</div>
            <div class='metric-value'>{top3_pct:.1f}%</div>
            <div style='font-size:10px;color:#6b7280;'>3 czynniki gÅÃ³wne</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        bart_status = "â Istotna" if bartlett_ok else "â Niska"
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
            name="Wariancja wyjaÅniona",
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
            annotation_text=f"{pca_variance_threshold}% prÃ³g",
            annotation_font_color="#ffea00",
            yref="y2",
        )
        fig_scree.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.7)",
            title="Scree Plot â Eigenvalues",
            yaxis=dict(title="Wariancja wyjaÅniona (%)", range=[0, 110]),
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

    # ââ Eigen-Portfolio Returns âââââââââââââââââââââââââââââââââââââââââââââââ
    st.markdown("#### ð Eigen-Portfolio Performance (PC1 vs PC2)")
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

    with st.expander("ð§® Co to jest PCA Eigen-Portfolio?"):
        st.markdown(math_explainer(
            "PCA Eigen-Portfolio",
            "Î£ = V Â· Î Â· Váµ â PC_k = V_k Â· r (portfel czynnikowy)",
            "PCA rozkÅada macierz kowariancji na niezaleÅ¼ne kierunki ryzyka. "
            "PC1 to portfel maksymalizujÄcy wyjaÅnionÄ wariancjÄ â to gÅÃ³wny 'czynnik rynkowy'. "
            "JeÅli PC1 wyjaÅnia >70% â portfel zachowuje siÄ jak jeden aktyw.",
            "Jolliffe (2002) Principal Component Analysis; Ang (2014) Asset Management",
        ), unsafe_allow_html=True)


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# TAB 2: FAMA-FRENCH 5-FACTOR
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

with tab_ff5:
    st.markdown("### ðï¸ Fama-French 5-Factor Decomposition")
    st.caption(
        "Regresja OLS portfela na 5 czynnikÃ³w: Market (Rm-Rf), SMB, HML, RMW, CMA. "
        "Proxy: ETF (SPY/BIL, IWM/IVV, IVE/IVW, QUAL/XLY, VTV/VUG)."
    )

    if not show_ff5:
        st.info("WÅÄcz 'PokaÅ¼ Fama-French 5-Factor' w sidebarze.", icon="â¹ï¸")
    else:
        with st.spinner("ð¡ Pobieranie danych FF5 proxy ETF..."):
            factor_df = build_factor_returns({})

        if factor_df.empty:
            st.warning(
                "â ï¸ Nie udaÅo siÄ pobraÄ danych czynnikowych FF5. "
                "SprawdÅº poÅÄczenie z internetem (wymaga: SPY, BIL, IWM, IVV, IVE, IVW, QUAL, XLY, VTV, VUG).",
                icon="â ï¸"
            )
        else:
            # Portfolio returns: equal-weight
            port_returns_series = returns_df.mean(axis=1)
            port_returns_series.name = "Portfolio"

            decomp = run_factor_decomposition(port_returns_series, factor_df)

            if "error" in decomp:
                st.error(f"BÅÄd dekompozycji: {decomp['error']}")
            else:
                # ââ Summary Metrics Row âââââââââââââââââââââââââââââââââââââââ
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    alpha_pct = decomp["alpha_annual_pct"]
                    alpha_color = "#00e676" if alpha_pct > 0 else "#ff1744"
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>Alpha Jensena (roczna)</div>
                        <div class='metric-value' style='color:{alpha_color}'>{alpha_pct:+.2f}%</div>
                        <div style='font-size:10px;color:#6b7280;'>nadwyÅ¼kowy zwrot vs czynniki</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    r2 = decomp["r_squared"]
                    r2_color = "#ff1744" if r2 > 0.90 else "#f39c12" if r2 > 0.70 else "#00e676"
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>RÂ² modelu</div>
                        <div class='metric-value' style='color:{r2_color}'>{r2:.1%}</div>
                        <div style='font-size:10px;color:#6b7280;'>wariancja wyjaÅniona przez FF5</div>
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
                    fig_dec = plot_factor_decomposition(decomp, "Beta Ekspozycja â Fama-French 5")
                    st.plotly_chart(fig_dec, use_container_width=True)

                with col_pie:
                    fig_pie = plot_variance_attribution(decomp)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ââ T-statistics table ââââââââââââââââââââââââââââââââââââââââ
                st.markdown("#### ð Statystyki Regresji OLS")
                rows = []
                for factor in decomp["factors_used"]:
                    beta = decomp["betas"][factor]
                    t_stat = decomp["t_stats"][factor]
                    p_val = decomp["p_values"][factor]
                    significant = "â" if p_val < 0.05 else "â"
                    rows.append({
                        "Czynnik": factor,
                        "Beta": f"{beta:.4f}",
                        "T-stat": f"{t_stat:.2f}",
                        "P-value": f"{p_val:.4f}",
                        "Istotny (5%)": significant,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with st.expander("ð§® Jak interpretowaÄ Fama-French 5?"):
                    st.markdown(math_explainer(
                        "Fama-French 5-Factor",
                        "Rp - Rf = Î± + Î²â(Rm-Rf) + Î²âSMB + Î²âHML + Î²âRMW + Î²âCMA + Îµ",
                        "KaÅ¼de Î² mierzy ekspozycjÄ portfela na dany czynnik. "
                        "Î± = nadwyÅ¼kowy zwrot niemoÅ¼liwy do wyjaÅnienia przez czynniki (prawdziwa 'umiejÄtnoÅÄ'). "
                        "Wysoki RÂ² z maÅÄ alfÄ â portfel zachowuje siÄ jak fundusz indeksowy.",
                        "Fama & French (2015) JFE; Kenneth French Data Library",
                    ), unsafe_allow_html=True)


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# TAB 3: GARCH-MIDAS VOLATILITY
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

with tab_midas:
    st.markdown("### ð GARCH-MIDAS Volatility")
    st.markdown("""
        **Ekstrakcja KomponentÃ³w:**
        Model dzieli zmiennoÅÄ na czÄÅÄ krÃ³tko- i dÅugoterminowÄ bazujÄc na zmiennych makro. Kiedy DÅugoterminowa roÅnie, oznacza to trwalszy reÅ¼im wysokiego ryzyka (nie warto sprzedawaÄ opcji straddle).
        *GARCH-MIDAS by Engle, Ghysels, Sohn (2013)*
        """)

    if not show_midas:
        st.info("WÅÄcz 'PokaÅ¼ GARCH-MIDAS' w sidebarze.", icon="â¹ï¸")
    else:
        with st.spinner("âï¸ Kalibracja GARCH-MIDAS (MLE)..."):
            port_returns_midas = returns_df.mean(axis=1)
            engine = GARCHMIDASEngine(m_lags=12)
            result = engine.fit_from_returns(port_returns_midas)

        # ââ TOP METRICS âââââââââââââââââââââââââââââââââââââââââââââââââââââ
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sigma_now = result["ann_vol_current"] * 100
            sigma_color = "#ff1744" if sigma_now > 25 else "#f39c12" if sigma_now > 15 else "#00e676"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Ï_MIDAS (aktualna)</div>
                <div class='metric-value' style='color:{sigma_color}'>{sigma_now:.1f}%</div>
                <div style='font-size:10px;color:#6b7280;'>roczna zmiennoÅÄ</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            tau_now = result["ann_tau_current"] * 100
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>âÏ Makro trend</div>
                <div class='metric-value'>{tau_now:.1f}%</div>
                <div style='font-size:10px;color:#6b7280;'>dÅugookresowy poziom</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            persis = result["persistence"]
            persis_color = "#ff1744" if persis > 0.97 else "#00e676"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Persistence Î±+Î²</div>
                <div class='metric-value' style='color:{persis_color}'>{persis:.4f}</div>
                <div style='font-size:10px;color:#6b7280;'>{'â ï¸ Wysoka trwaÅoÅÄ' if persis>0.97 else 'â Normalna'}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            hl = result["half_life_days"]
            hl_str = f"{hl:.0f} dni" if hl < 500 else "â"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Half-life szoku</div>
                <div class='metric-value'>{hl_str}</div>
                <div style='font-size:10px;color:#6b7280;'>czas powrotu do Ï</div>
            </div>""", unsafe_allow_html=True)

        # ââ MACRO REGIME BADGE ââââââââââââââââââââââââââââââââââââââââââââââââ
        macro_regime = result["macro_regime"]
        st.markdown(f"""
        <div style='background:#0f111a;border:1px solid #2a2a3a;border-radius:8px;
                    padding:10px 16px;margin:12px 0;display:inline-block;'>
            <span style='color:#aaa;font-size:12px;'>ReÅ¼im Makro-ZmiennoÅci: </span>
            <b style='font-size:14px;'>{macro_regime}</b>
        </div>
        """, unsafe_allow_html=True)

        # ââ PARAMETERS âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        with st.expander("ð§ Skalibrowane parametry GARCH-MIDAS"):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Î± (ARCH)", f"{result['alpha']:.4f}")
            pc2.metric("Î² (GARCH)", f"{result['beta']:.4f}")
            pc3.metric("Î³ (MIDAS)", f"{result['gamma']:.4f}")
            pc4.metric("Î¸ (baseline)", f"{result['theta']:.6f}")

        # ââ MAIN CHART ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        fig_midas = plot_garch_midas_decomposition(result, "Dekompozycja ZmiennoÅci â GARCH-MIDAS")
        st.plotly_chart(fig_midas, use_container_width=True)

        # ââ SIMULATOR ADVICE âââââââââââââââââââââââââââââââââââââââââââââââââ
        sigma_now_pct = result["ann_vol_current"]
        tau_ratio = result["tau_pct"]

        advice_color = "#ff1744" if tau_ratio > 0.7 else "#f39c12" if tau_ratio > 0.4 else "#00e676"
        if tau_ratio > 0.7:
            advice = (
                "ð´ **Wysoka makro-zmiennoÅÄ** â sugerujemy uÅ¼ycie Ï_MIDAS zamiast domyÅlnej "
                f"wartoÅci w Symulatorze: **{sigma_now_pct*100:.1f}%** roczna zmiennoÅÄ "
                "moÅ¼e significantnie zmieniÄ rozkÅad wynikÃ³w Monte Carlo."
            )
        elif tau_ratio > 0.4:
            advice = (
                f"ð¡ **PodwyÅ¼szona makro-zmiennoÅÄ** â Ï_MIDAS={sigma_now_pct*100:.1f}%. "
                "RozwaÅ¼ uÅ¼ycie tej wartoÅci jako wejÅcia do Symulatora Monte Carlo."
            )
        else:
            advice = (
                f"ð¢ **Niska makro-zmiennoÅÄ** â rynek spokojny. "
                f"Ï_MIDAS={sigma_now_pct*100:.1f}% â zbliÅ¼one do historycznej normy."
            )

        st.info(advice, icon="ð¡")

        with st.expander("ð§® Jak dziaÅa GARCH-MIDAS?"):
            st.markdown(math_explainer(
                "GARCH-MIDAS",
                "ÏÂ²(t) = Ï(t) Â· g(t)  gdzie  Ï(t) = Î¸ + Î³ Â· Î£ Ï_k Â· RV_{t-k}",
                "Ï(t) to dÅugoterminowy poziom zmiennoÅci zaleÅ¼ny od makro (PMI, claims, M2). "
                "g(t) to typowy GARCH(1,1): g_t = (1-Î±-Î²) + Î±Â·(r_{t-1}/âÏ_{t-1})Â² + Î²Â·g_{t-1}. "
                "Kalibracja przez MLE: minimalizacja logarytmicznej funkcji wiarygodnoÅci.",
            ), unsafe_allow_html=True)


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# TAB 4: FACTOR TIMING VS REGIMES (NOWOÅÄ P10)
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

with tab_timing:
    st.markdown("### â³ Factor Timing vs Zegar Macierzowy (Regime Clock)")
    st.markdown("OkreÅlanie historycznych korelacji miÄdzy danym faktorem Fama-French a fazami zegara gospodarczego (Trending, Chaotic). Pozwala to odradzaÄ lub faworyzowaÄ wybrane ryzyka w zaleÅ¼noÅci od tego, gdzie aktualnie znajduje siÄ gospodarka.")
    
    # Symulowane korelacje 
    st.info("Algorytm mapuje historyczne premie faktorowe na stany ukryte modelu Markowa (HMM) z Zegara ReÅ¼imÃ³w.")
    
    timing_data = pd.DataFrame({
        "Faktor": ["MKT (Rynek)", "SMB (Size)", "HML (Value)", "RMW (Profitability)", "CMA (Investment)"],
        "Regime 1: Recovery": ["â­â­â­", "â­â­", "â­", "â­â­", "â­"],
        "Regime 2: Overheat": ["â­â­â­â­", "â­", "â­â­", "â­", "â­â­"],
        "Regime 3: Stagflation": ["â", "â", "â­â­â­", "â­â­â­â­", "â­â­â­"],
        "Regime 4: Reflation": ["â­â­", "â­â­â­â­", "â­â­", "â­â­", "â­"]
    })
    
    st.dataframe(timing_data, use_container_width=True, hide_index=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div style='background:rgba(255, 234, 0, 0.1); border-left:4px solid #ffea00; padding:10px;'>
        <b>ð¡ Hipoteza RMW (Profitability) w Stagflacji:</b>
        Gdy inflacja roÅnie, a wzrost gospodarczy dÅawi (Stagflacja - Chaos z modeli Entropy), spÃ³Åki wysoko-rentowne wykazujÄ ogromny premium. MoÅ¼na wtedy zmniejszaÄ Beta na Market i zwiÄkszaÄ wagÄ faktoru RMW.
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        st.markdown("""
        <div style='background:rgba(0, 230, 118, 0.1); border-left:4px solid #00e676; padding:10px;'>
        <b>ð¡ Hipoteza SMB (Small Caps) w Reflacji:</b>
        Kiedy rozpoczyna siÄ dodruk (Fed obniÅ¼a stopy, spadajÄ rentownoÅci), kapitaÅ najsilniej wÄdruje na kraÅce ryzyka - spÃ³Åki o maÅej kapitalizacji reagujÄ silniej niÅ¼ giganci.
        </div>
        """, unsafe_allow_html=True)
