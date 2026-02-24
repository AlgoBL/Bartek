"""
factor_model.py — Fama-French 5-Factor Decomposition
Implementuje pełną analizę czynnikową portfela.

Czynniki:
  Rm-Rf — Market Premium (CAPM)
  SMB   — Small Minus Big (rozmiar)
  HML   — High Minus Low (value/growth)
  RMW   — Robust Minus Weak (rentowność operacyjna)
  CMA   — Conservative Minus Aggressive (inwestycje)

Referencje:
  Fama & French (1993), Fama & French (2015).
  Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.linalg import lstsq
import plotly.graph_objects as go
import plotly.express as px
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ─── Proxy Czynnikowe (bez dostępu do Kenneth French Data Library) ────────────
# Używamy publicznie dostępnych ETF jako proxy:
#   Rm-Rf: S&P 500 (SPY) - 3M T-Bill (BIL)
#   SMB:   IWM (small cap) - IVV (large cap)
#   HML:   IVE (Value) - IVW (Growth)
#   RMW:   QUAL (Quality/High ROE) - proxy wg analizy
#   CMA:   VTV (conservative) - VUG (aggressive growth)

FF5_PROXY_TICKERS = {
    "market":       ("SPY",  "BIL"),   # (risky, safe)
    "smb":          ("IWM",  "IVV"),
    "hml":          ("IVE",  "IVW"),
    "rmw":          ("QUAL", "XLY"),   # proxy: quality vs. discretionary
    "cma":          ("VTV",  "VUG"),
}


def _ols_regression(y: np.ndarray, X: np.ndarray):
    """
    OLS regression with standard errors. Returns coefficients, t-stats, p-values.
    """
    n, k = X.shape
    b, _, _, _ = lstsq(X, y)
    resid = y - X @ b
    s2    = float(np.dot(resid, resid)) / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se    = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 0))
    t_vals = b / (se + 1e-12)
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), df=max(n - k, 1)))
    r2     = 1 - np.var(resid) / (np.var(y) + 1e-10)
    return b, se, t_vals, p_vals, r2


def build_factor_returns(
    factor_prices: dict[str, pd.Series],
    rf_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Buduje codzienne stopy zwrotu dla 5 czynników FF z danych cenowych.

    Parameters
    ----------
    factor_prices : dict z kluczami matching FF5_PROXY_TICKERS,
                    wartości = pd.DataFrame z kolumnami (long_ticker, short_ticker)
    rf_series     : bezrisk. stopa (dzienna). Jeśli None → zakładamy 0.

    Returns pd.DataFrame z kolumnami [Rm_Rf, SMB, HML, RMW, CMA].
    """
    try:
        from modules.data_provider import fetch_data
    except ImportError:
        return pd.DataFrame()

    all_tickers = list({t for pair in FF5_PROXY_TICKERS.values() for t in pair})
    try:
        raw = fetch_data(
            all_tickers, period="3y", auto_adjust=True
        )
    except Exception as e:
        logger.error(f"Błąd pobierania FF5 proxy dla {all_tickers}: {e}")
        return pd.DataFrame()

    def _get_returns(ticker: str) -> pd.Series:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[ticker]["Close"].pct_change().dropna()
            return raw["Close"].pct_change().dropna()
        except Exception as e:
            logger.debug(f"Brak danych lub błąd dla FF5 proxy {ticker}: {e}")
            return pd.Series(dtype=float)

    factor_df = pd.DataFrame()

    factor_names = {"market": "Rm_Rf", "smb": "SMB", "hml": "HML",
                    "rmw": "RMW", "cma": "CMA"}

    for key, (long_t, short_t) in FF5_PROXY_TICKERS.items():
        r_long  = _get_returns(long_t)
        r_short = _get_returns(short_t)
        aligned = pd.concat([r_long, r_short], axis=1, join="inner")
        if aligned.empty:
            continue
        factor_series = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        factor_df[factor_names[key]] = factor_series

    # Subtract RF from market factor if available
    if rf_series is not None and "Rm_Rf" in factor_df.columns:
        factor_df["Rm_Rf"] = factor_df["Rm_Rf"].subtract(rf_series, fill_value=0)

    return factor_df.dropna()


def run_factor_decomposition(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    n_factors: int = 5,
) -> dict:
    """
    OLS regresja portfela na czynniki Fama-French.

    Returns
    -------
    dict z:
      alpha       : Jensen's alpha (dzienny)
      alpha_annual: Jensen's alpha (roczny)
      betas       : dict {czynnik: beta}
      t_stats     : dict {czynnik: t-stat}
      p_values    : dict {czynnik: p-value}
      r_squared   : R² modelu
      factor_attribution : % wyjaśnionej zmienności per czynnik
      idiosyncratic_return: niewyjaśniony zwrot (alfa)
    """
    # Align na wspólnym indeksie
    if isinstance(portfolio_returns, pd.Series):
        port = portfolio_returns.copy()
    else:
        port = pd.Series(portfolio_returns, dtype=float)

    common = port.index.intersection(factor_returns.index)
    if len(common) < 30:
        return {"error": "Za mało wspólnych obserwacji (min 30)."}

    y = port.loc[common].values
    factor_cols = factor_returns.columns.tolist()[:n_factors]
    X_raw = factor_returns.loc[common, factor_cols].values
    X     = np.column_stack([np.ones(len(y)), X_raw])  # add intercept

    b, se, t_vals, p_vals, r2 = _ols_regression(y, X)

    alpha_daily  = float(b[0])
    alpha_annual = alpha_daily * 252
    betas        = {f: float(b[i+1]) for i, f in enumerate(factor_cols)}
    t_stats      = {f: float(t_vals[i+1]) for i, f in enumerate(factor_cols)}
    p_values     = {f: float(p_vals[i+1]) for i, f in enumerate(factor_cols)}

    # Factor attribution (variance decomposition)
    total_var = float(np.var(y))
    factor_var = {}
    for i, f in enumerate(factor_cols):
        explained = float(betas[f] ** 2 * np.var(X_raw[:, i]))
        factor_var[f] = min(explained / max(total_var, 1e-10), 1.0)

    idio_pct = max(0.0, 1.0 - sum(factor_var.values()))

    return {
        "alpha_daily":          alpha_daily,
        "alpha_annual":         alpha_annual,
        "alpha_annual_pct":     alpha_annual * 100,
        "betas":                betas,
        "t_stats":              t_stats,
        "p_values":             p_values,
        "r_squared":            float(r2),
        "factor_attribution":   factor_var,
        "idiosyncratic_pct":    idio_pct,
        "n_observations":       len(y),
        "factors_used":         factor_cols,
    }


def plot_factor_decomposition(result: dict, title: str = "FF5 Factor Decomposition") -> go.Figure:
    """
    Plotly bar chart: beta exposure + variance attribution.
    """
    if "error" in result:
        fig = go.Figure()
        fig.add_annotation(text=result["error"], x=0.5, y=0.5, showarrow=False)
        return fig

    betas = result["betas"]
    colors = ["#00ff88" if v >= 0 else "#ff4444" for v in betas.values()]

    fig = go.Figure(go.Bar(
        x=list(betas.keys()),
        y=list(betas.values()),
        marker_color=colors,
        text=[f"{v:.3f}" for v in betas.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{title} | α={result['alpha_annual_pct']:.2f}%/yr | R²={result['r_squared']:.2%}",
        xaxis_title="Czynnik",
        yaxis_title="Ekspozycja Beta",
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        font=dict(color="white"),
        yaxis=dict(zeroline=True, zerolinecolor="gray", zerolinewidth=1),
    )
    return fig


def plot_variance_attribution(result: dict) -> go.Figure:
    """
    Plotly pie: variance attribution per factor + idiosyncratic.
    """
    if "error" in result:
        return go.Figure()

    labels  = list(result["factor_attribution"].keys()) + ["Idiosyncratic (α)"]
    values  = list(result["factor_attribution"].values()) + [result["idiosyncratic_pct"]]
    palette = ["#00ff88", "#00ccff", "#ffaa00", "#ff6b35", "#e040fb", "#aaaaaa"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=palette[:len(labels)]),
        texttemplate="%{label}<br>%{percent:.1%}",
        hole=0.35,
    ))
    fig.update_layout(
        title="Dekompozycja Wariancji (FF5)",
        template="plotly_dark",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig
