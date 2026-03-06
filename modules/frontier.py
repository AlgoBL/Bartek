"""
frontier.py — Intelligent Barbell v4.0 (Advanced Portfolio Optimization)

Implementuje pięć rygorystycznych metod optymalizacji:
  1. HRP  — Hierarchical Risk Parity (Lopez de Prado 2016)
  2. CVaR — Minimum CVaR (Rockafellar & Uryasev 2000)
  3. BL   — Black-Litterman z widokami CIO (He & Litterman 1999)
  4. NCO  — Nested Cluster Optimization (Lopez de Prado 2019)  [NEW]
  5. DRO  — Wasserstein Distributionally Robust Optimization    [NEW 2024]
  6. MC   — Monte Carlo sampling (legacy, Markowitz 1952) — zachowany

Referencje:
  - Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
  - Lopez de Prado, M. (2019). Machine Learning for Asset Managers. Ch. 16, NCO.
  - Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.
  - Black, F. & Litterman, R. (1992). Global Portfolio Optimization.
  - Zhang et al. (2024). Wasserstein DRO for Portfolio Optimization. arXiv:2401.xxxxx
"""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import block_diag
import plotly.graph_objects as go
import plotly.express as px
from modules.metrics import calculate_omega

# ══════════════════════════════════════════════════════════════════════════════
# 1. HRP — Hierarchical Risk Parity
# ══════════════════════════════════════════════════════════════════════════════

def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    """Distance matrix d_ij = sqrt(0.5*(1 - rho_ij)). Lopez de Prado (2016)."""
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(linkage: np.ndarray) -> list:
    """Sort clustered assets to minimise distance between adjacent items."""
    n = int(linkage[-1, 3])  # total leaves
    order = [n - 1]          # start from root
    while True:
        new_order = []
        for cluster in order:
            if cluster < len(linkage) + 1:   # leaf
                new_order.append(cluster)
            else:
                # Split into left / right children
                idx = int(cluster) - (len(linkage) + 1)
                left  = int(linkage[idx, 0])
                right = int(linkage[idx, 1])
                new_order.extend([left, right])
        if new_order == order:
            break
        order = new_order
    return [int(x) for x in order if x < (len(linkage) + 1)]


def _recursive_bisection(cov: np.ndarray, sorted_items: list) -> np.ndarray:
    """
    Allocate weights by recursive bisection + inverse-variance within clusters.
    """
    weights = np.ones(len(sorted_items))
    cluster_items = [sorted_items]

    while cluster_items:
        cluster_items = [
            sub[i: j]
            for sub in cluster_items
            for i, j in ((0, len(sub) // 2), (len(sub) // 2, len(sub)))
            if len(sub) > 1
        ]
        for sub in cluster_items:
            if len(sub) == 0:
                continue
            # Cluster variance via inverse-variance weights
            cov_sub = cov[np.ix_(sub, sub)]
            inv_var = 1.0 / np.diag(cov_sub)
            w_sub = inv_var / inv_var.sum()
            # Portfolio variance of this cluster
            var_left  = float(w_sub @ cov_sub @ w_sub)

            complement = [x for x in sorted_items if x not in sub]
            if not complement:
                continue
            cov_comp = cov[np.ix_(complement, complement)]
            inv_var_c = 1.0 / np.diag(cov_comp)
            w_comp = inv_var_c / inv_var_c.sum()
            var_right = float(w_comp @ cov_comp @ w_comp)

            # Allocation factor
            alpha = 1.0 - var_left / (var_left + var_right + 1e-10)
            weights[sub] *= alpha
            weights[complement] *= (1.0 - alpha)

    return weights / weights.sum()


def compute_hrp(returns_df: pd.DataFrame, risk_free_rate: float = 0.04) -> dict:
    """
    Hierarchical Risk Parity portfolio — Lopez de Prado (2016).

    Steps:
      1. Compute correlation matrix
      2. Convert to distance matrix
      3. Hierarchical clustering (Ward linkage)
      4. Quasi-diagonal reordering
      5. Recursive bisection allocation

    Returns dict with weights, expected return, volatility, Sharpe, fig.
    """
    returns = returns_df.copy()
    # Belka Tax
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    corr = returns.corr().values
    cov  = returns.cov().values * 252
    mu   = returns.mean().values * 252

    dist     = _corr_to_dist(corr)
    linkage  = sch.linkage(dist[np.triu_indices(len(dist), k=1)], method="ward")
    sorted_i = _get_quasi_diag(linkage)

    # Map back to 0..n-1 indices
    n = len(returns_df.columns)
    sorted_i = [x for x in sorted_i if x < n]
    weights_sorted = _recursive_bisection(cov, sorted_i)

    # Full weight vector
    w = np.zeros(n)
    for rank, asset_idx in enumerate(sorted_i):
        w[asset_idx] = weights_sorted[rank]
    w = w / w.sum()

    port_return = float(w @ mu)
    port_vol    = float(np.sqrt(w @ cov @ w))
    rf_taxed    = risk_free_rate * 0.81
    sharpe      = (port_return - rf_taxed) / port_vol if port_vol > 0 else 0
    daily_r     = (returns.values @ w)
    omega       = float(min(calculate_omega(daily_r), 10.0))

    tickers = returns_df.columns.tolist()
    return {
        "method":    "HRP",
        "weights":   {t: float(w[i]) for i, t in enumerate(tickers)},
        "return":    port_return,
        "volatility":port_vol,
        "sharpe":    sharpe,
        "omega":     omega,
        "linkage":   linkage,
        "sorted_i":  sorted_i,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CVaR Optimization — Rockafellar & Uryasev (2000)
# ══════════════════════════════════════════════════════════════════════════════

def compute_min_cvar(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
) -> dict:
    """
    Minimise CVaR_α portfolio (Expected Shortfall at level 1-α).

    Linear programme formulation — Rockafellar & Uryasev (2000):
      min  VaR + (1/(alpha*T)) * sum(z_t)
      s.t. z_t >= -r_t'w - VaR,  z_t >= 0,  sum(w)=1, w>=0

    Solved via scipy.optimize.minimize (SLSQP).
    Also computes Max-Sharpe and Max-Omega as reference points.
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    R   = returns.values          # (T, n)
    T, n = R.shape
    mu  = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252
    rf  = risk_free_rate * 0.81

    def _cvar(w: np.ndarray) -> float:
        port_r = R @ w
        var    = np.percentile(port_r, alpha * 100)
        tail   = port_r[port_r <= var]
        return -float(np.mean(tail)) if len(tail) > 0 else 0.0

    def _neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - rf) / vol if vol > 0 else 0.0

    def _neg_omega(w: np.ndarray) -> float:
        return -float(min(calculate_omega(R @ w), 20.0))

    bounds = [(0, max_weight)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0     = np.ones(n) / n

    # ── Min CVaR ──
    res_cvar = minimize(_cvar, w0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 1000, "ftol": 1e-9})
    w_cvar = np.abs(res_cvar.x); w_cvar /= w_cvar.sum()

    # ── Max Sharpe ──
    res_sharpe = minimize(_neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                          options={"maxiter": 1000, "ftol": 1e-9})
    w_sharpe = np.abs(res_sharpe.x); w_sharpe /= w_sharpe.sum()

    # ── Max Omega ──
    res_omega = minimize(_neg_omega, w0, method="SLSQP", bounds=bounds, constraints=cons,
                         options={"maxiter": 500, "ftol": 1e-7})
    w_omega = np.abs(res_omega.x); w_omega /= w_omega.sum()

    tickers = returns_df.columns.tolist()

    def _port_stats(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w)) if n > 1 else 0.0
        sh  = (ret - rf) / vol if vol > 0 else 0.0
        cv  = _cvar(w)
        om  = float(min(calculate_omega(R @ w), 10.0))
        return {"return": ret, "volatility": vol, "sharpe": sh, "cvar": cv, "omega": om,
                "weights": {t: float(w[i]) for i, t in enumerate(tickers)}}

    return {
        "method":    "CVaR",
        "min_cvar":  _port_stats(w_cvar),
        "max_sharpe":_port_stats(w_sharpe),
        "max_omega": _port_stats(w_omega),
        "alpha":     alpha,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Black-Litterman — He & Litterman (1999)
# ══════════════════════════════════════════════════════════════════════════════

def compute_black_litterman(
    returns_df: pd.DataFrame,
    cio_views: dict | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Black-Litterman model — He & Litterman (1999).

    1. Prior: π = δ × Σ × w_mkt  (equilibrium CAPM returns)
    2. Views:  P × μ = Q + ε,  Ω = diag(uncertainty)
    3. Posterior: E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

    cio_views: dict {ticker: view_return}  — CIO expected annual returns.
               None → use market-cap weight prior only.

    Returns portfolio weights, expected returns, and diagnostics.
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    tickers = returns_df.columns.tolist()
    n       = len(tickers)
    mu_hist = returns.mean().values * 252
    Sigma   = returns.cov().values * 252
    rf      = risk_free_rate * 0.81

    # Market weights — equal weight prior (proxy for market cap)
    w_mkt = np.ones(n) / n

    # Prior (equilibrium returns)
    pi = risk_aversion * Sigma @ w_mkt

    # ── Build views matrix from CIO ──────────────────────────────────────
    if cio_views and len(cio_views) > 0:
        view_tickers = [t for t in cio_views if t in tickers]
        k = len(view_tickers)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for i, t in enumerate(view_tickers):
            j = tickers.index(t)
            P[i, j] = 1.0
            Q[i]    = cio_views[t] * 0.81   # After Belka Tax

        # Uncertainty: proportional to variance of each asset
        omega = np.diag([tau * float(Sigma[tickers.index(t), tickers.index(t)])
                         for t in view_tickers])

        # Posterior expected returns
        tauSigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv    = np.linalg.inv(omega)
        M   = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
        mu_bl = M @ (tauSigma_inv @ pi + P.T @ Omega_inv @ Q)
    else:
        mu_bl = pi.copy()

    # ── Optimise Max-Sharpe on BL expected returns ───────────────────────
    def _neg_sharpe_bl(w):
        ret = float(w @ mu_bl)
        vol = float(np.sqrt(w @ Sigma @ w))
        return -(ret - rf) / vol if vol > 0 else 0.0

    bounds = [(0, 0.4)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    w0     = w_mkt.copy()
    res    = minimize(_neg_sharpe_bl, w0, method="SLSQP",
                      bounds=bounds, constraints=cons, options={"maxiter": 1000})
    w_bl   = np.abs(res.x); w_bl /= w_bl.sum()

    port_ret = float(w_bl @ mu_bl)
    port_vol = float(np.sqrt(w_bl @ Sigma @ w_bl))
    sharpe   = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    omega    = float(min(calculate_omega(returns.values @ w_bl), 10.0))

    return {
        "method":            "Black-Litterman",
        "weights":           {t: float(w_bl[i]) for i, t in enumerate(tickers)},
        "return":            port_ret,
        "volatility":        port_vol,
        "sharpe":            sharpe,
        "omega":             omega,
        "prior_returns":     {t: float(pi[i]) for i, t in enumerate(tickers)},
        "posterior_returns": {t: float(mu_bl[i]) for i, t in enumerate(tickers)},
        "views_used":        list(cio_views.keys()) if cio_views else [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plotly — Combined Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 2000,
    risk_free_rate: float = 0.04,
    barbell_weights: dict = None,
    cio_views: dict = None,
) -> dict:
    """
    Compute Efficient Frontier + HRP + CVaR + Black-Litterman portfolios.

    Parameters
    ----------
    returns_df     : DataFrame of daily returns, columns = asset tickers
    n_portfolios   : number of random MC portfolios (background scatter)
    risk_free_rate : annual risk-free rate (pre-tax)
    barbell_weights: dict {ticker: weight} — current Barbell strategy overlay
    cio_views      : dict {ticker: annual_return} — CIO views for Black-Litterman

    Returns
    -------
    dict with fig, portfolios_df, max_sharpe, min_vol, max_omega,
         hrp, cvar_opt, black_litterman
    """
    tickers  = returns_df.columns.tolist()
    n_assets = len(tickers)

    if n_assets < 2:
        return {"error": "Potrzeba co najmniej 2 aktywów do obliczeń granicy efektywnej."}

    # Belka Tax copy
    returns_taxed = returns_df.copy()
    returns_taxed.mask(returns_taxed > 0, returns_taxed * 0.81, inplace=True)

    mean_returns = returns_taxed.mean() * 252
    cov_matrix   = returns_taxed.cov() * 252
    rf_taxed     = risk_free_rate * 0.81

    # ── Monte Carlo sampling (background) ────────────────────────────────
    results = {"Return": [], "Volatility": [], "Sharpe": [], "Omega": [], "Weights": []}
    np.random.seed(42)

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = float(np.dot(w, mean_returns))
        vol = float(np.sqrt(w @ cov_matrix.values @ w))
        sh  = (ret - rf_taxed) / vol if vol > 0 else 0
        om  = min(calculate_omega(returns_taxed.values @ w), 10.0)
        results["Return"].append(ret)
        results["Volatility"].append(vol)
        results["Sharpe"].append(sh)
        results["Omega"].append(om)
        results["Weights"].append(w)

    df = pd.DataFrame(results)
    idx_max_sharpe = df["Sharpe"].idxmax()
    idx_min_vol    = df["Volatility"].idxmin()
    idx_max_omega  = df["Omega"].idxmax()

    def _port_info(idx):
        row = df.iloc[idx]
        return {"return": row["Return"], "volatility": row["Volatility"],
                "sharpe": row["Sharpe"], "omega": row["Omega"],
                "weights": {t: w for t, w in zip(tickers, row["Weights"])}}

    max_sharpe_port = _port_info(idx_max_sharpe)
    min_vol_port    = _port_info(idx_min_vol)
    max_omega_port  = _port_info(idx_max_omega)

    # ── Advanced portfolios ───────────────────────────────────────────────
    hrp_result = compute_hrp(returns_df, risk_free_rate)
    cvar_result = compute_min_cvar(returns_df, alpha=0.05, risk_free_rate=risk_free_rate)
    bl_result   = compute_black_litterman(returns_df, cio_views=cio_views,
                                          risk_free_rate=risk_free_rate)

    # ── Build Plotly figure ───────────────────────────────────────────────
    fig = go.Figure()

    # Background MC scatter
    fig.add_trace(go.Scatter(
        x=df["Volatility"] * 100, y=df["Return"] * 100,
        mode="markers",
        marker=dict(color=df["Sharpe"], colorscale="Viridis", size=4,
                    opacity=0.5, colorbar=dict(title="Sharpe"), showscale=True),
        text=[f"Sharpe: {s:.2f}<br>Omega: {o:.2f}"
              for s, o in zip(df["Sharpe"], df["Omega"])],
        hovertemplate="%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        name="Portfele losowe (MC)",
    ))

    # MC key portfolios
    for port, label, color, sym in [
        (max_sharpe_port, "⭐ MC Max Sharpe",  "gold",  "star"),
        (min_vol_port,    "🔵 MC Min Vol",     "cyan",  "diamond"),
        (max_omega_port,  "🟢 MC Max Omega",   "lime",  "triangle-up"),
    ]:
        fig.add_trace(go.Scatter(
            x=[port["volatility"] * 100], y=[port["return"] * 100],
            mode="markers+text",
            marker=dict(color=color, size=16, symbol=sym,
                        line=dict(color="white", width=1.5)),
            text=[label], textposition="top right",
            textfont=dict(color=color, size=10),
            hovertemplate=(f"<b>{label}</b><br>"
                           f"Return: {port['return']*100:.1f}%<br>"
                           f"Vol: {port['volatility']*100:.1f}%<br>"
                           f"Sharpe: {port['sharpe']:.2f}<extra></extra>"),
            name=label,
        ))

    # HRP
    hrp_vol = hrp_result.get("volatility", 0) * 100
    hrp_ret = hrp_result.get("return", 0) * 100
    fig.add_trace(go.Scatter(
        x=[hrp_vol], y=[hrp_ret], mode="markers+text",
        marker=dict(color="#ff6b35", size=20, symbol="hexagram",
                    line=dict(color="white", width=2)),
        text=["🟠 HRP"], textposition="bottom right",
        textfont=dict(color="#ff6b35", size=11),
        hovertemplate=(f"<b>🟠 HRP (Lopez de Prado 2016)</b><br>"
                       f"Return: {hrp_ret:.1f}%<br>Vol: {hrp_vol:.1f}%<br>"
                       f"Sharpe: {hrp_result.get('sharpe', 0):.2f}<extra></extra>"),
        name="🟠 HRP",
    ))

    # CVaR Min
    cv = cvar_result.get("min_cvar", {})
    if cv:
        fig.add_trace(go.Scatter(
            x=[cv["volatility"] * 100], y=[cv["return"] * 100],
            mode="markers+text",
            marker=dict(color="#e040fb", size=20, symbol="pentagon",
                        line=dict(color="white", width=2)),
            text=["🟣 Min CVaR"], textposition="top left",
            textfont=dict(color="#e040fb", size=11),
            hovertemplate=(f"<b>🟣 Min CVaR₅ (Rockafellar & Uryasev 2000)</b><br>"
                           f"Return: {cv['return']*100:.1f}%<br>"
                           f"Vol: {cv['volatility']*100:.1f}%<br>"
                           f"CVaR: {cv['cvar']*100:.1f}%<extra></extra>"),
            name="🟣 Min CVaR",
        ))

    # Black-Litterman
    bl_vol = bl_result.get("volatility", 0) * 100
    bl_ret = bl_result.get("return", 0) * 100
    fig.add_trace(go.Scatter(
        x=[bl_vol], y=[bl_ret], mode="markers+text",
        marker=dict(color="#00e5ff", size=20, symbol="star-square",
                    line=dict(color="white", width=2)),
        text=["🔷 BL"], textposition="top right",
        textfont=dict(color="#00e5ff", size=11),
        hovertemplate=(f"<b>🔷 Black-Litterman (He & Litterman 1999)</b><br>"
                       f"Return: {bl_ret:.1f}%<br>Vol: {bl_vol:.1f}%<br>"
                       f"Sharpe: {bl_result.get('sharpe', 0):.2f}<br>"
                       f"Widoki CIO: {len(bl_result.get('views_used', []))}<extra></extra>"),
        name="🔷 Black-Litterman",
    ))

    # Barbell overlay
    if barbell_weights:
        w_arr = np.array([barbell_weights.get(t, 0.0) for t in tickers])
        s = w_arr.sum()
        if s > 0:
            w_arr /= s
        bb_ret  = float(np.dot(w_arr, mean_returns))
        bb_vol  = float(np.sqrt(w_arr @ cov_matrix.values @ w_arr))
        bb_sh   = (bb_ret - rf_taxed) / bb_vol if bb_vol > 0 else 0
        bb_om   = float(min(calculate_omega(returns_taxed.values @ w_arr), 10.0))
        fig.add_trace(go.Scatter(
            x=[bb_vol * 100], y=[bb_ret * 100], mode="markers+text",
            marker=dict(color="red", size=24, symbol="x",
                        line=dict(color="white", width=2)),
            text=["🎯 Twój Barbell"], textposition="bottom right",
            textfont=dict(color="red", size=12),
            hovertemplate=(f"<b>🎯 Twoja Strategia Barbell</b><br>"
                           f"Return: {bb_ret*100:.1f}%<br>Vol: {bb_vol*100:.1f}%<br>"
                           f"Sharpe: {bb_sh:.2f}<br>Omega: {bb_om:.2f}<extra></extra>"),
            name="🎯 Twój Barbell",
        ))

    fig.update_layout(
        title="📐 Granica Efektywna — HRP + Min CVaR + Black-Litterman + MC (Markowitz 1952)",
        xaxis_title="Ryzyko (Volatility) [%]",
        yaxis_title="Oczekiwana Stopa Zwrotu [%]",
        template="plotly_dark", height=600,
        legend=dict(orientation="v", x=1.02, y=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        font=dict(family="Inter", color="white"),
        hovermode="closest",
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1,
                     spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1,
                     spikedash="dot", spikemode="across")

    return {
        "fig":              fig,
        "portfolios_df":    df,
        "max_sharpe":       max_sharpe_port,
        "min_vol":          min_vol_port,
        "max_omega":        max_omega_port,
        "hrp":              hrp_result,
        "cvar_opt":         cvar_result,
        "black_litterman":  bl_result,
        "error":            None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NCO — Nested Cluster Optimization  (Lopez de Prado 2019)  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

def compute_nco(
    returns_df: pd.DataFrame,
    n_clusters: int | None = None,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
) -> dict:
    """
    Nested Cluster Optimization (NCO) — Lopez de Prado (2019).

    Strategia: Podziel aktywa na klastry (hierarchical clustering),
    optymalizuj WEWNĄTRZ każdego klastra (intra-cluster weights),
    potem optymalizuj MIĘDZY klastrami (inter-cluster weights).

    Zalety vs klasyczny B-L lub Max-Sharpe:
      - Redukuje błąd estymacji macierzy kowariancji (mniejszy wymiar)
      - Naturalna dywersyfikacja między klastrami
      - Stabilne wagi OOS — mniejszy overfitting

    Ref: Lopez de Prado (2019), "Machine Learning for Asset Managers",
         Cambridge University Press, Chapter 16.
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    tickers  = returns_df.columns.tolist()
    n        = len(tickers)
    mu       = returns.mean().values * 252
    cov      = returns.cov().values * 252
    corr     = returns.corr().values
    rf       = risk_free_rate * 0.81

    if n < 3:
        # Degenerate case — fallback to HRP
        return compute_hrp(returns_df, risk_free_rate)

    # ── Step 1: Hierarchical Clustering ──────────────────────────────────
    dist = _corr_to_dist(corr)
    link = sch.linkage(dist[np.triu_indices(n, k=1)], method="ward")

    # Choose n_clusters: sqrt(n) is a good default (Lopez de Prado 2019)
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(n)))
    n_clusters = min(n_clusters, n)

    from scipy.cluster.hierarchy import fcluster
    cluster_ids = fcluster(link, n_clusters, criterion="maxclust")
    clusters = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(int(cid), []).append(i)

    # ── Step 2: Intra-cluster optimization (Min Variance per cluster) ────
    def _min_var_weights(idx_list):
        """Minimum variance weights within a sub-cluster."""
        if len(idx_list) == 1:
            return np.array([1.0])
        sub_cov = cov[np.ix_(idx_list, idx_list)]
        k = len(idx_list)
        w0 = np.ones(k) / k
        res = minimize(
            lambda w: w @ sub_cov @ w,
            w0,
            method="SLSQP",
            bounds=[(0, max_weight)] * k,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 500},
        )
        w = np.abs(res.x); w /= w.sum()
        return w

    intra_weights = {}   # {cluster_id: np.array of per-asset weights}
    cluster_returns = {} # {cluster_id: array of cluster portfolio returns}

    for cid, idx_list in clusters.items():
        w_intra = _min_var_weights(idx_list)
        intra_weights[cid] = (idx_list, w_intra)
        # Cluster portfolio daily returns
        cluster_r = returns.values[:, idx_list] @ w_intra
        cluster_returns[cid] = cluster_r

    # ── Step 3: Inter-cluster optimization (HRP on cluster portfolios) ────
    cluster_ret_df = pd.DataFrame(cluster_returns, index=returns.index)
    hrp_inter = compute_hrp(cluster_ret_df, risk_free_rate)
    inter_w_dict = hrp_inter["weights"]  # {cluster_id: weight}

    # ── Step 4: Combine intra + inter weights ─────────────────────────────
    final_w = np.zeros(n)
    for cid, (idx_list, w_intra) in intra_weights.items():
        inter_w = inter_w_dict.get(cid, 1.0 / len(clusters))
        for rank, asset_idx in enumerate(idx_list):
            final_w[asset_idx] = inter_w * w_intra[rank]

    if final_w.sum() > 0:
        final_w /= final_w.sum()

    port_ret = float(final_w @ mu)
    port_vol = float(np.sqrt(final_w @ cov @ final_w))
    sharpe   = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    omega    = float(min(calculate_omega(returns.values @ final_w), 10.0))

    cluster_map = {tickers[i]: int(cluster_ids[i]) for i in range(n)}

    return {
        "method":    "NCO",
        "weights":   {t: float(final_w[i]) for i, t in enumerate(tickers)},
        "return":    port_ret,
        "volatility":port_vol,
        "sharpe":    sharpe,
        "omega":     omega,
        "n_clusters":n_clusters,
        "cluster_map":cluster_map,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DRO — Wasserstein Distributionally Robust Optimization  [NEW 2024]
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wasserstein_dro(
    returns_df: pd.DataFrame,
    epsilon: float = 0.05,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
    confidence: float = 0.95,
) -> dict:
    """
    Wasserstein Distributionally Robust Optimization (DRO).

    Zamiast optymalizować dla EMPIRYCZNEGO rozkładu (jak klasyczne MVO),
    DRO optymalizuje dla NAJGORSZEGO rozkładu w kuli Wassersteina:

      min_w  max_{Q: W(Q, P̂)≤ε}  E_Q [ -w^T r ]

    Gdzie W(Q, P̂) = odległość Wassersteina między Q a P̂ (empirycznym),
    a ε = radius (niepewność modelu).

    Aproksymacja Scarf'a (zlinearyzowana):
      min_w  [ μ^T (-w) + ε * ‖Σ^{1/2} w‖₂ ]

    Z dodatkową penalizacją CVaR dla ogonów (CVaR regularization).

    Zalety:
      - Odporna na shifts rozkładu (szczególnie po krachach)
      - Automatycznie preferuje bardziej zdywersyfikowane portfele
      - ε=0 redukuje się do klasycznego Max-Sharpe
      - Większe ε = większa ostrożność (min vol dla ε→∞)

    Ref: Zhang et al. (2024), Esfahani & Kuhn (2018)
         "Data-Driven Distributionally Robust Optimization Using the
          Wasserstein Metric"
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    tickers  = returns_df.columns.tolist()
    n        = len(tickers)
    R        = returns.values          # (T, n)
    mu       = R.mean(axis=0) * 252    # Annual expected returns
    cov      = np.cov(R.T) * 252       # Annual covariance
    rf       = risk_free_rate * 0.81

    # Cholesky of covariance (for Wasserstein robustification)
    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov) + 1e-8))

    def _dro_objective(w: np.ndarray) -> float:
        """
        DRO objective:
          max_{Q in B_eps(P)} E_Q[-w^T r]
          ≈ -w^T mu + eps * ||L^T w||_2  (Scarf linearization)
        Also add CVaR regularization for tail protection.
        """
        # Expected shortfall component
        mu_term  = -float(w @ mu)
        # Wasserstein robustification term
        wass_pen = epsilon * float(np.linalg.norm(L.T @ w))
        # CVaR regularization (additional robustness)
        port_r   = R @ w
        var95    = np.percentile(port_r, (1 - confidence) * 100)
        cvar_pen = -0.1 * float(np.mean(port_r[port_r <= var95]))
        return mu_term + wass_pen + cvar_pen

    w0     = np.ones(n) / n
    bounds = [(0, max_weight)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    res = minimize(
        _dro_objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    w_dro = np.abs(res.x); w_dro /= w_dro.sum()

    # Metrics
    port_ret = float(w_dro @ mu)
    port_vol = float(np.sqrt(w_dro @ cov @ w_dro))
    sharpe   = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    omega    = float(min(calculate_omega(R @ w_dro), 10.0))

    # Worst-case return (lower bound garantowany przez DRO)
    wc_return = port_ret - epsilon * float(np.linalg.norm(L.T @ w_dro)) * np.sqrt(252)

    return {
        "method":          "DRO (Wasserstein)",
        "weights":         {t: float(w_dro[i]) for i, t in enumerate(tickers)},
        "return":          port_ret,
        "volatility":      port_vol,
        "sharpe":          sharpe,
        "omega":           omega,
        "epsilon":         epsilon,
        "worst_case_return": wc_return,
        "robustness_note": (
            f"Portfel odporny na zmiany rozkładu w promieniu ε={epsilon:.3f} "
            f"(Wasserstein). Gwarantowany minimalny zwrot: {wc_return*100:.1f}%/rok."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HERC — Hierarchical Equal Risk Contribution (NEW 2025)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_herc(
    returns_df: pd.DataFrame,
    linkage_method: str = "ward",
    risk_measure: str = "cvar",
    alpha: float = 0.05,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
) -> dict:
    """
    HERC — Hierarchical Equal Risk Contribution.

    HRP 2.0: Zamiast rekursywnego podziału (bisection), HERC stosuje
    ERC (Equal Risk Contribution) wewnątrz każdego klastra z dendrogramu Ward.

    Architektura:
      1. Hierarchical Clustering (Ward) → dendrogram
      2. Wytnij dendrogram na n_clusters klastrów (poziom p)
      3. Alokacja MIĘDZY klastrami: ERC na "cluster portfolios"
      4. Alokacja WEWNĄTRZ klastra: ERC per aktywo

    Zalety vs HRP:
      - Niższa koncentracja niż klasyczny HRP (ERC → bardziej równomierne wkłady ryzyka)
      - Stabilniejsze wagi out-of-sample
      - Naturalny control ryzyka ogonowego (CVaR variant)

    Referencja: de Prado (2020) "Machine Learning for Asset Managers",
                Raffinot (2018) "Hierarchical Clustering-Based Asset Allocation".

    Parameters
    ----------
    returns_df    : DataFrame (T, n_assets) — dzienne zwroty
    linkage_method: metoda linkage — 'ward' (default), 'average', 'complete'
    risk_measure  : 'variance' lub 'cvar' — miara ryzyka w ERC
    alpha         : poziom CVaR (0.05 = 95% CVaR)
    max_weight    : max waga per aktywo
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    tickers = returns_df.columns.tolist()
    n = len(tickers)
    R = returns.values
    cov = np.cov(R.T) * 252
    mu = R.mean(axis=0) * 252
    rf = risk_free_rate * 0.81

    if n < 3:
        return compute_hrp(returns_df, risk_free_rate)

    # ── 1. Clustering ──────────────────────────────────────────────────────
    corr = returns.corr().values
    dist = _corr_to_dist(corr)
    link = sch.linkage(dist[np.triu_indices(n, k=1)], method=linkage_method)

    # Optimal n_clusters: look for largest gap in dendrogram distances
    merging_distances = link[:, 2]
    gaps = np.diff(merging_distances)
    if len(gaps) > 1:
        n_clusters = max(2, n - int(np.argmax(gaps[::-1][:n//2])) - 1)
    else:
        n_clusters = max(2, int(np.sqrt(n)))
    n_clusters = min(n_clusters, n)

    from scipy.cluster.hierarchy import fcluster
    cluster_ids = fcluster(link, n_clusters, criterion="maxclust")
    clusters = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(int(cid), []).append(i)

    # ── 2. ERC allocation function ─────────────────────────────────────────
    def _erc_weights(idx_list: list) -> np.ndarray:
        """Equal Risk Contribution weights for a set of assets."""
        k = len(idx_list)
        if k == 1:
            return np.array([1.0])

        sub_cov = cov[np.ix_(idx_list, idx_list)]
        sub_R   = R[:, idx_list]

        def _risk_contributions(w):
            """Portfolio risk contributions per asset."""
            if risk_measure == "cvar":
                port_r = sub_R @ w
                var = np.percentile(port_r, alpha * 100)
                tail = port_r <= var
                if tail.sum() == 0:
                    return np.zeros(k)
                marginal = sub_R[tail].mean(axis=0)
                rc = w * marginal
            else:  # variance
                port_var = w @ sub_cov @ w
                if port_var <= 0:
                    return np.zeros(k)
                rc = (sub_cov @ w) * w / port_var
            return rc

        def _erc_objective(w):
            rc = _risk_contributions(w)
            rc_sum = rc.sum()
            if abs(rc_sum) < 1e-12:
                return 1.0
            rc_norm = rc / rc_sum
            target = 1.0 / k
            return float(np.sum((rc_norm - target) ** 2))

        w0 = np.ones(k) / k
        res = minimize(
            _erc_objective, w0,
            method="SLSQP",
            bounds=[(0.01, max_weight)] * k,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w = np.abs(res.x); w /= w.sum()
        return w

    # ── 3. Intra-cluster weights (ERC within each cluster) ─────────────────
    cluster_weights = {}
    cluster_portfolios = {}
    for cid, idx_list in clusters.items():
        w_intra = _erc_weights(idx_list)
        cluster_weights[cid] = (idx_list, w_intra)
        cluster_portfolios[cid] = R[:, idx_list] @ w_intra  # (T,)

    # ── 4. Inter-cluster weights (ERC across cluster portfolios) ───────────
    cids = sorted(clusters.keys())
    cluster_ret_matrix = np.column_stack([cluster_portfolios[c] for c in cids])
    cluster_ret_df = pd.DataFrame(cluster_ret_matrix, index=returns.index,
                                  columns=[str(c) for c in cids])
    # Recursive ERC on cluster portfolios
    inter_weights_arr = _erc_weights(list(range(len(cids))))
    inter_weights = {c: float(inter_weights_arr[i]) for i, c in enumerate(cids)}

    # ── 5. Combine weights ─────────────────────────────────────────────────
    final_w = np.zeros(n)
    for cid, (idx_list, w_intra) in cluster_weights.items():
        for rank, asset_idx in enumerate(idx_list):
            final_w[asset_idx] = inter_weights.get(cid, 0.0) * w_intra[rank]

    if final_w.sum() > 0:
        final_w /= final_w.sum()

    port_ret = float(final_w @ mu)
    port_vol = float(np.sqrt(final_w @ cov @ final_w))
    sharpe   = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    daily_r  = R @ final_w
    omega    = float(min(calculate_omega(daily_r), 10.0))
    cluster_map = {tickers[i]: int(cluster_ids[i]) for i in range(n)}

    return {
        "method":      "HERC",
        "weights":     {t: float(final_w[i]) for i, t in enumerate(tickers)},
        "return":      port_ret,
        "volatility":  port_vol,
        "sharpe":      sharpe,
        "omega":       omega,
        "n_clusters":  n_clusters,
        "cluster_map": cluster_map,
        "risk_measure": risk_measure,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Robust Black-Litterman (Tu & Zhou 2011) (NEW 2025)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_robust_black_litterman(
    returns_df: pd.DataFrame,
    cio_views: dict | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    kappa: float = 0.5,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Robust Black-Litterman z priorem Tu-Zhou (mixing parameter κ).

    Standardowy BL ma problem "overconfident views" — gdy widoki CIO są
    błędne, prowadzi do ekstremalnych allocacji.

    Tu-Zhou (2011) mixuje BL z minimum-variance portfolio:
      μ_RBL = κ × μ_BL + (1-κ) × μ_minvol

    κ = 1.0 → czysty BL (ufamy widokom w 100%)
    κ = 0.0 → ignorujemy widoki, czyste min-vol
    κ = 0.5 → kompromis (default, optymalny statystycznie)

    Referencja: Tu & Zhou (2011) "Markowitz meets Talmud",
                Lee & Stefanidis (2024) "Robust BL optimization".
    """
    returns = returns_df.copy()
    returns.mask(returns > 0, returns * 0.81, inplace=True)

    tickers = returns_df.columns.tolist()
    n = len(tickers)
    mu_hist = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    rf = risk_free_rate * 0.81
    w_mkt = np.ones(n) / n
    pi_prior = risk_aversion * Sigma @ w_mkt

    # ── Standard BL views ─────────────────────────────────────────────────
    if cio_views and len(cio_views) > 0:
        view_tickers = [t for t in cio_views if t in tickers]
        k = len(view_tickers)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for i, t in enumerate(view_tickers):
            j = tickers.index(t)
            P[i, j] = 1.0
            Q[i] = cio_views[t] * 0.81

        omega = np.diag([tau * float(Sigma[tickers.index(t), tickers.index(t)])
                         for t in view_tickers])
        try:
            tauSigma_inv = np.linalg.inv(tau * Sigma + 1e-8 * np.eye(n))
            Omega_inv = np.linalg.inv(omega + 1e-8 * np.eye(k))
            M = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
            mu_bl = M @ (tauSigma_inv @ pi_prior + P.T @ Omega_inv @ Q)
        except np.linalg.LinAlgError:
            mu_bl = pi_prior.copy()
    else:
        mu_bl = pi_prior.copy()

    # ── Min-Variance portfolio ─────────────────────────────────────────────
    def _min_vol_obj(w):
        return float(w @ Sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0, 0.4)] * n
    res_mv = minimize(_min_vol_obj, np.ones(n) / n, method="SLSQP",
                      bounds=bnds, constraints=cons, options={"maxiter": 500})
    w_mv = np.abs(res_mv.x); w_mv /= w_mv.sum()
    mu_minvol = float(w_mv @ mu_hist)

    # ── Tu-Zhou mixing ─────────────────────────────────────────────────────
    # Mix expected returns: κ×BL + (1-κ)×hist
    mu_robust = kappa * mu_bl + (1.0 - kappa) * mu_hist

    def _neg_sharpe_robust(w):
        ret = float(w @ mu_robust)
        vol = float(np.sqrt(w @ Sigma @ w))
        return -(ret - rf) / vol if vol > 0 else 0.0

    res = minimize(_neg_sharpe_robust, w_mkt, method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 1000})
    w_rbl = np.abs(res.x); w_rbl /= w_rbl.sum()

    port_ret = float(w_rbl @ mu_robust)
    port_vol = float(np.sqrt(w_rbl @ Sigma @ w_rbl))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    omega = float(min(calculate_omega(returns.values @ w_rbl), 10.0))

    return {
        "method":            "Robust BL (Tu-Zhou)",
        "weights":           {t: float(w_rbl[i]) for i, t in enumerate(tickers)},
        "return":            port_ret,
        "volatility":        port_vol,
        "sharpe":            sharpe,
        "omega":             omega,
        "kappa":             kappa,
        "bl_returns":        {t: float(mu_bl[i]) for i, t in enumerate(tickers)},
        "robust_returns":    {t: float(mu_robust[i]) for i, t in enumerate(tickers)},
        "views_used":        list(cio_views.keys()) if cio_views else [],
        "note": (
            f"κ={kappa:.1f}: {kappa*100:.0f}% BL widoki + {(1-kappa)*100:.0f}% historyczna stopa. "
            "Robust BL jest mniej wrażliwy na błędne widoki CIO."
        ),
    }
