"""
walk_forward.py — Walk-Forward Validation + Bootstrap CI
Implementuje rygorystyczny backtesting z walidacją poza próbą.

Walk-Forward Validation (WFV):
  Podział rolujący: [Train N_train][Test N_test] → [Train][Test] → ...
  Każde okno treningowe przesuwa się o N_test do przodu.
  Metryki są zbierane tylko z okien testowych (out-of-sample).

Bootstrap Confidence Intervals:
  BCa bootstrap (bias-corrected accelerated) — najdokładniejsza metoda.
  Dla metryk: Sharpe, CAGR, Max Drawdown, Omega, Sortino.

Referencje:
  - Prado M.L. de (2018). Advances in Financial Machine Learning.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
  - Kiefer & Vogelsang (2005). A New Asymptotic Theory for HAC.
"""
import numpy as np
import pandas as pd
from typing import Callable
import warnings
from modules.logger import setup_logger

logger = setup_logger(__name__)

try:
    from modules.metrics import (
        calculate_sharpe, calculate_sortino, calculate_calmar,
        calculate_max_drawdown, calculate_omega,
        calculate_sterling_ratio, calculate_rachev_ratio,
    )
except ImportError:
    # Minimal fallback
    def calculate_sharpe(r, rf=0.04, periods=252):
        return (np.mean(r) * periods - rf) / (np.std(r) * np.sqrt(periods) + 1e-10)
    def calculate_sortino(r, rf=0.04, periods=252, target_return=0.0):
        dd = np.minimum(r - target_return, 0)
        return (np.mean(r)*periods - rf) / (np.sqrt(np.mean(dd**2))*np.sqrt(periods) + 1e-10)
    def calculate_max_drawdown(prices):
        peaks = np.maximum.accumulate(prices)
        return float(np.min((prices - peaks) / (peaks + 1e-10)))
    def calculate_omega(r, threshold=0.0):
        excess = r - threshold
        g = np.sum(np.maximum(excess, 0))
        l = np.sum(np.maximum(-excess, 0))
        return g / l if l > 0 else np.inf
    def calculate_sterling_ratio(cagr, prices): return cagr
    def calculate_rachev_ratio(r): return 1.0


# ── Metryki zbierane per okno ─────────────────────────────────────────────────

def _compute_window_metrics(
    equity: np.ndarray,
    n_years: float,
    rf: float = 0.04,
) -> dict:
    """Compute full metric set for one WFV test window."""
    if len(equity) < 5 or equity[0] <= 0:
        return {}

    returns  = np.diff(equity) / (equity[:-1] + 1e-10)
    cagr     = float((equity[-1] / equity[0]) ** (1 / max(n_years, 1e-3)) - 1)
    mdd      = float(calculate_max_drawdown(equity))
    sharpe   = float(calculate_sharpe(returns, rf=rf * 0.81))
    sortino  = float(calculate_sortino(returns, rf=rf * 0.81))
    omega    = float(min(calculate_omega(returns), 20.0))
    calmar   = cagr / abs(mdd) if mdd != 0 else 0.0
    sterling = float(calculate_sterling_ratio(cagr, equity))
    rachev   = float(min(calculate_rachev_ratio(returns), 20.0))
    vol      = float(np.std(returns) * np.sqrt(252))
    total_return = float(equity[-1] / equity[0] - 1)

    return {
        "cagr":         cagr,
        "total_return": total_return,
        "mdd":          mdd,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "omega":        omega,
        "calmar":       calmar,
        "sterling":     sterling,
        "rachev":       rachev,
        "volatility":   vol,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(
    equity_curve: pd.Series,
    train_days: int = 504,    # 2 years
    test_days: int  = 126,    # 6 months
    step_days: int  | None = None,  # default = test_days (non-overlapping)
    rf: float = 0.04,
    strategy_fn: Callable | None = None,
) -> dict:
    """
    Walk-Forward Validation of an equity curve (or strategy).

    Parameters
    ----------
    equity_curve  : pd.Series — portfolio value over time
    train_days    : training window length (default: 504 = 2 years)
    test_days     : test window length (default: 126 = 6 months)
    step_days     : step size between windows (default: test_days)
    rf            : annual risk-free rate (pre-tax)
    strategy_fn   : optional callable(train_prices, test_prices) → equity
                    If None, uses raw equity_curve slices as-is.

    Returns
    -------
    dict with:
      windows         : list of per-window metric dicts
      aggregate       : mean ± std of each metric across OOS windows
      oos_equity      : concatenated out-of-sample equity curve
      n_windows       : number of WFV windows
    """
    if step_days is None:
        step_days = test_days

    prices = equity_curve.values
    dates  = equity_curve.index
    n      = len(prices)
    windows = []
    oos_pieces = []

    start = 0
    while start + train_days + test_days <= n:
        train_end   = start + train_days
        test_end    = train_end + test_days
        test_end    = min(test_end, n)

        train_prices = prices[start:train_end]
        test_prices  = prices[train_end:test_end]
        test_dates   = dates[train_end:test_end]

        if strategy_fn is not None:
            try:
                test_equity = strategy_fn(train_prices, test_prices)
            except Exception as e:
                logger.error(f"Błąd evaluacji funkcji strategii: {e}")
                test_equity = test_prices
        else:
            test_equity = test_prices

        n_test_years = len(test_equity) / 252.0
        m = _compute_window_metrics(np.asarray(test_equity, dtype=float), n_test_years, rf)

        if m:
            m["window_start"] = str(dates[start].date()) if hasattr(dates[start], 'date') else str(start)
            m["window_test_start"] = str(test_dates[0].date()) if len(test_dates) > 0 and hasattr(test_dates[0], 'date') else str(train_end)
            m["window_test_end"]   = str(test_dates[-1].date()) if len(test_dates) > 0 and hasattr(test_dates[-1], 'date') else str(test_end)
            windows.append(m)

        # Build OOS equity
        # Normalize to start from last OOS value for continuity
        if oos_pieces and len(test_equity) > 0 and test_equity[0] != 0:
            scale = oos_pieces[-1][-1] / test_equity[0]
        else:
            scale = 1.0
        oos_pieces.append(np.asarray(test_equity) * scale)

        start += step_days

    if not windows:
        return {"error": "Za mało danych dla Walk-Forward (min train+test)."}

    # Aggregate metrics
    metric_keys = [k for k in windows[0] if isinstance(windows[0][k], float)]
    aggregate = {}
    for key in metric_keys:
        vals = np.array([w[key] for w in windows if key in w], dtype=float)
        aggregate[key] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "median": float(np.median(vals)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
            "n":      len(vals),
        }

    oos_equity = np.concatenate(oos_pieces) if oos_pieces else np.array([])

    return {
        "windows":     windows,
        "aggregate":   aggregate,
        "oos_equity":  oos_equity,
        "n_windows":   len(windows),
        "train_days":  train_days,
        "test_days":   test_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def _bca_ci(
    data: np.ndarray,
    stat_fn: Callable,
    n_boot: int = 2000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Bias-Corrected and Accelerated (BCa) Bootstrap CI.
    Returns (lower, estimate, upper).
    Referencja: Efron (1987), DiCiccio & Efron (1996).
    """
    from scipy.stats import norm as _norm

    n     = len(data)
    obs   = stat_fn(data)
    boots = np.array([stat_fn(data[np.random.choice(n, n, replace=True)]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]

    if len(boots) < 10:
        return float(obs), float(obs), float(obs)

    # Bias correction (z0)
    z0 = _norm.ppf(np.mean(boots < obs) + 1e-6)

    # Acceleration (a) via jackknife
    jk_stats = np.array([stat_fn(np.delete(data, i)) for i in range(min(n, 200))])
    jk_mean  = np.mean(jk_stats)
    num   = np.sum((jk_mean - jk_stats) ** 3)
    denom = 6 * (np.sum((jk_mean - jk_stats) ** 2) ** 1.5 + 1e-10)
    a     = num / denom

    alpha = (1 - ci_level) / 2
    z_lo  = _norm.ppf(alpha)
    z_hi  = _norm.ppf(1 - alpha)

    p_lo  = _norm.cdf(z0 + (z0 + z_lo)  / (1 - a * (z0 + z_lo)))
    p_hi  = _norm.cdf(z0 + (z0 + z_hi)  / (1 - a * (z0 + z_hi)))

    lo  = float(np.percentile(boots, max(0, min(100, p_lo * 100))))
    hi  = float(np.percentile(boots, max(0, min(100, p_hi * 100))))
    return lo, float(obs), hi


def bootstrap_ci(
    equity_curve,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    block_size: int = 21,
    rf: float = 0.04,
) -> dict:
    """
    BCa Bootstrap Confidence Intervals for key portfolio metrics.
    Uses BLOCK bootstrap (preserves autocorrelation in financial returns).

    Parameters
    ----------
    equity_curve : array-like — portfolio values
    n_boot       : number of bootstrap samples
    ci_level     : confidence level (default 0.95 → 95% CI)
    block_size   : block length for block bootstrap (default 21 = 1 month)
    rf           : annual risk-free rate (pre-tax)

    Returns
    -------
    dict {metric: {'lo': float, 'est': float, 'hi': float}}
    """
    prices  = np.asarray(equity_curve, dtype=float)
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    n       = len(returns)

    if n < 30:
        return {"error": "Za mało obserwacji (min 30)."}

    # Block bootstrap: resample blocks of returns
    def _block_resample(r: np.ndarray) -> np.ndarray:
        n_r = len(r)
        n_blocks = int(np.ceil(n_r / block_size))
        starts = np.random.randint(0, max(1, n_r - block_size), size=n_blocks)
        blocks = [r[s: s + block_size] for s in starts]
        return np.concatenate(blocks)[:n_r]

    def _boot_returns(r: np.ndarray) -> np.ndarray:
        return _block_resample(r)

    rf_daily = (1 + rf * 0.81) ** (1 / 252) - 1

    # Metric functions
    def _sharpe(r):
        excess = r - rf_daily
        return np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252)

    def _cagr(r):
        return float((np.prod(1 + r)) ** (252 / max(len(r), 1)) - 1)

    def _mdd(r):
        p = np.cumprod(1 + r)
        pk = np.maximum.accumulate(p)
        return float(np.min((p - pk) / (pk + 1e-10)))

    def _omega(r):
        g = np.sum(np.maximum(r, 0))
        l = np.sum(np.maximum(-r, 0))
        return min(g / l, 20.0) if l > 0 else 20.0

    def _sortino(r):
        dd = np.minimum(r - 0.0, 0)
        ds = np.sqrt(np.mean(dd ** 2)) * np.sqrt(252) + 1e-10
        return (np.mean(r) * 252 - rf * 0.81) / ds

    # Generate boot samples once
    boot_samples = [_boot_returns(returns) for _ in range(n_boot)]

    results = {}
    for metric_name, fn in [
        ("sharpe",  _sharpe),
        ("cagr",    _cagr),
        ("mdd",     _mdd),
        ("omega",   _omega),
        ("sortino", _sortino),
    ]:
        obs   = fn(returns)
        boots = np.array([fn(b) for b in boot_samples])
        boots = boots[np.isfinite(boots)]

        if len(boots) < 10:
            results[metric_name] = {"lo": obs, "est": obs, "hi": obs, "ci_level": ci_level}
            continue

        alpha = (1 - ci_level) / 2
        lo = float(np.percentile(boots, alpha * 100))
        hi = float(np.percentile(boots, (1 - alpha) * 100))
        results[metric_name] = {
            "lo":       lo,
            "est":      float(obs),
            "hi":       hi,
            "std":      float(np.std(boots)),
            "ci_level": ci_level,
            "n_boot":   len(boots),
        }

    return results


# ── Plotly summary charts ─────────────────────────────────────────────────────

def plot_wfv_results(wfv_result: dict) -> "go.Figure":
    """Bar chart: per-window Sharpe ratio with mean line."""
    import plotly.graph_objects as go

    windows = wfv_result.get("windows", [])
    if not windows:
        return go.Figure()

    sharpes = [w.get("sharpe", 0) for w in windows]
    labels  = [w.get("window_test_start", str(i)) for i, w in enumerate(windows)]
    colors  = ["#00ff88" if s > 0 else "#ff4444" for s in sharpes]
    mean_s  = wfv_result["aggregate"].get("sharpe", {}).get("mean", 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=sharpes,
        marker_color=colors,
        text=[f"{s:.2f}" for s in sharpes],
        textposition="outside",
        name="Sharpe (OOS)",
    ))
    fig.add_hline(y=mean_s, line_dash="dash", line_color="gold",
                  annotation_text=f"Śr. OOS={mean_s:.2f}")
    fig.add_hline(y=0, line_color="red", line_width=1)
    fig.update_layout(
        title="Walk-Forward Validation — Sharpe Ratio (Out-of-Sample)",
        xaxis_title="Okno Testowe (start)",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark", height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        font=dict(color="white"),
    )
    return fig


def plot_bootstrap_ci(ci_result: dict) -> "go.Figure":
    """Horizontal error bar chart: metric estimate ± bootstrap CI."""
    import plotly.graph_objects as go

    if "error" in ci_result:
        return go.Figure()

    metrics = list(ci_result.keys())
    ests = [ci_result[m]["est"] for m in metrics]
    los  = [ci_result[m]["lo"]  for m in metrics]
    his  = [ci_result[m]["hi"]  for m in metrics]
    err_lo = [e - l for e, l in zip(ests, los)]
    err_hi = [h - e for e, h in zip(ests, his)]

    fig = go.Figure(go.Scatter(
        x=ests, y=metrics,
        mode="markers",
        marker=dict(color="#00ff88", size=12),
        error_x=dict(
            type="data",
            symmetric=False,
            arrayminus=err_lo,
            array=err_hi,
            color="rgba(0,255,136,0.5)",
            thickness=3,
            width=8,
        ),
        name="Estimate ± CI",
    ))
    fig.update_layout(
        title=f"Bootstrap Confidence Intervals ({ci_result[metrics[0]]['ci_level']:.0%} BCa)",
        xaxis_title="Wartość Metryki",
        template="plotly_dark", height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        font=dict(color="white"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO  (Bailey & Lopez de Prado 2014)  [NEW]
# ══════════════════════════════════════════════════════════════════════════════

def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
    rf: float = 0.04,
    periods: int = 252,
) -> dict:
    """
    Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado (2014).

    PSR = Φ[ (SR - SR*) * sqrt(T-1) / sqrt(1 - γ₃·SR + (γ₄-1/4)·SR²) ]

    Gdzie:
      SR  = estimated annualized Sharpe Ratio
      SR* = benchmark Sharpe (zazwyczaj 0 lub poprzedni najlepszy)
      γ₃  = skewness zwrotów (korekta nienormalności)
      γ₄  = kurtosis zwrotów (korekta grubych ogonów)
      T   = liczba obserwacji
      Φ   = CDF normalnej standardowej

    Interpretacja: PSR > 0.95 → strategia jest statystycznie lepsza
                               od SR* z 95% pewnością.

    Ref: Bailey, D.H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
         Frontier. Journal of Risk, 15(2), 3-44.
    """
    from scipy.stats import norm as _norm
    from scipy.stats import skew as _skew, kurtosis as _kurt

    r   = np.asarray(returns, dtype=float)
    r   = r[np.isfinite(r)]
    T   = len(r)

    if T < 30:
        return {"error": "Za mało obs. dla PSR (min 30)."}

    # Annualized Sharpe from sample
    rf_daily  = (1 + rf * 0.81) ** (1 / periods) - 1
    excess    = r - rf_daily
    sr_hat    = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12)) * np.sqrt(periods)

    # Higher moments
    sk = float(_skew(r))
    ku = float(_kurt(r, fisher=False))  # non-excess kurtosis (so Normal → 3)

    # PSR formula
    sr_adj = benchmark_sharpe / np.sqrt(periods)  # daily SR benchmark
    sr_hat_daily = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12))

    variance = (1 - sk * sr_hat_daily + (ku - 1) / 4 * sr_hat_daily ** 2) / (T - 1)
    if variance <= 0:
        variance = 1e-10

    z = (sr_hat_daily - sr_adj) / np.sqrt(variance)
    psr = float(_norm.cdf(z))

    return {
        "psr":             psr,
        "sr_hat":          sr_hat,
        "benchmark_sr":    benchmark_sharpe,
        "skewness":        sk,
        "kurtosis":        ku,
        "n_obs":           T,
        "interpretation": (
            f"P(SR > {benchmark_sharpe:.2f}) = {psr:.1%} "
            f"({'✅ Istotne' if psr > 0.95 else '⚠️ Nieistotne'} at 95%)"
        ),
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    rf: float = 0.04,
    periods: int = 252,
) -> dict:
    """
    Deflated Sharpe Ratio (DSR) — korekta Sharpe'a za wielokrotne testowanie.

    Problem: jeśli przetestowałeś N strategii, OCZEKIWANA wartość maksymalnego
    Sharpe'a z samego przypadku (data snooping) rośnie z N.

    DSR = PSR ze SR* = E[max_SR(N, T, σ)] = optymalny benchmark po N próbach.

    E[max] ≈ σ * [(1 - γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))]
    gdzie σ = std zwrotów, γ = stała Eulera–Mascheroniego ≈ 0.5772.

    Ref: Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
         Financial Analysts Journal, 70(5), 94-107.
    """
    from scipy.stats import norm as _norm

    r   = np.asarray(returns, dtype=float)
    r   = r[np.isfinite(r)]
    T   = len(r)

    if T < 60:
        return {"error": "Za mało obs. dla DSR (min 60)."}
    if n_trials < 1:
        return {"error": "n_trials musi być >= 1."}

    gamma_em = 0.5772156649  # Euler–Mascheroni constant

    # Expected max Sharpe under repeated testing (Bailey & de Prado 2014)
    if n_trials == 1:
        e_max_sr = 0.0
    else:
        e_max_sr = float(
            (1 - gamma_em) * _norm.ppf(1 - 1 / n_trials)
            + gamma_em * _norm.ppf(1 - 1 / (n_trials * np.e))
        )

    # Annualise the benchmark
    benchmark_sr = max(0.0, e_max_sr)
    psr_result = probabilistic_sharpe_ratio(returns, benchmark_sr, rf, periods)

    if "error" in psr_result:
        return psr_result

    psr_result["dsr_benchmark"] = benchmark_sr
    psr_result["n_trials"]      = n_trials
    psr_result["dsr_interpretation"] = (
        f"Po testowaniu {n_trials} strategii, oczekiwany Sharpe z przypadku = {benchmark_sr:.2f}. "
        f"P(SR > benchmark) = {psr_result['psr']:.1%} "
        f"({'✅ Prawdziwa przewaga' if psr_result['psr'] > 0.95 else '❌ Może być data snooping'})"
    )
    return psr_result
