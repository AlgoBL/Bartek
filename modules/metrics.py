import numpy as np
from scipy.stats import norm
import pandas as pd

def calculate_sharpe(returns, rf=0.0324, periods=252):
    """
    Calculates Sharpe Ratio: (Mean Return - Risk Free) / Volatility
    """
    if len(returns) < 2: return 0.0
    
    # Convert annual RF to daily if needed, here we assume rf is annual 4%
    # Excess daily return = daily_return - (rf / 252)
    rf_daily = (1 + rf)**(1/periods) - 1
    excess_returns = returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0: return 0.0
    
    return (mean_excess / std_excess) * np.sqrt(periods)

def calculate_sortino(returns, rf=0.0324, periods=252, target_return=0.0):
    """
    Calculates Sortino Ratio: (Mean Return - Risk Free) / Downside Deviation
    """
    if len(returns) < 2: return 0.0
    
    rf_daily = (1 + rf)**(1/periods) - 1
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation relative to target (usually 0 or rf)
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.inf # No downside risk
        
    # We calculate geometric downside deviation or standard deviation of negative returns
    # Sortino denominator is sqrt(mean(min(0, R - target)^2))
    downside_diff = np.minimum(0, returns - target_return)**2
    downside_dev = np.sqrt(np.mean(downside_diff))
    
    if downside_dev == 0: return np.inf
    
    return (mean_excess / downside_dev) * np.sqrt(periods)

def calculate_calmar(cagr, max_drawdown):
    """
    Calculates Calmar Ratio: CAGR / abs(Max Drawdown)
    """
    if max_drawdown == 0: return np.inf
    return cagr / abs(max_drawdown)

def calculate_max_drawdown(prices):
    """
    Calculates Max Drawdown from a price series.
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) == 0:
        return 0.0
    
    peaks = np.maximum.accumulate(prices)
    drawdowns = (prices - peaks) / peaks
    return np.min(drawdowns)

def calculate_var_cvar(returns, confidence=0.95):
    """
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    Returns (VaR, CVaR) as positive percentages (losses).
    """
    if len(returns) == 0: return 0.0, 0.0
    
    # VaR is the quantile
    # e.g. 95% confidence means looking at 5% worst returns
    cutoff = (1.0 - confidence) * 100
    var = np.percentile(returns, cutoff)
    
    # CVaR is the mean of returns worse than VaR
    cvar = returns[returns <= var].mean()
    
    return var, cvar

def calculate_alpha_beta(asset_returns, benchmark_returns, periods=252):
    """
    Calculates Alpha and Beta relative to a benchmark.
    Returns (Alpha_Annual, Beta).
    """
    # Align data
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30: return 0.0, 1.0 # Not enough data
    
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    
    covariance = np.cov(x, y)[0][1]
    variance = np.var(x)
    
    if variance == 0: return 0.0, 0.0
    
    beta = covariance / variance
    
    # Alpha (Jensen's Alpha) = R_p - [R_f + Beta * (R_m - R_f)]
    # Simplified Alpha from regression intercept: y = alpha + beta*x
    # alpha_daily = mean(y) - beta * mean(x)
    alpha_daily = np.mean(y) - beta * np.mean(x)
    alpha_annual = alpha_daily * periods
    
    return alpha_annual, beta

def calculate_information_ratio(asset_returns, benchmark_returns):
    """
    Calculates Information Ratio: (R_p - R_b) / Tracking Error
    """
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30: return 0.0
    
    diff = asset_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    
    mean_active_return = np.mean(diff)
    tracking_error = np.std(diff)
    
    if tracking_error == 0: return 0.0
    
    return (mean_active_return / tracking_error) * np.sqrt(252)

def calculate_trade_stats(equity_curve):
    """
    Approximates trading stats from equity curve (daily resolution).
    Ideally needs trade logs, but we can infer 'winning days' vs 'losing days'.
    """
    returns = equity_curve.pct_change().dropna()
    
    winning_days = returns[returns > 0]
    losing_days = returns[returns < 0]
    
    win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
    
    avg_win = winning_days.mean() if len(winning_days) > 0 else 0
    avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 0
    
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    gross_win = winning_days.sum()
    gross_loss = abs(losing_days.sum())
    
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward": risk_reward,
        "profit_factor": profit_factor
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW SCIENTIFIC METRICS (2024 upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_omega(returns, threshold=0.0):
    """
    Omega Ratio (Shadwick & Keating, 2002).
    Omega = Integral of (1-F(r)) dr above threshold
           / Integral of F(r) dr below threshold
    Simplified as: sum of gains above threshold / sum of losses below threshold.
    Omega > 1 means more weighted return above threshold than below.
    Perfect for Barbell strategies with asymmetric return profiles.
    """
    if len(returns) == 0:
        return 1.0
    excess = returns - threshold
    gains = np.sum(np.maximum(excess, 0))
    losses = np.sum(np.maximum(-excess, 0))
    if losses == 0:
        return np.inf
    return gains / losses


def calculate_ulcer_index(prices):
    """
    Ulcer Index (Martin & McCann, 1989).
    Measures the depth and duration of drawdowns — investor 'pain'.
    UI = sqrt(mean(drawdown_pct^2))
    Unlike Max Drawdown, UI punishes prolonged drawdowns, not just deep ones.
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)
    if len(arr) < 2:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdown_pct = ((arr - peaks) / peaks) * 100  # as percentage
    return np.sqrt(np.mean(drawdown_pct ** 2))


def calculate_pain_index(prices):
    """
    Pain Index — average drawdown depth over the entire period.
    Simpler than Ulcer Index but useful as companion metric.
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)
        
    if len(arr) == 0:
        return 0.0
        
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return np.mean(np.abs(drawdowns))


def calculate_drawdown_analytics(prices):
    """
    Full drawdown analytics suite.
    Returns dict with:
    - max_drawdown: deepest single drawdown
    - avg_drawdown_depth: average depth across all drawdown periods
    - avg_drawdown_duration: average number of days to recover
    - ulcer_index: Martin & McCann UI
    - pain_index: mean absolute drawdown
    - drawdown_at_risk_95: worst 5% case drawdown (analogous to CVaR)
    Reference: Magdon-Ismail & Atiya (2004), Chekhlov et al. (2005)
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)

    if len(arr) == 0:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown_depth": 0.0,
            "avg_drawdown_duration_days": 0.0,
            "max_drawdown_duration_days": 0,
            "ulcer_index": 0.0,
            "pain_index": 0.0,
            "drawdown_at_risk_95": 0.0,
        }

    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks  # non-positive values

    # Max drawdown
    max_dd = float(np.min(drawdowns))

    # Ulcer Index and Pain Index
    dd_pct = drawdowns * 100
    ulcer = float(np.sqrt(np.mean(dd_pct ** 2)))
    pain = float(np.mean(np.abs(drawdowns)))

    # Drawdown-at-Risk 95% (worst 5% of daily drawdown values)
    dd_at_risk_95 = float(np.percentile(drawdowns, 5))  # 5th percentile (most negative)

    # Drawdown periods: duration analysis
    in_drawdown = drawdowns < 0
    durations = []
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        durations.append(current_duration)

    avg_duration = float(np.mean(durations)) if durations else 0.0
    max_duration = int(np.max(durations)) if durations else 0

    return {
        "max_drawdown": max_dd,
        "avg_drawdown_depth": float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0,
        "avg_drawdown_duration_days": avg_duration,
        "max_drawdown_duration_days": max_duration,
        "ulcer_index": ulcer,
        "pain_index": pain,
        "drawdown_at_risk_95": dd_at_risk_95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NOWE METRYKI NAUKOWE v3.0 (2026 upgrade)
# Referencje: Bailey & de Prado (2012), Rachev et al. (2007), Sortino & Satchell (2001)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sterling_ratio(cagr: float, prices) -> float:
    """
    Sterling Ratio = CAGR / |Average Drawdown Depth|.
    Lepsza od Calmar: uwzględnia średnią głębokość, nie tylko max.
    Referencja: Sortino & Satchell (2001).
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.asarray(prices, dtype=float)
        
    if len(arr) == 0:
        return 0.0
        
    peaks = np.maximum.accumulate(arr)
    dds   = (arr - peaks) / (peaks + 1e-10)
    avg_dd = float(np.mean(dds[dds < 0])) if np.any(dds < 0) else -1e-10
    return cagr / abs(avg_dd) if avg_dd != 0 else np.inf


def calculate_burke_ratio(
    returns,
    cagr: float,
    rf: float = 0.04,
    modified: bool = True,
) -> float:
    """
    Burke Ratio = (CAGR - RF) / sqrt(sum(DD_i^2)).
    Modified = True → divides by sqrt(T) for length-independence.
    Penalizuje wiele drawdownów, nie tylko największy.
    Referencja: Burke (1994).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    # Reconstruct price series from returns
    prices = np.cumprod(1 + returns)
    if len(prices) == 0:
        return 0.0
    peaks  = np.maximum.accumulate(prices)
    dds    = (prices - peaks) / (peaks + 1e-10)
    sum_dd_sq = float(np.sum(dds ** 2))
    if sum_dd_sq <= 0:
        return np.inf
    denominator = np.sqrt(sum_dd_sq / len(returns)) if modified else np.sqrt(sum_dd_sq)
    return (cagr - rf * 0.81) / denominator if denominator > 0 else 0.0


def calculate_rachev_ratio(
    returns,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> float:
    """
    Rachev Ratio = ETL_alpha(gain) / ETL_beta(loss).
    ETL = Expected Tail Loss (= CVaR).

    Mierzy asymetrię ogonów — IDEALNA dla strategii Barbell:
      Rachev >> 1 → gruby prawy ogon (zyski) dominuje nad lewym (straty).
    Referencja: Biglova et al. (2004), Rachev et al. (2007).
    """
    returns = np.asarray(returns, dtype=float)
    # Gain tail (upper alpha quantile)
    var_gain = np.percentile(returns, (1 - alpha) * 100)
    upper    = returns[returns >= var_gain]
    etl_gain = float(np.mean(upper)) if len(upper) > 0 else 0.0

    # Loss tail (lower beta quantile)
    var_loss = np.percentile(returns, beta * 100)
    lower    = returns[returns <= var_loss]
    etl_loss = float(np.mean(np.abs(lower))) if len(lower) > 0 else 1e-10

    return etl_gain / etl_loss if etl_loss > 0 else np.inf


def calculate_probabilistic_sharpe(
    observed_sr: float,
    benchmark_sr: float,
    n: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Probabilistic Sharpe Ratio — P(SR* > SR_benchmark).
    Uwzględnia skewness i kurtosis przy ocenie istotności statystycznej SR.
    Referencja: Bailey & de Prado (2012).

    PSR > 0.95 → SR jest statystycznie istotne (p > 5%).
    """
    if n < 2:
        return 0.5
    sr_std = np.sqrt(
        (1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr ** 2)
        / (n - 1)
    )
    if sr_std <= 0:
        return 1.0 if observed_sr > benchmark_sr else 0.0
    z = (observed_sr - benchmark_sr) / sr_std
    return float(norm.cdf(z))


def calculate_deflated_sharpe(
    observed_sr: float,
    n: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio — korekta na data mining / multiple testing.
    DSR = PSR(SR_benchmark_max), gdzie SR_benchmark uwzględnia liczbę testów.
    Referencja: Bailey & de Prado (2012) — "The Sharpe Ratio Efficient Frontier".

    DSR bliskie 0 → SR to prawdopodobnie efekt przeszukiwania danych (p-hacking).
    """
    # Expected maximum SR under multiple testing (approx. Extreme Value Theory)
    if n_trials < 1:
        n_trials = 1
    euler_gamma = 0.5772156649
    sr_expected_max = (
        (1 - euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return calculate_probabilistic_sharpe(
        observed_sr, sr_expected_max, n, skewness, kurtosis
    )


def calculate_marginal_cvar(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Marginal CVaR = ∂CVaR/∂w_i.
    Wkład każdego aktywa w ryzyko ogonowe portfela.

    Metodologia: komponent CVaR = w_i × β_i,
    gdzie β_i = E[r_i | r_portfolio <= VaR].
    Referencja: Scaillet (2004), Rockafellar & Uryasev (2000).

    Returns vector of length n_assets.
    """
    weights   = np.asarray(weights, dtype=float)
    port_ret  = returns_matrix @ weights
    var_level = np.percentile(port_ret, alpha * 100)
    tail_mask = port_ret <= var_level

    if tail_mask.sum() == 0:
        return np.zeros(len(weights))

    # Component CVaR: E[r_i | tail] × w_i
    tail_returns   = returns_matrix[tail_mask]
    mean_tail      = tail_returns.mean(axis=0)        # (n_assets,)
    marginal_cvar  = weights * mean_tail               # component contribution
    return marginal_cvar


def calculate_tci(
    returns_matrix: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Tail Correlation Index (TCI) — kondycjonalna korelacja w ogonach.
    TCI_ij = Corr(r_i, r_j | r_i <= VaR_i  AND  r_j <= VaR_j).

    Wysoki TCI → aktywa crash razem (ryzyko systemowe, bad for Barbell).
    Niski TCI  → dywersyfikacja utrzymuje się w kryzysach (good for Barbell).
    Referencja: Joe (1997), Embrechts et al. (2002).

    Returns (n_assets, n_assets) matrix.
    """
    R = np.asarray(returns_matrix, dtype=float)
    n = R.shape[1]
    # VaR per asset (alpha-quantile)
    var_per_asset = np.percentile(R, alpha * 100, axis=0)

    tci = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Joint tail: both in lower alpha-tail simultaneously
            mask = (R[:, i] <= var_per_asset[i]) & (R[:, j] <= var_per_asset[j])
            if mask.sum() < 5:
                tci[i, j] = tci[j, i] = 0.0
                continue
            r_i = R[mask, i]
            r_j = R[mask, j]
            if r_i.std() > 0 and r_j.std() > 0:
                c = float(np.corrcoef(r_i, r_j)[0, 1])
            else:
                c = 0.0
            tci[i, j] = tci[j, i] = c
    return tci


# ─────────────────────────────────────────────────────────────────────────────
# NOWE METRYKI NAUKOWE v4.0 (2025 upgrade — Modernizacja Plan)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_range_var_cvar(
    returns,
    alpha_lo: float = 0.95,
    alpha_hi: float = 0.99,
) -> dict:
    """
    Range VaR i Range CVaR — uśrednianie między dwoma poziomami ufności.

    Mniej wrażliwe na wybór pojedynczego α niż klasyczne VaR/CVaR.
    Rekomendacja EBA Stress Testing Guidelines 2024, Cont et al. (2010).

    Formuła:
      Range_VaR(α_lo, α_hi) = (1/(α_hi - α_lo)) ∫_{α_lo}^{α_hi} VaR_α dα
      Range_CVaR              = mean(CVaR_α) dla α ∈ [α_lo, α_hi]

    Returns dict z keys: range_var, range_cvar, var_lo, var_hi, cvar_lo, cvar_hi
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {k: 0.0 for k in ["range_var", "range_cvar", "var_lo", "var_hi", "cvar_lo", "cvar_hi"]}

    n_steps = 20
    alphas = np.linspace(alpha_lo, alpha_hi, n_steps)
    vars_, cvars_ = [], []
    for a in alphas:
        cutoff_pct = (1.0 - a) * 100
        v = np.percentile(returns, cutoff_pct)
        c = returns[returns <= v].mean() if np.any(returns <= v) else v
        vars_.append(v)
        cvars_.append(c)

    return {
        "range_var":  float(np.mean(vars_)),
        "range_cvar": float(np.mean(cvars_)),
        "var_lo":     float(vars_[0]),
        "var_hi":     float(vars_[-1]),
        "cvar_lo":    float(cvars_[0]),
        "cvar_hi":    float(cvars_[-1]),
    }


def calculate_component_es(
    weights: np.ndarray,
    returns_df,
    alpha: float = 0.05,
) -> dict:
    """
    Component Expected Shortfall — dekompozycja CVaR per aktywo.

    ES_component_i = w_i × E[r_i | r_portfolio <= VaR_α]
    Sum(ES_component_i) = ES_portfolio  (additive decomposition)

    Kluczowe zastosowania:
    - Identyfikacja aktywów dominujących w ryzyku ogonowym portfela
    - Risk budgeting oparty na CVaR (nie tylko wariancji)
    - Alokacja kapitału regulacyjnego

    Referencja: Tasche (2008) "Capital Allocation to Business Units and
    Sub-Portfolios: The Euler Principle". Pakiet risktools 2024.

    Parameters
    ----------
    weights     : array (n_assets,) — wagi portfela (suma = 1)
    returns_df  : DataFrame lub ndarray (T, n_assets) — dzienne zwroty
    alpha       : poziom ogona (0.05 = 95% CVaR)

    Returns
    -------
    dict z: component_es (array), pct_contribution (array), port_es (float),
            asset_names (list)
    """
    if hasattr(returns_df, "values"):
        R = returns_df.values
        names = list(returns_df.columns)
    else:
        R = np.asarray(returns_df, dtype=float)
        names = [f"A{i}" for i in range(R.shape[1])]

    w = np.asarray(weights, dtype=float)
    port_ret = R @ w
    var_level = np.percentile(port_ret, alpha * 100)
    tail_mask = port_ret <= var_level

    if tail_mask.sum() == 0:
        return {
            "component_es": np.zeros(len(w)),
            "pct_contribution": np.zeros(len(w)),
            "port_es": 0.0,
            "asset_names": names,
        }

    tail_returns = R[tail_mask]                    # (n_tail, n_assets)
    # Euler decomposition: ES_i = w_i * E[r_i | tail]
    mean_tail_per_asset = tail_returns.mean(axis=0)
    component_es = w * mean_tail_per_asset         # (n_assets,)
    port_es = float(np.mean(port_ret[tail_mask]))
    pct = component_es / (port_es + 1e-12)         # fraction of total ES

    return {
        "component_es":    component_es,
        "pct_contribution": pct,
        "port_es":         port_es,
        "asset_names":     names,
    }


def calculate_expectile_risk(
    returns,
    tau: float = 0.05,
) -> float:
    """
    Expectile Risk Measure — alternatywa dla CVaR.

    Expectile e_τ minimalizuje:
      E[|τ - 1(r < e)| · (r - e)²]

    Właściwości matematyczne (Jones 2023):
    - Elicitable: możliwa statystyczna weryfikacja prognoz (CVaR jej nie ma!)
    - Subadditive: spełnia aksjomaty miar ryzyka spójnego
    - Powiązanie z CVaR: ER(τ) ≈ CVaR(1-2τ) dla małych τ

    Referencja: Newey & Powell (1987), Jones (2023) "Revisiting Expectile Risk".

    Parameters
    ----------
    returns : array zwrotów
    tau     : poziom asymetrii (0.05 ≈ 95% quantile analog)

    Returns
    -------
    float — expectile value (ujemny = strata)
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    # Iterative Weighted Least Squares (IWLS)
    e = np.quantile(returns, tau)  # init with quantile
    for _ in range(200):
        below = returns < e
        weights = np.where(below, tau, 1.0 - tau)
        e_new = np.sum(weights * returns) / np.sum(weights)
        if abs(e_new - e) < 1e-10:
            break
        e = e_new
    return float(e)


def calculate_srr_score(wealth_paths: np.ndarray) -> float:
    """
    Sequence-of-Returns Risk (SRR) Score.

    Mierzy wrażliwość końcowego majątku na KOLEJNOŚĆ zwrotów.
    Wysoki SRR → portfel bardzo podatny na złą sekwencję (szczególnie groźne
    w fazie wypłat z emerytalnego).

    Metodologia:
    1. Oblicz CAGR per ścieżka
    2. Oblicz wariancję CAGR po podziale na decyle trajektorii
    3. SRR = std(CAGR_by_decile) / mean(CAGR_total)

    SRR < 0.1  → niskie ryzyko sekwencji
    SRR > 0.3  → wysokie ryzyko sekwencji (ostrzeżenie!)

    Referencja: Kitces & Pfau (2014), Blanchett (2024) "Sequence Risk Revisited".

    Parameters
    ----------
    wealth_paths : (n_sims, n_periods) — macierz ścieżek bogactwa z MC

    Returns
    -------
    float — SRR score (wyższy = gorszy)
    """
    if wealth_paths.ndim != 2 or wealth_paths.shape[1] < 2:
        return 0.0

    n_sims, n_periods = wealth_paths.shape
    # CAGR per simulation
    final_wealth = wealth_paths[:, -1]
    init_wealth = wealth_paths[:, 0]
    years = n_periods / 252.0 if n_periods > 252 else n_periods / 12.0
    # Avoid division by zero
    valid = (init_wealth > 0) & (final_wealth >= 0)
    if valid.sum() < 10:
        return 0.0

    cagr = (final_wealth[valid] / init_wealth[valid]) ** (1.0 / max(years, 1.0)) - 1.0

    # Divide simulations into deciles of early-period performance
    mid = n_periods // 4
    early_perf = wealth_paths[valid, mid] / init_wealth[valid]
    decile_labels = pd.qcut(early_perf, 10, labels=False, duplicates="drop")
    decile_cagr_means = pd.Series(cagr).groupby(decile_labels).mean()

    if len(decile_cagr_means) < 2:
        return 0.0

    srr = float(decile_cagr_means.std() / (abs(decile_cagr_means.mean()) + 1e-10))
    return srr
