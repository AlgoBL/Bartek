
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats.qmc import Sobol
from numba import jit
from modules.ai.observer import get_market_regimes
from modules.ai.architect import PortfolioArchitect
from modules.ai.optimizer import GeneticOptimizer
from modules.ai.trader import RLTrader
import streamlit as st
from config import TAX_BELKA, RISK_FREE_RATE_PL
from modules.logger import setup_logger

logger = setup_logger(__name__)

@jit(nopython=True, cache=True)
def _compute_garch_variance(var_series, eps, omega_g, alpha_g, beta_g, total_days):
    for day in range(1, total_days):
        for s in range(var_series.shape[0]):
            prev_eps = eps[s, day - 1]
            prev_var = var_series[s, day - 1]
            var_series[s, day] = omega_g + alpha_g * (prev_eps * np.sqrt(prev_var)) ** 2 + beta_g * prev_var
    return var_series

@jit(nopython=True, cache=True)
def _compute_wealth_paths(val_safe, val_risky, wealth_paths, daily_safe_rate, risky_returns, alloc_safe, threshold_percent, total_days, days_per_year, rebalance_strategy_id):
    n_sims = val_safe.shape[0]
    for day in range(1, total_days + 1):
        for s in range(n_sims):
            vs = val_safe[s, day-1] * (1.0 + daily_safe_rate)
            vr = val_risky[s, day-1] * (1.0 + risky_returns[s, day-1])
            cw = vs + vr
            
            should_rebalance = False
            if rebalance_strategy_id == 1 and day % days_per_year == 0:
                should_rebalance = True
            elif rebalance_strategy_id == 2 and day % 21 == 0:
                should_rebalance = True
            elif rebalance_strategy_id == 3:
                current_risky_weight = vr / cw
                target_weight = 1.0 - alloc_safe
                lower_bound = target_weight * (1.0 - threshold_percent)
                upper_bound = target_weight * (1.0 + threshold_percent)
                if current_risky_weight < lower_bound or current_risky_weight > upper_bound:
                    should_rebalance = True
                    
            if should_rebalance:
                vs = cw * alloc_safe
                vr = cw * (1.0 - alloc_safe)
                
            val_safe[s, day] = vs
            val_risky[s, day] = vr
            wealth_paths[s, day] = cw

    return wealth_paths

def generate_student_t_copula_shocks(n_sims, n_days, n_assets, df=4, correlation_matrix=None):
    """
    Generates multidimensional shocks using a Student-t Copula.
    This captures 'tail dependence' — assets crashing together during crises.
    """
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets)
        
    # Cholesky decomposition of the correlation matrix
    try:
        chol = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        chol = np.eye(n_assets)

    z = np.random.normal(0, 1, size=(n_sims, n_days, n_assets))
    correlated_z = np.einsum('ij,kMj->kMi', chol, z)
    chi2 = np.random.chisquare(df, size=(n_sims, n_days, 1))
    t_shocks = correlated_z * np.sqrt(df / chi2)
    std_t = np.sqrt(df / (df - 2)) if df > 2 else 1.0
    return t_shocks / std_t


# ══════════════════════════════════════════════════════════════════════════════
# ARCHIMEDEAN COPULAS — Referencja: Nelson (2006), Joe (1997)
# ══════════════════════════════════════════════════════════════════════════════

def _sample_clayton_copula(n: int, theta: float = 2.0) -> np.ndarray:
    """
    Clayton Copula — dolna zależność ogonów (lower tail dependence).
    Modeluje kryzysy gdzie wszystko spada razem.
    Zależność ogonowa: λ_L = 2^(-1/θ), λ_U = 0.
    θ → 0: niezależność; θ → ∞: komonotoniczność.
    """
    # Conditional inversion method (bivariate)
    u   = np.random.uniform(0, 1, n)
    p   = np.random.uniform(0, 1, n)
    # v = conditional quantile of Clayton
    exponent = -theta / (1 + theta)
    v = u * ((p ** exponent) - 1 + u ** (-theta)) ** (-1 / theta)
    v = np.clip(v, 1e-8, 1 - 1e-8)
    return np.column_stack([u, v])


def _sample_gumbel_copula(n: int, theta: float = 2.0) -> np.ndarray:
    """
    Gumbel Copula — górna zależność ogonów (upper tail dependence).
    Modeluje boom gdzie wszystko rośnie razem.
    Zależność ogonowa: λ_U = 2 - 2^(1/θ), λ_L = 0.
    θ = 1: niezależność; θ → ∞: komonotoniczność.
    """
    # Stable distribution simulation (Gumbel Fréchet method)
    from scipy.stats import levy_stable
    alpha_s = 1.0 / theta
    # Simulate from Positive Stable distribution
    phi = np.pi * (np.random.uniform(0, 1, n) - 0.5)
    e   = np.random.exponential(1, n)
    W   = (np.sin(alpha_s * phi) / np.cos(phi)) ** (1 / alpha_s) \
          * (np.cos(phi - alpha_s * phi) / e) ** ((1 - alpha_s) / alpha_s)
    # Transform uniform marginals through Gumbel generator inverse
    e1 = np.random.exponential(1, n)
    e2 = np.random.exponential(1, n)
    u  = np.exp(-(e1 / W) ** (1 / theta))
    v  = np.exp(-(e2 / W) ** (1 / theta))
    u  = np.clip(u, 1e-8, 1 - 1e-8)
    v  = np.clip(v, 1e-8, 1 - 1e-8)
    return np.column_stack([u, v])


def _sample_frank_copula(n: int, theta: float = 3.0) -> np.ndarray:
    """
    Frank Copula — symetryczna zależność ogonów.
    Brak ekstremalnej zależności ogonów: λ_L = λ_U = 0.
    Modeluje liniową zależność bez efektu kryzysowego.
    """
    u = np.random.uniform(0, 1, n)
    p = np.random.uniform(0, 1, n)
    if abs(theta) < 1e-6:
        return np.column_stack([u, p])  # independence
    # Conditional inverse
    num = -np.log(1 - (1 - np.exp(-theta)) / (np.exp(-theta * p) * (1 / u - 1) + 1))
    v   = np.clip(num / theta, 1e-8, 1 - 1e-8)
    return np.column_stack([u, v])


def generate_archimedean_copula_shocks(
    n_sims: int, n_days: int,
    family: str = "clayton",
    theta: float = 2.0,
    df_marginal: float = 4.0,
) -> np.ndarray:
    """
    Generuje wstrząsy z Kopuły Archimedejskiej (bivariate → rozszerzenie przez PIT).

    Parameters
    ----------
    n_sims      : liczba symulacji
    n_days      : liczba dni
    family      : 'clayton' | 'gumbel' | 'frank'
    theta       : parametr kopuły (siła zależności)
    df_marginal : stopnie swobody dla marginalnego rozkładu t

    Returns
    -------
    (n_sims, n_days) array of standardized shocks
    """
    total = n_sims * n_days
    if family == "gumbel":
        uv = _sample_gumbel_copula(total, theta)
    elif family == "frank":
        uv = _sample_frank_copula(total, theta)
    else:  # clayton (default)
        uv = _sample_clayton_copula(total, theta)

    from scipy.stats import t as t_dist
    std_t = np.sqrt(df_marginal / (df_marginal - 2)) if df_marginal > 2 else 1.0
    # PIT: uniform → Student-t marginals
    u_clipped = np.clip(uv[:, 0], 1e-6, 1 - 1e-6)
    shocks = t_dist.ppf(u_clipped, df=df_marginal) / std_t
    return shocks.reshape(n_sims, n_days)


# ══════════════════════════════════════════════════════════════════════════════
# ROUGH VOLATILITY — Rough Bergomi (Gatheral et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_rough_bergomi_vol(
    n_sims: int,
    n_days: int,
    H: float = 0.1,
    xi0: float = 0.04,     # initial variance level (e.g. vol²=20% → xi0=0.04)
    eta: float = 1.9,      # vol-of-vol
    rho: float = -0.70,    # spot-vol correlation (leverage effect)
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rough Bergomi Stochastic Volatility (Gatheral, Jaisson & Rosenbaum 2018).

    Volatility obeys fractional Brownian motion with Hurst H ≈ 0.1 (rough):
      V(t) = xi(t) * exp(eta * W^H(t) - 0.5*eta²*t^(2H))
    gdzie W^H jest fBM z wykładnikiem H.

    Kluczowe różnice od GARCH:
    - Hurst H ≈ 0.1 → bardzo "szorstka" ścieżka zmienności
    - Power-law decay covariancji: Cov(V_s, V_t) ~ |t-s|^(2H)
    - Brak wykładniczego zanikania jak w GARCH
    - Lepsza dopasowanie do implied vol surface (Gatheral 2018)

    Returns
    -------
    vol_paths  : (n_sims, n_days) instantaneous volatiliy paths
    spot_shocks: (n_sims, n_days) correlated spot shocks (ρ z vol)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / 252.0
    times = np.arange(1, n_days + 1) * dt  # (n_days,)

    # ── Approximate fBM via Cholesky of covariance matrix ─────────────────
    # Cov(W^H_s, W^H_t) = 0.5*(s^{2H} + t^{2H} - |t-s|^{2H})
    # For large n_days this is expensive — cap at 252 then tile
    cov_len = min(n_days, 252)
    cov_T   = times[:cov_len]
    Cov = np.zeros((cov_len, cov_len))
    for i in range(cov_len):
        for j in range(i, cov_len):
            s, t = cov_T[i], cov_T[j]
            c = 0.5 * (s**(2*H) + t**(2*H) - abs(t - s)**(2*H))
            Cov[i, j] = Cov[j, i] = c
    # Regularise
    Cov += np.eye(cov_len) * 1e-8
    try:
        L = np.linalg.cholesky(Cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(Cov)))

    vol_paths   = np.zeros((n_sims, n_days))
    spot_shocks = np.zeros((n_sims, n_days))

    for sim in range(n_sims):
        # Simulate fBM path (tile if n_days > 252)
        fbm_full = np.zeros(n_days)
        ptr = 0
        while ptr < n_days:
            z_indep = np.random.normal(0, 1, cov_len)
            fbm_seg = L @ z_indep
            take    = min(cov_len, n_days - ptr)
            fbm_full[ptr: ptr + take] = fbm_seg[:take]
            ptr += take

        # Rough Bergomi variance process
        var_t = xi0 * np.exp(
            eta * fbm_full
            - 0.5 * eta**2 * times[:n_days]**(2*H)
        )
        vol_t = np.sqrt(np.maximum(var_t, 1e-8))
        vol_paths[sim] = vol_t

        # Correlated spot shocks: w_spot = ρ*w_vol + sqrt(1-ρ²)*w_indep
        z_vol   = fbm_full / (np.std(fbm_full) + 1e-10)  # normalised
        z_indep = np.random.normal(0, 1, n_days)
        spot_shocks[sim] = rho * z_vol + np.sqrt(max(1 - rho**2, 0)) * z_indep

    return vol_paths, spot_shocks

def simulate_barbell_strategy(
    n_years=10, 
    n_simulations=1000, 
    initial_captial=10000,
    safe_rate=RISK_FREE_RATE_PL,
    risky_mean=0.08, 
    risky_vol=0.20, 
    risky_kurtosis=6.0,
    alloc_safe=0.85,
    rebalance_strategy="None",
    threshold_percent=0.20,
    use_qmc=False,
    use_garch=False,
    use_jump_diffusion=True,
    jump_lambda=1.5,
    jump_mean=-0.03,
    jump_std=0.04,
    # ==== NOWE v3.0 ====
    copula_family: str = "student_t",   # 'student_t' | 'clayton' | 'gumbel' | 'frank'
    copula_theta: float = 2.0,          # Archimedean copula strength
    use_rough_vol: bool = False,        # Rough Bergomi (Gatheral 2018)
    rough_hurst: float = 0.1,          # Hurst exponent H≈0.1 for rough
    rough_eta: float = 1.9,            # vol-of-vol
    rough_rho: float = -0.70,          # spot-vol correlation
    custom_scenarios: list = None,     # [{"year": 5, "drop_pct": 0.40}]
):
    """
    Monte Carlo Simulation z:
    - GARCH(1,1) lub Rough Bergomi zmiennością
    - Kopuła t-Studenta (default) lub Archimedejska (Clayton/Gumbel/Frank)
    - Merton Jump-Diffusion
    - Belka Tax (19%) na zyski
    """
    days_per_year = 252
    total_days = n_years * days_per_year
    dt = 1/days_per_year
    
    df = max(2.1, risky_kurtosis)
    
    # ─── Random Shocks: Copula selection ────────────────────────────────────
    if use_rough_vol:
        # Rough Bergomi: zwraca vol_paths i spot_shocks z korelacją spot-vol
        vol_paths, spot_shocks_2d = simulate_rough_bergomi_vol(
            n_sims=n_simulations, n_days=total_days,
            H=rough_hurst, xi0=risky_vol**2,
            eta=rough_eta, rho=rough_rho, seed=None,
        )
        standardized_shocks = spot_shocks_2d
        rough_daily_vols    = vol_paths
    elif copula_family in ("clayton", "gumbel", "frank"):
        standardized_shocks = generate_archimedean_copula_shocks(
            n_sims=n_simulations, n_days=total_days,
            family=copula_family, theta=copula_theta,
            df_marginal=max(2.1, risky_kurtosis),
        )
        rough_daily_vols = None
    elif use_qmc:
        df = max(2.1, risky_kurtosis)
        try:
            if total_days <= 21201:
                sampler = Sobol(d=total_days, scramble=True)
                n_sobol = int(2 ** np.ceil(np.log2(n_simulations)))
                uniform_samples = sampler.random(n_sobol)[:n_simulations]
                from scipy.stats import t as t_dist
                std_t = np.sqrt(df / (df - 2))
                standardized_shocks = t_dist.ppf(np.clip(uniform_samples, 1e-8, 1 - 1e-8), df=df) / std_t
            else:
                raise ValueError("Too many days for Sobol")
        except Exception as e:
            logger.warning(f"Błąd generacji Sobola, fallback do t.rvs: {e}")
            random_shocks = t.rvs(df, size=(n_simulations, total_days))
            std_t = np.sqrt(df / (df - 2))
            standardized_shocks = random_shocks / std_t
        rough_daily_vols = None
    else:
        df = max(2.1, risky_kurtosis)
        random_shocks = t.rvs(df, size=(n_simulations, total_days))
        std_t = np.sqrt(df / (df - 2))
        standardized_shocks = random_shocks / std_t
        rough_daily_vols = None

    # ─── Merton Jump-Diffusion ──────────────────────────────────────────────
    jump_returns = np.zeros((n_simulations, total_days))
    if use_jump_diffusion:
        prob_jump = jump_lambda * dt
        jump_occurrences = np.random.binomial(1, prob_jump, size=(n_simulations, total_days))
        jump_sizes = np.random.normal(jump_mean, jump_std, size=(n_simulations, total_days))
        jump_returns = jump_occurrences * jump_sizes


    # ─── GARCH(1,1) / Rough Bergomi Volatility ──────────────────────────────
    if use_rough_vol and rough_daily_vols is not None:
        # Rough Bergomi: spot_shocks already correlated with vol
        jump_var_annual = jump_lambda * (jump_mean**2 + jump_std**2) if use_jump_diffusion else 0
        jump_mean_annual = jump_lambda * (np.exp(jump_mean + 0.5*jump_std**2) - 1) if use_jump_diffusion else 0
        diff_mean = risky_mean - jump_mean_annual
        daily_mean = diff_mean * dt
        risky_returns = np.exp(daily_mean + rough_daily_vols * standardized_shocks) - 1 + jump_returns
    elif use_garch:
        alpha_g = 0.10
        beta_g  = 0.85
        # Compensation for jump variance to keep overall volatility target
        jump_var_annual = jump_lambda * (jump_mean**2 + jump_std**2) if use_jump_diffusion else 0
        target_diff_var = max(0.0001, (risky_vol ** 2) - jump_var_annual)
        
        omega_g = target_diff_var * (1 - alpha_g - beta_g) / days_per_year
        
        var_series = np.zeros((n_simulations, total_days))
        var_series[:, 0] = target_diff_var / days_per_year
        
        eps = standardized_shocks.copy()
        var_series = _compute_garch_variance(var_series, eps, omega_g, alpha_g, beta_g, total_days)
        
        daily_vols = np.sqrt(np.maximum(var_series, 1e-10))
        
        # Compensate mean for jumps
        jump_mean_annual = jump_lambda * (np.exp(jump_mean + 0.5*jump_std**2) - 1) if use_jump_diffusion else 0
        diff_mean = risky_mean - jump_mean_annual - 0.5 * target_diff_var
        daily_mean = diff_mean * dt
        
        risky_returns = np.exp(daily_mean + daily_vols * standardized_shocks) - 1 + jump_returns
    else:
        jump_var_annual = jump_lambda * (jump_mean**2 + jump_std**2) if use_jump_diffusion else 0
        target_diff_var = max(0.0001, (risky_vol ** 2) - jump_var_annual)
        
        jump_mean_annual = jump_lambda * (np.exp(jump_mean + 0.5*jump_std**2) - 1) if use_jump_diffusion else 0
        diff_mean = risky_mean - jump_mean_annual - 0.5 * target_diff_var
        
        daily_mean = diff_mean * dt
        daily_vol = np.sqrt(target_diff_var / days_per_year)
        
        risky_returns = np.exp(daily_mean + daily_vol * standardized_shocks) - 1 + jump_returns
    
    # Apply Custom Scenarios (Manual Crash Injection)
    if custom_scenarios:
        for scenario in custom_scenarios:
            yr = scenario.get("year", 1)
            drop = scenario.get("drop_pct", 0.0)
            day_idx = min(int(yr * days_per_year), total_days - 1)
            # Inject a direct negative return on that specific day across all simulations
            risky_returns[:, day_idx] -= drop

    # Apply 19% Belka Tax on risky gains (conservative daily approximation)
    risky_returns = np.where(risky_returns > 0, risky_returns * 0.81, risky_returns)
    
    # Apply 19% Belka Tax to safe rate interest
    daily_safe_rate = (1 + (safe_rate * 0.81))**(1/days_per_year) - 1
    
    wealth_paths = np.zeros((n_simulations, total_days + 1))
    wealth_paths[:, 0] = initial_captial
    
    val_safe = np.zeros((n_simulations, total_days + 1))
    val_risky = np.zeros((n_simulations, total_days + 1))
    
    val_safe[:, 0] = initial_captial * alloc_safe
    val_risky[:, 0] = initial_captial * (1 - alloc_safe)
    
    rebalance_map = {"None": 0, "Yearly": 1, "Monthly": 2, "Threshold": 3}
    reb_id = rebalance_map.get(rebalance_strategy, 0)
    
    wealth_paths = _compute_wealth_paths(
        val_safe, val_risky, wealth_paths, daily_safe_rate, risky_returns, 
        alloc_safe, threshold_percent, total_days, days_per_year, reb_id
    )

    return wealth_paths


from modules.metrics import calculate_sharpe, calculate_sortino, calculate_calmar, calculate_max_drawdown

def calculate_metrics(wealth_paths, years):
    """Calculates metrics for Monte Carlo paths (2D array)"""
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    returns = np.diff(wealth_paths, axis=1) / wealth_paths[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1)
    
    rf = 0.04 * 0.81 # Tax-adjusted risk free proxy
    excess_ret = cagr - rf
    sharpe = np.divide(excess_ret, volatility, out=np.zeros_like(excess_ret), where=volatility!=0)
    calmar = np.divide(cagr, np.abs(max_drawdowns), out=np.zeros_like(cagr), where=max_drawdowns!=0)
    
    metrics = {
        "mean_final_wealth": np.mean(final_wealth),
        "median_final_wealth": np.median(final_wealth),
        "std_final_wealth": np.std(final_wealth),
        "mean_cagr": np.mean(cagr),
        "median_cagr": np.median(cagr),
        "prob_loss": np.mean(final_wealth < initial_wealth[0]),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "worst_case_drawdown": np.min(max_drawdowns),
        
        "median_sharpe": np.median(sharpe),
        "median_sortino": 0.0,
        "median_calmar": np.median(calmar),
        "median_volatility": np.median(volatility),
        "var_95": np.percentile(final_wealth, 5),
        "cvar_95": final_wealth[final_wealth <= np.percentile(final_wealth, 5)].mean()
    }
    return metrics

def calculate_individual_metrics(wealth_paths, years):
    """
    Calculates metrics for EACH simulation path separately for 3D Scatter plots.
    """
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    n_sims, n_days = wealth_paths.shape
    
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1)
    
    returns = np.diff(wealth_paths, axis=1) / wealth_paths[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    
    rf = 0.04 * 0.81
    excess_returns = cagr - rf
    sharpe = np.divide(excess_returns, volatility, out=np.zeros_like(excess_returns), where=volatility!=0)
    calmar = np.divide(cagr, np.abs(max_drawdowns), out=np.zeros_like(cagr), where=max_drawdowns!=0)
    
    return pd.DataFrame({
        "FinalWealth": final_wealth,
        "CAGR": cagr,
        "MaxDrawdown": max_drawdowns,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Calmar": calmar
    })

from modules.metrics import calculate_sharpe, calculate_sortino, calculate_calmar, calculate_max_drawdown
from modules.risk_manager import RiskManager

def run_ai_backtest(
    safe_data, 
    risky_data, 
    initial_capital=100000,
    safe_type="Ticker",
    safe_fixed_rate=RISK_FREE_RATE_PL,
    allocation_mode="AI Dynamic", 
    alloc_safe_fixed=0.85,
    kelly_params=None,
    rebalance_strategy="Monthly",
    threshold_percent=0.20,
    progress_callback=None,
    risky_weights_dict=None,
    # ==== NOWE v3.0 (Risk & Costs) ====
    transaction_costs: dict = None,
    risk_params: dict = None,
):
    """
    Backtest z uwzględnieniem kosztów transakcyjnych i zarządzania ryzykiem (Stop-loss, Kelly).
    """
    risk_m = RiskManager(transaction_costs)
    stop_loss = risk_params.get("stop_loss", 0.0) if risk_params else 0.0
    trailing_stop = risk_params.get("trailing_stop", 0.0) if risk_params else 0.0
    vol_target = risk_params.get("vol_target", 0.0) if risk_params else 0.0

    if risky_data.empty:
        st.error("Risky data is empty.")
        return pd.DataFrame(), [], []
    risky_tickers = risky_data.columns.tolist()
    
    if safe_type == "Fixed":
        dates = risky_data.index
        # Apply 19% Belka Tax to safe fixed rate
        daily_rate = (1 + (safe_fixed_rate * 0.81))**(1/252) - 1
        safe_prices = [100.0]
        for _ in range(len(dates)-1):
            safe_prices.append(safe_prices[-1] * (1 + daily_rate))
            
        safe_data = pd.DataFrame({"FIXED_SAFE": safe_prices}, index=dates)
        safe_tickers = ["FIXED_SAFE"]
    else:
        if safe_data.empty:
             st.error("Safe data is empty.")
             return pd.DataFrame(), [], []
        safe_tickers = safe_data.columns.tolist()

    combined_data = pd.concat([safe_data, risky_data], axis=1).dropna()
    dates = combined_data.index
    
    portfolio_value = [initial_capital]
    weights_history = []
    regime_history = []
    
    architect = PortfolioArchitect()
    trader = RLTrader()
    
    risky_returns_df = combined_data[risky_tickers].pct_change().fillna(0)
    risky_proxy_returns = risky_returns_df.mean(axis=1)
    
    regimes, observer_model = get_market_regimes(risky_proxy_returns, progress_callback)
    
    if observer_model:
        high_vol_state = observer_model.high_vol_state
        normalized_regimes = np.zeros_like(regimes)
        normalized_regimes[regimes == high_vol_state] = 1
        regimes = normalized_regimes
    else:
        regimes = np.zeros(len(dates))
    
    current_holdings = {t: 0.0 for t in combined_data.columns}
    entry_prices = {t: 0.0 for t in combined_data.columns}
    max_prices = {t: 0.0 for t in combined_data.columns}
    is_stopped = {t: False for t in combined_data.columns}
    cash = initial_capital
    last_rebalance_idx = 0
    total_steps = len(combined_data)
    
    for i in range(total_steps):
        if i % 10 == 0 and progress_callback:
            pct = i / total_steps
            progress_callback(pct, f"Symulacja: {pct:.1%} ({i}/{total_steps})")
            
        date = dates[i]
        prices = combined_data.iloc[i]
        
        regime_desc = "Unknown"
        if allocation_mode == "AI Dynamic":
            regime = regimes[i]
            regime_desc = "High Volatility (Risk-Off)" if regime == 1 else "Low Volatility (Risk-On)"
        
        regime_history.append(regime_desc)
        
        should_rebalance = False
        if i == 0:
            should_rebalance = True
        elif allocation_mode == "AI Dynamic":
             if i % 21 == 0:
                 should_rebalance = True
        else:
            if rebalance_strategy == "Yearly":
                if (i - last_rebalance_idx) >= 252:
                    should_rebalance = True
            elif rebalance_strategy == "Monthly":
                if (i - last_rebalance_idx) >= 21:
                    should_rebalance = True
            elif rebalance_strategy == "Threshold (Shannon's Demon)":
                if val_now > 0:
                   current_risky_val = sum(current_holdings[t] * prices[t] for t in risky_tickers)
                   current_risky_w = current_risky_val / val_now
                   target_r = 1.0 - alloc_safe_fixed if allocation_mode == "Manual Fixed" else 0.5
                   if abs(current_risky_w - target_r) / target_r > threshold_percent:
                       should_rebalance = True
            
            # --- Stop Loss / Trailing Stop Check ---
            for t in combined_data.columns:
                if current_holdings[t] > 0 and not is_stopped[t]:
                    max_prices[t] = max(max_prices[t], prices[t])
                    if risk_m.check_stops(entry_prices[t], prices[t], max_prices[t], stop_loss, trailing_stop):
                        # Exit position immediately
                        cash += current_holdings[t] * prices[t] * (1 - risk_m.costs.get("bid_ask", 0.0002))
                        # Deduct trading cost
                        asset_cls = "crypto" if "BTC" in t or "ETH" in t else "etf"
                        cash -= risk_m.calculate_transaction_cost(asset_cls, current_holdings[t]*prices[t], False)
                        current_holdings[t] = 0
                        is_stopped[t] = True
        
        if should_rebalance:
            prev_portfolio_val = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash if i > 0 else initial_capital
            
            # 1. Strategic Split
            target_safe_pct = 0.5; target_risky_pct = 0.5
            if allocation_mode == "AI Dynamic":
                if "High Volatility" in regime_desc:
                    target_safe_pct = 0.80; target_risky_pct = 0.20
                else:
                    target_safe_pct = 0.20; target_risky_pct = 0.80
                
                vol_window = risky_proxy_returns.iloc[max(0, i-30):i].std() * np.sqrt(252)
                kelly_mult = trader.get_kelly_adjustment(vol_window, regime_desc)
                target_risky_pct *= kelly_mult
                target_safe_pct = 1.0 - target_risky_pct
            elif allocation_mode == "Manual Fixed":
                target_safe_pct = alloc_safe_fixed
                target_risky_pct = 1.0 - target_safe_pct
            elif allocation_mode == "Rolling Kelly":
                win_len = kelly_params.get('window', 252) if kelly_params else 252
                lookback_ret = risky_proxy_returns.iloc[max(0, i-win_len):i]
                if len(lookback_ret) > 60:
                    mu = lookback_ret.mean() * 252; sigma = lookback_ret.std() * np.sqrt(252)
                    r_safe = (safe_fixed_rate * 0.81) if safe_type == "Fixed" else 0.04 * 0.81
                    if sigma > 0:
                        kelly_full = (mu - r_safe) / (sigma**2)
                        k_opt = kelly_full * kelly_params.get('fraction', 1.0) * (1 - kelly_params.get('shrinkage', 0.0))
                        target_risky_pct = max(0.0, min(1.5, k_opt))
                        target_safe_pct = 1.0 - target_risky_pct
            
            # 2. Rebalance logic with tax on gains considered automatically by reduced future growth
            # (In reality, rebalance triggers tax on sold profitable assets. 
            #  Here we model it as tax on positive daily returns for simplicity)
            portfolio_val_now = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash if i > 0 else initial_capital
            
            # Clear holdings
            current_holdings = {t: 0.0 for t in combined_data.columns}
            cash = 0
            
            window_start = max(0, i-60)
            window_data = combined_data.iloc[window_start:i+1]
            
            if safe_type == "Fixed":
                safe_weights_internal = {"FIXED_SAFE": 1.0}
            else:
                safe_weights_internal = architect.allocate_hrp(window_data[safe_tickers]) if len(window_data) > 10 else {t: 1.0/len(safe_tickers) for t in safe_tickers}

            if risky_weights_dict:
                risky_weights_internal = {t: risky_weights_dict.get(t, 0.0) for t in risky_tickers}
            else:
                risky_weights_internal = architect.allocate_hrp(window_data[risky_tickers]) if len(window_data) > 10 else {t: 1.0/len(risky_tickers) for t in risky_tickers}
            
            global_weights = {}
            for t in safe_tickers: global_weights[t] = safe_weights_internal.get(t, 0) * target_safe_pct
            for t in risky_tickers: global_weights[t] = risky_weights_internal.get(t, 0) * target_risky_pct
            for t in combined_data.columns:
                is_stopped[t] = False # Reset stops on rebalance
                entry_prices[t] = prices[t]
                max_prices[t] = prices[t]
                
                asset_cls = "crypto" if "BTC" in t or "ETH" in t else "etf"
                if "FIXED_SAFE" in t: asset_cls = "bonds"
                
                target_val = portfolio_val_now * global_weights.get(t, 0)
                # Deduct transaction cost on entry/rebalance
                cost = risk_m.calculate_transaction_cost(asset_cls, target_val, True)
                portfolio_val_now -= cost
                
                current_holdings[t] = (portfolio_val_now * global_weights.get(t, 0)) / prices[t]
            
            weights_history.append(global_weights)
            last_rebalance_idx = i
        else:
            weights_history.append(weights_history[-1] if weights_history else {})
        
        # Daily return logic for tax in backtest
        if i > 0:
            prev_val = portfolio_value[-1]
            # Since backtest uses raw prices, we must manually subtract the tax from the daily gain
            # This is hard to do without mutating prices. 
            # Solution: we adjust the portfolio value calculation to reflect tax drag on gains.
            raw_val = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
            daily_gain = raw_val - prev_val
            if daily_gain > 0:
                # Deduct 19% from gain
                net_val = prev_val + (daily_gain * 0.81)
                # To keep math consistent, we adjust 'cash' to absorb the tax loss
                cash -= (daily_gain * TAX_BELKA)
                portfolio_value.append(net_val)
            else:
                portfolio_value.append(raw_val)
        else:
            portfolio_value.append(initial_capital)
        
    results = pd.DataFrame({
        "Date": dates,
        "PortfolioValue": portfolio_value[1:],
        "Regime": regime_history
    }).set_index("Date")
    
    return results, weights_history, regimes
