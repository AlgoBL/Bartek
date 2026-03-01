"""
drawdown_recovery_analyzer.py — Analiza Czasu Odrobienia Strat

Implementuje:
1. Underwater Period Analysis — pełna mapa wszystkich drawdownów
2. Sequence-of-Returns Risk — wpływ kolejności złych lat
3. Recovery Probability — Monte Carlo P(odrobienie w N lat)
4. Time-to-Ruin — kiedy portfel się wyczerpie
5. Break-Even Return Calculator — ile trzeba zarobić po stracie X%

Referencje:
  - Magdon-Ismail & Atiya (2004) — Maximum Drawdown
  - Milevsky (2004) — Sequence of Returns Risk
  - Bengen (1994) — The 4% Rule (Safe Withdrawal Rate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. UNDERWATER PERIOD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def underwater_analysis(equity_curve) -> dict:
    """
    Pełna analiza wszystkich drawdown periods.

    Dla każdego drawdown: peak, trough, recovery, depth, duration, recovery_days.

    Returns
    -------
    dict z:
      drawdown_periods : pd.DataFrame — każdy drawdown
      summary          : dict — max, avg, longest
      current_underwater : bool
      current_dd_depth   : float
    """
    # ── Coerce to clean 1-D float Series ──────────────────────────────────────
    if isinstance(equity_curve, pd.DataFrame):
        # Take first column if DataFrame was passed
        equity_curve = equity_curve.iloc[:, 0]
    s = pd.Series(equity_curve).dropna()

    # Ensure DatetimeIndex with no duplicates
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    s = s[~s.index.duplicated(keep="last")]
    s = s.astype(float)

    if len(s) < 5:
        return {"error": "Za mało danych"}

    # ── Work in numpy for all comparisons (zero pandas boolean ambiguity) ──────
    vals = s.values
    idx  = s.index
    n    = len(vals)

    running_max = np.maximum.accumulate(vals)
    dd_arr = (vals - running_max) / np.where(running_max == 0, np.nan, running_max)
    dd_arr = np.nan_to_num(dd_arr, nan=0.0)

    periods = []
    in_dd = False
    dd_start_i = None
    trough_i = None

    for i in range(n):
        v = vals[i]
        rm = running_max[i]
        under = (v < rm * 0.999) and (rm > 0)

        if under and not in_dd:
            in_dd = True
            dd_start_i = i
            trough_i = i
        elif under and in_dd:
            if vals[i] < vals[trough_i]:
                trough_i = i
        elif not under and in_dd:
            # Recovery point
            trough_val = float(vals[trough_i])
            # Find actual peak (last ATH before dd_start_i)
            peak_i = int(np.argmax(vals[:dd_start_i + 1]))
            peak_val = float(vals[peak_i])
            depth = (trough_val - peak_val) / peak_val if peak_val != 0 else 0.0

            trough_date = idx[trough_i]
            peak_date   = idx[peak_i]
            recov_date  = idx[i]

            try:
                duration = int((trough_date - peak_date).total_seconds() / 86400)
                rec_days = int((recov_date - trough_date).total_seconds() / 86400)
            except Exception:
                duration = trough_i - peak_i
                rec_days = i - trough_i

            periods.append({
                "peak_date":      peak_date,
                "trough_date":    trough_date,
                "recovery_date":  recov_date,
                "depth":          depth,
                "duration_days":  max(0, duration),
                "recovery_days":  max(0, rec_days),
                "total_days":     max(0, duration + rec_days),
                "recovered":      True,
            })
            in_dd = False
            dd_start_i = None
            trough_i = None

    # Handle still-open drawdown at end
    if in_dd and trough_i is not None:
        trough_val = float(vals[trough_i])
        peak_i = int(np.argmax(vals[:dd_start_i + 1]))
        peak_val = float(vals[peak_i])
        depth = (trough_val - peak_val) / peak_val if peak_val != 0 else 0.0
        try:
            duration = int((idx[trough_i] - idx[peak_i]).total_seconds() / 86400)
        except Exception:
            duration = trough_i - peak_i
        periods.append({
            "peak_date":      idx[peak_i],
            "trough_date":    idx[trough_i],
            "recovery_date":  None,
            "depth":          depth,
            "duration_days":  max(0, duration),
            "recovery_days":  0,
            "total_days":     max(0, duration),
            "recovered":      False,
        })

    current_in_dd = bool(in_dd)
    current_depth = float(dd_arr[-1])

    # Rebuild dd as Series for callers
    dd_series = pd.Series(dd_arr, index=idx)

    if periods:
        df_periods = pd.DataFrame(periods)
        summary = {
            "n_drawdowns":   len(periods),
            "max_depth":     float(df_periods["depth"].min()),
            "avg_depth":     float(df_periods["depth"].mean()),
            "max_duration":  int(df_periods["duration_days"].max()),
            "avg_duration":  float(df_periods["duration_days"].mean()),
            "max_recovery":  int(df_periods["recovery_days"].max()),
            "avg_recovery":  float(df_periods["recovery_days"].mean()),
        }
    else:
        df_periods = pd.DataFrame()
        summary = {"n_drawdowns": 0}

    return {
        "drawdown_periods":  df_periods,
        "summary":           summary,
        "current_underwater": current_in_dd,
        "current_dd_depth":  current_depth,
        "dd_series":         dd_series,
    }



# ══════════════════════════════════════════════════════════════════════════════
# 2. SEQUENCE-OF-RETURNS RISK
# ══════════════════════════════════════════════════════════════════════════════

def sequence_of_returns_risk(
    annual_returns: list[float],
    initial_capital: float = 100_000.0,
    annual_withdrawal: float = 4000.0,
    n_permutations: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """
    Demonstracja ryzyka kolejności stóp zwrotu.

    Ten sam CAGR, różna kolejność → completely different terminal wealth
    gdy inwestor wypłaca środki (emerytura, odsetki).

    Parameters
    ----------
    annual_returns    : lista historycznych rocznych stóp zwrotu
    initial_capital   : kapitał startowy
    annual_withdrawal : roczna wypłata
    n_permutations    : ile losowych permutacji sprawdzić

    Returns
    -------
    dict z:
      original_final    : float — wynik z oryginalną kolejnością
      reversed_final    : float — wynik z odwróconą kolejnością
      best_final        : float — najlepsza permutacja
      worst_final       : float — najgorsza permutacja
      cagr              : float — CAGR wspólny dla wszystkich
      sequence_impact   : float — ratio (best / worst)
      percentile_5      : float
      percentile_95     : float
      paths             : list[list[float]] — sample 20 ścieżek
    """
    r = np.array(annual_returns, dtype=float)
    if len(r) < 2:
        return {"error": "Min 2 lata danych"}

    cagr = float((1 + r).prod() ** (1 / len(r)) - 1)

    def simulate_path(returns_seq):
        capital = initial_capital
        path = [capital]
        for ret in returns_seq:
            capital = capital * (1 + ret) - annual_withdrawal
            path.append(max(capital, 0))
            if capital <= 0:
                path.extend([0] * (len(returns_seq) - len(path) + 1))
                break
        return path

    original_path = simulate_path(r)
    reversed_path = simulate_path(r[::-1])

    rng = np.random.default_rng(rng_seed)
    finals = []
    sample_paths = []
    for i in range(n_permutations):
        perm = rng.permutation(r)
        path = simulate_path(perm)
        finals.append(path[-1])
        if i < 20:
            sample_paths.append(path)

    finals = np.array(finals)

    return {
        "original_final": original_path[-1],
        "reversed_final": reversed_path[-1],
        "best_final": float(finals.max()),
        "worst_final": float(finals.min()),
        "cagr": cagr,
        "sequence_impact": float(finals.max() / (finals.min() + 1e-10)),
        "percentile_5": float(np.percentile(finals, 5)),
        "percentile_95": float(np.percentile(finals, 95)),
        "median_final": float(np.median(finals)),
        "original_path": original_path,
        "reversed_path": reversed_path,
        "sample_paths": sample_paths,
        "annual_withdrawal": annual_withdrawal,
        "initial_capital": initial_capital,
        "n_years": len(r),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. BREAK-EVEN RETURN CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def breakeven_calculator(losses: list[float] = None) -> dict:
    """
    Ile % zysku potrzeba żeby odrobić daną stratę %.

    Strata X% → potrzeba zysk Z% = X / (1 - X)

    Returns
    -------
    dict z:
      table : pd.DataFrame — {loss_pct, required_gain_pct, effort_ratio}
    """
    if losses is None:
        losses = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]

    rows = []
    for loss in losses:
        required = loss / (1 - loss)
        years_10pct = np.log(1 / (1 - loss)) / np.log(1.10) if loss < 1 else np.inf
        years_7pct = np.log(1 / (1 - loss)) / np.log(1.07) if loss < 1 else np.inf
        rows.append({
            "Strata": f"-{loss:.0%}",
            "loss_pct": -loss,
            "Zwrot do BE": f"+{required:.1%}",
            "required_gain": required,
            "Lata (CAGR=10%)": round(years_10pct, 1) if years_10pct < 100 else "∞",
            "Lata (CAGR=7%)": round(years_7pct, 1) if years_7pct < 100 else "∞",
            "Effort Ratio": round(required / loss, 2) if loss > 0 else 1.0,
        })
    return {"table": pd.DataFrame(rows)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECOVERY PROBABILITY (Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════════

def recovery_probability_mc(
    current_drawdown: float,
    returns: pd.Series,
    horizon_years: int = 5,
    n_sims: int = 5000,
    rng_seed: int = 42,
) -> dict:
    """
    Monte Carlo: P(pełne odrobienie strat w ciągu horizon_years).

    Parametryzuje rozkład Student-t z historycznych zwrotów (grube ogony).

    Parameters
    ----------
    current_drawdown : float — bieżący drawdown (np. -0.20)
    returns          : pd.Series — historia dziennych zwrotów
    horizon_years    : int — ile lat na odrobienie
    n_sims           : int — liczba symulacji

    Returns
    -------
    dict z:
      prob_recovery_full : float — P(odrobienie 100% straty)
      prob_recovery_50   : float — P(odrobienie 50% straty)
      median_years       : float — mediana czasu do odrobienia
      paths_recovered_pct: float — % ścieżek z pełnym odrobem
    """
    r = returns.dropna()
    if len(r) < 30 or current_drawdown >= 0:
        return {"error": "Niepoprawne dane: potrzeba drawdown < 0 i min 30 obserwacji"}

    # Fit Student-t
    from scipy.stats import t as tdist
    params = tdist.fit(r)
    df_t, loc_t, scale_t = params

    n_days = horizon_years * 252
    rng = np.random.default_rng(rng_seed)

    # Starting value pozycja w drawdown
    start = 1.0 + current_drawdown  # np. 0.80 jeśli -20%
    target = 1.0  # odrobienie pełne

    recovered_full = 0
    recovered_half = 0
    recovery_times = []

    for _ in range(n_sims):
        sims = tdist.rvs(df_t, loc=loc_t, scale=scale_t, size=n_days, random_state=rng)
        path = np.cumprod(1 + sims) * start
        # Check if touched target
        half_target = 1.0 - abs(current_drawdown) / 2
        if path.max() >= target:
            recovered_full += 1
            idx = np.argmax(path >= target)
            recovery_times.append(idx / 252)
        if path.max() >= half_target:
            recovered_half += 1

    p_full = recovered_full / n_sims
    p_half = recovered_half / n_sims
    median_time = float(np.median(recovery_times)) if recovery_times else np.inf

    return {
        "current_drawdown": current_drawdown,
        "horizon_years": horizon_years,
        "prob_recovery_full": p_full,
        "prob_recovery_50pct": p_half,
        "median_years_to_recovery": median_time,
        "n_sims": n_sims,
        "pct_recovered": p_full,
    }
