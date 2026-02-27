"""
rl_environment.py — Gymnasium environment dla portfolio allocation (PPO/SAC).

Implementuje custom Gymnasium Env symulujący decyzje alokacji portfela
w środowisku multi-asset z obserwacją makroekonomiczną.

Nagroda: Differential Sharpe Ratio (Moody & Saffell 1999)
  R_t = (Δμ_t - 0.5 * Δσ²_t / σ_{t-1}) / σ_{t-1}

Zoptymalizowane pod Stable-Baselines3 (PPO, SAC, TD3).

Referencje:
  Moody & Saffell (1999) — "Learning to Trade via Direct Reinforcement"
  Jiang, Xu & Liang (2017) — "A Deep Reinforcement Learning Framework for
                              the Financial Portfolio Management Problem"
  Mnih et al. (2015) — "Human-level control through deep RL" (DQN)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# Soft dep: gymnasium (może być niedostępne)
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class PortfolioEnv:
    """
    Custom Portfolio Allocation Environment.

    Jeśli gymnasium jest dostępne, dziedziczy po gym.Env i jest
    kompatybilny ze Stable-Baselines3. W przeciwnym razie działa
    jako standalone symulator (do backtestingu reguł RL).

    Przestrzeń obserwacji (state):
      [current_weights (n),
       returns_window_60d (n × 60),
       macro_features (8)]
      → łącznie: n + n*60 + 8

    Przestrzeń akcji:
      Continuous ∈ [0, 1]^n — wagi portfela (normalizowane do sumy 1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        returns_df: pd.DataFrame,
        macro_features_df: pd.DataFrame | None = None,
        initial_capital: float = 100_000.0,
        lookback: int = 60,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.04,
    ):
        """
        Parameters
        ----------
        returns_df        : (T, n_assets) dzienna macierz zwrotów
        macro_features_df : (T, 8) cechy makroekonomiczne (opcjonalne)
        initial_capital   : kapitał startowy
        lookback          : okno historyczne w obserwacji
        transaction_cost  : koszt transakcyjny (jednostronny)
        risk_free_rate    : stopa wolna od ryzyka (roczna)
        """
        self.returns_df     = returns_df.reset_index(drop=True)
        self.n_assets       = returns_df.shape[1]
        self.asset_names    = list(returns_df.columns)
        self.T              = len(returns_df)
        self.lookback       = lookback
        self.initial_capital = initial_capital
        self.tc             = transaction_cost
        self.rf_daily       = (1 + risk_free_rate) ** (1 / 252) - 1

        # Makro cechy
        if macro_features_df is not None:
            self.macro_df = macro_features_df.reset_index(drop=True)
            self.n_macro  = macro_features_df.shape[1]
        else:
            self.macro_df = None
            self.n_macro  = 0

        # Wymiary obserwacji
        self.obs_dim = self.n_assets + self.n_assets * lookback + self.n_macro

        # Gymnasium spaces (jeśli dostępne)
        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_assets,), dtype=np.float32,
            )

        # Stan wewnętrzny
        self._reset_state()

    def _reset_state(self) -> None:
        self.t           = self.lookback
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self._portfolio_history = [self.initial_capital]
        self._return_history    = []
        # Dla Differential Sharpe Ratio
        self._mu_t  = 0.0
        self._var_t = 1e-6

    def _get_obs(self) -> np.ndarray:
        """Buduje wektor obserwacji."""
        # Okno zwrotów
        window = self.returns_df.iloc[self.t - self.lookback:self.t].values  # (60, n)
        window_flat = window.T.flatten().astype(np.float32)                   # (n*60,)
        # Bieżące wagi
        weights = self.current_weights.astype(np.float32)
        # Cechy makro
        if self.macro_df is not None:
            macro = self.macro_df.iloc[self.t].values.astype(np.float32)
            macro = np.nan_to_num(macro, nan=0.0)
        else:
            macro = np.array([], dtype=np.float32)

        obs = np.concatenate([weights, window_flat, macro])
        return np.clip(obs, -10.0, 10.0)

    def _differential_sharpe_reward(self, r_t: float) -> float:
        """
        Moody & Saffell (1999) — Differential Sharpe Ratio.
        Nagradza faktyczną poprawę stosunku zysk/ryzyko vs oczekiwany.

        Δt = r_t - μ_t
        R_t = (μ_{t-1}*Δt - 0.5 * σ²_{t-1} * Δt²) / (σ²_{t-1})^{3/2} * η
        """
        eta = 0.01  # smoothing factor
        delta = r_t - self._mu_t
        reward = (self._var_t * delta - 0.5 * delta**2) / max(self._var_t**1.5, 1e-8)
        reward *= eta
        # Update running moments
        self._mu_t  = (1 - eta) * self._mu_t  + eta * r_t
        self._var_t = (1 - eta) * self._var_t + eta * delta**2
        return float(np.clip(reward, -10.0, 10.0))

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Gymnasium-compatible reset."""
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Wykonuje krok symulacji.

        action : (n_assets,) — target weights (surowe, przed normalizacją)
        """
        # Normalizacja wag do sumy 1
        action = np.maximum(action, 0.0)
        action_sum = action.sum()
        if action_sum < 1e-8:
            weights_new = np.ones(self.n_assets) / self.n_assets
        else:
            weights_new = action / action_sum

        # Koszty transakcyjne (proporcjonalne do obrotu)
        turnover = np.sum(np.abs(weights_new - self.current_weights))
        tc_cost  = self.tc * turnover

        # Zwrot portfela
        daily_returns = self.returns_df.iloc[self.t].values
        port_return   = float(weights_new @ daily_returns) - tc_cost - self.rf_daily

        # Nagroda
        reward = self._differential_sharpe_reward(port_return)

        # Update portfolio value
        self.portfolio_value *= (1.0 + port_return + self.rf_daily)
        self._portfolio_history.append(self.portfolio_value)
        self._return_history.append(port_return)
        self.current_weights = weights_new
        self.t += 1

        terminated = self.t >= self.T - 1
        truncated  = self.portfolio_value < self.initial_capital * 0.01  # bankructwo
        obs        = self._get_obs() if not terminated and not truncated else self._get_obs()

        info = {
            "portfolio_value":  self.portfolio_value,
            "portfolio_return":  port_return,
            "weights":          weights_new.tolist(),
            "step":             self.t,
        }
        return obs, reward, terminated, truncated, info

    def get_episode_summary(self) -> dict:
        """Zwraca podsumowanie epizodu (do callbacków treningowych)."""
        pv = np.array(self._portfolio_history)
        rets = np.array(self._return_history)
        peaks = np.maximum.accumulate(pv)
        max_dd = float(np.min((pv - peaks) / np.where(peaks > 0, peaks, 1)))
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 0 else 0.0
        cagr = float((pv[-1] / pv[0]) ** (252 / max(len(rets), 1)) - 1) if len(pv) > 1 else 0.0
        return {
            "final_value": float(pv[-1]),
            "cagr":        cagr,
            "max_drawdown": max_dd,
            "volatility":   vol,
            "sharpe":       cagr / max(vol, 1e-6),
        }


# Rejestruj env w Gymnasium (jeśli dostępne)
if _GYM_AVAILABLE:
    try:
        gym.register(
            id="PortfolioAllocation-v1",
            entry_point="modules.ai.rl_environment:PortfolioEnv",
        )
    except Exception:
        pass  # Już zarejestrowane lub błąd — ignoruj
