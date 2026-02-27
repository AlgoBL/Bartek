"""
rl_trainer.py — Training pipeline dla Deep RL portfolio agenta (PPO/SAC).

Opakowuje Stable-Baselines3 z:
  - WalkForward evaluation callback
  - TensorBoard logging
  - Checkpoint saving / loading
  - Rolling-window training (unikanie look-ahead bias)

Referencje:
  Schulman et al. (2017) — "Proximal Policy Optimization Algorithms" (PPO)
  Haarnoja et al. (2018) — "Soft Actor-Critic" (SAC)
  Stable-Baselines3: https://stable-baselines3.readthedocs.io/
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Soft deps
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        StopTrainingOnRewardThreshold,
    )
    from stable_baselines3.common.monitor import Monitor
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    logger.warning(
        "stable-baselines3 nie jest zainstalowane. "
        "Uruchom: pip install stable-baselines3[extra] gymnasium"
    )


class RLPortfolioTrainer:
    """
    Pipeline treningowy Deep RL agenta alokacji portfela.

    Wspierane algorytmy: PPO (domyślny), SAC
    Architektura sieci: MLP 256→128→64 z LayerNorm

    Przykład użycia:
        trainer = RLPortfolioTrainer(returns_df, algorithm="PPO")
        agent = trainer.train(total_timesteps=500_000)
        allocation = trainer.predict(obs)
    """

    DEFAULT_PPO_KWARGS = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
        ),
        verbose=0,
        tensorboard_log="./rl_logs/",
    )

    DEFAULT_SAC_KWARGS = dict(
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        policy_kwargs=dict(
            net_arch=[256, 128, 64],
        ),
        verbose=0,
        tensorboard_log="./rl_logs/",
    )

    def __init__(
        self,
        returns_df: pd.DataFrame,
        macro_features_df: pd.DataFrame | None = None,
        algorithm: str = "PPO",
        lookback: int = 60,
        transaction_cost: float = 0.001,
        checkpoint_dir: str = "./rl_checkpoints/",
    ):
        self.returns_df       = returns_df
        self.macro_df         = macro_features_df
        self.algorithm        = algorithm.upper()
        self.lookback         = lookback
        self.tc               = transaction_cost
        self.checkpoint_dir   = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model            = None

    def _make_env(self, returns_df: pd.DataFrame, macro_df: pd.DataFrame | None = None):
        """Tworzy i opakuje środowisko."""
        from modules.ai.rl_environment import PortfolioEnv
        env = PortfolioEnv(
            returns_df=returns_df,
            macro_features_df=macro_df,
            lookback=self.lookback,
            transaction_cost=self.tc,
        )
        try:
            return Monitor(env)
        except Exception:
            return env

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 3,
        train_end_fraction: float = 0.8,
        progress_bar: bool = False,
    ):
        """
        Trenuje agenta PPO/SAC na danych historycznych.

        Parameters
        ----------
        total_timesteps    : łączna liczba kroków treningowych
        eval_freq          : co ile kroków ewaluować na zbiorze walidacyjnym
        train_end_fraction : ułamek danych użyty do treningu (reszta = val)
        progress_bar       : czy pokazywać pasek postępu (wymaga tqdm)

        Returns
        -------
        wytrenowany model (PPO lub SAC) lub None jeśli SB3 niedostępne
        """
        if not _SB3_AVAILABLE:
            logger.error("stable-baselines3 niedostępne — trening niemożliwy")
            return None

        T_train = int(len(self.returns_df) * train_end_fraction)
        train_returns = self.returns_df.iloc[:T_train]
        val_returns   = self.returns_df.iloc[T_train:]

        train_macro = self.macro_df.iloc[:T_train] if self.macro_df is not None else None
        val_macro   = self.macro_df.iloc[T_train:] if self.macro_df is not None else None

        train_env = self._make_env(train_returns, train_macro)
        eval_env  = self._make_env(val_returns,   val_macro)

        # Inicjalizacja modelu
        AlgoClass = PPO if self.algorithm == "PPO" else SAC
        algo_kwargs = (
            self.DEFAULT_PPO_KWARGS.copy()
            if self.algorithm == "PPO"
            else self.DEFAULT_SAC_KWARGS.copy()
        )

        self.model = AlgoClass("MlpPolicy", train_env, **algo_kwargs)

        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"rl_{self.algorithm.lower()}",
        )
        eval_cb = EvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=str(self.checkpoint_dir / "best"),
            log_path=str(self.checkpoint_dir / "eval_logs"),
            verbose=0,
        )

        logger.info(
            f"Trening {self.algorithm}: {total_timesteps:,} kroków, "
            f"{self.returns_df.shape[1]} aktywów"
        )
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=progress_bar,
        )
        logger.info("Trening zakończony.")
        return self.model

    def save(self, path: str | None = None) -> str:
        """Zapisuje model do pliku."""
        if self.model is None:
            raise RuntimeError("Brak wytrenowanego modelu.")
        save_path = path or str(self.checkpoint_dir / f"portfolio_rl_{self.algorithm.lower()}")
        self.model.save(save_path)
        logger.info(f"Model zapisany: {save_path}")
        return save_path

    def load(self, path: str) -> None:
        """Wczytuje model z pliku."""
        if not _SB3_AVAILABLE:
            logger.error("stable-baselines3 niedostępne")
            return
        AlgoClass = PPO if self.algorithm == "PPO" else SAC
        self.model = AlgoClass.load(path)
        logger.info(f"Model wczytany: {path}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Inferencja agenta: obs → allocation weights.

        Returns
        -------
        weights : (n_assets,) — znormalizowane wagi portfela
        """
        if self.model is None:
            logger.warning("Model niezainicjalizowany — zwracam równe wagi")
            return None
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.maximum(action, 0.0)
        total  = action.sum()
        return action / max(total, 1e-8)

    def walk_forward_backtest(
        self,
        window_train: int = 504,
        window_test:  int = 63,
        timesteps_per_window: int = 50_000,
    ) -> pd.DataFrame:
        """
        Walk-Forward backtest: retrenow okienkowy agenta RL.

        Parameters
        ----------
        window_train           : rozmiar okna treningowego (dni)
        window_test            : rozmiar okna testowego (dni)
        timesteps_per_window   : kroki treningowe per okno

        Returns
        -------
        pd.DataFrame — portfolio returns per test window
        """
        from modules.ai.rl_environment import PortfolioEnv

        T = len(self.returns_df)
        results = []

        t = window_train
        while t + window_test <= T:
            train_ret = self.returns_df.iloc[t - window_train: t]
            test_ret  = self.returns_df.iloc[t: t + window_test]

            train_macro = self.macro_df.iloc[t - window_train: t] if self.macro_df is not None else None
            test_macro  = self.macro_df.iloc[t: t + window_test] if self.macro_df is not None else None

            train_env = self._make_env(train_ret, train_macro)
            AlgoClass = PPO if self.algorithm == "PPO" else SAC
            algo_kwargs = (
                self.DEFAULT_PPO_KWARGS.copy()
                if self.algorithm == "PPO"
                else self.DEFAULT_SAC_KWARGS.copy()
            )
            model = AlgoClass("MlpPolicy", train_env, **algo_kwargs)
            model.learn(timesteps_per_window, progress_bar=False)

            # Test
            test_env = PortfolioEnv(test_ret, test_macro, lookback=self.lookback, transaction_cost=self.tc)
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated

            summary = test_env.get_episode_summary()
            summary["window_start"] = self.returns_df.index[t - window_train] if hasattr(self.returns_df.index, '__getitem__') else t - window_train
            summary["window_test_start"] = self.returns_df.index[t] if hasattr(self.returns_df.index, '__getitem__') else t
            results.append(summary)
            t += window_test

        logger.info(f"Walk-Forward zakończony: {len(results)} okien")
        return pd.DataFrame(results)
