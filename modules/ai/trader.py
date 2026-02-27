import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RLTrader:
    def __init__(self):
        """
        Inicjalizuje RLTrader.
        W trybie heurystycznym działa bez pre-trenowanego modelu.
        W trybie RL wymaga wczytania modelu przez load_pretrained().
        """
        self._rl_model = None   # SB3 model (PPO/SAC) — opcjonalny
        self._n_assets  = None
        
    def predict_action(self, state):
        """
        Mock prediction function.
        State: [Market Trend, Volatility, Portfolio Value]
        Action: Target Leverage / Position Size adjustment (-1 to 1)
        """
        # Feature extraction (Mock)
        trend = state[0] # Positive = Bull, Negative = Bear
        volatility = state[1] # Low to High
        
        # RL Logic (Approximated Policy)
        # Low Vol + Bull Trend = Leverage Up (Aggressive Kelly)
        # High Vol + Bear Trend = Leverage Down (Cash/Bond)
        
        if trend > 0 and volatility < 0.15:
            action = 0.5 # Increase exposure
        elif trend < 0 or volatility > 0.30:
            action = -0.5 # Decrease exposure
        else:
            action = 0.0 # Hold
            
        return action

    def get_kelly_adjustment(self, current_volatility, regime, extra_evidence=0.0):
        """
        Returns a multiplier for the Kelly Fraction based on RL logic / Market State.
        Uses Bayesian Probability updating (Vanguard V7.0).
        """
        from modules.vanguard_math import bayesian_kelly_update
        
        prior = 0.5 
        evidence = extra_evidence
        
        if regime == "High Volatility (Risk-Off)":
            evidence -= 6.0
        elif regime == "Low Volatility (Risk-On)":
            evidence += 3.0
            
        if current_volatility < 0.12:
            evidence += 3.0
        elif current_volatility > 0.25:
            evidence -= 4.0
            
        posterior = bayesian_kelly_update(prior, evidence, max_evidence_score=10.0)
        multiplier = posterior * 2.0
        return max(0.0, min(1.5, multiplier))

    # ─── Deep RL Inference (PPO / SAC) ─────────────────────────────────────────

    def load_pretrained(self, path: str, algorithm: str = "PPO") -> bool:
        """
        Wczytuje pre-trenowany model PPO/SAC (Stable-Baselines3).

        Parameters
        ----------
        path      : ścieżka do pliku .zip (SB3 format)
        algorithm : 'PPO' lub 'SAC'

        Returns True jeśli sukces, False jeśli SB3 niedostępne lub błąd.
        """
        try:
            from modules.ai.rl_trainer import RLPortfolioTrainer
            trainer = RLPortfolioTrainer(
                returns_df=None, algorithm=algorithm
            )
            trainer.load(path)
            self._rl_model  = trainer
            logger.info(f"RLTrader: wczytano model {algorithm} z {path}")
            return True
        except Exception as e:
            logger.warning(f"RLTrader.load_pretrained failed: {e}")
            return False

    def predict_allocation(
        self,
        obs: np.ndarray,
        n_assets: int | None = None,
    ) -> np.ndarray | None:
        """
        Inferencja Deep RL: obserwacja → wagi portfela.

        Jeśli model nie jest załadowany, zwraca None
        (caller powinien użyć fallback heurystyki).

        Parameters
        ----------
        obs       : (obs_dim,) wektor obserwacji z PortfolioEnv
        n_assets  : liczba aktywów (do równych wag fallback)

        Returns
        -------
        weights : (n_assets,) znormalizowane wagi lub None
        """
        if self._rl_model is None:
            return None
        try:
            return self._rl_model.predict(obs)
        except Exception as e:
            logger.warning(f"RLTrader.predict failed: {e}")
            return None

    def is_rl_active(self) -> bool:
        """Sprawdza czy pre-trenowany model RL jest załadowany."""
        return self._rl_model is not None
