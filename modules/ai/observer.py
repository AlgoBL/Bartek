
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import streamlit as st

# Regime label constants
REGIME_BULL = "Low Volatility (Risk-On / Bull)"
REGIME_BEAR = "Medium Volatility (Risk-Off / Bear)"
REGIME_CRISIS = "High Volatility (Crisis / Crash)"


class MarketObserver:
    def __init__(self, n_regimes=3):
        """
        Market regime detector using Gaussian Mixture Models (GMM).
        Upgraded from 2-state to 3-state (Bull / Bear / Crisis).
        Input features: returns + rolling_vol + momentum + skewness.
        Reference: Kritzman et al. (2012) — Regime Shifts: Implications for Dynamic Strategies.
        """
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,          # Run 5 inits to avoid bad local optima
            random_state=42
        )
        self.trained = False
        self.regime_labels = {}   # state_idx -> label string
        self.high_vol_state = 0
        self.low_vol_state = 1
        self.mid_vol_state = 2

    def _build_features(self, returns_series: pd.Series) -> np.ndarray:
        """
        Build multi-feature matrix from raw returns.
        Features: [return, rolling_vol_21, momentum_63, rolling_skew_63]
        """
        df = pd.DataFrame({"r": returns_series})
        df["rol_vol"] = df["r"].rolling(21, min_periods=5).std().fillna(df["r"].std())
        df["momentum"] = df["r"].rolling(63, min_periods=10).mean().fillna(0)
        df["rol_skew"] = df["r"].rolling(63, min_periods=10).skew().fillna(0)
        return df[["r", "rol_vol", "momentum", "rol_skew"]].values

    def fit(self, data: pd.Series):
        """
        Fits the GMM to multi-feature data.
        Assigns regime labels by sorting states on mean volatility.
        """
        X = self._build_features(data)
        self.model.fit(X)
        self.trained = True

        # Identify state volatilities from GMM means (feature index 1 = rol_vol)
        vol_means = [self.model.means_[i][1] for i in range(self.n_regimes)]
        sorted_states = np.argsort(vol_means)  # ascending: low vol → high vol

        if self.n_regimes >= 3:
            self.low_vol_state = int(sorted_states[0])
            self.mid_vol_state = int(sorted_states[1])
            self.high_vol_state = int(sorted_states[-1])
        else:
            self.low_vol_state = int(sorted_states[0])
            self.high_vol_state = int(sorted_states[-1])
            self.mid_vol_state = int(sorted_states[0])

        self.regime_labels = {
            self.low_vol_state: REGIME_BULL,
            self.high_vol_state: REGIME_CRISIS,
        }
        if self.n_regimes >= 3:
            self.regime_labels[self.mid_vol_state] = REGIME_BEAR

    def predict_regime(self, data: pd.Series) -> np.ndarray:
        """
        Predict hard regime label (integer) for each time step.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        X = self._build_features(data)
        return self.model.predict(X)

    def predict_regime_proba(self, data: pd.Series) -> np.ndarray:
        """
        Predict soft regime probabilities. Shape: (n_samples, n_regimes).
        Columns ordered as: [P(Bull), P(Bear), P(Crisis)]
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        X = self._build_features(data)
        proba = self.model.predict_proba(X)  # (n, n_regimes)
        # Reorder columns: low_vol, mid_vol, high_vol
        col_order = [self.low_vol_state]
        if self.n_regimes >= 3:
            col_order.append(self.mid_vol_state)
        col_order.append(self.high_vol_state)
        return proba[:, col_order]

    def get_regime_desc(self, state: int) -> str:
        return self.regime_labels.get(state, f"Regime {state}")


def get_market_regimes(returns_data: pd.Series, progress_callback=None):
    """
    Pipeline: fit 3-state GMM → predict regimes and probabilities.
    Returns (regimes_array, observer_model)
    """
    observer = MarketObserver(n_regimes=3)

    if progress_callback:
        progress_callback(0.1, "Trenowanie 3-stanowego modelu GMM (Bull/Bear/Crisis)...")

    observer.fit(returns_data)

    if progress_callback:
        progress_callback(0.5, "Przewidywanie reżimów rynkowych...")

    regimes = observer.predict_regime(returns_data)

    if progress_callback:
        progress_callback(0.9, "Analiza reżimów zakończona.")

    return regimes, observer
