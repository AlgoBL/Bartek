
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import streamlit as st

# Regime label constants (v3.0 - 4-state HMM style)
REGIME_BULL_QUIET = "Bull Quiet (Low Vol, Risk-On)"
REGIME_BULL_VOL   = "Bull Volatile (High Vol, Risk-On)"
REGIME_BEAR       = "Bear Market (Negative Return, Risk-Off)"
REGIME_CRISIS     = "Crisis/Crash (Extreme Vol, Risk-Off)"


class MarketObserver:
    def __init__(self, n_regimes=4):
        """
        Market regime detector using Gaussian Mixture Models (GMM).
        Upgraded to 4-state (Bull Quiet / Bull Volatile / Bear / Crisis).
        Input features: returns + rolling_vol + momentum + skewness + kurtosis.
        Reference: Kritzman et al. (2012) — Regime Shifts.
        """
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=42
        )
        self.trained = False
        self.regime_labels = {}
        # Key states for backtest decision logic
        self.high_vol_state = 0
        self.crisis_state = 0
        self.bull_quiet_state = 1

    def _build_features(self, returns_series: pd.Series) -> np.ndarray:
        """
        Features: [return, rolling_vol_21, momentum_63, rolling_skew_63, rolling_kurt_63]
        """
        df = pd.DataFrame({"r": returns_series})
        df["rol_vol"] = df["r"].rolling(21, min_periods=5).std().fillna(df["r"].std())
        df["momentum"] = df["r"].rolling(63, min_periods=10).mean().fillna(0)
        df["rol_skew"] = df["r"].rolling(63, min_periods=10).skew().fillna(0)
        df["rol_kurt"] = df["r"].rolling(63, min_periods=10).kurt().fillna(0)
        return df[["r", "rol_vol", "momentum", "rol_skew", "rol_kurt"]].values

    def fit(self, data: pd.Series):
        """
        Assigns regime labels based on mean vol (feature 1) and mean return (feature 0).
        """
        X = self._build_features(data)
        self.model.fit(X)
        self.trained = True

        # Means: [mu_ret, mu_vol, mu_mom, mu_skew, mu_kurt]
        means = self.model.means_
        
        # Sort by volatility first
        vol_order = np.argsort(means[:, 1]) # ascending
        
        # 1. Bull Quiet: Lowest Vol
        self.bull_quiet_state = int(vol_order[0])
        self.regime_labels[self.bull_quiet_state] = REGIME_BULL_QUIET
        
        # 2. Crisis: Extreme Vol
        self.crisis_state = int(vol_order[-1])
        self.high_vol_state = self.crisis_state
        self.regime_labels[self.crisis_state] = REGIME_CRISIS
        
        # 3. Handle middle two: check returns
        mid_indices = vol_order[1:3]
        if means[mid_indices[0], 0] > means[mid_indices[1], 0]:
            bull_vol = mid_indices[0]
            bear = mid_indices[1]
        else:
            bull_vol = mid_indices[1]
            bear = mid_indices[0]
            
        self.regime_labels[int(bull_vol)] = REGIME_BULL_VOL
        self.regime_labels[int(bear)] = REGIME_BEAR

    def predict_regime(self, data: pd.Series) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model not trained.")
        X = self._build_features(data)
        return self.model.predict(X)

    def predict_regime_proba(self, data: pd.Series) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model not trained.")
        X = self._build_features(data)
        return self.model.predict_proba(X)

    def get_regime_desc(self, state: int) -> str:
        return self.regime_labels.get(state, f"Regime {state}")


def get_market_regimes(returns_data: pd.Series, progress_callback=None):
    """
    Pipeline: fit 4-state GMM (Bull Quiet / Bull Vol / Bear / Crisis)
    """
    observer = MarketObserver(n_regimes=4)
    if progress_callback:
        progress_callback(0.1, "Trenowanie 4-stanowego modelu (HMM-style)...")
    observer.fit(returns_data)
    if progress_callback:
        progress_callback(0.5, "Klasyfikacja reżimów Barbella...")
    regimes = observer.predict_regime(returns_data)
    return regimes, observer
