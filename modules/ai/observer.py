
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import streamlit as st

class MarketObserver:
    def __init__(self, n_regimes=2):
        """
        Initializes the Market Observer using Gaussian Mixture Models (GMM).
        GMM is a probabilistic model that works similarly to HMM for clustering regimes,
        but doesn't model the transition probabilities explicitly (stateless clustering).
        It is much easier to install than hmmlearn on Windows.
        """
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes, 
            covariance_type="full", 
            random_state=42
        )
        self.trained = False
        
    def fit(self, data):
        """
        Fits the GMM to the provided data (e.g., returns).
        """
        # Reshape for Sklearn: [n_samples, n_features]
        X = data.values.reshape(-1, 1)
        self.model.fit(X)
        self.trained = True
        
        # Identify which hidden state corresponds to High Volatility
        # The state with the highest variance is the "Crash/Crisis" regime
        variances = [self.model.covariances_[i][0][0] for i in range(self.n_regimes)]
        self.regime_volatility_map = {i: var for i, var in enumerate(variances)}
        self.high_vol_state = np.argmax(variances)
        
    def predict_regime(self, data):
        """
        Predicts the regime for each time step.
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
            
        X = data.values.reshape(-1, 1)
        hidden_states = self.model.predict(X)
        
        return hidden_states

    def get_regime_desc(self, state):
        """
        Returns a human-readable description of the regime.
        """
        if state == self.high_vol_state:
            return "High Volatility (Risk-Off)"
        else:
            return "Low Volatility (Risk-On)"
            
def get_market_regimes(returns_data, progress_callback=None):
    """
    Helper function to run the pipeline.
    """
    observer = MarketObserver(n_regimes=2)
    
    if progress_callback:
        progress_callback(0.1, "Trenowanie modelu HMM (GMM)...")
        
    observer.fit(returns_data)
    
    if progress_callback:
        progress_callback(0.5, "Przewidywanie reżimów...")
        
    regimes = observer.predict_regime(returns_data)
    
    if progress_callback:
        progress_callback(0.9, "Analiza zakończona.")
        
    return regimes, observer
