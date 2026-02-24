import numpy as np
import pandas as pd
import streamlit as st

class RLTrader:
    def __init__(self):
        """
        Initializes the RL Trader.
        For this simplified web-app version, we use a heuristic-based "Pre-trained" logic
        because training a TD3 agent from scratch takes too long for a user session.
        """
        pass
        
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
        Now uses Bayesian Probability updating (Vanguard V7.0)
        """
        from modules.vanguard_math import bayesian_kelly_update
        
        # Base neutral prior
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
        
        # Scale posterior [0, 1] to a Kelly Multiplier [0, ~2.0]
        # 0.5 prior -> 1.0 multiplier
        multiplier = posterior * 2.0
        return max(0.0, min(1.5, multiplier)) # Cap at 1.5x leverage

