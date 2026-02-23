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

    def get_kelly_adjustment(self, current_volatility, regime):
        """
        Returns a multiplier for the Kelly Fraction based on RL logic / Market State.
        """
        if regime == "High Volatility (Risk-Off)":
            return 0.0 # Sit out
        elif current_volatility < 0.10: # Calm bull market
            return 1.2 # Boost
        else:
            return 1.0 # Normal
