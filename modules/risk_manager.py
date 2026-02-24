"""
risk_manager.py — Zaawansowane zarządzanie ryzykiem i position sizing.

Implementuje:
1. Empirical Kelly — Position sizing oparty na momentach rozkładu (skew, kurtosis).
2. Risk Budgeting — Alokacja oparta na udziale w CVaR.
3. Volatility Targeting — Skalowanie pozycji do docelowej zmienności portfela.
4. Stop-Loss & Trailing Stop — Mechanizmy ochronne.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class RiskManager:
    def __init__(self, transaction_costs=None):
        self.costs = transaction_costs or {
            "equity_pl":    0.0019,
            "etf":          0.0005,
            "crypto":       0.0060,
            "bonds":        0.0000,
            "bid_ask":      0.0002,
        }

    # ─── 1. Empirical Kelly ───────────────────────────────────────────────────
    
    def calculate_empirical_kelly(self, returns: pd.Series, rf: float = 0.04) -> float:
        """
        Oblicza Kelly Criterion na podstawie EMPIRYCZNYCH momentów (nie zakłada rozkładu normalnego).
        Optymalizuje f = argmax E[log(1 + f*r)].
        """
        r = returns.values
        rf_daily = (1 + rf)**(1/252) - 1
        excess = r - rf_daily
        
        def log_wealth(f):
            # Penalizujemy bankructwo (f*r < -1)
            wealth = 1 + f * excess
            if np.any(wealth <= 0):
                return 1e10
            return -np.mean(np.log(wealth))

        res = minimize(log_wealth, 
                       x0=[0.5], 
                       bounds=[(0, 2.0)], # Max dźwignia 2x
                       method='SLSQP')
        
        return float(res.x[0])

    # ─── 2. Risk Budgeting (CVaR) ─────────────────────────────────────────────
    
    def allocate_risk_budget(self, returns_df: pd.DataFrame, target_cvar_contribs: np.ndarray = None) -> np.ndarray:
        """
        Alokacja ERC (Equal Risk Contribution) pod kątem CVaR.
        Każde aktywo wnosi tyle samo do całkowitego CVaR portfela.
        """
        n = returns_df.shape[1]
        if target_cvar_contribs is None:
            target_cvar_contribs = np.ones(n) / n
            
        def objective(w):
            w = w / np.sum(w)
            port_ret = returns_df.values @ w
            var = np.percentile(port_ret, 5)
            cvar = -np.mean(port_ret[port_ret <= var])
            
            # Marginalny wkład: w_i * E[r_i | port_loss]
            tail_indices = np.where(port_ret <= var)[0]
            if len(tail_indices) == 0:
                return 1.0
            
            marginal_cvar = -np.mean(returns_df.values[tail_indices, :], axis=0)
            actual_contribs = w * marginal_cvar
            # Błąd sumy kwadratów od celu
            return np.sum((actual_contribs / cvar - target_cvar_contribs)**2)

        w0 = np.ones(n) / n
        res = minimize(objective, w0, bounds=[(0.01, 0.5)]*n, 
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        return res.x / np.sum(res.x)

    # ─── 3. Volatility Targeting ──────────────────────────────────────────────
    
    def get_vol_target_multiplier(self, current_vol: float, target_vol: float = 0.15) -> float:
        """
        Zwraca mnożnik dźwigni by osiągnąć docelowe vol (np. 15%).
        """
        if current_vol <= 0:
            return 1.0
        return target_vol / current_vol

    # ─── 4. Stop-Loss & Trailing Stop ─────────────────────────────────────────

    def check_stops(self, entry_price: float, current_price: float, 
                    max_price_since_entry: float, 
                    stop_loss_pct: float = 0.10, 
                    trailing_stop_pct: float = 0.05) -> bool:
        """
        Zwraca True jeśli należy zamknąć pozycję.
        """
        # Hard Stop Loss
        if current_price < entry_price * (1 - stop_loss_pct):
            return True
        
        # Trailing Stop
        if current_price < max_price_since_entry * (1 - trailing_stop_pct):
            return True
            
        return False

    def calculate_transaction_cost(self, asset_class: str, value: float, is_rebalance: bool = True) -> float:
        """Oblicza koszt transakcyjny (buy+sell jeśli rebalance)."""
        cost_rate = self.costs.get(asset_class, self.costs["etf"]) + self.costs["bid_ask"]
        multiplier = 2 if is_rebalance else 1
        return value * cost_rate * multiplier

