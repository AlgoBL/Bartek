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
        if stop_loss_pct > 0.0:
            if current_price < entry_price * (1 - stop_loss_pct):
                return True
        
        # Trailing Stop
        if trailing_stop_pct > 0.0:
            if current_price < max_price_since_entry * (1 - trailing_stop_pct):
                return True
            
        return False

    def calculate_transaction_cost(self, asset_class: str, value: float, is_rebalance: bool = True) -> float:
        """Oblicza koszt transakcyjny (buy+sell jeśli rebalance)."""
        cost_rate = self.costs.get(asset_class, self.costs["etf"]) + self.costs["bid_ask"]
        multiplier = 2 if is_rebalance else 1
        return value * cost_rate * multiplier

    # ─── 5. Extreme Value Theory — POT (Peaks-Over-Threshold) ─────────────────

    def fit_evt_pot(
        self,
        returns: pd.Series,
        threshold_pct: float = 0.95,
    ) -> dict:
        """
        Dopasowuje GPD (Generalized Pareto Distribution) do ogona strat
        metodą POT (Peaks-Over-Threshold).

        Metodologia:
          1. Wyznacz próg u = empiryczny kwantyl threshold_pct strat
          2. Zbierz straty przekraczające próg: y_i = loss_i - u
          3. Dopasuj GPD: F(y|ξ,σ) = 1 - (1 + ξy/σ)^{-1/ξ}
          4. Szacuj EVT-VaR i EVT-CVaR z zamkniętych formuł

        Referencje:
          Balkema & de Haan (1974) — Pickands-Balkema-de Haan Theorem
          Embrechts, Klüppelberg & Mikosch (1997) — "Modelling Extremal Events"
          McNeil & Frey (2000) — "Estimation of tail-related risk measures for
                                  heteroscedastic financial time series"

        Parameters
        ----------
        returns       : pd.Series zwrotów portfela (mogą być ujemne)
        threshold_pct : kwantyl progu, np. 0.95 → 95. percentyl strat

        Returns
        -------
        dict z: xi (shape), sigma (scale), threshold, N_u, N_total,
                excesses (array), error (str jeśli błąd)
        """
        from scipy.stats import genpareto

        losses = -returns.dropna()  # straty = ujemne zwroty ze znakiem +

        if len(losses) < 50:
            return {"error": "Za mało obserwacji (min 50)."}

        u = float(np.percentile(losses, threshold_pct * 100))
        excesses = losses[losses > u].values - u

        if len(excesses) < 15:
            return {
                "error": (
                    f"Za mało pomiarów ponad próg u={u:.4f} "
                    f"(znaleziono {len(excesses)}, min 15). "
                    "Zmniejsz threshold_pct."
                )
            }

        # MLE fit GPD (loc=0 wymuszony przez POT — mierzymy od progu)
        try:
            shape, loc, scale = genpareto.fit(excesses, floc=0)
        except Exception as e:
            return {"error": f"GPD fit failed: {e}"}

        return {
            "xi":        float(shape),       # kształt: >0 heavy tail, =0 exponential
            "sigma":     float(scale),       # skala ogona
            "threshold": float(u),
            "N_u":       int(len(excesses)),
            "N_total":   int(len(losses)),
            "excesses":  excesses,
            "threshold_pct": threshold_pct,
        }

    def evt_var(self, evt_params: dict, confidence: float = 0.99) -> float:
        """
        EVT-VaR (Value at Risk) przy zadanym poziomie ufności przez GPD.

        Formuła: VaR_p = u + (σ/ξ) * [((1-p) * N/N_u)^{-ξ} - 1]

        Dla ξ=0: VaR_p = u + σ * ln((1-p) * N/N_u)^{-1}
        """
        if "error" in evt_params:
            return float("nan")

        xi    = evt_params["xi"]
        sigma = evt_params["sigma"]
        u     = evt_params["threshold"]
        N     = evt_params["N_total"]
        N_u   = evt_params["N_u"]
        p     = confidence

        exceedance_prob = (1 - p) * N / N_u  # P(X > VaR) / P(X > u)

        if abs(xi) < 1e-6:
            # Exponential tail (ξ → 0)
            return float(u + sigma * np.log(1.0 / exceedance_prob))
        else:
            return float(u + (sigma / xi) * (exceedance_prob ** (-xi) - 1.0))

    def evt_cvar(self, evt_params: dict, confidence: float = 0.99) -> float:
        """
        EVT-CVaR (Expected Shortfall) przez GPD — formuła zamknięta.

        Formuła: CVaR_p = VaR_p / (1 - ξ) + (σ - ξ*u) / (1 - ξ)

        Warunek istnienia: ξ < 1 (spełniony empirycznie; ξ>1 → nieskończona EV)
        """
        if "error" in evt_params:
            return float("nan")

        xi    = evt_params["xi"]
        sigma = evt_params["sigma"]
        u     = evt_params["threshold"]

        if xi >= 1.0:
            # CVaR nieskończone dla ξ ≥ 1 (Cauchy-like heavy tail)
            return float("inf")

        var = self.evt_var(evt_params, confidence)
        cvar = var / (1.0 - xi) + (sigma - xi * u) / (1.0 - xi)
        return float(cvar)

    def evt_full_metrics(self, returns: pd.Series, threshold_pct: float = 0.95) -> dict:
        """
        Kompletne metryki EVT: VaR i CVaR na poziomach 95%, 99%, 99.9%.

        Returns
        -------
        dict z: evt_params, var_95, cvar_95, var_99, cvar_99, var_999, cvar_999,
                tail_index (shape ξ), tail_type (str interpretacja)
        """
        params = self.fit_evt_pot(returns, threshold_pct)

        if "error" in params:
            return {"error": params["error"]}

        xi = params["xi"]
        if xi <= 0:
            tail_type = "Thin tail (exponential/sub-exponential) — bezpieczne ogony"
        elif xi < 0.5:
            tail_type = "Moderate heavy tail — typowe dla akcji (Pareto-like)"
        elif xi < 1.0:
            tail_type = "Heavy tail — typowe dla krypto/rynków wschodzących"
        else:
            tail_type = "EKSTREMALNE ogony! ξ≥1 → CVaR nieskończone (ostrożnie!)"

        return {
            "evt_params":  params,
            "var_95":      self.evt_var(params, 0.95),
            "cvar_95":     self.evt_cvar(params, 0.95),
            "var_99":      self.evt_var(params, 0.99),
            "cvar_99":     self.evt_cvar(params, 0.99),
            "var_999":     self.evt_var(params, 0.999),
            "cvar_999":    self.evt_cvar(params, 0.999),
            "tail_index":  xi,
            "tail_scale":  params["sigma"],
            "threshold":   params["threshold"],
            "n_excesses":  params["N_u"],
            "threshold_pct": threshold_pct,
            "tail_type":   tail_type,
        }

    def mean_excess_plot_data(
        self, returns: pd.Series, n_thresholds: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dane do wykresu Mean Excess Function (MEF / Mean Residual Life Plot).
        Wzrastająca MEF wskazuje na heavy tail (GPD z ξ > 0 jest uzasadnione).

        Returns: (thresholds, mean_excesses)
        """
        losses = (-returns.dropna()).values
        losses_sorted = np.sort(losses)
        n = len(losses_sorted)

        thresholds = np.percentile(
            losses_sorted, np.linspace(50, 95, n_thresholds)
        )
        mean_excesses = np.array([
            np.mean(losses_sorted[losses_sorted > u] - u)
            if np.any(losses_sorted > u) else 0.0
            for u in thresholds
        ])
        return thresholds, mean_excesses

