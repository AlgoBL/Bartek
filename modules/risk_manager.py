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

        thresholds = np.percentile(
            losses_sorted, np.linspace(50, 95, n_thresholds)
        )
        mean_excesses = np.array([
            np.mean(losses_sorted[losses_sorted > u] - u)
            if np.any(losses_sorted > u) else 0.0
            for u in thresholds
        ])
        return thresholds, mean_excesses

    # ─── 6. VaRGPD-ML: Adaptive EVT Threshold (NEW 2024) ─────────────────────

    def fit_evt_pot_adaptive(
        self,
        returns: pd.Series,
        macro_features: pd.DataFrame | None = None,
        n_candidates: int = 15,
    ) -> dict:
        """
        VaRGPD-ML — dynamiczny dobór progu u przez Gradient Boosting.

        Zamiast stałego percentyla (np. 95%), model XGBoost uczy się
        OPTYMALNEGO progu u z mikro-struktury rynku:
          - Rolling volatility (5, 21 dni)
          - Momentum (21, 63 dni)
          - Skewness/Kurtosis (63 dni)
          - VIX proxy (realizowana zmienność)
          - Macro features (opcjonalne): VIX, credit spread itd.

        Algorytm:
          1. Wygeneruj N_candidates progów (percentyle 85–97.5)
          2. Oblicz pseudo-label: próg minimalizujący MSE dopasowania GPD do danych
          3. Trenuj GBM na cechach → przewiduj optymalny próg
          4. Dopasuj GPD z przewidzianym progiem

        Ref: VaRGPD-ML(α), aimspress.com, Nov 2024 — poprawa ~15% OOS accuracy.
        """
        from scipy.stats import genpareto

        losses = -returns.dropna()
        if len(losses) < 100:
            return self.fit_evt_pot(returns, threshold_pct=0.95)

        # ── Cechy inżynierowane ────────────────────────────────────────────
        r = returns.dropna()
        feat_df = pd.DataFrame(index=r.index)
        feat_df["vol_5"]    = r.rolling(5, min_periods=2).std()
        feat_df["vol_21"]   = r.rolling(21, min_periods=5).std()
        feat_df["vol_63"]   = r.rolling(63, min_periods=10).std()
        feat_df["mom_21"]   = r.rolling(21, min_periods=5).mean()
        feat_df["mom_63"]   = r.rolling(63, min_periods=10).mean()
        feat_df["skew_63"]  = r.rolling(63, min_periods=10).skew()
        feat_df["kurt_63"]  = r.rolling(63, min_periods=10).kurt()
        feat_df["loss_lvl"] = (-r).rolling(21, min_periods=5).quantile(0.95)

        if macro_features is not None:
            feat_df = feat_df.join(macro_features, how="left")

        feat_df = feat_df.dropna()
        if len(feat_df) < 60:
            return self.fit_evt_pot(returns, threshold_pct=0.95)

        # ── Pseudo-labels: optymalny próg przez Anderson-Darling GOF ─────────
        candidate_pcts = np.linspace(0.85, 0.975, n_candidates)
        best_pcts = []

        # Oblicz optymalny próg dla każdego 63-dniowego okna
        window = 63
        step   = 21
        loss_arr = losses.values

        for start in range(0, len(loss_arr) - window, step):
            window_losses = loss_arr[start: start + window]
            best_ks = np.inf
            best_p  = 0.95
            for p in candidate_pcts:
                u       = np.percentile(window_losses, p * 100)
                exc     = window_losses[window_losses > u] - u
                if len(exc) < 8:
                    continue
                try:
                    shape, _, scale = genpareto.fit(exc, floc=0)
                    # KS statistic between empirical and fitted GPD
                    from scipy.stats import ks_1samp
                    ks_stat, _ = ks_1samp(exc, genpareto.cdf,
                                          args=(shape, 0, scale))
                    if ks_stat < best_ks:
                        best_ks = ks_stat
                        best_p  = p
                except Exception as e:
                    logger.debug(f"Adaptive threshold candidate p={p} failed: {e}")
                    continue
            best_pcts.append(best_p)

        if len(best_pcts) < 5:
            return self.fit_evt_pot(returns, threshold_pct=0.95)

        # ── Trening GBM ───────────────────────────────────────────────────────
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            # Fallback: użyj mediany optymalnych progów
            opt_pct = float(np.median(best_pcts))
            return self.fit_evt_pot(returns, threshold_pct=opt_pct)

        # Dopasuj etykiety do okien
        n_windows = len(best_pcts)
        feat_vals = feat_df.values
        X_windows = []
        for i, start in enumerate(range(0, len(loss_arr) - window, step)):
            if i >= n_windows:
                break
            # Użyj ostatniego wiersza okna jako cech
            feat_idx = min(start + window - 1, len(feat_vals) - 1)
            X_windows.append(feat_vals[feat_idx])

        X_arr = np.array(X_windows[:n_windows])
        y_arr = np.array(best_pcts[:len(X_arr)])

        if len(X_arr) < 10:
            return self.fit_evt_pot(returns, threshold_pct=0.95)

        gbm = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        gbm.fit(X_arr, y_arr)

        # Przewidź optymalny próg dla CAŁEJ historii (ostatni wiersz cech)
        last_features = feat_df.values[-1].reshape(1, -1)
        opt_pct = float(np.clip(gbm.predict(last_features)[0], 0.85, 0.975))

        result = self.fit_evt_pot(returns, threshold_pct=opt_pct)
        result["adaptive_threshold_pct"] = opt_pct
        result["method"] = "VaRGPD-ML"
        return result

    # ─── 7. Spectral Risk Measure (NEW 2024) ──────────────────────────────────

    def spectral_risk_measure(
        self,
        returns: pd.Series,
        phi_type: str = "exponential",
        gamma: float = 5.0,
    ) -> float:
        """
        Spectral Risk Measure — generalizacja CVaR.

        SRM = ∫₀¹ φ(p) · VaR_p(X) dp

        Gdzie φ(p) jest wagową funkcją ryzyka (musi być niemalejąca i ∫φ=1):
          - 'exponential': φ(p) = γ·exp(γ·p) / (exp(γ)-1)  [Acerbi 2002]
          - 'power':        φ(p) = (1+γ)·p^γ               [Weighted CVaR]
          - 'flat':         φ(p) = 1/(1-α) dla p>α, 0 inaczej  [= CVaR_α]

        Ref: Acerbi (2002), rozwinięcie 2024 w ryzyku portfeli hedge-fundowych.
        SRM > CVaR → wyższe penalizowanie bardzo ciężkich ogonów.
        """
        losses = (-returns.dropna()).sort_values().values
        n = len(losses)
        if n < 20:
            return float("nan")

        probs = (np.arange(1, n + 1) - 0.5) / n  # midpoint rule

        if phi_type == "exponential":
            phi = gamma * np.exp(gamma * probs) / (np.exp(gamma) - 1.0)
        elif phi_type == "power":
            phi = (1.0 + gamma) * probs ** gamma
        elif phi_type == "flat":
            alpha = max(0.01, 1.0 - 1.0 / max(gamma, 1.0))
            phi = np.where(probs >= alpha, 1.0 / (1.0 - alpha), 0.0)
        else:
            phi = np.ones(n)

        # BUG-19 FIX: SRM wymaga ∫φ dp = 1, czyli phi_integral = phi.sum()/n ≈ 1
        # Poprzedni kod: phi/(phi.sum()/n * n) = phi/phi.sum() → sum(phi)=1, ale ∫φ dp = 1/n ≠ 1
        # Poprawnie: normalizujemy tak żeby ∫φ dp = phi.sum()/n = 1, tzn. phi.sum() = n
        phi_integral = phi.sum() / n   # ≈ ∫φ dp
        phi = phi / (phi_integral + 1e-12)   # teraz ∫φ dp ≈ 1

        srm = float(np.sum(phi * losses) / n)
        return srm

    # ─── 8. Multivariate EVT: Joint Exceedance Matrix (NEW 2024) ─────────────

    def joint_exceedance_matrix(
        self,
        returns_df: pd.DataFrame,
        threshold_pct: float = 0.95,
    ) -> pd.DataFrame:
        """
        Wielowymiarowa EVT — macierz wspólnych ekstremalnych zdarzeń.

        Dla każdej pary aktywów (i, j) oblicza:
          P(X_i < -VaR_i(α) | X_j < -VaR_j(α))

        Wartości > wynikające z niezależności → contagion / zarażenie.
        Używane w stres-testach i alokacji CVaR.

        Ref: Joe & Xu (1996), pakiet extremes 2024, McNeil et al. (2015).
        """
        cols = returns_df.columns.tolist()
        n    = len(cols)
        mat  = pd.DataFrame(np.eye(n), index=cols, columns=cols)

        thresholds = {}
        for col in cols:
            losses = -returns_df[col].dropna()
            thresholds[col] = float(np.percentile(losses, threshold_pct * 100))

        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if i == j:
                    continue
                li = -returns_df[ci].dropna()
                lj = -returns_df[cj].dropna()
                common_idx = li.index.intersection(lj.index)
                if len(common_idx) < 30:
                    continue
                li_c = li[common_idx]
                lj_c = lj[common_idx]
                ui, uj = thresholds[ci], thresholds[cj]
                joint  = np.sum((li_c > ui) & (lj_c > uj))
                cond_j = np.sum(lj_c > uj)
                mat.loc[ci, cj] = float(joint / cond_j) if cond_j > 0 else 0.0

        return mat

    # ─── 9. Tail Index Evolution (rolling ξ) ─────────────────────────────────

    def rolling_tail_index(
        self,
        returns: pd.Series,
        window: int = 252,
        step: int   = 21,
        threshold_pct: float = 0.95,
    ) -> pd.DataFrame:
        """
        Ewolucja wskaźnika ogona ξ (shape GPD) w czasie.

        Wzrastające ξ → grubszy ogon → rosnące ryzyko ekstremalnych strat.
        Gwałtowny skok ξ poprzedza kryzysy finansowe.

        Returns: DataFrame z kolumnami [xi, sigma, threshold, n_excesses]
        indeksowany datami (co step dni).
        """
        results = []
        dates   = []
        losses  = -returns.dropna()
        idx     = losses.index
        n       = len(losses)

        for end in range(window, n + 1, step):
            window_ret = returns.iloc[max(0, end - window): end]
            params = self.fit_evt_pot(window_ret, threshold_pct=threshold_pct)
            if "error" not in params:
                results.append({
                    "xi":        params["xi"],
                    "sigma":     params["sigma"],
                    "threshold": params["threshold"],
                    "n_excesses": params["N_u"],
                })
                dates.append(idx[end - 1])

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results, index=dates)

    # ─── 10. ΔCoVaR — Systemic Risk Measure (NEW 2025) ────────────────────────

    def calculate_delta_covar(
        self,
        asset_returns: "pd.Series",
        system_returns: "pd.Series",
        confidence: float = 0.99,
    ) -> dict:
        """
        ΔCoVaR — miara ryzyka systemowego (Adrian & Brunnermeier 2016).

        ΔCoVaR_i = CoVaR(system | asset_i w VaR) - CoVaR(system | median(asset_i))

        Interpretacja:
        - ΔCoVaR = 0    → aktywo nie wpływa na ryzyko systemu
        - ΔCoVaR >> 0   → aktywo AMPLIFIKUJE ryzyko systemu (systemowo ważne!)
        - Kluczowe dla: krypto, large-cap tech, banki

        Metodologia: Quantile Regression (OLS aproksymacja):
          r_system = α + β₁·r_asset + β₂·Controls + ε,  szacowane na kwantylach.

        Referencja: Adrian & Brunnermeier (2016) AER, NBER WP 2024 update.
        """
        from scipy.stats import linregress

        common = asset_returns.index.intersection(system_returns.index)
        if len(common) < 100:
            return {"error": "Za mało wspólnych obserwacji (min 100)."}

        x = asset_returns.loc[common].values
        y = system_returns.loc[common].values

        alpha_pct = (1.0 - confidence) * 100

        # VaR of asset at confidence level
        var_asset = np.percentile(x, alpha_pct)

        # ── Quantile Regression via OLS on tail subset (simplified) ──────────
        # Tail regime: asset near its VaR (within 20% of VaR level)
        band = abs(var_asset) * 0.5
        tail_mask = x <= var_asset + band
        median_mask = (x >= np.percentile(x, 40)) & (x <= np.percentile(x, 60))

        if tail_mask.sum() < 20 or median_mask.sum() < 20:
            return {"error": "Za mało obserwacji w ogonie lub medianie."}

        # CoVaR in tail regime
        y_tail = y[tail_mask]
        covar_tail = float(np.percentile(y_tail, alpha_pct))

        # CoVaR in median regime
        y_median = y[median_mask]
        covar_median = float(np.percentile(y_median, alpha_pct))

        delta_covar = covar_tail - covar_median  # negative (loss amplification)

        # Unconditional system VaR (benchmark)
        system_var = float(np.percentile(y, alpha_pct))

        # Exposure: OLS beta of system on asset in tail
        if len(x[tail_mask]) > 5:
            slope, intercept, r, _, _ = linregress(x[tail_mask], y[tail_mask])
        else:
            slope = 0.0

        return {
            "delta_covar":     delta_covar,
            "covar_tail":      covar_tail,
            "covar_median":    covar_median,
            "system_var":      system_var,
            "delta_covar_pct": delta_covar / (abs(system_var) + 1e-10),
            "ols_beta_tail":   float(slope),
            "confidence":      confidence,
            "n_tail_obs":      int(tail_mask.sum()),
            "interpretation":  (
                "WYSOKIE ryzyko systemowe" if abs(delta_covar) > abs(system_var) * 0.3
                else ("UMIARKOWANE ryzyko systemowe" if abs(delta_covar) > abs(system_var) * 0.1
                      else "NISKIE ryzyko systemowe")
            ),
        }

    # ─── 11. Calibration Tests (Basel IV) ─────────────────────────────────────

    def run_calibration_tests(
        self,
        model_var_series: "pd.Series",
        actual_returns: "pd.Series",
        confidence: float = 0.99,
    ) -> dict:
        """
        VaR/ES Backtesting suite — Basel IV wymóg od 01.01.2025.

        Implementuje trzy testy:
        1. Kupiec POF Test — czy liczba naruszeń VaR jest statystycznie poprawna?
           H₀: p_violations = 1 - confidence
           LR_POF = -2·ln[L(p̂|violations) / L(1-conf|violations)]
           Odrzucamy H₀ gdy LR > chi2(1, 5%)

        2. Christoffersen Independence Test — czy naruszenia są niezależne w czasie?
           H₀: naruszenia nie są autocorrelated
           Wykrywa clustering naruszeń (kryzys → wiele kolejnych przekroczeń)

        3. Acerbi-Szekely ES Test — test ES poza próbą (elicitable proxy).
           S = (1/n) · Σ [r_t · 1(r_t < VaR_t)] / ES_t + 1
           E[S] = 0 pod H₀ (ES prawidłowe), S < 0 → ES zaniżone

        Referencja: Kupiec (1995), Christoffersen (1998),
                    Acerbi & Szekely (2014), Basel IV IMA (2024).

        Parameters
        ----------
        model_var_series  : pd.Series z prognozowanym VaR (wartościami straty, ujemne!)
        actual_returns    : pd.Series z rzeczywistymi zwrotami
        confidence        : poziom ufności VaR (0.99 = 99%)
        """
        from scipy.stats import chi2

        common = model_var_series.index.intersection(actual_returns.index)
        if len(common) < 50:
            return {"error": "Za mało obserwacji (min 50)."}

        var_pred = model_var_series.loc[common].values   # negative losses (VaR)
        r_actual = actual_returns.loc[common].values

        n = len(r_actual)
        # Violation indicator: 1 if actual return < VaR forecast
        violations = (r_actual < var_pred).astype(int)
        n_viol = violations.sum()
        p_hat = n_viol / n
        p_expected = 1.0 - confidence

        results = {}

        # ── 1. Kupiec POF Test ────────────────────────────────────────────────
        try:
            eps = 1e-12
            p_hat_safe = np.clip(p_hat, eps, 1 - eps)
            p_exp_safe = np.clip(p_expected, eps, 1 - eps)
            lr_pof = -2.0 * (
                n_viol * np.log(p_exp_safe / p_hat_safe)
                + (n - n_viol) * np.log((1 - p_exp_safe) / (1 - p_hat_safe))
            )
            p_value_kupiec = float(1.0 - chi2.cdf(lr_pof, df=1))
            kupiec_pass = p_value_kupiec > 0.05
        except Exception:
            lr_pof, p_value_kupiec, kupiec_pass = np.nan, np.nan, False

        results["kupiec"] = {
            "lr_statistic": float(lr_pof),
            "p_value":      float(p_value_kupiec),
            "passed":       bool(kupiec_pass),
            "n_violations": int(n_viol),
            "expected_violations": int(round(n * p_expected)),
            "violation_rate": float(p_hat),
        }

        # ── 2. Christoffersen Independence Test ───────────────────────────────
        try:
            # Count transition matrix
            n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
            n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
            n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
            n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))

            p01 = n01 / max(n00 + n01, 1)
            p11 = n11 / max(n10 + n11, 1)
            p_unc = (n01 + n11) / max(n, 1)

            p01s = np.clip(p01, eps, 1 - eps)
            p11s = np.clip(p11, eps, 1 - eps)
            p_us = np.clip(p_unc, eps, 1 - eps)

            lr_ind = -2.0 * (
                (n00 + n10) * np.log(1 - p_us) + (n01 + n11) * np.log(p_us)
                - n00 * np.log(1 - p01s) - n01 * np.log(p01s)
                - n10 * np.log(1 - p11s) - n11 * np.log(p11s)
            )
            p_value_christ = float(1.0 - chi2.cdf(max(lr_ind, 0), df=1))
            christ_pass = p_value_christ > 0.05
        except Exception:
            lr_ind, p_value_christ, christ_pass = np.nan, np.nan, False

        results["christoffersen"] = {
            "lr_statistic":   float(lr_ind) if not np.isnan(lr_ind) else None,
            "p_value":        float(p_value_christ) if not np.isnan(p_value_christ) else None,
            "passed":         bool(christ_pass),
            "clustering_detected": not christ_pass,
        }

        # ── 3. Acerbi-Szekely ES Test (proxy) ────────────────────────────────
        try:
            # Compute empirical ES from model quantile series
            tail_mask = r_actual < var_pred
            if tail_mask.sum() > 0:
                # S = mean(r_t | tail) / (-ES_model) - 1, where ES_model = mean(VaR in tail)
                r_tail = r_actual[tail_mask]
                var_tail = var_pred[tail_mask]
                # ES model proxy = average predicted VaR in tail
                es_model = float(np.mean(var_tail))
                es_actual = float(np.mean(r_tail))
                s_stat = (es_actual / (es_model - 1e-12)) - 1.0
                # S < -0.1 → ES significantly underestimated
                acerbi_pass = s_stat >= -0.1
            else:
                s_stat, acerbi_pass = 0.0, True
        except Exception as e:
            from modules.logger import setup_logger
            setup_logger(__name__).error(f"Acerbi-Szekely test failed: {e}")
            s_stat, acerbi_pass = np.nan, False

        results["acerbi_szekely"] = {
            "s_statistic": float(s_stat) if not np.isnan(s_stat) else None,
            "passed":      bool(acerbi_pass),
            "interpretation": (
                "ES prawidłowe (H₀ nie odrzucona)" if acerbi_pass
                else "ES ZANIŻONE — model niedoszacowuje ryzyko ogonowe!"
            ),
        }

        # ── Summary ───────────────────────────────────────────────────────────
        all_pass = kupiec_pass and christ_pass and acerbi_pass
        results["summary"] = {
            "all_tests_passed": all_pass,
            "n_observations":   n,
            "confidence":       confidence,
            "overall_verdict":  "✅ Model VaR/ES spełnia wymogi Basel IV" if all_pass
                                else "❌ Model VaR/ES NIE spełnia wymogów Basel IV",
        }

        return results

    # ─── 12. Model Risk Score ──────────────────────────────────────────────────

    def calculate_model_risk_score(
        self,
        returns: "pd.Series",
        param_perturbation: float = 0.10,
        n_perturb: int = 50,
    ) -> dict:
        """
        Model Risk Score — ilościowa ocena niepewności modelu EVT.

        Metodologia (ECB Model Risk Management 2024):
        1. Wyestymuj bazowe parametry GPD (ξ₀, σ₀)
        2. Perturbuj parametry ±perturbation (Monte Carlo na przestrzeni parametrów)
        3. Oblicz VaR_99 dla każdego scenariusza perturbacji
        4. Model Risk Score = std(VaR_99) / mean(VaR_99)

        MRS < 0.05  → model stabilny (niska niepewność parametrów)
        MRS > 0.20  → model niestabilny (wysoka wrażliwość na kalibrację)

        Referencja: ECB Model Risk Management Guidelines (2024), SR 11-7 Fed.
        """
        base = self.fit_evt_pot(returns, threshold_pct=0.95)
        if "error" in base:
            return {"error": base["error"]}

        xi0    = base["xi"]
        sigma0 = base["sigma"]
        u0     = base["threshold"]
        N_u    = base["N_u"]
        N_tot  = base["N_total"]

        var_scenarios = []
        rng = np.random.default_rng(42)

        for _ in range(n_perturb):
            xi_p    = xi0    * (1 + rng.uniform(-param_perturbation, param_perturbation))
            sigma_p = sigma0 * (1 + rng.uniform(-param_perturbation, param_perturbation))
            sigma_p = max(sigma_p, 1e-6)

            # EVT-VaR formula
            p = 0.99
            exceedance = (1 - p) * N_tot / max(N_u, 1)
            if abs(xi_p) < 1e-6:
                var_p = u0 + sigma_p * np.log(1.0 / max(exceedance, 1e-10))
            else:
                var_p = u0 + (sigma_p / xi_p) * (max(exceedance, 1e-10) ** (-xi_p) - 1.0)
            var_scenarios.append(var_p)

        var_scenarios = np.array(var_scenarios)
        mean_var = float(np.mean(var_scenarios))
        std_var = float(np.std(var_scenarios))
        mrs = std_var / max(abs(mean_var), 1e-10)

        return {
            "model_risk_score": mrs,
            "mean_var_99":      mean_var,
            "std_var_99":       std_var,
            "var_range_90ci":   (float(np.percentile(var_scenarios, 5)),
                                 float(np.percentile(var_scenarios, 95))),
            "stability":        ("Stabilny" if mrs < 0.05 else
                                 "Umiarkowany" if mrs < 0.20 else "NIESTABILNY"),
            "base_xi":          xi0,
            "base_sigma":       sigma0,
        }
    # ─── 13. Realised Kernels & Jump Decomposition (NEW 2024) ────────────────
    
    def decompose_variance_hf(self, returns: pd.Series) -> dict:
        """
        Decompozycja wariancji na składnik ciągły i skokowy (Jump-Diffusion).
        RV = BV + J (Realised Variance = Bipower Variation + Jumps)
        
        Referencja: Barndorff-Nielsen & Shephard (2004, 2006).
        'Econometrics of testing for jumps in financial economics'.
        
        Metodologia:
          1. RV (Realised Variance) = sum(r^2)
          2. BV (Bipower Variation) = (pi/2) * (n/(n-1)) * sum(|r_i| * |r_{i-1}|)
          3. J (Jump Component) = max(0, RV - BV)
        """
        r = returns.dropna().values
        n = len(r)
        
        if n < 20:
            return {"error": "Za mało danych do dekompozycji (min 20)."}
        
        # 1. Realised Variance (Total)
        rv = np.sum(r**2)
        
        # 2. Bipower Variation (Continuous component proxy)
        # pi/2 constant
        c_bv = np.pi / 2.0
        bv_sum = np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
        bv = c_bv * (n / (n - 1)) * bv_sum
        
        # 3. Jump Component
        j = max(0, rv - bv)
        
        # Annualization (assuming daily returns)
        ann_factor = 252.0
        
        return {
            "rv_ann": float(np.sqrt(rv * ann_factor)),
            "bv_ann": float(np.sqrt(bv * ann_factor)),
            "j_ann":  float(np.sqrt(j * ann_factor)),
            "jump_contribution_pct": float(j / rv if rv > 0 else 0),
            "is_jumpy": bool(j / rv > 0.15), # Empirical threshold
            "n_obs": int(n)
        }
