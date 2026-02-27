"""
black_litterman.py — Model Black-Litterman z AI-generowanymi views.

Implementuje kompletny model B-L łączący:
  1. CAPM Prior (market-implied returns Π = λΣw_mkt)
  2. AI Views (P, Q, Ω) generowane przez agentów LocalCIO / LocalEconomist
  3. Posterior (μ_BL): bayesowskie połączenie prioirytu z views

Referencje:
  Black & Litterman (1992) — "Global Portfolio Optimization"
  He & Litterman (1999) — "The Intuition Behind Black-Litterman Model Portfolios"
  Idzorek (2005) — "A Step-by-Step Guide to the Black-Litterman Model"
  Meucci (2010) — "The Black-Litterman Approach: Original Model and Extensions"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modules.logger import setup_logger

logger = setup_logger(__name__)


class BlackLittermanEngine:
    """
    Model Black-Litterman z automatycznym mapowaniem views z agentów AI.

    Użycie:
        engine = BlackLittermanEngine()
        pi = engine.compute_implied_returns(sigma, w_mkt)
        P, Q, omega = engine.build_views_from_agents(cio_thesis, econ_analysis, assets)
        mu_bl, sigma_bl = engine.posterior_returns(sigma, pi, P, Q, omega)
        w_opt = engine.optimize_portfolio(mu_bl, sigma_bl)
    """

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        """
        Parameters
        ----------
        risk_aversion : λ — współczynnik awersji do ryzyka (typowo 2.5–3.5)
        tau           : skala niepewności prioru (typowo 1/T lub 0.05)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

    # ─── 1. CAPM PRIOR ────────────────────────────────────────────────────────

    def compute_implied_returns(
        self,
        sigma: np.ndarray,
        w_mkt: np.ndarray,
    ) -> np.ndarray:
        """
        Oblicza implikowane zwroty równowagi rynkowej (CAPM prior).

        Π = λ · Σ · w_mkt

        gdzie:
          λ = risk_aversion (typowo 2.5)
          Σ = macierz kowariancji aktywów
          w_mkt = wagi rynkowe (np. kapitalizacja / równoważne)
        """
        return self.risk_aversion * sigma @ w_mkt

    # ─── 2. AI VIEWS → (P, Q, Ω) ──────────────────────────────────────────────

    def build_views_from_agents(
        self,
        cio_thesis: dict,
        econ_analysis: dict,
        asset_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Automatycznie mapuje wyniki agentów AI na macierze modelu B-L.

        Reguły mapowania:
          - risk_off + yield_curve_inverted  → Bonds > Equities (+3%/yr, conf=0.80)
          - risk_off + credit_spread > 3.5%  → Bonds > High_Yield (+5%/yr, conf=0.85)
          - risk_on + vix < 15               → Risky > Bonds (+5%/yr, conf=0.60)
          - VIX Backwardation               → Equities negative (-5%/yr, conf=0.90)
          - DXY > 108                        → Commodities/EM negative (-3%/yr, conf=0.70)
          - neutral                         → brak views (P empty)

        Returns
        -------
        P : (k, n) — view matrix
        Q : (k,)   — expected returns per view (annualized)
        Ω : (k, k) — uncertainty diagonal (variance of view errors)
        """
        n = len(asset_names)
        names_lower = [a.lower() for a in asset_names]
        views_P: list[np.ndarray] = []
        views_Q: list[float]      = []
        confidences: list[float]  = []

        regime    = cio_thesis.get("regime", "neutral")
        raw_sigs  = econ_analysis.get("raw_signals", {})
        vix_sig   = raw_sigs.get("vix_level", 0.3)          # 0=low,1=high
        yc_sig    = raw_sigs.get("yield_curve", 0.3)         # 0=normal,1=inverted
        cs_sig    = raw_sigs.get("credit_spread", 0.2)
        dxy_sig   = raw_sigs.get("dxy_strength", 0.3)

        def _idx(keywords: list[str]) -> int | None:
            """Zwraca indeks pierwszego aktywa pasującego do słów kluczowych."""
            for kw in keywords:
                for i, name in enumerate(names_lower):
                    if kw in name:
                        return i
            return None

        bond_idx   = _idx(["bond", "bil", "sgov", "tlt", "tob", "obligacj"])
        equity_idx = _idx(["spy", "equity", "akcj", "stock", "qqq", "iwm", "ivv"])
        risky_idx  = _idx(["btc", "crypto", "krypto", "risky", "etf"])
        em_idx     = _idx(["eem", "em ", "emerging", "rynki wschodzące"])
        comm_idx   = _idx(["gld", "gold", "złoto", "oil", "ropa", "copper"])

        # View 1: Risk-Off → Bonds outperform Equities
        if regime == "risk_off" and bond_idx is not None and equity_idx is not None:
            p_view         = np.zeros(n)
            p_view[bond_idx]   =  1.0
            p_view[equity_idx] = -1.0
            views_P.append(p_view)
            views_Q.append(0.03 + cs_sig * 0.04)   # +3–7%/yr
            confidences.append(0.75 + yc_sig * 0.15)

        # View 2: VIX Backwardation → Equities negative (absolute view)
        vix_backwardation = econ_analysis.get("score", 0) > 5.5 and vix_sig > 0.70
        if vix_backwardation and equity_idx is not None:
            p_view            = np.zeros(n)
            p_view[equity_idx] = 1.0
            views_P.append(p_view)
            views_Q.append(-0.05)   # -5%/yr oczekiwany zwrot
            confidences.append(0.85)

        # View 3: Risk-On → Risky assets outperform Bonds
        if regime == "risk_on" and risky_idx is not None and bond_idx is not None:
            p_view             = np.zeros(n)
            p_view[risky_idx]  =  1.0
            p_view[bond_idx]   = -1.0
            views_P.append(p_view)
            views_Q.append(0.05 - vix_sig * 0.03)  # +2–5%/yr
            confidences.append(0.55 + (1 - vix_sig) * 0.15)

        # View 4: Strong USD → EM/Commodities negative
        if dxy_sig > 0.65 and comm_idx is not None:
            p_view             = np.zeros(n)
            p_view[comm_idx]   = 1.0
            views_P.append(p_view)
            views_Q.append(-0.03)   # -3%/yr
            confidences.append(0.65)

        # Brak views → zwróć puste macierze (posterrior = prior)
        if not views_P:
            logger.info("Black-Litterman: brak views od agentów (neutral regime)")
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P = np.array(views_P)    # (k, n)
        Q = np.array(views_Q)    # (k,)

        # Idzorek (2005): Ω_ii = (1 - conf_i) / conf_i * (P_i Σ P_i^T)
        # Modeluje niepewność view jako funkcję confidence i wariancji portfela
        k = len(views_Q)
        omega_diag = np.zeros(k)
        for i in range(k):
            conf = np.clip(confidences[i], 0.01, 0.99)
            # Placeholder sigma (brak Σ w tym scope — użyj jednostkowej)
            p_var = np.dot(P[i], P[i])  # simplified: P_i * I * P_i^T
            omega_diag[i] = ((1.0 - conf) / conf) * p_var
        omega_diag = np.maximum(omega_diag, 1e-6)
        omega = np.diag(omega_diag)

        logger.info(
            f"Black-Litterman: {k} views wygenerowanych | "
            f"regime={regime} | Q={np.round(Q*100, 1)}"
        )
        return P, Q, omega

    def build_views_from_agents_with_sigma(
        self,
        cio_thesis: dict,
        econ_analysis: dict,
        asset_names: list[str],
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Jak build_views_from_agents, ale używa prawdziwej macierzy Σ dla Ω (Idzorek 2005).
        """
        P, Q, omega_placeholder = self.build_views_from_agents(
            cio_thesis, econ_analysis, asset_names
        )
        if P.shape[0] == 0:
            return P, Q, omega_placeholder

        k = P.shape[0]
        omega_diag = np.zeros(k)
        # Re-derive confidences from omega_placeholder diagonal (reverse Idzorek)
        for i in range(k):
            p_sigma_pt = float(P[i] @ sigma @ P[i])
            p_var_unit = float(np.dot(P[i], P[i]))
            if p_var_unit < 1e-10:
                omega_diag[i] = 1e-4
                continue
            ratio = omega_placeholder[i, i] / p_var_unit
            conf  = 1.0 / (1.0 + ratio)
            omega_diag[i] = ((1.0 - conf) / conf) * max(p_sigma_pt, 1e-6)
        return P, Q, np.diag(np.maximum(omega_diag, 1e-8))

    # ─── 3. POSTERIOR RETURNS ─────────────────────────────────────────────────

    def posterior_returns(
        self,
        sigma: np.ndarray,
        pi: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oblicza Black-Litterman posterior mean returns i macierz kowariancji.

        μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
        Σ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ + Σ

        Returns
        -------
        mu_bl    : (n,) posterior expected returns
        sigma_bl : (n, n) posterior covariance
        """
        if P.shape[0] == 0:
            # Brak views → posterior = prior
            return pi.copy(), sigma.copy()

        tau_sigma         = self.tau * sigma
        tau_sigma_inv     = np.linalg.inv(tau_sigma + np.eye(len(pi)) * 1e-8)
        omega_inv         = np.linalg.inv(omega + np.eye(len(Q)) * 1e-8)
        P_omega_inv_P     = P.T @ omega_inv @ P
        posterior_prec    = tau_sigma_inv + P_omega_inv_P
        try:
            posterior_cov_tau = np.linalg.inv(posterior_prec + np.eye(len(pi)) * 1e-8)
        except np.linalg.LinAlgError:
            posterior_cov_tau = np.linalg.pinv(posterior_prec)

        bracket  = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl    = posterior_cov_tau @ bracket
        # Full posterior covariance (He & Litterman formulation)
        sigma_bl = sigma + posterior_cov_tau

        return mu_bl, sigma_bl

    # ─── 4. PORTFOLIO OPTIMIZATION ────────────────────────────────────────────

    def optimize_portfolio(
        self,
        mu_bl: np.ndarray,
        sigma_bl: np.ndarray,
        risk_aversion: float | None = None,
        constraints: list[dict] | None = None,
        bounds: list[tuple] | None = None,
    ) -> np.ndarray:
        """
        Optymalizacja mean-variance z posterior B-L.

        Maksymalizuje: μ_BL^T w - (λ/2) w^T Σ_BL w

        Returns
        -------
        w_opt : (n,) optymalne wagi (suma = 1, wagi >= 0)
        """
        lam  = risk_aversion or self.risk_aversion
        n    = len(mu_bl)
        w0   = np.ones(n) / n

        def neg_utility(w):
            ret  = mu_bl @ w
            var  = w @ sigma_bl @ w
            return -(ret - 0.5 * lam * var)

        default_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        all_constraints     = (constraints or []) + default_constraints
        default_bounds      = [(0.0, 1.0)] * n
        all_bounds          = bounds or default_bounds

        res = minimize(
            neg_utility, w0,
            method="SLSQP",
            bounds=all_bounds,
            constraints=all_constraints,
            options={"ftol": 1e-8, "maxiter": 500},
        )

        w_opt = res.x / (np.sum(res.x) + 1e-12)
        return np.maximum(w_opt, 0.0)

    # ─── 5. FULL PIPELINE ─────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        returns_df: "pd.DataFrame",
        cio_thesis: dict,
        econ_analysis: dict,
        market_cap_weights: np.ndarray | None = None,
    ) -> dict:
        """
        Kompletny pipeline Black-Litterman od danych do wag portfela.

        Parameters
        ----------
        returns_df           : pd.DataFrame z dziennymi zwrotami aktywów
        cio_thesis           : wynik LocalCIO.synthesize_thesis()
        econ_analysis        : wynik LocalEconomist.analyze_macro()
        market_cap_weights   : wagi rynkowe; jeśli None → równe

        Returns
        -------
        dict z kluczami: pi, mu_bl, sigma_bl, weights_bl, weights_mkt,
                         views_count, regime, views_Q
        """
        asset_names = list(returns_df.columns)
        n           = len(asset_names)

        # Σ z danych historycznych (roczna)
        sigma = returns_df.cov().values * 252

        # Wagi rynkowe
        w_mkt = (
            market_cap_weights
            if market_cap_weights is not None
            else np.ones(n) / n
        )

        # Prior
        pi = self.compute_implied_returns(sigma, w_mkt)

        # Views z agentów
        P, Q, omega = self.build_views_from_agents_with_sigma(
            cio_thesis, econ_analysis, asset_names, sigma
        )

        # Posterior
        mu_bl, sigma_bl = self.posterior_returns(sigma, pi, P, Q, omega)

        # Optymalne wagi
        w_bl  = self.optimize_portfolio(mu_bl, sigma_bl)
        w_mkt_opt = self.optimize_portfolio(pi, sigma)

        return {
            "asset_names":    asset_names,
            "pi":             pi,
            "mu_bl":          mu_bl,
            "sigma":          sigma,
            "sigma_bl":       sigma_bl,
            "weights_bl":     w_bl,
            "weights_mkt":    w_mkt_opt,
            "views_count":    P.shape[0],
            "regime":         cio_thesis.get("regime", "neutral"),
            "views_Q":        Q.tolist() if len(Q) > 0 else [],
            "views_P":        P.tolist() if P.shape[0] > 0 else [],
        }
