"""
modules/advisor_engine.py
=========================
Silnik Doradcy Inwestycyjnego — zbiera sygnały z:
  - GlobalPortfolio (portfel użytkownika z globalnych ustawień)
  - Heartbeat Cache (dane makro: VIX, YC, TED, GEX, HY, itp.)
  - Reguły heurystyczne → syntetyczne rekomendacje

Nie uruchamia żadnych modułów analitycznych na nowo.
Używa gotowych danych dostępnych w cache + session_state.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from modules.emerytura import run_mc_retirement, retirement_readiness_score, cape_adjusted_swr, compute_survival_curve


# ── Stałe ──────────────────────────────────────────────────────────────────────

SCORE_MAX = 100
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "market_cache.json")
# fall back ścieżka (w katalogu root projektu)
_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE_CANDIDATES = [
    os.path.join(_ROOT, "data", "market_cache.json"),
    os.path.join(_ROOT, "heartbeat_cache.json"),
]


def _load_cache() -> Tuple[dict, dict]:
    """Wczytuje makro i geo z pliku cache Heartbeat Engine."""
    for path in _CACHE_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    packet = json.load(f)
                if packet.get("status") == "success":
                    return packet.get("macro", {}), packet.get("geo_report", {})
            except Exception:
                pass
    return {}, {}


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AdvisorAction:
    """Jedna konkretna rekomendacja dla użytkownika."""
    priority: int          # 1 = krytyczne, 2 = ważne, 3 = optymalizacja
    category: str          # "Ryzyko" / "Alokacja" / "Dywersyfikacja" / "Timing" / "Portfel"
    icon: str
    title: str
    description: str
    confidence: float      # 0.0–1.0
    horizon_months: int    # od kiedy ma znaczenie
    # XAI — Explainability (CFA Institute 2025)
    evidence: list = field(default_factory=list)          # dane ktore uruchomily regule
    explanation_chain: str = ""                           # "Dlaczego?" w 1 zdaniu
    related_signals: list = field(default_factory=list)   # powiazane sygnaly makro


@dataclass
class AdvisorSignal:
    """Sygnał z jednego wskaźnika makro lub portfelowego."""
    name: str
    value: Optional[float]
    threshold_ok: float
    threshold_warn: float
    direction: str         # "lower_better" | "higher_better"
    current_state: str     # "ok" | "warn" | "alarm"
    weight: float          # waga w ogólnym score (suma = 1.0)

    def score(self) -> float:
        """Zwraca znormalizowany score 0–100 dla tego sygnału."""
        if self.value is None:
            return 50.0
        v = self.value
        lo, hi = self.threshold_ok, self.threshold_warn
        if self.direction == "lower_better":
            if v <= lo:   return 90.0
            if v >= hi:   return 10.0
            return 90.0 - 80.0 * (v - lo) / max(hi - lo, 1e-9)
        else:  # higher_better
            if v >= lo:   return 90.0
            if v <= hi:   return 10.0
            return 10.0 + 80.0 * (v - hi) / max(lo - hi, 1e-9)


@dataclass
class AdvisorReport:
    """Kompletny raport doradczy."""
    horizon_months: int
    profile_name: str

    # Scores 0–100
    score_protection: float = 50.0   # Ochrona kapitału
    score_growth: float = 50.0       # Potencjał wzrostu
    score_risk: float = 50.0         # Ryzyko (wyższy = większe ryzyko)
    score_overall: float = 50.0      # Ocena ogólna

    # Wymiary radaru (0–100)
    radar_safety: float = 50.0
    radar_growth: float = 50.0
    radar_liquidity: float = 70.0
    radar_diversification: float = 50.0
    radar_inflation: float = 50.0
    radar_currency: float = 50.0

    # Listy
    actions: List[AdvisorAction] = field(default_factory=list)
    signals: List[AdvisorSignal] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

    # Tekstowe podsumowania
    headline: str = ""
    market_context: str = ""
    portfolio_assessment: str = ""

    # Dane surowe do wykresów
    timeline_labels: List[str] = field(default_factory=list)
    timeline_values: List[float] = field(default_factory=list)

    # Wkłady poszczególnych sygnałów do score (waterfall)
    score_contributions: List[Dict] = field(default_factory=list)

    # Sparkline data per-asset (ticker -> list of close prices)
    sparkline_data: Dict[str, List[float]] = field(default_factory=dict)

    # Dane emerytalne FIRE (Inteligentny moduł)
    ret_readiness_score: float = 0.0
    ret_success_prob: float = 0.0
    ret_adjusted_swr: float = 0.0
    ret_years_to_fire: float = 0.0
    ret_assessment: str = ""
    ret_timeline_labels: List[str] = field(default_factory=list)
    ret_timeline_values: List[float] = field(default_factory=list)


# ── Główna klasa ───────────────────────────────────────────────────────────────

class AdvisorEngine:
    """
    Silnik syntezy wniosków inwestycyjnych.

    Używa:
    - GlobalPortfolio (gs) — portfel użytkownika
    - macro dict — dane z Heartbeat Cache
    - geo dict — dane geopolityczne z cache
    """

    def __init__(self, gs=None, macro: dict = None, geo: dict = None, 
                 sim_safe_pct: float = None, sim_risky_pct: float = None):
        if gs is None:
            from modules.global_settings import get_gs
            gs = get_gs()
        self.gs = gs

        if macro is None or geo is None:
            _m, _g = _load_cache()
            self.macro = macro if macro is not None else _m
            self.geo   = geo   if geo   is not None else _g
        else:
            self.macro = macro
            self.geo   = geo

        # Fazy 1: What-If (Symulowane wagi)
        self.safe_pct = sim_safe_pct if sim_safe_pct is not None else gs.alloc_safe_pct
        self.risky_pct = sim_risky_pct if sim_risky_pct is not None else gs.alloc_risky_pct

        self._signals: List[AdvisorSignal] = []

        # Obliczenia korelacji
        self.avg_correlation = 0.0
        self._compute_correlation()

        self.ret_metrics = self._compute_retirement_metrics()
        self._score_ci: tuple = (50.0, 50.0)  # bootstrap CI placeholder

        self._build_signals()

    # ── Logika Fazy 2: Korelacje ──────────────────────────────────────────────
    def _compute_correlation(self):
        """Pobiera historię 1M/1Y dla części ryzykownej i wylicza średnią korelację."""
        gs = self.gs
        tickers = [a["ticker"] for a in getattr(gs, "risky_assets", [])]
        if len(tickers) < 2:
            self.avg_correlation = 0.0
            return

        from modules.data_provider import fetch_data
        import pandas as pd
        import numpy as np
        
        try:
            data = fetch_data(tickers, period="1y", auto_adjust=True)
            if data.empty: return
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    prices = data['Close']
                else: return
            else:
                prices = data

            returns = prices.pct_change().dropna()
            corr_matrix = returns.corr()

            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            valid_vals = corr_matrix.values[mask]
            
            self.avg_correlation = np.nanmean(valid_vals) if len(valid_vals) > 0 else 0.0

        except Exception as e:
            from modules.logger import setup_logger
            log = setup_logger(__name__)
            log.warning(f"Błąd wyliczania korelacji w AdvisorEngine: {e}")
            self.avg_correlation = 0.0

    # ── Metryki Emerytalne (FIRE) ─────────────────────────────────────────────
    def _compute_retirement_metrics(self) -> dict:
        """Kalkulacje oparte na emerytura.py (lite MC 500 sims)."""
        gs = self.gs

        init_cap = gs.initial_capital
        annual_expenses = gs.ret_monthly_expense * 12
        annual_contrib = gs.ret_monthly_contribution * 12

        vix = self.macro.get("VIX_1M", 20) or 20
        base_vol = np.clip(vix / 100.0, 0.08, 0.8)
        macro_score = self._compute_macro_score()
        base_return = 0.08 + (macro_score - 50) * 0.001

        vol = base_vol * self.risky_pct + 0.05 * self.safe_pct
        ret = base_return * self.risky_pct + gs.safe_rate * self.safe_pct

        # Oblicz CAPE adjusted SWR
        cape = self.macro.get("CAPE", 30.0) # zakładamy 30 jeśli brak
        adj_swr = cape_adjusted_swr(cape, base_swr=gs.ret_swr_rate)

        years_to_retire = max(0, gs.ret_target_age - gs.ret_current_age)
        horizon = min(max(years_to_retire + 30, 40), 60) # stały horyzont ok 40-60 lat

        try:
            wealth_matrix, _ = run_mc_retirement(
                init_cap=init_cap, annual_expenses=annual_expenses, annual_contrib=annual_contrib,
                ret_return=ret, ret_vol=vol, horizon=horizon, n_sims=500,
                years_to_retirement=years_to_retire, inflation_base=gs.ret_inflation_rate,
                stochastic_inflation=gs.ret_stochastic_inflation, enable_contributions=gs.ret_enable_contributions,
                contrib_during_retirement=gs.ret_contrib_during_retirement, 
                withdrawal_strategy=gs.ret_withdrawal_strategy.split(" ")[0],
                use_glide_path=gs.ret_use_glide_path, tax_regime=gs.ret_tax_regime,
                use_spending_smile=gs.ret_use_spending_smile,
                zus_monthly=gs.ret_zus_monthly, floor_amount=gs.ret_floor_amount,
                medical_inflation_rate=gs.ret_medical_inflation
            )
            survival = compute_survival_curve(wealth_matrix)
            success_prob = float(survival[-1]) if len(survival) > 0 else 0.0
            p50_wealth = np.percentile(wealth_matrix, 50, axis=0)
        except Exception:
            success_prob = 0.5
            p50_wealth = np.zeros(horizon+1)

        zus_annual = 2500 * 12 # typowy ZUS
        zus_cov = min(1.0, zus_annual / (annual_expenses if annual_expenses>0 else 1.0))

        cur_swr = annual_expenses / (init_cap if init_cap > 0 else 1.0)
        fire_number = (annual_expenses - zus_annual) / adj_swr if (annual_expenses - zus_annual) > 0 else 0
        cap_vs_fire = init_cap / fire_number if fire_number > 0 else 1.0

        years_to_fire = (fire_number - init_cap) / (annual_contrib + init_cap*ret) if (annual_contrib + init_cap*ret) > 0 else 99
        years_to_fire = float(np.clip(years_to_fire, 0, 99))

        readiness = retirement_readiness_score(success_prob, cur_swr, zus_cov, years_to_fire, cap_vs_fire)

        tl_labels = [f"Wiek {gs.ret_current_age + i}" for i in range(0, horizon+1, 5)]
        tl_values = [p50_wealth[i] for i in range(0, horizon+1, 5)]

        return {
            "readiness": readiness,
            "success_prob": success_prob,
            "adj_swr": adj_swr,
            "years_to_fire": years_to_fire,
            "tl_labels": tl_labels,
            "tl_values": tl_values,
        }


    # ── Budowanie sygnałów ────────────────────────────────────────────────────

    def _build_signals(self):
        m = self.macro
        gs = self.gs

        def safe(key, default=None):
            v = m.get(key, default)
            return float(v) if v is not None else None

        # --- Makro sygnały ---
        self._signals = [
            AdvisorSignal("VIX", safe("VIX_1M"), 18, 30, "lower_better",
                          self._state(safe("VIX_1M"), 18, 30, "lower_better"), 0.15),
            AdvisorSignal("Yield Curve (10Y-3M)", safe("Yield_Curve_Spread"), 0.3, -0.3, "higher_better",
                          self._state(safe("Yield_Curve_Spread"), 0.3, -0.3, "higher_better"), 0.12),
            AdvisorSignal("HY Spread", safe("FRED_HY_Spread"), 350, 550, "lower_better",
                          self._state(safe("FRED_HY_Spread"), 350, 550, "lower_better"), 0.10),
            AdvisorSignal("TED Spread", safe("FRED_TED_Spread"), 0.3, 0.6, "lower_better",
                          self._state(safe("FRED_TED_Spread"), 0.3, 0.6, "lower_better"), 0.08),
            AdvisorSignal("Financial Stress (FCI)", safe("FRED_Financial_Stress_Index"), -0.5, 1.5, "lower_better",
                          self._state(safe("FRED_Financial_Stress_Index"), -0.5, 1.5, "lower_better"), 0.10),
            AdvisorSignal("GEX (Gamma Exposure)", safe("total_gex_billions"), 0, -3, "higher_better",
                          self._state(safe("total_gex_billions"), 0, -3, "higher_better"), 0.07),
            AdvisorSignal("Breadth Momentum", safe("Breadth_Momentum"), 0.005, -0.02, "higher_better",
                          self._state(safe("Breadth_Momentum"), 0.005, -0.02, "higher_better"), 0.06),
            AdvisorSignal("Crypto Fear&Greed", safe("Crypto_FearGreed"), 65, 20, "lower_better",
                          self._state(safe("Crypto_FearGreed"), 65, 20, "lower_better"), 0.04),
        ]

        # --- Sygnały portfelowe ---
        safe_pct = self.safe_pct
        risky_pct = self.risky_pct
        n_risky = len(gs.risky_assets)

        div_score = min(n_risky / 5.0, 1.0) * 80 + 10
        self._signals.append(AdvisorSignal(
            "Liczba i niska korelacja aktywów", float(n_risky) * (1 - self.avg_correlation*0.5), 4, 1.5, "higher_better",
            "ok" if n_risky >= 3 and self.avg_correlation < 0.7 else ("alarm" if n_risky < 2 or self.avg_correlation > 0.85 else "warn"), 0.08
        ))

        # Ekspozycja na ryzyko
        self._signals.append(AdvisorSignal(
            "Alokacja Bezpieczna", safe_pct * 100, 70, 40, "higher_better",
            "ok" if safe_pct >= 0.70 else ("warn" if safe_pct >= 0.40 else "alarm"), 0.10
        ))

        # Sentyment
        sent = self.geo.get("compound_sentiment", 0.0)
        self._signals.append(AdvisorSignal(
            "Sentyment News (NLP)", sent, 0.1, -0.2, "higher_better",
            "ok" if sent > 0.05 else ("warn" if sent > -0.15 else "alarm"), 0.06
        ))

        self._signals.append(AdvisorSignal(
            "M2 Money Supply YoY", safe("FRED_M2_YoY_Growth"), 2, -2, "higher_better",
            self._state(safe("FRED_M2_YoY_Growth"), 2, -2, "higher_better"), 0.04
        ))

        # --- SORR Signal: Sequence of Returns Risk (Pfau & Kitces 2024) ---
        years_to_ret = max(0, getattr(gs, 'ret_target_age', 65) - getattr(gs, 'ret_current_age', 45))
        sorr_sensitivity = float(max(0.0, 1.0 - years_to_ret / 10.0))  # 0=daleko, 1=juz na emeryturze
        self._signals.append(AdvisorSignal(
            "SORR — Ryzyko Sekwencji Zwrotow", sorr_sensitivity, 0.30, 0.70, "lower_better",
            self._state(sorr_sensitivity, 0.30, 0.70, "lower_better"), 0.08
        ))

        # --- Dodatkowe sygnały makro ---
        unemployment = safe("FRED_Unemployment_Rate")
        if unemployment is not None:
            self._signals.append(AdvisorSignal(
                "Bezrobocie (USA)", unemployment, 5.0, 7.5, "lower_better",
                self._state(unemployment, 5.0, 7.5, "lower_better"), 0.04
            ))

        pce = safe("FRED_PCE_YoY")
        if pce is not None:
            self._signals.append(AdvisorSignal(
                "PCE Inflacja YoY", pce, 2.5, 4.5, "lower_better",
                self._state(pce, 2.5, 4.5, "lower_better"), 0.04
            ))

    @staticmethod
    def _state(val, ok_thresh, warn_thresh, direction: str) -> str:
        if val is None:
            return "ok"
        if direction == "lower_better":
            if val <= ok_thresh:   return "ok"
            if val >= warn_thresh: return "alarm"
            return "warn"
        else:
            if val >= ok_thresh:   return "ok"
            if val <= warn_thresh: return "alarm"
            return "warn"

    # ── Obliczanie scores ──────────────────────────────────────────────────────

    def _compute_score_bootstrap_ci(self, n_bootstrap: int = 150) -> tuple:
        """Bootstrap ±zakres dla score_overall — transparentnosc modelu (XAI 2025)."""
        import numpy as np
        if not self._signals:
            return (50.0, 50.0)
        
        weights = np.array([s.weight for s in self._signals])
        base_scores = np.array([s.score() for s in self._signals])
        total_w = max(weights.sum(), 1e-9)
        
        noise_matrix = np.random.normal(0, 3.0, (n_bootstrap, len(base_scores)))
        perturbed = np.clip(base_scores + noise_matrix, 0, 100)
        macro_scores = np.clip(np.sum(perturbed * weights, axis=1) / total_w, 0, 100)
        
        q25 = float(np.percentile(macro_scores, 25))
        q75 = float(np.percentile(macro_scores, 75))
        return (round(q25, 1), round(q75, 1))

    def _compute_macro_score(self) -> float:
        """Ważona średnia score makro sygnałów."""
        total_w = sum(s.weight for s in self._signals)
        if total_w == 0:
            return 50.0
        return sum(s.score() * s.weight for s in self._signals) / total_w

    def compute_scores(self) -> Dict[str, float]:
        """Zwraca dict z wszystkimi scores 0–100."""
        gs = self.gs
        macro_score = self._compute_macro_score()

        # Protection: wysoka alokacja bezpieczna + niskie VIX = dobra ochrona
        vix = self.macro.get("VIX_1M", 20)
        vix_protection = max(0, 1 - (vix - 10) / 40) * 40 if vix else 20
        safe_protection = self.safe_pct * 50
        score_protection = min(100, safe_protection + vix_protection + 10)

        # Growth: wysoka alokacja ryzykowna + dobry sentyment + dobry macro
        risky_potential = self.risky_pct * 60
        macro_boost = (macro_score - 50) * 0.3
        score_growth = min(100, max(0, risky_potential + macro_boost + 20))

        # Risk: odwrotność macro score
        score_risk = min(100, max(0, 100 - macro_score + 5))

        # Overall: średnia ważona (protection * 0.4 + growth * 0.3 + (100-risk) * 0.3)
        score_overall = score_protection * 0.4 + score_growth * 0.3 + (100 - score_risk) * 0.3

        # Radar dimensions
        yc = self.macro.get("Yield_Curve_Spread", 0) or 0
        radar_safety  = min(100, self.safe_pct * 80 + (20 if yc >= 0 else 0))
        radar_growth  = min(100, self.risky_pct * 70 + max(0, macro_score - 50))
        radar_liquid  = 75.0  # zakładamy dobrą płynność ETF
        n_risky = len(gs.risky_assets)
        corr_pen = max(0, self.avg_correlation - 0.4) * 40
        radar_div     = min(100, max(0, 15 + n_risky * 20 - corr_pen))
        ry = self.macro.get("FRED_Real_Yield_10Y")
        radar_infl    = 70 - (ry * 15 if ry and ry > 0 else 0) if ry is not None else 55
        fx_enabled    = getattr(gs, "currency_risk_enabled", False)
        radar_fx      = 60 - (20 if fx_enabled else 0) + (self.safe_pct * 30)

        # Dynamiczny benchmark radaru — zalezy od profilu ryzyka (Konserwatywny/Agresywny)
        profile = getattr(self.gs, 'profile_name', '').lower()
        if 'konserw' in profile or 'bezpieczn' in profile:
            self.radar_benchmark = [70, 30, 80, 40, 65, 55]
        elif 'agresyw' in profile or 'dynamicz' in profile:
            self.radar_benchmark = [30, 80, 55, 60, 40, 40]
        else:  # zrownowazony / default
            self.radar_benchmark = [55, 55, 70, 55, 55, 50]

        return {
            "protection": round(score_protection, 1),
            "growth":     round(score_growth, 1),
            "risk":       round(score_risk, 1),
            "overall":    round(score_overall, 1),
            "radar_safety":         round(min(100, max(0, radar_safety)), 1),
            "radar_growth":         round(min(100, max(0, radar_growth)), 1),
            "radar_liquidity":      round(min(100, max(0, radar_liquid)), 1),
            "radar_diversification":round(min(100, max(0, radar_div)), 1),
            "radar_inflation":      round(min(100, max(0, radar_infl)), 1),
            "radar_currency":       round(min(100, max(0, radar_fx)), 1),
        }

    # ── Generowanie akcji ──────────────────────────────────────────────────────

    def generate_actions(self, horizon_months: int = 12) -> List[AdvisorAction]:
        """Generuje listę priorytetowanych rekomendacji."""
        gs = self.gs
        m = self.macro
        actions: List[AdvisorAction] = []

        vix = m.get("VIX_1M")
        yc  = m.get("Yield_Curve_Spread")
        hy  = m.get("FRED_HY_Spread")
        gex = m.get("total_gex_billions")
        fci = m.get("FRED_Financial_Stress_Index")
        ry  = m.get("FRED_Real_Yield_10Y")
        sent = self.geo.get("compound_sentiment", 0.0)
        n_risky = len(gs.risky_assets)

        # ── Priorytet 1 — Krytyczne alerty ────────────────────────────────────
        if vix and vix > 35:
            actions.append(AdvisorAction(
                1, "Ryzyko", "🚨", "ALARM: Panika rynkowa (VIX > 35)",
                f"VIX wynosi {vix:.1f} — poziom krachu. Rozważ natychmiastowe zmniejszenie ekspozycji "
                f"na część ryzykowną o 10–20%. Kup opcje ochronne lub zwiększ alokację TOS.",
                0.92, 0,
                evidence=[f"VIX = {vix:.1f} (próg alarmowy: 35)"],
                explanation_chain=f"VIX > 35 historycznie poprzedza spadki >20% w ciągu 3-6 mies. (CBOE dane 1990-2024).",
                related_signals=["VIX", "GEX (Gamma Exposure)", "HY Spread"]
            ))

        if yc is not None and yc < -0.5:
            actions.append(AdvisorAction(
                1, "Makro", "📉", "Silna inwersja krzywej rentowności",
                f"Spread 10Y-3M wynosi {yc:+.2f}% — historycznie poprzedza recesję o 12–18 mies. "
                f"Zwiększ część bezpieczną (obligacje długoterminowe). Horyzont krytyczny: ~12 mies.",
                0.85, 12,
                evidence=[f"Yield Curve Spread = {yc:+.2f}% (próg: -0.5%)"],
                explanation_chain="Inwersja 10Y-3M jest najlepszym predyktorem recesji (AUC ~0.85 wg NY Fed 2024).",
                related_signals=["Yield Curve (10Y-3M)", "HY Spread", "Financial Stress (FCI)"]
            ))

        if hy and hy > 600:
            actions.append(AdvisorAction(1, "Ryzyko", "💳", "Kryzys kredytowy (HY Spread > 600 bps)",
                f"Spread HY wynosi {hy:.0f} bps — poziom alarmu kredytowego. "
                f"Akcje spółek o niskim ratingu i krypto są szczególnie wrażliwe. Przejdź do jakości.",
                0.88, 0))

        if gex is not None and gex < -5:
            actions.append(AdvisorAction(1, "Ryzyko", "⚡", "Ekstremalny Short Gamma (GEX < -5B)",
                f"GEX wynosi {gex:.1f}B — dealerzy nie stabilizują rynku. "
                f"Spodziewaj się amplifikacji ruchów cenowych. Nie zwiększaj lewarowania.",
                0.80, 0))

        # ── Priorytet 2 — Ważne rekomendacje ──────────────────────────────────
        if n_risky <= 1:
            actions.append(AdvisorAction(2, "Dywersyfikacja", "🌐", "Niewystarczająca dywersyfikacja",
                f"Część ryzykowna zawiera tylko {n_risky} aktywo. Dodaj 2–4 nieskorelowane ETF "
                f"(np. QQQ, GLD, EEM) lub sektory, by obniżyć koncentrację.",
                0.90, 1))
        elif self.avg_correlation > 0.8:
            actions.append(AdvisorAction(2, "Dywersyfikacja", "🧲", "Krytyczne stężenie korelacji!",
                f"Średnia korelacja rynkowa wynosi ~{self.avg_correlation:.2f}. "
                f"Twój koszyk ryzykowny nie posiada prawdziwej dywersyfikacji. Zastosuj złoto lub surowce.",
                0.85, 1))

        if self.safe_pct < 0.50 and horizon_months <= 12:
            actions.append(AdvisorAction(2, "Alokacja", "🔒", "Za mała ochrona dla krótkoterminowego horyzontu",
                f"Alokacja bezpieczna (obecna/symulowana) to {self.safe_pct:.0%}. Przy horyzoncie {horizon_months} mies. "
                f"zalecane minimum to 50–65%.",
                0.82, 0))

        if ry is not None and ry > 2.0 and self.safe_pct < 0.70:
            actions.append(AdvisorAction(2, "Inflacja", "📊", "Wysokie realne stopy — presja na wyceny",
                f"Realny yield 10Y = {ry:.2f}%. Akcje (zwłaszcza growth) są wyceniane wyżej niż sugeruje "
                f"model DCF. Preferuj aktywa o niskim duration lub value ETF.",
                0.75, 3))

        if fci and fci > 1.0:
            actions.append(AdvisorAction(2, "Makro", "🏦", "Restrykcyjne warunki finansowe (FCI > 1)",
                f"STLFSI = {fci:.2f} — warunki finansowe zacieśnione. "
                f"Unikaj nowych pozycji spekulacyjnych przez najbliższe {min(6, horizon_months)} mies.",
                0.78, 0))

        if sent < -0.2:
            actions.append(AdvisorAction(2, "Sentyment", "😨", "Głęboko negatywny sentyment mediów",
                f"Sentyment NLP wynosi {sent:.2f}. Paradoks: ekstremalny pesymizm bywa kontrariańskim "
                f"sygnałem kupna dla długoterminowych inwestorów. Nie sprzedawaj w panice.",
                0.65, 0))

        # ── Priorytet 3 — Optymalizacje ───────────────────────────────────────
        if horizon_months >= 24 and self.risky_pct < 0.20:
            actions.append(AdvisorAction(3, "Alokacja", "📈", "Zwiększ ekspozycję dla długiego horyzontu",
                f"Przy horyzoncie {horizon_months} mies. niska alokacja ryzykowna ({self.risky_pct:.0%}) "
                f"może skutkować stratą na inflacji. Rozważ powiększenie do 20-30%.",
                0.70, 6))

        if n_risky > 0:
            tickers = [a["ticker"] for a in gs.risky_assets]
            if "GLD" not in tickers and "IAU" not in tickers:
                actions.append(AdvisorAction(3, "Dywersyfikacja", "🥇", "Brak złota w portfelu",
                    "Złoto jest naturalnym hedgem na recesję i osłabienie dolara. "
                    "Alokacja 3–7% w GLD lub IAU poprawia Sharpe Ratio portfela.",
                    0.68, 3))

            crypto = [a for a in gs.risky_assets if "BTC" in a["ticker"] or "ETH" in a["ticker"]]
            if crypto and gs.alloc_risky_pct > 0:
                crypto_wt = sum(a["weight"] for a in crypto) / max(sum(a["weight"] for a in gs.risky_assets), 1)
                if crypto_wt > 0.30:
                    actions.append(AdvisorAction(3, "Ryzyko", "₿", "Wysoka koncentracja krypto",
                        f"Krypto stanowi ~{crypto_wt:.0%} części ryzykownej. "
                        f"Przy horyzoncie {horizon_months} mies. rozważ redukcję do max 15–20%.",
                        0.72, 0))

        if gs.alloc_safe_pct >= 0.80 and horizon_months >= 36:
            actions.append(AdvisorAction(3, "Alokacja", "⚖️", "Portfel za konserwatywny w długim terminie",
                f"Alokacja {gs.alloc_safe_pct:.0%} bezpieczna jest bezpieczna krótkoterminowo, "
                f"ale przy horyzoncie {horizon_months} mies. inflacja realnie obniży wartość. "
                f"Rozważ strategię stopniowego (DCA) zwiększania ekspozycji na akcje.",
                0.65, 12))

        # ── Emerytura ─────────────────────────────────────────────────────────
        ret_r = self.ret_metrics.get("readiness", 50.0)
        if ret_r < 40:
            actions.append(AdvisorAction(1, "Emerytura", "🔥", "Krytycznie niska gotowość emerytalna",
                f"Twój wskaźnik Readiness to {ret_r}/100. Prawdopodobieństwo sukcesu wynosi tylko {self.ret_metrics.get('success_prob', 0):.0%}. "
                f"Zwiększ wpłaty miesięczne w module Ustawień lub przemyśl opóźnienie przejścia na emeryturę o kilka lat.",
                0.90, horizon_months=24))
        elif ret_r < 65:
            actions.append(AdvisorAction(2, "Emerytura", "⚠️", "Wymagana optymalizacja podatkowa SWR",
                f"SWR skorygowany o CAPE wynosi {self.ret_metrics.get('adj_swr', 0.04):.1%}. Twój plan posiada luki, a portfel może nie przetrwać całej emerytury. "
                f"Zaleca się zminimalizowanie narzutu podatkowego (wykorzystaj limity IKE/IKZE) oraz obniżenie prognozowanego poziomu konsumpcji.",
                0.78, horizon_months=12))

        # SORR action — Pfau 2024
        years_to_ret = max(0, getattr(gs, 'ret_target_age', 65) - getattr(gs, 'ret_current_age', 45))
        if years_to_ret <= 5 and self.risky_pct > 0.4:
            actions.append(AdvisorAction(
                2, "Emerytura", "⏳", f"SORR Alert: Emerytura za {years_to_ret} lat — zbyt duże ryzyko",
                f"Przy {years_to_ret} latach do emerytury alokacja ryzykowna {self.risky_pct:.0%} jest za wysoka. "
                f"Pfau (2024): krach w 1. roku emerytury trwale obniża dochód o 20-40%. Rozpocznij Glide Path.",
                0.88, 0,
                evidence=[f"Lata do emerytury: {years_to_ret}", f"Alokacja ryzykowna: {self.risky_pct:.0%}"],
                explanation_chain="Sequence of Returns Risk: straty wczesnej emerytury są nieodwracalne bo kapital jest najwyższy.",
                related_signals=["SORR — Ryzyko Sekwencji Zwrotow", "VIX"]
            ))

        if not actions:
            actions.append(AdvisorAction(
                3, "Portfel", "✅", "Portfel w dobrej kondycji",
                "Na podstawie dostępnych danych makroekonomicznych i struktury portfela "
                "nie zidentyfikowano pilnych kwestii do działania. Kontynuuj regularny rebalancing.",
                0.75, 0,
                explanation_chain="Wszystkie monitorowane sygnały makro i portfelowe w normie."
            ))

        # Sortuj: priorytet → confidence
        actions.sort(key=lambda a: (a.priority, -a.confidence))
        return actions

    # ── Tekstowe podsumowania ─────────────────────────────────────────────────

    def _generate_headline(self, scores: Dict[str, float]) -> str:
        overall = scores["overall"]
        vix = self.macro.get("VIX_1M")
        yc  = self.macro.get("Yield_Curve_Spread")

        if overall >= 70:
            return "✅ Środowisko inwestycyjne sprzyjające — portfel dobrze skalibrowany"
        if overall >= 50:
            if vix and vix > 25:
                return "⚡ Umiarkowane ryzyko — VIX podwyższony, zachowaj ostrożność"
            return "⚖️ Środowisko neutralne — utrzymuj obecną alokację z czujnością"
        if overall >= 30:
            return "⚠️ Zwiększone ryzyko systemowe — rekomendujemy redukcję ekspozycji"
        return "🚨 ALARM — warunki rynkowe ekstremalne, priorytet: ochrona kapitału"

    def _generate_market_context(self) -> str:
        m = self.macro
        vix = m.get("VIX_1M")
        yc  = m.get("Yield_Curve_Spread")
        hy  = m.get("FRED_HY_Spread")
        sent = self.geo.get("compound_sentiment", 0.0)

        parts = []
        if vix:
            lvl = "niski" if vix < 18 else ("umiarkowany" if vix < 25 else ("wysoki" if vix < 35 else "ekstremalny"))
            parts.append(f"**VIX {vix:.1f}** — strach {lvl}")
        if yc is not None:
            s = "normalna" if yc > 0.5 else ("płaska" if yc > 0 else "ODWRÓCONA ⚠️")
            parts.append(f"**Krzywa {yc:+.2f}%** — {s}")
        if hy:
            lvl = "spokojny" if hy < 350 else ("podwyższony" if hy < 550 else "alarmowy")
            parts.append(f"**HY {hy:.0f}bps** — kredyt {lvl}")
        if sent:
            s = "pozytywny" if sent > 0.1 else ("neutralny" if sent > -0.1 else "negatywny")
            parts.append(f"**Sentyment {sent:+.2f}** — narracja {s}")

        if not parts:
            return "Brak aktualnych danych makroekonomicznych. Uruchom Heartbeat Engine."
        return " | ".join(parts)

    def _generate_portfolio_assessment(self) -> str:
        gs = self.gs
        safe_pct = self.safe_pct

        lines = [
            f"Wariant kalkulacji: **Alokacja symulowana ({safe_pct:.0%} Bezp., {self.risky_pct:.0%} Ryzykowna)**" if safe_pct != gs.alloc_safe_pct else f"Profil obecny: **{gs.profile_name}**",
            f"Alokacja zadeklarowana bazowo: **{gs.alloc_safe_pct:.0%} bezpieczna** / **{gs.alloc_risky_pct:.0%} ryzykowna** | Kapitał: {gs.initial_capital:,.0f} PLN",
        ]
        if gs.risky_assets:
            top3 = ", ".join(f"{a['ticker']} ({a['weight']:.0f}%)" for a in gs.risky_assets[:3])
            if len(gs.risky_assets) > 3:
                top3 += f" +{len(gs.risky_assets)-3} więcej"
            lines.append(f"Aktywa ryzykowne: **{top3}**")
        if self.avg_correlation > 0.8:
            lines.append("⚠️ **Krytycznie niska faktyczna dywersyfikacja** – wysoka korelacja oznacza złudną zmienność")
        if safe_pct >= 0.85:
            lines.append("🛡️ Silna ochrona kapitału — Barbell Strategy dobrze skalibrowana")
        return "  \n".join(lines)

    # ── Kontrybucje do score (Waterfall) ─────────────────────────────────────

    def compute_score_contributions(self) -> List[Dict]:
        """Zwraca listę wkładów każdego sygnału do score_overall (dla waterfall chart)."""
        macro_score = self._compute_macro_score()
        contributions = []
        for sig in self._signals:
            sc = sig.score()        # 0–100
            delta = (sc - 50.0) * sig.weight  # Wkład: dodatni = pomaga, ujemny = szkodzi
            contributions.append({
                "name": sig.name,
                "score": sc,
                "weight": sig.weight,
                "delta": round(delta, 2),
                "state": sig.current_state,
                "category": "Makro" if sig.name in (
                    "VIX", "Yield Curve (10Y-3M)", "HY Spread", "TED Spread",
                    "Financial Stress (FCI)", "GEX (Gamma Exposure)",
                    "Breadth Momentum", "M2 Money Supply YoY",
                    "Bezrobocie (USA)", "PCE Inflacja YoY"
                ) else "Portfel",
            })
        contributions.sort(key=lambda x: x["delta"], reverse=True)
        return contributions

    # ── Sparklines 30D per aktywo ─────────────────────────────────────────────

    def fetch_sparklines(self, tickers: List[str]) -> Dict[str, List[float]]:
        """Pobiera 30-dniowe ceny zamknięcia dla listy tickerów (lazy, opcjonalne)."""
        if not tickers:
            return {}
        try:
            from modules.data_provider import fetch_data
            import pandas as pd
            data = fetch_data(tickers, period="1mo", auto_adjust=True)
            result: Dict[str, List[float]] = {}
            if data.empty:
                return result
            if isinstance(data.columns, pd.MultiIndex):
                prices = data.get("Close", data)
            else:
                prices = data
            for tkr in tickers:
                if tkr in prices.columns:
                    vals = prices[tkr].dropna().tolist()
                    result[tkr] = [round(v, 2) for v in vals[-30:]]
            return result
        except Exception:
            return {}

    # ── Funkcje Fazy 1 i 3: Rebalancing & Stochastics ─────────────────────────

    def generate_rebalancing_orders(self) -> List[dict]:
        """Kalkulator odchyleń portfela do docelowej alokacji (What-If)."""
        gs = self.gs
        cap = gs.initial_capital
        orders = []
        if cap <= 0: return orders

        # Cel z symulatora (lub domyślnie z GlobalPortfolio)
        target_safe_pln    = cap * self.safe_pct
        target_risky_pln   = cap * self.risky_pct

        # Obecna wartość z GlobalPortfolio (ta zadeklarowana bazowo)
        current_safe_pln = cap * gs.alloc_safe_pct

        diff_safe = target_safe_pln - current_safe_pln
        if abs(diff_safe) > cap * 0.01:
            orders.append({
                "asset": "TOS / Detaliczne (Bezpieczna)",
                "action": "KUP" if diff_safe > 0 else "SPRZEDAJ",
                "diff_pln": abs(diff_safe),
                "type": "SAFE"
            })
            
        if gs.risky_assets:
            total_w = sum([a["weight"] for a in gs.risky_assets])
            for a in gs.risky_assets:
                if total_w <= 0: continue
                target_pln = target_risky_pln * (a["weight"] / total_w)
                orders.append({
                    "asset": a.get("name", a["ticker"]),
                    "action": "DOCELOWA POZYCJA",
                    "diff_pln": target_pln,
                    "target_pct": (a["weight"]/total_w)*gs.alloc_risky_pct*100,
                    "type": "RISK"
                })

        orders.sort(key=lambda x: x["diff_pln"], reverse=True)
        return orders

    def _build_stochastic_timeline(self, horizon_months: int) -> Dict[str, List[float]]:
        """Symulacja Fan-Chart (Monte Carlo) portfela."""
        import numpy as np
        
        cap = self.gs.initial_capital
        monthly_safe = (1 + self.gs.safe_rate) ** (1/12) - 1
        vix = self.macro.get("VIX_1M", 20) or 20
        annual_vol = np.clip(vix / 100.0, 0.08, 0.8)
        monthly_vol = annual_vol / np.sqrt(12)
        
        expected_annual_ret = 0.09 + (self._compute_macro_score() - 50) * 0.001
        m_ret = expected_annual_ret / 12
        
        paths = 500
        horizon = max(horizon_months, 1)
        
        safe_arr = np.zeros(horizon+1)
        safe_arr[0] = cap * self.safe_pct
        for m in range(1, horizon+1): safe_arr[m] = safe_arr[m-1] * (1 + monthly_safe)
            
        risky_start = cap * self.risky_pct
        risky_sims = np.zeros((paths, horizon+1))
        risky_sims[:, 0] = risky_start
        
        for m in range(1, horizon+1):
            shocks = np.random.normal(m_ret - 0.5 * monthly_vol**2, monthly_vol, paths)
            risky_sims[:, m] = risky_sims[:, m-1] * np.exp(shocks)
            
        total_sims = risky_sims + safe_arr
        
        labels, indices = [], []
        for m in range(1, horizon+1):
            if m == 1 or m % max(1, horizon // 10) == 0 or m == horizon:
                labels.append(f"M{m}" if m <= 12 else (f"Y{m//12}" if m % 12 == 0 else f"M{m}"))
                indices.append(m)
                
        p10 = np.percentile(total_sims, 10, axis=0)
        p50 = np.percentile(total_sims, 50, axis=0)
        p90 = np.percentile(total_sims, 90, axis=0)
        
        return {
            "labels": labels,
            "p10": [p10[i] for i in indices],
            "p50": [p50[i] for i in indices],
            "p90": [p90[i] for i in indices],
            "conservative": [safe_arr[i] + (risky_start if risky_start > 0 else 0) for i in indices],
            "max_drawdown": self._estimate_dd(annual_vol)
        }
        
    def _estimate_dd(self, vol: float) -> str:
        p_vol = vol * self.risky_pct
        if p_vol < 0.05: return "< 5%"
        if p_vol < 0.10: return "~ 10-15%"
        if p_vol < 0.20: return "~ 15-25%"
        return "> 30% (KRYTYCZNE)"

    def generate_report(self, horizon_months: int = 12, fetch_sparklines: bool = True) -> AdvisorReport:
        """Generuje kompletny raport doradczy."""
        scores = self.compute_scores()
        actions = self.generate_actions(horizon_months)
        timeline_data = self._build_stochastic_timeline(horizon_months)
        self._score_ci = self._compute_score_bootstrap_ci()
        contributions = self.compute_score_contributions()

        alerts = []
        for sig in self._signals:
            if sig.current_state == "alarm":
                alerts.append(f"🔴 {sig.name}: wartość krytyczna ({sig.value:.2f})" if sig.value is not None else f"🔴 {sig.name}: wartość krytyczna")
            elif sig.current_state == "warn":
                alerts.append(f"🟡 {sig.name}: podwyższona czujność ({sig.value:.2f})" if sig.value is not None else f"🟡 {sig.name}: podwyższona czujność")

        # Sparklines (opcjonalne — nie blokuje raportu przy błędzie)
        sparklines: Dict[str, List[float]] = {}
        if fetch_sparklines:
            risky_tickers = [a["ticker"] for a in getattr(self.gs, "risky_assets", [])]
            if risky_tickers:
                sparklines = self.fetch_sparklines(risky_tickers[:6])  # max 6 tickerów

        report = AdvisorReport(
            horizon_months=horizon_months,
            profile_name=self.gs.profile_name,
            score_protection=scores["protection"],
            score_growth=scores["growth"],
            score_risk=scores["risk"],
            score_overall=scores["overall"],
            radar_safety=scores["radar_safety"],
            radar_growth=scores["radar_growth"],
            radar_liquidity=scores["radar_liquidity"],
            radar_diversification=scores["radar_diversification"],
            radar_inflation=scores["radar_inflation"],
            radar_currency=scores["radar_currency"],
            actions=actions,
            signals=self._signals,
            alerts=alerts,
            headline=self._generate_headline(scores),
            market_context=self._generate_market_context(),
            portfolio_assessment=self._generate_portfolio_assessment(),
            timeline_labels=timeline_data["labels"],
            timeline_values=timeline_data["p50"],
            score_contributions=contributions,
            sparkline_data=sparklines,
            ret_readiness_score=self.ret_metrics.get("readiness", 50.0),
            ret_success_prob=self.ret_metrics.get("success_prob", 0.0),
            ret_adjusted_swr=self.ret_metrics.get("adj_swr", 0.04),
            ret_years_to_fire=self.ret_metrics.get("years_to_fire", 0.0),
            ret_timeline_labels=self.ret_metrics.get("tl_labels", []),
            ret_timeline_values=self.ret_metrics.get("tl_values", []),
            ret_assessment=f"Zaktualizowany bezpieczny poziom wypłat (SWR - uwzględniający wyceny CAPE) wynosi **{self.ret_metrics.get('adj_swr', 0.04):.2%}**. Szansa na przetrwanie kapitału przez całą emeryturę (w wariancie bazowym) to **{self.ret_metrics.get('success_prob', 0)*100:.0f}%**."
        )
        report.timeline_full_data = timeline_data
        report.rebalancing_orders = self.generate_rebalancing_orders()
        return report
