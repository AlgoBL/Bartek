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

# ── Stałe ──────────────────────────────────────────────────────────────────────

SCORE_MAX = 100
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "heartbeat_cache.json")
# fall back ścieżka (w katalogu root projektu)
_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE_CANDIDATES = [
    os.path.join(_ROOT, "data", "heartbeat_cache.json"),
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


# ── Główna klasa ───────────────────────────────────────────────────────────────

class AdvisorEngine:
    """
    Silnik syntezy wniosków inwestycyjnych.

    Używa:
    - GlobalPortfolio (gs) — portfel użytkownika
    - macro dict — dane z Heartbeat Cache
    - geo dict — dane geopolityczne z cache
    """

    def __init__(self, gs=None, macro: dict = None, geo: dict = None):
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

        self._signals: List[AdvisorSignal] = []
        self._build_signals()

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
        safe_pct = gs.alloc_safe_pct
        risky_pct = gs.alloc_risky_pct
        n_risky = len(gs.risky_assets)
        portfolio_assets = gs.portfolio_assets if hasattr(gs, "portfolio_assets") and gs.portfolio_assets else []

        # Dywersyfikacja: więcej aktywów = lepiej (max 5+)
        div_score = min(n_risky / 5.0, 1.0) * 80 + 10
        self._signals.append(AdvisorSignal(
            "Liczba aktywów ryzykownych", float(n_risky), 4, 1, "higher_better",
            "ok" if n_risky >= 3 else ("warn" if n_risky == 2 else "alarm"), 0.08
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
        safe_protection = gs.alloc_safe_pct * 50
        score_protection = min(100, safe_protection + vix_protection + 10)

        # Growth: wysoka alokacja ryzykowna + dobry sentyment + dobry macro
        risky_potential = gs.alloc_risky_pct * 60
        macro_boost = (macro_score - 50) * 0.3
        score_growth = min(100, max(0, risky_potential + macro_boost + 20))

        # Risk: odwrotność macro score
        score_risk = min(100, max(0, 100 - macro_score + 5))

        # Overall: średnia ważona (protection * 0.4 + growth * 0.3 + (100-risk) * 0.3)
        score_overall = score_protection * 0.4 + score_growth * 0.3 + (100 - score_risk) * 0.3

        # Radar dimensions
        yc = self.macro.get("Yield_Curve_Spread", 0) or 0
        radar_safety  = min(100, gs.alloc_safe_pct * 80 + (20 if yc >= 0 else 0))
        radar_growth  = min(100, gs.alloc_risky_pct * 70 + max(0, macro_score - 50))
        radar_liquid  = 75.0  # zakładamy dobrą płynność ETF
        n_risky = len(gs.risky_assets)
        radar_div     = min(100, 15 + n_risky * 17)
        ry = self.macro.get("FRED_Real_Yield_10Y")
        radar_infl    = 70 - (ry * 15 if ry and ry > 0 else 0) if ry is not None else 55
        fx_enabled    = getattr(gs, "currency_risk_enabled", False)
        radar_fx      = 60 - (20 if fx_enabled else 0) + (gs.alloc_safe_pct * 30)

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
            actions.append(AdvisorAction(1, "Ryzyko", "🚨", "ALARM: Panika rynkowa (VIX > 35)",
                f"VIX wynosi {vix:.1f} — poziom krachu. Rozważ natychmiastowe zmniejszenie ekspozycji "
                f"na część ryzykowną o 10–20%. Kup opcje ochronne lub zwiększ alokację TOS.",
                0.92, 0))

        if yc is not None and yc < -0.5:
            actions.append(AdvisorAction(1, "Makro", "📉", "Silna inwersja krzywej rentowności",
                f"Spread 10Y-3M wynosi {yc:+.2f}% — historycznie poprzedza recesję o 12–18 mies. "
                f"Zwiększ część bezpieczną (obligacje długoterminowe). Horyzont krytyczny: ~12 mies.",
                0.85, 12))

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
                f"(np. QQQ, GLD, EEM) lub sektory, by obniżyć koncentrację o ~30%.",
                0.90, 1))

        if gs.alloc_safe_pct < 0.50 and horizon_months <= 12:
            actions.append(AdvisorAction(2, "Alokacja", "🔒", "Za mała ochrona dla krótkoterminowego horyzontu",
                f"Alokacja bezpieczna wynosi {gs.alloc_safe_pct:.0%}. Przy horyzoncie {horizon_months} mies. "
                f"zalecane minimum to 50–65%. Zwiększ pozycję w TOS lub obligacjach.",
                0.82, 0))

        if ry is not None and ry > 2.0 and gs.alloc_safe_pct < 0.70:
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
        if horizon_months >= 24 and gs.alloc_risky_pct < 0.20:
            actions.append(AdvisorAction(3, "Alokacja", "📈", "Zwiększ ekspozycję dla długiego horyzontu",
                f"Przy horyzoncie {horizon_months} mies. niska alokacja ryzykowna ({gs.alloc_risky_pct:.0%}) "
                f"może skutkować stopami poniżej inflacji. Rozważ stopniowe zwiększanie do 20–30%.",
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

        if not actions:
            actions.append(AdvisorAction(3, "Portfel", "✅", "Portfel w dobrej kondycji",
                "Na podstawie dostępnych danych makroekonomicznych i struktury portfela "
                "nie zidentyfikowano pilnych kwestii do działania. Kontynuuj regularny rebalancing.",
                0.75, 0))

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
        n = len(gs.risky_assets)
        safe_pct = gs.alloc_safe_pct

        lines = [
            f"Profil: **{gs.profile_name}** | Kapitał: **{gs.initial_capital:,.0f} PLN**",
            f"Alokacja: **{safe_pct:.0%} bezpieczna** / **{gs.alloc_risky_pct:.0%} ryzykowna**",
        ]
        if gs.risky_assets:
            top3 = ", ".join(f"{a['ticker']} ({a['weight']:.0f}%)" for a in gs.risky_assets[:3])
            if len(gs.risky_assets) > 3:
                top3 += f" +{len(gs.risky_assets)-3} więcej"
            lines.append(f"Aktywa ryzykowne: **{top3}**")
        if n == 1:
            lines.append("⚠️ Portfel jednoskładnikowy — brak dywersyfikacji wewnątrz części ryzykownej")
        if safe_pct >= 0.85:
            lines.append("🛡️ Silna ochrona kapitału — Barbell Strategy dobrze skalibrowana")
        return "  \n".join(lines)

    # ── Timeline ───────────────────────────────────────────────────────────────

    def _build_timeline(self, horizon_months: int, scores: Dict[str, float]) -> Tuple[List[str], List[float]]:
        """Buduje oś czasu z prognozowanymi wartościami portfela."""
        import math
        gs = self.gs
        cap = gs.initial_capital
        safe_rate = gs.safe_rate
        risky_nominal = 0.08  # zakładamy średnią dla aktywów ryzykownych

        # Stopa miesięczna
        monthly_safe  = (1 + safe_rate) ** (1/12) - 1
        monthly_risky = (1 + risky_nominal) ** (1/12) - 1

        # Modyfikator makro (VIX, YC)
        vix = self.macro.get("VIX_1M", 20) or 20
        macro_mod = 1.0 - max(0, (vix - 20) / 100)

        labels, values = [], []
        current = cap

        for m in range(1, horizon_months + 1):
            safe_return  = current * gs.alloc_safe_pct * monthly_safe
            risky_return = current * gs.alloc_risky_pct * monthly_risky * macro_mod
            current += safe_return + risky_return

            if m == 1 or m % max(1, horizon_months // 10) == 0 or m == horizon_months:
                if m <= 12:
                    labels.append(f"M{m}")
                else:
                    labels.append(f"Y{m//12}" if m % 12 == 0 else f"M{m}")
                values.append(round(current, 0))

        return labels, values

    # ── Główna metoda ─────────────────────────────────────────────────────────

    def generate_report(self, horizon_months: int = 12) -> AdvisorReport:
        """Generuje kompletny raport doradczy."""
        scores = self.compute_scores()
        actions = self.generate_actions(horizon_months)
        labels, values = self._build_timeline(horizon_months, scores)

        alerts = []
        for sig in self._signals:
            if sig.current_state == "alarm":
                alerts.append(f"🔴 {sig.name}: wartość krytyczna")
            elif sig.current_state == "warn":
                alerts.append(f"🟡 {sig.name}: podwyższona czujność")

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
            timeline_labels=labels,
            timeline_values=values,
        )
        return report
