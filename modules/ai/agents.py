
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class LocalEconomist:
    """
    Ekonomista Makro v2.0 — Wielowymiarowy Nowcast Ryzyka.
    
    Zamiast prostych reguł IF/ELSE, oblicza ważony Z-Score z 7 zmiennych makro.
    Metodologia: Ang & Bekaert (2002), Hamilton (1989), OECD CLI.
    
    Misja Barbellowa: Identyfikuje fazy cyklu koniunkturalnego, by wskazać
    Skaner w stronę OCHRONY (Safe Sleeve) lub ATAKU na Wypukłość (Risky Sleeve).
    """

    # Wagi czynników makro (suma = 1.0), oparte na literaturze empirycznej
    _FACTOR_WEIGHTS = {
        "yield_curve":       0.25,  # Hamilton (1989) — najsilniejszy predyktor recesji
        "vix_level":         0.20,  # Whaley (2009) — Investor Fear Gauge
        "vix_term_structure":0.15,  # Carr & Wu (2006) — contango = spokój, backwardation = panika
        "dxy_strength":      0.10,  # Luchtenberg & Vu (2015) — silny USD = risk-off EM
        "credit_spread":     0.15,  # Fama (1986) — spread korp. = ryzyko kredytowe systemu
        "cu_au_ratio":       0.10,  # Czerwona miedź vs. Złoto — barometr globalnego wzrostu
        "copper_trend":      0.05,  # Momentum miedzi = leading indicator PKB (12M)
    }

    def analyze_macro(self, oracle_snapshot: dict) -> dict:
        signals = {}
        details = []

        # 1. Yield Curve (Hamilton 1989) ─────────────────────────────────
        spread = oracle_snapshot.get("Yield_Curve_Spread", 0.5)
        if spread is None: spread = 0.5
        # Normalizujemy: negatywny spread → ryzyko 1.0, spread > 2.0 → brak ryzyka 0.0
        signals["yield_curve"] = float(np.clip(-spread / 2.0 + 0.5, 0, 1))
        if spread < 0:
            details.append(f"KRYTYCZNE ⛔: Inwersja Krzywej ({spread:.2f}%). Historycznie recesja w ciągu 12-18M.")
        elif spread < 0.5:
            details.append(f"OSTRZEŻENIE ⚠️: Płaska Krzywa ({spread:.2f}%). Trudne warunki kredytowe.")
        else:
            details.append(f"ZDROWE ✅: Stroma Krzywa (+{spread:.2f}%). Ekspansja gospodarcza.")

        # 2. VIX Poziom (Whaley 2009) ────────────────────────────────────
        vix = oracle_snapshot.get("VIX_1M") or oracle_snapshot.get("VIX_Volatility") or 15.0
        # VIX 10 = bardzo spokój, VIX 45 = kryzys (normalizacja 10–50)
        signals["vix_level"] = float(np.clip((vix - 10) / 40, 0, 1))
        if vix > 35:
            details.append(f"PANIKA 🔴 (VIX={vix:.0f}): Historyczny krach. Idealne środowisko dla Safe Sleeve.")
        elif vix > 20:
            details.append(f"PODWYŻSZONE RYZYKO 🟡 (VIX={vix:.0f}): Rynkowe turbulencje.")
        else:
            details.append(f"SPOKÓJ 🟢 (VIX={vix:.0f}): Risk-On. Szukamy asymetrycznych aktywów.")

        # 3. VIX Term Structure ────────────────────────────────────────────
        ts_ratio = oracle_snapshot.get("VIX_TS_Ratio")
        backwardation = oracle_snapshot.get("VIX_Backwardation", False)
        if ts_ratio is not None:
            # Backwardation (spot > futures) = krótkoterminowy strach strukturalny
            signals["vix_term_structure"] = float(np.clip((ts_ratio - 0.8) / 0.4, 0, 1))
            if backwardation:
                details.append(f"VIX BACKWARDATION 🔴 (ratio={ts_ratio:.2f}): Panika krótkoterminowa. Typowe przy nagłych krachach.")
            else:
                details.append(f"VIX Contango 🟢 (ratio={ts_ratio:.2f}): Rynek spokojny. Stale rosnąca zmienność długoterm.")
        else:
            signals["vix_term_structure"] = 0.3  # Brak danych = neutralny

        # 4. DXY (Dollar Strength) ────────────────────────────────────────
        dxy = oracle_snapshot.get("US_Dollar_Index") or 100.0
        signals["dxy_strength"] = float(np.clip((dxy - 95) / 20, 0, 1))
        if dxy > 108:
            details.append(f"SILNY DOLAR 🔴 (DXY={dxy:.0f}): Zaciskanie płynności — szkodzi Krypto, EM, Surowcom.")
        elif dxy < 98:
            details.append(f"SŁABY DOLAR 🟢 (DXY={dxy:.0f}): Luźna płynność — sprzyja aktywom ryzykownym.")
        else:
            details.append(f"NEUTRALNY DOLAR 🟡 (DXY={dxy:.0f})")

        # 5. Credit Spread (FRED BAA10Y) ──────────────────────────────────
        credit_spread = oracle_snapshot.get("FRED_Credit_Spread_BAA_AAA")
        if credit_spread is not None and credit_spread > 0:
            # Historycznie: < 1.5% = spokój, > 4% = kryzys (Fama 1986)
            signals["credit_spread"] = float(np.clip((credit_spread - 1.5) / 3.5, 0, 1))
            if credit_spread > 3.5:
                details.append(f"KRYZYS KREDYTOWY ⛔ (spread={credit_spread:.2f}%): Rynek obligacji korporacyjnych w strachu.")
            elif credit_spread > 2.0:
                details.append(f"STRES KREDYTOWY 🟡 (spread={credit_spread:.2f}%): Ostrożność.")
            else:
                details.append(f"ZDROWY DŁUG 🟢 (spread={credit_spread:.2f}%): Niskie ryzyko defaultu.")
        else:
            signals["credit_spread"] = 0.2  # Brak FRED → neutralny

        # 6. Copper/Gold Ratio (barometr wzrostu) ─────────────────────────
        cu_au = oracle_snapshot.get("CuAu_Ratio")
        if cu_au is not None:
            # Rosnący Cu/Au = ryzyko ON (miedź > złoto = wzrost gospodarczy dominuje)
            # Typ. zakres: 0.15–0.40 dla Cu/Au (oz na oz)
            signals["cu_au_ratio"] = float(np.clip(1.0 - (cu_au - 0.1) / 0.3, 0, 1))  # Niski = dobrze
        else:
            signals["cu_au_ratio"] = 0.3

        # 7. Copper Trend (momentum) ───────────────────────────────────────
        copper = oracle_snapshot.get("Copper")
        if copper:
            # Prosta heurystyka: jeśli cena Cu > 4.0 USD/lb → boom przemysłowy
            signals["copper_trend"] = float(np.clip(1.0 - (copper - 3.0) / 2.0, 0, 1))
        else:
            signals["copper_trend"] = 0.3

        # 8. Composite Nowcast Z-Score ─────────────────────────────────────
        composite_risk = sum(
            signals[key] * self._FACTOR_WEIGHTS[key]
            for key in self._FACTOR_WEIGHTS
            if key in signals
        )
        # Skalujemy do 0–8 dla zachowania kompatybilności
        risk_score = float(composite_risk * 8.0)

        # 9. Faza cyklu na podstawie composite score
        if risk_score >= 5.5:
            phase = "OBRONA (Ryzyko Recesji / Panika Systemowa)"
            color = "red"
        elif risk_score >= 3.0:
            phase = "NEUTRALNY (Podwyższona Czujność)"
            color = "orange"
        else:
            phase = "ATAK (Risk-On / Goldilocks — Szukaj Wypukłości!)"
            color = "green"

        return {
            "score":          round(risk_score, 2),
            "composite_risk": round(composite_risk, 3),
            "phase":          phase,
            "color":          color,
            "details":        details,
            "raw_signals":    {k: round(v, 3) for k, v in signals.items()},
        }


class LocalGeopolitics:
    """
    Geopolityk v2.0 — VADER NLP z wagowaniem wiadomości ekonomicznych.
    
    Analiza sentymentu nagłówków z RSS. Wiadomości finansowe są ważniejsze
    niż ogólne (wyższy multiplikator). Dodano dekompozycję na pos/neg/neu.
    """

    FINANCIAL_KEYWORDS = {
        "recession", "crash", "crisis", "default", "collapse", "panic",
        "bank run", "inflation", "rate hike", "fed", "sanctions",
        "bull", "rally", "growth", "boom", "expansion"
    }

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_news(self, oracle_news: list) -> dict:
        if not oracle_news:
            return {
                "compound_sentiment": 0.0, "label": "NEUTRALNY (Brak Danych)",
                "color": "gray", "analyzed_articles": 0,
                "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0
            }

        total_compound = 0.0
        pos_count = neg_count = neu_count = 0
        analyzed_count = 0

        for news in oracle_news:
            text = news["title"] + ". " + news.get("summary", "")
            scores = self.analyzer.polarity_scores(text)

            # Waga: wiadomości finansowe są 2× ważniejsze
            weight = 2.0 if any(kw in text.lower() for kw in self.FINANCIAL_KEYWORDS) else 1.0

            total_compound += scores["compound"] * weight

            if scores["compound"] >= 0.05:
                pos_count += 1
            elif scores["compound"] <= -0.05:
                neg_count += 1
            else:
                neu_count += 1

            analyzed_count += 1

        n = max(analyzed_count, 1)
        avg_sentiment = total_compound / n

        if avg_sentiment <= -0.20:
            label = "STRUKTURALNY STRACH ⛔ (Dominują złe nagłówki)"
            color = "red"
        elif avg_sentiment <= -0.05:
            label = "NEGATYWNY SZUM 🟡 (Lekka presja negatywna)"
            color = "orange"
        elif avg_sentiment >= 0.15:
            label = "GLOBALNY OPTYMIZM 🟢 (Dominują dobre nagłówki)"
            color = "green"
        else:
            label = "SZUM INFORMACYJNY ⚪ (Neutralna prasa)"
            color = "gray"

        return {
            "compound_sentiment": round(avg_sentiment, 3),
            "label":              label,
            "color":              color,
            "analyzed_articles":  analyzed_count,
            "positive_pct":       round(pos_count / n * 100, 1),
            "negative_pct":       round(neg_count / n * 100, 1),
            "neutral_pct":        round(neu_count / n * 100, 1),
        }


class LocalCIO:
    """
    Główny Dyrektor Inwestycyjny v2.0 — Barbell Allocation Engine.
    
    Syntetyzuje Ekonomistę + Geopolityka → decyzja o trybie Barbella.
    Kluczowa rola: wybrać FOKUS dla skanera EVT:
      • Risk-On  → Szukaj aktywów z dodatnim prawym ogonem (krypto, tech, commodities)
      • Risk-Off → Priorytet: obligacje krótkoterminowe, złoto, USD cash
    
    Metodologia: Risk Parity Portfolio (Bridgewater), Regime-Conditional Allocation
    (Ang & Bekaert 2004), Fat-Tail Kelly (Thorp 2006).
    """

    # Mapy klas aktywów per reżim (dla Skanera EVT)
    ASSET_MAP = {
        "risk_on": {
            "target_classes":    ["Tech", "Crypto", "Emerging Markets", "Growth", "Commodities"],
            "etf_focus":         ["QQQ", "TQQQ", "BTC-USD", "EEM", "XLE", "ARKK"],
            "kelly_multiplier":  1.0,   # Pełna ekspozycja Kelly
            "description": "Środowisko Goldilocks. Skaner szuka aktywów o maksymalnej WYPUKŁOŚCI (fat right tail). "
                           "Idealni kandydaci: wysoka beta, dodatni skew, Hurst > 0.55.",
            "gauge": 10,
        },
        "neutral": {
            "target_classes":    ["Value", "Dividend", "Quality", "Diversified", "Real Assets"],
            "etf_focus":         ["VYM", "SCHD", "IVV", "GLD", "VNQ"],
            "kelly_multiplier":  0.5,   # Połowa Kelly (ostrożnie)
            "description": "Środowisko ostrzegawcze. Skaner szuka stabilnych aktywów z dobrą relacją Omega > 1.0 "
                           "i niskim Max Drawdown. Dywidendy + Realne Aktywa.",
            "gauge": 50,
        },
        "risk_off": {
            "target_classes":    ["Short-Duration Bonds", "Gold", "Cash USD", "Defensive"],
            "etf_focus":         ["BIL", "SGOV", "GLD", "UUP", "XLU"],
            "kelly_multiplier":  0.1,   # Minimalny Kelly — ochrona kapitału
            "description": "Środowisko kryzysu. Skaner przełącza się w tryb BUNKER: "
                           "szuka aktywów z negatywną korelacją do akcji (Safe Haven). "
                           "Część ryzykowna Barbella zmniejsza się do minimum.",
            "gauge": 90,
        },
    }

    def synthesize_thesis(self, econ_analysis: dict, geo_analysis: dict, horizon_years: int) -> dict:
        """
        Oblicza Master Risk Score i decyduje o trybie Barbella.
        
        Ekonomista: ~ 70% wagi (twarde dane)
        Geopolityka: ~ 30% wagi (miękkie sygnały rynkowe)
        
        Wynik: 0–100 (0 = pełen Risk-On, 100 = pełen Risk-Off).
        """
        # Ekonomista: score 0–8 → skalujemy do 0–0.7
        econ_score_norm = (econ_analysis["score"] / 8.0) * 0.70

        # Geopolityk: sentiment -1..+1 → ryzyko = (1 - sentiment) / 2 → 0..1 → * 0.30
        geo_sentiment  = geo_analysis["compound_sentiment"]
        geo_risk_norm  = ((1.0 - geo_sentiment) / 2.0) * 0.30  # -1.0 → 1.0, 1.0 → 0.0

        # Łącznie: composite_risk 0..1
        composite_risk = econ_score_norm + geo_risk_norm
        gauge_value    = int(np.clip(composite_risk * 100, 0, 100))

        # Horyzont — długi horyzont zachęca do większej tolerancji ryzyka
        if horizon_years >= 10 and composite_risk < 0.65:
            composite_risk = max(0, composite_risk - 0.08)  # Lekka korekta w dół (Risk-On bias)

        # Tryb Barbella
        if composite_risk >= 0.60:
            regime = "risk_off"
            mode   = "BUNKIER (Risk-Off)"
        elif composite_risk >= 0.35:
            regime = "neutral"
            mode   = "PANCERNY PORTFEL (Neutral-Defensive)"
        else:
            regime = "risk_on"
            mode   = "PEŁNY ATAK (Risk-On — Szukaj Wypukłości!)"

        asset_info = self.ASSET_MAP[regime]

        return {
            "master_risk_score":   round(composite_risk * 25, 2),  # 0–25 dla wstecznej kompatybilności
            "gauge_risk_percent":  gauge_value,
            "composite_risk_0_1":  round(composite_risk, 3),
            "mode":                mode,
            "regime":              regime,
            "description":         asset_info["description"],
            "target_asset_classes":asset_info["target_classes"],
            "etf_focus":           asset_info["etf_focus"],
            "kelly_multiplier":    asset_info["kelly_multiplier"],
        }
