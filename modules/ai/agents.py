
import numpy as np

# ─── Sentiment Backend (FinBERT > VADER fallback) ────────────────────────────
_SENTIMENT_BACKEND = "none"

try:
    from modules.logger import setup_logger
    logger = setup_logger(__name__)
    from transformers import pipeline as _hf_pipeline
    import torch as _torch

    # Try loading FinBERT; falls back to distilroberta if unavailable
    _FINBERT_MODEL = "ProsusAI/finbert"
    _finbert_pipe = _hf_pipeline(
        "text-classification",
        model=_FINBERT_MODEL,
        device=0 if _torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
    )
    _SENTIMENT_BACKEND = "finbert"
except Exception as e:
    logger.warning(f"Błąd ładowania FinBERT: {e} -> Fallback do VADER")
    _finbert_pipe = None

if _SENTIMENT_BACKEND != "finbert":
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADERAnalyzer
        _vader_analyzer = _VADERAnalyzer()
        _SENTIMENT_BACKEND = "vader"
    except Exception as e:
        logger.error(f"Całkowity brak NLP (Vader fail): {e}")
        _vader_analyzer = None
        _SENTIMENT_BACKEND = "none"


def _score_text(text: str) -> float:
    """
    Returns compound sentiment score in [-1, +1].
    Priority: FinBERT → VADER → 0.0 (neutral fallback).
    """
    if _SENTIMENT_BACKEND == "finbert" and _finbert_pipe is not None:
        try:
            result = _finbert_pipe(text[:512])[0]
            label = result["label"].lower()   # 'positive', 'negative', 'neutral'
            score = float(result["score"])    # confidence 0-1
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Błąd analizy FinBERT: {e}")
            pass

    if _SENTIMENT_BACKEND == "vader" and _vader_analyzer is not None:
        try:
            return float(_vader_analyzer.polarity_scores(text)["compound"])
        except Exception as e:
            logger.warning(f"Błąd analizy VADER: {e}")
            pass

    return 0.0


def get_sentiment_backend() -> str:
    """Returns which backend is active: 'finbert', 'vader', or 'none'."""
    return _SENTIMENT_BACKEND


# ─── LocalEconomist ───────────────────────────────────────────────────────────

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
        signals["yield_curve"] = float(np.clip(-spread / 2.0 + 0.5, 0, 1))
        if spread < 0:
            details.append(f"KRYTYCZNE ⛔: Inwersja Krzywej ({spread:.2f}%). Historycznie recesja w ciągu 12-18M.")
        elif spread < 0.5:
            details.append(f"OSTRZEŻENIE ⚠️: Płaska Krzywa ({spread:.2f}%). Trudne warunki kredytowe.")
        else:
            details.append(f"ZDROWE ✅: Stroma Krzywa (+{spread:.2f}%). Ekspansja gospodarcza.")

        # 2. VIX Poziom (Whaley 2009) ────────────────────────────────────
        vix = oracle_snapshot.get("VIX_1M") or oracle_snapshot.get("VIX_Volatility") or 15.0
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
            signals["vix_term_structure"] = float(np.clip((ts_ratio - 0.8) / 0.4, 0, 1))
            if backwardation:
                details.append(f"VIX BACKWARDATION 🔴 (ratio={ts_ratio:.2f}): Panika krótkoterminowa.")
            else:
                details.append(f"VIX Contango 🟢 (ratio={ts_ratio:.2f}): Rynek spokojny.")
        else:
            signals["vix_term_structure"] = 0.3

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
            signals["credit_spread"] = float(np.clip((credit_spread - 1.5) / 3.5, 0, 1))
            if credit_spread > 3.5:
                details.append(f"KRYZYS KREDYTOWY ⛔ (spread={credit_spread:.2f}%): Rynek obligacji korporacyjnych w strachu.")
            elif credit_spread > 2.0:
                details.append(f"STRES KREDYTOWY 🟡 (spread={credit_spread:.2f}%): Ostrożność.")
            else:
                details.append(f"ZDROWY DŁUG 🟢 (spread={credit_spread:.2f}%): Niskie ryzyko defaultu.")
        else:
            signals["credit_spread"] = 0.2

        # 6. Copper/Gold Ratio ────────────────────────────────────────────
        cu_au = oracle_snapshot.get("CuAu_Ratio")
        if cu_au is not None:
            signals["cu_au_ratio"] = float(np.clip(1.0 - (cu_au - 0.1) / 0.3, 0, 1))
        else:
            signals["cu_au_ratio"] = 0.3

        # 7. Copper Trend ─────────────────────────────────────────────────
        copper = oracle_snapshot.get("Copper")
        if copper:
            signals["copper_trend"] = float(np.clip(1.0 - (copper - 3.0) / 2.0, 0, 1))
        else:
            signals["copper_trend"] = 0.3

        # 8. Composite Nowcast Z-Score ─────────────────────────────────────
        composite_risk = sum(
            signals[key] * self._FACTOR_WEIGHTS[key]
            for key in self._FACTOR_WEIGHTS
            if key in signals
        )
        risk_score = float(composite_risk * 8.0)

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


# ─── LocalGeopolitics ─────────────────────────────────────────────────────────

class LocalGeopolitics:
    """
    Geopolityk v3.0 — FinBERT NLP (z fallbackiem VADER).

    Zamiana VADER (2014) na FinBERT (ProsusAI/finbert) — model BERT
    pretrenowany na tekstach finansowych (Financial PhraseBank + Reuters).
    FinBERT rozumie kontekst finansowy: "rate hike" → negative (nie neutral),
    "beat expectations" → positive (nie neutral).

    Hierarchia backendu:
      1. FinBERT (transformers) — najdokładniejszy dla tekstów finansowych
      2. VADER (vaderSentiment)  — szybki, reguły leksykalne
      3. Neutral 0.0             — brak bibliotek

    Wagi: wiadomości finansowe (słowa kluczowe) mają wagę 2×.
    Metodologia: Malo et al. (2014) FinancialPhraseBank, Huang et al. (2023) FinBERT.
    """

    FINANCIAL_KEYWORDS = {
        "recession", "crash", "crisis", "default", "collapse", "panic",
        "bank run", "inflation", "rate hike", "fed", "sanctions",
        "bull", "rally", "growth", "boom", "expansion", "earnings",
        "gdp", "cpi", "ppi", "fomc", "ecb", "rate cut", "quantitative",
        "yield", "spread", "unemployment", "layoffs", "bankruptcy",
    }

    def analyze_news(self, oracle_news: list) -> dict:
        backend = get_sentiment_backend()

        if not oracle_news:
            return {
                "compound_sentiment": 0.0,
                "label": "NEUTRALNY (Brak Danych)",
                "color": "gray",
                "analyzed_articles": 0,
                "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
                "sentiment_backend": backend,
            }

        total_compound = 0.0
        pos_count = neg_count = neu_count = 0
        analyzed_count = 0

        for news in oracle_news:
            text = news["title"] + ". " + news.get("summary", "")
            is_financial = any(kw in text.lower() for kw in self.FINANCIAL_KEYWORDS)
            weight = 2.0 if is_financial else 1.0

            score = _score_text(text)
            total_compound += score * weight

            if score >= 0.05:
                pos_count += 1
            elif score <= -0.05:
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
            "sentiment_backend":  backend,
        }


# ─── LocalCIO ─────────────────────────────────────────────────────────────────

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

    ASSET_MAP = {
        "risk_on": {
            "target_classes":    ["Tech", "Crypto", "Emerging Markets", "Growth", "Commodities"],
            "etf_focus":         ["QQQ", "TQQQ", "BTC-USD", "EEM", "XLE", "ARKK"],
            "kelly_multiplier":  1.0,
            "description": "Środowisko Goldilocks. Skaner szuka aktywów o maksymalnej WYPUKŁOŚCI (fat right tail). "
                           "Idealni kandydaci: wysoka beta, dodatni skew, Hurst > 0.55.",
            "gauge": 10,
        },
        "neutral": {
            "target_classes":    ["Value", "Dividend", "Quality", "Diversified", "Real Assets"],
            "etf_focus":         ["VYM", "SCHD", "IVV", "GLD", "VNQ"],
            "kelly_multiplier":  0.5,
            "description": "Środowisko ostrzegawcze. Skaner szuka stabilnych aktywów z dobrą relacją Omega > 1.0 "
                           "i niskim Max Drawdown. Dywidendy + Realne Aktywa.",
            "gauge": 50,
        },
        "risk_off": {
            "target_classes":    ["Short-Duration Bonds", "Gold", "Cash USD", "Defensive"],
            "etf_focus":         ["BIL", "SGOV", "GLD", "UUP", "XLU"],
            "kelly_multiplier":  0.1,
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
        econ_score_norm = (econ_analysis["score"] / 8.0) * 0.70
        geo_sentiment   = geo_analysis["compound_sentiment"]
        geo_risk_norm   = ((1.0 - geo_sentiment) / 2.0) * 0.30

        composite_risk = econ_score_norm + geo_risk_norm
        gauge_value    = int(np.clip(composite_risk * 100, 0, 100))

        if horizon_years >= 10 and composite_risk < 0.65:
            composite_risk = max(0, composite_risk - 0.08)

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
            "master_risk_score":   round(composite_risk * 25, 2),
            "gauge_risk_percent":  gauge_value,
            "composite_risk_0_1":  round(composite_risk, 3),
            "mode":                mode,
            "regime":              regime,
            "description":         asset_info["description"],
            "target_asset_classes":asset_info["target_classes"],
            "etf_focus":           asset_info["etf_focus"],
            "kelly_multiplier":    asset_info["kelly_multiplier"],
        }
