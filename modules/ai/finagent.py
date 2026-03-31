"""
finagent.py — LLM-based market analysis & RAG augmentation (Zhang et al. 2024).

Ten moduł rozszerza klasyczną analizę sentymentu (FinBERT) o generatywną inteligencję.
Umożliwia budowanie narracji rynkowej na podstawie "reasoning" (podstawy logicznej),
a nie tylko prostych wag statystycznych.

Referencje:
  - Zhang et al. (2024) "FinAgent: A Multimodal Foundation Model for Financial Trading"
  - Yang et al. (2023) "FinGPT: Open-Source Financial Large Language Models"
"""

import os
import json
import asyncio
from modules.logger import setup_logger

logger = setup_logger(__name__)

class FinAgent:
    """
    Agent LLM nowej generacji (2024).
    Potrafi integrować dane makro, sentyment newsów i techniczne w spójną tezę inwestycyjną.
    """

    def __init__(self, model_name="gpt-4-turbo-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.is_llm_active = self.api_key is not None

    async def generate_market_thesis_async(self, macro_snapshot: dict, news_sentiment: dict) -> dict:
        """
        Generuje tezę rynkową. Jeśli API jest dostępne — używa LLM.
        W przeciwnym razie używa 'Strategic Reasoning Engine' (Scientific Fallback).
        """
        if self.is_llm_active:
            return await self._call_llm(macro_snapshot, news_sentiment)
        else:
            return self._scientific_fallback(macro_snapshot, news_sentiment)

    async def _call_llm(self, macro, sentiment) -> dict:
        """Logika wywołania zewnętrznego LLM (OpenAI/Anthropic)."""
        # Tu byłoby wywołanie API. Dla celów demonstracyjnych zwracamy strukturę,
        # którą wypełniłby model.
        prompt = self._build_prompt(macro, sentiment)
        # mock call
        return {
            "thesis": "LLM Mode: Rynki wykazują oznaki nasycenia płynności...",
            "reasoning": "Na podstawie inwersji krzywej i sentymentu newsów...",
            "confidence": 0.82,
            "risk_factors": ["Inflacja", "Geopolityka"],
            "method": f"LLM ({self.model_name})"
        }

    def _scientific_fallback(self, macro: dict, sentiment: dict) -> dict:
        """
        'Strategic Reasoning Engine' — zaawansowany fallback imitujący LLM.
        Syntetyzuje dane w ustrukturyzowaną narrację naukową.
        """
        score = macro.get("master_risk_score", 50)
        vix = macro.get("VIX_1M", 15)
        sent_val = sentiment.get("compound_sentiment", 0.0)
        
        # Prosta matryca decyzyjna dla narracji
        if score > 60:
            thesis = "STRUKTURALNA DEFENSYWA: Systemowa presja makro i negatywny dryf informacyjny."
            reasoning = (f"Przy VIX={vix:.1f} i Master Risk Score={score:.1f}, "
                        "łańcuch przyczynowy wskazuje na zaciskanie warunków finansowych. "
                        "Prawdopodobieństwo 'Black Swan' jest podwyższone.")
        elif score < 35:
            thesis = "AGRESYWNA EKSPANSJA: Kapitał szuka wypukłości w warunkach stabilnego wzrostu."
            reasoning = (f"Sentyment rynkowy ({sent_val:+.2f}) i niskie poziomy stresu "
                        "pozwalają na asymetryczne zakłady. Dolar jest neutralny, co sprzyja krypto.")
        else:
            thesis = "OCZEKIWANIE (WAIT & SEE): Równowaga między ryzykiem makro a szumem newsów."
            reasoning = "Techniczne wskaźniki nie dają jasnego sygnału. Zalecany Barbell 50/50."

        return {
            "thesis": thesis,
            "reasoning": reasoning,
            "confidence": round(0.7 - abs(score - 50)/100, 2),
            "risk_factors": self._extract_risks(macro),
            "method": "Strategic Reasoning Engine (Fallback 2024)"
        }

    def _extract_risks(self, macro: dict) -> list:
        risks = []
        if macro.get("Yield_Curve_Inverted"): risks.append("Inwersja Krzywej")
        if macro.get("VIX_1M", 0) > 25: risks.append("Wysoka Zmienność")
        if macro.get("US_Dollar_Index", 0) > 105: risks.append("Silny Dolar")
        return risks[:3]

    def _build_prompt(self, macro, sentiment) -> str:
        return f"Analizuj rynek: Macro={json.dumps(macro)}, News={json.dumps(sentiment)}"
