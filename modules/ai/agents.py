import json
import google.generativeai as genai
from modules.secrets_manager import load_api_key

class BaseAgent:
    """Klasa bazowa dla Agentów z podpiętym API Gemini."""
    def __init__(self, role_prompt: str):
        self.api_key = load_api_key()
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=role_prompt)
        else:
            self.model = None

    def analyze(self, data: str) -> str:
        if not self.model:
            return "BŁĄD: Brak klucza API Gemini w ustawieniach."
        try:
            response = self.model.generate_content(data)
            return response.text
        except Exception as e:
            return f"Błąd LLM: {str(e)}"

class EconomistAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role_prompt="Jesteś wybitnym ekonomistą makroekonomicznym i strategiem quant. "
                        "Jesteś chłodny w obyciu, oszczędny w słowach. Rozumiesz Teorię Wartości Ekstremalnych "
                        "oraz inwersję krzywej dochodowości. Twoim zadaniem jest ocena środowiska makroekonomicznego."
        )

    def analyze_macro(self, oracle_snapshot: dict) -> str:
        prompt = f"""
        Przeanalizuj poniższe parametry makroekonomiczne pochodzące ze Skanera V5:
        {json.dumps(oracle_snapshot, indent=2)}
        
        Skomentuj ryzyko recesji (patrząc na Yield Curve Spread), ogólne środowisko płynnościowe i siłę dolara.
        Zakończ jedną jasną konkluzją (np. Ryzyko Off, Stagflacja, Ryzyko On).
        """
        return self.analyze(prompt)

class GeopoliticsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role_prompt="Jesteś czołowym analitykiem wywiadu geopolitycznego. "
                        "Czytasz między wierszami nagłówków gazet i identyfikujesz strukturalne megatrendy. "
                        "Twoim głównym zmartwieniem są szoki podażowe, wojny i protekcjonizm."
        )

    def analyze_news(self, oracle_news: list) -> str:
        news_text = "\n".join([f"- {n['title']}" for n in oracle_news])
        prompt = f"""
        Przeanalizuj poniższe najnowsze światowe nagłówki finansowe:
        {news_text}
        
        Wypisz max 3 najważniejsze globalne ryzyka i megatrendy z nich wynikające. 
        Czy dominują tematy deflacyjne, czy stagflacyjne/zbrojeniowe?
        """
        return self.analyze(prompt)

class ChiefInvestmentOfficerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role_prompt="Jesteś Dyrektorem Inwestycyjnym (CIO) funduszu hedgingowego. "
                        "Twój cel to fuzja opinii Ekonomisty i Geopolityka w jeden konkretny plan działania. "
                        "Rekomendujesz, jakie aktywa mają grube prawe ogony i są najbardziej asymetryczne w tym środowisku."
        )

    def synthesize_thesis(self, economist_report: str, geopolitics_report: str, horizon_years: int) -> str:
        prompt = f"""
        Oto raporty twoich podwładnych:
        
        --- RAPORT EKONOMISTY (Dane twarde): ---
        {economist_report}
        
        --- RAPORT GEOPOLITYKA (Sentyment miekki): ---
        {geopolitics_report}
        
        Opracuj strategię 'Barbell' (zbudowaną z nudnego kapitału chroniącego przed stratą oraz hiper-ryzykownego ogona zysków)
        pod horyzont: {horizon_years} lat.
        Wymień w 3 punktach, jakich branż, krajów (ETF-ów) lub klas aktywów Skaner powinien szukać w następnym kroku.
        """
        return self.analyze(prompt)
