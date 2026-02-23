from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class LocalEconomist:
    """
    Ekonomista Makro oparty o Reguły Wnioskowania (Heuristics).
    Nie używa LLM. Przypisuje punkty ryzyka na podstawie twardych danych rynkowych.
    """
    def __init__(self):
        pass

    def analyze_macro(self, oracle_snapshot: dict) -> dict:
        risk_score = 0
        details = []

        # 1. Yield Curve Inversion (Najgroźniejszy wskaźnik recesji)
        spread = oracle_snapshot.get('Yield_Curve_Spread', 0.0)
        inverted = oracle_snapshot.get('Yield_Curve_Inverted', False)
        if inverted or spread < 0:
            risk_score += 4
            details.append("KRYTYCZNE: Inwersja Krzywej Dochodowości. Wysokie ryzyko recesji (Hard Landing).")
        elif spread < 0.5:
            risk_score += 1
            details.append("OSTRZEŻENIE: Płaska Krzywa Dochodowości. Trudne warunki kredytowe.")
        else:
            details.append("POZYTYWNE: Stroma Krzywa Dochodowości. Zdrowe warunki kredytowe.")

        # 2. VIX Volatility (Strach na rynku)
        vix = oracle_snapshot.get('VIX_Volatility', 15.0)
        if vix is not None:
            if vix > 30:
                risk_score += 3
                details.append(f"PANIKA (VIX {vix:.1f}): Ekstremalny strach na giełdzie.")
            elif vix > 20:
                risk_score += 1
                details.append(f"ZANIEPOKOJENIE (VIX {vix:.1f}): Podwyższona zmienność rynkowa.")
            else:
                details.append(f"SPOKÓJ (VIX {vix:.1f}): Rynek byka / letarg.")

        # 3. DXY Systemowy (Płynność dolara)
        dxy = oracle_snapshot.get('US_Dollar_Index', 100.0)
        if dxy is not None and dxy > 105:
            risk_score += 1
            details.append(f"TWARDA PŁYNNOŚĆ: Silny dolar (DXY {dxy:.1f}) szkodzi aktywom ryzykownym (Krypto, Rynki Wschodzące).")
            
        # Wyznaczanie fazy
        if risk_score >= 5:
            phase = "DEFENSYWA (Ryzyko Recesji / Panika)"
            color = "red"
        elif risk_score >= 2:
            phase = "NEUTRALNY (Podwyższona Czujność)"
            color = "orange"
        else:
            phase = "ATAK (Ryzyko ON / Goldilocks)"
            color = "green"

        return {
            "score": risk_score, # 0 do 8 (im więcej tym gorzej)
            "phase": phase,
            "color": color,
            "details": details
        }

class LocalGeopolitics:
    """
    Geopolityk oparty o Dedykowany Model NLP (VaderSentiment) działający LOKALNIE na CPU.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_news(self, oracle_news: list) -> dict:
        if not oracle_news:
             return {"compound_sentiment": 0.0, "label": "NEUTRALNY (Brak Danych)", "color": "gray"}
             
        total_compound = 0.0
        analyzed_count = 0
        
        for news in oracle_news:
            # Analizujemy Tytuł i Fragment (Summary)
            text_to_analyze = news['title'] + ". " + news.get('summary', '')
            sentiment = self.analyzer.polarity_scores(text_to_analyze)
            total_compound += sentiment['compound']
            analyzed_count += 1
            
        avg_sentiment = total_compound / analyzed_count if analyzed_count > 0 else 0.0
        
        if avg_sentiment <= -0.15:
            label = "STRUKTURALNY STRACH (Zła Prasa)"
            color = "red"
        elif avg_sentiment >= 0.15:
            label = "GLOBALNY OPTYMIZM (Dobra Prasa)"
            color = "green"
        else:
            label = "SZUM INFORMACYJNY (Neutralnie)"
            color = "orange"
            
        return {
            "compound_sentiment": round(avg_sentiment, 3), # Od -1.0 (złe) do 1.0 (dobre)
            "label": label,
            "color": color,
            "analyzed_articles": analyzed_count
        }

class LocalCIO:
    """
    Główny Dyrektor Inwestycyjny. Scala matematycznie twarde fakty (Ekonomista)
    z miękkimi emocjami Świata (Geopolityk).
    """
    def __init__(self):
        pass

    def synthesize_thesis(self, econ_analysis: dict, geo_analysis: dict, horizon_years: int) -> dict:
        master_score = 0 
        
        # Ekonomista jest ważniejszy niż prasa
        # Econ score idzie od 0 (dobrze) do 8 (źle)
        master_score += econ_analysis['score'] * 2.5 
        
        # Geopolityk idzie od -1.0 (źle) do 1.0 (dobrze) -> odwracamy to, żeby zły sentyment dodawał punkty ryzyka
        geo_sentiment = geo_analysis['compound_sentiment']
        if geo_sentiment < -0.1: master_score += 3
        if geo_sentiment < -0.3: master_score += 2 # Max +5 z newsów
        if geo_sentiment > 0.2: master_score -= 2
        
        # Max master_score ~ 25. Min ~ -2.
        
        # Ostateczna decyzja funduszu
        if master_score >= 12:
            mode = "BUNKIER (Risk-Off)"
            description = "Środowisko skrajnego stresu (Inwersje krzywych lub panika prasy). Algorytm będzie faworyzował w skanerze: Dług USA, Złoto, Dolara i Defensywne ETF-y."
            target_asset_classes = ["Bonds", "Gold", "Defensive", "USD"]
            gauge_value = 85 # Bardzo defensywnie
        elif master_score >= 5:
            mode = "PANCERNY PORTFEL (Neutral-Defensive)"
            description = "Środowisko ostrzegawcze. Skaner skupi się na mniejszej zmienności, dywidendach, stabilnych spółkach i częściowym cashu."
            target_asset_classes = ["Value", "Dividend", "Commodities", "Diversified"]
            gauge_value = 50
        else:
            mode = "PEŁNY ATAK (Risk-On)"
            description = "Czyste niebo inwestycyjne. Algorytm szuka aktywów o wysokiej becie: Krypto, Nasdaq, TQQQ, Rynki Wschodzące."
            target_asset_classes = ["Tech", "Crypto", "Emerging Markets", "Growth"]
            gauge_value = 15
            
        return {
            "master_risk_score": master_score,
            "gauge_risk_percent": gauge_value, # Dla wykresu (0 - 100%)
            "mode": mode,
            "description": description,
            "target_asset_classes": target_asset_classes
        }
