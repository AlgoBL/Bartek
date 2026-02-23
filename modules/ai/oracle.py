import yfinance as yf
import pandas as pd
import feedparser
import time

class TheOracle:
    """
    Warstwa 1: Moduł odpowiedzialny za zbieranie "brudnych" danych ze świata,
    które posłużą Komitotowi Inwestycyjnemu AI do podjęcia decyzji.
    """
    
    def __init__(self):
        # Słownik kluczowych wskaźników makro z Yahoo Finance
        self.macro_tickers = {
            "10Y_Treasury": "^TNX",
            "3M_Treasury": "^IRX",
            "VIX_Volatility": "^VIX",
            "US_Dollar_Index": "DX-Y.NYB",
            "Gold": "GC=F",
            "Copper": "HG=F",
            "Crude_Oil": "CL=F"
        }
        
        # Źródła RSS z wiadomościami finansowymi (Yahoo Finance, Google News)
        self.news_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://news.google.com/rss/search?q=economy+finance+markets+geopolitics&hl=en-US&gl=US&ceid=US:en"
        ]

    def get_macro_snapshot(self) -> dict:
        """Pobiera aktualne wartości wskaźników makro z YFinance."""
        snapshot = {}
        try:
            for name, ticker in self.macro_tickers.items():
                data = yf.Ticker(ticker).history(period="5d")
                if not data.empty:
                    # Bierzemy ostatnią cenę zamknięcia
                    snapshot[name] = round(data['Close'].iloc[-1], 3)
                else:
                    snapshot[name] = None
                    
            # Obliczanie kluczowego wskaźnika: Inwersja Krzywej Dochodowości
            if snapshot.get("10Y_Treasury") and snapshot.get("3M_Treasury"):
                # Spread = Długoterminowe - Krótkoterminowe
                spread = snapshot["10Y_Treasury"] - snapshot["3M_Treasury"]
                snapshot["Yield_Curve_Spread"] = round(float(spread), 3)
                snapshot["Yield_Curve_Inverted"] = bool(spread < 0)
        except Exception as e:
            print(f"Błąd TheOracle podczas pobierania Makro: {e}")
            
        return snapshot
        
    def get_latest_news_headlines(self, max_items: int = 50) -> list:
        """
        Pobiera z RSS najnowsze, gorące nagłówki do analizy sentymentu.
        """
        headlines = []
        try:
            for feed_url in self.news_feeds:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    headlines.append({
                        "title": entry.title,
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:200] + "..." # Truncate summary
                    })
        except Exception as e:
            print(f"Błąd TheOracle podczas parsowania Newsów: {e}")
            
        # Odfiltrowanie duplikatów i sortowanie jeśli możliwe (RSS zazwyczaj jest chronologiczne)
        unique_titles = set()
        clean_headlines = []
        for h in headlines:
            if h["title"] not in unique_titles:
                unique_titles.add(h["title"])
                clean_headlines.append(h)
                
        return clean_headlines[:max_items]

    def generate_oracle_report(self) -> str:
        """
        Konsoliduje dane w przyjazny dla LLM string (raport tekstowy).
        """
        macro = self.get_macro_snapshot()
        news = self.get_latest_news_headlines(max_items=30)
        
        report = "=== GLOBAL MACROECONOMIC SNAPSHOT ===\n\n"
        for key, val in macro.items():
            report += f"- {key}: {val}\n"
            
        report += "\n=== LATEST GEOPOLITICAL & MARKET NEWS ===\n\n"
        for i, n in enumerate(news):
            report += f"{i+1}. {n['title']} ({n['published']})\n"
            
        return report

# Prosty test wbudowany
if __name__ == "__main__":
    oracle = TheOracle()
    print("Zbieranie danych z wyroczni...")
    print(oracle.generate_oracle_report())
