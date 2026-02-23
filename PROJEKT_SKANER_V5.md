# Koncepcja V5.0: Autonomiczny Makro-Skaner Wypukłości (AI Quant Agent)

To, co opisujesz, to przejście z poziomu "Narzędzie Kwalifikacji Ryzyka" na poziom **"Autonomiczny Fundusz Hedgingowy (AI Hedge Fund)"**. W obecnej wersji Skaner potrzebuje, żebyś podał mu listę tickerów (np. S&P500) i on je matematycznie pozycjonuje. 

W nowej wizji podajesz tylko **Horyzont Czasowy (np. 5 lat)**, a algorytm samodzielnie buduje tezę inwestycyjną opartą na świecie zewnętrznym.

Oto projekt architektury takiego modułu i moje propozycje rozwoju tej koncepcji, podzielone na warstwy:

---

## 🌍 Warstwa 1: Połykacz Danych Makro i Geopolitycznych (The Oracle)

Zanim matematyka oceni wykres, system musi zrozumieć *stan świata*.

1. **Integracja z FRED API (The Federal Reserve):**
   - System każdego dnia automatycznie pobiera kluczowe wskaźniki gospodarcze: Podaż pieniądza (M2), Inflację (CPI), Bezrobocie, bazowe Stopy Procentowe oraz kluczowy **Spread Krzywej Dochodowości (10Y minus 2Y)** — ostateczny predyktor recesji.
2. **Globalny Sentyment Geopolityczny (NLP na News API):**
   - Podpięcie pod źródło wiadomości finansowych (np. Alpaca News API lub Finnhub).
   - Przetwarzanie naturalnego języka (NLP): Duży model językowy w locie (np. używając lekkiego API) czyta 1000 ostatnich nagłówków światowych agencji (Reuters, Bloomberg).
   - Ekstrakcja kluczowych słów: *np. "wojna celna", "subwencje na półprzewodniki", "kryzys energetyczny", "stymulacja w Chinach"*.
3. **Alternatywne Wskaźniki Kwantowe:**
   - Wrzucenie do modelu rynkowego indeksu strachu (VIX), indeksu frachtu morskiego (Baltic Dry Index - pokazującego, czy światowy handel zwalnia), indeksu siły dolara (DXY) i cen miedzi (Dr. Copper).

## 🧠 Warstwa 2: AI Makro-Stratedzy (Multi-Agent System)

Zamiast jednego płaskiego skryptu, budujemy wirtualny komitet inwestycyjny:

1. **Agent Ekonomista:** Analizuje dane z FRED. Stwierdza np. *"Jesteśmy w fazie Stagflacji (niski wzrost, wysoka inflacja)."* Z historycznych korelacji wie, że wtedy wygrywają surowce, rynki wschodzące i spółki dywidendowe borykające się z twardą infrastrukturą.
2. **Agent Geopolityk:** Na bazie newsów wyłapuje strukturalne megatrendy. Np. *"Napięcia na linii USA-Chiny prowadzą do nearshoringu (przenoszenia fabryk) do Meksyku, Indii, i Wietnamu. Wyceniam ryzyko wpadnięcia Europy w recesję jako wysokie."*
3. **Synteza Lidera (Chief Investment Officer):** Model na podstawie rad dwóch pierwszych agentów i **Twojego horyzontu (np. 10 lat)** generuje ścisłą tezę. 
   - *Wynik:* Skupiamy się na ETF-ach reprezentujących rynki energii jądrowej (URA), gospodarki wschodzące ościenne (EWW - Meksyk) i globalne srebro (SLV). Odpadają spółki technologiczne Growth (zbyt zależne od stóp ujemnych).

## 🔬 Warstwa 3: Mikro-Skaner Finansowy (Screening & Filtracja)

Gdy Agenci wybiorą sektory i kierunki, przechodzimy do konkretów:

1. **Globalny Screener API (Automatyczne Zaciąganie Tickerów):**
   - System posiada wbudowany Screener (np. pakiety `yahoo-fin` lub poprzez FMP API), z którego automatycznie ściąga listę 2000 np. spółek lub ETF-ów pasujących do wybranej tezy z Warstwy 2.
2. **Analiza Fundamentalna (Piotroski F-Score / Altman Z-Score):**
   - Skaner ściąga bilanse spółek. Odrzuca z automatu firmy o ogromnym zrolowanym zadłużeniu i słabym przepływie wolnej gotówki (Free Cash Flow). Na placu boju zostaje 200 najlepszych jakościowo "kandydatów".

## ⚙️ Warstwa 4: Ostateczna Egzekucja Matematyczna (EVT i Kopuły)

Oto moment, w którym do gry wkracza to, co zrobiliśmy do tej pory, ale w sterydach:

1. Przez sito fundamentalne i makroekonomiczne przeszło np. 200 aktywów.
2. Odpalamy nasz ulepszony skaner z Teorii Wartości Ekstremalnych (POT). Algorytm sprawdza *kształt powrotów* tych 200 aktywów. Wybiera 10% tych, które mają najczęstsze "pozytywne niespodzianki" (bardzo grube prawe ogony i ucięte lewe).
3. System wrzuca je na nową wizualizację (Hierarchical Dendrogram) i pilnuje, by nie wybrać 5 rzeczy skorelowanych ze sobą (Kowariancja informacyjna).

## 🚀 Jak wyglądałby interfejs (UX) nowej aplikacji?

**Ekran 1: "The Command Center"**
- Suwak: *Horyzont inwestycyjny (1 - 30 lat)*.
- Przełącznik: *Preferowany poziom ryzyka*.
- Przycisk: **"Odpal Globalny Syntezator V5"**.

**Ekran 2: Wynik na żywo (Proces myślowy AI)**
Pasek przewija się i pokazuje kolejne kroki, podobnie jak na filmach hakerskich:
- *"Połączono z FRED. Wykryto inwersję krzywej rentowności."*
- *"Analiza Newsów: Dominujący sentyment: Zbrojenia, Ograniczenia półprzewodników, Twarde Lądowanie (Hard Landing)."*
- *"Formułowanie wektora inwestycyjnego: Defensywne surowce, Ochrona kapitału w 70%, 30% w asymetryczne rynki obrzeżne."*
- *"Pobrano 8 412 tickerów z giełd światowych -> Odrzucono 7 900 przez filtry F-Score."*
- *"Przeprowadzanie Teorii Wartości Ekstremalnych na 512 aktywach..."*

**Ekran 3: Gotowy Raport i Trade Ideas**
Aplikacja wypluwa:
1. **Tezę Makroekonomiczną (PDF/Tekst):** Napisaną ludzkim językiem analizę tego, dokąd zmierza świat w Twoim horyzoncie czasowym.
2. **Rekomendowany Portfel Barbell:** Np. "75% amerykańskie bony skarbowe (SHV) + 25% podziału między ETF URA, Bitcoin i konkretne akcje obronne spółki X".
3. **Mapa Korelacji i EVT:** Graficzne potwierdzenie Twoim zaufanym Dendrogramem i Ridge-Plotem, dlaczego akurat te tickery mają w sobie zaszytą matematyczną i makroekonomiczną nagrodę.

---

Oznaczałoby to stworzenie systemu, którego zadaniem byłoby być w 100% obiektywnym, chłodnym obliczeniowo, globalnym zarządcą. Wdrożenie tego polegałoby głównie na podłączeniu zewnętrznych darmowych i płatnych bramek z danymi masowymi (News + Makro) oraz zintegrowaniu silnika ustrukturyzowanych zapytań do LLMów np. z wykorzystaniem `langchain`.

**Brzmi ambitnie?**
