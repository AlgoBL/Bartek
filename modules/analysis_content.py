
import streamlit as st

def display_analysis_report():
    with st.expander("📊 AUTOMATYCZNY RAPORT ANALITYCZNY I REKOMENDACJE (AI)", expanded=True):
        st.markdown("""
        ## Synteza Strategii Sztangi i Rekomendacje Inwestycyjne

        Poniższa analiza stanowi podsumowanie matematycznych podstaw zastosowanych w symulacji oraz rekomendacji dla inwestora, zgodnie z paradygmatem Antykruchości (Nassim Taleb) i Optymalizacji Portfela (Kelly, Shannon, Markowitz).

        ---

        ### 1. Architektura Matematyczna: Dlaczego to działa?

        Twoja strategia opiera się na odrzuceniu "środka" krzywej ryzyka (tradycyjne 60/40) na rzecz ekstremów. Jest to odpowiedź na strukturalne błędy tradycyjnych finansów:
        *   **Błąd Gaussa vs Prawa Potęgowe**: Rynki nie mają rozkładu normalnego. Posiadają "grube ogony" (Fat Tails). Tradycyjne modele nie doszacowują ryzyka krachu. Twoja strategia zakłada, że rzadkie zdarzenia są normą, a nie anomalią.
        *   **Opór Wariancji (Variance Drag)**: Zmienność zabija zysk składany. Wzór $R_G \\approx R_A - \\frac{\\sigma^2}{2}$ pokazuje, że im wyższa zmienność, tym niższy realny wzrost kapitału.
        *   **Rozwiązanie - Strategia Sztangi**:
            *   **Bezpieczna Część (Kotwica)**: Ma zerową wariancję. Jej celem nie jest zysk, lecz eliminacja Opru Wariancji dla ~90% kapitału.
            *   **Ryzykowna Część (Wypukłość)**: Ma dodatnią skośność (nieliniowe zyski). Izolujemy zmienność w małej części portfela, aby nie "zatruła" całego kapitału.

        ---

        ### 2. Zastosowane Modele i Algorytmy

        W przeprowadzonej symulacji (w trybie AI/Backtest) wykorzystano następujące zaawansowane mechanizmy:

        *   **Kryterium Kelly'ego (Zarządzanie Wielkością Pozycji)**:
            *   Klasyczny Kelly maksymalizuje wzrost geometryczny, ale jest zbyt ryzykowny ("ściana Kelly'ego").
            *   **Zastosowanie**: Użyliśmy Ułamkowego Kelly'ego z **Faktorem Kurczenia (Shrinkage)** (wg Bakera-McHale'a), aby uwzględnić błąd estymacji i uniknąć ruiny.
        *   **Demon Shannona (Zbieranie Zmienności)**:
            *   Systematyczne rebalansowanie między nieskorelowanymi aktywami generuje dodatkowy zwrot ("Premia z Rebalansowania").
            *   **Implementacja**: Rebalansowanie Progowe (Threshold), symulujące pasma Davisa-Normana. Rebalansujemy tylko, gdy wagi odchylą się znacząco (np. +/- 20%), co minimalizuje koszty transakcyjne i maksymalizuje efekt "kupuj tanio, sprzedawaj drogo".
        *   **Teoria Wartości Ekstremalnych (EVT)**:
            *   Dobór aktywów do części ryzykownej opierał się na poszukiwaniu "Grubych Ogonów" (Estymator Hilla). Szukamy aktywów o potencjale nieliniowego wzrostu (Opcje, Krypto, Tech).
        *   **Sztuczna Inteligencja - wnioski ze stosowania**:
            *   **Architect (HRP)**: Buduje zdywersyfikowany portfel wewnątrz koszyków. W przeciwieństwie do Markowitza, HRP nie "wariuje" przy wysokiej korelacji. Na wykresie struktury portfela (poniżej) zobaczysz, jak Architect dynamicznie zmienia wagi aktywów ryzykownych, reagując na zmieniające się korelacje. To zapewnia stabilność.
            *   **Trader (RL Agent/Kelly)**: Dynamicznie zarządza lewarem (Kelly). To jest "gaz i hamulec". Trader obserwuje reżim rynkowy. Gdy zmienność spada (hossa), zwiększa ekspozycję (lewaruje). Gdy wykrywa turbulencje (Risk-Off), tnie pozycje szybciej niż jakikolwiek człowiek. Wykres "Pozycja Tradera" (poniżej) pokazuje te decyzje w czasie.
            *   **Hierarchiczny Parytet Ryzyka (HRP)**: Zastępuje tradycyjną korelację (która zawodzi w krachach) strukturą drzewiastą, lepiej dywersyfikując ryzyko.
            *   **Ukryte Modele Markowa (HMM)**: Wykrywają reżimy rynkowe (Risk-On/Risk-Off), działając jako filtr bezpieczeństwa.

        ---

        ### 3. Wnioski i Rekomendacje dla Inwestora (2025-2026)

        Na podstawie wyników symulacji oraz analizy makroekonomicznej, rekomendujemy następującą strukturę portfela:

        #### A. Struktura Docelowa
        | Część Portfela | Alokacja | Aktywa | Rola |
        | :--- | :---: | :--- | :--- |
        | **Bezpieczna (Safe)** | **85-90%** | **SGOV/BIL** (Krótkie Obligacje USA), **GLD** (Złoto) | Ochrona kapitału, płynność do rebalansowania ("Suche Proch"). Unikaj długich obligacji (TLT) w środowisku inflacyjnym. |
        | **Ryzykowna (Risky)** | **10-15%** | **TAIL** (Opcje Put), **DBMF** (Trend Following), **Bitcoin/Tech** | "Crisis Alpha" (zysk w chaosie) i asymetryczny wzrost. Ekspozycja limitowana przez ułamek Kelly'ego. |

        #### B. Zasady Zarządzania
        1.  **Nie rebalansuj kalendarzowo**: Rebalansowanie co miesiąc/rok jest suboptymalne.
        2.  **Użyj Pasm Rebalansowania**: Dokonuj transakcji TYLKO, gdy waga części ryzykownej przekroczy ustalony próg (np. spadnie poniżej 12% lub wzrośnie powyżej 18%). To jest sekret Demona Shannona.
        3.  **Akceptuj Małe Straty**: Część ryzykowna będzie często tracić. Traktuj to jako koszt ubezpieczenia (cost of business) w oczekiwaniu na rzadkie, skokowe wzrosty (Fat Tail Events).
        4.  **Dywersyfikacja Wewnątrz Koszyków**: Używaj HRP (lub równych wag) wewnątrz części ryzykownej, aby nie stawiać wszystkiego na jedną kartę.

        > **Finalna Myśl**: Celem tego portfela nie jest bycie najlepszym każdego dnia, ale przetrwanie każdego krachu i czerpanie zysków z nieuchronnej zmienności rynku. Jesteś teraz "Antykruchy".
        """)
