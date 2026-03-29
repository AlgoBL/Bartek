# Plan Rozszerzenia: Life OS — Algorytm Łowcy Wielorybów

## Kontekst i analiza aktualnego modułu

Aktualny moduł (`pages/20_Life_OS.py`, ~1428 linii) zawiera **21 sekcji** pokrywających:

| # | Sekcja | Nauka bazowa |
|---|--------|-------------|
| 1 | Świat Potęgowy vs Gaussowski | Pareto, Black Swans (Taleb) |
| 2 | Ergodyczność i Przetrwanie | Ole Peters (LML) |
| 3 | Dopamina i RPE | Wolfram Schultz (Cambridge) |
| 4 | Teoria Gier i Sygnalizacja | Nash, BATNA |
| 5 | Kelly Criterion i Barbell | Kelly, Taleb |
| 6 | Teoria Sieci (Burt) | Ronald Burt (structural holes) |
| 7 | Daily OS Algorithm | Cal Newport, Deep Work |
| 8 | Chronobiologia | Satchin Panda, Huberman |
| 9 | Architektura Przepływu | Csíkszentmihályi |
| 10 | Hormeza i Antykruchość | Mark Mattson |
| 11 | Fizyka Nawyków (Fogg) | B.J. Fogg Behavior Model |
| 12 | Dyskonto Hiperboliczne | Kahneman, Thaler |
| 13 | Ewolucja Współpracy (IPD) | Axelrod, Nowak |
| 14 | Mechanism Design | Revelation Principle |
| 15 | Eksploracja vs Eksploatacja (37%) | Multi-Armed Bandit |
| 16 | Teoria Sygnałów (Post-AGI) | Costly Signaling Theory |
| 17 | Wewnątrzpersonalna Teoria Gier | Dual-Self Model |
| 18 | Backward Induction | Subgame-perfect Nash |
| 19 | Antykruchość (Jensen) | Wypukłość Jensena |
| 20 | Ekonomia Uwagi | Governance Overload |
| 21 | Wnioskowanie Przyczynowe | Hawkes Processes |

## Luki zidentyfikowane w module

Moduł jest wyjątkowo bogaty, ale **brakuje mu** kluczowych obszarów:

1. ❌ **Filozofia stoicka** (dyktomia kontroli, negatywna wizualizacja) — fundament odporności psychicznej
2. ❌ **Biologia stresu społecznego** (Sapolsky, hierarchia, kortyzol) — zrozumienie własnych biologicznych reakcji
3. ❌ **Teoria Informacji / Epistemologia** (Shannon Entropy, Bayesian update, filtrowanie rzeczywistości)
4. ❌ **Nauki o złożoności SOC** (Self-Organized Criticality, krawędź chaosu) — kiedy system jest gotowy na przełom
5. ❌ **System 3 — AI i Cognitive Surrender** — najnowsze badania 2024-2026 o atrofii systemu 2
6. ❌ **Metacognition & Noise** (Kahneman 2021: *Noise*) — różnica między biasem a szumem decyzyjnym
7. ❌ **Biologia hierarchii społecznej** (testosteron, dominacja a współpraca) — wchodzenie w relacje z "wielorybami"
8. ❌ **Optimal Transport / Portfolio Life** — matematyka alokacji zasobów życiowych między domenami

---

## Proponowane Nowe Sekcje (22–29)

---

### Sekcja 22 — Stoicka Dychotomia Kontroli i Negatywna Wizualizacja

**Nauka bazowa:**
- Epictetus, *Enchiridion* + Marcus Aurelius, *Meditations*
- Badania CBT/ACT (2024): interwencje stoickie redukują lęk o 34% (ResearchGate, IJISRT)
- "Reserve Clause" - cel z klauzulą `fate permitting`

**Komponenty interaktywne:**
- **Kalkulator Dichotomii Kontroli**: Użytkownik wpisuje swoje obawy, klasyfikuje je jako "w mojej władzy" / "poza moją władzą", wizualizacja jako diagram Venna
- **Negatywna Wizualizacja (Premeditatio Malorum)**: Slider czasu — co stracę jeśli X nie wyjdzie? Jak długo to przetrwa?
- **Evening Stoic Review**: Formularz 3 pytań (Marcus Aurelius format)
- **Reserve Clause Builder**: Generator planów z klauzulą `fate permitting`

**Dlaczego to ważne dla Łowcy Wielorybów:** Stoicyzm to protokół zarządzania emocjami podczas długich "zimowych" faz bez wyników. Zapobiega panic-sell i desperation moves.

---

### Sekcja 23 — Biologia Hierarchii: Kortyzol, Testosteron i Status Społeczny (Sapolsky)

**Nauka bazowa:**
- Robert Sapolsky: badania na pawianach w Serengeti (30 lat)
- Niestabilna hierarchia = wyższy stres u dominujących
- Testosteron ≠ agresja; testosteron = **utrzymanie statusu** (adaptacyjny)
- "The Winner Effect" — Matthew Walker, Ian Robertson: wygrana podnosi T i poprawia wyniki kolejnej rundy

**Komponenty interaktywne:**
- **Mapa Hierarchii Społecznej**: Wykres sieciowy z klastrami — Ty vs wieloryby, gdzie jesteś w lokalnej hierarchii?
- **Symulator Kortyzolu Środowiskowego**: Które środowiska podnoszą Twój bazowy kortyzol (praca open-space, social media, niestabilne relacje)?
- **"Winner Effect" Tracker**: Symulacja jak seria małych zwycięstw modyfikuje hormonalny stan startowy przed trudnymi rozmowami

**Dlaczego ważne:** Wieloryby wyczuwają biologiczne sygnały statusu. Zrozumienie jak budować "wygrywającą biochemię" przed kluczowymi spotkaniami.

---

### Sekcja 24 — Teoria Informacji: Entropia Shannona i Filtrowanie Rzeczywistości

**Nauka bazowa:**
- Claude Shannon (1948): H = -Σ p(x) log₂ p(x)
- "Predictive Coding" (Karl Friston, 2010+): mózg to maszyna do minimalizacji entropii
- Bayesian Updating: P(H|E) = P(E|H)·P(H) / P(E)
- Kahneman 2021, *Noise*: zmienność = drugie obok biasu źródło złych decyzji

**Komponenty interaktywne:**
- **Kalkulator Entropii Informacyjnej**: Zdywersyfikuj swoje źródła informacji — ile "bitów unikalnej informacji" zawiera Twój dzienny input?
- **Bayesian Belief Updater**: Podajesz swoje prior (przekonanie 0-100%) + nowy dowód → posterior. Wizualizacja jak powinno zmieniać się Twoje przekonanie
- **Noise vs Bias Diagnozer**: Classifier decyzji — czy Twoje złe decyzje to skutek systematycznego biasu (przewidywalny błąd) czy szumu (losowa wariancja)?
- **Filtr Informacyjny**: Ile kanałów informacyjnych konsumujesz? Jaki % jest korelowany (nie dodaje nowej entropii)?

**Dlaczego ważne:** Łowca wielorybów musi mieć "edge informacyjny" — unikalny dostęp lub przetworzenie info. Shannon daje matematyczny język do tej analizy.

---

### Sekcja 25 — Nauki o Złożoności: Krawędź Chaosu i Krytyczność (SOC)

**Nauka bazowa:**
- Santa Fe Institute: Per Bak (Self-Organized Criticality, 1987)
- Stuart Kauffman: NK Model → systemy na "krawędzi chaosu" są najbardziej adaptacyjne
- Complexity Economics (W. Brian Arthur) — agent-based models
- 2021 Nobel z Fizyki dla Giorgio Parisi (złożone systemy)

**Komponenty interaktywne:**
- **Sandpile Model (SOC Simulator)**: Wizualizacja krytyczności — kiedy system jest gotowy na kaskadę zmiany (wieloryba)?
- **NK Landscape Explorer**: Suwak N (liczba zmiennych życiowych) × K (wzajemne zależności) — jak różne konfiguracje wpływają na dostępność "peaks of fitness"
- **Phase Transition Detector**: Które sygnały wskazują że Twój system (projekt, relacja, kariera) zbliża się do punktu krytycznego?

**Dlaczego ważne:** Wieloryby nie pojawiają się losowo — pojawiają się gdy system jest na krawędzi krytyczności. Zrozumienie SOC daje narzędzia do rozpoznanienia KIEDY naciskać.

---

### Sekcja 26 — AI i Cognitive Surrender: System 3 (Badania Wharton 2024-2026)

**Nauka bazowa:**
- Wharton School (2025): "System 3" — AI jako zewnętrzny system kognitywny
- "Cognitive Surrender": Użytkownicy AI zatrzymują ocenę output i przyjmują go jako własne myślenie
- Atrofia Systemu 2: systematyczne zaniedbywanie deliberatywnego myślenia → osłabienie mięśnia krytycznego
- "AI-Mediated Overconfidence" — badania Nature (2024)

**Komponenty interaktywne:**
- **Cognitive Independence Score**: Ile % Twoich kluczowych decyzji podjąłeś bez AI w tym tygodniu?
- **System 3 Risk Calculator**: Mapa obszarów gdzie korzystasz z AI — gdzie ryzykujesz atrofię vs gdzie jest to uzasadnione
- **Deliberative Practice Logger**: Tracker momentów gdzie świadomie odmówiłeś pomocy AI by wzmocnić własną zdolność analityczną
- **Signal vs AI Noise**: Czy Twoja intuicja rynkowa jest coraz słabsza po przejściu na AI-generated insights?

**Dlaczego ważne:** Paradoks: Łowca wielorybów używa AI jako narzędzia nauki, ale musi chronić swoją intuicję i zdolność do rozpoznawania sygnałów, które AI nie rozumie.

---

### Sekcja 27 — Metacognition i Noise: Kalibrowanie Własnych Uprzedzeń (Kahneman 2021)

**Nauka bazowa:**
- Kahneman, Sibony, Sunstein (2021): *Noise: A Flaw in Human Judgment*
- Różnica: **Bias** (systematyczna, kierunkowa pomyłka) vs **Noise** (losowa wariancja w identycznych sytuacjach)
- Overconfidence Effect, Anchoring, Planning Fallacy
- Metacognitive accuracy = korelacja między pewnością siebie a dokładnością

**Komponenty interaktywne:**
- **Calibration Test**: Seria pytań z zakresem pewności (90% confidence intervals) → pomiar czy jesteś over/under-confident
- **Decision Journal Analyzer**: Loguj decyzje z prognozą i datą weryfikacji → wykres kalibracji w czasie
- **Bias Identifier Matrix**: Zidentyfikuj swoje top 5 systematycznych uprzedzeń ze 180+ znanych biasów
- **Noise Audit**: Symulacja: jak bardzo Twoje decyzje zmieniają się gdy podejmujesz je w różnych stanach (rano vs wieczór, po spaniu vs po nieprzespanej nocy)?

**Dlaczego ważne:** Łowca wielorybów musi znać własne "faulty sensors". Metacognition to zdolność wyjścia poza własny umysł i oceny jego jakości.

---

### Sekcja 28 — Biologia Społeczna: Oksytocyna, Zaufanie i Więzi Wysokiej Wartości

**Nauka bazowa:**
- Paul Zak (Claremont Graduate Univ.): "Moral Molecule" — oksytocyna = waluta zaufania
- Robin Dunbar: liczba Dunbara (150) + hierarchia kręgów zaufania (5-15-50-150-500)
- Oxytocin-Trust Experiment: jedyrazowe podanie oksytocyny podnosi skłonność do zaufania o 17%
- "Vulnerability = Trust Amplifier" (Brené Brown + neurobiologia)

**Komponenty interaktywne:**
- **Dunbar Circle Mapper**: Wizualizacja Twoich kręgów społecznych — ile osób w każdym pierścieniu? Czy jest zdrowa dystrybucja?
- **Oxytocin Protocol Checklist**: Jakie zachowania budują oksytocynę u "wieloryba" — autentyczność, wrażliwość, fizyczna obecność, wspólny wysiłek
- **Trust Debt Calculator**: Ile "zaufania" budujesz vs zużywasz tygodniowo w relacjach wysokiej wartości?
- **High-Value Bond Index**: Mapa relacji z oceną długości, głębokości, wzajemności

**Dlaczego ważne:** Wieloryby to przede wszystkim LUDZIE. Neurobiologia zaufania wyjaśnia dlaczego niektóre zachowania natychmiastowo budują więź, a inne ją niszczą.

---

### Sekcja 29 — Optimal Transport Life: Matematyczna Alokacja Zasobów Życiowych

**Nauka bazowa:**
- Gaspard Monge (1781) + Kantorovich (Nobel 1975): Optimal Transport Theory
- Cédric Villani (Fields Medal 2010): Wasserstein Distance
- "Life Portfolio Theory" — wielodomenowa optymalizacja (zdrowie, finanse, relacje, projekty)
- Roy Baumeister: Ego Depletion i skończoność zasobów kognitywnych

**Komponenty interaktywne:**
- **Life Resource Allocation Map**: Ważona mapa wszystkich domen życia (zdrowie, finanse, relacje, projekty, wiedza) — ile "energii" alokujesz gdzie?
- **Wasserstein Distance Calculator**: Odległość między Twoją aktualną dystrybucją zasobów a Twoją "idealną" — co trzeba zmienić? 
- **Trade-off Analyzer**: Interaktywny Pareto front — w którym punkcie więcej zasobów w karierze odbiera relacjom i zdrowiu?
- **Energy Budget Optimizer**: 168 godzin tygodnia — jak je optymalnie dystrybuować między domains by maksymalizować wielowymiarową funkcję wartości?

**Dlaczego ważne:** Błąd sub-optimizacji: optymalizowanie jednej domeny kosztem innych. Optimal Transport daje matematykę do całościowego podejścia.

---

## Podsumowanie Rozszerzeń

| # | Sekcja | Główny naukowiec | Kluczowe narzędzie interaktywne |
|---|--------|-----------------|--------------------------------|
| 22 | Stoicka Dychotomia Kontroli | Epictetus, Marcus Aurelius | Kalkulator Dichotomii + Evening Review |
| 23 | Biologia Hierarchii (Sapolsky) | Robert Sapolsky | Mapa Hierarchii + Winner Effect Tracker |
| 24 | Teoria Informacji (Shannon) | Shannon, Friston, Kahneman | Bayesian Belief Updater + Noise Diagnozer |
| 25 | Krawędź Chaosu (SOC) | Per Bak, Kauffman, SFI | Sandpile Simulator + Phase Detector |
| 26 | AI & Cognitive Surrender | Wharton Lab (2025) | Cognitive Independence Score |
| 27 | Metacognition i Noise | Kahneman (2021) | Calibration Test + Decision Journal |
| 28 | Biologia Społeczna (Oksytocyna) | Paul Zak, Robin Dunbar | Dunbar Circle Mapper + Trust Debt |
| 29 | Optimal Transport Life | Villani, Kantorovich | Life Resource Allocation Map |

## Open Questions

> [!IMPORTANT]
> **Wybór priorytetów**: Sekcji jest 8. Czy implementuję wszystkie naraz, czy zaczynamy od np. 4 najważniejszych? Moja propozycja priorytetu: **22, 24, 26, 27** (najbardziej "umysłowe" i natychmiast praktyczne).

> [!WARNING]
> **Rozmiar pliku**: Aktualny moduł ma 1428 linii (~81KB). 8 nowych sekcji doda ok. 1200-1600 linii. Plik stanie się bardzo duży. Warto rozważyć podział na zakładki lub podmodule.

> [!NOTE]
> **Język**: Wszystkie nowe sekcje są po polsku, spójne ze stylem obecnego modułu.

## Verification Plan

### Automated
- Uruchomienie streamlit i sprawdzenie że wszystkie sekcje renderują poprawnie
- Check czy nie ma ImportError dla nowych bibliotek

### Manual
- Sprawdzenie interaktywności wszystkich sliderów i kalkulatorów
- Weryfikacja poprawności matematycznej (Bayes, Shannon entropy, Wasserstein)
