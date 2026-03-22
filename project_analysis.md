# Analiza projektu: Plan Ulepszeń Naukowych i UX

## 1. Kontekst — Co już jest w projekcie

Projekt jest **wyjątkowo zaawansowany naukowe**. Poniżej skrót aktualnego stanu:

| Obszar | Implementacja |
|--------|--------------|
| Symulacja stochastyczna | Rough Bergomi (FFT Davies-Harte O(N log N)), Hawkes self-exciting jumps, GARCH(1,1), fBM |
| Kopuły | Student-t, Clayton, Gumbel, Frank (Archimedean) + tail dependence matrix |
| Metryki ryzyka | Sharpe, Sortino, Calmar, Sterling, Burke, Rachev, PSR, DSR, Marginal CVaR, Component ES, Expectile Risk, Range VaR/CVaR, SRR, Ulcer, Pain, TCI |
| AI/ML | TFT (Temporal Fusion Transformer), TCN, LSTM, GMM reżimy, GAE autoencoder, RL trader |
| Optymalizacja | Black-Litterman z AI-views, frontier Markowitz, Kelly bayesowski, genetic optimizer |
| Analiza ogonów | EVT/GPD (POT), MEF plot, QQ-Plot GPD, EVT-VaR vs Hist vs Norm |
| Topologia | TDA Betti-0 (crash detector), TDA Betti-1 (cykl detektor), Path Signatures depth-3 |
| Makro | 5-filarowy Control Center, GEX/opcje, Dark Pools, business cycle, reżim zegar |
| Podatki | Belka 19%, tax-loss harvesting, optymalizator podatkowy PL |

---

## 2. Luki Naukowe — Co Warto Dodać

### 🔬 PRIORYTET WYSOKI

#### 2.1 GARCH-MIDAS (Mixed-Data Sampling Volatility)
**Czego brakuje:** Obecny GARCH(1,1) używa tylko danych dziennych. GARCH-MIDAS (Engle & Rangel 2008) łączy:
- **Krótkoterminowa składowa**: GARCH(1,1) na zwrotach dziennych
- **Długoterminowa składowa τ(t)**: zmienne makro z niską częstotliwością (PMI, PKB, CPI co miesiąc)

**Wzór:** `σ²(t) = τ(t) · g(t)`, gdzie `g(t)` = GARCH daily, `τ(t)` = MIDAS macro trend

**Wartość dla projektu:** Zmienność reaguje na makro (PMI < 50 → wyższe τ). Połączenie z Control Center — `τ(t)` steruje wejściowym `risky_vol` w Symulatorze.

**Implementacja:** ~150 linii w `simulation.py`, tab w Symulatorze.

---

#### 2.2 Factor Zoo PCA — Dekompozycja Zwrotów
**Czego brakuje:** Obecny moduł `factor_model.py` jest szkieletem. Brakuje:
- **PCA czynnikowa** na macierzy zwrotów portfela (ile czynników wyjaśnia ≥95% wariancji?)
- **Fama-French 5-factor** decomposition (Market, Size, Value, Profitability, Investment)
- **Principal Risk Factor** visualization (Eigen-portfolio analysis)

**Wartość:** Odpowiada na pytanie: "Czy Barbell jest prawdziwie zdywersyfikowany, czy ma ukrytą ekspozycję na 1 czynnik?" Kruczek zarządzania ryzykiem.

**Ref:** Fama & French (2015), Ang (2014) "Asset Management: A Systematic Approach to Factor Investing"

---

#### 2.3 Walk-Forward Optimisation Scorecard + Bias Tests
**Czego brakuje:** Moduł `walk_forward.py` istnieje, ale brakuje:
- **Combinatorial Purged Cross-Validation** (CPCV — de Prado 2018): n_splits, n_test_splits → chroni przed data leakage
- **Deflated Sharpe Scorecard**: ile strategii było testowanych? DSR koryguje o p-hacking.
- **Backtest Overfitting Test**: probability of backtest overfitting (PBO) Bailey & de Prado (2012)

**Wartość:** Eliminuje fałszywe strategie. De Prado: "99% backtest results are false discoveries."

---

#### 2.4 Lévy-Stable Processes (α-Stable Distribution)
**Czego brakuje:** Skoki Mertona parametryzowane są log-normalnie. Rynki finansowe wykazują Lévy-stable z α ≈ 1.7 (Mantegna & Stanley 1999). Brakuje:
- Parametryzacja α-stable (α, β, σ_s, μ_s) w Symulatorze
- Porównanie ścieżek MC: Gaussian vs Student-t vs α-Stable
- Związek z EVT: dla α < 2 ogony mają nieskończoną wariancję

**Wartość:** Bardziej realistyczny model dla krypto (BTC α ≈ 1.6) i SPY flash crashów.

---

#### 2.5 Signal Decomposition + Spectral Analysis
**Czego brakuje:** Brakuje analizy częstotliwości sygnałów rynkowych:
- **Fourier Power Spectrum** zwrotów → identyfikacja cykli (Juglar 8-11 lat, Kitchin 3-5 lat)
- **Wavelet Decomposition** (Morlet/Haar) → lokalna analiza czasowo-częstotliwościowa trendu
- **Empirical Mode Decomposition (EMD)** → nielinearne składowe rynku (Huang 1998)

**Wartość:** Identyfikacja gdzie jesteśmy w cyklu (Reżim Zegar + Wavelet = mocna para). Dodane do strony `11_Regime_Clock.py`.

---

### 🔬 PRIORYTET ŚREDNI

#### 2.6 SHAP Interpretability dla TCN/TFT
**Czego brakuje:** Model TCN/TFT nie wyjaśnia DLACZEGO klasyfikuje reżim określonym sposobem. Brakuje:
- **SHAP Values** (Lundberg & Lee 2017) dla modelu reżimów
- Waterfall chart: które cechy (vol_21, mom_63, skew) napędzają predykcję Bull/Bear
- **Attention Weight Visualization** dla TFT (które dni historyczne są kluczowe)

**Wartość:** Zrozumienie modelu → zaufanie do decyzji alokacyjnych.

---

#### 2.7 Markov Switching Model (MSM) jako benchmark
**Czego brakuje:** Klasyczny Hamilton (1989) Markov Switching GARCH jako punkt odniesienia dla TCN. Umożliwia:
- Porównanie P(reżim | dane historyczne) z Markov vs TCN vs GMM
- Explicit transition matrix (prawdopodobieństwo przejścia Bull→Bear)
- Likelihood ratio test: czy TCN rzeczywiście bije MSM?

**Implementacja:** `statsmodels.tsa.regime_switching.markov_switching`

---

#### 2.8 Calibration Surface — Implied Volatility Fitting
**Czego brakuje:** Obecny GEX kalkulator pobiera IV ale nie fits surface. Brakuje:
- **SABR model fit** (Hagan 2002): α, β, ρ, ν → pełna powierzchnia IV
- **SVI parametrization** (Gatheral 2004) — minimalistyczny model 5-parametrowy
- Porównanie: Implied Vol Surface vs Rough Bergomi Vol Paths

**Wartość:** Połączenie z symulatorem: kalibracja parametrów Rough Bergomi z rynkowych danych opcyjnych, a nie z estimacji historycznej.

---

#### 2.9 Copula Calibration (Historical Data)
**Czego brakuje:** Obecne kopuły używają predefiniowanych parametrów (theta=2.0). Brakuje:
- **Maximum Likelihood Estimation** parametru θ z danych historycznych
- **AIC/BIC selection**: który typ kopuły najlepiej opisuje daną parę aktywów?
- Dynamiczna kopuła (DCC-Copula): jak zależność ogonowa zmienia się w czasie?

---

### 🔬 PRIORYTET NISKI / DO ROZWAŻENIA

#### 2.10 Rough Path Machine Learning (RPML)
- Obecne path signatures (depth=3) nie są jeszcze używane jako features ML
- Dodanie feed-forward sieci nad sygnaturami → klasyfikacja reżimów drugą metodą
- Ref: Kiraly & Oberhauser (2019), Morrill et al. (2021)

#### 2.11 Nested Monte Carlo (dla ryzyka modelowego)
- Zewnętrzna pętla MC losuje parametry modelu (μ, σ, H) z priorów Bayesowskich
- Wewnętrzna pętla MC dla każdego zestawu parametrów
- Wynik: dystrybucja dystrybucji (model uncertainty, nie tylko parametryczna)
- Ref: Broadie et al. (2011)

---

## 3. Modernizacje UX/UI

> [!IMPORTANT]
> **Zasada:** Nie zmieniamy baz danych. Tylko interfejs, nawigacja, interaktywność.

### 🎯 PRIORYTET WYSOKI (Łatwość Użytkowania)

#### 3.1 Command Palette (⌘K / Ctrl+K)
**Problem:** 22 strony to za dużo do nawigacji przez sidebar. Użytkownik nie wie co gdzie jest.

**Rozwiązanie:** Globalny shortcut `Ctrl+K` → floating search box z fuzzy search po nazwach stron i funkcji.

**Implementacja:** JavaScript overlay + `st.components.v1.html()`. Widoczny na wszystkich stronach przez `app.py`.

---

#### 3.2 Porównanie Multi-Asset Side-by-Side
**Problem:** Aby porównać 2 strategie w Symulatorze, trzeba uruchamiać ją dwa razy i zapamiętywać wyniki.

**Rozwiązanie:** Przycisk "📌 Zapisz Scenariusz A/B/C" w Symulatorze → `st.session_state["scenarios"]`. Zakładka "📊 Porównanie" wyświetla zderzenie 3 zapisanych scenariuszy w jednym wykresie.

---

#### 3.3 Interaktywne Adnotacje na Wykresach
**Problem:** Wykresy equity pokazują zakres 2 lat, ale nie wiadomo które spadki to 2022 bear, które to COVID-19.

**Rozwiązanie:** Nakładka Plotly z historycznymi eventami → `fig.add_vrect()` dla kluczowych wydarzeń (COVID, 2022 Fed, SVB). Checkbox "Pokaż kryzysy" w sidebar.

---

#### 3.4 Tryb Szybkiej Diagnostyki (Quick Dashboard)
**Problem:** Użytkownik musi otworzyć 5-7 stron aby zobaczyć pełen obraz portfela.

**Rozwiązanie:** Nowa zakładka w **Control Center** (app.py): "⚡ Szybka Diagnostyka" — 1 strona z:
- Health Score portfela (z 8_Health_Monitor)
- Reżim + Business Cycle (z app.py)
- Top 3 alerty aktywne
- Regime allocation suggestion (z 12_Regime_Allocation)
- Kelly fraction aktualna
Wszystko na jednym scrollu, bez przełączania stron.

---

#### 3.5 Sticky Metric Cards + Baseline Comparison
**Problem:** Metryki w Symulatorze znikają po zmianie parametru.

**Rozwiązanie:** "📌 Zablokuj jako baseline" → ostatni wynik zapisywany w `session_state["baseline_metrics"]`. Kolejny bieg pokazuje delta (zamiast absolutna): np. Sharpe: 1.23 **(+0.18 vs baseline)** w kolorze zielonym/czerwonym.

---

#### 3.6 Keyboard Shortcuts Guide
**Problem:** Brak skrótów klawiszowych — nawigacja wymaga klikania.

**Rozwiązanie:** Podstrona "⌨️ Skróty" lub floating button "?" → dialog z listą.

| Skrót | Akcja |
|-------|-------|
| `Ctrl+K` | Command palette |
| `Ctrl+Enter` | Uruchom symulację/skan |
| `Ctrl+S` | Zapisz ustawienia globalne |
| `Ctrl+R` | Reset do domyślnych |
| `G + S` | Nawiguj do Symulatora |
| `G + C` | Nawiguj do Control Center |

---

### 🎯 PRIORYTET ŚREDNI

#### 3.7 Personalizowany Dashboard (Moduły do Konfiguracji)
**Problem:** Każdy użytkownik używa różnych modułów. Wealth Optimizer może być bardziej ważny niż Alt Risk Premia dla jednego użytkownika.

**Rozwiązanie:** W Globalnych Ustawieniach: lista modułów z checkboxami "Pokaż w menu". Ukryte moduły znikają z sidebara. Preferencje zapisywane do `global_settings.json` jako `visible_modules: [...]`.

---

#### 3.8 Tooltips "Naukowe" z Ekspanderem do Formul
**Problem:** Każda metryka wymaga wizyty w dokumentacji. Ekspandery `st.expander("🧮 Co to jest?")` istnieją tylko w EVT.

**Rozwiązanie:** Ujednolicony system: każdy `st.metric()` i `st.plotly_chart()` dostaje `help=` z krótkim formulą lub tooltip HTML. Spójny wzorzec z `math_explainer()` (już zaimplementowany w modules/styling.py).

---

#### 3.9 Tryb "Nauczania" vs "Ekspercki"
**Problem:** Zbyt dużo tekstu dla eksperta, zbyt mało wyjaśnień dla nowicjusza.

**Rozwiązanie:** Toggle w Globalnych Ustawieniach: `mode: expert | educational`. 
- **Educational**: wyświetla math explainers automatycznie, pokazuje definicje terminów
- **Expert**: ukrywa wszystkie ekspandery, pokazuje tylko liczby, szybszą nawigację

---

#### 3.10 Export do PDF/Excel
**Problem:** Brak możliwości eksportu wyników symulacji.

**Rozwiązanie:** Przycisk "📤 Eksportuj Raport" w Symulatorze:
- **Excel**: `pd.DataFrame(metrics).to_excel()` + wykresy jako PNG
- **PDF**: `reportlab` lub `fpdf2` → elegancki raport z logo

---

## 4. Podsumowanie Priorytetów

| # | Ulepszenie | Typ | Wpływ | Wysiłek | Priorytet |
|---|-----------|-----|-------|---------|-----------|
| 1 | GARCH-MIDAS + Macro τ | Naukowy | ⭐⭐⭐⭐⭐ | Średni | 🔴 WYSOKI |
| 2 | Factor Zoo PCA | Naukowy | ⭐⭐⭐⭐ | Średni | 🔴 WYSOKI |
| 3 | Quick Dashboard (1-strona) | UX | ⭐⭐⭐⭐⭐ | Niski | 🔴 WYSOKI |
| 4 | Scenario A/B/C Comparison | UX | ⭐⭐⭐⭐ | Niski | 🔴 WYSOKI |
| 5 | Walk-Forward CPCV + PBO | Naukowy | ⭐⭐⭐⭐ | Wysoki | 🟡 ŚREDNI |
| 6 | Lévy-Stable Processes | Naukowy | ⭐⭐⭐ | Średni | 🟡 ŚREDNI |
| 7 | Spectral/Wavelet Analysis | Naukowy | ⭐⭐⭐ | Średni | 🟡 ŚREDNI |
| 8 | SHAP dla TCN/TFT | Naukowy | ⭐⭐⭐ | Średni | 🟡 ŚREDNI |
| 9 | Sticky Baseline Comparison | UX | ⭐⭐⭐⭐ | Niski | 🟡 ŚREDNI |
| 10 | Command Palette (Ctrl+K) | UX | ⭐⭐⭐ | Niski | 🟡 ŚREDNI |
| 11 | Personalizowany Dashboard | UX | ⭐⭐⭐ | Średni | 🟢 NISKI |
| 12 | Markov Switching benchmark | Naukowy | ⭐⭐⭐ | Niski | 🟢 NISKI |
| 13 | Export PDF/Excel | UX | ⭐⭐ | Niski | 🟢 NISKI |
| 14 | Copula MLE Calibration | Naukowy | ⭐⭐⭐ | Wysoki | 🟢 NISKI |

---

## 5. Rekomendacja Kolejności Implementacji

```
Faza 1 (Quick Win):
  - Quick Dashboard (app.py tab "Szybka Diagnostyka")
  - Scenario A/B Comparison (Symulator)
  - Sticky Baseline (Symulator)

Faza 2 (Naukowa):
  - GARCH-MIDAS z macro τ (simulation.py + Symulator UI)
  - Factor Zoo PCA (nowa strona 22_Factor_Analysis.py)
  - Wavelet + Spectral (11_Regime_Clock.py tab dodatkowy)

Faza 3 (Zaawansowane):
  - Walk-Forward CPCV + PBO Scorecard
  - SHAP Values dla TCN/TFT
  - Lévy-Stable distribution (Symulator opcja)
  - Command Palette (JavaScript overlay)
```
