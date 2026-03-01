# 📊 Analiza Projektu — Barbell Strategy Quant Platform
**Data raportu:** 2026-03-01 | **Wersja:** v1.0

---

## 🏗️ STAN OBECNY — CO JUŻ MAMY

### Strony (Pages)

| # | Strona | Co robi |
|---|--------|---------|
| 0 | **Control Center** (`app.py`) | Dashboard makro: VIX, TED Spread, Yield Curve, Credit Spreads, GEX, M2, Baltic Dry, Miedź/Złoto, Fear&Greed, Breadth. AI Agents: LocalEconomist + LocalGeopolitics + LocalCIO → Master Risk Score (0-100) |
| 1 | **Symulator** (`1_Symulator.py`) | Monte Carlo portfela Barbell: GARCH, Rough Bergomi/Heston, Student-t Copula, Clayton/Gumbel/Frank, Sobol QMC, Walk-Forward Validation, Bootstrap CI, Tax Belka |
| 2 | **Skaner** (`2_Skaner.py`) | EVT Scanner aktywów (S&P500, STOXX50, WIG20, ETFs, Crypto): EVT right/left tail, Hurst, Omega, Amihud, Momentum, HRP Dendrogram, MST Network |
| 3 | **Stress Test** (`3_Stress_Test.py`) | Historyczne kryzysy (COVID, GFC, Stagflacja, dot-com), scenariusze syntetyczne (Klimat ECB 2024, AI/Geopolityka), Reverse Stress Test |
| 4 | **Emerytura** (`4_Emerytura.py`) | Monte Carlo planowania emerytalnego: inflacja stochastyczna, dożycie (Gompertz), strategie wypłat (stała, % portfela, bucket) |
| 5 | **EVT Analysis** (`5_EVT_Analysis.py`) | Zaawansowana analiza EVT-GPD: VaR/CVaR 95/99/99.9%, Spectral Risk Measure, Joint Exceedance, Mean Excess Plot |
| 6 | **BL Dashboard** (`6_BL_Dashboard.py`) | Black-Litterman z AI views, optymalizacja mean-variance posterior |
| 7 | **DCC Dashboard** (`7_DCC_Dashboard.py`) | DCC-GARCH korelacje dynamiczne, Autoencoder latent factors, Factor Model |

### Kluczowe Moduły

| Moduł | Implementacja |
|-------|--------------|
| `risk_manager.py` | Empirical Kelly, ERC Risk Budgeting, Vol Targeting, Stop-Loss, EVT-POT, Adaptive VaRGPD-ML (XGBoost), Spectral Risk Measure, Joint Exceedance Matrix |
| `simulation.py` | GARCH(1,1), Rough Bergomi, Rough Heston-Hawkes, Student-t Copula, Clayton/Gumbel/Frank Copulas, fBM, Sobol QMC, Numba JIT |
| `vanguard_math.py` | TDA Betti-0 (crash indicator), TDA Betti-1 (cycle detection), Dynamic Copulas, fBM paths, GEX/Options Skew, Bayesian Kelly, Path Signatures |
| `metrics.py` | Sharpe, Sortino, Calmar, Sterling, Burke, Omega, Rachev, Ulcer Index, Pain Index, Probabilistic Sharpe (PSR), Drawdown Analytics |
| `frontier.py` | HRP (Lopez de Prado), Min-CVaR LP, Black-Litterman, Efficient Frontier, Max-Sharpe, Max-Omega |
| `stress_test.py` | 8 kryzysów historycznych, 4 scenariusze syntetyczne (Klimat, AI, Geopolityka), Reverse Stress Test (Basel III) |
| `scanner.py` | EVT POT right/left tail, Hurst, Omega, Amihud, Momentum 12-1, Composite Z-Score, HRP Dendrogram, MST Network |
| `black_litterman.py` | Pełny B-L z AI Views (CAPM Prior → Posterior), Idzorek confidence |
| `walk_forward.py` | Rolling WFV, BCa Bootstrap CI, Block Bootstrap |
| `dcc_garch.py` | DCC-GARCH dynamiczne korelacje |
| `factor_model.py` | Multi-factor model (Fama-French style) |
| `autoencoder_factors.py` | Autoencoder: unsupervised latent risk factors |
| `emerytura.py` | Stochastic retirement planner |
| `ai/agents.py` | LocalEconomist, LocalGeopolitics, LocalCIO (FinBERT/VADER NLP) |
| `ai/oracle.py` | TheOracle: dane makro (FRED, YFinance, RSS) |
| `ai/lstm_observer.py` | LSTM market regime prediction |
| `ai/rl_trainer.py` | Reinforcement Learning portfolio agent |
| `data_provider.py` | YFinance → Stooq fallback |

---

## 🚀 NOWE MODUŁY — REKOMENDOWANE DO IMPLEMENTACJI

> Skupiam się **wyłącznie** na modułach merytorycznych, które pomagają **rozumieć ryzyko**, **chronić** i **powiększać majątek**. Nie ma tutaj zmian kosmetycznych ani UI-only.

---

### 🔴 PRIORYTET 1 — KRYTYCZNE (Natychmiastowa ochrona kapitału)

---

#### MODULE 1: `portfolio_health_monitor.py` — Ciągły Monitoring Zdrowia Portfela

**Problem który rozwiązuje:** Brak alertów w czasie rzeczywistym gdy portfel zbliża się do granicy ryzyka. Teraz musisz ręcznie sprawdzać wskaźniki.

**Co implementuje:**
- **Drawdown early warning** — alert gdy portfel spada >5%, >10%, >15% od szczytu (ATH tracking)
- **Volatility spike detector** — gdy realized vol portfela wzrasta >2 odchylenia standardowe od 90-dniowej średniej
- **Correlation breakdown alert** — gdy korelacje w portfelu nagłe rosną (sygnał kryzysu — aktywa przestają być dywersyfikowane)
- **Kelly fraction monitor** — alert gdy bieżąca pozycja przekracza optymalny Kelly sizing
- **Liquidation cascade risk** — szacowanie ryzyka wymuszonej sprzedaży w warunkach margin call (dla portfeli z dźwignią)
- **Eksport alertów do e-mail/webhook** (Pushover/Telegram)

**Dlaczego to ważne:** Pasywne dashboardy nie chronią kapitału. Aktywne alerty to pierwsza linia obrony.

**Nowe dane wejściowe:** Portfel użytkownika (wagi, ceny wejścia, wartości pozycji z CSV/API brokera)

---

#### MODULE 2: `regime_adaptive_allocation.py` — Dynamiczne Przełączanie Reżimów

**Problem który rozwiązuje:** Symulator i BL Dashboard dają statyczne alokacje. Rynek ma reżimy (bull/bear/crisis/sideways) i optymalne wagi są inne w każdym reżimie.

**Co implementuje:**
- **Hidden Markov Model (HMM) 3-state** — automatyczne rozpoznawanie reżimu: Risk-On / Risk-Off / Crisis (Hamilton 1989)
- **Regime-conditional covariance** — osobna macierz Σ dla każdego reżimu (kryzys ma 3× wyższe korelacje)
- **Smooth transition weights** — zamiast skokowego przejścia, wygładzenie przez sigmoid (unikamy market impact)
- **Regime persistence forecasting** — ile jeszcze potrwa obecny reżim? (duration model)
- **Backtested regime switching** — vs Buy & Hold na danych historycznych od 1990

**Połączenie z istniejącymi:** Pobiera Master Risk Score z Control Center; przekazuje wagi do Symulatora.

**Dlaczego to ważne:** Strategia Barbella w hossy wymaga innych proporcji niż w bessie. HMM automatyzuje tę decyzję.

---

#### MODULE 3: `tail_risk_hedging.py` — Systematyczne Zabezpieczenia Ogonowe

**Problem który rozwiązuje:** Mamy EVT i scenariusze — ale brak odpowiedzi: **co kupić żeby się zabezpieczyć?**

**Co implementuje:**
- **Put Option hedging calculator** — dla danej ekspozycji na akcje: ile OTM putów kupić, na jaki strike/termin, żeby osiągnąć docelowy max drawdown ≤ X%
- **Cost-benefit analysis zabezpieczeń** — roczny koszt hedgingu (theta decay) vs oczekiwana ochrona (CVaR reduction)
- **Praktyczne instrumenty** — kalkulacje dla: VIX calls, VIXY ETF, SPXU/SQQQ, złoto, obligacje długoterminowe (TLT), CHF, JPY, BTC jako hedge inflacji
- **Collar strategy calculator** — finansowanie zabezpieczeń przez sprzedaż opcji call (zero-cost collar)
- **Inflacja tail hedge** — TIPS, złoto, REIT, commodities: optymalny mix gdy inflacja >5%

**Model matematyczny:** Minimalizacja `CVaR(portfela z hedgiem)` przy ograniczeniu kosztu ≤ X% NAV/rok

**Dlaczego to ważne:** Wiedza o ryzyku bez narzędzi do jego redukcji to teoria. Ten moduł przechodzi do praktyki.

---

### 🟡 PRIORYTET 2 — WAŻNE (Lepsze rozumienie ryzyka i sytuacji)

---

#### MODULE 4: `macro_regime_clock.py` — Zegar Biznesowy (Investment Clock)

**Problem który rozwiązuje:** Control Center monitoruje wskaźniki, ale brak syntetycznego widoku **w jakim punkcie cyklu koniunkturalnego jesteśmy** i **co historycznie najlepiej działało**.

**Co implementuje:**
- **Merrill Lynch Investment Clock** — automatyczna klasyfikacja: Reflation / Recovery / Overheat / Stagflation na podstawie: wzrost PKB (CLI), inflacja (CPI), stopy procentowe
- **Asset class performance matrix** — dla każdej fazy zegara: średnia stopa zwrotu historyczna dla akcji, obligacji, surowców, cash, złota (dane od 1970)
- **Current clock position** — gdzie jesteśmy teraz + niepewność (bootstrap CI fazy)
- **Clock transition probability** — HMM: P(przejście do następnej fazy w ciągu 6M)
- **PLN-specific overlay** — dostosowanie do polskiej gospodarki: RPP decyzje, polskie obligacje, WIG sezonowość

**Dlaczego to ważne:** Zegar to jeden z najbardziej uznanych frameworków makro dla alokacji aktywów. Integruje się naturalnie z istniejącym Control Center.

---

#### MODULE 5: `liquidity_risk_analyzer.py` — Analiza Ryzyka Płynności

**Problem który rozwiązuje:** Nieobecna w projekcie — a kryzys płynności zabija portfele szybciej niż straty papierowe.

**Co implementuje:**
- **Bid-ask spread monitor** — pobiera spreads bid/ask dla aktywów z portfela; alert gdy spread staje się >5× normalny (sygnał kryzysu płynności)
- **Market depth scoring** — ocena głębokości rynku (volume * avg_price): ile możemy sprzedać bez 1% impact
- **Liquidity-adjusted VaR (LVaR)** — VaR + koszt likwidacji w warunkach kryzysu (Dowd 2005)
- **Redemption risk** — dla ETF/funduszy: analiza historycznych outflows w krizysach (ETF liquidity mismatch risk)
- **Liquidity ladder** — zestawienie: ile aktywów możemy spieniężyć w 1 dzień / 1 tydzień / 1 miesiąc bez >1% market impact
- **Fire-sale contagion** — gdy inne fundusze sprzedają te same aktywa (overlapping portfolio risk; Greenwood et al. 2015)

**Dlaczego to ważne:** W 2020 nawet „bezpieczne" ETF-y obligacyjne straciły płynność. Liquidty risk = survival risk.

---

#### MODULE 6: `concentration_risk_monitor.py` — Monitor Ryzyka Koncentracji

**Problem który rozwiązuje:** Portfel może wyglądać zdywersyfikowany (10 aktywów) ale być skoncentrowany faktycznie (wszystko koreluje z US tech).

**Co implementuje:**
- **Effective N (HHI)** — Herfindahl-Hirschman Index aktywów i sektorów (prawdziwa efektywna liczba niezależnych zakładów)
- **Factor concentration** — ile portfela faktycznie jest na ryzyku: Rynku / Momentum / Value / Low-Vol / Quality (Fama-French 5-factor)
- **Geographic concentration** — USD exposure, EUR, PLN, EM, single-country risk
- **Sector overlap** — szczególnie: crypto + tech + growth → triple exposure w risk-off
- **PCA concentration** — ile % wariancji wyjaśnia pierwszy PC? (jeśli >70% → brak dywersyfikacji mimo ilości aktywów)
- **PLN fx risk** — ile portfela denominowanego w obcych walutach, koszt hedgingu walutowego

**Dlaczego to ważne:** Główna iluzja dywersyfikacji to posiadanie wielu aktywów które faktycznie są jednym ryzykiem.

---

#### MODULE 7: `drawdown_recovery_analyzer.py` — Analiza Czasu Odrobienia Strat

**Problem który rozwiązuje:** Wiemy jaki jest max drawdown (mamy to w metrics.py) ale NIE WIEMY: jak długo trwa recovery i czy w ogóle zdążymy odrobić straty.

**Co implementuje:**
- **Underwater period analysis** — dla każdego historycznego drawdownu: czas trwania, czas do recovery, czy odrobiono przed emeryturą
- **Sequence-of-returns risk** — wizualizacja: jak kolejność złych lat wpływa na portfel emerytalny (ten sam CAGR, różna kolejność → ogromna różnica w wartości końcowej)
- **Recovery probability** — Monte Carlo: P(odrobienie strat w ciągu N lat) w zależności od reżimu rynkowego
- **Time-to-ruin analysis** — dla zadanego portfela y wypłat: kiedy portfel się wyczerpie przy różnych scenariuszach
- **Break-even return calculator** — po stracie X%: ile trzeba zarobić żeby wrócić do zera i ile to zajmie

**Dlaczego to ważne:** Strata 50% wymaga zysku 100% żeby wrócić do zera. Wizualizacja tego dramatycznie zmienia podejście do ryzyka.

---

### 🟢 PRIORYTET 3 — ZAAWANSOWANE (Powiększanie majątku)

---

#### MODULE 8: `smart_rebalancing_engine.py` — Inteligentny Rebalancing

**Problem który rozwiązuje:** Brak modułu do decydowania KIEDY i JAK rebalansować portfel minimalizując podatki i koszty transakcyjne.

**Co implementuje:**
- **Threshold-based rebalancing** — rebalansuj tylko gdy wagi odchyliły się >X% od celu (nie calendariowo — efektywniejsze podatkowo)
- **Tax-aware rebalancing** — priorytet rebalansowania przez nowe wpłaty; sell only losers (tax loss harvesting w Polsce: odliczenie strat od zysków Belka)
- **Transaction cost optimizer** — minimalizacja obrotu portfela (rebalance minimum trades do celu)
- **Volatility-based trigger** — częściej rebalansuj gdy vol wysoka (dryfowanie ryzyka), rzadziej gdy niska
- **Rebalancing backtester** — porównanie: Monthly vs Threshold vs Band (5% corridors) vs Buy-and-Hold  na historycznych danych
- **Optimal band calculator** — oblicza optymalne pasmo rebalansowania per aktywo minimalizując: koszty + Belka + tracking error

**Uwzględnia polskie prawo podatkowe:** Podatek Belki (19%), brak offsetu zysków/strat w tym samym roku (polska specyfika).

---

#### MODULE 9: `alternative_risk_premia.py` — Premie Alternatywne i Trend Following

**Problem który rozwiązuje:** Portfel Barbell bazuje na akcje+obligacje+crypto. Brak dostępu do strategii generujących *niezależne* od rynku zwroty.

**Co implementuje:**
- **CTA/Trend Following simulator** — Time Series Momentum na futures (Moskowitz, Ooi, Pedersen 2012): jak dodanie 10% MTUM/CTA ETF zmienia portfel
- **Carry strategy** — bond carry (długi koniec vs krótki), currency carry (AUD/JPY), commodity carry; Sharpe ratio i korelacja z portfelem
- **Value factor overlay** — systematyczny przechył portfela akcyjnego ku value (low P/B, P/E) na danych historycznych
- **Low Volatility anomaly** — backtesst: portfel min-vol vs cap-weighted (Frazzini & Pedersen BAB Factor)
- **Risk Parity overlay** — ile dodanie risk parity component poprawia Sharpe bez zwiększania drawdown
- **Korelacja z istniejącym portfelem** — każda premia ryzyka oceniana pod kątem: czy faktycznie dywersyfikuje?

**Dlaczego to ważne:** ARP strategie mają dokumentowane 30-letnie track record z niską korelacją do equity/bonds. To brakujący „trzeci koszyk".

---

#### MODULE 10: `wealth_protection_optimizer.py` — Optymalizator Ochrony Majątku

**Problem który rozwiązuje:** Brak całościowego narzędzia łączącego ochronę kapitału z celami życiowymi (emerytura, dzieci, dziedziczenie).

**Co implementuje:**
- **Goal-based investing framework** — podział majątku na cele z różnym horyzontem: bezpieczeństwo (1-3 lata), wzrost (3-10 lat), dziedzictwo (>10 lat); osobna optymalizacja każdego bucket
- **Liability-driven investing (LDI)** — dopasowanie aktywów do zobowiązań (rata kredytu, czesne dziecka, emerytura): minimalizacja ryzyka niedofinansowania celu
- **Real wealth preservation** — portfel budowany tak, żeby zachować siłę nabywczą po inflacji i podatkach (realna stopa zwrotu >0% po CPI + Belka)
- **Estate planning optimizer** — wpływ podatku od spadków, optymalna struktura portfela dla dziedziczenia
- **Human capital integration** — portfel powinien uwzględniać „ludzki kapitał" (praca = obligacja): młody pracownik z bezpieczną pracą może mieć więcej akcji w portfelu finansowym

**Dlaczego to ważne:** Zarządzanie ryzykiem bez celu to optymalizacja w próżni. Ten moduł łączy matematykę z życiowymi priorytetami.

---

#### MODULE 11: `sentiment_flow_tracker.py` — Tracker Przepływów i Nastrojów

**Problem który rozwiązuje:** Mamy sentiment w Control Center (Fear&Greed, VIX) ale brak głębszej analizy GDZIE płynie kapitał i jakiego sentymentu szukają profesjonaliści.

**Co implementuje:**
- **ETF fund flows** — tygodniowe przepływy do/z głównych ETF-ów: SPY, QQQ, TLT, GLD, IEF (sygnał instytucjonalny)
- **CFTC Commitment of Traders (CoT)** — pozycje Large Speculators vs Commercials na futures (S&P, Gold, Oil, EUR/USD); ekstremalny positioning = contrarian signal
- **Options put/call ratio tracking** — dla SPY/QQQ: 20-dniowa MA P/C ratio vs signal
- **Insider transactions monitor** — filing SEC Form 4: gdy insiders masowo kupują → bullish signal (Seyhun 1998)
- **Short interest tracker** — top short positions, short squeeze risk (days-to-cover ratio)
- **Smart money vs dumb money** — composite indicator: gdy rozbieżność duża → contrarian opportunity

**Dlaczego to ważne:** Rynek jest grą między uczestnikami. Wiedza kto i co robi (instytucje, insiderzy, CoT) daje realną przewagę informacyjną.

---

#### MODULE 12: `tax_optimizer_pl.py` — Optymalizator Podatkowy (Polska)

**Problem który rozwiązuje:** Projekt nie uwzględnia systematycznej optymalizacji podatkowej poza prostym Tax Belka w symulatorze.

**Co implementuje:**
- **Tax Loss Harvesting automatyczny** — identyfikacja pozycji ze stratą, która może być zrealizowana i odliczona od zysków Belka; zastąpienie similar asset (wash-sale risk minimization)
- **Optymalna kolejność sprzedaży** — FIFO vs LIFO vs specific lot: która kolejność daje najniższy podatek Belka w danym roku
- **IKE/IKZE optimizer** — kalkulator: ile zaoszczędzić na podatkach przez maksymalne wypełnienie IKE (ulga 19% Belka) i IKZE (odliczenie od PIT)
- **Dywidenda vs growth stocks** — po uwzględnieniu Belki: kiedy opłaca się dividend reinvestment zamiast dywidend (podatek przy wypłacie vs przy sprzedaży)
- **Walutowy PIT** — PLN/USD: jak księgować zyski z aktywów zagranicznych (różnice kursowe, metoda FIFO FX)
- **Roczny raport PIT-8C simulator** — szacowanie podatku na koniec roku przed jego złożeniem

**Dlaczego to ważne:** Różnica między gross return a net return (po podatkach) to często 25-40% zysku. Optymalizacja podatkowa to bezryzykowny zysk.

---

## 📊 MAPA PRIORYTETÓW

```
OCHRONA KAPITAŁU               WZROST KAPITAŁU
         │                              │
   Priorytet 1                   Priorytet 3
─────────────────────────────────────────────
P1: Portfolio Health Monitor    P3: Smart Rebalancing Engine
P1: Regime Adaptive Allocation  P3: Alternative Risk Premia  
P1: Tail Risk Hedging           P3: Wealth Protection Optimizer
                                P3: Sentiment Flow Tracker
   Priorytet 2                  P3: Tax Optimizer PL
─────────────────────────────────────────────
P2: Macro Regime Clock
P2: Liquidity Risk Analyzer
P2: Concentration Risk Monitor
P2: Drawdown Recovery Analyzer
```

---

## ⚡ SZACOWANY NAKŁAD PRACY (Kompleksowość)

| Moduł | Złożoność | Szacowany czas |
|-------|-----------|----------------|
| Portfolio Health Monitor | Średnia | 2-3 dni |
| Regime Adaptive Allocation (HMM) | Wysoka | 3-4 dni |
| Tail Risk Hedging | Wysoka | 3-4 dni |
| Macro Regime Clock | Średnia | 2-3 dni |
| Liquidity Risk Analyzer | Średnia | 2 dni |
| Concentration Risk Monitor | Niska | 1-2 dni |
| Drawdown Recovery Analyzer | Niska | 1-2 dni |
| Smart Rebalancing Engine | Średnia | 2-3 dni |
| Alternative Risk Premia | Wysoka | 4-5 dni |
| Wealth Protection Optimizer | Wysoka | 3-4 dni |
| Sentiment Flow Tracker | Średnia | 2-3 dni |
| Tax Optimizer PL | Średnia | 2-3 dni |

---

## 🎯 REKOMENDACJA KOLEJNOŚCI IMPLEMENTACJI

**Faza 1 (Ochrona — zrób najpierw):**
1. `portfolio_health_monitor.py` — podstawa bez której reszta to teoria
2. `concentration_risk_monitor.py` — szybki win, odkrywa ukryte ryzyka
3. `drawdown_recovery_analyzer.py` — zmienia sposób myślenia o ryzyku
4. `macro_regime_clock.py` — natural extension Control Center

**Faza 2 (Zaawansowane zarządzanie ryzykiem):**
5. `regime_adaptive_allocation.py` — wymaga HMM i danych historycznych
6. `liquidity_risk_analyzer.py` — ważne szczególnie dla większych portfeli
7. `tail_risk_hedging.py` — wymaga integracji danych opcyjnych
8. `tax_optimizer_pl.py` — bezryzykowny zysk dla polskiego inwestora

**Faza 3 (Powiększanie majątku):**
9. `smart_rebalancing_engine.py` — wymaga portfela użytkownika
10. `sentiment_flow_tracker.py` — dane CoT, ETF flows
11. `alternative_risk_premia.py` — najbardziej zaawansowany matematycznie
12. `wealth_protection_optimizer.py` — wymaga danych personalnych użytkownika
