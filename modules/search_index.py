"""
modules/search_index.py
=======================
Generator indeksu wyszukiwania dla Spotlight Command Palette v2.

Indeksuje:
  - Strony/Moduły (pages/*.py) — tytuły, tagi, kategorie
  - Wskaźniki makro (VIX, TED, GEX, Gold, ...) — z docstringów get_help_* z app.py
  - Metody quantitative (Monte Carlo, GARCH, Black-Litterman, ...)
  - Ustawienia portfela (GlobalPortfolio fields)
  - Akcje systemowe (szybkie komendy)
  - Tagi z i18n.py (oba języki jako słowa kluczowe)

Użycie:
    from modules.search_index import build_search_index
    index = build_search_index()   # lista dict, @st.cache_resource
"""
from __future__ import annotations
import re
import os
import ast
from typing import List, Dict, Any


# -- Mapa: page_file -> URL nawigacyjny w Streamlit
# Sidebar ma linki grupowe. Kazda podstrona mapuje sie na URL grupy.
PAGE_TO_NAV_URL = {
    'app.py':                             '/',
    'pages/0_Globalne_Ustawienia.py':     '/Globalne_Ustawienia',
    'pages/28_Doradca.py':                '/Doradca',
    'pages/2_Skaner.py':                  '/module_makro',
    'pages/11_Regime_Clock.py':           '/module_makro',
    'pages/12_Regime_Allocation.py':      '/module_makro',
    'pages/26_Recession_Nowcasting.py':   '/module_makro',
    'pages/40_FX_Dynamics.py':            '/module_makro',
    'pages/41_Commodity_Cycles.py':       '/module_makro',
    'pages/8_Health_Monitor.py':          '/module_portfel',
    'pages/16_Rebalancing.py':            '/module_portfel',
    'pages/15_Tax_Optimizer.py':          '/module_portfel',
    'pages/19_Wealth_Optimizer.py':       '/module_portfel',
    'pages/3_Stress_Test.py':             '/module_ryzyko',
    'pages/9_Concentration_Risk.py':      '/module_ryzyko',
    'pages/13_Liquidity_Risk.py':         '/module_ryzyko',
    'pages/5_EVT_Analysis.py':            '/module_ryzyko',
    'pages/27_Systemic_Risk.py':          '/module_ryzyko',
    'pages/10_Drawdown_Recovery.py':      '/module_ryzyko',
    'pages/14_Tail_Hedging.py':           '/module_ryzyko',
    'pages/32_Inzynieria_Opcji.py':       '/module_ryzyko',
    'pages/36_Fixed_Income_v2.py':        '/module_ryzyko',
    'pages/36_Portfolio_Insurance.py':    '/module_ryzyko',
    'pages/6_BL_Dashboard.py':            '/module_quant',
    'pages/24_HERC_Portfolio.py':         '/module_quant',
    'pages/7_DCC_Dashboard.py':           '/module_quant',
    'pages/22_Factor_Analysis.py':        '/module_quant',
    'pages/33_Sieci_Przyczynowe.py':      '/module_quant',
    'pages/48_Stochastic_Errors.py':      '/module_quant',
    'pages/18_Alt_Risk_Premia.py':        '/module_quant',
    'pages/35_Stochastic_Calculus.py':    '/module_quant',
    'pages/35_Vol_Surface.py':            '/module_quant',
    'pages/37_Covariance_Shrinkage.py':   '/module_quant',
    'pages/38_Market_Impact.py':          '/module_quant',
    'pages/42_Econophysics.py':           '/module_quant',
    'pages/43_Entropy_Finance.py':        '/module_quant',
    'pages/47_Metacognition.py':          '/module_meta',
    'pages/4_Emerytura.py':               '/module_majatku',
    'pages/34_Przewaga_Informacyjna.py':  '/module_meta',
    'pages/29_Kalkulator_Bayesa.py':      '/module_meta',
    'pages/31_Asymetria_Informacji.py':   '/module_meta',
    'pages/30_Teoria_Gier.py':            '/module_meta',
    'pages/49_Chaos_Deterministyczny.py': '/module_meta',
    'pages/45_Epigenetics_Investing.py':  '/module_meta',
    'pages/46_4E_Cognition.py':           '/module_meta',
    'pages/1_Symulator.py':               '/module_aktywne',
    'pages/21_Day_Trading.py':            '/module_aktywne',
    'pages/17_Sentiment_Flow.py':         '/module_aktywne',
    'pages/23_Walk_Forward.py':           '/module_aktywne',
    'pages/20_Life_OS.py':                '/module_aktywne',
}

# ─── Mapa stron projektu ───────────────────────────────────────────────────────
# Format: (file_path, title_pl, title_en, icon, category, tags)
PAGES_MAP = [
    # Strona główna
    ("app.py",
     "Control Center — Dashboard Główny",
     "Control Center — Main Dashboard",
     "🌐",
     "⚙️ System",
     ["dashboard", "control center", "makro", "macro", "vix", "regime score",
      "business cycle", "wskaźniki", "indicators", "główna", "home"]),

    # Strony
    ("pages/0_Globalne_Ustawienia.py",
     "Globalne Ustawienia Portfela",
     "Global Portfolio Settings",
     "🌐",
     "⚙️ System",
     ["ustawienia", "settings", "portfel", "portfolio", "alokacja", "allocation",
      "kelly", "bezpieczna", "safe sleeve", "ryzykowna", "risky sleeve",
      "obligacje", "bonds", "profil", "profile", "heartbeat", "waluta", "currency"]),

    ("pages/1_Symulator.py",
     "Symulator Monte Carlo / Barbell",
     "Monte Carlo / Barbell Simulator",
     "⚖️",
     "🛡️ Portfel i Ochrona",
     ["symulator", "simulator", "monte carlo", "barbell", "backtest",
      "garch", "kelly", "lévy", "fbm", "hurst", "sobol", "kopuła", "copula",
      "skok", "jump diffusion", "merton", "alokacja", "allocation",
      "rebalancing", "fx risk", "waluta", "cagr", "sharpe", "sortino",
      "var", "cvar", "drawdown", "symulacja", "simulation"]),

    ("pages/2_Skaner.py",
     "Skaner Rynku — Antykruchość",
     "Market Scanner — Antifragility",
     "🔍",
     "📊 Analiza Rynkowa",
     ["skaner", "scanner", "antykruchość", "antifragility", "barbell score",
      "ranking", "korelacja", "correlation", "dendrogram", "ogony", "tails",
      "log-log", "nowcast", "barbell nowcast", "etf", "akcje", "stocks",
      "krypto", "crypto", "watchlist"]),

    ("pages/3_Stress_Test.py",
     "Stress Test — Historyczne Kryzysy",
     "Stress Test — Historical Crises",
     "⚡",
     "🛡️ Portfel i Ochrona",
     ["stress test", "kryzys", "crisis", "crash", "2008", "covid", "dow jones",
      "drawdown", "reverse stress test", "punkt pęknięcia", "break point",
      "szok", "shock", "historyczny", "historical", "copula", "zaraza", "contagion"]),

    ("pages/4_Emerytura.py",
     "Emerytura / FIRE — Planer Emerytalny",
     "Retirement / FIRE Planner",
     "🏖️",
     "🛡️ Portfel i Ochrona",
     ["emerytura", "retirement", "fire", "finanse osobiste", "personal finance",
      "swr", "safe withdrawal rate", "ike", "ikze", "ppk", "zus",
      "income", "dochód", "inflacja", "inflation", "glide path",
      "spending smile", "merton lifecycle", "dekumulacja"]),

    ("pages/5_EVT_Analysis.py",
     "EVT — Analiza Wartości Ekstremalnych",
     "EVT — Extreme Value Analysis",
     "📐",
     "📊 Analiza Rynkowa",
     ["evt", "extreme value theory", "gumbel", "frechet", "weibull",
      "tail risk", "ogon ryzyka", "var", "cvar", "pot", "peak over threshold",
      "generalized pareto", "gev", "black swan", "czarny łabędź"]),

    ("pages/6_BL_Dashboard.py",
     "Black-Litterman AI Dashboard",
     "Black-Litterman AI Dashboard",
     "🎯",
     "📊 Analiza Rynkowa",
     ["black-litterman", "bl", "bayesian", "bayesowski", "prior", "posterior",
      "views", "widoki", "market equilibrium", "reverse optimization",
      "portfel ai", "ai portfolio", "covariance", "kowariancja"]),

    ("pages/7_DCC_Dashboard.py",
     "DCC-GARCH — Dynamiczne Korelacje",
     "DCC-GARCH — Dynamic Correlations",
     "🔗",
     "📊 Analiza Rynkowa",
     ["dcc", "garch", "dynamiczne korelacje", "dynamic correlations",
      "volatility clustering", "klastrowanie zmienności", "spillover",
      "time-varying", "zmienna w czasie"]),

    ("pages/8_Health_Monitor.py",
     "Portfolio Health Monitor",
     "Portfolio Health Monitor",
     "🏥",
     "🛡️ Portfel i Ochrona",
     ["health monitor", "zdrowie portfela", "drawdown alert", "alert drawdown",
      "volatility spike", "spike zmienności", "korelacja", "correlation",
      "ath", "equity", "monitoring", "rebalancing trigger"]),

    ("pages/9_Concentration_Risk.py",
     "Ryzyko Koncentracji Portfela",
     "Portfolio Concentration Risk",
     "🎯",
     "📊 Analiza Rynkowa",
     ["koncentracja", "concentration risk", "herfindahl", "hhi",
      "diversification", "dywersyfikacja", "sector", "sektor",
      "geographic", "geograficzny"]),

    ("pages/10_Drawdown_Recovery.py",
     "Drawdown Recovery Analyzer",
     "Drawdown Recovery Analyzer",
     "📉",
     "🛡️ Portfel i Ochrona",
     ["drawdown", "recovery", "odrabianie strat", "underwater",
      "pain index", "ulcer index", "calmar", "max drawdown"]),

    ("pages/11_Regime_Clock.py",
     "Zegar Reżimów Makro",
     "Macro Regime Clock",
     "🕐",
     "📊 Analiza Rynkowa",
     ["regime clock", "zegar reżimów", "cykl koniunkturalny", "business cycle",
      "pmi", "inflacja", "inflation", "yield curve", "krzywa dochodowości",
      "expansion", "ekspansja", "recession", "recesja", "fidelity clock"]),

    ("pages/12_Regime_Allocation.py",
     "Regime-Adaptive Allocation",
     "Regime-Adaptive Allocation",
     "🎯",
     "📊 Analiza Rynkowa",
     ["regime adaptive", "alokacja adaptacyjna", "dynamic allocation",
      "dynamiczna alokacja", "bull", "bear", "crisis", "reżim",
      "momentum", "mean reversion"]),

    ("pages/13_Liquidity_Risk.py",
     "Liquidity Risk Analyzer",
     "Liquidity Risk Analyzer",
     "💧",
     "📊 Analiza Rynkowa",
     ["liquidity risk", "ryzyko płynności", "bid-ask spread",
      "volume", "wolumen", "market impact", "illiquidity",
      "amihud", "kyle lambda", "płynność"]),

    ("pages/14_Tail_Hedging.py",
     "Tail Risk Hedging — Ochrona Ogonów",
     "Tail Risk Hedging",
     "🛡️",
     "🛡️ Portfel i Ochrona",
     ["tail hedging", "tail risk", "ochrona ogonów", "opcje", "options",
      "put", "collar", "hedging", "zabezpieczenie", "spx", "vix",
      "black swan protection", "antykruchość"]),

    ("pages/15_Tax_Optimizer.py",
     "Tax Optimizer PL — Optymalizacja Podatkowa",
     "Tax Optimizer PL",
     "🧾",
     "🛡️ Portfel i Ochrona",
     ["tax optimizer", "podatki", "taxes", "belka", "capital gains",
      "zyski kapitałowe", "ike", "ikze", "ppk", "optymalizacja podatkowa",
      "tax loss harvesting", "loss harvesting"]),

    ("pages/16_Rebalancing.py",
     "Smart Rebalancing Engine",
     "Smart Rebalancing Engine",
     "⚖️",
     "🛡️ Portfel i Ochrona",
     ["rebalancing", "rebalansowanie", "threshold", "progi",
      "calendar", "kalendaryczne", "tactical", "taktyczne",
      "drift", "odchylenie", "shannon demon"]),

    ("pages/17_Sentiment_Flow.py",
     "Sentiment Flow Tracker — NLP",
     "Sentiment Flow Tracker — NLP",
     "🌊",
     "📊 Analiza Rynkowa",
     ["sentiment", "sentyment", "nlp", "vader", "natural language processing",
      "fear greed", "strach chciwość", "news", "wiadomości",
      "social media", "flow", "przepływ"]),

    ("pages/18_Alt_Risk_Premia.py",
     "Alternative Risk Premia",
     "Alternative Risk Premia",
     "🎲",
     "📊 Analiza Rynkowa",
     ["alternative risk premia", "risk premia", "momentum", "value",
      "carry", "low volatility", "quality", "size", "factor", "faktor",
      "smart beta", "premia za ryzyko"]),

    ("pages/19_Wealth_Optimizer.py",
     "Wealth Protection Optimizer",
     "Wealth Protection Optimizer",
     "🏰",
     "🛡️ Portfel i Ochrona",
     ["wealth optimizer", "ochrona majątku", "goal-based investing",
      "human capital", "kapitał ludzki", "ldi", "liability driven",
      "funding ratio", "hrp", "hierarchical risk parity",
      "life goals", "cele życiowe", "merton lifecycle"]),

    ("pages/20_Life_OS.py",
     "Life OS — System Operacyjny Życia",
     "Life OS — Life Operating System",
     "🧠",
     "🧬 Osobiste",
     ["life os", "system życia", "osobiste finanse", "personal finance",
      "habit tracking", "nawyki", "goals", "cele", "health", "zdrowie",
      "productivity", "produktywność", "biohacking"]),

    ("pages/21_Day_Trading.py",
     "Day Trading Module",
     "Day Trading Module",
     "📊",
     "📊 Analiza Rynkowa",
     ["day trading", "trading", "technical analysis", "analiza techniczna",
      "rsi", "macd", "bollinger", "sma", "ema", "pairs trading",
      "cointegration", "kointegracja", "intraday"]),

    ("pages/22_Factor_Analysis.py",
     "Factor Zoo & PCA — Analiza Faktorów",
     "Factor Zoo & PCA — Factor Analysis",
     "🔬",
     "📊 Analiza Rynkowa",
     ["factor analysis", "analiza faktorów", "pca", "principal component",
      "autoencoder", "fama french", "carhart", "momentum factor",
      "value factor", "size factor", "quality factor", "zoo faktorów"]),

    ("pages/23_Walk_Forward.py",
     "Walk-Forward CPCV Walidacja",
     "Walk-Forward CPCV Validation",
     "🔄",
     "📊 Analiza Rynkowa",
     ["walk-forward", "cpcv", "cross-validation", "cross validation",
      "walidacja", "backtest bias", "overfitting", "przeuczenie",
      "purged k-fold", "combinatorial"]),

    ("pages/24_HERC_Portfolio.py",
     "HERC — Hierarchical Equal Risk Contribution",
     "HERC — Hierarchical Equal Risk Contribution",
     "🌳",
     "🛡️ Portfel i Ochrona",
     ["herc", "hierarchical equal risk contribution",
      "risk parity", "równy wkład ryzyka", "dendrogram",
      "clustering", "klasteryzacja", "min variance"]),

    ("pages/26_Recession_Nowcasting.py",
     "Recession Nowcasting — Prognoza Recesji",
     "Recession Nowcasting",
     "📡",
     "📊 Analiza Rynkowa",
     ["recession nowcasting", "nowcasting recesji", "yield curve",
      "krzywa dochodowości", "fred", "pmi", "leading indicators",
      "wskaźniki wyprzedzające", "probability recession",
      "prawdopodobieństwo recesji"]),

    ("pages/27_Systemic_Risk.py",
     "Systemic Risk & CoVaR",
     "Systemic Risk & CoVaR",
     "⚠️",
     "📊 Analiza Rynkowa",
     ["systemic risk", "ryzyko systemowe", "covar", "srisk",
      "too big to fail", "contagion", "zaraza finansowa",
      "network risk", "ryzyko sieciowe"]),

    ("pages/28_Doradca.py",
     "AI Doradca Inwestycyjny",
     "AI Investment Advisor",
     "🤖",
     "⚙️ System",
     ["doradca", "advisor", "ai", "sztuczna inteligencja",
      "rekomendacje", "recommendations", "portfolio advice",
      "robo-advisor", "chatbot"]),

    ("pages/29_Kalkulator_Bayesa.py",
     "Kalkulator Bayesowski",
     "Bayesian Calculator",
     "🎲",
     "📊 Analiza Rynkowa",
     ["bayesian", "bayesowski", "kalkulator bayesa", "bayesian calculator",
      "prior", "posterior", "likelihood", "prawdopodobieństwo bayesowskie",
      "conditional probability", "prawdopodobieństwo warunkowe",
      "bayes theorem", "twierdzenie bayesa"]),

    ("pages/30_Teoria_Gier.py",
     "Teoria Gier — Game Theory",
     "Game Theory",
     "♟️",
     "📊 Analiza Rynkowa",
     ["teoria gier", "game theory", "nash equilibrium", "równowaga nasha",
      "prisoner dilemma", "dylemat więźnia", "dominant strategy",
      "minimax", "zero sum", "cooperative"]),

    ("pages/31_Asymetria_Informacji.py",
     "Asymetria Informacji — Information Theory",
     "Information Asymmetry",
     "🔬",
     "📊 Analiza Rynkowa",
     ["asymetria informacji", "information asymmetry", "insider trading",
      "adverse selection", "moral hazard", "signaling", "sygnalizacja",
      "information theory", "teoria informacji", "entropy"]),

    ("pages/32_Inzynieria_Opcji.py",
     "Inżynieria Opcji — Options Engineering",
     "Options Engineering",
     "⚗️",
     "📊 Analiza Rynkowa",
     ["opcje", "options", "black-scholes", "delta", "gamma", "vega", "theta",
      "put", "call", "collar", "straddle", "iron condor",
      "greeks", "grecy", "implied volatility", "zmienność implikowana"]),

    ("pages/33_Sieci_Przyczynowe.py",
     "Sieci Przyczynowe — Causal Networks",
     "Causal Networks",
     "🕸️",
     "📊 Analiza Rynkowa",
     ["sieci przyczynowe", "causal networks", "causality", "przyczynowość",
      "granger causality", "dag", "directed acyclic graph",
      "bayesian network", "sieć bayesowska"]),

    ("pages/34_Przewaga_Informacyjna.py",
     "Przewaga Informacyjna",
     "Information Edge",
     "🔍",
     "📊 Analiza Rynkowa",
     ["przewaga informacyjna", "information edge", "alpha",
      "market efficiency", "efektywność rynku", "kelly criterion",
      "edge ratio", "batting average"]),

    ("pages/35_Stochastic_Calculus.py",
     "Stochastic Calculus — Rachunek Stochastyczny",
     "Stochastic Calculus",
     "∫",
     "📊 Analiza Rynkowa",
     ["stochastic calculus", "rachunek stochastyczny", "ito",
      "brownian motion", "ruch browna", "sde",
      "stochastic differential equation", "geometric brownian motion",
      "gbm", "wiener process"]),

    ("pages/35_Vol_Surface.py",
     "Volatility Surface — Powierzchnia Zmienności",
     "Volatility Surface",
     "📈",
     "📊 Analiza Rynkowa",
     ["volatility surface", "powierzchnia zmienności", "vol surface",
      "implied volatility", "zmienność implikowana", "smile",
      "skew", "term structure", "vix term structure", "sabr"]),

    ("pages/36_Fixed_Income_v2.py",
     "Fixed Income — Obligacje",
     "Fixed Income",
     "📊",
     "🛡️ Portfel i Ochrona",
     ["fixed income", "obligacje", "bonds", "duration", "convexity",
      "yield to maturity", "ytm", "nss", "nelson siegel",
      "treasury", "skarbowe", "corporate bonds", "obligacje korporacyjne"]),

    ("pages/36_Portfolio_Insurance.py",
     "Portfolio Insurance — Ubezpieczenie Portfela",
     "Portfolio Insurance",
     "🛡️",
     "🛡️ Portfel i Ochrona",
     ["portfolio insurance", "ubezpieczenie portfela", "cppi",
      "constant proportion portfolio insurance", "obpi",
      "option based protection", "floor", "podłoga"]),

    ("pages/37_Covariance_Shrinkage.py",
     "Covariance Shrinkage — Ledoit-Wolf",
     "Covariance Shrinkage",
     "📐",
     "📊 Analiza Rynkowa",
     ["covariance shrinkage", "ledoit wolf", "kowariancja",
      "shrinkage", "kurczenie macierzy", "regularization",
      "regularyzacja", "efficient frontier", "granica efektywna"]),

    ("pages/38_Market_Impact.py",
     "Market Impact — Wpływ na Rynek",
     "Market Impact",
     "💥",
     "📊 Analiza Rynkowa",
     ["market impact", "wpływ na rynek", "price impact", "slippage",
      "execution cost", "koszty transakcyjne", "almgren chriss",
      "optimal execution", "optimal trading"]),

    ("pages/40_FX_Dynamics.py",
     "FX Dynamics — Dynamika Walut",
     "FX Dynamics",
     "💱",
     "📊 Analiza Rynkowa",
     ["fx dynamics", "forex", "waluta", "currency", "usd pln",
      "eur usd", "gbp", "carry trade", "purchasing power parity",
      "ppp", "uncovered interest parity", "uip"]),

    ("pages/41_Commodity_Cycles.py",
     "Commodity Cycles — Cykle Surowcowe",
     "Commodity Cycles",
     "⛏️",
     "📊 Analiza Rynkowa",
     ["commodity cycles", "cykle surowcowe", "commodities", "surowce",
      "copper", "miedź", "gold", "złoto", "oil", "ropa",
      "super cycle", "supercykl", "energy", "energia"]),

    ("pages/42_Econophysics.py",
     "Econophysics — Ekonofizyka",
     "Econophysics",
     "⚛️",
     "📊 Analiza Rynkowa",
     ["econophysics", "ekonofizyka", "power law", "prawo potęgowe",
      "scale invariance", "self-organized criticality", "zipf law",
      "log-periodic", "log periodyczny", "sornette"]),

    ("pages/43_Entropy_Finance.py",
     "Entropy Finance — Entropia Rynków",
     "Entropy Finance",
     "🌀",
     "📊 Analiza Rynkowa",
     ["entropy finance", "entropia finansów", "entropy", "entropia",
      "information entropy", "max entropy", "maksymalna entropia",
      "shannon entropy", "thermodynamics", "termostatystyka"]),

    ("pages/47_Metacognition.py",
     "Metacognition — Psychologia Inwestowania",
     "Metacognition — Investor Psychology",
     "🧠",
     "🧬 Osobiste",
     ["metacognition", "metakognicja", "investor psychology",
      "psychologia inwestowania", "behavioral finance", "finanse behawioralne",
      "cognitive bias", "błędy poznawcze", "loss aversion",
      "awersja do straty", "overconfidence"]),

    ("pages/48_Stochastic_Errors.py",
     "Stochastic Errors — Błędy Stochastyczne",
     "Stochastic Errors",
     "📉",
     "📊 Analiza Rynkowa",
     ["stochastic errors", "błędy stochastyczne", "noise", "szum",
      "signal to noise", "ratio signal noise", "estimation error",
      "błąd estymacji", "model risk", "ryzyko modelu"]),

    ("pages/49_Chaos_Deterministyczny.py",
     "Chaos Deterministyczny",
     "Deterministic Chaos",
     "🌀",
     "📊 Analiza Rynkowa",
     ["chaos", "deterministyczny", "deterministic chaos", "lorenz attractor",
      "atraktor lorenza", "lyapunov exponent", "wykładnik lyapunova",
      "bifurcation", "bifurkacja", "fractals", "fraktale",
      "mandelbrot", "sensitive dependence"]),
]

# ─── Wskaźniki makro (z app.py get_help_* functions) ──────────────────────────
INDICATORS = [
    {
        "id": "ind_vix",
        "title": "VIX 1M — Indeks Strachu",
        "subtitle": "CBOE Volatility Index. Implikowana zmienność 30-dniowa opcji na S&P 500. Znany jako 'Indeks Strachu'.",
        "category": "📊 Wskaźnik Makro",
        "icon": "📈",
        "tags": ["vix", "volatility", "zmienność", "strach", "fear", "cboe",
                 "implied volatility", "s&p500", "risk", "ryzyko", "indeks strachu"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_vts",
        "title": "VIX Term Structure (Backwardation/Contango)",
        "subtitle": "Porównanie implikowanej zmienności krótkoterminowej (VIX 1M) z średnioterminową (VXMT 3M).",
        "category": "📊 Wskaźnik Makro",
        "icon": "📉",
        "tags": ["vix term structure", "backwardation", "contango", "vxmt",
                 "term structure", "volatility curve", "stres", "stress"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_ted",
        "title": "TED Spread — Zaufanie Bankowe",
        "subtitle": "Różnica między LIBOR 3M (ryzyko bankowe) a T-Bill 3M. Mierzy zaufanie na rynku międzybankowym.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🏦",
        "tags": ["ted spread", "libor", "t-bill", "treasury bill", "interbank",
                 "kredyt", "credit", "zaufanie bankowe", "banking stress",
                 "płynność", "liquidity"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_fci",
        "title": "STLFSI — Financial Stress Index",
        "subtitle": "St. Louis Fed Financial Stress Index. Agreguje 18 wskaźników rynkowych. Wartość 0 = norma historyczna.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🏛️",
        "tags": ["stlfsi", "financial stress index", "st louis fed", "fred",
                 "stress finansowy", "financial stress", "warunki finansowe",
                 "financial conditions", "systemic stress"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_yield_curve",
        "title": "Krzywa Dochodowości (10Y-3M)",
        "subtitle": "Różnica rentowności 10-letnich i 3-miesięcznych obligacji USA. Najlepszy predyktor recesji (trafność ~77%).",
        "category": "📊 Wskaźnik Makro",
        "icon": "📊",
        "tags": ["yield curve", "krzywa dochodowości", "inwersja", "inversion",
                 "10y 3m", "treasury", "recesja", "recession", "obligacje",
                 "bonds", "stopy procentowe", "interest rates"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_real_yield",
        "title": "Real 10Y Yield (TIPS)",
        "subtitle": "Rentowność 10-letnich obligacji USA skorygowana o inflację (indeksowane TIPS). Realny koszt pieniądza.",
        "category": "📊 Wskaźnik Makro",
        "icon": "📈",
        "tags": ["real yield", "tips", "real interest rate", "realna stopa",
                 "inflacja", "inflation", "breaks even", "breakeven inflation",
                 "pieniądz", "money cost"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_gex",
        "title": "Dark Pool GEX — Gamma Exposure",
        "subtitle": "Gamma Exposure dealers opcyjnych na rynek SPY/SPX (mld USD). Dodatni GEX stabilizuje, ujemny amplifikuje zmienność.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🎯",
        "tags": ["gex", "gamma exposure", "dark pool", "dealer gamma",
                 "short gamma", "long gamma", "spx", "spy", "options flow",
                 "market maker", "zmienność", "volatility stabilizer"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_hy_spread",
        "title": "HY Spread (OAS) — High Yield",
        "subtitle": "High Yield Option-Adjusted Spread: różnica rentowności obligacji śmieciowych vs Treasuries.",
        "category": "📊 Wskaźnik Makro",
        "icon": "💳",
        "tags": ["hy spread", "high yield", "junk bonds", "obligacje śmieciowe",
                 "credit spread", "credit risk", "ryzyko kredytowe",
                 "oas", "option adjusted spread", "kryzys kredytowy"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_credit_spread",
        "title": "Credit Spread (BAA-AAA)",
        "subtitle": "Różnica rentowności obligacji korporacyjnych BAA vs AAA. Mierzy premię za ryzyko upadłości.",
        "category": "📊 Wskaźnik Makro",
        "icon": "📊",
        "tags": ["credit spread", "baa aaa", "corporate bonds", "obligacje korporacyjne",
                 "investment grade", "default risk", "ryzyko default",
                 "premia za ryzyko", "spread"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_m2",
        "title": "M2 Money Supply YoY",
        "subtitle": "Roczna zmiana agregatu M2 (gotówka + depozyty + fundusze). Silny wzrost M2 napędza ceny aktywów.",
        "category": "📊 Wskaźnik Makro",
        "icon": "💵",
        "tags": ["m2", "money supply", "podaż pieniądza", "monetarny",
                 "monetary", "inflation", "inflacja", "friedman",
                 "liquidity", "płynność", "quantitative easing", "qe"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_copper",
        "title": "Dr. Copper — Miedź (Barometr Wzrostu)",
        "subtitle": "Miedź ($/lb) jako barometr globalnego wzrostu przemysłowego. Barometr PKB.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🔧",
        "tags": ["copper", "miedź", "dr copper", "commodities", "surowce",
                 "gdp barometer", "leading indicator", "wskaźnik wyprzedzający",
                 "przemysłowy", "industrial"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_cuau",
        "title": "Copper/Gold Ratio — Risk-On/Off Sygnał",
        "subtitle": "Stosunek ceny miedzi do złota. Rosnący ratio = rynki preferują wzrost (risk-on). Koreluje z rentownościami 10Y.",
        "category": "📊 Wskaźnik Makro",
        "icon": "⚖️",
        "tags": ["copper gold ratio", "cu au", "risk on risk off", "risk on",
                 "risk off", "miedź złoto", "gundlach", "leading indicator",
                 "10y yield", "obligacje"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_sentiment",
        "title": "NLP Sentiment — Analiza Sentymentu Mediów",
        "subtitle": "Analiza sentymentu globalnych nagłówków finansowych przez model VADER. Wartości: -1.0 do +1.0.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🧠",
        "tags": ["sentiment", "sentyment", "nlp", "vader", "natural language processing",
                 "news analysis", "analiza mediów", "fear greed", "media finansowe",
                 "compound sentiment", "ai sentiment"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_breadth",
        "title": "Market Breadth — Szerokość Rynku",
        "subtitle": "Momentum szerokości rynku: zwrot RSP (Equal-Weight S&P500) minus SPY (Cap-Weight).",
        "category": "📊 Wskaźnik Makro",
        "icon": "📊",
        "tags": ["market breadth", "szerokość rynku", "equal weight", "rsp spy",
                 "narrow rally", "wąska hossa", "advance decline", "breadth momentum"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_gold",
        "title": "Złoto — Safe Haven",
        "subtitle": "Gold Spot $/oz. Klasyczne aktywo safe-haven. Rośnie przy strachu, inflacji i słabym dolarze.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🥇",
        "tags": ["gold", "złoto", "safe haven", "bezpieczna przystań",
                 "precious metals", "metale szlachetne", "inflation hedge",
                 "risk off", "store of value"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_dxy",
        "title": "USD Index (DXY) — Siła Dolara",
        "subtitle": "Siła dolara względem koszyka 6 walut (EUR, JPY, GBP, CAD, SEK, CHF).",
        "category": "📊 Wskaźnik Makro",
        "icon": "💵",
        "tags": ["usd", "dxy", "dollar index", "indeks dolara", "dollar strength",
                 "siła dolara", "fx", "forex", "emerging markets", "rynki wschodzące"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_oil",
        "title": "Ropa Naftowa (WTI/Brent)",
        "subtitle": "Cena ropy wpływa bezpośrednio na inflację CPI i koszty produkcji.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🛢️",
        "tags": ["oil", "ropa", "crude oil", "wti", "brent", "energy",
                 "energia", "opec", "inflation", "inflacja", "stagflacja"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_baltic",
        "title": "Baltic Dry Index — Handel Globalny",
        "subtitle": "Indeks kosztu frachtu morskiego suchego ładunku. Puls globalnego handlu surowcami.",
        "category": "📊 Wskaźnik Makro",
        "icon": "🚢",
        "tags": ["baltic dry index", "bdi", "bdry", "freight", "fracht",
                 "global trade", "handel globalny", "shipping", "żegluga",
                 "commodities", "surowce"],
        "page": "app.py",
        "action": "navigate_home",
    },
    {
        "id": "ind_fng",
        "title": "Crypto Fear & Greed Index",
        "subtitle": "Indeks sentymentu rynku kryptowalut (alternative.me). Zakres 0-100.",
        "category": "📊 Wskaźnik Makro",
        "icon": "₿",
        "tags": ["crypto fear greed", "strach chciwość krypto", "bitcoin sentiment",
                 "krypto", "crypto", "btc", "alternative.me", "sentiment"],
        "page": "app.py",
        "action": "navigate_home",
    },
]

# ─── Metody i koncepty quant ───────────────────────────────────────────────────
QUANT_METHODS = [
    {
        "id": "qm_monte_carlo",
        "title": "Monte Carlo Simulation",
        "subtitle": "Stochastyczne ścieżki cenowe. Sobol QMC (10x szybsza zbieżność) + standardowa pseudolosowa.",
        "category": "🔬 Metoda Quant",
        "icon": "🎲",
        "tags": ["monte carlo", "mc simulation", "stochastic paths", "sobol",
                 "qmc", "quasi monte carlo", "random walk", "losowe ścieżki"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_garch",
        "title": "GARCH(1,1) — Klastrowanie Zmienności",
        "subtitle": "Generalized Autoregressive Conditional Heteroskedasticity. Modeluje volatility clustering.",
        "category": "🔬 Metoda Quant",
        "icon": "📈",
        "tags": ["garch", "garch 1 1", "volatility clustering", "klastrowanie zmienności",
                 "heteroskedasticity", "bollerslev", "arch", "conditional volatility"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_merton_jump",
        "title": "Merton Jump-Diffusion",
        "subtitle": "Modeluje nagłe luki cenowe (Czarne Łabędzie) poprzez proces Poissona. Merton (1976).",
        "category": "🔬 Metoda Quant",
        "icon": "⚡",
        "tags": ["merton", "jump diffusion", "poisson process", "skok cen",
                 "black swan", "czarny łabędź", "fat tails", "grube ogony",
                 "jump risk", "ryzyko skoku"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_levy_stable",
        "title": "Lévy-Stable Processes (Heavy Tails)",
        "subtitle": "Procesy o nieskończonej wariancji z parametrem α wg CMS (Chambers-Mallows-Stuck 1976).",
        "category": "🔬 Metoda Quant",
        "icon": "∞",
        "tags": ["lévy", "levy", "stable processes", "heavy tails", "grube ogony",
                 "alpha stable", "infinite variance", "nieskończona wariancja",
                 "cms", "chambers mallows stuck", "stable distribution"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_fbm",
        "title": "Fractional Brownian Motion (fBM)",
        "subtitle": "Mandelbrot's fBM. Modeluje długoterminową pamięć rynków zamiast błądzenia losowego.",
        "category": "🔬 Metoda Quant",
        "icon": "🌊",
        "tags": ["fbm", "fractional brownian motion", "hurst exponent", "wykładnik hursta",
                 "long memory", "długa pamięć", "mandelbrot", "fractal markets",
                 "rynki fraktalne", "self similarity"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_copula",
        "title": "Copula — Zależność Ogonowa",
        "subtitle": "Clayton, Gumbel, Frank, Student-t. Modeluje zależności ogonowe między aktywami.",
        "category": "🔬 Metoda Quant",
        "icon": "🕸️",
        "tags": ["copula", "kopuła", "tail dependence", "zależność ogonowa",
                 "clayton", "gumbel", "frank", "student t", "joint distribution",
                 "mle", "max likelihood", "zaraza", "contagion"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_black_litterman",
        "title": "Black-Litterman Model",
        "subtitle": "Bayesowski model alokacji. Łączy równowagę rynkową (CAPM) z subiektami widokami inwestora.",
        "category": "🔬 Metoda Quant",
        "icon": "🎯",
        "tags": ["black litterman", "bl model", "bayesian portfolio",
                 "market equilibrium", "równowaga rynkowa", "views", "widoki",
                 "capm", "reverse optimization", "tau"],
        "page": "pages/6_BL_Dashboard.py",
    },
    {
        "id": "qm_kelly",
        "title": "Kryterium Kelly'ego",
        "subtitle": "Optymalna frakcja kapitału do zainwestowania w część ryzykowną. John Kelly (1956).",
        "category": "🔬 Metoda Quant",
        "icon": "🎯",
        "tags": ["kelly criterion", "kryterium kelly", "kelly fraction",
                 "optimal bet sizing", "fortune formula", "growth optimal",
                 "half kelly", "fractional kelly"],
        "page": "pages/1_Symulator.py",
    },
    {
        "id": "qm_evt",
        "title": "Extreme Value Theory (EVT)",
        "subtitle": "GEV, GPD. Modelowanie ekstremalnych zdarzeń — ogony rozkładów finansowych.",
        "category": "🔬 Metoda Quant",
        "icon": "📐",
        "tags": ["evt", "extreme value theory", "gev", "gpd", "generalized extreme value",
                 "generalized pareto", "pot", "peak over threshold",
                 "tail risk modeling", "extreme events", "black swan"],
        "page": "pages/5_EVT_Analysis.py",
    },
    {
        "id": "qm_hrp",
        "title": "Hierarchical Risk Parity (HRP)",
        "subtitle": " Lopez de Prado (2016). Alokacja bez odwracania macierzy kowariancji — hierarchiczne klasterowanie.",
        "category": "🔬 Metoda Quant",
        "icon": "🌳",
        "tags": ["hrp", "hierarchical risk parity", "lopez de prado",
                 "dendrogram", "hierarchical clustering", "ward linkage",
                 "risk parity", "equal risk", "covariance matrix"],
        "page": "pages/19_Wealth_Optimizer.py",
    },
    {
        "id": "qm_dcc_garch",
        "title": "DCC-GARCH — Dynamiczne Korelacje",
        "subtitle": "Dynamic Conditional Correlation GARCH. Korelacje między aktywami zmienne w czasie.",
        "category": "🔬 Metoda Quant",
        "icon": "🔗",
        "tags": ["dcc garch", "dynamic conditional correlation", "dcc",
                 "time varying correlation", "zmienne korelacje",
                 "engle", "engle sheppard", "multivariate garch"],
        "page": "pages/7_DCC_Dashboard.py",
    },
    {
        "id": "qm_walk_forward",
        "title": "Walk-Forward Validation (CPCV)",
        "subtitle": "Combinatorial Purged Cross-Validation. Walidacja strategii bez overfittingu danych czasowych.",
        "category": "🔬 Metoda Quant",
        "icon": "🔄",
        "tags": ["walk forward", "cpcv", "combinatorial purged cross validation",
                 "backtest validation", "no data snooping", "purging",
                 "embargo", "time series cv"],
        "page": "pages/23_Walk_Forward.py",
    },
]

# ─── Ustawienia systemowe (akcje szybkie) ──────────────────────────────────────
SYSTEM_ACTIONS = [
    {
        "id": "act_open_settings",
        "title": "⚙️ Otwórz Globalne Ustawienia",
        "subtitle": "Przejdź do konfiguracji portfela, alokacji i parametrów globalnych.",
        "category": "⚡ Akcja",
        "icon": "⚙️",
        "tags": ["ustawienia", "settings", "konfiguracja", "configuration",
                 "portfel", "portfolio", "otwórz", "open", "przejdź"],
        "page": "pages/0_Globalne_Ustawienia.py",
        "score_boost": 2.0,
    },
    {
        "id": "act_run_simulator",
        "title": "🚀 Uruchom Symulator",
        "subtitle": "Przejdź bezpośrednio do Symulatora Monte Carlo.",
        "category": "⚡ Akcja",
        "icon": "🚀",
        "tags": ["symulator", "simulator", "uruchom", "run", "monte carlo",
                 "szybki dostęp", "quick access"],
        "page": "pages/1_Symulator.py",
        "score_boost": 2.0,
    },
    {
        "id": "act_stress_test",
        "title": "⚡ Stress Test — Testuj Portfel",
        "subtitle": "Natychmiastowy stress test dla Twojego portfela.",
        "category": "⚡ Akcja",
        "icon": "⚡",
        "tags": ["stress test", "testuj", "test", "kryzys", "crisis",
                 "szybki dostęp", "quick access"],
        "page": "pages/3_Stress_Test.py",
        "score_boost": 2.0,
    },
    {
        "id": "act_control_center",
        "title": "🌐 Control Center — Dashboard",
        "subtitle": "Główna strona z wskaźnikami makro i Regime Score.",
        "category": "⚡ Akcja",
        "icon": "🌐",
        "tags": ["control center", "dashboard", "główna", "home",
                 "makro", "regime score", "wskaźniki"],
        "page": "app.py",
        "score_boost": 1.5,
    },
    {
        "id": "act_life_os",
        "title": "🧠 Life OS",
        "subtitle": "System Operacyjny Życia — osobiste finanse, nawyki, cele.",
        "category": "⚡ Akcja",
        "icon": "🧠",
        "tags": ["life os", "życie", "life", "osobiste", "personal",
                 "szybki dostęp", "nawyki", "habits"],
        "page": "pages/20_Life_OS.py",
        "score_boost": 1.5,
    },
]


def _extract_i18n_tags() -> Dict[str, List[str]]:
    """
    Parsuje TRANSLATIONS z i18n.py i zwraca słownik:
    page_key -> [lista tagów z tłumaczeń PL + EN]
    """
    tag_map: Dict[str, List[str]] = {}

    # Mapowanie prefix kluczy i18n → ID strony
    prefix_to_page = {
        "sim_":  "pages/1_Symulator.py",
        "scan_": "pages/2_Skaner.py",
        "st_":   "pages/3_Stress_Test.py",
        "em_":   "pages/4_Emerytura.py",
        "hm_":   "pages/8_Health_Monitor.py",
        "wo_":   "pages/19_Wealth_Optimizer.py",
        "gs_":   "pages/0_Globalne_Ustawienia.py",
        "cc_":   "app.py",
        "bl_":   "pages/6_BL_Dashboard.py",
        "dcc_":  "pages/7_DCC_Dashboard.py",
        "dt_":   "pages/21_Day_Trading.py",
        "evt_":  "pages/5_EVT_Analysis.py",
        "tail_": "pages/14_Tail_Hedging.py",
        "tax_":  "pages/15_Tax_Optimizer.py",
        "rebal_": "pages/16_Rebalancing.py",
        "sent_": "pages/17_Sentiment_Flow.py",
        "alt_":  "pages/18_Alt_Risk_Premia.py",
        "lifeos_": "pages/20_Life_OS.py",
        "conc_": "pages/9_Concentration_Risk.py",
        "ddr_":  "pages/10_Drawdown_Recovery.py",
        "rc_":   "pages/11_Regime_Clock.py",
        "ra_":   "pages/12_Regime_Allocation.py",
        "liq_":  "pages/13_Liquidity_Risk.py",
    }

    try:
        from modules.i18n import TRANSLATIONS
        for key, langs in TRANSLATIONS.items():
            # Wyciągnij tekst z obu języków
            for lang_key in ("pl", "en"):
                text = langs.get(lang_key, "")
                if not text or not isinstance(text, str):
                    continue
                # Usuń emoji, tagi HTML, {placeholders}
                clean = re.sub(r'\{[^}]+\}', '', text)
                clean = re.sub(r'[^\w\s\-/&]', ' ', clean, flags=re.UNICODE)
                words = [w.lower().strip() for w in clean.split() if len(w) > 2]

                # Przypisz do właściwej strony
                for prefix, page in prefix_to_page.items():
                    if key.startswith(prefix):
                        if page not in tag_map:
                            tag_map[page] = []
                        tag_map[page].extend(words)
                        break
    except Exception:
        pass

    # Deduplikacja
    for page in tag_map:
        tag_map[page] = list(set(tag_map[page]))

    return tag_map


def _extract_docstring_tags(pages_dir: str = "pages") -> Dict[str, List[str]]:
    """
    Parsuje docstringi i komentarze z plików pages/*.py i modules/*.py
    i zwraca słownik file_path -> [słowa kluczowe].
    Używa AST dla bezpiecznego parsowania — bez importu modułów.
    """
    tags: Dict[str, List[str]] = {}
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)))

    dirs_to_scan = [
        os.path.join(root, "pages"),
        os.path.join(root, "modules"),
    ]

    for scan_dir in dirs_to_scan:
        if not os.path.exists(scan_dir):
            continue
        for fname in os.listdir(scan_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(scan_dir, fname)
            rel_path = os.path.relpath(fpath, root).replace("\\", "/")

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()

                # Zbierz słowa z komentarzy (#)
                comment_words = []
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        comment_words.extend(
                            w.lower() for w in re.split(r'[\s\-_,;:\.!?]+', stripped[1:])
                            if len(w) > 2 and w.isalpha()
                        )

                # Zbierz słowa z docstringów via AST
                docstring_words = []
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                            ds = ast.get_docstring(node)
                            if ds:
                                docstring_words.extend(
                                    w.lower() for w in re.split(r'[\s\-_,;:\.!?()]+', ds)
                                    if len(w) > 2 and w.isalpha()
                                )
                except SyntaxError:
                    pass

                all_words = list(set(comment_words + docstring_words))
                if all_words:
                    tags[rel_path] = all_words

            except Exception:
                continue

    return tags


def build_search_index() -> List[Dict[str, Any]]:
    """
    Buduje pełny indeks wyszukiwania projektu.
    Zwraca listę elementów gotową do JSON serialization.

    Każdy element:
    {
        "id": str,
        "title": str,
        "subtitle": str,
        "category": str,      # kategoria wyświetlana w wynikach
        "icon": str,          # emoji
        "tags": List[str],    # słowa kluczowe do wyszukiwania
        "page": str,          # ścieżka do strony Streamlit
        "score_boost": float, # opcjonalny mnożnik trafności (1.0 = default)
    }
    """
    index: List[Dict[str, Any]] = []

    # ── 1. Tagi z i18n.py ──────────────────────────────────────────────────────
    i18n_tags = _extract_i18n_tags()

    # ── 2. Tagi z docstringów i komentarzy ────────────────────────────────────
    docstring_tags = _extract_docstring_tags()

    # ── 3. Strony ────────────────────────────────────────────────────────────
    for (page_path, title_pl, title_en, icon, category, static_tags) in PAGES_MAP:
        # Złącz tagi: statyczne + i18n + docstringi
        all_tags = list(static_tags)
        all_tags.extend(i18n_tags.get(page_path, []))
        all_tags.extend(docstring_tags.get(page_path, []))

        # Tytuły też jako tagi
        for word in re.split(r'[\s\-/&]+', title_pl.lower()):
            if len(word) > 2:
                all_tags.append(word)
        for word in re.split(r'[\s\-/&]+', title_en.lower()):
            if len(word) > 2:
                all_tags.append(word)

        # Deduplikacja i filtracja
        all_tags = list(set(t.lower().strip() for t in all_tags if t and len(t) > 1))

        index.append({
            "id": f"page_{page_path.replace('/', '_').replace('.', '_')}",
            "title": title_pl,
            "subtitle": title_en,
            "category": category,
            "icon": icon,
            "tags": all_tags,
            "page": page_path,
            "nav_url": PAGE_TO_NAV_URL.get(page_path, "/"),
            "score_boost": 1.5,
        })

    # ── 4. Wskaźniki makro ───────────────────────────────────────────────────────
    for ind in INDICATORS:
        entry = dict(ind)
        entry["score_boost"] = entry.get("score_boost", 1.2)
        entry["nav_url"] = PAGE_TO_NAV_URL.get(entry.get("page", ""), "/")
        index.append(entry)

    # ── 5. Metody quant ─────────────────────────────────────────────────────────
    for qm in QUANT_METHODS:
        entry = dict(qm)
        entry["score_boost"] = entry.get("score_boost", 1.0)
        entry["nav_url"] = PAGE_TO_NAV_URL.get(entry.get("page", ""), "/")
        index.append(entry)

    # ── 6. Akcje systemowe ────────────────────────────────────────────────────────
    for act in SYSTEM_ACTIONS:
        entry = dict(act)
        entry["nav_url"] = PAGE_TO_NAV_URL.get(entry.get("page", ""), "/")
        index.append(entry)

    return index


def get_search_index_json() -> str:
    """
    Zwraca indeks jako JSON string gotowy do wstrzyknięcia do JavaScript.
    Używa cache by nie przebudowywać przy każdym rerunie.
    """
    import json
    index = build_search_index()
    return json.dumps(index, ensure_ascii=False)
