"""
modules/module_registry.py
===========================
Centralny rejestr wszystkich modułów projektu Barbell Strategy Dashboard.

Jest to JEDYNE źródło prawdy dla metadanych modułów.
Używany przez:
  - tools/project_updater.py  → synchronizuje Mapę, search_index, hubs
  - modules/search_index.py   → buduje indeks wyszukiwania (importuje REGISTRY)
  - pages/00_Mapa_Projektu.py → buduje graf (importuje REGISTRY)

Dodawanie nowego modułu:
  1. Dodaj wpis ModuleEntry do listy REGISTRY poniżej
  2. Uruchom: python tools/project_updater.py sync
  LUB użyj kreatora:  python tools/project_updater.py new
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import glob as _glob


# ─────────────────────────────────────────────────────────────────────────────
#  DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleEntry:
    page_file:   str                    # np. "pages/50_Obligacje_Skarbowe.py"
    title_pl:    str                    # Tytuł PL (wyświetlany w UI)
    title_en:    str                    # Tytuł EN
    icon:        str                    # Emoji ikona
    category:    str                    # Klucz sekcji z SECTIONS poniżej
    nav_url:     str                    # URL Streamlit np. "/Obligacje_Skarbowe"
    tags:        list[str]              # Słowa kluczowe do wyszukiwania
    module_file: Optional[str] = None  # np. "modules/obligacje_skarbowe.py"
    hub:         Optional[str] = None  # np. "pages/module_majatku.py" lub None
    render_fn:   Optional[str] = None  # np. "render_obligacje_module"
    status:      str = "active"        # "active" | "wip" | "deprecated"


# ─────────────────────────────────────────────────────────────────────────────
#  SEKCJE — struktura hierarchii Mapy Projektu
#  Klucz musi być identyczny z kluczem w REGISTRY[i].category
# ─────────────────────────────────────────────────────────────────────────────

SECTIONS = {
    "🌐 Centrum Dowodzenia": {
        "color": ("#2979ff", "#fff"),
        "hub":   None,
    },
    "📉 Środowisko Makro i Reżimy": {
        "color": ("#ff9100", "#000"),
        "hub":   {"title": "Makro i Reżimy", "page_file": "pages/module_makro.py"},
    },
    "⚖️ Zarządzanie Portfelem": {
        "color": ("#ffea00", "#000"),
        "hub":   {"title": "Zarządzanie Portfelem", "page_file": "pages/module_portfel.py"},
    },
    "🛡️ Centrum Ryzyka": {
        "color": ("#00e676", "#000"),
        "hub":   {"title": "Centrum Ryzyka", "page_file": "pages/module_ryzyko.py"},
    },
    "🧬 Laboratorium Quant i AI": {
        "color": ("#d500f9", "#fff"),
        "hub":   {"title": "Laboratorium Quant", "page_file": "pages/module_quant.py"},
    },
    "💰 Planowanie Majątku (FIRE)": {
        "color": ("#ff1744", "#fff"),
        "hub":   {"title": "Planowanie Majątku", "page_file": "pages/module_majatku.py"},
    },
    "♟️ Meta-Decyzje i Teoria": {
        "color": ("#00e5ff", "#000"),
        "hub":   {"title": "Meta-Decyzje", "page_file": "pages/module_meta.py"},
    },
    "🎯 Moduły Aktywne i Trening": {
        "color": ("#ff6d00", "#fff"),
        "hub":   {"title": "Moduły Aktywne", "page_file": "pages/module_aktywne.py"},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  REGISTRY — lista wszystkich modułów
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: list[ModuleEntry] = [

    # ── Centrum Dowodzenia ────────────────────────────────────────────────────
    ModuleEntry(
        page_file="app.py",
        title_pl="Control Center — Dashboard Główny",
        title_en="Control Center — Main Dashboard",
        icon="🌐",
        category="🌐 Centrum Dowodzenia",
        nav_url="/",
        hub=None,
        tags=["dashboard", "control center", "makro", "vix", "regime score",
              "business cycle", "wskaźniki", "główna", "home"],
    ),
    ModuleEntry(
        page_file="pages/0_Globalne_Ustawienia.py",
        title_pl="Globalne Ustawienia Portfela",
        title_en="Global Portfolio Settings",
        icon="⚙️",
        category="🌐 Centrum Dowodzenia",
        nav_url="/Globalne_Ustawienia",
        hub=None,
        tags=["ustawienia", "settings", "portfel", "portfolio", "alokacja",
              "kelly", "bezpieczna", "safe sleeve", "ryzykowna",
              "obligacje", "profil", "heartbeat", "waluta"],
    ),
    ModuleEntry(
        page_file="pages/28_Doradca.py",
        title_pl="AI Doradca Inwestycyjny",
        title_en="AI Investment Advisor",
        icon="🤖",
        category="🌐 Centrum Dowodzenia",
        nav_url="/Doradca",
        hub=None,
        module_file="modules/advisor_engine.py",
        render_fn="render_advisor",
        tags=["doradca", "advisor", "ai", "sztuczna inteligencja",
              "rekomendacje", "recommendations", "portfolio advice", "robo-advisor"],
    ),
    ModuleEntry(
        page_file="pages/00_Mapa_Projektu.py",
        title_pl="Mapa Projektu",
        title_en="Project Map",
        icon="🗺️",
        category="🌐 Centrum Dowodzenia",
        nav_url="/Mapa_Projektu",
        hub=None,
        tags=["mapa", "map", "projekt", "project", "nawigacja", "navigation", "graf"],
    ),

    # ── Środowisko Makro i Reżimy ─────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/2_Skaner.py",
        title_pl="Skaner Rynku — Antykruchość",
        title_en="Market Scanner — Antifragility",
        icon="🔍",
        category="📉 Środowisko Makro i Reżimy",
        nav_url="/module_makro",
        hub="pages/module_makro.py",
        module_file="modules/scanner.py",
        tags=["skaner", "scanner", "antykruchość", "antifragility", "barbell score",
              "ranking", "korelacja", "nowcast", "etf", "krypto"],
    ),
    ModuleEntry(
        page_file="pages/11_Regime_Clock.py",
        title_pl="Zegar Reżimów Makro",
        title_en="Macro Regime Clock",
        icon="🕐",
        category="📉 Środowisko Makro i Reżimy",
        nav_url="/module_makro",
        hub="pages/module_makro.py",
        module_file="modules/macro_regime_clock.py",
        tags=["regime clock", "zegar reżimów", "cykl koniunkturalny", "pmi",
              "inflacja", "yield curve", "expansion", "recession"],
    ),
    ModuleEntry(
        page_file="pages/12_Regime_Allocation.py",
        title_pl="Regime-Adaptive Allocation",
        title_en="Regime-Adaptive Allocation",
        icon="🎯",
        category="📉 Środowisko Makro i Reżimy",
        nav_url="/module_makro",
        hub="pages/module_makro.py",
        module_file="modules/regime_adaptive_allocation.py",
        tags=["regime adaptive", "alokacja adaptacyjna", "dynamic allocation",
              "bull", "bear", "crisis", "momentum", "mean reversion"],
    ),
    ModuleEntry(
        page_file="pages/26_Recession_Nowcasting.py",
        title_pl="Recession Nowcasting — Prognoza Recesji",
        title_en="Recession Nowcasting",
        icon="📡",
        category="📉 Środowisko Makro i Reżimy",
        nav_url="/module_makro",
        hub="pages/module_makro.py",
        module_file="modules/recession_models.py",
        tags=["recession nowcasting", "yield curve", "fred", "pmi",
              "leading indicators", "probability recession"],
    ),

    # ── Zarządzanie Portfelem ─────────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/8_Health_Monitor.py",
        title_pl="Portfolio Health Monitor",
        title_en="Portfolio Health Monitor",
        icon="🏥",
        category="⚖️ Zarządzanie Portfelem",
        nav_url="/module_portfel",
        hub="pages/module_portfel.py",
        module_file="modules/portfolio_health_monitor.py",
        tags=["health monitor", "zdrowie portfela", "drawdown alert",
              "volatility spike", "korelacja", "ath", "monitoring"],
    ),
    ModuleEntry(
        page_file="pages/16_Rebalancing.py",
        title_pl="Smart Rebalancing Engine",
        title_en="Smart Rebalancing Engine",
        icon="⚖️",
        category="⚖️ Zarządzanie Portfelem",
        nav_url="/module_portfel",
        hub="pages/module_portfel.py",
        module_file="modules/smart_rebalancing_engine.py",
        tags=["rebalancing", "rebalansowanie", "threshold", "calendar",
              "tactical", "drift", "shannon demon"],
    ),
    ModuleEntry(
        page_file="pages/15_Tax_Optimizer.py",
        title_pl="Tax Optimizer PL — Optymalizacja Podatkowa",
        title_en="Tax Optimizer PL",
        icon="🧾",
        category="⚖️ Zarządzanie Portfelem",
        nav_url="/module_portfel",
        hub="pages/module_portfel.py",
        module_file="modules/tax_optimizer_pl.py",
        tags=["tax optimizer", "podatki", "belka", "capital gains",
              "ike", "ikze", "ppk", "tax loss harvesting"],
    ),
    ModuleEntry(
        page_file="pages/19_Wealth_Optimizer.py",
        title_pl="Wealth Protection Optimizer",
        title_en="Wealth Protection Optimizer",
        icon="🏰",
        category="⚖️ Zarządzanie Portfelem",
        nav_url="/module_portfel",
        hub="pages/module_portfel.py",
        module_file="modules/wealth_protection_optimizer.py",
        tags=["wealth optimizer", "ochrona majątku", "goal-based investing",
              "human capital", "ldi", "hrp", "life goals", "merton lifecycle"],
    ),

    # ── Centrum Ryzyka ────────────────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/3_Stress_Test.py",
        title_pl="Stress Test — Historyczne Kryzysy",
        title_en="Stress Test — Historical Crises",
        icon="⚡",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        module_file="modules/stress_test.py",
        tags=["stress test", "kryzys", "crisis", "crash", "2008", "covid",
              "drawdown", "szok", "historyczny", "copula", "contagion"],
    ),
    ModuleEntry(
        page_file="pages/9_Concentration_Risk.py",
        title_pl="Ryzyko Koncentracji Portfela",
        title_en="Portfolio Concentration Risk",
        icon="🎯",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        module_file="modules/concentration_risk_monitor.py",
        tags=["koncentracja", "concentration risk", "herfindahl", "hhi",
              "diversification", "sector", "geographic"],
    ),
    ModuleEntry(
        page_file="pages/13_Liquidity_Risk.py",
        title_pl="Liquidity Risk Analyzer",
        title_en="Liquidity Risk Analyzer",
        icon="💧",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        module_file="modules/liquidity_risk_analyzer.py",
        tags=["liquidity risk", "ryzyko płynności", "bid-ask spread",
              "volume", "market impact", "amihud", "kyle lambda"],
    ),
    ModuleEntry(
        page_file="pages/5_EVT_Analysis.py",
        title_pl="EVT — Analiza Wartości Ekstremalnych",
        title_en="EVT — Extreme Value Analysis",
        icon="📐",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        tags=["evt", "extreme value theory", "gumbel", "frechet", "weibull",
              "tail risk", "var", "cvar", "pot", "gev", "black swan"],
    ),
    ModuleEntry(
        page_file="pages/27_Systemic_Risk.py",
        title_pl="Systemic Risk & CoVaR",
        title_en="Systemic Risk & CoVaR",
        icon="⚠️",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        tags=["systemic risk", "ryzyko systemowe", "covar", "srisk",
              "too big to fail", "contagion", "network risk"],
    ),
    ModuleEntry(
        page_file="pages/10_Drawdown_Recovery.py",
        title_pl="Drawdown Recovery Analyzer",
        title_en="Drawdown Recovery Analyzer",
        icon="📉",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        module_file="modules/drawdown_recovery_analyzer.py",
        tags=["drawdown", "recovery", "odrabianie strat", "underwater",
              "pain index", "ulcer index", "calmar", "max drawdown"],
    ),
    ModuleEntry(
        page_file="pages/14_Tail_Hedging.py",
        title_pl="Tail Risk Hedging — Ochrona Ogonów",
        title_en="Tail Risk Hedging",
        icon="🛡️",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        module_file="modules/tail_risk_hedging.py",
        tags=["tail hedging", "tail risk", "opcje", "options", "put", "collar",
              "hedging", "spx", "vix", "antykruchość"],
    ),
    ModuleEntry(
        page_file="pages/32_Inzynieria_Opcji.py",
        title_pl="Inżynieria Opcji — Options Engineering",
        title_en="Options Engineering",
        icon="⚗️",
        category="🛡️ Centrum Ryzyka",
        nav_url="/module_ryzyko",
        hub="pages/module_ryzyko.py",
        tags=["opcje", "options", "black-scholes", "delta", "gamma", "vega",
              "greeks", "implied volatility"],
    ),

    # ── Laboratorium Quant i AI ───────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/6_BL_Dashboard.py",
        title_pl="Black-Litterman AI Dashboard",
        title_en="Black-Litterman AI Dashboard",
        icon="🎯",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/black_litterman.py",
        tags=["black-litterman", "bl", "bayesian", "prior", "posterior",
              "views", "market equilibrium", "reverse optimization"],
    ),
    ModuleEntry(
        page_file="pages/24_HERC_Portfolio.py",
        title_pl="HERC — Hierarchical Equal Risk Contribution",
        title_en="HERC — Hierarchical Equal Risk Contribution",
        icon="🌳",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/herc_optimizer.py",
        tags=["herc", "hierarchical equal risk contribution",
              "risk parity", "dendrogram", "clustering", "min variance"],
    ),
    ModuleEntry(
        page_file="pages/7_DCC_Dashboard.py",
        title_pl="DCC-GARCH — Dynamiczne Korelacje",
        title_en="DCC-GARCH — Dynamic Correlations",
        icon="🔗",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/dcc_garch.py",
        tags=["dcc", "garch", "dynamiczne korelacje", "volatility clustering",
              "spillover", "time-varying"],
    ),
    ModuleEntry(
        page_file="pages/22_Factor_Analysis.py",
        title_pl="Factor Zoo & PCA — Analiza Faktorów",
        title_en="Factor Zoo & PCA — Factor Analysis",
        icon="🔬",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/factor_model.py",
        tags=["factor analysis", "pca", "principal component", "autoencoder",
              "fama french", "carhart", "momentum factor", "zoo faktorów"],
    ),
    ModuleEntry(
        page_file="pages/33_Sieci_Przyczynowe.py",
        title_pl="Sieci Przyczynowe — Causal Networks",
        title_en="Causal Networks",
        icon="🕸️",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/causal_risk.py",
        tags=["sieci przyczynowe", "causal networks", "causality",
              "granger causality", "dag", "bayesian network"],
    ),
    ModuleEntry(
        page_file="pages/48_Stochastic_Errors.py",
        title_pl="Stochastic Errors — Błędy Stochastyczne",
        title_en="Stochastic Errors",
        icon="📉",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/stochastic_errors.py",
        tags=["stochastic errors", "błędy stochastyczne", "noise",
              "signal to noise", "estimation error", "model risk"],
    ),
    ModuleEntry(
        page_file="pages/18_Alt_Risk_Premia.py",
        title_pl="Alternative Risk Premia",
        title_en="Alternative Risk Premia",
        icon="🎲",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        module_file="modules/alternative_risk_premia.py",
        tags=["alternative risk premia", "momentum", "value", "carry",
              "low volatility", "quality", "size", "smart beta"],
    ),
    ModuleEntry(
        page_file="pages/35_Stochastic_Calculus.py",
        title_pl="Stochastic Calculus — Rachunek Stochastyczny",
        title_en="Stochastic Calculus",
        icon="∫",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["stochastic calculus", "ito", "brownian motion", "sde",
              "geometric brownian motion", "gbm", "wiener process"],
    ),
    ModuleEntry(
        page_file="pages/35_Vol_Surface.py",
        title_pl="Volatility Surface — Powierzchnia Zmienności",
        title_en="Volatility Surface",
        icon="📈",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["volatility surface", "vol surface", "implied volatility",
              "smile", "skew", "term structure", "vix term structure", "sabr"],
    ),
    ModuleEntry(
        page_file="pages/37_Covariance_Shrinkage.py",
        title_pl="Covariance Shrinkage — Ledoit-Wolf",
        title_en="Covariance Shrinkage",
        icon="📐",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["covariance shrinkage", "ledoit wolf", "kowariancja",
              "shrinkage", "regularization", "efficient frontier"],
    ),
    ModuleEntry(
        page_file="pages/38_Market_Impact.py",
        title_pl="Market Impact — Wpływ na Rynek",
        title_en="Market Impact",
        icon="💥",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["market impact", "price impact", "slippage",
              "execution cost", "almgren chriss", "optimal execution"],
    ),
    ModuleEntry(
        page_file="pages/42_Econophysics.py",
        title_pl="Econophysics — Ekonofizyka",
        title_en="Econophysics",
        icon="⚛️",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["econophysics", "ekonofizyka", "power law", "scale invariance",
              "self-organized criticality", "zipf law", "sornette"],
    ),
    ModuleEntry(
        page_file="pages/43_Entropy_Finance.py",
        title_pl="Entropy Finance — Entropia Rynków",
        title_en="Entropy Finance",
        icon="🌀",
        category="🧬 Laboratorium Quant i AI",
        nav_url="/module_quant",
        hub="pages/module_quant.py",
        tags=["entropy finance", "entropy", "information entropy",
              "max entropy", "shannon entropy", "thermodynamics"],
    ),

    # ── Planowanie Majątku (FIRE) ─────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/4_Emerytura.py",
        title_pl="Emerytura / FIRE — Planer Emerytalny",
        title_en="Retirement / FIRE Planner",
        icon="🏖️",
        category="💰 Planowanie Majątku (FIRE)",
        nav_url="/module_majatku",
        hub="pages/module_majatku.py",
        module_file="modules/emerytura.py",
        render_fn="render_emerytura_module",
        tags=["emerytura", "retirement", "fire", "finanse osobiste",
              "swr", "safe withdrawal rate", "ike", "ikze", "ppk", "zus",
              "income", "inflacja", "glide path", "dekumulacja"],
    ),
    ModuleEntry(
        page_file="pages/36_Fixed_Income_v2.py",
        title_pl="Fixed Income — Obligacje",
        title_en="Fixed Income",
        icon="📊",
        category="💰 Planowanie Majątku (FIRE)",
        nav_url="/module_majatku",
        hub="pages/module_majatku.py",
        tags=["fixed income", "obligacje", "bonds", "duration", "convexity",
              "yield to maturity", "ytm", "nss", "nelson siegel",
              "treasury", "skarbowe", "corporate bonds"],
    ),
    ModuleEntry(
        page_file="pages/36_Portfolio_Insurance.py",
        title_pl="Portfolio Insurance — Ubezpieczenie Portfela",
        title_en="Portfolio Insurance",
        icon="🛡️",
        category="💰 Planowanie Majątku (FIRE)",
        nav_url="/module_majatku",
        hub="pages/module_majatku.py",
        tags=["portfolio insurance", "cppi",
              "constant proportion portfolio insurance", "obpi", "floor"],
    ),
    # ── NOWE: Obligacje Skarbowe ────────────────────────────────────
    ModuleEntry(
        page_file="pages/50_Obligacje_Skarbowe.py",
        title_pl="Kalkulator Obligacji Skarbowych",
        title_en="Treasury Bond Calculator",
        icon="🏦",
        category="💰 Planowanie Majątku (FIRE)",
        nav_url="/Obligacje_Skarbowe",
        hub="pages/module_majatku.py",
        module_file="modules/obligacje_skarbowe.py",
        render_fn="render_obligacje_module",
        tags=["obligacje skarbowe", "treasury bonds", "ots", "ror", "dor",
              "tos", "coi", "edo", "belka", "podatek", "inflacja",
              "oprocentowanie", "wcześniejszy wykup", "odsetki",
              "obligacje 3-miesięczne", "obligacje roczne", "obligacje 10-letnie",
              "bonds", "fixed income", "safe haven", "bezpieczna część portfela"],
    ),

    # ── Meta-Decyzje i Teoria ─────────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/34_Przewaga_Informacyjna.py",
        title_pl="Przewaga Informacyjna",
        title_en="Information Edge",
        icon="🔍",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        module_file="modules/information_theory_edge.py",
        tags=["przewaga informacyjna", "information edge", "alpha",
              "market efficiency", "kelly criterion", "edge ratio"],
    ),
    ModuleEntry(
        page_file="pages/29_Kalkulator_Bayesa.py",
        title_pl="Kalkulator Bayesowski",
        title_en="Bayesian Calculator",
        icon="🎲",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        tags=["bayesian", "bayesowski", "prior", "posterior", "likelihood",
              "conditional probability", "bayes theorem"],
    ),
    ModuleEntry(
        page_file="pages/31_Asymetria_Informacji.py",
        title_pl="Asymetria Informacji — Information Theory",
        title_en="Information Asymmetry",
        icon="🔬",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        tags=["asymetria informacji", "information asymmetry", "insider trading",
              "adverse selection", "moral hazard", "signaling", "entropy"],
    ),
    ModuleEntry(
        page_file="pages/30_Teoria_Gier.py",
        title_pl="Teoria Gier — Game Theory",
        title_en="Game Theory",
        icon="♟️",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        module_file="modules/game_theory_engine.py",
        tags=["teoria gier", "game theory", "nash equilibrium",
              "prisoner dilemma", "dominant strategy", "minimax", "zero sum"],
    ),
    ModuleEntry(
        page_file="pages/49_Chaos_Deterministyczny.py",
        title_pl="Chaos Deterministyczny",
        title_en="Deterministic Chaos",
        icon="🌀",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        tags=["chaos", "deterministic chaos", "lorenz attractor",
              "lyapunov exponent", "bifurcation", "fractals", "mandelbrot"],
    ),
    ModuleEntry(
        page_file="pages/47_Metacognition.py",
        title_pl="Metacognition — Psychologia Inwestowania",
        title_en="Metacognition — Investor Psychology",
        icon="🧠",
        category="♟️ Meta-Decyzje i Teoria",
        nav_url="/module_meta",
        hub="pages/module_meta.py",
        tags=["metacognition", "metakognicja", "investor psychology",
              "psychologia inwestowania", "behavioral finance",
              "cognitive bias", "loss aversion", "overconfidence"],
    ),

    # ── Moduły Aktywne i Trening ──────────────────────────────────────────────
    ModuleEntry(
        page_file="pages/1_Symulator.py",
        title_pl="Symulator Monte Carlo / Barbell",
        title_en="Monte Carlo / Barbell Simulator",
        icon="⚖️",
        category="🎯 Moduły Aktywne i Trening",
        nav_url="/module_aktywne",
        hub="pages/module_aktywne.py",
        module_file="modules/simulation.py",
        tags=["symulator", "simulator", "monte carlo", "barbell", "backtest",
              "garch", "kelly", "lévy", "fbm", "hurst", "sobol",
              "cagr", "sharpe", "sortino", "var", "cvar", "drawdown"],
    ),
    ModuleEntry(
        page_file="pages/21_Day_Trading.py",
        title_pl="Day Trading Module",
        title_en="Day Trading Module",
        icon="📊",
        category="🎯 Moduły Aktywne i Trening",
        nav_url="/module_aktywne",
        hub="pages/module_aktywne.py",
        tags=["day trading", "trading", "technical analysis", "rsi", "macd",
              "bollinger", "sma", "ema", "pairs trading", "cointegration"],
    ),
    ModuleEntry(
        page_file="pages/17_Sentiment_Flow.py",
        title_pl="Sentiment Flow Tracker — NLP",
        title_en="Sentiment Flow Tracker — NLP",
        icon="🌊",
        category="🎯 Moduły Aktywne i Trening",
        nav_url="/module_aktywne",
        hub="pages/module_aktywne.py",
        module_file="modules/sentiment_flow_tracker.py",
        tags=["sentiment", "sentyment", "nlp", "vader",
              "fear greed", "news", "social media", "flow"],
    ),
    ModuleEntry(
        page_file="pages/23_Walk_Forward.py",
        title_pl="Walk-Forward CPCV Walidacja",
        title_en="Walk-Forward CPCV Validation",
        icon="🔄",
        category="🎯 Moduły Aktywne i Trening",
        nav_url="/module_aktywne",
        hub="pages/module_aktywne.py",
        module_file="modules/walk_forward.py",
        tags=["walk-forward", "cpcv", "cross-validation", "walidacja",
              "backtest bias", "overfitting", "purged k-fold"],
    ),
    ModuleEntry(
        page_file="pages/20_Life_OS.py",
        title_pl="Life OS — System Operacyjny Życia",
        title_en="Life OS — Life Operating System",
        icon="🧠",
        category="🎯 Moduły Aktywne i Trening",
        nav_url="/module_aktywne",
        hub="pages/module_aktywne.py",
        tags=["life os", "system życia", "osobiste finanse", "habit tracking",
              "nawyki", "goals", "cele", "health", "zdrowie", "productivity"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  FUNKCJE POMOCNICZE
# ─────────────────────────────────────────────────────────────────────────────

def get_by_category(category: str) -> list[ModuleEntry]:
    """Zwraca wszystkie moduły dla danej sekcji."""
    return [e for e in REGISTRY if e.category == category]


def get_by_hub(hub_file: str) -> list[ModuleEntry]:
    """Zwraca wszystkie moduły podpięte pod dany hub."""
    return [e for e in REGISTRY if e.hub == hub_file]


def get_by_page(page_file: str) -> Optional[ModuleEntry]:
    """Zwraca wpis dla podanego pliku strony (lub None)."""
    for e in REGISTRY:
        if e.page_file == page_file:
            return e
    return None


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_unregistered_pages() -> list[str]:
    """
    Skanuje pages/[0-9]*.py i zwraca listę plików
    nieobecnych w REGISTRY. Ścieżki relatywne do root projektu.
    """
    registered = {e.page_file.replace("\\", "/") for e in REGISTRY}
    pattern = os.path.join(_BASE_DIR, "pages", "[0-9]*.py")
    found = []
    for p in _glob.glob(pattern):
        rel = os.path.relpath(p, _BASE_DIR).replace("\\", "/")
        if rel not in registered:
            found.append(rel)
    return sorted(found)


def build_menu_structure() -> dict:
    """
    Buduje słownik MENU_STRUCTURE kompatybilny z 00_Mapa_Projektu.py
    na podstawie REGISTRY i SECTIONS.
    """
    menu = {}
    for sec_name, sec_data in SECTIONS.items():
        entries = get_by_category(sec_name)
        if not entries and sec_name not in ("🌐 Centrum Dowodzenia",):
            continue
        subpages = [{"title": e.title_pl, "page_file": e.page_file}
                    for e in entries]
        menu[sec_name] = {
            "color": sec_data["color"],
            "hub":   sec_data["hub"],
            "subpages": subpages,
        }
    return menu


def build_pages_map() -> list[tuple]:
    """
    Buduje listę PAGES_MAP kompatybilną z search_index.py.
    Format: (file_path, title_pl, title_en, icon, category, tags)
    """
    result = []
    for e in REGISTRY:
        result.append((
            e.page_file,
            e.title_pl,
            e.title_en,
            e.icon,
            e.category,
            e.tags,
        ))
    return result


def build_page_to_nav_url() -> dict:
    """Buduje słownik PAGE_TO_NAV_URL na podstawie REGISTRY."""
    return {e.page_file: e.nav_url for e in REGISTRY}
