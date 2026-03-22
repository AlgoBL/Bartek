"""
modules/global_settings.py
==========================
Centralny moduł globalnych ustawień portfela Barbell Strategy.

Architektura przepływu danych:
  global_settings.json (dysk) ──► load_global_settings() ──► session_state["_gs"]
                                                                     │
                                               get_gs() ◄────────────┘
                                                  │
                    inject_into_session_state() ◄─┘  (propaguje do _s.* kluczy)
                             │
                 1_Symulator.py, 3_Stress_Test.py, ... (czytają _s.* jak zawsze)
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st

from config import (
    GLOBAL_SETTINGS_PATH,
    DEFAULT_SAFE_ALLOCATION,
    DEFAULT_SAFE_RATE,
    DEFAULT_SAFE_TYPE,
    DEFAULT_SAFE_TICKERS,
    DEFAULT_RISKY_ASSETS,
    DEFAULT_INITIAL_CAPITAL,
)

# ─── Stałe ────────────────────────────────────────────────────────────────────

SESSION_KEY = "_gs"  # klucz w st.session_state dla obiektu GlobalPortfolio

# Presety gotowych profili
PRESET_PROFILES: Dict[str, dict] = {
    "🛡️ Konserwatywny": {
        "safe_type": "fixed",
        "safe_rate": 0.0551,
        "safe_tickers": [],
        "risky_assets": [{"ticker": "SPY", "weight": 100.0, "asset_class": "ETF US"}],
        "alloc_safe_pct": 0.95,
        "initial_capital": 100_000.0,
        "profile_name": "Konserwatywny",
    },
    "⚖️ Zrównoważony": {
        "safe_type": "fixed",
        "safe_rate": 0.0551,
        "safe_tickers": [],
        "risky_assets": [
            {"ticker": "SPY", "weight": 60.0, "asset_class": "ETF US"},
            {"ticker": "QQQ", "weight": 40.0, "asset_class": "ETF Tech"},
        ],
        "alloc_safe_pct": 0.70,
        "initial_capital": 100_000.0,
        "profile_name": "Zrównoważony",
    },
    "🎯 Agresywny Barbell": {
        "safe_type": "fixed",
        "safe_rate": 0.0551,
        "safe_tickers": [],
        "risky_assets": [
            {"ticker": "SPY",     "weight": 40.0, "asset_class": "ETF US"},
            {"ticker": "QQQ",     "weight": 30.0, "asset_class": "ETF Tech"},
            {"ticker": "NVDA",    "weight": 20.0, "asset_class": "Akcja"},
            {"ticker": "BTC-USD", "weight": 10.0, "asset_class": "Krypto"},
        ],
        "alloc_safe_pct": 0.85,
        "initial_capital": 100_000.0,
        "profile_name": "Agresywny Barbell",
    },
    "🚀 Spekulacyjny": {
        "safe_type": "fixed",
        "safe_rate": 0.0551,
        "safe_tickers": [],
        "risky_assets": [
            {"ticker": "BTC-USD", "weight": 40.0, "asset_class": "Krypto"},
            {"ticker": "NVDA",    "weight": 30.0, "asset_class": "Akcja"},
            {"ticker": "MSTR",    "weight": 20.0, "asset_class": "Akcja"},
            {"ticker": "QQQ",     "weight": 10.0, "asset_class": "ETF Tech"},
        ],
        "alloc_safe_pct": 0.60,
        "initial_capital": 100_000.0,
        "profile_name": "Spekulacyjny",
    },
}


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class GlobalPortfolio:
    """Struktura danych globalnego portfela inwestycyjnego."""

    # Część bezpieczna
    safe_type: str = DEFAULT_SAFE_TYPE          # "fixed" | "tickers"
    safe_rate: float = DEFAULT_SAFE_RATE        # oprocentowanie obligacji (np. 0.0551)
    safe_tickers: List[str] = field(default_factory=lambda: list(DEFAULT_SAFE_TICKERS))

    # Część ryzykowna – lista słowników {"ticker": str, "weight": float, "asset_class": str}
    risky_assets: List[Dict] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_RISKY_ASSETS)
    )

    # Podział portfela (jako ułamek, np. 0.85 = 85% bezpieczna)
    alloc_safe_pct: float = DEFAULT_SAFE_ALLOCATION

    # Kapitał startowy
    initial_capital: float = DEFAULT_INITIAL_CAPITAL

    # Tło (Heartbeat Engine)
    bg_refresh_enabled: bool = True
    bg_refresh_interval_minutes: int = 15

    # Język interfejsu / Interface language
    language: str = "pl"  # "pl" | "en"

    # Personalizacja Menu (Visible Modules)
    visible_modules: List[str] = field(default_factory=list)

    # Metadane
    profile_name: str = "Domyślny"
    last_updated: str = ""

    # ── Właściwości pomocnicze ──────────────────────────────────────────────

    @property
    def alloc_risky_pct(self) -> float:
        return 1.0 - self.alloc_safe_pct

    @property
    def risky_tickers_str(self) -> str:
        """Zwraca tickery ryzykowne jako string oddzielony przecinkami."""
        return ", ".join(a["ticker"] for a in self.risky_assets)

    @property
    def safe_tickers_str(self) -> str:
        """Zwraca tickery bezpieczne jako string (gdy safe_type=tickers)."""
        return ", ".join(self.safe_tickers)

    @property
    def risky_weights_dict(self) -> Dict[str, float]:
        """Zwraca dict {ticker: fraction} (wagi jako ułamek 0–1)."""
        total = sum(a["weight"] for a in self.risky_assets) or 100.0
        return {a["ticker"]: a["weight"] / total for a in self.risky_assets}

    @property
    def blended_rate(self) -> float:
        """Efektywna stopa mieszana portfela (safe + risky nominal mean 8%)."""
        risky_nominal = 0.08  # przybliżona oczekiwana stopa ryzykownych
        return self.alloc_safe_pct * self.safe_rate + self.alloc_risky_pct * risky_nominal

    def total_risky_weight(self) -> float:
        return sum(a["weight"] for a in self.risky_assets)


# ─── Funkcje I/O ──────────────────────────────────────────────────────────────

def load_global_settings() -> GlobalPortfolio:
    """
    Wczytuje GlobalPortfolio z pliku JSON.
    Jeśli plik nie istnieje lub jest uszkodzony — zwraca wartości fabryczne.
    """
    if not os.path.exists(GLOBAL_SETTINGS_PATH):
        return GlobalPortfolio()

    try:
        with open(GLOBAL_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data: dict = json.load(f)

        gs = GlobalPortfolio(
            safe_type=data.get("safe_type", DEFAULT_SAFE_TYPE),
            safe_rate=float(data.get("safe_rate", DEFAULT_SAFE_RATE)),
            safe_tickers=data.get("safe_tickers", list(DEFAULT_SAFE_TICKERS)),
            risky_assets=data.get("risky_assets", copy.deepcopy(DEFAULT_RISKY_ASSETS)),
            alloc_safe_pct=float(data.get("alloc_safe_pct", DEFAULT_SAFE_ALLOCATION)),
            initial_capital=float(data.get("initial_capital", DEFAULT_INITIAL_CAPITAL)),
            bg_refresh_enabled=bool(data.get("bg_refresh_enabled", True)),
            bg_refresh_interval_minutes=int(data.get("bg_refresh_interval_minutes", 15)),
            language=data.get("language", "pl"),
            visible_modules=data.get("visible_modules", []),
            profile_name=data.get("profile_name", "Domyślny"),
            last_updated=data.get("last_updated", ""),
        )
        return gs

    except Exception as e:
        # Uszkodzony plik — fallback do domyślnych
        import warnings
        warnings.warn(f"global_settings.py: błąd odczytu JSON ({e}). Używam wartości domyślnych.")
        return GlobalPortfolio()


def save_global_settings(gs: GlobalPortfolio) -> bool:
    """
    Zapisuje GlobalPortfolio do pliku JSON w katalogu głównym projektu.
    Zwraca True przy sukcesie, False przy błędzie.
    """
    try:
        data = asdict(gs)
        data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(GLOBAL_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Aktualizuj też last_updated w obiekcie
        gs.last_updated = data["last_updated"]
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"global_settings.py: błąd zapisu JSON ({e}).")
        return False


# ─── session_state bridge ─────────────────────────────────────────────────────

def get_gs() -> GlobalPortfolio:
    """
    Pobiera aktualny GlobalPortfolio z session_state.
    Jeśli nie istnieje — wczytuje z pliku JSON (lub tworzy domyślny) i zapisuje w session_state.
    Bezpieczne do wywołania na początku każdej strony Streamlit.
    """
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = load_global_settings()
    return st.session_state[SESSION_KEY]


def set_gs(gs: GlobalPortfolio) -> None:
    """Zapisuje GlobalPortfolio do session_state."""
    st.session_state[SESSION_KEY] = gs


def apply_gs_to_session(gs: Optional[GlobalPortfolio] = None) -> None:
    """
    Wstrzykuje wartości GlobalPortfolio do kluczy `_s.*` używanych przez
    istniejące moduły (Symulator, Stress Test itp.) jako wartości domyślne.

    Klucze `_s.*` są zapisywane przez mechanizm _save()/_saved() w modułach.
    Wstrzykujemy je TYLKO jeśli dany klucz jeszcze nie istnieje w session_state
    (żeby nie nadpisywać lokalnych, ręcznych zmian użytkownika).
    """
    if gs is None:
        gs = get_gs()

    # ── Język interfejsu ─────────────────────────────────────────────────────
    st.session_state["_lang"] = gs.language

    # ── Symulator MC ────────────────────────────────────────────────────────
    _set_default("_s.mc_alloc_safe",  int(round(gs.alloc_safe_pct * 100)))
    _set_default("_s.mc_cap",         int(gs.initial_capital))

    # ── Symulator AI Backtest ───────────────────────────────────────────────
    if gs.safe_type == "fixed":
        _set_default("_s.ai_safe_type", "Holistyczne Obligacje Skarbowe (TOS 5.51%)")
    else:
        _set_default("_s.ai_safe_type", "Tickers (Yahoo)")
        _set_default("_s.ai_safe_tickers", gs.safe_tickers_str)

    _set_default("_s.ai_safe_rate",         round(gs.safe_rate * 100, 4))
    _set_default("_s.ai_risky_tickers",     gs.risky_tickers_str)
    _set_default("_s.ai_alloc_safe_slider", int(round(gs.alloc_safe_pct * 100)))
    _set_default("_s.ai_cap",               int(gs.initial_capital))

    # ── Stress Test ─────────────────────────────────────────────────────────
    _set_default("_s.st_safe",  gs.safe_tickers_str if gs.safe_tickers else "TLT, GLD")
    _set_default("_s.st_risky", gs.risky_tickers_str)
    _set_default("_s.st_sw",    int(round(gs.alloc_safe_pct * 100)))
    _set_default("_s.st_cap",   int(gs.initial_capital))

    # ── Emerytura ───────────────────────────────────────────────────────────
    _set_default("rem_initial_capital", gs.initial_capital)

    # ── Wealth Optimizer ────────────────────────────────────────────────────
    _set_default("_gs_wealth_total", gs.initial_capital)


def _set_default(key: str, value) -> None:
    """Ustawia wartość w session_state tylko jeśli klucz jeszcze nie istnieje."""
    if key not in st.session_state:
        st.session_state[key] = value


def force_apply_gs_to_session(gs: Optional[GlobalPortfolio] = None) -> None:
    """
    Jak apply_gs_to_session, ale NADPISUJE istniejące klucze.
    Używane gdy użytkownik kliknie "↩ Przywróć z Globalnych".
    """
    if gs is None:
        gs = get_gs()

    st.session_state["_s.mc_alloc_safe"]  = int(round(gs.alloc_safe_pct * 100))
    st.session_state["_s.mc_cap"]         = int(gs.initial_capital)

    if gs.safe_type == "fixed":
        st.session_state["_s.ai_safe_type"] = "Holistyczne Obligacje Skarbowe (TOS 5.51%)"
    else:
        st.session_state["_s.ai_safe_type"]    = "Tickers (Yahoo)"
        st.session_state["_s.ai_safe_tickers"] = gs.safe_tickers_str

    st.session_state["_s.ai_safe_rate"]         = round(gs.safe_rate * 100, 4)
    st.session_state["_s.ai_risky_tickers"]     = gs.risky_tickers_str
    st.session_state["_s.ai_alloc_safe_slider"] = int(round(gs.alloc_safe_pct * 100))
    st.session_state["_s.ai_cap"]               = int(gs.initial_capital)

    st.session_state["_s.st_safe"]  = gs.safe_tickers_str if gs.safe_tickers else "TLT, GLD"
    st.session_state["_s.st_risky"] = gs.risky_tickers_str
    st.session_state["_s.st_sw"]    = int(round(gs.alloc_safe_pct * 100))
    st.session_state["_s.st_cap"]   = int(gs.initial_capital)

    st.session_state["rem_initial_capital"] = gs.initial_capital
    st.session_state["_gs_wealth_total"]    = gs.initial_capital


def gs_sidebar_badge() -> None:
    """
    Renderuje miniaturkę aktualnych ustawień globalnych na dole sidebara.
    Wywołaj na końcu sidebara w każdym module.
    """
    from modules.i18n import t
    gs = get_gs()
    risky_preview = ", ".join(
        f"{a['ticker']} {a['weight']:.0f}%" for a in gs.risky_assets[:3]
    )
    if len(gs.risky_assets) > 3:
        risky_preview += f" +{len(gs.risky_assets)-3}"

    safe_suffix = "@ " + f"{gs.safe_rate*100:.2f}%" if gs.safe_type == "fixed" else t("gs_safe_basket")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="
            background: rgba(0,204,255,0.08);
            border: 1px solid rgba(0,204,255,0.25);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 11px;
            color: #94a3b8;
            margin-top: 4px;
        ">
        🌐 <b style="color:#00ccff">{t('gs_global_badge')}</b><br>
        🔒 {t('gs_badge_safe')}: <b>{gs.alloc_safe_pct:.0%}</b>
        {safe_suffix}<br>
        ⚡ {t('gs_badge_risky')}: {risky_preview}<br>
        💰 {t('gs_badge_capital')}: <b>{gs.initial_capital:,.0f} PLN</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button(t("gs_change_btn"), key="_gs_badge_btn", use_container_width=True):
        st.switch_page("pages/0_Globalne_Ustawienia.py")
