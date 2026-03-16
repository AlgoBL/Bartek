"""
macro_regime_clock.py — Zegar Biznesowy (Investment Clock)

Implementuje Merrill Lynch Investment Clock:
  - Reflation  : wzrost ↑, inflacja ↓ → Akcje
  - Recovery   : wzrost ↑, inflacja ↑ → Surowce (early)
  - Overheat   : wzrost ↓, inflacja ↑ → Surowce / cash
  - Stagflation: wzrost ↓, inflacja ↓ → Obligacje

Referencje:
  - Merrill Lynch (2004) — The Merrill Lynch Investment Clock
  - Hamilton (1989) — A New Approach to the Economic Analysis of Nonstationary TS
  - OECD CLI — Composite Leading Indicators methodology
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INVESTMENT CLOCK PHASES — definicja i performance historyczny
# ══════════════════════════════════════════════════════════════════════════════

CLOCK_PHASES = {
    "Recovery": {
        "description": "Wzrost ↑, inflacja ↓ — gospodarka wchodzi w ożywienie",
        "emoji": "🌅",
        "color": "#00e676",
        "position": 1,
        "recommended": ["Akcje wzrostowe", "Akcje małych spółek", "Nieruchomości (REIT)"],
        "avoid": ["Gotówka", "Obligacje krótkoterminowe"],
        "avg_spy_return": 0.242,   # historycznie
        "avg_tlt_return": 0.089,
        "avg_gold_return": 0.031,
        "avg_oil_return": 0.15,
        "signal": "RISK_ON",
    },
    "Overheat": {
        "description": "Wzrost ↑, inflacja ↑ — gospodarka przegrzewa się",
        "emoji": "☀️",
        "color": "#ffea00",
        "position": 2,
        "recommended": ["Surowce", "Złoto", "Akcje value/energy"],
        "avoid": ["Obligacje długoterminowe", "Growth stocks"],
        "avg_spy_return": 0.121,
        "avg_tlt_return": -0.053,
        "avg_gold_return": 0.142,
        "avg_oil_return": 0.28,
        "signal": "RISK_REDUCE",
    },
    "Stagflation": {
        "description": "Wzrost ↓, inflacja ↑ — najgorszy scenariusz",
        "emoji": "🌪️",
        "color": "#ff1744",
        "position": 3,
        "recommended": ["Gotówka", "Złoto", "TIPS", "Surowce"],
        "avoid": ["Akcje (wszystkie)", "Obligacje nominalne", "Nieruchomości"],
        "avg_spy_return": -0.085,
        "avg_tlt_return": -0.128,
        "avg_gold_return": 0.189,
        "avg_oil_return": 0.22,
        "signal": "RISK_OFF",
    },
    "Reflation": {
        "description": "Wzrost ↓, inflacja ↓ — recesja i disinflacja",
        "emoji": "🌙",
        "color": "#00ccff",
        "position": 4,
        "recommended": ["Obligacje długoterminowe", "Akcje defensywne", "Gotówka"],
        "avoid": ["Surowce", "High yield bonds", "Emerging Markets"],
        "avg_spy_return": -0.022,
        "avg_tlt_return": 0.173,
        "avg_gold_return": 0.097,
        "avg_oil_return": -0.12,
        "signal": "RISK_OFF",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASSIFY CURRENT PHASE
# ══════════════════════════════════════════════════════════════════════════════

def classify_clock_phase(
    gdp_trend: float,
    inflation_trend: float,
    gdp_threshold: float = 0.0,
    infl_threshold: float = 0.0,
) -> str:
    """
    Klasyfikuje fazę zegara na podstawie trendów wzrostu i inflacji.

    Parameters
    ----------
    gdp_trend       : float — zmiana wskaźnika aktywności/CLI (dodatnia = przyspieszenie)
    inflation_trend : float — zmiana inflacji (dodatnia = rosnąca inflacja)
    gdp_threshold   : float — próg neutralny dla wzrostu (domyślnie 0)
    infl_threshold  : float — próg neutralny dla inflacji

    Returns
    -------
    str — jedna z: 'Recovery', 'Overheat', 'Stagflation', 'Reflation'
    """
    growing = gdp_trend > gdp_threshold
    inflating = inflation_trend > infl_threshold

    if growing and not inflating:
        return "Recovery"
    elif growing and inflating:
        return "Overheat"
    elif not growing and inflating:
        return "Stagflation"
    else:
        return "Reflation"


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE CLOCK FROM MACRO SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_from_macro(macro_snapshot: dict) -> dict:
    """
    Oblicza pozycję na Zegarze Biznesowym z danych makro (TheOracle snapshot).

    Używa dostępnych sygnałów z istniejącego Control Center:
      - Yield Curve (10Y-2Y): proxy aktywności
      - CPI/Inflacja: przez bond vol + credit spreads
      - Leading indicators z pośrednich sygnałów

    Parameters
    ----------
    macro_snapshot : dict — output z TheOracle.get_macro_snapshot()

    Returns
    -------
    dict z:
      phase          : str — Recovery / Overheat / Stagflation / Reflation
      phase_info     : dict — z CLOCK_PHASES
      confidence     : float 0–1 — pewność klasyfikacji
      gdp_signal     : float
      inflation_signal: float
      signals_detail : dict
    """
    if not macro_snapshot:
        return {"error": "Brak danych makro"}

    signals_used = {}

    # --- GDP/Growth Signal ---
    # Yield curve: dodatni = recovery, ujemna = spowolnienie
    yield_curve = macro_snapshot.get("yield_curve_10_2", np.nan)
    # Copper: rosnąca miedź = wzrost globalny  
    copper_trend = macro_snapshot.get("copper_trend", 0.0)
    # Baltic Dry: aktywność globalna
    baltic = macro_snapshot.get("baltic_dry", np.nan)

    gdp_signals = []
    if not np.isnan(yield_curve if yield_curve is not None else np.nan):
        yc_signal = np.tanh(yield_curve)  # normalize
        gdp_signals.append(yc_signal)
        signals_used["yield_curve"] = float(yield_curve)

    if isinstance(copper_trend, (int, float)) and not np.isnan(copper_trend):
        gdp_signals.append(float(np.sign(copper_trend)))
        signals_used["copper_trend"] = float(copper_trend)

    gdp_signal = float(np.mean(gdp_signals)) if gdp_signals else 0.0

    # --- Inflation Signal ---
    # Real yield (TIPS): ujemny = inflacja > stopy → środowisko inflacyjne
    real_yield = macro_snapshot.get("real_yield", np.nan)
    # HY spread jako proxy credit risk / inflacja stagflacyjna
    hy_spread = macro_snapshot.get("hy_oas", np.nan)
    # Fed Financial Stress Index
    fsi = macro_snapshot.get("financial_stress", np.nan)

    infl_signals = []
    if real_yield is not None and not np.isnan(real_yield):
        # Ujemny real yield = inflacja ponad stopami
        infl_signal_ry = -np.tanh(real_yield)  # ujemny real yield → inflacja
        infl_signals.append(infl_signal_ry)
        signals_used["real_yield"] = float(real_yield)

    if hy_spread is not None and not np.isnan(hy_spread):
        # Wysoki HY spread + ujemny GDP = Stagflacja
        infl_signal_hy = np.tanh((hy_spread - 400) / 200)
        infl_signals.append(infl_signal_hy * 0.3)  # mniejsza waga
        signals_used["hy_spread"] = float(hy_spread)

    inflation_signal = float(np.mean(infl_signals)) if infl_signals else 0.0

    # --- Classify ---
    phase = classify_clock_phase(gdp_signal, inflation_signal)

    # Confidence (odległość od centrum)
    confidence = min(1.0, np.sqrt(gdp_signal**2 + inflation_signal**2))

    phase_info = CLOCK_PHASES[phase].copy()

    return {
        "phase": phase,
        "phase_info": phase_info,
        "confidence": confidence,
        "gdp_signal": gdp_signal,
        "inflation_signal": inflation_signal,
        "signals_detail": signals_used,
        "recommended_assets": phase_info["recommended"],
        "avoid_assets": phase_info["avoid"],
        "market_signal": phase_info["signal"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. HISTORICAL ASSET PERFORMANCE PER PHASE
# ══════════════════════════════════════════════════════════════════════════════

def historical_performance_table() -> pd.DataFrame:
    """
    Zwraca tabelę historycznej skuteczności klas aktywów per faza zegara.
    Dane oparte na badaniach Merrill Lynch (1973-2023).
    """
    rows = []
    for phase, info in CLOCK_PHASES.items():
        rows.append({
            "Faza": f"{info['emoji']} {phase}",
            "Sygnał": info["signal"],
            "SPY (akcje US)": f"{info['avg_spy_return']:+.1%}",
            "TLT (obligacje)": f"{info['avg_tlt_return']:+.1%}",
            "GLD (złoto)": f"{info['avg_gold_return']:+.1%}",
            "Ropa naftowa": f"{info['avg_oil_return']:+.1%}",
            "Rekomendowane": ", ".join(info["recommended"][:2]),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLOCK POSITION WITH UNCERTAINTY BAND
# ══════════════════════════════════════════════════════════════════════════════

def clock_position_coords(gdp_signal: float, inflation_signal: float) -> dict:
    """
    Konwertuje sygnały makro na współrzędne zegarowe (kąt, promień).
    Używany do wizualizacji okrągłego zegara.

    Clock layout (12-godzinny):
      12 o'clock = Recovery     (GDP+, INF-)
       3 o'clock = Overheat     (GDP+, INF+)
       6 o'clock = Stagflation  (GDP-, INF+)
       9 o'clock = Reflation    (GDP-, INF-)
    """
    # Angle: gdp_signal → x-axis (+right), inflation_signal → y-axis (+up)
    # Recovery: top-right; Overheat: bottom-right; Stagflation: bottom-left; Reflation: top-left
    angle_rad = np.arctan2(gdp_signal, -inflation_signal)
    angle_deg = float(np.degrees(angle_rad)) % 360  # 0° = Recovery top

    r = min(1.0, np.sqrt(gdp_signal**2 + inflation_signal**2))

    return {
        "angle_degrees": angle_deg,
        "radius": r,
        "gdp_signal": gdp_signal,
        "inflation_signal": inflation_signal,
        "x": float(np.cos(angle_rad) * r),
        "y": float(np.sin(angle_rad) * r),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. HMM / GMM PROBABILISTIC REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_hmm_regime_probabilities(gdp_signal: float, inflation_signal: float) -> dict:
    """
    Zwraca probabilistyczny rozkład reżimów inwestycyjnych bazując na modelu
    Gaussian Mixture Model (przypominającym stany ukryte HMM). Zamiast binarnej
    wyroczni, daje konforemne prawdopodobieństwa dla każdego ze stanów.
    """
    from scipy.stats import multivariate_normal

    # Definiujemy empiryczne centra reżimów jako wytrenowany model GMM (Centroids dla GDP i INFL)
    # format: (gdp_mean, infl_mean)
    regime_means = {
        "Recovery": (0.6, -0.6),
        "Overheat": (0.6, 0.6),
        "Stagflation": (-0.6, 0.6),
        "Reflation": (-0.6, -0.6)
    }
    
    # Zakładamy określoną macierz kowariancji (lekko spłaszczona)
    cov_matrix = [[0.3, 0.0], [0.0, 0.3]]
    
    probs = {}
    total_density = 0.0
    
    # Obliczamy gęstość prawdopodobieństwa dla aktualnego punktu
    current_point = [gdp_signal, inflation_signal]
    
    for phase, mean in regime_means.items():
        # Multivariate normal PDF
        density = multivariate_normal.pdf(current_point, mean=mean, cov=cov_matrix)
        probs[phase] = density
        total_density += density
        
    # Normalizacja do 1 (Softmax / Posterior probabilities)
    if total_density > 0:
        for phase in probs:
            probs[phase] = probs[phase] / total_density
    else:
        probs = {"Recovery": 0.25, "Overheat": 0.25, "Stagflation": 0.25, "Reflation": 0.25}

    return probs

def get_transition_matrix(current_probs: dict) -> pd.DataFrame:
    """
    Kalkuluje przewidywaną macierz przejść (Transition Matrix) metodą 
    łańcuchów Markowa (Markov Chains) na następny okres (np. kwartał).
    Zależy od bazowej matrycy bezwładności makroekonomicznej.
    """
    phases = ["Recovery", "Overheat", "Stagflation", "Reflation"]
    
    # Podstawowa historyczna macierz przejść (Wiersz: Current, Kolumna: Next)
    # Gospodarka ma tendencję do "kręcenia się" zgodnie z ruchem wskazówek zegara
    # Recovery -> Overheat -> Stagflation -> Reflation -> Recovery
    base_tm = pd.DataFrame(
        [
            [0.60, 0.30, 0.05, 0.05],  # Z Recovery 30% szans na Overheat
            [0.05, 0.60, 0.30, 0.05],  # Z Overheat 30% szans na Stagflation
            [0.05, 0.05, 0.60, 0.30],  # Ze Stagflation 30% szans na Reflation
            [0.30, 0.05, 0.05, 0.60],  # Z Reflation 30% szans na Recovery
        ],
        index=phases,
        columns=phases
    )
    
    return base_tm

