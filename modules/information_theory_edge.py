"""
information_theory_edge.py — Informational Edge & Decision Theory
====================================================================
Generalizacja narzędzi Teorii Informacji (Shannon/Thorp/Jaynes) do oceny 
jakości decyzji inwestycyjnych, wyliczania kapitału i detekcji anomalii 
("Smart Money" na giełdzie, zmiany reżimów).

Moduł ten służy jako warstwa decyzyjna (Meta-Layer) nad innymi modułami 
(np. uśrednianie sygnałów makro, ocena ryzyka zmiany w portfolio).

Funkcjonalności:
1. KL-Divergence: Ocena Przewagi (Edge) nad rynkiem/konsensusem.
2. Kelly Sizing: Agresywność alokacji kapitału na bazie Przewagi.
3. Max-Entropy Fusion: Fuzja sygnałów od różnych "wyroczni" w 1 wagę.
4. Smart Money Entropy: Zapadanie entropii wolumenu jako wskaźnik akumulacji.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCENA PRZEWAGI INFORMACYJNEJ (KL-DIVERGENCE)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_information_edge(p_mine: float, p_market: float) -> float:
    """
    Oblicza przewagę (edge) w bitach informacji (Rozbieżność Kullbacka-Leiblera).
    
    Zastosowanie w inwestycjach/życiu:
    - `p_mine`: Twoje oszacowanie sukcesu (np. start nowego biznesu, zysk ETF).
    - `p_market`: Statystyka rynkowa / obiektywna baza (np. 90% firm upada -> p=0.1).
    
    Interpretacja:
    > 0.10 bita - Silny edge informacyjny, warto podejmować ryzyko.
    < 0.05 bita - Brak edge, rzucasz monetą z rynkiem.
    """
    p_mine = np.clip(p_mine, 0.01, 0.99)
    p_market = np.clip(p_market, 0.01, 0.99)
    q_mine = 1 - p_mine
    q_market = 1 - p_market
    
    edge = p_mine * np.log2(p_mine / p_market) + q_mine * np.log2(q_mine / q_market)
    return float(edge)

# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTYMALIZACJA ALOKACJI (KELLY CRITERION)
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(p_mine: float, risk_reward_ratio: float = 1.0) -> float:
    """
    Ogólne Kryterium Kelly'ego dla dowolnego zakładu / inwestycji.
    
    - `p_mine`: Szansa na sukces (0.0 - 1.0).
    - `risk_reward_ratio`: Stosunek Zysku do Ryzyka (np. ryzykujesz 1k PLN, żeby wygrać 3k PLN -> ratio = 3.0).
    
    Zwraca procent kapitału (0.0 - 1.0), który maksymalizuje długoterminowy wzrost majątku (CAGR).
    W praktyce stosuje się "Half-Kelly" lub "Quarter-Kelly" dla złagodzenia zmienności.
    """
    if p_mine <= 0.0:
        return 0.0
    
    q_mine = 1 - p_mine
    kelly = p_mine - (q_mine / risk_reward_ratio)
    
    return float(np.clip(kelly, 0.0, 1.0))

def calculate_bet_sizing(p_mine: float, p_market: float, is_symmetric: bool = True) -> dict:
    """Oblicza pełny pakiet decyzyjny dla typowych szacunków."""
    edge_bits = calculate_information_edge(p_mine, p_market)
    
    # Dla giełdy, jeśli zakładamy symetrię, r/r wylicza się z p_market
    rr_ratio = (1 - p_market) / p_market if is_symmetric else 1.0
    
    full_kelly = kelly_fraction(p_mine, rr_ratio)
    
    return {
        "edge_bits": edge_bits,
        "full_kelly": full_kelly,
        "half_kelly": full_kelly / 2.0,
        "status": "Działaj" if edge_bits >= 0.10 and full_kelly > 0 else "Odpuść / Zbieraj dane"
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. FUZJA SYGNAŁÓW (MAX ENTROPY FUSION)
# ─────────────────────────────────────────────────────────────────────────────

def max_entropy_fusion(signals: List[float], confidences: List[float]) -> float:
    """
    Meta-Warstwa dla Twojej aplikacji.
    Łączy sygnały (np. z modułów: macro, sentiment, pairs) w jedną decyzję.
    
    Zasada Max-Entropy zakazuje dodawania "ukrytych założeń". Wynikiem jest
    czysta, ważona entropijnie średnia, odporna na overfitting modeli.
    """
    signals = np.array(signals)
    confidences = np.array(confidences)
    
    if np.sum(confidences) == 0:
        return float(np.mean(signals))
        
    fused_prob = np.sum(signals * confidences) / np.sum(confidences)
    return float(np.clip(fused_prob, 0.01, 0.99))

# ─────────────────────────────────────────────────────────────────────────────
# 4. DETEKCJA SMART MONEY (ENTROPY COLLAPSE NA WOLUMENIE)
# ─────────────────────────────────────────────────────────────────────────────

def volume_entropy_collapse(df: pd.DataFrame, window: int = 14, k: float = 2.5) -> pd.DataFrame:
    """
    Mierzy Entropię Profilu Wolumenu (Zapadanie Entropii).
    Zamiast rynków predykcyjnych, badamy tradycyjne giełdy (np. ETFy, akcje).
    
    Jak to działa?
    Jeśli wolumen rozkłada się równo w ciągu ostatnich 14 dni -> Entropia wysoka.
    Jeśli wolumen drastycznie kumuluje się w 1-2 dni (instytucje kupują) -> Entropia zapada się.
    
    Zwraca DataFrame z kolumną 'SmartMoney_Alert'.
    """
    d = df.copy()
    if 'Volume' not in d.columns:
        raise ValueError("DataFrame musi zawierać kolumnę 'Volume'")
        
    def calc_shannon_vol(vol_array):
        vol_array = vol_array[vol_array > 0]
        if len(vol_array) == 0:
            return 0.0
        p = vol_array / np.sum(vol_array)
        return -np.sum(p * np.log2(p))

    # Oblicz kroczącą entropię wolumenu
    d['Vol_Entropy'] = d['Volume'].rolling(window=window).apply(calc_shannon_vol, raw=True)
    
    # Delta entropii (tempo zmian)
    d['dH_dt'] = d['Vol_Entropy'].diff()
    d['sigma'] = d['dH_dt'].rolling(window=window).std()
    
    # Alert: Entropia zapada się bardzo szybko (poniżej -k odchyleń)
    d['SmartMoney_Alert'] = (d['dH_dt'] < 0) & (d['dH_dt'] < -k * d['sigma'])
    
    return d

# ─────────────────────────────────────────────────────────────────────────────
# WIZUALIZACJE PLOTLY DO STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def plot_smart_money_entropy(df: pd.DataFrame, ticker_name: str = "Aktywo") -> go.Figure:
    """Dwupanelowy wykres Ceny/Wolumenu i zapadania entropii (Smart Money)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{ticker_name} - Cena", "Entropia Wolumenu (Smart Money Alerts)"),
        row_heights=[0.6, 0.4]
    )

    if 'Close' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', line=dict(color="#e2e4f0", width=1.5), name="Cena"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Vol_Entropy'], mode='lines', line=dict(color="#00ccff", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,255,0.06)", name="Entropia Wol."
    ), row=2, col=1)

    alerts = df[df.get('SmartMoney_Alert', False)]
    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts.index, y=alerts['Vol_Entropy'], mode='markers',
            marker=dict(color="#00e676", size=12, symbol='star', line=dict(color="black", width=1)),
            name="Akumulacja Instytucjonalna"
        ), row=2, col=1)
        # Znaczniki na wykresie ceny dla widoczności
        if 'Close' in df.columns:
            fig.add_trace(go.Scatter(
                x=alerts.index, y=alerts['Close'], mode='markers',
                marker=dict(color="#00e676", size=10, symbol='triangle-up'),
                name="Sygnał"
            ), row=1, col=1)

    fig.update_layout(
        title=dict(text=f"Detektor Śladów Smart Money - {ticker_name}", font=dict(color="#e2e4f0")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        height=550, margin=dict(l=40, r=20, t=50, b=30), showlegend=False
    )
    fig.update_xaxes(gridcolor="#1c1c2e")
    fig.update_yaxes(gridcolor="#1c1c2e")
    return fig

def demo_information_theory() -> dict:
    """Generuje dane syntetyczne dla celów testowania modułu w UI."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    
    close = np.cumsum(np.random.normal(0, 1, 100)) + 100
    volume = np.random.lognormal(mean=10, sigma=0.5, size=100)
    
    # Symulacja "Smart Money" - gigantyczny wolumen przy bocznej cenie
    volume[50:52] = volume[50:52] * 8
    
    df = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
    df_analyzed = volume_entropy_collapse(df)
    
    return {
        "df": df_analyzed,
        "fig_smart_money": plot_smart_money_entropy(df_analyzed, "Syntetyczny ETF")
    }
