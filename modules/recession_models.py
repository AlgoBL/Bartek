"""
recession_models.py — Rozbudowane modele recesji i nowcastingu
=============================================================
Modele:
  - Reguła Sahm (Claudia Sahm, Fed)
  - Estrella-Mishkin Probit (Fed NY)
  - Reguła 2/10 (popularna heurystyka)
  - Composite Leading Indicator (CLI)
  - Hamilton Markov Switching (uproszczony)
  - Baumeister-Hamilton Oil Price Impact
  - Kansas City Fed: Dual Mandate Tracker
  - Wizualizacja Plotly (cyberpunk theme)
"""

import numpy as np
import pandas as pd
import datetime
from scipy.stats import norm
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1. REGUŁA SAHM
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sahm_rule(unemployment_series: pd.Series) -> pd.DataFrame:
    """
    Reguła Sahm: recesja gdy 3-mies. SMA bezrobocia wzrośnie ≥ 0.50pp
    powyżej minimum z poprzednich 12 miesięcy.
    Ref: Sahm (2019) "Direct Stimulus Payments to Individuals"
    """
    if unemployment_series.empty or len(unemployment_series) < 12:
        return pd.DataFrame()

    df = pd.DataFrame({'unrate': unemployment_series})
    df['3m_sma'] = df['unrate'].rolling(window=3).mean()
    df['12m_min_3m_sma'] = df['3m_sma'].rolling(window=12, min_periods=3).min()
    df['sahm_indicator'] = df['3m_sma'] - df['12m_min_3m_sma']
    df['is_recession_signal'] = df['sahm_indicator'] >= 0.50
    df['alert_level'] = df['sahm_indicator'].apply(
        lambda x: 'RECESJA' if x >= 0.50 else ('OSTRZEŻENIE' if x >= 0.30 else 'NORMALNIE')
    )
    return df.dropna(subset=['sahm_indicator'])


# ─────────────────────────────────────────────────────────────────────────────
#  2. ESTRELLA-MISHKIN PROBIT (Yield Curve)
# ─────────────────────────────────────────────────────────────────────────────

def prob_recession_estrella_mishkin(spread_10y_3m: float) -> float:
    """
    Model probitowy Estrella-Mishkin.
    P(Recesja za 12 mies.) = Φ(-2.17 - 0.76 * Spread)
    Ref: Estrella & Mishkin (1998) Federal Reserve Bank NY
    """
    z = -2.17 - 0.76 * spread_10y_3m
    return float(norm.cdf(z))


def batch_recession_estrella_mishkin(spread_history: pd.Series) -> pd.Series:
    """Batchowa wersja modelu E-M dla serii historycznych."""
    z = -2.17 - 0.76 * spread_history
    return pd.Series(norm.cdf(z), index=spread_history.index, name="recession_prob_em")


# ─────────────────────────────────────────────────────────────────────────────
#  3. REGUŁA WRIGHT (Stopy krótkoterminowe + Bezrobocie)
# ─────────────────────────────────────────────────────────────────────────────

def prob_recession_wright(spread_10y_3m: float, ff_rate: float) -> float:
    """
    Model Wrighta (2006): uwzględnia zarówno spread jak i poziom stóp FF.
    P(Recesja) = Φ(-1.29 - 2.45*spread - 0.70*ff_rate²)
    Ref: Wright (2006) "The Yield Curve and Predicting Recessions" Fed Reserve.
    """
    z = -1.29 - 2.45 * spread_10y_3m - 0.70 * (ff_rate ** 2)
    return float(norm.cdf(z))


def batch_recession_wright(spread_df: pd.DataFrame) -> pd.Series:
    """
    spread_df must have columns: 'spread_10y_3m', 'ff_rate'
    """
    if 'spread_10y_3m' not in spread_df or 'ff_rate' not in spread_df:
        return pd.Series(dtype=float)
    z = -1.29 - 2.45 * spread_df['spread_10y_3m'] - 0.70 * (spread_df['ff_rate'] ** 2)
    return pd.Series(norm.cdf(z), index=spread_df.index, name="recession_prob_wright")


# ─────────────────────────────────────────────────────────────────────────────
#  4. COMPOSITE RECESSION SCORE (agregat kilku modeli)
# ─────────────────────────────────────────────────────────────────────────────

def composite_recession_score(
    spread_10y_3m: float,
    ff_rate: float = 4.5,
    sahm_indicator: float = 0.0,
    hy_spread_bps: float = 400.0,
    leading_index_mom: float = 0.0,  # % MoM change in LEI
    weights: Optional[dict] = None
) -> dict:
    """
    Zagregowany wskaźnik recesji z wielu modeli.
    
    Returns:
        composite_prob: 0-1 prawdopodobieństwo recesji
        component_scores: dict z wkładami każdego modelu
        risk_label: 'LOW' / 'ELEVATED' / 'HIGH' / 'RECESSION'
    """
    if weights is None:
        weights = {
            'estrella_mishkin': 0.30,
            'wright': 0.20,
            'sahm': 0.20,
            'hy_spread': 0.15,
            'lei': 0.15,
        }

    # 1. Estrella-Mishkin
    em = prob_recession_estrella_mishkin(spread_10y_3m)

    # 2. Wright
    wr = prob_recession_wright(spread_10y_3m, ff_rate)

    # 3. Sahm rule → probability proxy
    sahm_prob = min(1.0, max(0.0, sahm_indicator / 0.5))  # 0.5 = trigger

    # 4. HY Spread → probability proxy (400bps normal, 800bps crisis)
    hy_prob = min(1.0, max(0.0, (hy_spread_bps - 300) / 500))

    # 5. LEI MoM → falling = recession signal
    lei_prob = min(1.0, max(0.0, -leading_index_mom / 2.0 + 0.5))

    components = {
        'estrella_mishkin': em,
        'wright': wr,
        'sahm': sahm_prob,
        'hy_spread': hy_prob,
        'lei': lei_prob,
    }

    composite = sum(components[k] * weights.get(k, 0) for k in components)
    composite = float(min(1.0, max(0.0, composite)))

    if composite < 0.15:
        label = 'LOW'
    elif composite < 0.35:
        label = 'ELEVATED'
    elif composite < 0.60:
        label = 'HIGH'
    else:
        label = 'RECESSION'

    return {
        'composite_prob': composite,
        'composite_pct': composite * 100,
        'component_scores': components,
        'risk_label': label,
        'estrella_mishkin_prob': em,
        'wright_prob': wr,
        'sahm_indicator': sahm_indicator,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. HAMILTON MARKOV-SWITCHING (uproszczony, 2-state)
# ─────────────────────────────────────────────────────────────────────────────

def markov_switching_filter(gdp_growth: pd.Series,
                             mu_expansion: float = 0.7,
                             mu_recession: float = -0.5,
                             sigma: float = 1.5,
                             p_stay_exp: float = 0.95,
                             p_stay_rec: float = 0.75) -> pd.DataFrame:
    """
    Uproszczony Hamilton (1989) Hidden Markov Model (2 stany).
    State 0 = Ekspansja, State 1 = Recesja.
    
    Ref: Hamilton (1989) "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
    """
    if len(gdp_growth) < 4:
        return pd.DataFrame()

    T = len(gdp_growth)
    y = gdp_growth.values

    # Transition matrix
    P_mat = np.array([[p_stay_exp, 1 - p_stay_exp],
                      [1 - p_stay_rec, p_stay_rec]])

    # Emission probabilities (Gaussian)
    def gaussian_pdf(x, mu, sig):
        return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    # Forward algorithm
    prob_recession = np.zeros(T)
    filtered = np.zeros((T, 2))  # [p_expansion, p_recession]

    # Init: assume starting in expansion
    filtered[0] = np.array([0.8, 0.2])

    for t in range(1, T):
        em = np.array([
            gaussian_pdf(y[t], mu_expansion, sigma),
            gaussian_pdf(y[t], mu_recession, sigma)
        ])
        predicted = P_mat.T @ filtered[t - 1]
        updated = em * predicted
        total = updated.sum()
        filtered[t] = updated / max(total, 1e-15)

    prob_recession = filtered[:, 1]

    return pd.DataFrame({
        'gdp_growth': y,
        'prob_expansion': filtered[:, 0],
        'prob_recession': prob_recession,
        'regime': (prob_recession > 0.5).astype(int),
        'regime_label': ['RECESJA' if r > 0.5 else 'EKSPANSJA' for r in prob_recession],
    }, index=gdp_growth.index)


# ─────────────────────────────────────────────────────────────────────────────
#  6. DANE SYMULOWANE (fallback gdy brak FRED)
# ─────────────────────────────────────────────────────────────────────────────

def load_simulated_sahm_data() -> pd.Series:
    """Mockowane historyczne dane bezrobocia UNRATE z realistycznymi cyklami."""
    dates = pd.date_range(start="2000-01-01", end=datetime.date.today(), freq="ME")
    import math
    base = 4.5
    values = []
    for d in dates:
        v = base
        if 2001 <= d.year <= 2003:
            v += 1.5 * math.sin(math.pi * (d.year - 2001) / 2.0)
        if 2008 <= d.year <= 2011:
            v += 5.0 * math.sin(math.pi * (d.year - 2008) / 3.0)
        if d.year == 2020 and d.month > 3:
            v += 8.0 * math.exp(-(d.month - 4))
        if 2023 <= d.year <= datetime.date.today().year:
            v += 0.3 * (d.year - 2022) + 0.05 * d.month
        v += np.random.normal(0, 0.1)
        values.append(max(v, 3.4))
    np.random.seed(None)
    return pd.Series(values, index=dates, name="UNRATE")


def load_simulated_yield_spread() -> pd.Series:
    """Mockowany 10Y-3M spread z historycznymi inwersami przed recesjami."""
    dates = pd.date_range(start="2000-01-01", end=datetime.date.today(), freq="ME")
    import math
    values = []
    for d in dates:
        v = 1.5
        if 2000 <= d.year <= 2001: v = -0.5 + np.random.normal(0, 0.15)
        elif 2006 <= d.year <= 2007: v = -0.6 + np.random.normal(0, 0.15)
        elif d.year == 2019: v = -0.1 + np.random.normal(0, 0.2)
        elif 2022 <= d.year <= 2023: v = -1.2 + np.random.normal(0, 0.2)
        elif d.year >= 2024: v = 0.1 + (d.year - 2024) * 0.2 + np.random.normal(0, 0.15)
        else: v = 1.5 + np.random.normal(0, 0.2)
        values.append(v)
    return pd.Series(values, index=dates, name="Spread_10Y_3M")


def load_simulated_gdp_growth() -> pd.Series:
    """Mockowany kwartalny wzrost PKB (% QoQ ann.) z recesyjnymi dołkami."""
    dates = pd.date_range(start="2000-01-01", end=datetime.date.today(), freq="QE")
    import math
    values = []
    for d in dates:
        v = 2.5
        if 2001 <= d.year <= 2002: v = -0.5 + np.random.normal(0, 0.5)
        elif 2008 <= d.year <= 2009: v = -3.5 + np.random.normal(0, 1.0)
        elif d.year == 2020 and d.month <= 6: v = -7.0 + np.random.normal(0, 1.5)
        elif d.year == 2020 and d.month > 6: v = 6.0 + np.random.normal(0, 1.0)
        else: v = 2.5 + np.random.normal(0, 0.5)
        values.append(v)
    return pd.Series(values, index=dates, name="GDP_QoQ_Ann")
