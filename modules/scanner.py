
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import streamlit as st
from modules.metrics import calculate_sharpe, calculate_sortino, calculate_max_drawdown

def hill_estimator(returns, tail_fraction=0.05):
    """
    Oblicza Estymator Hilla (Hill Index) dla prawego ogona rozkładu zwrotów.
    Alpha < 3.0 oznacza gruby ogon (Fat Tail).
    Alpha < 2.0 oznacza nieskończoną wariancję (ekstremalne ryzyko/zysk).
    """
    if len(returns) < 20:
        return np.nan
        
    # Sortujemy zwroty malejąco (największe zyski na początku)
    sorted_returns = np.sort(returns)[::-1]
    
    # Bierzemy tylko dodatnie zwroty do analizy prawego ogona
    positive_returns = sorted_returns[sorted_returns > 0]
    
    if len(positive_returns) < 10:
        return np.nan

    # Ustalamy liczbę obserwacji w ogonie (k)
    k = int(len(positive_returns) * tail_fraction)
    if k < 2:
        k = 2
        
    # Wybieramy k największych zwrotów
    tail_returns = positive_returns[:k]
    x_k_plus_1 = positive_returns[k] # Próg odcięcia
    
    # Wzór Hilla: 1 / (mean(ln(Xi / X_k+1)))
    log_ratios = np.log(tail_returns / x_k_plus_1)
    gamma = np.mean(log_ratios)
    
    if gamma == 0:
        return np.nan
        
    alpha = 1.0 / gamma
    return alpha

def calculate_convecity_metrics(ticker, price_series, benchmark_series=None):
    """
    Oblicza zestaw metryk dla Skanera Wypukłości.
    """
    # Obliczamy zwroty logarytmiczne
    returns = np.log(price_series / price_series.shift(1)).dropna()
    
    if len(returns) < 30:
        return None

    # 1. Podstawowe statystyki
    vol_ann = returns.std() * np.sqrt(252)
    mean_ann = returns.mean() * 252
    
    # 2. Wyższe momenty (odrzucamy Gaussianity)
    skew_val = skew(returns)
    kurt_val = kurtosis(returns) # Excess kurtosis (Fisher)
    
    # 3. Professional Metrics
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_dd = calculate_max_drawdown(price_series)
    
    # 3. Estymator Hilla (Prawy Ogon - Zyski)
    alpha_hill = hill_estimator(returns.values)
    
    # 4. Ryzyko Oporu Wariancji (Variance Drag)
    # R_Geom approx R_Arith - 0.5 * sigma^2
    var_drag = 0.5 * (vol_ann ** 2)
    
    # 5. Kelly (Uproszczony dla 0 stopy wolnej od ryzyka, lub hardcoded)
    risk_free = 0.04
    if vol_ann > 0:
        kelly_full = (mean_ann - risk_free) / (vol_ann ** 2)
        # Factor kurczenia (Shrinkage) - Hardcoded 50% safety
        kelly_safe = kelly_full * 0.5
    else:
        kelly_full = 0
        kelly_safe = 0
        
    return {
        "Ticker": ticker,
        "Annual Return": mean_ann,
        "Volatility": vol_ann,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "Hill Alpha (Tail)": alpha_hill,
        
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        
        "Variance Drag": var_drag,
        "Kelly Full": kelly_full,
        "Kelly Safe (50%)": kelly_safe
    }

def score_asset(metrics):
    """
    Ocenia aktywo punktowo pod kątem przydatności do strategii Barbell.
    Nagradzamy: Niskie Alpha Hilla, Wysoki Skew, Zmiennosc (jesli skew > 0).
    """
    if metrics is None:
        return -999
        
    score = 0
    
    # 1. Hill Alpha (Im niżej tym lepiej, celujemy w 1.5 - 2.5)
    alpha = metrics["Hill Alpha (Tail)"]
    if not np.isnan(alpha):
        if 1.0 < alpha < 3.0:
            score += 50
        if 1.5 < alpha < 2.5: # Sweet spot
            score += 20
        # Penalizacja za zbyt cienki ogon
        if alpha > 4.0:
            score -= 20
            
    # 2. Skośność (Musi być dodatnia)
    if metrics["Skewness"] > 0:
        score += 30 * metrics["Skewness"] # Promujemy wysoki skew
    else:
        score -= 50 # Dyskwalifikacja ujemnej skośności (ryzyko lewego ogona)
        
    # 3. Kelly (Musi być dodatni - aktywo musi zarabiać)
    if metrics["Kelly Full"] > 0.1:
        score += 20
    elif metrics["Kelly Full"] <= 0:
        score -= 30
        
    return score
