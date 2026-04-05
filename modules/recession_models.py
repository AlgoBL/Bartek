import numpy as np
import pandas as pd
import datetime

def calculate_sahm_rule(unemployment_series: pd.Series) -> pd.DataFrame:
    """
    Kalkulator Reguły Sahm.
    Zasada Sahm mówi, że rynki wchodzą w recesję, gdy 3-miesięczna średnia ruchoma
    krajowej stopy bezrobocia (U3) wzrośnie o 0.50 punktu procentowego lub więcej względem
    swojego minimum z poprzednich 12 miesięcy.
    """
    if unemployment_series.empty or len(unemployment_series) < 12:
        return pd.DataFrame()
        
    df = pd.DataFrame({'unrate': unemployment_series})
    df['3m_sma'] = df['unrate'].rolling(window=3).mean()
    
    # Minimum z ostatnich 12 miesięcy 3m_sma
    df['12m_min_3m_sma'] = df['3m_sma'].rolling(window=12, min_periods=1).min()
    
    df['sahm_indicator'] = df['3m_sma'] - df['12m_min_3m_sma']
    df['is_recession_signal'] = df['sahm_indicator'] >= 0.50
    
    return df.dropna()

def prob_recession_estrella_mishkin(spread_10y_3m: float) -> float:
    """
    Model probitowy Estrella-Mishkin. Przewiduje prawdopodobieństwo recesji w czasie t+12.
    Równanie: P(Recesja) = Φ(-2.17 - 0.76 * Spread)
    gdzie Spread to (10-Year Treasury - 3-Month Treasury) w punktach procentowych.
    """
    from scipy.stats import norm
    
    z = -2.17 - 0.76 * spread_10y_3m
    prob = norm.cdf(z)
    
    return prob

def batch_recession_estrella_mishkin(spread_history: pd.Series) -> pd.Series:
    from scipy.stats import norm
    z = -2.17 - 0.76 * spread_history
    return pd.Series(norm.cdf(z), index=spread_history.index, name="recession_prob")

def load_simulated_sahm_data() -> pd.Series:
    """ Zwraca mockowane historyczne dane bezrobocia (UNRATE) jeśli brak API FRED. """
    dates = pd.date_range(start="2000-01-01", end=datetime.date.today(), freq="ME")
    # Generujmy sztuczne cykle bezrobocia (wzrosty rzędu 5-10% co +- 10 lat)
    # T+=10, 2008, 2020 peaks
    
    base = 4.5
    values = []
    import math
    for d in dates:
        v = base
        # Dotcom
        if 2001 <= d.year <= 2003:
            v += 1.5 * math.sin(math.pi * (d.year - 2001)/2.0)
        # GFC
        if 2008 <= d.year <= 2011:
            v += 5.0 * math.sin(math.pi * (d.year - 2008)/3.0)
        # Covid
        if d.year == 2020:
             if d.month > 3:
                 v += 8.0 * math.exp(-(d.month-4))
        
        # Ostatnie dane (aktualne podwyżenie bezrobocia)
        if 2023 <= d.year <= datetime.date.today().year:
             v += 0.3 * (d.year - 2022) + 0.1 * d.month
            
        # Szum
        v += np.random.normal(0, 0.1)
        values.append(max(v, 3.4))
        
    return pd.Series(values, index=dates)

def load_simulated_yield_spread() -> pd.Series:
    """ Zwraca mockowane 10Y-3M spread. Ujemny przed recesjami. """
    dates = pd.date_range(start="2000-01-01", end=datetime.date.today(), freq="ME")
    
    values = []
    import math
    for d in dates:
        v = 1.5 # normalna stroma krzywa
        # Zmiany przed krachami
        if 2000 <= d.year <= 2001: v = -0.5
        if 2006 <= d.year <= 2007: v = -0.6
        if 2019 == d.year: v = -0.1
        if 2022 <= d.year <= 2023: v = -1.2
        if 2024 <= d.year: v = 0.1 + (d.year - 2024) * 0.2
        
        v += np.random.normal(0, 0.2)
        values.append(v)
        
    return pd.Series(values, index=dates)
