import numpy as np
from scipy.stats import norm

def black_scholes_merton(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call"):
    """
    Wycena opcji europejskich modelem Blacka-Scholesa-Mertona oraz wyliczenie Greków.
    S - cena instrumentu bazowego
    K - cena wykonania (strike)
    T - czas do wygaśnięcia (w latach)
    r - stopa wolna od ryzyka
    sigma - zmienność implikowana
    q - stopa dywidendy (ciągła)
    """
    if T <= 0:
        if option_type == "call":
            return {"price": max(0.0, S - K), "delta": 1.0 if S > K else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        else:
            return {"price": max(0.0, K - S), "delta": -1.0 if K > S else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Cena i Delta
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    # Gamma, Vega, Theta
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    if option_type == "call":
        theta = (- (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2) 
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        theta = (- (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0, # Zazwyczaj wyrażana dla zmiany sigma o 1 pkt proc.
        "theta": theta / 365.0 # Zazwyczaj wyrażana w dniach
    }

def binomial_tree_real_option(S: float, K: float, T: float, r: float, sigma: float, steps: int = 100, option_type: str = "call"):
    """
    Wycena opcji realnych (często amerykańskich) za pomocą drzewa dwumianowego Cox-Ross-Rubinstein.
    Pozwala na wczesne wykonanie (w przeciwieństwie do BSM).
    Używane np. do wyceny opcji ekspansji (call) lub opcji porzucenia projektu (put).
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Inicjalizacja liści drzewa (ceny bazowe w momencie T)
    asset_prices = np.zeros(steps + 1)
    option_values = np.zeros(steps + 1)

    for j in range(steps + 1):
        asset_prices[j] = S * (u ** (steps - j)) * (d ** j)
        if option_type == "call":
            option_values[j] = max(0.0, asset_prices[j] - K)
        else:
            option_values[j] = max(0.0, K - asset_prices[j])

    # Propagacja wsteczna (Backward Induction) z możliwością wczesnego wykonania
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
            # Cena instrumentu bazowego w tym węźle
            current_asset_price = S * (u ** (i - j)) * (d ** j)
            
            # Warunek wczesnego wykonania (Amerykańska/Realna)
            if option_type == "call":
                early_exercise = max(0.0, current_asset_price - K)
            else:
                early_exercise = max(0.0, K - current_asset_price)
                
            option_values[j] = max(option_values[j], early_exercise)

    return option_values[0]

def merton_structural_credit_risk(V: float, D: float, T: float, r: float, sigma_v: float):
    """
    Model Mertona wyceny ryzyka kredytowego.
    Traktuje kapitał własny (Equity) firmy jako opcję Call na jej Aktywa (V) z ceną wykonania równą Długowi (D).
    Wylicza Prawdopodobieństwo Bankructwa (Probability of Default - PD) w ujęciu miary ryzyka naturalnego.
    V - Wartość Aktywów
    D - Zobowiązania (Wartość nominalna długu)
    T - Czas do zapadalności długu
    r - Stopa wolna od ryzyka
    sigma_v - Zmienność wartości aktywów
    """
    # Odległość do bankructwa (Distance to Default - DD)
    # Zauważ, że PD = N(-d2) w mierze Black-Scholesa
    
    if V <= 0 or D <= 0 or T <= 0:
        return {"distance_to_default": 0, "probability_of_default": 1.0, "equity_value": 0.0}
        
    d1 = (np.log(V / D) + (r + 0.5 * sigma_v**2) * T) / (sigma_v * np.sqrt(T))
    d2 = d1 - sigma_v * np.sqrt(T)
    
    # Kapitał własny (Equity)
    equity = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    
    # Rynkowa wartość długu
    debt_value = V - equity
    
    # Credit Spread (Premia za ryzyko kredytowe)
    # Yield długu = (D / debt_value)^(1/T) - 1 => stopa ciągła to (1/T) * ln(D / debt_value)
    if debt_value > 0:
        yield_debt = (1 / T) * np.log(D / debt_value)
        credit_spread = yield_debt - r
    else:
        credit_spread = float('inf')

    probability_of_default = norm.cdf(-d2)
    
    return {
        "equity_value": equity,
        "debt_market_value": debt_value,
        "distance_to_default": d2,
        "probability_of_default": probability_of_default,
        "credit_spread_bps": credit_spread * 10000 if credit_spread != float('inf') else float('inf')
    }
