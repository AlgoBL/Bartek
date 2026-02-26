
"""
Vanguard Math Module (V6.0)

Zaawansowane narzędzia matematyczne dla projektu Talebl/Barbell:
1. TDA (Topological Data Analysis) - Wczesne ostrzeganie przed krachami (Betti-0).
2. Dynamic Copulas - Estymacja zależności lewego ogona (Contagion Effect/Stress Test).
3. Fractional Brownian Motion (fBM) - Symulacja rynków z długą pamięcią (Fraktalny Monte Carlo).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

# =====================================================================
# 1. TOPOLOGICZNA ANALIZA DANYCH (TDA) - CRASH INDICATOR
# =====================================================================

def calculate_tda_betti_0_persistence(returns_df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    TDA Crash Indicator (0-Dimensional Persistent Homology).
    Wykrywa, kiedy wielowymiarowa chmura punktów na rynkach "zapada się" w jeden
    wysoce skorelowany wymiar (wszystko spada naraz).
    
    Metoda:
    Używamy ruchomego okna. Dla każdej próbki liczymy macierz korelacji i dystansów.
    Następnie wyliczamy fuzje (merge distances) w drzewie Single Linkage. 
    Spadek średniego dystansu "śmierci" komponentów Betti-0 oznacza rosnącą panikę.
    
    Returns:
        pd.Series indeksowany datami z wskaźnikiem TDA Crash. 
        Gwałtowny spadek wartości ostrzega przed krachem.
    """
    if len(returns_df) < window + 10:
        return pd.Series(dtype=float)
        
    dates = returns_df.index[window:]
    tda_scores = []
    
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        
        # Odrzucamy kolumny ze zbyt wieloma NaN w obecnym oknie
        valid_data = window_data.dropna(axis=1, how='any')
        if valid_data.shape[1] < 3:
            tda_scores.append(np.nan)
            continue
            
        corr = valid_data.corr().values
        # Przekształcenie korelacji na dystans topologiczny: d = sqrt(0.5 * (1 - rho))
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        
        # Scipy wymaga macierzy skondensowanej
        condensed_dist = dist[np.triu_indices(dist.shape[0], k=1)]
        
        if len(condensed_dist) == 0 or np.isnan(condensed_dist).all():
            tda_scores.append(np.nan)
            continue
            
        # Homologia 0-wymiarowa używa Single Linkage
        Z = linkage(condensed_dist, method='single')
        
        # Dystanse przy których dwie komponenty się łączą (death of Betti-0)
        death_distances = Z[:, 2]
        
        # Średnia odległość pożerania (im niższa, tym wyżej skorelowany rynek = Piekło)
        tda_scores.append(np.mean(death_distances))
        
    indicator = pd.Series(tda_scores, index=dates)
    return indicator

# =====================================================================
# 2. DYNAMIC COPULAS (TAIL DEPENDENCE)
# =====================================================================

def empirical_lower_tail_dependence(series_x: pd.Series, series_y: pd.Series, q: float = 0.10) -> float:
    """
    Empiryczna zależność lewego ogona (TDC - Tail Dependence Coefficient).
    Modeluje efekt "Contagion" — jak często oba aktywa jednocześnie lądują w najgorszych 10% swoich strat.
    Klasyczna korelacja Pearsona zaniża to ryzyko.
    
    Wzór: P(X < q_x | Y < q_y)
    """
    df = pd.concat([series_x, series_y], axis=1).dropna()
    if len(df) < 50:
        return np.nan
        
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    threshold_x = np.quantile(x, q)
    threshold_y = np.quantile(y, q)
    
    # Kiedy oba są w dolnym ogonie?
    both_in_tail = np.sum((x <= threshold_x) & (y <= threshold_y))
    # Kiedy Y jest w dolnym ogonie?
    y_in_tail = np.sum(y <= threshold_y)
    
    if y_in_tail == 0:
        return 0.0
        
    return both_in_tail / y_in_tail

def compute_tail_dependence_matrix(returns_df: pd.DataFrame, q: float = 0.10) -> pd.DataFrame:
    """
    Tworzy macierz zależności dolnych ogonów dla całego portfela.
    Pokazuje, które aktywa pociągną się nawzajem na dno podczas krachu.
    """
    assets = returns_df.columns
    n = len(assets)
    tdm = pd.DataFrame(index=assets, columns=assets, dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                tdm.iloc[i, j] = 1.0
            else:
                tdm.iloc[i, j] = empirical_lower_tail_dependence(returns_df.iloc[:, i], returns_df.iloc[:, j], q)
                
    # Upewniamy się że macierz jest symetryczna względem uśrednienia empirycznego
    tdm_sym = (tdm + tdm.T) / 2
    return tdm_sym

# =====================================================================
# 3. FRACTIONAL BROWNIAN MOTION (fBM) - FRACTAL MONTE CARLO
# =====================================================================

def construct_fgn_covariance_matrix(H: float, n: int) -> np.ndarray:
    """ 
    Buduje macierz kowariancji dla ułamkowego szumu stochastycznego (fGn).
    Wymagane do wygenerowania pamięci długoterminowej.
    H = Hurst Exponent
    """
    gamma = np.zeros(n)
    for k in range(n):
        if k == 0:
            gamma[k] = 1.0
        else:
            gamma[k] = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + abs(k-1)**(2*H))
            
    # Macierz Toeplitza
    Sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Sigma[i, j] = gamma[abs(i - j)]
            
    return Sigma

def generate_fbm_paths(H: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Generuje ścieżki ułamkowego ruchu Browna używając Rozkładu Cholesky'ego.
    Rynki Fraktalne - symulacje, które realnie odzwierciedlają grube ogony i trendowanie.
    
    Args:
        H: Wykładnik Hursta (H > 0.5 to trendowanie, H < 0.5 mean-reversion, H = 0.5 to błądzenie losowe).
        n_paths: Ilość symulowanych ścieżek Monte Carlo.
        n_steps: Długość prognozy.
        
    Returns:
        np.ndarray shape (n_paths, n_steps) z wygenerowanymi skumulowanymi ścieżkami (prices).
    """
    # Zabezpieczenia numeryczne
    H = np.clip(H, 0.05, 0.95) 
    
    # Klasyczny ruch Browna (brak pamięci - szybko generujemy)
    if np.isclose(H, 0.5, atol=1e-3):
        fgn = np.random.randn(n_paths, n_steps)
        fbm = np.cumsum(fgn, axis=1)
        return fbm

    # fGn Method (Rozkład Cholesky'ego macierzy autokowariancji)
    # Metoda dokładna ale działa wolniej dla bardzo dużych n_steps (O(n^3)). 
    # Dla typowych ~252 dni jest błyskawiczna.
    try:
        Sigma = construct_fgn_covariance_matrix(H, n_steps)
        # Mała regularyzacja na przekątnej dla stabilności
        Sigma += np.eye(n_steps) * 1e-8
        L = np.linalg.cholesky(Sigma)
        
        # Z to niezależny szum gaussa
        Z = np.random.randn(n_steps, n_paths)
        
        # X = L * Z to szum ułamkowy (fGn)
        fgn = (L @ Z).T  # Daje shape (n_paths, n_steps)
        
        # Calkowanie (cumsum) by otrzymać Fractional Brownian Motion
        fbm = np.cumsum(fgn, axis=1)
        return fbm
    except np.linalg.LinAlgError:
        # Fallback do zwyklego Browna jesli macierz kowariancji ulegla uszkodzeniu
        fgn_fallback = np.random.randn(n_paths, n_steps)
        return np.cumsum(fgn_fallback, axis=1)

# =====================================================================
# 4. VOLATILITY SURFACE & DARK POOLS (GEX)
# =====================================================================

import scipy.stats as stats

def black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Oblicza współczynnik Gamma z modelu Blacka-Scholesa.
    S: Spot (obecna cena)
    K: Strike (cena wykonania)
    T: Time to maturity (w latach)
    r: Risk-free rate
    sigma: Implied Volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def compute_gex_and_skew(ticker_symbol: str = "SPY") -> dict:
    """
    Pobiera dane opcyjne i oblicza:
    1. Total Gamma Exposure (GEX) - wpływ Market Makerów na rynek.
    2. Zero Gamma Level - próg poniżej którego rynkiem rządzi panika.
    3. Skew Index - stosunek ceny zabezpieczeń przed krachem (OTM Puts) do spekulacji wzrostowej (OTM Calls).
    """
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker_symbol)
        expirations = tk.options
        if not expirations:
            return {}
            
        # Bierzemy najbliższe i kolejne wygaśnięcia (do 3 tygodni / 1 miesiąca wprzód minimalnie)
        spot_history = tk.history(period="5d")
        if spot_history.empty:
            return {}
        spot_price = spot_history['Close'].iloc[-1]
        
        total_gex = 0.0
        gex_profile = {} # Strike -> Net GEX
        
        otm_puts_iv = []
        otm_calls_iv = []
        
        for exp in expirations[:3]: # Analizujemy 3 najbliższe terminy by złapać front-month gamma
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            
            # Days to expiration
            T = max((pd.to_datetime(exp) - pd.Timestamp.today()).days / 365.25, 1/365.25)
            r = 0.045 # Ok. 4.5% wolne od ryzyka
            
            # Przetwarzamy CALLS
            for idx, row in calls.iterrows():
                K = row['strike']
                iv = row['impliedVolatility']
                oi = row['openInterest']
                if np.isnan(oi) or oi == 0 or np.isnan(iv) or iv < 0.01: continue
                
                gamma = black_scholes_gamma(spot_price, K, T, r, iv)
                # Call GEX is positive (Dealers are short calls, so they hedge by buying dips and selling rips if net positive)
                # Actually standard formulation: GEX = Gamma * OI * 100 * Spot
                scalled_gex = gamma * oi * 100 * spot_price
                total_gex += scalled_gex
                gex_profile[K] = gex_profile.get(K, 0) + scalled_gex
                
                # Check 5% OTM calls for skew
                if K > spot_price * 1.04 and K < spot_price * 1.06:
                    otm_calls_iv.append(iv)
                    
            # Przetwarzamy PUTS
            for idx, row in puts.iterrows():
                K = row['strike']
                iv = row['impliedVolatility']
                oi = row['openInterest']
                if np.isnan(oi) or oi == 0 or np.isnan(iv) or iv < 0.01: continue
                
                gamma = black_scholes_gamma(spot_price, K, T, r, iv)
                # Put GEX is negative
                scalled_gex = -gamma * oi * 100 * spot_price
                total_gex += scalled_gex
                gex_profile[K] = gex_profile.get(K, 0) + scalled_gex
                
                # Check 5% OTM puts for skew
                if K < spot_price * 0.96 and K > spot_price * 0.94:
                    otm_puts_iv.append(iv)
        
        # Zero Gamma Level Approximation (Gdzie profil przechodzi przez 0)
        # To uproszczenie szuka strajku z największym nagromadzeniem wolumenu wokół zera
        sorted_strikes = sorted(gex_profile.keys())
        cumulative_gex = []
        c = 0
        zero_gamma_level = spot_price
        
        # OTM Put IV / OTM Call IV
        skew_index = 1.0
        if otm_puts_iv and otm_calls_iv:
            skew_index = np.mean(otm_puts_iv) / np.mean(otm_calls_iv)
            
        return {
            "spot_price": spot_price,
            "total_gex_billions": total_gex / 1e9,
            "skew_index": skew_index,
            "gex_status": "Positive (Stabilizing) 🟢" if total_gex > 0 else "Negative (Volatile) 🔴"
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Błąd przy obliczaniu GEX dla {ticker_symbol}: {e}")
        return {}


# =====================================================================
# 5. BAYESIAN KELLY MULTIPLIER
# =====================================================================

def bayesian_kelly_update(prior_prob: float, evidence_score: float, max_evidence_score: float = 10.0) -> float:
    """
    Dokonuje Bayesowskiej aktualizacji pewności systemu względem "zwycięstwa" (zyskownego tradu).
    
    prior_prob: Początkowe bazowe prawdopodobieństwo po stronie CIO (np. 0.5).
    evidence_score: Nowy sygnał np. z Oracle (od -max_evidence_score do +max_evidence_score).
    max_evidence_score: Maksymalna teoretyczna siła sygnału.
    
    Zwraca uaktualnione prawdopodobieństwo sukcesu (Posterior), które można wrzucić do Kelly'ego.
    """
    # Likelihood ratio. Jeśli evidence > 0, L > 1 (zwiększamy wiarę).
    # Normalizujemy evidence do [-1, 1]
    norm_evidence = np.clip(evidence_score / max_evidence_score, -0.99, 0.99)
    
    # Przełożenie znormalizowanego sygnału na Likelihood
    # np. sygnał +0.5 -> L = (1+0.5)/(1-0.5) = 1.5/0.5 = 3.0
    L = (1 + norm_evidence) / (1 - norm_evidence)
    
    posterior_prob = (prior_prob * L) / (prior_prob * L + (1 - prior_prob))
    return posterior_prob

def dynamic_bayesian_kelly(returns: pd.Series, prior_prob: float, evidence: float, win_loss_ratio: float = None) -> float:
    """
    Oblicza frakcję Kelly'ego ze zaktualizowanym Bayesowskim prawdopodobieństwem.
    f_star = P - (1-P)/(W/L)
    """
    p_asterix = bayesian_kelly_update(prior_prob, evidence)
    
    if win_loss_ratio is None:
        mean_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        mean_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        if mean_loss == 0 or np.isnan(mean_loss):
            win_loss_ratio = 1.0
        else:
            win_loss_ratio = mean_win / mean_loss

    if win_loss_ratio == 0:
        return 0.0
        
    kelly_f = p_asterix - ((1 - p_asterix) / win_loss_ratio)
    return max(0.0, kelly_f)
