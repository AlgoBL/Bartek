"""
signature_features.py — Neural CDEs & Log-Signature Features (A.4)
===================================================================
Implementacja ekstrakcji cech sygnaturowych (Path Signatures) wykorzystywanych
jako dane wejściowe dla lekkich modeli ML (zamiast głębokich LSTM/RNN) wg
Morrill et al. (2021) "Neural Controlled Differential Equations".

Redukcja wymiaru z użyciem Log-Signatures dla zmniejszenia klątwy wymiarowości
(O(k·d) a nie O(d^k) jak przy płaskich Signatures).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False
    logger.warning("iisignature library not found. Falling back to simple heuristic for signatures. Run 'pip install iisignature'.")

def compute_log_signatures(paths: np.ndarray, depth: int = 4) -> np.ndarray:
    """
    Oblicza Log-Signature dla zestawu ścieżek wielowymiarowych.
    Log-Signature efektywnie i jednoznacznie koduje ewolucję ścieżki (Morrill 2021).
    
    Parameters
    ----------
    paths : np.ndarray
        (N_samples, Time_steps, Features)
    depth : int
        Głębokość obcięcia (truncation level). Dla rynków wystarcza d=4 lub 5.
        
    Returns
    -------
    np.ndarray
        (N_samples, LogSig_Features) gotowe do wrzucenia do XGBoost/Ridge
    """
    n_samples, time_steps, features = paths.shape
    
    if HAS_IISIGNATURE:
        # Standardowa ewaluacja:
        # iisignature expect shape (time_steps, features)
        # Przygotowujemy log-signature support (basis)
        s = iisignature.prepare(features, depth)
        
        results = []
        for i in range(n_samples):
            # Compute log signature (redukuje redundantne wymiary za bazy słów)
            log_sig = iisignature.logsig(paths[i], s)
            results.append(log_sig)
        return np.array(results)
    
    else:
        # FALLBACK: Ręczne, przybliżone i bezpieczne (niekompletne, ale no-crash fallback)
        # W prawdziwym wdrożeniu należy zainstalować 'iisignature'
        
        # Increments
        diffs = np.diff(paths, axis=1) # (N, T-1, D)
        
        # Zamiast pełnej sygnatury robimy "Poor man's signature":
        # - L1 (Zwykłe zsumowane przyrosty)
        L1 = np.sum(diffs, axis=1) # (N, D)
        
        # - L2 (Uproszczone pole powierzchni Lead-Lag)
        results = []
        for i in range(n_samples):
            d_path = diffs[i] # (T-1, D)
            # Area matrix
            area = np.zeros((features, features))
            for t in range(d_path.shape[0] - 1):
                # Z(t) * dZ(t+1) uproszczony trapezoid area
                area += np.outer(d_path[t], d_path[t+1])
            
            # Wektoryzacja górnego trójkąta by uniknąć symetrycznych duplikatów
            upper_area = area[np.triu_indices(features, k=1)]
            
            row = np.concatenate([L1[i], upper_area])
            results.append(row)
            
        return np.array(results)

def generate_cde_features(df: pd.DataFrame, window_size: int = 21, depth: int = 4) -> pd.DataFrame:
    """
    Sliding window feature extractor na podstawie Log-Signatures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Multivariate time-series data (e.g., Returns, Volatility, Spread)
    window_size : int
        Długość okna (np. 21 dni)
        
    Returns
    -------
    pd.DataFrame
        Nowe features, index zrównany z oryginalnym (shifted).
    """
    n_rows, n_cols = df.shape
    if n_rows <= window_size:
        return pd.DataFrame()
        
    arr = df.values
    data_windows = []
    indices = []
    
    for i in range(window_size, n_rows):
        w = arr[i-window_size : i, :]
        data_windows.append(w)
        indices.append(df.index[i])
        
    data_windows = np.array(data_windows)  # (N, window_size, d)
    
    log_sigs = compute_log_signatures(data_windows, depth)
    
    cols = [f"LogSig_D{depth}_{k}" for k in range(log_sigs.shape[1])]
    
    return pd.DataFrame(log_sigs, index=indices, columns=cols)
