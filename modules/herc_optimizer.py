"""
herc_optimizer.py — Hierarchical Equal Risk Contribution (A.2)
==============================================================
Algorytm Tomasa Raffinota (2018) łączący clustering hierarchiczny z 
Equal Risk Contribution. W przeciwieństwie do standardowego HRP (López de Prado),
który przydziela ryzyko równo na każdym węźle, HERC przydziela ryzyko
równo w obrębie zoptymalizowanych klastrów (Machine Learning).

Złożoność zredukowana w stosunku do Critical Line Algorithm (CLA).
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

def compute_herc_weights(cov_matrix: pd.DataFrame, n_clusters: int = None) -> pd.Series:
    """
    Kalkuluje wagi portfela za pomocą algorytmu HERC.
    
    1. Konwersja macierzy kowariancji na macierz korelacji i dystansu
    2. Hierarchical Clustering (Ward linkage)
    3. Wybór liczby klastrów (Gap Statistic / Silhouette lub podana ręcznie)
    4. Top-down Equal Risk Contribution wzdłuż drzewa klastrów.
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Macierz kowariancji stóp zwrotu aktywów
    n_clusters : int, optional
        Zdefiniowana liczna klastrów, jeśli None - heurystyka (np. k=3).
        
    Returns
    -------
    pd.Series
        Wagi w wektorze o sumie 1.0 (long-only).
    """
    tickers = cov_matrix.columns
    n_assets = len(tickers)
    
    if n_assets <= 1:
        return pd.Series(1.0, index=tickers)
        
    # 1. Korelacja z Kowariancji
    std = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix.divide(std, axis=0).divide(std, axis=1)
    
    # Numeryczne błędy (clip)
    corr_matrix = corr_matrix.clip(-1.0, 1.0)
    
    # 2. Dystans (musi być metryką -> sqrt(0.5*(1-corr)))
    dist_matrix = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
    
    # condensed distance matrix dla linkage
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # 3. Clustering (Ward linkage minimizes within-cluster variance)
    Z = linkage(condensed_dist, method='ward')
    
    # Określanie klastrów
    if n_clusters is None:
        n_clusters = min(max(2, n_assets // 2), 6) # Prosta heurystyka
        
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # 4. Naive Risk Contribution w klastrach (Inverse Volatility wewnątrz klastra)
    cluster_weights = pd.Series(0.0, index=tickers)
    cluster_variances = {}
    
    for c in range(1, n_clusters + 1):
        assets_in_cluster = tickers[cluster_labels == c]
        if len(assets_in_cluster) == 0: continue
            
        cluster_cov = cov_matrix.loc[assets_in_cluster, assets_in_cluster]
        inv_vols = 1.0 / np.sqrt(np.diag(cluster_cov))
        inv_vols_sum = np.sum(inv_vols)
        
        weights_in_cluster = inv_vols / inv_vols_sum
        cluster_weights[assets_in_cluster] = weights_in_cluster
        
        # Obliczenie wariancji klastra
        w = np.array(weights_in_cluster)
        c_var = w.T @ cluster_cov.values @ w
        cluster_variances[c] = c_var

    # 5. Equal Risk Contribution na poziomie klastrów (Equal Risk Allocation)
    # HERC alokuje kapitał tak, by każdy KLASter wnosił tyle samo ryzyka
    cluster_allocation = {}
    total_inv_risk = sum([1.0 / np.sqrt(var) for var in cluster_variances.values()])
    
    for c, var in cluster_variances.items():
        cluster_allocation[c] = (1.0 / np.sqrt(var)) / total_inv_risk
        
    # 6. Finalne wagi = alokacja_klastra * alokacja_wewnatrz_klastra
    final_weights = pd.Series(0.0, index=tickers)
    
    for c in range(1, n_clusters + 1):
        assets_in_cluster = tickers[cluster_labels == c]
        final_weights[assets_in_cluster] = cluster_weights[assets_in_cluster] * cluster_allocation[c]
        
    return final_weights / final_weights.sum()  # Normalizacja
