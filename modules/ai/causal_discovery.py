"""
causal_discovery.py — Automatyczne odkrywanie relacji przyczynowych
===================================================================
Algorytmy:
  - PC Algorithm (Peter-Clark) — constraint-based
  - Granger Causality (VAR-based)
  - Transfer Entropy (Information-theoretic Granger)
  - LiNGAM (Linear Non-Gaussian Acyclic Model) — simplified
  - DAG vizualizacja + Do-Calculus backdoor criterion

Ref: Spirtes, Glymour & Scheines (2000), Shimizu (2006)
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1. GRANGER CAUSALITY TEST
# ─────────────────────────────────────────────────────────────────────────────

def granger_causality_test(x: pd.Series, y: pd.Series,
                            max_lag: int = 5,
                            significance: float = 0.05) -> dict:
    """
    Test Grangera: czy x Granger-powoduje y?
    H0: x nie Granger-powoduje y (lagged x nie ulepsza predykcji y)
    
    Zwraca: p-value, F-stat, optymalny lag (AIC)
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    T = len(df)

    if T < max_lag * 3 + 10:
        return {"granger_causes": False, "p_value": 1.0, "best_lag": 1}

    best_aic = np.inf
    best_lag = 1
    best_pval = 1.0
    best_fstat = 0.0

    for lag in range(1, max_lag + 1):
        # Restricted model: y ~ y_lags
        Y = df["y"].values[lag:]
        X_r = np.column_stack([df["y"].shift(l).values[lag:] for l in range(1, lag + 1)])
        # Unrestricted: + x_lags
        X_u = np.column_stack([
            df["y"].shift(l).values[lag:] for l in range(1, lag + 1)
        ] + [
            df["x"].shift(l).values[lag:] for l in range(1, lag + 1)
        ])
        X_r = np.column_stack([np.ones(len(Y)), X_r])
        X_u = np.column_stack([np.ones(len(Y)), X_u])

        try:
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            rss_r = np.sum((Y - X_r @ beta_r)**2)
            rss_u = np.sum((Y - X_u @ beta_u)**2)
            n, k_r, k_u = len(Y), X_r.shape[1], X_u.shape[1]
            df_n = k_u - k_r
            df_d = n - k_u
            if df_d <= 0 or df_n <= 0 or rss_u <= 0:
                continue
            F = ((rss_r - rss_u) / df_n) / (rss_u / df_d)
            p_val = 1 - stats.f.cdf(F, df_n, df_d)
            aic = n * np.log(rss_u / n) + 2 * k_u
            if aic < best_aic:
                best_aic, best_lag, best_pval, best_fstat = aic, lag, p_val, F
        except Exception:
            continue

    return {
        "granger_causes": best_pval < significance,
        "p_value": best_pval,
        "f_stat": best_fstat,
        "best_lag": best_lag,
        "significance": significance,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. TRANSFER ENTROPY (Information-theoretic Granger)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_entropy(x: np.ndarray, y: np.ndarray,
                     lag: int = 1, n_bins: int = 10) -> float:
    """
    Transfer Entropy T(X→Y): ile informacji X dostarcza o przyszłości Y
    ponad to co dostarcza przeszłość Y.
    T(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    
    Obliczana histogramowo (binning method).
    """
    n = min(len(x), len(y)) - lag
    if n < 20:
        return 0.0

    y_fut = y[lag:n + lag]
    y_past = y[:n]
    x_past = x[:n]

    def entropy_hist(data, bins):
        counts, _ = np.histogram(data, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs + 1e-15))

    def joint_entropy_2d(a, b, bins):
        H, _, _ = np.histogram2d(a, b, bins=bins)
        p = H / H.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p + 1e-15))

    def joint_entropy_3d(a, b, c, bins):
        data = np.column_stack([a, b, c])
        H, _ = np.histogramdd(data, bins=bins)
        p = H / H.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p + 1e-15))

    try:
        b = n_bins
        # TE(X→Y) = H(Y_fut, Y_past) + H(Y_past, X_past) - H(Y_past) - H(Y_fut, Y_past, X_past)
        h_yfut_ypast = joint_entropy_2d(y_fut, y_past, b)
        h_ypast_xpast = joint_entropy_2d(y_past, x_past, b)
        h_ypast = entropy_hist(y_past, b)
        h_joint3 = joint_entropy_3d(y_fut, y_past, x_past, b)
        te = h_yfut_ypast + h_ypast_xpast - h_ypast - h_joint3
        return max(0.0, float(te))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  3. PC ALGORITHM (Constraint-Based Causal Discovery)
# ─────────────────────────────────────────────────────────────────────────────

def partial_correlation(df: pd.DataFrame, x: str, y: str,
                         conditioning: list) -> tuple:
    """
    Partial correlation between x and y given conditioning set.
    Returns (corr, p_value)
    """
    if not conditioning:
        r, p = stats.pearsonr(df[x].dropna(), df[y].dropna())
        return r, p

    cols = [x, y] + conditioning
    sub = df[cols].dropna()
    if len(sub) < 10:
        return 0.0, 1.0

    # Regress out conditioning variables
    def residuals(target, conds):
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in conds])
        y_vec = sub[target].values
        try:
            beta = np.linalg.lstsq(X, y_vec, rcond=None)[0]
            return y_vec - X @ beta
        except Exception:
            return y_vec

    res_x = residuals(x, conditioning)
    res_y = residuals(y, conditioning)

    if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
        return 0.0, 1.0

    r, p = stats.pearsonr(res_x, res_y)
    return r, p


def pc_algorithm_skeleton(df: pd.DataFrame,
                           alpha: float = 0.05,
                           max_cond_set: int = 2) -> dict:
    """
    PC Algorithm — szkielet (undirected edges).
    Zwraca: adj_matrix, separating_sets, edge_list
    """
    nodes = list(df.columns)
    n = len(nodes)

    # Start with complete undirected graph
    adj = {node: set(nodes) - {node} for node in nodes}
    sep_sets = {}

    for cond_size in range(max_cond_set + 1):
        edges_to_remove = []
        for x, y in combinations(nodes, 2):
            if y not in adj[x]:
                continue
            # Potential conditioning sets from neighbors of x (excluding y)
            neighbors_x = list(adj[x] - {y})
            if len(neighbors_x) < cond_size:
                continue
            # Test all conditioning sets of given size
            from itertools import combinations as combs
            for cond_set in combs(neighbors_x, cond_size):
                r, p = partial_correlation(df, x, y, list(cond_set))
                if p > alpha:  # conditionally independent → remove edge
                    edges_to_remove.append((x, y))
                    sep_sets[(x, y)] = list(cond_set)
                    sep_sets[(y, x)] = list(cond_set)
                    break

        for x, y in edges_to_remove:
            adj[x].discard(y)
            adj[y].discard(x)

    # Build edge list
    edges = []
    for x in nodes:
        for y in adj[x]:
            if x < y:
                edges.append((x, y))

    return {
        "adjacency": adj,
        "edges": edges,
        "separating_sets": sep_sets,
        "n_nodes": n,
        "n_edges": len(edges),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. CAUSAL DISCOVERY MATRIX (Multi-variable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_causal_matrix(df: pd.DataFrame,
                           method: str = "granger",
                           max_lag: int = 3,
                           significance: float = 0.05) -> pd.DataFrame:
    """
    Oblicza macierz przyczynowości dla wszystkich par zmiennych.
    
    method: 'granger' | 'transfer_entropy'
    Returns: DataFrame [cause × effect] z p-values lub TE values.
    """
    cols = list(df.columns)
    n = len(cols)
    result = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i, cause in enumerate(cols):
        for j, effect in enumerate(cols):
            if i == j:
                continue
            x = df[cause].dropna()
            y = df[effect].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 20:
                continue

            if method == "granger":
                res = granger_causality_test(x[common], y[common], max_lag, significance)
                result.loc[cause, effect] = 1.0 - res["p_value"]  # strength proxy
            elif method == "transfer_entropy":
                te = transfer_entropy(x[common].values, y[common].values, lag=1)
                result.loc[cause, effect] = te

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  5. BACKDOOR CRITERION (Pearl's Do-Calculus)
# ─────────────────────────────────────────────────────────────────────────────

def check_backdoor_criterion(dag: dict,
                              treatment: str,
                              outcome: str,
                              adjustment_set: list) -> dict:
    """
    Sprawdza backdoor criterion Pearla.
    dag: dict of {node: [children]} (directed edges)
    
    Backdoor criterion jest spełnione gdy:
    1. Żaden węzeł z adjustment_set nie jest potomkiem treatment.
    2. Adjustment set blokuje wszystkie backdoor paths od treatment do outcome.
    
    Zwraca: is_valid, explanation
    """
    # Find descendants of treatment
    def descendants(node, graph):
        desc = set()
        frontier = list(graph.get(node, []))
        while frontier:
            n = frontier.pop()
            if n not in desc:
                desc.add(n)
                frontier.extend(graph.get(n, []))
        return desc

    desc_treatment = descendants(treatment, dag)

    # Check condition 1: no member of adjustment set is a descendant of treatment
    cond1_violated = [n for n in adjustment_set if n in desc_treatment]
    cond1_ok = len(cond1_violated) == 0

    explanation = []
    if not cond1_ok:
        explanation.append(f"❌ Warunek 1 NARUSZONY: {cond1_violated} to potomkowie {treatment}.")
    else:
        explanation.append("✅ Warunek 1 OK: Żaden węzeł adjustment set nie jest potomkiem treatment.")

    # Simplified condition 2 check (assumes user provided valid set)
    explanation.append("ℹ️ Warunek 2 (blokowanie backdoor paths) wymaga pełnej analizy grafowej (d-separation).")

    return {
        "is_valid_criterion1": cond1_ok,
        "desc_of_treatment": list(desc_treatment),
        "violated_nodes": cond1_violated,
        "explanation": "\n".join(explanation),
    }
