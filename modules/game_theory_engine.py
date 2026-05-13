"""
game_theory_engine.py
=====================
Silnik teorii gier: rozszerzona implementacja Równowag Nasha.
Fazy:
  1. Solver NxM (support enumeration, bez nashpy)
  2. Algorytm Gale-Shapley (Stable Matching)
  3. Correlated Equilibrium (Aumann 1987) — LP solver
  4. Bayesian Nash Equilibrium (typy graczy)
  5. Stackelberg Equilibrium (ciągła przestrzeń)
  6. Uproszczony Mean-Field Game (LQ)
"""

import numpy as np
from scipy.optimize import linprog, minimize
from itertools import combinations
from typing import Optional


# =============================================================================
# 1. NxM NASH EQUILIBRIUM SOLVER (support enumeration method)
# =============================================================================

def find_pure_nash(payoff_r: np.ndarray, payoff_c: np.ndarray) -> list[tuple]:
    """
    Czyste Równowagi Nasha dla gry NxM.
    Zwraca listę (i, j) będących NE w strategiach czystych.
    """
    n, m = payoff_r.shape
    nash_eqs = []
    for i in range(n):
        for j in range(m):
            # Gracz wierszowy nie chce dewiować
            if payoff_r[i, j] >= np.max(payoff_r[:, j]):
                # Gracz kolumnowy nie chce dewiować
                if payoff_c[i, j] >= np.max(payoff_c[i, :]):
                    nash_eqs.append((i, j))
    return nash_eqs


def _solve_mixed_for_support(payoff_r: np.ndarray, payoff_c: np.ndarray,
                              support_r: tuple, support_c: tuple) -> Optional[tuple]:
    """
    Dla zadanego nośnika strategii próbuje znaleźć mieszaną NE.
    Zwraca (sigma_r, sigma_c) lub None jeśli nie istnieje.
    """
    k = len(support_r)
    l_ = len(support_c)

    # Gracz R jest obojętny między strategiami w support_c
    # Dla każdej pary (j1, j2) w support_c: sum_i sigma_r[i] * A[i,j1] = sum_i sigma_r[i] * A[i,j2]
    # Kolumna kolumnowego gracza: payoff_c[support_r, :][:, support_c]
    A_sub = payoff_r[np.ix_(list(support_r), list(support_c))]
    B_sub = payoff_c[np.ix_(list(support_r), list(support_c))]

    # Znajdź sigma_c (mieszanie kolumnowego gracza) — musi wyrównać zyski wierszowego
    if l_ > 1:
        # Układ: A_sub @ sigma_c = const * ones → (A_sub[:-1] - A_sub[1:]) @ sigma_c = 0
        M_c = (A_sub[:-1, :] - A_sub[1:, :]).T  # (l_, k-1)
        b_c = np.zeros(k - 1)
        # plus suma = 1
        M_full = np.vstack([M_c.T, np.ones((1, l_))])
        b_full = np.append(b_c, 1.0)
        try:
            sigma_c_full = np.linalg.lstsq(M_full, b_full, rcond=None)[0]
        except Exception:
            return None
        if np.any(sigma_c_full < -1e-8) or abs(sigma_c_full.sum() - 1) > 1e-6:
            return None
        sigma_c_full = np.clip(sigma_c_full, 0, None)
        sigma_c_full /= sigma_c_full.sum()
    else:
        sigma_c_full = np.array([1.0])

    # Znajdź sigma_r
    if k > 1:
        M_r = (B_sub[:, :-1] - B_sub[:, 1:])
        b_r = np.zeros(l_ - 1)
        M_full_r = np.vstack([M_r.T, np.ones((1, k))])
        b_full_r = np.append(b_r, 1.0)
        try:
            sigma_r_full = np.linalg.lstsq(M_full_r, b_full_r, rcond=None)[0]
        except Exception:
            return None
        if np.any(sigma_r_full < -1e-8) or abs(sigma_r_full.sum() - 1) > 1e-6:
            return None
        sigma_r_full = np.clip(sigma_r_full, 0, None)
        sigma_r_full /= sigma_r_full.sum()
    else:
        sigma_r_full = np.array([1.0])

    # Weryfikacja: żadna strategia poza nośnikiem nie powinna być lepsza
    n, m = payoff_r.shape
    sigma_r_full_expanded = np.zeros(n)
    sigma_c_full_expanded = np.zeros(m)
    for idx, i in enumerate(support_r):
        sigma_r_full_expanded[i] = sigma_r_full[idx]
    for idx, j in enumerate(support_c):
        sigma_c_full_expanded[j] = sigma_c_full[idx]

    u_r_eq = payoff_r[list(support_r)[0], :] @ sigma_c_full_expanded
    u_c_eq = sigma_r_full_expanded @ payoff_c[:, list(support_c)[0]]

    for i in range(n):
        if payoff_r[i, :] @ sigma_c_full_expanded > u_r_eq + 1e-8:
            return None
    for j in range(m):
        if sigma_r_full_expanded @ payoff_c[:, j] > u_c_eq + 1e-8:
            return None

    return sigma_r_full_expanded, sigma_c_full_expanded


def find_all_nash(payoff_r: np.ndarray, payoff_c: np.ndarray) -> list[dict]:
    """
    Wszystkie Równowagi Nasha (czyste + mieszane) dla gry NxM.
    Metoda: support enumeration.
    Zwraca listę słowników z polami:
        type       — 'pure' lub 'mixed'
        sigma_r    — strategia wierszowego (wektor prawdopodobieństw)
        sigma_c    — strategia kolumnowego
        payoff_r   — oczekiwana wypłata wierszowego
        payoff_c   — oczekiwana wypłata kolumnowego
    """
    n, m = payoff_r.shape
    results = []
    seen = []

    for size_r in range(1, n + 1):
        for size_c in range(1, m + 1):
            for sup_r in combinations(range(n), size_r):
                for sup_c in combinations(range(m), size_c):
                    result = _solve_mixed_for_support(payoff_r, payoff_c, sup_r, sup_c)
                    if result is None:
                        continue
                    sr, sc = result
                    # Deduplikacja
                    duplicate = False
                    for prev_sr, prev_sc in seen:
                        if np.allclose(sr, prev_sr, atol=1e-6) and np.allclose(sc, prev_sc, atol=1e-6):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    seen.append((sr, sc))
                    eq_type = "pure" if size_r == 1 and size_c == 1 else "mixed"
                    results.append({
                        "type": eq_type,
                        "sigma_r": sr,
                        "sigma_c": sc,
                        "payoff_r": float(sr @ payoff_r @ sc),
                        "payoff_c": float(sr @ payoff_c @ sc),
                    })
    return results


# =============================================================================
# 2. CORRELATED EQUILIBRIUM (Aumann 1987) — LP
# =============================================================================

def find_correlated_equilibrium(payoff_r: np.ndarray, payoff_c: np.ndarray,
                                 objective: str = "welfare") -> dict:
    """
    Oblicza Correlated Equilibrium przez linear programming (scipy HiGHS).

    Zmienne: p(i,j) = prawdopodobieństwo rekomendacji sygnału (i,j).
    Ograniczenia IC:
      Dla gracza R: dla każdego i, i': sum_j p(i,j)*(u_R(i,j) - u_R(i',j)) >= 0
      Dla gracza C: dla każdego j, j': sum_i p(i,j)*(u_C(i,j) - u_C(i,j')) >= 0
    Cel:
      'welfare'  — maks. dobrobyt społeczny (suma wypłat)
      'row'      — maks. wypłata wierszowego
      'col'      — maks. wypłata kolumnowego

    Zwraca dict z kluczami: success, distribution, social_welfare, payoff_r, payoff_c
    """
    n, m = payoff_r.shape
    n_vars = n * m

    idx = lambda i, j: i * m + j  # indeks zmiennej p(i,j)

    # Cel
    if objective == "welfare":
        c_obj = -(payoff_r + payoff_c).flatten()
    elif objective == "row":
        c_obj = -payoff_r.flatten()
    else:
        c_obj = -payoff_c.flatten()

    # Ograniczenia IC
    ic_rows = []
    # Gracz R (wierszowy)
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            row = np.zeros(n_vars)
            for j in range(m):
                row[idx(i, j)] += payoff_r[i, j] - payoff_r[i2, j]
            ic_rows.append(-row)  # linprog: A_ub @ x <= b_ub, więc negujemy >= 0

    # Gracz C (kolumnowy)
    for j in range(m):
        for j2 in range(m):
            if j == j2:
                continue
            row = np.zeros(n_vars)
            for i in range(n):
                row[idx(i, j)] += payoff_c[i, j] - payoff_c[i, j2]
            ic_rows.append(-row)

    A_ub = np.array(ic_rows)
    b_ub = np.zeros(len(ic_rows))

    # Suma = 1
    A_eq = np.ones((1, n_vars))
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_vars

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return {"success": False, "message": res.message}

    p = res.x.reshape(n, m)
    p = np.clip(p, 0, None)
    p /= p.sum()

    return {
        "success": True,
        "distribution": p,
        "social_welfare": float((p * (payoff_r + payoff_c)).sum()),
        "payoff_r": float((p * payoff_r).sum()),
        "payoff_c": float((p * payoff_c).sum()),
    }


# =============================================================================
# 3. ALGORYTM GALE-SHAPLEY (Stable Matching)
# =============================================================================

def gale_shapley(proposers_prefs: dict, receivers_prefs: dict) -> dict:
    """
    Algorytm Deferred Acceptance (Gale-Shapley 1962).
    Zwraca stabilne dopasowanie optymalne dla strony proponującej.

    Args:
        proposers_prefs: {P1: [R2, R1, R3], P2: [...], ...}
        receivers_prefs: {R1: [P2, P1, P3], R2: [...], ...}

    Returns:
        {P1: R2, P2: R1, ...}   (dopasowanie proposer -> receiver)
    """
    free_proposers = list(proposers_prefs.keys())
    next_proposal = {p: 0 for p in proposers_prefs}   # indeks następnej propozycji
    current_match = {}    # receiver -> proposer (tymczasowe)

    # Ranking receiverów (szybkie porównania)
    receiver_rank = {
        r: {p: idx for idx, p in enumerate(prefs)}
        for r, prefs in receivers_prefs.items()
    }

    while free_proposers:
        p = free_proposers[0]
        prefs = proposers_prefs[p]

        if next_proposal[p] >= len(prefs):
            free_proposers.pop(0)
            continue

        r = prefs[next_proposal[p]]
        next_proposal[p] += 1

        if r not in current_match:
            current_match[r] = p
            free_proposers.pop(0)
        else:
            current_holder = current_match[r]
            # Receiver woli p zamiast current_holder?
            r_prefs = receiver_rank.get(r, {})
            rank_p = r_prefs.get(p, 999)
            rank_holder = r_prefs.get(current_holder, 999)
            if rank_p < rank_holder:
                current_match[r] = p
                free_proposers.pop(0)
                free_proposers.append(current_holder)

    return {p: r for r, p in current_match.items()}


def check_stability(matching: dict, proposers_prefs: dict, receivers_prefs: dict) -> list[tuple]:
    """
    Sprawdza czy dopasowanie jest stabilne. Zwraca listę blokujących par (p, r).
    Stabilne dopasowanie <=> pusta lista.
    """
    receiver_rank = {
        r: {p: idx for idx, p in enumerate(prefs)}
        for r, prefs in receivers_prefs.items()
    }
    proposer_rank = {
        p: {r: idx for idx, r in enumerate(prefs)}
        for p, prefs in proposers_prefs.items()
    }
    reverse_match = {r: p for p, r in matching.items()}
    blocking = []

    for p, r_matched in matching.items():
        p_prefs = proposers_prefs[p]
        rank_matched = proposer_rank[p].get(r_matched, 999)
        for r in p_prefs:
            rank_r = proposer_rank[p].get(r, 999)
            if rank_r >= rank_matched:
                break  # p woli r_matched lub jest równie dobry
            # p woli r. Czy r woli p od swojego obecnego partnera?
            current_p = reverse_match.get(r)
            rank_current = receiver_rank[r].get(current_p, 999)
            rank_p_for_r = receiver_rank[r].get(p, 999)
            if rank_p_for_r < rank_current:
                blocking.append((p, r))
    return blocking


# =============================================================================
# 4. BAYESIAN NASH EQUILIBRIUM (2 typy graczy, 2 strategie)
# =============================================================================

def solve_bne_two_types(
    payoff_HH: np.ndarray, payoff_HL: np.ndarray,
    payoff_LH: np.ndarray, payoff_LL: np.ndarray,
    prob_H: float
) -> dict:
    """
    Bayesian Nash Equilibrium dla gry z 2 typami (High/Low) każdego gracza.
    Zakłada niezależne typy. prob_H = P(Gracz jest typem H).

    payoff_XY — macierz 2x2 wypłat gdy Row=typ X, Col=typ Y.
    Zwraca słownik z równowagami Bayesowskimi (strategie warunkowe).

    Uproszczona wersja: szuka równowagi w strategiach symetrycznych.
    Każdy typ gra strategię czystą lub mieszaną (p_H, p_L) = prob grania strategii 1.
    """
    p_L = 1 - prob_H

    results = []

    # Przeszukaj przestrzeń (p_H, p_L) w siatce
    grid = np.linspace(0, 1, 21)

    for qH in grid:
        for qL in grid:
            # Oczekiwana wypłata gracza H grającego strategię 0 vs 1
            # Przeciwnik gra H z prob prob_H (qH) i L z prob p_L (qL)
            # E[u_H | play 0] = prob_H * u_HH(0, qH) + p_L * u_HL(0, qL)
            eu_H0 = (prob_H * (payoff_HH[0, :] @ np.array([1 - qH, qH]))
                     + p_L  * (payoff_HL[0, :] @ np.array([1 - qL, qL])))
            eu_H1 = (prob_H * (payoff_HH[1, :] @ np.array([1 - qH, qH]))
                     + p_L  * (payoff_HL[1, :] @ np.array([1 - qL, qL])))

            eu_L0 = (prob_H * (payoff_LH[0, :] @ np.array([1 - qH, qH]))
                     + p_L  * (payoff_LL[0, :] @ np.array([1 - qL, qL])))
            eu_L1 = (prob_H * (payoff_LH[1, :] @ np.array([1 - qH, qH]))
                     + p_L  * (payoff_LL[1, :] @ np.array([1 - qL, qL])))

            # Sprawdź best response
            br_H = 1.0 if eu_H1 > eu_H0 + 1e-6 else (0.0 if eu_H0 > eu_H1 + 1e-6 else qH)
            br_L = 1.0 if eu_L1 > eu_L0 + 1e-6 else (0.0 if eu_L0 > eu_L1 + 1e-6 else qL)

            if abs(br_H - qH) < 1e-6 and abs(br_L - qL) < 1e-6:
                results.append({
                    "p_H": round(qH, 2),
                    "p_L": round(qL, 2),
                    "eu_H": round(max(eu_H0, eu_H1), 3),
                    "eu_L": round(max(eu_L0, eu_L1), 3),
                    "type": "pure" if qH in (0, 1) and qL in (0, 1) else "mixed"
                })

    # Deduplikacja
    seen = set()
    unique = []
    for r in results:
        key = (r["p_H"], r["p_L"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return {"equilibria": unique, "prob_H": prob_H}


# =============================================================================
# 5. STACKELBERG EQUILIBRIUM (ciągła przestrzeń, duopol Cournota)
# =============================================================================

def stackelberg_cournot(a: float, b: float, c_leader: float, c_follower: float) -> dict:
    """
    Równowaga Stackelberga w modelu duopolu Cournota (liniowy popyt).
    Cena: P = a - b*(q1 + q2)
    Koszty marginalne: c_leader (lider), c_follower (naśladowca)

    Lider wybiera q1 wiedząc, że Follower zoptymalizuje q2 = (a - c_f - b*q1) / (2b).
    """
    # Funkcja reakcji followera: q2*(q1) = max(0, (a - c_f - b*q1) / (2b))
    def follower_br(q1):
        return max(0.0, (a - c_follower - b * q1) / (2 * b))

    # Zysk lidera: pi1 = (P - c_l) * q1
    def leader_profit(q1):
        q2 = follower_br(q1[0])
        price = a - b * (q1[0] + q2)
        return -(price - c_leader) * q1[0]  # negujemy bo minimize

    res = minimize(leader_profit, x0=[1.0], bounds=[(0, None)], method='L-BFGS-B')
    q1_star = max(0.0, res.x[0])
    q2_star = follower_br(q1_star)
    price_star = a - b * (q1_star + q2_star)

    # Równowaga Nasha (Cournot symultaniczny) dla porównania
    q_nash = (a - 2 * c_leader + c_follower) / (3 * b) if b > 0 else 0
    q_nash = max(0.0, q_nash)

    return {
        "q_leader": round(q1_star, 4),
        "q_follower": round(q2_star, 4),
        "price": round(price_star, 4),
        "profit_leader": round((price_star - c_leader) * q1_star, 4),
        "profit_follower": round((price_star - c_follower) * q2_star, 4),
        "total_output": round(q1_star + q2_star, 4),
        "nash_q_symmetric": round(q_nash, 4),
        "first_mover_advantage": round(
            (price_star - c_leader) * q1_star
            - (a - b*(q_nash+q_nash) - c_leader) * q_nash, 4
        ),
    }


# =============================================================================
# 6. MEAN-FIELD GAME (LQ — przepływ kapitału)
# =============================================================================

def lq_mean_field_game(T: float = 10.0, n_steps: int = 200,
                        alpha: float = 0.5, beta: float = 1.0,
                        sigma: float = 0.3, x0: float = 1.0) -> dict:
    """
    Uproszczony Linear-Quadratic Mean-Field Game.
    Modeluje optymalny przepływ kapitału w populacji inwestorów.

    Dynamika: dx = u*dt + sigma*dW
    Koszt:    integral(beta*u^2 + alpha*(x - m)^2) dt + (x_T)^2
    gdzie m = mean field (średni stan populacji).

    W LQ-MFG mean field spełnia: m(t) = x0 * exp(-alpha*t / (2*beta))
    Optymalne sterowanie: u*(t) = -(1/beta) * P(t) * (x - m(t))

    Zwraca trajektorię średniego stanu i optymalną politykę.
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Rozwiązanie równania Riccatiego (backward)
    P = np.zeros(n_steps + 1)
    P[-1] = 1.0  # koszt końcowy
    for k in range(n_steps - 1, -1, -1):
        dP = -2 * alpha + (P[k+1]**2) / beta
        P[k] = P[k+1] - dP * dt

    # Mean-field trajektoria (deterministyczna)
    m = np.zeros(n_steps + 1)
    m[0] = x0
    for k in range(n_steps):
        u_m = -(P[k] / beta) * (m[k] - m[k])  # = 0 w równowadze (m = m*)
        m[k+1] = m[k] + u_m * dt

    # Przykładowa trajektoria pojedynczego agenta (z szumem)
    np.random.seed(42)
    x = np.zeros(n_steps + 1)
    x[0] = x0 * 1.3  # agent startuje z odchyleniem +30%
    u_path = np.zeros(n_steps)
    for k in range(n_steps):
        u_path[k] = -(P[k] / beta) * (x[k] - m[k])
        x[k+1] = x[k] + u_path[k] * dt + sigma * np.sqrt(dt) * np.random.randn()

    return {
        "t": t,
        "mean_field": m,
        "agent_path": x,
        "optimal_control": u_path,
        "riccati_P": P,
    }
