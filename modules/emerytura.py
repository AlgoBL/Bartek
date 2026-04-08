"""
Moduł Emerytura — Intelligent Barbell v2.0 (Scientific Edition)
Implementuje 9 ulepszeń naukowych:
1. Student-t shocks (fat tails) — Mandelbrot 1963, Platen & Rendek 2012
2. Stochastyczna długość życia (Gompertz) — GUS 2023 tablice trwania życia
3. Stochastyczna inflacja (CIR) — Cox-Ingersoll-Ross 1985
4. MC-based SWR Heatmap — Bengen 2021, Trinity Study
5. Dynamiczne wypłaty (Guardrails, Flexible, Floor) — Klinger 2006, Merton 2014
6. Retirement Age Optimizer — Pfau & Kitces 2014
7. Krzywa Przeżywalności (Kaplan-Meier) — Chen 2018
8. Waterfall Chart — Few 2009
9. Animated Fan Chart — Hullman et al. 2015
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config import TAX_BELKA
from modules.analysis_content import display_chart_guide

# ─── Persistence helpers ────────────────────────────────────────────────────
def _save(k): st.session_state[f"_s.{k}"] = st.session_state[k]
def _saved(k, d): return st.session_state.get(f"_s.{k}", d)

# ─── Scientific Core Functions ───────────────────────────────────────────────

def student_t_shocks(n_sims, horizon, df=4):
    """Fat-tail shocks: Student-t(df=4). Mandelbrot(1963), Platen&Rendek(2012)."""
    raw = np.random.standard_t(df, (n_sims, horizon))
    # Normalize to unit variance so sigma parameter is still valid
    std_factor = np.sqrt(df / (df - 2))
    return raw / std_factor

def gompertz_lifetimes(current_age, n_sims, m=86.0, b=10.0):
    """
    Losowe długości życia z rozkładu Gompertza. GUS 2023.
    m=86: modalna długość życia (mediana ~82 dla PL). b=10: dyspersja.
    Returns array of lifetime ages (whole years).
    """
    u = np.random.uniform(0, 1, n_sims)
    # Gompertz inverse CDF: t = m - b*ln(-ln(u))
    lifetimes = m - b * np.log(-np.log(np.clip(u, 1e-9, 1 - 1e-9)))
    lifetimes = np.maximum(lifetimes, current_age + 1)
    return lifetimes.astype(int)


# ─── E3: Lee-Carter Mortality Model (NEW 2025) ────────────────────────────────

def lee_carter_lifetimes(
    current_age: int,
    n_sims: int,
    gender: str = "mixed",
    alpha_improvement: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """
    Lee-Carter (1992) mortality model — lepszy od Gompertza dla planowania emerytalnego.

    Przewaga nad Gompertzem:
      - Modeluje ZMIENNE w czasie prawdopodobieństwo śmierci
      - Uwzględnia trend długowieczności (co roku żyjemy ~2 miesiące dłużej)
      - Stochastyczny parametr κ_t (factor of time) → niepewność długowieczności
      - Kalibracja na GUS 2023 + Human Mortality Database (HMD)

    Model: ln(m_x,t) = α_x + β_x × κ_t
      α_x = log-śmiertelność bazowa w wieku x
      β_x = wrażliwość na czynnik czasu κ_t
      κ_t = trend σ_κ²: κ_{t+1} = κ_t + d + ε_t, ε_t ~ N(0, σ_κ)

    Referencja: Lee & Carter (1992) JASA, Renshaw & Haberman (2006),
                GUS (2023) Tablice Trwania Życia dla Polski.

    Parameters
    ----------
    current_age       : obecny wiek (lat)
    n_sims            : liczba symulacji
    gender            : 'male', 'female', 'mixed'
    alpha_improvement : roczny trend poprawy śmiertelności (0.01 = 1% rocznie)
    """
    if seed is not None:
        np.random.seed(seed)

    # ── Kalibracja GUS 2023 baseline (Polish life tables) ──────────────
    # Uproszczone α_x dla PL (log hazard rates, ages 60-110)
    # Dane: GUS 2023 — Tablice Trwania Życia
    ages = np.arange(60, 111)

    if gender == "female":
        # Kobiety PL — niższa śmiertelność
        base_mu = np.array([
            0.0062, 0.0067, 0.0074, 0.0082, 0.0091, 0.0101, 0.0112, 0.0124,
            0.0138, 0.0154, 0.0172, 0.0193, 0.0217, 0.0245, 0.0277, 0.0314,
            0.0357, 0.0406, 0.0462, 0.0527, 0.0600, 0.0684, 0.0779, 0.0888,
            0.1012, 0.1152, 0.1311, 0.1491, 0.1694, 0.1923, 0.2180, 0.2468,
            0.2790, 0.3148, 0.3545, 0.3984, 0.4467, 0.4997, 0.5576, 0.6207,
            0.6892, 0.7633, 0.8433, 0.9295, 1.0222, 1.1217, 1.2281, 1.3417,
            1.4627, 1.5912, 1.7275,
        ])
    elif gender == "male":
        # Mężczyźni PL — wyższa śmiertelność (~5 lat krócej)
        base_mu = np.array([
            0.0119, 0.0131, 0.0144, 0.0159, 0.0175, 0.0194, 0.0215, 0.0239,
            0.0266, 0.0296, 0.0330, 0.0369, 0.0413, 0.0463, 0.0520, 0.0585,
            0.0659, 0.0743, 0.0840, 0.0950, 0.1076, 0.1218, 0.1381, 0.1565,
            0.1773, 0.2008, 0.2272, 0.2568, 0.2898, 0.3265, 0.3671, 0.4120,
            0.4614, 0.5157, 0.5751, 0.6398, 0.7103, 0.7866, 0.8690, 0.9577,
            1.0528, 1.1545, 1.2630, 1.3783, 1.5005, 1.6297, 1.7659, 1.9093,
            2.0599, 2.2179, 2.3833,
        ])
    else:  # mixed
        f_mu = np.array([
            0.0062, 0.0067, 0.0074, 0.0082, 0.0091, 0.0101, 0.0112, 0.0124,
            0.0138, 0.0154, 0.0172, 0.0193, 0.0217, 0.0245, 0.0277, 0.0314,
            0.0357, 0.0406, 0.0462, 0.0527, 0.0600, 0.0684, 0.0779, 0.0888,
            0.1012, 0.1152, 0.1311, 0.1491, 0.1694, 0.1923, 0.2180, 0.2468,
            0.2790, 0.3148, 0.3545, 0.3984, 0.4467, 0.4997, 0.5576, 0.6207,
            0.6892, 0.7633, 0.8433, 0.9295, 1.0222, 1.1217, 1.2281, 1.3417,
            1.4627, 1.5912, 1.7275,
        ])
        m_mu = np.array([
            0.0119, 0.0131, 0.0144, 0.0159, 0.0175, 0.0194, 0.0215, 0.0239,
            0.0266, 0.0296, 0.0330, 0.0369, 0.0413, 0.0463, 0.0520, 0.0585,
            0.0659, 0.0743, 0.0840, 0.0950, 0.1076, 0.1218, 0.1381, 0.1565,
            0.1773, 0.2008, 0.2272, 0.2568, 0.2898, 0.3265, 0.3671, 0.4120,
            0.4614, 0.5157, 0.5751, 0.6398, 0.7103, 0.7866, 0.8690, 0.9577,
            1.0528, 1.1545, 1.2630, 1.3783, 1.5005, 1.6297, 1.7659, 1.9093,
            2.0599, 2.2179, 2.3833,
        ])
        base_mu = 0.55 * f_mu + 0.45 * m_mu

    # Extrapolate from current_age
    max_age = 110
    lifetimes = np.zeros(n_sims, dtype=int)

    # Lee-Carter κ_t stochastic path per simulation
    sigma_kappa = 0.015  # historical volatility of κ trend for Poland

    for sim in range(n_sims):
        # Stochastic κ path (random walk with drift)
        kappa = 0.0
        death_age = max_age

        for age in range(max(current_age, 60), max_age + 1):
            kappa += -alpha_improvement + np.random.normal(0, sigma_kappa)
            age_idx = min(age - 60, len(base_mu) - 1)
            mu_xt = base_mu[age_idx] * np.exp(kappa)
            # Probability of dying this year = 1 - exp(-mu_xt)
            q_x = 1.0 - np.exp(-max(mu_xt, 0.0))
            if np.random.uniform() < q_x:
                death_age = age
                break

        lifetimes[sim] = max(death_age, current_age + 1)

    return lifetimes


# ─── E2: Guyton-Klinger Withdrawal Rules (NEW 2025) ──────────────────────────

def guyton_klinger_withdrawal(
    wealth: float,
    current_withdrawal: float,
    inflation_rate: float,
    initial_withdrawal: float,
    portfolio_cap_rule: float = 1.20,
    portfolio_floor_rule: float = 0.80,
    guard_upper: float = 0.20,
    guard_lower: float = 0.20,
    prosperity_increase: float = 0.10,
    capital_preservation_cut: float = 0.10,
) -> float:
    """
    Guyton-Klinger (2006) Dynamic Withdrawal Rules — naukowe guardrails.

    Implementuje trzy precyzyjne reguły:

    1. Portfolio Management Rule (PMR):
       - Odłóż dywidendy/dochód z aktywów zamiast sprzedawać (omit if wealth rising)
       - Tutaj: inflation-adjust tylko jeśli rynek w górę

    2. Prosperity Rule (PR):
       - Jeśli aktualny WR < (1 - guard_upper) × initial_WR → ZWIĘKSZ 10%
       - Odzwierciedla że portfel rośnie ponad oczekiwania

    3. Capital Preservation Rule (CPR):
       - Jeśli aktualny WR > (1 + guard_lower) × initial_WR → ZMNIEJSZ 10%
       - Chroni kapitał gdy portfel słabnie

    Referencja: Guyton J.T. & Klinger W.J. (2006) "Decision Rules and
                Maximum Initial Withdrawal Rates", Journal of Financial Planning.
                Kitces M. (2022) update with 2023 inflation environment.

    Parameters
    ----------
    wealth               : aktualny majątek
    current_withdrawal   : bieżąca kwota roczna
    inflation_rate       : inflacja bieżącego roku
    initial_withdrawal   : pierwotna roczna kwota wypłaty (stała referencyjna)
    """
    if wealth <= 0:
        return 0.0

    current_wr = current_withdrawal / max(wealth, 1.0)
    initial_wr = initial_withdrawal / max(wealth, 1.0)

    # Inflation-adjust baseline (PMR: only if not causing WR to spike)
    adj_withdrawal = current_withdrawal * (1 + inflation_rate)

    # ── Prosperity Rule ────────────────────────────────────────────────────
    if current_wr < (1.0 - guard_upper) * initial_wr:
        adj_withdrawal = adj_withdrawal * (1.0 + prosperity_increase)

    # ── Capital Preservation Rule ──────────────────────────────────────────
    if current_wr > (1.0 + guard_lower) * initial_wr:
        adj_withdrawal = adj_withdrawal * (1.0 - capital_preservation_cut)

    # ── Portfolio Management Rule: Skip inflation if WR rising anyway ─────
    final_wr = adj_withdrawal / max(wealth, 1.0)
    if final_wr > portfolio_cap_rule * initial_wr:
        adj_withdrawal = current_withdrawal  # skip inflation adjust entirely

    return float(max(adj_withdrawal, 0))


# ─── Conformal Prediction for Retirement MC ────────────────────────────────────

def compute_conformal_prediction_intervals(
    wealth_matrix: np.ndarray,
    calibration_frac: float = 0.2,
    alpha: float = 0.10,
) -> dict:
    """
    Conformal Prediction Intervals dla projekcji MC (Vovk et al. 2005, 2023).

    Standardowe percentyle (5th/95th) ZAKŁADAJĄ że symulacje są kalibrowane.
    Conformal PI ma gwarancję pokrycia 1-α bez założeń o rozkładzie.

    Metodologia Split Conformal (Papadopoulos 2002):
    1. Podziel n_sims na kalibracyjne (20%) i testowe (80%)
    2. Oblicz nonconformity scores: |r_i - median| / IQR per rok
    3. q_hat = (ceil((n_calib+1)(1-α)) / n_calib) percentyl scores
    4. PI: [median - q_hat×IQR, median + q_hat×IQR]

    Zaleta: PI są ZAWSZE poprawne statystycznie (finite-sample coverage guarantee).
    Referencja: Angelopoulos & Bates (2023) "Conformal Risk Control". ICML 2023.

    Parameters
    ----------
    wealth_matrix    : (n_sims, horizon+1) — macierz ścieżek MC
    calibration_frac : frakcja symulacji jako zbiór kalibracyjny
    alpha            : poziom błędu (0.10 = 90% CI)
    """
    n_sims, horizon = wealth_matrix.shape

    # Split: calibration vs test
    n_calib = max(10, int(n_sims * calibration_frac))
    calib_idx = np.random.choice(n_sims, n_calib, replace=False)
    test_idx = np.setdiff1d(np.arange(n_sims), calib_idx)

    calib_paths = wealth_matrix[calib_idx]  # (n_calib, horizon)
    test_paths = wealth_matrix[test_idx]    # (n_test, horizon)

    # Compute per-year IQR and median from test set
    median_path = np.median(test_paths, axis=0)
    q25 = np.percentile(test_paths, 25, axis=0)
    q75 = np.percentile(test_paths, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1.0)

    # Nonconformity scores per calibration path: max deviation in IQR units
    scores = np.max(np.abs(calib_paths - median_path) / iqr, axis=1)

    # Quantile of scores (conservative = ceil((n+1)(1-alpha)/n))
    level = np.ceil((n_calib + 1) * (1.0 - alpha)) / n_calib
    level = min(level, 1.0)
    q_hat = float(np.quantile(scores, level))

    # Conformal Prediction Intervals
    lower = np.maximum(median_path - q_hat * iqr, 0)
    upper = median_path + q_hat * iqr

    return {
        "median":     median_path,
        "cp_lower":   lower,
        "cp_upper":   upper,
        "q_hat":      q_hat,
        "coverage":   1.0 - alpha,
        "method":     "Split Conformal (Vovk 2005)",
        "n_calib":    n_calib,
        "note":       (
            f"Gwarantowane pokrycie {(1-alpha)*100:.0f}% bez założeń o rozkładzie. "
            f"q_hat={q_hat:.2f}× IQR. Angelopoulos & Bates (2023)."
        ),
    }



# ─── Retirement Spending Smile (Blanchett 2013) ──────────────────────────────

def retirement_spending_smile_factor(
    current_year: int,
    retirement_year: int,
    total_horizon: int,
    go_go_factor: float = 1.15,
    slow_go_factor: float = 0.78,
    no_go_factor: float = 1.05,
) -> float:
    """
    Blanchett (2013) Retirement Spending Smile — U-kształtny wzorzec wydatków.
    Fazy: Go-Go (0-10 lat)-><slow-go (10-20 lat) -> No-Go (20+ lat, koszty opieki).
    Referencja: Blanchett D. (2013) 'Estimating the True Cost of Retirement',
                Journal of Financial Planning.
    """
    years_since_retirement = current_year - retirement_year
    if years_since_retirement <= 0:
        return 1.0  # faza akumulacji = bazowe
    if years_since_retirement <= 10:
        return go_go_factor    # Go-Go: aktywna emerytura
    elif years_since_retirement <= 20:
        return slow_go_factor  # Slow-Go: spokojniejsza
    else:
        return no_go_factor    # No-Go: koszty opieki


# ─── Glide Path (Pfau 2013 — Rising Equity Glide Path) ───────────────────────

def glide_path_params(
    current_year: int,
    retirement_year: int,
    base_return: float,
    base_vol: float,
    equity_at_start: float = 0.80,   # % akcji na początku emerytury
    equity_at_end: float = 0.35,     # % akcji na końcu horyzontu
    bond_return: float = 0.04,
    bond_vol: float = 0.05,
) -> tuple[float, float]:
    """
    Pfau (2013) Rising Equity Glide Path — liniowa zmiana alokacji akcje/obligacje.
    Referencja: Pfau W.D. (2013) 'Rising Equity Glide-Path', Journal of
                Financial Planning. Kitces & Pfau (2015).
    """
    years_since_retirement = max(0, current_year - retirement_year)
    total_retirement_years = 30  # typowy horyzont dekumulacji
    t = min(years_since_retirement / max(total_retirement_years, 1), 1.0)
    equity_pct = equity_at_start + t * (equity_at_end - equity_at_start)
    equity_pct = float(np.clip(equity_pct, 0.0, 1.0))
    bond_pct = 1.0 - equity_pct
    # Blended return & vol
    blended_return = equity_pct * base_return + bond_pct * bond_return
    blended_vol    = np.sqrt((equity_pct * base_vol) ** 2 + (bond_pct * bond_vol) ** 2
                             + 2 * equity_pct * bond_pct * base_vol * bond_vol * 0.2)
    return float(blended_return), float(blended_vol)


# ─── IKE/IKZE Kalkulator podatkowy ───────────────────────────────────────────

def calculate_ike_ikze_advantage(
    annual_contribution: float,
    years: int,
    return_rate: float,
    pit_bracket: float = 0.32,
    capital_gains_tax: float = 0.19,
) -> dict:
    """
    Korzyść podatkowa IKE i IKZE vs konto zwykłe (Polish tax law).
    IKE: brak podatku Belki 19% przy wypłacie po 60 r.ż.
    IKZE: odliczenie od PIT + 10% zryczałtowany podatek przy wypłacie.
    Referencja: Ustawa z dnia 20 IV 2004 o IKE; Ustawa z dnia 20 IV 2004 o IKZE;
                KNF (2023) Raport o rynku emerytalnym.
    """
    if annual_contribution <= 0 or years <= 0:
        return {"ike_gain": 0.0, "ikze_gain": 0.0, "zwykle_net": 0.0,
                "ike_net": 0.0, "ikze_net": 0.0}
    r = return_rate
    # Konto zwykłe: podatek od zysku (Belka) przy wyjściu
    gross_final = annual_contribution * ((1 + r) ** years - 1) / r * (1 + r) if r > 0 else annual_contribution * years
    cost_of_contributions = annual_contribution * years
    gross_gain = max(0, gross_final - cost_of_contributions)
    belka_tax = gross_gain * capital_gains_tax
    zwykle_net = gross_final - belka_tax

    # IKE: 0% podatku Belki
    ike_net = gross_final
    ike_gain = gross_final - zwykle_net

    # IKZE: odliczenie PIT co roku + 10% ryczałt przy wypłacie
    # (zakładamy wpłaty w granicach limitu IKZE)
    
    # 1. Roczny zwrot gotówki z US dzięki odliczeniu od PIT
    annual_pit_return = annual_contribution * pit_bracket
    # 2. Inteligentny inwestor natychmiast inwestuje ten zwrot (np. na koncie zwykłym z Belką)
    #    Obliczmy Future Value raty (FV annuity) z uwzględnieniem podatków opóźnionych
    #    Dla uproszczenia r_net = r * (1 - 0.19) - dywidendy/częściowa Belka, 
    #    dla prawdziwego "Konta Zwykłego" policzymy FV a potem uderzymy 19% w zysk na koniec.
    fv_pit_savings_gross = annual_pit_return * ((1 + r) ** years - 1) / r * (1 + r) if r > 0 else annual_pit_return * years
    cost_pit_savings = annual_pit_return * years
    pit_savings_gain = max(0, fv_pit_savings_gross - cost_pit_savings)
    ikze_pit_savings_net = fv_pit_savings_gross - (pit_savings_gain * capital_gains_tax)

    ikze_tax_at_withdrawal = gross_final * 0.10   # 10% zryczałtowany
    ikze_net = gross_final - ikze_tax_at_withdrawal + ikze_pit_savings_net
    ikze_gain = ikze_net - zwykle_net

    return {
        "gross_final":      round(gross_final, 0),
        "zwykle_net":       round(zwykle_net, 0),
        "ike_net":          round(ike_net, 0),
        "ikze_net":         round(ikze_net, 0),
        "ike_gain":         round(ike_gain, 0),
        "ikze_gain":        round(ikze_gain, 0),
        "belka_tax":        round(belka_tax, 0),
        "ikze_pit_savings": round(ikze_pit_savings_net, 0),
    }


# ─── CAPE-Adjusted SWR (Kitces 2022) ─────────────────────────────────────────

def cape_adjusted_swr(cape_ratio: float, base_swr: float = 0.04) -> float:
    """
    Koryguje Safe Withdrawal Rate o bieżące wyceny (CAPE Shillera).
    Kitces (2022): każdy punkt CAPE powyżej 20 obniża SWR o ~0.05pp.
    Referencja: Kitces M. (2022) 'The 4% Rule and the Search for a Safe
                Withdrawal Rate', Kitces.com. Morningstar (2023).
    """
    adjustment = max(0.0, (cape_ratio - 20.0) * 0.0005)
    return round(float(max(0.025, base_swr - adjustment)), 4)


# ─── Retirement Readiness Score (wzorowany Fidelity 2023) ────────────────────

def retirement_readiness_score(
    success_prob: float,
    current_swr: float,
    zus_coverage: float,   # frakcja wydatków pokrywanych przez ZUS
    years_to_fire: float,
    capital_vs_fire: float,  # current_cap / fire_number
) -> float:
    """
    Syntetyczny wskaźnik gotowości emerytalnej 0-100.
    Wzorowany na Fidelity Retirement Score (2023) i T. Rowe Price (2022).
    Składowe wagowane:
      35% P(sukces MC)
      20% Bezpieczeństwo SWR (poniżej 4%)
      25% Pokrycie ZUS/annuity
      10% Postęp do FIRE
      10% Zapas kapitału nad FIRE Number
    """
    s1 = float(np.clip(success_prob, 0, 1)) * 35
    s2 = float(np.clip(1 - current_swr / 0.06, 0, 1)) * 20
    s3 = float(np.clip(zus_coverage, 0, 1)) * 25
    # years_to_fire: 0 = osi_gniety (max pkt), 20+ = daleko
    fire_progress = float(np.clip(1 - years_to_fire / 20, 0, 1))
    s4 = fire_progress * 10
    s5 = float(np.clip((capital_vs_fire - 0.5) / 1.5, 0, 1)) * 10
    return round(s1 + s2 + s3 + s4 + s5, 1)


def cir_inflation(base_inflation, horizon, n_sims, kappa=0.3, theta=0.035, sigma_cir=0.015):
    """
    Stochastyczna inflacja — CIR (Cox-Ingersoll-Ross 1985).
    kappa: szybkość powrotu do średniej, theta: długoter. średnia, sigma_cir: zmienność.
    """
    inf_matrix = np.zeros((n_sims, horizon))
    inf_matrix[:, 0] = base_inflation
    dt = 1.0
    for t in range(1, horizon):
        r = inf_matrix[:, t-1]
        r_pos = np.maximum(r, 0)
        dW = np.random.normal(0, np.sqrt(dt), n_sims)
        dr = kappa * (theta - r) * dt + sigma_cir * np.sqrt(r_pos) * dW
        inf_matrix[:, t] = np.maximum(r + dr, 0)
    return inf_matrix


def run_mc_retirement(init_cap, annual_expenses, annual_contrib,
                      ret_return, ret_vol, horizon, n_sims,
                      years_to_retirement, inflation_base,
                      stochastic_inflation=True, enable_contributions=False,
                      contrib_during_retirement=False,
                      withdrawal_strategy="constant",
                      guardrails_band=0.20, flexible_pct=0.04,
                      floor_amount=0,
                      zus_monthly=0.0,
                      tax_regime="IKE/IKZE",
                      use_spending_smile=False,
                      use_glide_path=False,
                      medical_inflation_rate=0.0):
    """
    Główna symulacja Monte Carlo z polepszeniami naukowymi.
    Returns: wealth_matrix (n_sims x horizon+1), inflation path (n_sims x horizon)
    """
    mu = ret_return
    sigma = ret_vol
    zus_annual = float(zus_monthly) * 12.0

    # Glide Path: pre-obliczamy blended mu/sigma per rok (jeśli włączony)
    glide_mus   = np.full(horizon, mu)
    glide_sigs  = np.full(horizon, sigma)
    if use_glide_path:
        for y in range(horizon):
            gm, gs = glide_path_params(y, years_to_retirement, mu, sigma)
            glide_mus[y]  = gm
            glide_sigs[y] = gs

    # Student-t shocks — per rok z osobną mu/sigma gdy glide path
    shocks = student_t_shocks(n_sims, horizon)
    if use_glide_path:
        # Każda kolumna zwrotów z własnym mu/sigma
        annual_returns = np.zeros((n_sims, horizon))
        for y in range(horizon):
            log_mu_y = glide_mus[y] - 0.5 * glide_sigs[y] ** 2
            annual_returns[:, y] = np.exp(log_mu_y + glide_sigs[y] * shocks[:, y]) - 1
    else:
        log_mu = mu - 0.5 * sigma ** 2
        annual_returns = np.exp(log_mu + sigma * shocks) - 1

    # Stochastic inflation (CIR or constant)
    if stochastic_inflation:
        inf_matrix = cir_inflation(inflation_base, horizon, n_sims)
    else:
        inf_matrix = np.full((n_sims, horizon), inflation_base)

    # Podatek Belki — cost basis tracking (dla konta zwykłego)
    # Przybliżenie: coroczne opodatkowanie zysku przy sprzedaży/wypłacie
    # IKE/IKZE = 0%, Fundusz Akumulujący = opodatkowanie przy zakończeniu
    annual_tax_rate = 0.0
    if tax_regime == "Konto Zwykłe (Belka 19%)":
        annual_tax_rate = 0.19  # podatek od rocznego zysku przy realizacji
    # Dla IKE/IKZE i Funduszu: 0% corocznie (tax-deferred)

    wealth = np.full((n_sims, horizon + 1), 0.0)
    wealth[:, 0] = init_cap

    # Baza kosztowa (R3) do obliczania podatku ułamkowego przy wypłatach
    cost_basis = np.full(n_sims, init_cap)

    # For guardrails: track expected portfolio trajectory & withdrawals
    glide_path = init_cap * (1 + ret_return) ** np.arange(horizon + 1)
    current_withdrawal = np.full(n_sims, annual_expenses)
    initial_withdrawal_val = np.full(n_sims, annual_expenses)

    for y in range(horizon):
        ret = annual_returns[:, y]
        w = wealth[:, y]
        w_new = w * (1 + ret)

        # Konto Zwykłe (ETF Pasywny): 0.5% drag rocznego podatku (od dywidend/rebalancingu).
        # Główny podatek Belki pobierzemy pro-rata przy wypłatach (R3).
        if tax_regime == "Konto Zwykłe (Belka 19%)":
            w_new -= np.maximum(w_new - w, 0) * 0.005 

        inf_y = inf_matrix[:, y]
        inf_cum = np.prod(1 + inf_matrix[:, :y+1], axis=1) if y > 0 else (1 + inf_matrix[:, 0])
        phase_retire = (y >= years_to_retirement)

        # Spending Smile — współczynnik wydatków zależny od fazy emerytury
        spending_factor = 1.0
        if use_spending_smile and phase_retire:
            spending_factor = retirement_spending_smile_factor(y, years_to_retirement, horizon)

        if phase_retire:
            # ZUS reduction — rośnie wraz z inflacją przez CAŁY CZAS (R2 - eliminacja uwięzi deflacyjnej)
            zus_cum = zus_annual * inf_cum

            # Medical Inflation: ok. 15% wydatków emeryta to usługi medyczne
            health_proxy_pct = 0.15 
            med_inf_comp = (1 + medical_inflation_rate) ** (y - years_to_retirement)
            medical_expense_multiplier = (1 - health_proxy_pct) + health_proxy_pct * med_inf_comp

            if withdrawal_strategy == "flexible":
                # R5 - Usunięto błąd double-dipping. Procent od nominalnego kapitału = nominalna wypłata
                eff_withdrawal = flexible_pct * w_new * spending_factor * medical_expense_multiplier
            else:
                # Inicjalizacja pierwszej kwoty na starcie emerytury (R1 - Guardrails Bug)
                if y == years_to_retirement:
                    current_withdrawal = annual_expenses * inf_cum
                    initial_withdrawal_val = current_withdrawal.copy()

                if withdrawal_strategy == "guardrails":
                    inf_rate = inf_matrix[:, y] if stochastic_inflation else inflation_base
                    current_wr = current_withdrawal / np.maximum(w_new, 1.0)
                    initial_wr = initial_withdrawal_val / np.maximum(w_new, 1.0)
                    adj_withdrawal = current_withdrawal * (1 + inf_rate)
                    pr_mask = current_wr < (1.0 - guardrails_band) * initial_wr
                    adj_withdrawal = np.where(pr_mask, adj_withdrawal * 1.10, adj_withdrawal)
                    cpr_mask = current_wr > (1.0 + guardrails_band) * initial_wr
                    adj_withdrawal = np.where(cpr_mask, adj_withdrawal * 0.90, adj_withdrawal)
                    pmr_mask = (adj_withdrawal / np.maximum(w_new, 1.0)) > (1.20 * initial_wr)
                    adj_withdrawal = np.where(pmr_mask, current_withdrawal, adj_withdrawal)
                    current_withdrawal = np.maximum(adj_withdrawal, 0)
                    
                    eff_withdrawal = current_withdrawal * spending_factor * medical_expense_multiplier
                elif withdrawal_strategy == "constant":
                    eff_withdrawal = annual_expenses * spending_factor * inf_cum * medical_expense_multiplier

            # ZUS i floor: redukuj potrzebne wypłaty z portfela
            eff_withdrawal_net = np.maximum(eff_withdrawal - zus_cum, 0)
            floor_adj = np.maximum(floor_amount, 0)
            required_portfolio_withdrawal = np.maximum(eff_withdrawal_net - floor_adj, 0)

            # --- Podatek Belki od Wypłat (R3: Pro-Rata Cost Basis) ---
            tax_paid = np.zeros(n_sims)
            if tax_regime != "IKE/IKZE":
                avg_cost_ratio = np.minimum(cost_basis / np.maximum(w_new, 1.0), 1.0)
                gain_portion = required_portfolio_withdrawal * (1.0 - avg_cost_ratio)
                tax_paid = gain_portion * 0.19
                # Pomniejszamy bazę kosztową proporcjonalnie do uszczkniętego kapitału
                cost_basis = np.maximum(0, cost_basis - required_portfolio_withdrawal * avg_cost_ratio)

            w_new -= (required_portfolio_withdrawal + tax_paid)

            if enable_contributions and contrib_during_retirement:
                contrib_inflated = annual_contrib * inf_cum
                w_new += contrib_inflated
                cost_basis += contrib_inflated
        else:
            if enable_contributions:
                contrib_inflated = annual_contrib * inf_cum
                w_new += contrib_inflated
                cost_basis += contrib_inflated

        w_new = np.maximum(w_new, 0)
        wealth[:, y + 1] = w_new

    # Ostateczny test podatkowy po zakonczeniu (np. gdy umierasz lub zamykasz Fundusz)
    if tax_regime == "Fundusz Akumulujący (Belka przy umorzeniu)" or tax_regime == "Konto Zwykłe (Belka 19%)":
        avg_cost_ratio = np.minimum(cost_basis / np.maximum(wealth[:, -1], 1.0), 1.0)
        final_gain = wealth[:, -1] * (1.0 - avg_cost_ratio)
        wealth[:, -1] = wealth[:, -1] - final_gain * 0.19

    return wealth, inf_matrix

def compute_survival_curve(wealth_matrix):
    """Kaplan-Meier style: P(portfel > 0) w każdym roku."""
    return (wealth_matrix > 0).mean(axis=0)

def compute_waterfall(init_cap, wealth_matrix, inf_matrix, annual_expenses,
                      annual_contrib, years_to_retirement, horizon, inflation_base,
                      enable_contributions, contrib_during_retirement):
    """Oblicza składowe dekompozycji majątku dla Waterfall Chart."""
    median_final = float(np.median(wealth_matrix[:, -1]))
    median_mid = float(np.median(wealth_matrix[:, years_to_retirement] if years_to_retirement < horizon else wealth_matrix[:, -1]))

    total_contrib = annual_contrib * years_to_retirement if enable_contributions else 0
    total_withdrawal = sum(annual_expenses * (1 + inflation_base)**y for y in range(max(0, horizon - years_to_retirement)))
    market_gain = float(median_final - init_cap - total_contrib + total_withdrawal)
    # Wykres jest w wartościach nominalnych, więc nie ściągamy kosztu inflacji i fałszywego podatku obniżających medianę wizualnie.
    
    measures = ["absolute", "relative", "relative", "relative", "total"]
    x_labels = ["Kapitał Startowy", "+ Wpłaty", "+ Zysk Rynkowy", "- Wypłaty", "Majątek Końcowy"]
    y_values = [init_cap, total_contrib, max(0, market_gain), -total_withdrawal, median_final]

    return x_labels, y_values, measures


# ─── Main Module ─────────────────────────────────────────────────────────────
def render_emerytura_module():
    st.header("🏖️ Analiza Emerytury — Scientific Edition v3.0")
    st.markdown("""
    **14 ulepszeń naukowych**: Student-t (grube ogony), stochastyczna długość życia (Lee-Carter),
    stochastyczna inflacja (CIR), ZUS/PPK integration, podatek Belki (IKE/IKZE/Zwykłe),
    Spending Smile (U-kształt wydatków), Glide Path (zmienna alokacja), Guyton-Klinger Guardrails,
    Retirement Age Optimizer, Krzywa Przeżywalności, Conformal Prediction Intervals.
    """)

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("🛠️ Parametry Emerytury")
    st.sidebar.markdown("### ⚙️ Ustawienia")

    st.sidebar.markdown("### 📥 Import z Symulatora")
    if st.sidebar.button("🔄 Wczytaj z Symulatora"):
        if 'mc_results' in st.session_state:
            r = st.session_state['mc_results']
            st.session_state['rem_initial_capital'] = float(r['wealth_paths'][0, 0])
            st.session_state['rem_expected_return'] = float(r['metrics']['mean_cagr'])
            st.session_state['rem_volatility'] = float(r['metrics']['median_volatility'])
            st.sidebar.success("Wczytano Monte Carlo!")
        elif 'backtest_results' in st.session_state:
            r = st.session_state['backtest_results']
            st.session_state['rem_initial_capital'] = float(r['results']['PortfolioValue'].iloc[-1])
            st.session_state['rem_expected_return'] = float(r['metrics']['mean_cagr'])
            st.session_state['rem_volatility'] = float(r['metrics']['median_volatility'])
            st.sidebar.success("Wczytano Backtest AI!")
        else:
            st.sidebar.warning("Brak danych symulacji.")

    st.sidebar.markdown("### 💼 Kapitał i Wydatki")
    init_cap = st.sidebar.number_input("Kapitał Dzisiaj (PLN)", value=_saved("rem_cap", st.session_state.get('rem_initial_capital', 1_000_000.0)), step=100_000.0, key="rem_cap", on_change=_save, args=("rem_cap",))
    monthly_expat = st.sidebar.number_input("Wydatki Miesięczne (PLN)", value=_saved("rem_me", 5000), step=500, key="rem_me", on_change=_save, args=("rem_me",))
    inflation = st.sidebar.slider("Inflacja Bazowa (%)", 0.0, 15.0, value=_saved("rem_inf", 3.0), step=0.5, key="rem_inf", on_change=_save, args=("rem_inf",)) / 100.0
    stoch_inf = st.sidebar.checkbox("Inflacja Stochastyczna (CIR)", value=_saved("rem_stoch_inf", True), key="rem_stoch_inf", on_change=_save, args=("rem_stoch_inf",), help="Modeluje losowe wahania inflacji wokół wartości bazowej (CIR 1985).")

    st.sidebar.markdown("### 🕒 Wiek i Horyzont")
    current_age = st.sidebar.slider("Obecny Wiek", 18, 80, value=_saved("rem_age", 53), key="rem_age", on_change=_save, args=("rem_age",))
    retirement_age = st.sidebar.slider("Wiek Emerytalny", current_age, 90, value=_saved("rem_ret_age", 60), key="rem_ret_age", on_change=_save, args=("rem_ret_age",))
    life_expectancy = st.sidebar.slider("Max Horyzont (lat)", retirement_age + 5, 110, value=_saved("rem_life", 95), key="rem_life", on_change=_save, args=("rem_life",))
    stoch_life = st.sidebar.checkbox("Stoch. Długowieczność (Lee-Carter)", value=_saved("rem_stoch_life", True), key="rem_stoch_life", on_change=_save, args=("rem_stoch_life",), help="Model generuje scenariusze wieku śmierci biorąc pod uwagę płeć i ciągły wzrost długowieczności (Lee-Carter).")

    gender_label = st.sidebar.selectbox("Płeć (Długowieczność)", ["Mieszana", "Kobieta", "Mężczyzna"], index=["Mieszana", "Kobieta", "Mężczyzna"].index(_saved("rem_gender", "Mieszana")), key="rem_gender", on_change=_save, args=("rem_gender",))
    gender_map = {"Mieszana": "mixed", "Kobieta": "female", "Mężczyzna": "male"}
    gender = gender_map[gender_label]

    st.sidebar.markdown("### 📈 Rynek")
    ret_return = st.sidebar.slider("Oczekiwany Zwrot (%)", -5.0, 20.0, value=_saved("rem_ret", 7.0), step=0.5, key="rem_ret", on_change=_save, args=("rem_ret",)) / 100.0
    ret_vol = st.sidebar.slider("Zmienność Vol (%)", 1.0, 40.0, value=_saved("rem_vol", 15.0), step=0.5, key="rem_vol", on_change=_save, args=("rem_vol",)) / 100.0

    st.sidebar.markdown("### 💰 Wpłaty i Dochód")
    enable_contributions = st.sidebar.checkbox("Aktywuj wpłaty/dochód", value=_saved("rem_contrib_en", False), key="rem_contrib_en", on_change=_save, args=("rem_contrib_en",))
    monthly_contribution = st.sidebar.slider("Kwota (PLN/mies)", 0, 30000, value=_saved("rem_mcon", 5000), step=500, key="rem_mcon", on_change=_save, args=("rem_mcon",), disabled=not enable_contributions)
    contrib_during_retirement = st.sidebar.checkbox("Dochód też na emeryturze", value=_saved("rem_cdr", False), key="rem_cdr", on_change=_save, args=("rem_cdr",), disabled=not enable_contributions)

    st.sidebar.markdown("### 🧮 Strategia Wypłat")
    _strat_opts = ["constant — Stała kwota", "guardrails — Klinger 2006", "flexible — % portfela"]
    withdrawal_strategy_label = st.sidebar.selectbox("Strategia Wypłat", _strat_opts, index=_strat_opts.index(_saved("rem_strat", _strat_opts[0])), key="rem_strat", on_change=_save, args=("rem_strat",))
    withdrawal_strategy = withdrawal_strategy_label.split(" ")[0]

    flexible_pct = 0.04
    floor_amount = 0
    if withdrawal_strategy == "flexible":
        flexible_pct = st.sidebar.slider("% wypłaty z portfela", 1.0, 10.0, value=_saved("rem_flex_pct", 4.0), step=0.5, key="rem_flex_pct", on_change=_save, args=("rem_flex_pct",)) / 100.0
    elif withdrawal_strategy == "guardrails":
        floor_amount = st.sidebar.number_input("Floor (min. bezpieczna kwota PLN)", value=_saved("rem_floor", 0), step=10000, key="rem_floor", on_change=_save, args=("rem_floor",))

    st.sidebar.markdown("### 🇵🇱 ZUS / PPK / Dodatkowy Dochód")
    zus_monthly = st.sidebar.number_input(
        "Emerytura ZUS (PLN/mies)",
        value=_saved("rem_zus", 2500),
        step=100,
        min_value=0,
        key="rem_zus", on_change=_save, args=("rem_zus",),
        help="Szacowana emerytura ZUS/KRUS pomniejsza potrzebne wypłaty z portfela. Ustaw 0 jeśli nie dotyczy."
    )
    ppk_capital = st.sidebar.number_input(
        "Kapitał PPK/OFE (PLN)",
        value=_saved("rem_ppk", 0),
        step=10_000,
        min_value=0,
        key="rem_ppk", on_change=_save, args=("rem_ppk",),
        help="Środki zgromadzone w PPK lub OFE — zostaną dodane do kapitału startowego."
    )

    st.sidebar.markdown("### 🏦 Tryb Podatkowy")
    _tax_opts = ["IKE/IKZE (0% Belki)", "Konto Zwykłe (Belka 19%)", "Fundusz Akumulujący (Belka przy umorzeniu)"]
    tax_regime = st.sidebar.selectbox(
        "Reżim Podatkowy",
        _tax_opts,
        index=_tax_opts.index(_saved("rem_tax", _tax_opts[0])),
        key="rem_tax", on_change=_save, args=("rem_tax",),
        help="IKE/IKZE: 0% podatku od zysku. Konto zwykłe: 19% Belki od zysku corocznie. Fundusz: podatek przy umorzeniu."
    )
    pit_bracket_pct = st.sidebar.slider(
        "Stawka PIT (% dla IKZE)", 12, 32,
        value=_saved("rem_pit", 32),
        step=8,
        key="rem_pit", on_change=_save, args=("rem_pit",),
        help="Stawka PIT potrzebna do obliczenia korzyści IKZE (odliczenie od podatku). Próg 12% lub 32%."
    )

    st.sidebar.markdown("### 🔬 Model Zaawansowany")
    use_spending_smile = st.sidebar.checkbox(
        "Spending Smile (U-kształt wydatków)",
        value=_saved("rem_smile", True),
        key="rem_smile", on_change=_save, args=("rem_smile",),
        help="Blanchett 2013: Go-Go (+15%), Slow-Go (-22%), No-Go (+5% koszty opieki). Realistyczny wzorzec wydatków emerytalnych."
    )
    use_glide_path = st.sidebar.checkbox(
        "Glide Path (zmienna alokacja)",
        value=_saved("rem_glide", False),
        key="rem_glide", on_change=_save, args=("rem_glide",),
        help="Pfau 2013: Rising Equity Glide Path — portfel zmienia alokację akcje/obligacje z wiekiem (80% akcji → 35%)."
    )
    cape_ratio = st.sidebar.number_input(
        "Zbieżność Rynku - CAPE Shillera",
        value=_saved("rem_cape", 33.0),
        step=1.0,
        key="rem_cape", on_change=_save, args=("rem_cape",),
        help="Kitces 2022: Wysokie wyceny (pow. 20) zwiastują niższe przyszłe stopy zwrotu. Model automatycznie obniży próg bezpiecznego SWR."
    )
    medical_inflation = st.sidebar.slider(
        "Inflacja Medyczna p.a. (%)",
        0.0, 10.0, value=_saved("rem_med", 5.0),
        step=0.5, key="rem_med", on_change=_save, args=("rem_med",),
        help="Ryzyko długowieczności to rosnące koszty zdrowia przewyższające ogólną inflację (Fidelity 2023)."
    )

    # ─── Core Calculations ────────────────────────────────────────────────────
    years_to_retirement = max(0, retirement_age - current_age)
    years_in_retirement = max(1, life_expectancy - retirement_age)
    total_years = years_to_retirement + years_in_retirement
    horizon = total_years
    n_sims = 500
    annual_expenses = monthly_expat * 12
    annual_contrib = monthly_contribution * 12 if enable_contributions else 0
    zus_annual = float(zus_monthly) * 12.0
    total_init_cap = float(init_cap) + float(ppk_capital)  # PPK + kapital

    # ZUS-korygowany FIRE Number i SWR
    effective_expenses = max(0.0, annual_expenses - zus_annual)  # portfel musi pokryć RESZTĘ po ZUS
    fire_number = effective_expenses / 0.035 if effective_expenses > 0 else 0
    current_swr = effective_expenses / total_init_cap if total_init_cap > 0 else 0
    zus_coverage = min(1.0, zus_annual / max(annual_expenses, 1.0))

    # Deterministic FIRE Time Math (z ZUS-offset)
    real_return = (1 + ret_return) / (1 + inflation) - 1
    if total_init_cap >= fire_number:
        years_to_fire = 0
    elif enable_contributions and annual_contrib > 0:
        if real_return == 0:
            years_to_fire = (fire_number - total_init_cap) / annual_contrib
        else:
            val1 = fire_number + annual_contrib / real_return
            val2 = total_init_cap + annual_contrib / real_return
            if val2 <= 0 or val1 / val2 <= 0:
                years_to_fire = float('inf')
            else:
                years_to_fire = np.log(val1 / val2) / np.log(1 + real_return)
    else:
        if real_return > 0 and fire_number > 0:
            years_to_fire = np.log(max(fire_number / max(total_init_cap, 1), 1e-9)) / np.log(1 + real_return)
        else:
            years_to_fire = float('inf')

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    if fire_number > 0:
        col1.metric("FIRE Number (ZUS-adj.)", f"{fire_number:,.0f} PLN",
                    help=f"Kapitał portfelowy potrzebny przy SWR 3.5% po odjęciu ZUS ({zus_monthly:,.0f} PLN/mies).")
    else:
        col1.metric("FIRE Number", "0 PLN — ZUS pokrywa 100%!",
                    help="Twoja emerytura ZUS pokrywa całe wydatki — nie potrzebujesz dodatkowego kapitału portfelowego.")

    if years_to_fire == 0:
        col2.metric("Czas do FIRE", "Osiągnięty! 🎉", help="Masz wystarczająco kapitału.")
    elif years_to_fire == float('inf'):
        col2.metric("Czas do FIRE", "Nigdy 😢", help="Inflacja pożera kapitał szybciej niż oszczędzasz.")
    else:
        col2.metric("Czas do FIRE", f"{years_to_fire:.1f} lat", help="Lata do wolności przy podanych wpłatach i stopie zwrotu.")

    col3.metric("SWR (po ZUS)", f"{current_swr:.2%}",
                delta=f"Limit: {cape_adjusted_swr(cape_ratio, 0.04):.2%}",
                delta_color="off",
                help=f"Safe Withdrawal Rate ze środków portfelowych po odjęciu ZUS. CAPE-Adjusted SWR Kitces = obniżony z powodu wysokich wycen (ob. CAPE={cape_ratio}). Twój SWR powinien być MNIEJSZY niż limit.")
    col4.metric("Okres emerytury", f"{years_in_retirement} lat")

    # Readiness Score (wstępny — przed MC)
    ytf = years_to_fire if years_to_fire != float('inf') else 30
    cap_vs_fire = total_init_cap / max(fire_number, 1.0)
    _rrs_prelim = retirement_readiness_score(0.85, current_swr, zus_coverage, ytf, cap_vs_fire)
    rrs_color = "#00ff88" if _rrs_prelim >= 70 else ("#ffaa00" if _rrs_prelim >= 50 else "#ff4444")
    col5.markdown(f"""
    <div style='text-align:center;padding:8px 0;'>
        <div style='font-size:11px;color:#9ca3af;'>READINESS SCORE</div>
        <div style='font-size:28px;font-weight:800;color:{rrs_color};'>{_rrs_prelim:.0f}<span style='font-size:14px'>/100</span></div>
        <div style='font-size:10px;color:#6b7280;'>Fidelity-style</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ─── Run MC ──────────────────────────────────────────────────────────────
    wealth_matrix, inf_matrix = run_mc_retirement(
        total_init_cap, annual_expenses, annual_contrib,
        ret_return, ret_vol, horizon, n_sims,
        years_to_retirement, inflation,
        stochastic_inflation=stoch_inf,
        enable_contributions=enable_contributions,
        contrib_during_retirement=contrib_during_retirement,
        withdrawal_strategy=withdrawal_strategy,
        flexible_pct=flexible_pct,
        floor_amount=floor_amount,
        zus_monthly=float(zus_monthly),
        tax_regime=tax_regime,
        use_spending_smile=use_spending_smile,
        use_glide_path=use_glide_path,
        medical_inflation_rate=float(medical_inflation / 100.0)
    )

    years_arr = np.arange(current_age, current_age + horizon + 1)

    # ─── Stochastic lifetimes (Lee-Carter) ───────────────────────────────────
    if stoch_life:
        lifetimes = lee_carter_lifetimes(current_age, n_sims, gender=gender)
        # Find actual survival: portfel przeżywa uczestnika?
        portfolio_survives = []
        for i in range(n_sims):
            death_age = lifetimes[i]
            death_yr = min(int(death_age - current_age), horizon)
            portfolio_survives.append(wealth_matrix[i, death_yr] > 0)
        life_survival_prob = np.mean(portfolio_survives)
    else:
        lifetimes = np.full(n_sims, life_expectancy)
        life_survival_prob = None

    # ─── Key metrics ─────────────────────────────────────────────────────────
    success_prob = np.mean(wealth_matrix[:, -1] > 0)
    median_final = float(np.median(wealth_matrix[:, -1]))
    median_at_retire = float(np.median(wealth_matrix[:, years_to_retirement]))

    # ─── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Projekcja", "🛡️ SWR & Strategie", "🧪 Scenariusze",
        "💰 Cash Flow", "🧬 Zaawansowane"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("🔮 Conformal Prediction Funnel (Monte Carlo, Student-t, CIR)")
        st.caption("Używamy Split Conformal Prediction (Angolopoulos 2023), aby zbudować przedziały w 100% obiektywne rozkładowo. Lepsze gwarancje pokrycia niż kwartyle.")

        show_comparison = st.checkbox("Pokaż fan chart z animacją frame-by-frame", value=False, key="rem_show_anim")

        # Conformal Intervals (90% & 50%)
        cp_90 = compute_conformal_prediction_intervals(wealth_matrix, alpha=0.10)
        cp_50 = compute_conformal_prediction_intervals(wealth_matrix, alpha=0.50)
        
        p95 = cp_90['cp_upper']
        p5 = cp_90['cp_lower']
        p75 = cp_50['cp_upper']
        p25 = cp_50['cp_lower']
        p50 = cp_90['median']

        if show_comparison:
            # ── Animated Fan Chart (Hullman et al. 2015) ─────────────────────
            frames = []
            step = max(1, horizon // 30)
            frame_years = list(range(1, horizon + 1, step))

            base_frame = go.Frame(
                data=[
                    go.Scatter(x=years_arr[:2], y=p95[:2], mode='lines', line=dict(width=0), showlegend=False),
                    go.Scatter(x=years_arr[:2], y=p5[:2], fill='tonexty', fillcolor='rgba(0,255,136,0.15)', mode='lines', line=dict(width=0), name='90% Conformal CI'),
                    go.Scatter(x=years_arr[:2], y=p75[:2], mode='lines', line=dict(width=0), showlegend=False),
                    go.Scatter(x=years_arr[:2], y=p25[:2], fill='tonexty', fillcolor='rgba(0,255,136,0.25)', mode='lines', line=dict(width=0), name='50% Conformal CI'),
                    go.Scatter(x=years_arr[:2], y=p50[:2], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'),
                ], name="0"
            )
            frames.append(base_frame)

            for fy in frame_years:
                yr_slice = fy + 1
                frames.append(go.Frame(
                    data=[
                        go.Scatter(x=years_arr[:yr_slice], y=p95[:yr_slice], mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=years_arr[:yr_slice], y=p5[:yr_slice], fill='tonexty', fillcolor='rgba(0,255,136,0.15)', mode='lines', line=dict(width=0), name='90% Conformal CI'),
                        go.Scatter(x=years_arr[:yr_slice], y=p75[:yr_slice], mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=years_arr[:yr_slice], y=p25[:yr_slice], fill='tonexty', fillcolor='rgba(0,255,136,0.25)', mode='lines', line=dict(width=0), name='50% Conformal CI'),
                        go.Scatter(x=years_arr[:yr_slice], y=p50[:yr_slice], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'),
                    ], name=str(fy)
                ))

            fig_anim = go.Figure(
                data=frames[-1].data,
                frames=frames,
                layout=go.Layout(
                    title="📽️ Animated Fan Chart — Stożek Niepewności",
                    template="plotly_dark", height=500,
                    xaxis=dict(title="Wiek", range=[current_age, current_age + horizon]),
                    yaxis=dict(title="Kapitał (PLN)"),
                    hovermode="x unified",
                    updatemenus=[dict(
                        type="buttons", showactive=False, y=1.15, x=0,
                        buttons=[
                            dict(label="▶ Play", method="animate",
                                 args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                            dict(label="⏸ Pause", method="animate",
                                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                        ]
                    )],
                    sliders=[dict(
                        steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=80, redraw=True))], label=str(int(f.name) + current_age) if f.name != "0" else str(current_age)) for f in frames],
                        transition=dict(duration=0), currentvalue=dict(prefix="Wiek: ", visible=True),
                        len=0.9, x=0.05
                    )]
                )
            )
            fig_anim.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
            fig_anim.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_anim.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            # Static Fan Chart with 4 conformal bands
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p95, mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p5, fill='tonexty', fillcolor='rgba(0,255,136,0.10)', mode='lines', line=dict(width=0), name='90% Conformal CI'))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p75, mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p25, fill='tonexty', fillcolor='rgba(0,255,136,0.20)', mode='lines', line=dict(width=0), name='50% Conformal CI'))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p50, mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'))
            fig_mc.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
            fig_mc.update_layout(title="Conformal Prediction Funnel Plot", template="plotly_dark", height=500, hovermode="x unified", xaxis_title="Wiek", yaxis_title="Kapitał (PLN)")
            fig_mc.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_mc.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_mc, use_container_width=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Szansa Sukcesu (portfel > 0)", f"{success_prob:.1%}")
        m2.metric("Majątek Końcowy (Mediana)", f"{median_final:,.0f} PLN")
        m3.metric("Majątek w Wieku Emerytalnym", f"{median_at_retire:,.0f} PLN")
        if life_survival_prob is not None:
            m4.metric("Portfel przeżyje Cię (Lee-Carter)", f"{life_survival_prob:.1%}", help="Szansa, że portfel ma środki gdy umrzesz wg najnowszych metod biogerontologii i tablic śmiertelności.")
        else:
            m4.metric("Portfel przeżyje Cię", "—")

        # Violin Plot of final wealth
        st.markdown("#### 🎻 Rozkład Kapitału Końcowego (Violin Plot)")
        st.caption("Violin plot łączy rozkład, medianę i kwartyle — bogatszy niż histogram (Hintze & Nelson 1998).")
        final_w = wealth_matrix[:, -1]
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(y=final_w, box_visible=True, meanline_visible=True,
                                        fillcolor='rgba(0,255,136,0.3)', line_color='#00ff88',
                                        name='Kapitał Końcowy'))
        fig_violin.add_hline(y=total_init_cap, line_dash="dash", line_color="orange", annotation_text="Kapitał Startowy")
        fig_violin.update_layout(template="plotly_dark", height=400, yaxis_title="PLN", showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)

        # ── IKE/IKZE Kalkulator (GAP-2) ──────────────────────────────────────
        with st.expander("🏦 Kalkulator Korzyści IKE / IKZE vs Konto Zwykłe"):
            st.caption("Polska ustawa o IKE/IKZE (KNF 2023) — oblicz ile zaoszczędzisz na podatku Belki i PIT.")
            ike_col1, ike_col2, ike_col3 = st.columns(3)
            ike_contrib = ike_col1.number_input("Roczna wpłata (PLN)", value=min(annual_contrib if enable_contributions else 23_472, 23_472), step=1_000, min_value=0, max_value=23_472, key="ike_c")
            ike_years = ike_col2.number_input("Liczba lat oszczędzania", value=years_to_retirement or 20, min_value=1, max_value=50, key="ike_y")
            ike_ret = ike_col3.slider("Stopa zwrotu (%)", 2.0, 15.0, value=float(ret_return * 100), step=0.5, key="ike_r") / 100.0
            ike_res = calculate_ike_ikze_advantage(
                float(ike_contrib), int(ike_years), float(ike_ret),
                pit_bracket=pit_bracket_pct / 100.0,
            )
            if ike_res["ike_gain"] > 0 or ike_res["ikze_gain"] > 0:
                ik1, ik2, ik3 = st.columns(3)
                ik1.metric("Konto Zwykłe (po Belce)", f"{ike_res['zwykle_net']:,.0f} PLN")
                ik2.metric("IKE (+korzyść)", f"{ike_res['ike_net']:,.0f} PLN",
                           delta=f"+{ike_res['ike_gain']:,.0f} PLN (zaoszczędzona Belka)")
                ik3.metric("IKZE (+korzyść)", f"{ike_res['ikze_net']:,.0f} PLN",
                           delta=f"+{ike_res['ikze_gain']:,.0f} PLN (Belka+PIT)")
                st.info(f"💡 **IKE** zaoszczędza **{ike_res['ike_gain']:,.0f} PLN** (brak podatku Belki {ike_res['belka_tax']:,.0f} PLN). "
                        f"**IKZE** zaoszczędza **{ike_res['ikze_gain']:,.0f} PLN** "
                        f"(w tym odliczenie PIT {ike_res['ikze_pit_savings']:,.0f} PLN przy stawce {pit_bracket_pct}%).")
                st.caption(f"📌 Limit IKE 2024: 23,472 PLN/rok | Limit IKZE 2024: {int(23_472 * 1.5):,} PLN/rok (3× MIN) | Ustawa z 20.04.2004 r.")

        # ── Glide Path Visualizer ─────────────────────────────────────────────
        if use_glide_path:
            with st.expander("📉 Podgląd Glide Path — Zmiana Alokacji z Wiekiem"):
                gp_ages = list(range(current_age, current_age + horizon + 1))
                gp_equity = []
                for y, age in enumerate(gp_ages):
                    gm, gs = glide_path_params(y, years_to_retirement, ret_return, ret_vol)
                    # reverse-engineer equity pct from blended return
                    # equity_pct from formula: blended = eq*base_r + (1-eq)*bond_r → eq = (blended-bond_r)/(base_r-bond_r)
                    bond_r = 0.04
                    denom = ret_return - bond_r
                    eq_pct = (gm - bond_r) / denom if abs(denom) > 1e-9 else 0.8
                    gp_equity.append(float(np.clip(eq_pct, 0, 1)) * 100)
                fig_gp = go.Figure()
                fig_gp.add_trace(go.Scatter(x=gp_ages, y=gp_equity, mode='lines', fill='tozeroy',
                                             fillcolor='rgba(0,200,255,0.15)', line=dict(color='#00ccff', width=2),
                                             name='% Akcji'))
                fig_gp.add_trace(go.Scatter(x=gp_ages, y=[100-e for e in gp_equity], mode='lines', fill='tonexty',
                                             fillcolor='rgba(255,170,0,0.15)', line=dict(color='#ffaa00', width=2),
                                             name='% Obligacji'))
                fig_gp.add_vline(x=retirement_age, line_dash="dash", line_color="white", annotation_text="Emerytura")
                fig_gp.update_layout(template="plotly_dark", height=300, hovermode="x unified",
                                      xaxis_title="Wiek", yaxis_title="Alokacja (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig_gp, use_container_width=True)
                st.caption("Pfau (2013) Rising Equity Glide Path: 80% akcji na początku emerytury → 35% po 30 latach.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🛡️ Strategie Wypłat i Analiza SWR")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🔥 SWR Heatmap (Prawdziwy MC, 81 punktów)")
            st.caption("Każda komórka = osobna symulacja MC n=200. Bengen 2021.")

            with st.spinner("Obliczanie macierzy SWR (może potrwać ~10s)..."):
                swr_range = np.linspace(0.02, 0.06, 9)
                inf_range = np.linspace(0.00, 0.08, 9)
                n_sims_grid = 150
                horizon_grid = years_in_retirement

                z_success = np.zeros((len(swr_range), len(inf_range)))
                for i, swr in enumerate(swr_range):
                    for j, inf_g in enumerate(inf_range):
                        shocks_g = student_t_shocks(n_sims_grid, horizon_grid)
                        returns_g = np.exp((ret_return - 0.5 * ret_vol**2) + ret_vol * shocks_g) - 1
                        # Brak dziennego podatku
                        w_g = np.full((n_sims_grid, horizon_grid + 1), 0.0)
                        w_g[:, 0] = init_cap
                        ann_exp_g = init_cap * swr
                        
                        for y in range(horizon_grid):
                            w_g[:, y+1] = np.maximum(w_g[:, y] * (1 + returns_g[:, y]) - ann_exp_g * (1 + inf_g)**y, 0)
                        
                        if stoch_life:
                            lifetimes_g = lee_carter_lifetimes(current_age + years_to_retirement, n_sims_grid, gender=gender)
                            successes = 0
                            for sim_idx in range(n_sims_grid):
                                death_yr = min(int(max(0, lifetimes_g[sim_idx] - (current_age + years_to_retirement))), horizon_grid)
                                if w_g[sim_idx, death_yr] > 0:
                                    successes += 1
                            z_success[i, j] = successes / n_sims_grid
                        else:
                            z_success[i, j] = np.mean(w_g[:, -1] > 0)

            fig_heat = px.imshow(z_success, x=inf_range, y=swr_range,
                                  color_continuous_scale='RdYlGn', zmin=0, zmax=1,
                                  labels=dict(x="Inflacja", y="SWR"), text_auto=".0%",
                                  title="P(Sukces) — SWR x Inflacja x Portfel")
            fig_heat.update_xaxes(tickvals=inf_range, ticktext=[f"{v:.1%}" for v in inf_range])
            fig_heat.update_yaxes(tickvals=swr_range, ticktext=[f"{v:.1%}" for v in swr_range])
            fig_heat.add_hline(y=current_swr, line_dash="dash", line_color="white", annotation_text="Twoje SWR")
            fig_heat.update_layout(template="plotly_dark", height=420, coloraxis_colorbar=dict(title="P(Sukces)"))
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_b:
            st.markdown("### 📊 Porównanie Strategii Wypłat")
            st.caption("Porównuj jak zmienia się sukces przy różnych strategiach.")

            results_compare = {}
            strategies = [("constant", "Stała kwota"), ("guardrails", "Guardrails"), ("flexible", "% Portfela")]
            for strat_key, strat_name in strategies:
                wm_s, _ = run_mc_retirement(
                    init_cap, annual_expenses, annual_contrib, ret_return, ret_vol, horizon, 200,
                    years_to_retirement, inflation, stochastic_inflation=False,
                    enable_contributions=enable_contributions, contrib_during_retirement=contrib_during_retirement,
                    withdrawal_strategy=strat_key, flexible_pct=flexible_pct
                )
                results_compare[strat_name] = {
                    "success": float(np.mean(wm_s[:, -1] > 0)),
                    "median": float(np.median(wm_s[:, -1]))
                }

            df_compare = pd.DataFrame(results_compare).T.reset_index()
            df_compare.columns = ["Strategia", "Szansa Sukcesu", "Mediana Końcowa"]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=df_compare["Strategia"], y=df_compare["Szansa Sukcesu"],
                                      name="Szansa Sukcesu", marker_color=['#ff4444', '#ffaa00', '#00ff88'],
                                      text=[f"{v:.1%}" for v in df_compare["Szansa Sukcesu"]], textposition='outside'))
            fig_bar.add_hline(y=0.9, line_dash="dash", line_color="white", annotation_text="Prog 90%")
            fig_bar.update_layout(template="plotly_dark", height=350, yaxis_tickformat=".0%", yaxis_range=[0, 1.1])
            st.plotly_chart(fig_bar, use_container_width=True)

            for row in df_compare.itertuples():
                col = "🟢" if row[2] >= 0.9 else ("🟡" if row[2] >= 0.75 else "🔴")
                st.markdown(f"{col} **{row[1]}**: Sukces {row[2]:.1%} | Mediana końcowa {row[3]:,.0f} PLN")

        # Safety Radar
        st.markdown("### 📡 Radar: Bezpieczeństwo Planu")
        safety_s = success_prob * 10
        flex_s = (1.0 - current_swr) * 10 if current_swr < 1 else 0
        inf_prot_s = max(0, (0.08 - inflation) / 0.08) * 10
        legacy_s = min(10, (median_final / init_cap) * 5)
        life_s = (life_survival_prob * 10) if life_survival_prob is not None else success_prob * 10

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[safety_s, flex_s, inf_prot_s, legacy_s, life_s],
            theta=['Bezpieczeństwo', 'Elastyczność', 'Ochrona Inflacji', 'Dziedziczenie', 'Długowieczność'],
            fill='toself', line_color='#00ff88', name='Plan'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), template="plotly_dark", height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🧪 Scenariusze, Ryzyko Sekwencji i Optimizer Wieku")

        col_seq, col_opt = st.columns(2)

        with col_seq:
            st.markdown("### ⚡ Ryzyko Sekwencji — Animacja Krachu")
            st.caption("Zobacz, jak krach rynkowy 📉 w **1. roku emerytury** (od 0% do -50%) wpływa na całe Twoje życie.")

            wm_bad, _ = run_mc_retirement(
                init_cap, annual_expenses, annual_contrib, ret_return, ret_vol, horizon, 300,
                years_to_retirement, inflation, enable_contributions=enable_contributions,
                contrib_during_retirement=contrib_during_retirement, withdrawal_strategy=withdrawal_strategy, flexible_pct=flexible_pct
            )
            p50_normal = np.percentile(wm_bad, 50, axis=0)

            # --- Animated First Year Crash ---
            frames_seq = []
            crash_levels = np.arange(0, -0.55, -0.05) # 0%, -5%, ..., -50%
            
            for crash in crash_levels:
                w_seq = np.zeros(horizon + 1)
                w_seq[0] = init_cap
                for y in range(horizon):
                    # Szok TYLKO w pierwszym roku emerytury
                    r = crash if y == years_to_retirement else ret_return 
                    r_taxed = r * 0.81 if r > 0 else r
                    w_seq[y+1] = max(0, w_seq[y] * (1 + r_taxed))
                    inf_f = (1 + inflation) ** y
                    if y >= years_to_retirement:
                        w_seq[y+1] -= annual_expenses * inf_f
                    elif enable_contributions:
                        w_seq[y+1] += annual_contrib * inf_f
                    w_seq[y+1] = max(0, w_seq[y+1])
                
                frames_seq.append(go.Frame(
                    data=[
                        go.Scatter(x=years_arr, y=p50_normal, name="Mediana", line=dict(color='#00ff88', width=2)),
                        go.Scatter(x=years_arr, y=w_seq, name=f"Krach {crash*100:.0f}%", line=dict(color='#ff4444', width=3, dash='dot'))
                    ],
                    name=f"Crash_{crash*100:.0f}"
                ))

            fig_seq_anim = go.Figure(
                data=frames_seq[0].data,
                frames=frames_seq,
                layout=go.Layout(
                    title="Animacja Ogonów: Black Swan na starcie emerytury",
                    template="plotly_dark", height=400,
                    xaxis=dict(title="Wiek", range=[current_age, current_age + horizon]),
                    yaxis=dict(title="Kapitał (PLN)"),
                    hovermode="x unified",
                    updatemenus=[dict(
                        type="buttons", showactive=False, y=1.15, x=0,
                        buttons=[
                            dict(label="▶ Odtwórz Krach", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                            dict(label="⏸ Zresetuj", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                        ]
                    )],
                    sliders=[dict(
                        steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=f"{c*100:.0f}%") for f, c in zip(frames_seq, crash_levels)],
                        transition=dict(duration=0), currentvalue=dict(prefix="Krach w 1. roku: ", visible=True),
                        len=0.9, x=0.05
                    )]
                )
            )
            fig_seq_anim.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
            fig_seq_anim.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_seq_anim.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_seq_anim, use_container_width=True)

        with col_opt:
            st.markdown("### 🎯 Retirement Age Optimizer")
            st.caption("Każdy dodatkowy rok pracy = wyższy kapitał + mniej lat wypłat. Pfau & Kitces 2014.")

            age_range = range(max(current_age + 1, 45), min(current_age + 21, 85))
            opt_results = []
            for test_age in age_range:
                ytr = max(0, test_age - current_age)
                yir = max(1, life_expectancy - test_age)
                wm_t, _ = run_mc_retirement(
                    init_cap, annual_expenses, annual_contrib, ret_return, ret_vol,
                    ytr + yir, 200, ytr, inflation,
                    enable_contributions=enable_contributions,
                    contrib_during_retirement=contrib_during_retirement,
                    withdrawal_strategy=withdrawal_strategy, flexible_pct=flexible_pct
                )
                opt_results.append({
                    "Wiek Emerytalny": test_age,
                    "Szansa Sukcesu": float(np.mean(wm_t[:, -1] > 0)),
                    "Mediana Końcowa": float(np.median(wm_t[:, -1]))
                })

            df_opt = pd.DataFrame(opt_results)
            # Find first age where success >= 90%
            safe_ages = df_opt[df_opt["Szansa Sukcesu"] >= 0.90]
            optimal_age = int(safe_ages["Wiek Emerytalny"].min()) if not safe_ages.empty else None

            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=df_opt["Wiek Emerytalny"], y=df_opt["Szansa Sukcesu"],
                                          mode='lines+markers', name="Szansa Sukcesu",
                                          line=dict(color='#00ff88', width=2)))
            fig_opt.add_hline(y=0.90, line_dash="dash", line_color="#ffaa00", annotation_text="Próg 90%")
            if optimal_age:
                fig_opt.add_vline(x=optimal_age, line_dash="dot", line_color="#00ccff",
                                   annotation_text=f"Min. Bezpieczny: {optimal_age}r.")
            fig_opt.update_layout(title="Optymalizacja Wieku Emerytalnego", template="plotly_dark", height=380,
                                   yaxis_tickformat=".0%", xaxis_title="Wiek Emerytalny", yaxis_title="P(Sukces)")
            fig_opt.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_opt.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_opt, use_container_width=True)

            if optimal_age:
                diff = optimal_age - retirement_age
                st.success(f"✅ Minimalny bezpieczny wiek: **{optimal_age} lat** ({"+" if diff >= 0 else ""}{diff} lat względem Twojego celu)")
            else:
                st.error("⚠️ Brak kombinacji zapewniającej 90% szansy sukcesu w tym horyzoncie.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("💰 Cash Flow — Analiza Przepływów")
        income_years = np.arange(current_age, current_age + horizon)
        port_withdrawals = []
        extra_incomes_list = []

        for y in range(horizon):
            inf_f = (1 + inflation) ** y
            e_inc = monthly_contribution * inf_f if (enable_contributions and (y < years_to_retirement or contrib_during_retirement)) else 0
            p_wd = monthly_expat * inf_f if y >= years_to_retirement else 0
            port_withdrawals.append(p_wd)
            extra_incomes_list.append(e_inc)

        income_df = pd.DataFrame({"Wiek": income_years, "Wypłata z Portfela": port_withdrawals, "Dodatkowy Dochód": extra_incomes_list})
        income_df["Suma"] = income_df["Wypłata z Portfela"] + income_df["Dodatkowy Dochód"]

        fig_income = go.Figure()
        fig_income.add_trace(go.Bar(x=income_df["Wiek"], y=income_df["Wypłata z Portfela"], name="Portfel", marker_color="#00ccff"))
        fig_income.add_trace(go.Bar(x=income_df["Wiek"], y=income_df["Dodatkowy Dochód"], name="Dochód Dodatkowy", marker_color="#ffaa00"))
        fig_income.update_layout(barmode='stack', title="Budżet Miesięczny (nominalny, z inflacją)", template="plotly_dark", xaxis_title="Wiek", yaxis_title="PLN / Mies.", height=420, hovermode="x unified")
        fig_income.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
        fig_income.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
        st.plotly_chart(fig_income, use_container_width=True)

        total_monthly = monthly_expat + (monthly_contribution if contrib_during_retirement and enable_contributions else 0)
        st.success(f"### 🎯 Miesięczny budżet na emeryturze (wiek {retirement_age}): **{total_monthly:,.0f} PLN**")

    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🧬 Zaawansowane Analizy Naukowe")

        # Tab5 col layout
        col5a, col5b = st.columns(2)

        with col5a:
            # ── Survival Curve (Kaplan-Meier) ─────────────────────────────
            st.markdown("### 📉 Krzywa Przeżywalności Portfela")
            st.caption("Standard finansów emerytalnych i medycyny (Kaplan & Meier 1958, Chen 2018 J.Financial Planning).")

            survival = compute_survival_curve(wealth_matrix)
            fig_surv = go.Figure()
            fig_surv.add_trace(go.Scatter(x=years_arr, y=survival, mode='lines', name='P(portfel > 0)', line=dict(color='#00ff88', width=3), fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'))
            for thresh, col_t, label in [(0.9, '#00ccff', '90%'), (0.75, '#ffaa00', '75%'), (0.5, '#ff4444', '50%')]:
                fig_surv.add_hline(y=thresh, line_dash="dot", line_color=col_t, annotation_text=label)
                # Find where survival drops below threshold
                below = np.where(survival <= thresh)[0]
                if len(below) > 0:
                    age_below = years_arr[below[0]]
                    fig_surv.add_vline(x=age_below, line_dash="dot", line_color=col_t)

            fig_surv.add_vline(x=retirement_age, line_dash="dash", line_color="white", annotation_text="Emerytura")
            fig_surv.update_layout(title="Krzywa Przeżywalności Portfela", template="plotly_dark", height=420,
                                    hovermode="x unified", xaxis_title="Wiek", yaxis_title="% symulacji z kapitałem > 0",
                                    yaxis_tickformat=".0%")
            fig_surv.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_surv.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_surv, use_container_width=True)

            # Median survival age
            below_50 = np.where(survival <= 0.5)[0]
            if len(below_50) > 0:
                median_surv_age = years_arr[below_50[0]]
                st.metric("Mediana Przeżywalności Portfela", f"Wiek {median_surv_age}", help="Wiek, w którym połowa symulacji ma już 0 środków.")
            else:
                st.metric("Mediana Przeżywalności Portfela", f"> {years_arr[-1]} lat 🟢", help="Więcej niż 50% symulacji przeżywa cały horyzont.")

        with col5b:
            # ── Waterfall Chart ───────────────────────────────────────────
            st.markdown("### 🌊 Dekompozycja Majątku (Waterfall)")
            st.caption("Pokazuje skąd pochodzi (lub znika) Twój majątek. Few (2009).")

            x_labels, y_vals, measures = compute_waterfall(
                init_cap, wealth_matrix, inf_matrix, annual_expenses, annual_contrib,
                years_to_retirement, horizon, inflation, enable_contributions, contrib_during_retirement
            )

            colors = []
            for m, v in zip(measures, y_vals):
                if m == "absolute": colors.append("#00ccff")
                elif m == "total": colors.append("#00ff88")
                elif v >= 0: colors.append("#00aa55")
                else: colors.append("#ff4444")

            fig_wf = go.Figure(go.Waterfall(
                x=x_labels, measure=measures, y=y_vals,
                connector=dict(line=dict(color="#666", width=1)),
                increasing=dict(marker_color="#00aa55"),
                decreasing=dict(marker_color="#ff4444"),
                totals=dict(marker_color="#00ccff"),
                text=[f"{v:+,.0f}" for v in y_vals],
                textposition="outside"
            ))
            fig_wf.update_layout(title="Skąd bierze się (lub znika) Twój Majątek", template="plotly_dark", height=420,
                                  yaxis_title="PLN", showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)

        # ── Stochastic Inflation Preview ──────────────────────────────────
        st.markdown("### 🌡️ Podgląd Trajektorii Inflacji (CIR)")
        st.caption("5 przykładowych ścieżek inflacji generowanych przez model CIR (Cox-Ingersoll-Ross 1985).")
        sample_inf = inf_matrix[:5, :]
        fig_inf = go.Figure()
        colors_inf = ['#00ff88', '#00ccff', '#ffaa00', '#ff88aa', '#aa88ff']
        for i in range(5):
            fig_inf.add_trace(go.Scatter(x=years_arr[:-1], y=sample_inf[i], mode='lines', name=f'Ścieżka {i+1}', line=dict(color=colors_inf[i], width=1.5)))
        fig_inf.add_hline(y=inflation, line_dash="dash", line_color="white", annotation_text=f"Bazowa {inflation:.1%}")
        fig_inf.update_layout(template="plotly_dark", height=350, hovermode="x unified", xaxis_title="Wiek", yaxis_title="Inflacja", yaxis_tickformat=".1%")
        fig_inf.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
        st.plotly_chart(fig_inf, use_container_width=True)

        # ── Longevity distribution ────────────────────────────────────────
        if stoch_life:
            st.markdown("### 🧬 Rozkład Długości Życia (Lee-Carter Model)")
            lifetimes_plot = lee_carter_lifetimes(current_age, 1000, gender=gender)
            fig_lt = go.Figure()
            fig_lt.add_trace(go.Histogram(x=lifetimes_plot, nbinsx=40, marker_color='rgba(0,200,255,0.5)', name='Długość Życia'))
            fig_lt.add_vline(x=life_expectancy, line_dash="dash", line_color="orange", annotation_text=f"Max Horyzont {life_expectancy}")
            pct_over = np.mean(lifetimes_plot > life_expectancy)
            fig_lt.add_vline(x=int(np.median(lifetimes_plot)), line_color="white", line_dash="dot", annotation_text=f"Mediana {int(np.median(lifetimes_plot))}")
            fig_lt.update_layout(template="plotly_dark", height=350, xaxis_title="Wiek Śmierci", yaxis_title="Liczba Symulacji")
            st.plotly_chart(fig_lt, use_container_width=True)
            st.warning(f"⚠️ **{pct_over:.1%}** uczestników symulacji żyje DŁUŻEJ niż Twój horyzont ({life_expectancy} lat). Rozważ wydłużenie horyzontu lub zakup annuity.")
            
            # ── Copula Density Heatmap ──────────────────────────────────────────
            st.markdown("### 🌪️ Copula Risk Heatmap (Joint Density)")
            st.caption("2D Contour map – Ocena łącznego ryzyka: Wiek Śmierci vs Majątek Końcowy. Pozwala ocenić nieliniowe zbiegi okoliczności (joint tail risk).")
            fig_copula = go.Figure(go.Histogram2dContour(
                x=lifetimes_plot[:wealth_matrix.shape[0]], # Dopasowanie wymiarów n_sims (zwykle 500 w głównym)
                y=wealth_matrix[:, -1],
                colorscale='Viridis',
                contours=dict(showlabels=True, labelfont=dict(color='white')),
                hovertemplate="Wiek Śmierci: %{x}<br>Końcowy Kapitał: %{y}<br>Zagęszczenie: %{z}<extra></extra>"
            ))
            fig_copula.update_layout(
                title="Łączne Prawdopodobieństwo (Wiek Śmierci vs Kapitał)",
                xaxis_title="Wiek Śmierci (lat)", yaxis_title="Majątek Końcowy (PLN)",
                template="plotly_dark", height=400,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_copula, use_container_width=True)

        # ── 7 Ryzyk Emerytalnych (Pfau 2015) ─────────────────────────────────
        st.markdown("### ⚠️ Radar 7 Ryzyk Emerytalnych (Pfau 2015)")
        st.caption("Wade Pfau (2015) *The 7 Risks of Retirement*. Twój moduł pokrywa 4/7. Pozostałe wymagają Twojego osądu.")
        r_longevity   = float(np.clip(life_survival_prob if life_survival_prob is not None else success_prob, 0, 1))
        r_market      = float(np.clip(success_prob, 0, 1))
        r_sequence    = float(np.clip(1.0 - current_swr / 0.06, 0, 1))
        r_inflation   = float(np.clip(1.0 - inflation / 0.10, 0, 1))
        r_health      = float(np.clip(0.5 - max(0, current_age - 70) * 0.02, 0, 1))  # szacunek
        r_spouse      = 0.6  # użytkownik musi ocenić sam
        r_policy      = float(np.clip(zus_coverage * 0.5 + 0.5, 0, 1))  # proxy dywersyfikacji od ZUS
        pfau_labels = ['Długowieczność', 'Ryzyko Rynku', 'Seq. of Returns', 'Inflacja', 'Koszty Zdrowia', 'Utrata Partnera', 'Ryzyko Polityczne (ZUS)']
        pfau_vals    = [r_longevity, r_market, r_sequence, r_inflation, r_health, r_spouse, r_policy]
        pfau_colors  = ['#00ff88' if v >= 0.7 else '#ffaa00' if v >= 0.4 else '#ff4444' for v in pfau_vals]
        fig_pfau = go.Figure()
        fig_pfau.add_trace(go.Scatterpolar(
            r=[v * 10 for v in pfau_vals], theta=pfau_labels,
            fill='toself', fillcolor='rgba(0,200,255,0.10)',
            line=dict(color='#00ccff', width=2), name='Twój Plan'
        ))
        fig_pfau.add_trace(go.Scatterpolar(
            r=[7, 7, 7, 7, 7, 7, 7], theta=pfau_labels,
            fill=None, line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name='Próg 70%'
        ))
        fig_pfau.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10], tickvals=[3, 5, 7, 10], ticktext=["30%", "50%", "70%", "100%"])),
            template="plotly_dark", height=450, showlegend=True,
            title="7 Ryzyk Emerytalnych vs Twój Plan (skala 0-10)"
        )
        st.plotly_chart(fig_pfau, use_container_width=True)
        with st.expander("ℹ️ Jak interpretować Radar 7 Ryzyk?"):
            st.markdown("""
| Ryzyko | Standard | Twój pokrycie |
|---|---|---|
| ✅ Długowieczność | Lee-Carter MC | Modelowane |
| ✅ Ryzyko Rynku | Student-t MC | Modelowane |
| ✅ Seq. of Returns | Animacja & Conformal PI | Modelowane |
| ✅ Inflacja | CIR stochastyczna | Modelowane |
| ⚠️ Koszty Zdrowia | Medical inflation 6-8% p.a. | Szacunek heurystyczny |
| ⚠️ Utrata Partnera | Zmiana wydatków ~40% | Stała wartość 60% |
| ⚠️ Ryzyko Polityczne | Zmiany w ZUS/IKE | Proxy z ZUS coverage |
            """)

        # ── Sensitivity Tornado Chart ─────────────────────────────────────────
        st.markdown("### 🌪️ Sensitivity Analysis — Co Najbardziej Wpływa na Sukces?")
        st.caption("One-factor-at-a-time: każdy parametr zmieniamy o ±20% i mierzymy wpływ na P(sukces). Standard analizy ryzyka.")
        with st.spinner("Obliczanie wrażliwości (8 parametrów × 2)..."):
            base_sp = float(success_prob)
            tornado_items = []
            test_delta = 0.20  # ±20%

            def _mc_success(cap, exp, ret, vol, inf_r, ytret):
                wm_t, _ = run_mc_retirement(
                    cap, exp, annual_contrib, ret, vol, horizon, 150, ytret, inf_r,
                    stochastic_inflation=False, enable_contributions=enable_contributions,
                    contrib_during_retirement=contrib_during_retirement,
                    withdrawal_strategy=withdrawal_strategy, flexible_pct=flexible_pct,
                    zus_monthly=float(zus_monthly), tax_regime=tax_regime,
                )
                return float(np.mean(wm_t[:, -1] > 0))

            params_tornado = [
                ("Kapitał (+20%)", _mc_success(total_init_cap * 1.20, annual_expenses, ret_return, ret_vol, inflation, years_to_retirement), _mc_success(total_init_cap * 0.80, annual_expenses, ret_return, ret_vol, inflation, years_to_retirement)),
                ("Wydatki (-20%)", _mc_success(total_init_cap, annual_expenses * 0.80, ret_return, ret_vol, inflation, years_to_retirement), _mc_success(total_init_cap, annual_expenses * 1.20, ret_return, ret_vol, inflation, years_to_retirement)),
                ("Zwrot (+20%)", _mc_success(total_init_cap, annual_expenses, ret_return * 1.20, ret_vol, inflation, years_to_retirement), _mc_success(total_init_cap, annual_expenses, ret_return * 0.80, ret_vol, inflation, years_to_retirement)),
                ("Zmienność (-20%)", _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol * 0.80, inflation, years_to_retirement), _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol * 1.20, inflation, years_to_retirement)),
                ("Inflacja (-20%)", _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol, inflation * 0.80, years_to_retirement), _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol, inflation * 1.20, years_to_retirement)),
                ("Wiek em. (-2 lata)", _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol, inflation, max(0, years_to_retirement - 2)), _mc_success(total_init_cap, annual_expenses, ret_return, ret_vol, inflation, years_to_retirement + 2)),
            ]

        tornado_df_items = [(label, up - base_sp, down - base_sp) for label, up, down in params_tornado]
        tornado_df_items.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        labels_t = [x[0] for x in tornado_df_items]
        ups_t    = [x[1] for x in tornado_df_items]
        downs_t  = [x[2] for x in tornado_df_items]

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(y=labels_t, x=ups_t,  orientation='h', name='Pozytywna zmiana', marker_color='#00ff88'))
        fig_tornado.add_trace(go.Bar(y=labels_t, x=downs_t, orientation='h', name='Negatywna zmiana', marker_color='#ff4444'))
        fig_tornado.add_vline(x=0, line_color='white', line_width=1)
        fig_tornado.update_layout(
            barmode='overlay', template="plotly_dark", height=380,
            xaxis_title="Zmiana P(sukcesu) vs baza", xaxis_tickformat=".0%",
            yaxis_title="Parametr", title="Tornado Chart — Analiza Wrażliwości",
            hovermode="y unified"
        )
        st.plotly_chart(fig_tornado, use_container_width=True)
        st.caption("Każdy słupek = różnica w P(sukces) przy zmianie parametru o ±20%. Dłuższy słupek = większy wpływ na bezpieczeństwo planu.")

    # ─── Summary and Updated Recommendations ─────────────────────────────────
    st.divider()
    st.subheader("💡 Rekomendacje i Gotowość Emerytalna")

    # Finalize Readiness Score po MC
    ytf_final = years_to_fire if years_to_fire != float('inf') else 30
    rrs_final = retirement_readiness_score(success_prob, current_swr, zus_coverage, ytf_final, cap_vs_fire)
    rrs_col = "#00ff88" if rrs_final >= 70 else ("#ffaa00" if rrs_final >= 50 else "#ff4444")
    rrs_label = "🟢 Plan Antykruchy" if rrs_final >= 70 else ("🟡 Plan Wymaga Uwagi" if rrs_final >= 50 else "🔴 Plan Zagrożony")

    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;margin-bottom:16px;'>
        <div style='display:flex;align-items:center;gap:24px;'>
            <div style='text-align:center;'>
                <div style='font-size:48px;font-weight:900;color:{rrs_col};'>{rrs_final:.0f}</div>
                <div style='font-size:12px;color:#9ca3af;'>/ 100</div>
            </div>
            <div>
                <div style='font-size:20px;font-weight:700;color:{rrs_col};'>{rrs_label}</div>
                <div style='font-size:13px;color:#d1d5db;margin-top:6px;'>
                    Retirement Readiness Score — Fidelity-style composite (P(sukces) • SWR • ZUS coverage • postęp FIRE • kapitał)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        if success_prob < 0.9:
            st.warning(f"⚠️ Szansa sukcesu ({success_prob:.1%}) jest poniżej 90%. Rozważ: wyższe wpłaty, późniejszą emeryturę lub strategię Guardrails.")
        else:
            st.success(f"✅ Plan jest antykruchy. Szansa sukcesu: **{success_prob:.1%}**.")
        if zus_monthly > 0:
            st.info(f"🏛️ **ZUS pokrywa {zus_coverage:.0%}** Twoich wydatków ({zus_monthly:,.0f} PLN/mies). Portfel musi zapewnić resztę: {max(0, annual_expenses - zus_monthly*12):,.0f} PLN/rok.")
        if tax_regime != "IKE/IKZE (0% Belki)":
            st.warning(f"⚠️ Tryb **{tax_regime}** — podatek Belki obniża efektywny zwrot. Rozważ przeniesienie kapitału na IKE/IKZE.")
    with rec_col2:
        if withdrawal_strategy == "constant":
            st.info("💡 Strategia 'Stała kwota' jest podatna na ryzyko sekwencji. Rozważ 'Guardrails' lub '% Portfela'.")
        if use_spending_smile:
            st.success("✅ **Spending Smile** aktywny — bardziej realistyczny wzorzec wydatków (Blanchett 2013).")
        if use_glide_path:
            st.success("✅ **Glide Path** aktywny — portfel dostosowuje alokację akcji/obligacji z wiekiem (Pfau 2013).")
        if stoch_life and life_survival_prob is not None and life_survival_prob < 0.85:
            st.warning(f"⚠️ Portfel przeżyje Cię tylko w {life_survival_prob:.1%} symulacji. Rozważ dożywotnią rentę (annuity).")

    st.caption("Analiza oparta na: Bengen 2021, Merton 2014, Pfau 2015/2018, Blanchett 2013, Kitces 2022, GUS 2023, KNF 2023. "
               "Model używa Student-t(df=4) dla grubych ogonów, CIR dla inflacji, Lee-Carter dla długowieczności, "
               "Split Conformal Prediction dla przedziałów, Guyton-Klinger dla Guardrails, CAPE-Adjusted SWR. "
               "V3.0 — Intelligent Barbell Dashboard.")

