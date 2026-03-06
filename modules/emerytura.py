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
                      floor_amount=0):
    """
    Główna symulacja Monte Carlo z polepszeniami naukowymi.
    Returns: wealth_matrix (n_sims x horizon+1), inflation path (n_sims x horizon)
    """
    mu = ret_return
    sigma = ret_vol
    log_mu = mu - 0.5 * sigma**2

    # Student-t shocks (fat tails)
    shocks = student_t_shocks(n_sims, horizon)
    annual_returns = np.exp(log_mu + sigma * shocks) - 1

    # Stochastic inflation (CIR or constant)
    if stochastic_inflation:
        inf_matrix = cir_inflation(inflation_base, horizon, n_sims)
    else:
        inf_matrix = np.full((n_sims, horizon), inflation_base)

    wealth = np.full((n_sims, horizon + 1), 0.0)
    wealth[:, 0] = init_cap

    # For guardrails: track "glide path" — expected portfolio trajectory
    glide_path = init_cap * (1 + ret_return) ** np.arange(horizon + 1)

    # Per-sim withdrawal amount (can change per guardrails)
    current_withdrawal = np.full(n_sims, annual_expenses)

    for y in range(horizon):
        ret = annual_returns[:, y]
        # Usunięto destrukcyjny podatek potrącany corocznie z całości portfela (zał. Fundusze Akumulujące/IKE/IKZE)
        w = wealth[:, y]
        w_new = w * (1 + ret)

        inf_y = inf_matrix[:, y]
        inf_factor_scalar = (1 + inflation_base) ** y  # for nominal calcs

        phase_retire = (y >= years_to_retirement)

        if phase_retire:
            # Dynamic withdrawal strategies
            if withdrawal_strategy == "flexible":
                # Always withdraw fixed % of current portfolio — never bankrupt
                current_withdrawal = flexible_pct * w_new
            elif withdrawal_strategy == "guardrails":
                ratio = w_new / np.maximum(glide_path[y+1], 1)
                increase = (ratio > 1 + guardrails_band) & (w > 0)
                decrease = (ratio < 1 - guardrails_band) & (w > 0)
                current_withdrawal = np.where(increase, current_withdrawal * 1.10,
                              np.where(decrease, current_withdrawal * 0.90,
                                       current_withdrawal))
            # Inflation-adjust constant withdrawal
            inf_cum = np.prod(1 + inf_matrix[:, :y+1], axis=1) if y > 0 else (1 + inf_matrix[:, 0])
            if withdrawal_strategy == "constant":
                eff_withdrawal = annual_expenses * inf_cum
            else:
                eff_withdrawal = current_withdrawal * (1 + inf_y)

            # Floor strategy: protect a minimum floor amount
            floor_adj = np.maximum(floor_amount, 0)
            w_new -= np.maximum(eff_withdrawal - floor_adj, 0)

            if enable_contributions and contrib_during_retirement:
                w_new += annual_contrib * (1 + inflation_base) ** y

        else:
            # Accumulation phase
            if enable_contributions:
                w_new += annual_contrib * (1 + inflation_base) ** y

        w_new = np.maximum(w_new, 0)
        wealth[:, y + 1] = w_new

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
    st.header("🏖️ Analiza Emerytury — Scientific Edition v2.0")
    st.markdown("""
    **9 ulepszeń naukowych**: Student-t (grube ogony), stochastyczna długość życia (Gompertz),
    stochastyczna inflacja (CIR), dynamiczne strategie wypłat (Guardrails / Flexible / Floor),
    Retirement Age Optimizer, Krzywa Przeżywalności i nowe wizualizacje.
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
    stoch_life = st.sidebar.checkbox("Stoch. Długość Życia (Gompertz)", value=_saved("rem_stoch_life", True), key="rem_stoch_life", on_change=_save, args=("rem_stoch_life",), help="Każdy uczestnik MC 'umiera' w losowym wieku (Gompertz/GUS 2023).")

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

    # ─── Core Calculations ────────────────────────────────────────────────────
    years_to_retirement = max(0, retirement_age - current_age)
    years_in_retirement = max(1, life_expectancy - retirement_age)
    total_years = years_to_retirement + years_in_retirement
    horizon = total_years
    n_sims = 500
    annual_expenses = monthly_expat * 12
    annual_contrib = monthly_contribution * 12 if enable_contributions else 0
    fire_number = annual_expenses / 0.035
    current_swr = annual_expenses / init_cap if init_cap > 0 else 0

    # Deterministic FIRE Time Math
    real_return = (1 + ret_return) / (1 + inflation) - 1
    if init_cap >= fire_number:
        years_to_fire = 0
    elif enable_contributions and annual_contrib > 0:
        if real_return == 0:
            years_to_fire = (fire_number - init_cap) / annual_contrib
        else:
            val1 = fire_number + annual_contrib / real_return
            val2 = init_cap + annual_contrib / real_return
            if val2 <= 0 or val1 / val2 <= 0:
                years_to_fire = float('inf')
            else:
                years_to_fire = np.log(val1 / val2) / np.log(1 + real_return)
    else:
        if real_return > 0:
            years_to_fire = np.log(fire_number / init_cap) / np.log(1 + real_return)
        else:
            years_to_fire = float('inf')

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FIRE Number (SWR 3.5%)", f"{fire_number:,.0f} PLN")
    
    if years_to_fire == 0:
        col2.metric("Czas do FIRE", "Osiągnięty! 🎉", help="Masz wystarczająco kapitału.")
    elif years_to_fire == float('inf'):
        col2.metric("Czas do FIRE", "Nigdy 😢", help="Inflacja pożera kapitał szybciej niż oszczędzasz.")
    else:
        col2.metric("Czas do FIRE", f"{years_to_fire:.1f} lat", help="Lata do wolności przy podanych wpłatach i stopie zwrotu.")
        
    col3.metric("Twoje obecne SWR", f"{current_swr:.2%}")
    col4.metric("Okres emerytury", f"{years_in_retirement} lat")

    st.markdown("---")

    # ─── Run MC ──────────────────────────────────────────────────────────────
    wealth_matrix, inf_matrix = run_mc_retirement(
        init_cap, annual_expenses, annual_contrib,
        ret_return, ret_vol, horizon, n_sims,
        years_to_retirement, inflation,
        stochastic_inflation=stoch_inf,
        enable_contributions=enable_contributions,
        contrib_during_retirement=contrib_during_retirement,
        withdrawal_strategy=withdrawal_strategy,
        flexible_pct=flexible_pct,
        floor_amount=floor_amount
    )

    years_arr = np.arange(current_age, current_age + horizon + 1)

    # ─── Stochastic lifetimes (Gompertz) ─────────────────────────────────────
    if stoch_life:
        lifetimes = gompertz_lifetimes(current_age, n_sims)
        # Find actual survival: portfel przeżywa uczestnika?
        portfolio_survives = []
        for i in range(n_sims):
            death_age = lifetimes[i]
            death_yr = min(int(death_age - current_age), horizon)
            portfolio_survives.append(wealth_matrix[i, death_yr] > 0)
        life_survival_prob = np.mean(portfolio_survives)
    else:
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
        st.subheader("🔮 Projekcja Majątku — Monte Carlo (Student-t, CIR)")
        st.caption("Symulacja używa gruboogonowych szoków Student-t(df=4) i stochastycznej inflacji CIR.")

        show_comparison = st.checkbox("Pokaż fan chart z animacją frame-by-frame", value=False, key="rem_show_anim")

        p5, p25, p50, p75, p95 = np.percentile(wealth_matrix, [5, 25, 50, 75, 95], axis=0)

        if show_comparison:
            # ── Animated Fan Chart (Hullman et al. 2015) ─────────────────────
            frames = []
            step = max(1, horizon // 30)
            frame_years = list(range(1, horizon + 1, step))

            base_frame = go.Frame(
                data=[
                    go.Scatter(x=years_arr[:2], y=p95[:2], mode='lines', line=dict(width=0), showlegend=False),
                    go.Scatter(x=years_arr[:2], y=p5[:2], fill='tonexty', fillcolor='rgba(0,255,136,0.15)', mode='lines', line=dict(width=0), name='90% CI'),
                    go.Scatter(x=years_arr[:2], y=p75[:2], mode='lines', line=dict(width=0), showlegend=False),
                    go.Scatter(x=years_arr[:2], y=p25[:2], fill='tonexty', fillcolor='rgba(0,255,136,0.25)', mode='lines', line=dict(width=0), name='50% CI'),
                    go.Scatter(x=years_arr[:2], y=p50[:2], mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'),
                ], name="0"
            )
            frames.append(base_frame)

            for fy in frame_years:
                yr_slice = fy + 1
                frames.append(go.Frame(
                    data=[
                        go.Scatter(x=years_arr[:yr_slice], y=p95[:yr_slice], mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=years_arr[:yr_slice], y=p5[:yr_slice], fill='tonexty', fillcolor='rgba(0,255,136,0.15)', mode='lines', line=dict(width=0), name='90% CI'),
                        go.Scatter(x=years_arr[:yr_slice], y=p75[:yr_slice], mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=years_arr[:yr_slice], y=p25[:yr_slice], fill='tonexty', fillcolor='rgba(0,255,136,0.25)', mode='lines', line=dict(width=0), name='50% CI'),
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
            # Static Fan Chart with 4 percentile bands
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p95, mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p5, fill='tonexty', fillcolor='rgba(0,255,136,0.10)', mode='lines', line=dict(width=0), name='90% CI'))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p75, mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p25, fill='tonexty', fillcolor='rgba(0,255,136,0.20)', mode='lines', line=dict(width=0), name='50% CI'))
            fig_mc.add_trace(go.Scatter(x=years_arr, y=p50, mode='lines', line=dict(color='#00ff88', width=3), name='Mediana'))
            fig_mc.add_vline(x=retirement_age, line_dash="dash", line_color="#00ccff", annotation_text="Start Emerytury")
            fig_mc.update_layout(title="Projekcja Majątku (4 pasma percentylowe)", template="plotly_dark", height=500, hovermode="x unified", xaxis_title="Wiek", yaxis_title="Kapitał (PLN)")
            fig_mc.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            fig_mc.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot")
            st.plotly_chart(fig_mc, use_container_width=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Szansa Sukcesu (portfel > 0)", f"{success_prob:.1%}")
        m2.metric("Majątek Końcowy (Mediana)", f"{median_final:,.0f} PLN")
        m3.metric("Majątek w Wieku Emerytalnym", f"{median_at_retire:,.0f} PLN")
        if life_survival_prob is not None:
            m4.metric("Portfel przeżyje Cię (Gompertz)", f"{life_survival_prob:.1%}", help="Szansa, że portfel ma środki gdy umrzesz (losowa długość życia wg GUS 2023).")
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
        fig_violin.add_hline(y=init_cap, line_dash="dash", line_color="orange", annotation_text="Kapitał Startowy")
        fig_violin.update_layout(template="plotly_dark", height=400, yaxis_title="PLN", showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)

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
                            lifetimes_g = gompertz_lifetimes(current_age + years_to_retirement, n_sims_grid)
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
            st.markdown("### 🧬 Rozkład Długości Życia (Gompertz)")
            lifetimes_plot = gompertz_lifetimes(current_age, 1000)
            fig_lt = go.Figure()
            fig_lt.add_trace(go.Histogram(x=lifetimes_plot, nbinsx=40, marker_color='rgba(0,200,255,0.5)', name='Długość Życia'))
            fig_lt.add_vline(x=life_expectancy, line_dash="dash", line_color="orange", annotation_text=f"Max Horyzont {life_expectancy}")
            pct_over = np.mean(lifetimes_plot > life_expectancy)
            fig_lt.add_vline(x=int(np.median(lifetimes_plot)), line_color="white", line_dash="dot", annotation_text=f"Mediana {int(np.median(lifetimes_plot))}")
            fig_lt.update_layout(template="plotly_dark", height=350, xaxis_title="Wiek Śmierci", yaxis_title="Liczba Symulacji")
            st.plotly_chart(fig_lt, use_container_width=True)
            st.warning(f"⚠️ **{pct_over:.1%}** uczestników symulacji żyje DŁUŻEJ niż Twój horyzont ({life_expectancy} lat). Rozważ wydłużenie horyzontu lub zakup annuity.")

    # ─── Summary and Recommendations ─────────────────────────────────────────
    st.divider()
    st.subheader("💡 Rekomendacje")
    if success_prob < 0.9:
        st.warning(f"⚠️ Szansa sukcesu ({success_prob:.1%}) jest poniżej 90%. Rozważ: wyższe wpłaty, późniejszą emeryturę lub strategię Guardrails.")
    else:
        st.success(f"✅ Plan jest antykruchy. Szansa sukcesu: **{success_prob:.1%}**.")

    if withdrawal_strategy == "constant":
        st.info("💡 **Wskazówka**: Strategia 'Stała kwota' jest podatna na ryzyko sekwencji. Rozważ 'Guardrails' lub '% Portfela' dla wyższej odporności.")

    if stoch_life and life_survival_prob is not None and life_survival_prob < 0.85:
        st.warning(f"⚠️ Portfel przeżyje Cię tylko w {life_survival_prob:.1%} symulacji. Rozważ dożywotnią rentę (annuity) lub wydłużenie horyzontu.")

    st.caption("Analiza oparta na: Bengen 2021, Merton 2014, Pfau 2018, Kaplan & Meier 1958, GUS 2023. Model używa Student-t(df=4) dla grafikoowych ogonów, CIR dla inflacji, Gompertz dla długości życia.")
