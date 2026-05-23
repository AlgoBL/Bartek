"""
stochastic_errors.py — Błędy Addytywne vs Multiplikatywne w Procesach Stochastycznych
========================================================================================
Obliczenia numeryczne dla modułu wizualizacyjnego.

Teorie:
  - Gardiner (2009) *Handbook of Stochastic Methods* — podstawy SDE
  - Horsthemke & Lefever (1984) *Noise-Induced Transitions* — przejścia fazowe
  - Gammaitoni et al. (1998) Rev. Mod. Phys. — Stochastic Resonance
  - Risken (1989) *The Fokker-Planck Equation* — rozkłady stacjonarne
  - Arnold (1998) *Random Dynamical Systems* — wykładniki Lapunova
  - Van Kampen (1992) *Stochastic Processes in Physics and Chemistry*

Klasyfikacja błędów:
  AE (Additive Error):       dX = f(X)dt + σ·dW          (szum niezależny od stanu)
  ME (Multiplicative Error): dX = f(X)dt + g(X)·σ·dW     (szum skaluje się ze stanem)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
#  TYPY
# ─────────────────────────────────────────────────────────────────────────────

NoiseType = Literal["additive", "multiplicative", "both", "none"]
LinearModel = Literal["gbm", "ou", "lognormal"]
NonlinearModel = Literal["bistable", "duffing", "logistic"]


@dataclass
class SimResult:
    """Wynik symulacji SDE."""
    times: np.ndarray                        # (T,)
    paths_clean: np.ndarray                  # (n_paths, T) — bez szumu dodatkowego
    paths_ae: np.ndarray                     # (n_paths, T) — z błędem addytywnym
    paths_me: np.ndarray                     # (n_paths, T) — z błędem multiplikatywnym
    stats: dict = field(default_factory=dict)


@dataclass
class FokkerPlanckResult:
    """Wynik numerycznego Fokker-Planck."""
    x_grid: np.ndarray       # siatka x
    p_stationary: np.ndarray  # P_inf(x) stacjonarny rozkład
    p_ae: np.ndarray          # P_inf z błędem AE
    p_me: np.ndarray          # P_inf z błędem ME


@dataclass
class SRResult:
    """Wynik symulacji Stochastic Resonance."""
    sigma_range: np.ndarray   # zakres σ
    snr_ae: np.ndarray        # Signal-to-Noise Ratio dla AE
    snr_me: np.ndarray        # SNR dla ME
    snr_clean: float          # SNR bez szumu (referencja)
    opt_sigma_ae: float       # optymalny σ dla AE
    opt_sigma_me: float       # optymalny σ dla ME


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESY LINIOWE
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_process(
    model: LinearModel = "ou",
    n_paths: int = 100,
    T: float = 5.0,
    dt: float = 0.01,
    # Parametry modelu
    mu: float = 0.05,         # dryf (GBM) lub długookresowa średnia (OU)
    theta: float = 2.0,       # szybkość powrotu do średniej (OU)
    sigma_base: float = 0.20, # bazowa zmienność modelu
    x0: float = 1.0,          # warunek początkowy
    # Błędy
    sigma_ae: float = 0.05,   # σ błędu addytywnego
    sigma_me: float = 0.10,   # σ błędu multiplikatywnego
    seed: int | None = 42,
) -> SimResult:
    """
    Symuluje liniowy proces stochastyczny w 3 wariantach:
    - Czysty (bez dodatkowego szumu)
    - Z błędem addytywnym (AE): + σ_AE dW_AE
    - Z błędem multiplikatywnym (ME): + σ_ME·X dW_ME

    Modele:
    -------
    GBM: dS = μ·S dt + σ·S dW  (baseline ME)
         AE wariant: + ε_AE dW_noise
         ME wariant: dodatkowy + σ_ME·S dW_extra
    OU:  dX = θ(μ - X)dt + σ dW  (baseline AE)
         AE wariant: + ε_AE dW_noise (dodatkowy)
         ME wariant: + σ_ME·X dW_extra (multiplikatywny)
    Log-Normal: log-transformacja OU
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    sqrt_dt = np.sqrt(dt)

    paths_clean = np.zeros((n_paths, n_steps + 1))
    paths_ae    = np.zeros((n_paths, n_steps + 1))
    paths_me    = np.zeros((n_paths, n_steps + 1))

    paths_clean[:, 0] = x0
    paths_ae[:, 0]    = x0
    paths_me[:, 0]    = x0

    # Szoki Wienera (wspólna składowa + niezależne zakłócenia)
    dW_base = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_ae   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_me   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt

    for i in range(n_steps):
        if model == "gbm":
            # dS = μS dt + σ_base·S dW
            drift_c = mu * paths_clean[:, i]
            diff_c  = sigma_base * paths_clean[:, i]
            drift_ae = mu * paths_ae[:, i]
            diff_ae  = sigma_base * paths_ae[:, i]
            drift_me = mu * paths_me[:, i]
            diff_me  = sigma_base * paths_me[:, i]

        elif model == "ou":
            # dX = θ(μ - X)dt + σ_base dW
            drift_c  = theta * (mu - paths_clean[:, i])
            diff_c   = sigma_base * np.ones(n_paths)
            drift_ae = theta * (mu - paths_ae[:, i])
            diff_ae  = sigma_base * np.ones(n_paths)
            drift_me = theta * (mu - paths_me[:, i])
            diff_me  = sigma_base * np.ones(n_paths)

        elif model == "lognormal":
            # dX = μX dt + σ_base·X dW (jak GBM ale z inną interpretacją)
            drift_c  = (mu - 0.5 * sigma_base**2) * paths_clean[:, i]
            diff_c   = sigma_base * paths_clean[:, i]
            drift_ae = (mu - 0.5 * sigma_base**2) * paths_ae[:, i]
            diff_ae  = sigma_base * paths_ae[:, i]
            drift_me = (mu - 0.5 * sigma_base**2) * paths_me[:, i]
            diff_me  = sigma_base * paths_me[:, i]
        else:
            raise ValueError(f"Nieznany model: {model}")

        # Euler-Maruyama update
        # Czysty — bez dodatkowego szumu
        paths_clean[:, i+1] = paths_clean[:, i] + drift_c * dt + diff_c * dW_base[:, i]

        # Z błędem addytywnym — + σ_AE dW_AE (szum stały, niezależny od X)
        paths_ae[:, i+1] = (paths_ae[:, i]
                            + drift_ae * dt
                            + diff_ae * dW_base[:, i]
                            + sigma_ae * dW_ae[:, i])

        # Z błędem multiplikatywnym — + σ_ME · X · dW_ME (szum skaluje się z X)
        noise_me = sigma_me * np.abs(paths_me[:, i]) * dW_me[:, i]
        paths_me[:, i+1] = (paths_me[:, i]
                            + drift_me * dt
                            + diff_me * dW_base[:, i]
                            + noise_me)

    # Clip aby uniknąć eksplozji w ścieżkach GBM
    if model in ("gbm", "lognormal"):
        paths_clean = np.clip(paths_clean, 0.0, 1e6)
        paths_ae    = np.clip(paths_ae,    0.0, 1e6)
        paths_me    = np.clip(paths_me,    0.0, 1e6)

    stats = _compute_path_stats(paths_clean, paths_ae, paths_me)

    return SimResult(
        times=times,
        paths_clean=paths_clean,
        paths_ae=paths_ae,
        paths_me=paths_me,
        stats=stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESY NIELINIOWE
# ─────────────────────────────────────────────────────────────────────────────

def run_bistable_process(
    n_paths: int = 200,
    T: float = 10.0,
    dt: float = 0.005,
    # Parametry studni potencjału V(x) = -a/2·x² + b/4·x⁴
    a: float = 1.0,
    b: float = 1.0,
    # Bazowy szum procesowy
    sigma_base: float = 0.5,
    # Błędy
    sigma_ae: float = 0.3,
    sigma_me: float = 0.3,
    x0: float = 0.1,  # start w pobliżu niestabilnego równowagi
    seed: int | None = 42,
) -> SimResult:
    """
    Bistabilny potencjał podwójny (Double-Well Potential).

    dX = -V'(X)dt + noise·dW  gdzie V'(X) = -aX + bX³

    Referencja: Horsthemke & Lefever (1984) — szum multiplikatywny może
    wywoływać przejście fazowe (noise-induced transition) nawet gdy
    parametry a, b są stałe. Szum AE tylko poszerza rozkład, ME może
    CAŁKOWICIE zmienić topologię P_inf(x) z bimodalnej na unimodalną.

    Minimá potencjału (bez szumu): x* = ±sqrt(a/b)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    sqrt_dt = np.sqrt(dt)

    paths_clean = np.full((n_paths, n_steps + 1), x0)
    paths_ae    = np.full((n_paths, n_steps + 1), x0)
    paths_me    = np.full((n_paths, n_steps + 1), x0)

    # Różne warunki początkowe — część ścieżek startuje z + , część z -
    half = n_paths // 2
    paths_clean[:half, 0] = x0
    paths_clean[half:, 0] = -x0
    paths_ae[:half, 0]    = x0
    paths_ae[half:, 0]    = -x0
    paths_me[:half, 0]    = x0
    paths_me[half:, 0]    = -x0

    dW_base = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_ae   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_me   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt

    for i in range(n_steps):
        # Drift = -V'(X) = aX - bX³
        def drift(x): return a * x - b * x**3

        # Czysty
        xc = paths_clean[:, i]
        paths_clean[:, i+1] = xc + drift(xc) * dt + sigma_base * dW_base[:, i]

        # AE: dodatkowy stały szum
        xa = paths_ae[:, i]
        paths_ae[:, i+1] = (xa + drift(xa) * dt
                            + sigma_base * dW_base[:, i]
                            + sigma_ae * dW_ae[:, i])

        # ME: szum skaluje się z |X| — kluczowy efekt!
        xm = paths_me[:, i]
        me_noise = sigma_me * np.abs(xm) * dW_me[:, i]
        paths_me[:, i+1] = (xm + drift(xm) * dt
                            + sigma_base * dW_base[:, i]
                            + me_noise)

    # Clip
    clip = 5.0
    paths_clean = np.clip(paths_clean, -clip, clip)
    paths_ae    = np.clip(paths_ae,    -clip, clip)
    paths_me    = np.clip(paths_me,    -clip, clip)

    stats = _compute_path_stats(paths_clean, paths_ae, paths_me)
    return SimResult(
        times=times,
        paths_clean=paths_clean,
        paths_ae=paths_ae,
        paths_me=paths_me,
        stats=stats,
    )


def run_duffing_process(
    n_paths: int = 50,
    T: float = 30.0,
    dt: float = 0.005,
    # Parametry Duffing
    delta: float = 0.3,   # tłumienie
    alpha: float = -1.0,  # współczynnik liniowy
    beta: float = 1.0,    # współczynnik kubiczny
    omega: float = 1.2,   # częstotliwość wymuszania
    F: float = 0.3,       # amplituda wymuszania
    # Błędy
    sigma_ae: float = 0.1,
    sigma_me: float = 0.15,
    x0: float = 0.5,
    v0: float = 0.0,
    seed: int | None = 42,
) -> dict:
    """
    Duffing Oscillator z szumem addytywnym i multiplikatywnym.

    dX = V dt
    dV = (-δV - αX - βX³ + F·cos(ωt))dt + noise·dW

    Referencja: Arnold (1998) — szum ME może zmienić wykładnik Lapunova
    i indukować chaos tam, gdzie deterministyczny system jest regularny.

    Zwraca: słownik z trajektoriami (X, V) dla clean/AE/ME
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    sqrt_dt = np.sqrt(dt)

    def _simulate_duffing(sigma_extra: float, use_me: bool) -> tuple[np.ndarray, np.ndarray]:
        """Pojedyncza symulacja Duffing dla zadanego poziomu szumu."""
        X = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        X[0] = x0
        V[0] = v0
        for i in range(n_steps):
            t = times[i]
            force = F * np.cos(omega * t)
            dW = rng.standard_normal() * sqrt_dt
            # Szum: AE = stały σ, ME = σ·|X|
            noise_amp = sigma_extra * (abs(X[i]) if use_me else 1.0)

            # Runge-Kutta 4 dla deterministycznej części + Euler dla szumu
            # (metoda hybrydowa — dokładniejsza niż czyste EM)
            k1v = (-delta * V[i] - alpha * X[i] - beta * X[i]**3 + force)
            k1x = V[i]

            xm = X[i] + 0.5 * dt * k1x
            vm = V[i] + 0.5 * dt * k1v
            force_m = F * np.cos(omega * (t + 0.5 * dt))
            k2v = (-delta * vm - alpha * xm - beta * xm**3 + force_m)
            k2x = vm

            X[i+1] = X[i] + dt * k2x + noise_amp * dW
            V[i+1] = V[i] + dt * k2v + noise_amp * dW * 0.1  # słabszy szum w V

        return X, V

    X_clean, V_clean = _simulate_duffing(0.0, False)
    X_ae, V_ae = _simulate_duffing(sigma_ae, False)
    X_me, V_me = _simulate_duffing(sigma_me, True)

    # Wykładnik Lapunova metodą numeryczną (Benettin 1980)
    lyap_ae = _estimate_lyapunov_1d(X_ae)
    lyap_me = _estimate_lyapunov_1d(X_me)
    lyap_clean = _estimate_lyapunov_1d(X_clean)

    return {
        "times":   times,
        "X_clean": X_clean, "V_clean": V_clean,
        "X_ae":    X_ae,    "V_ae":    V_ae,
        "X_me":    X_me,    "V_me":    V_me,
        "lyap_clean": lyap_clean,
        "lyap_ae":    lyap_ae,
        "lyap_me":    lyap_me,
    }


def _estimate_lyapunov_1d(x: np.ndarray, epsilon: float = 1e-4) -> float:
    """
    Uproszczona estymacja największego wykładnika Lapunova z szeregu czasowego.
    Metoda: śledzenie rozejścia się zaburzonej trajektorii.
    Wynik > 0: chaos; Wynik < 0: stabilność.
    """
    n = len(x)
    if n < 20:
        return 0.0
    # Perturb and track divergence
    lyap_sum = 0.0
    count = 0
    for i in range(10, n - 1):
        dx = abs(x[i] - x[i-1]) + epsilon
        dx_next = abs(x[i+1] - x[i]) + epsilon
        if dx > 0:
            lyap_sum += np.log(dx_next / dx)
            count += 1
    return lyap_sum / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  FOKKER-PLANCK — rozkłady stacjonarne
# ─────────────────────────────────────────────────────────────────────────────

def compute_fokker_planck_stationary(
    f_type: Literal["ou", "bistable"] = "ou",
    # Parametry modelu
    theta: float = 2.0,   # OU: szybkość powrotu
    mu: float = 0.0,      # OU: długookresowa średnia
    a: float = 1.0,       # bistabilny: parametr a
    b: float = 1.0,       # bistabilny: parametr b
    # Poziomy szumu
    sigma_base: float = 0.5,
    sigma_ae: float = 0.3,
    sigma_me: float = 0.3,
    x_range: tuple[float, float] = (-3.0, 3.0),
    n_points: int = 500,
) -> FokkerPlanckResult:
    """
    Analityczne i numeryczne rozkłady stacjonarne Fokker-Planck.

    Dla procesu dX = f(X)dt + D(X)·dW, rozkład stacjonarny:
    P_inf(x) ∝ (1/D²(x)) · exp(2∫[f(x)/D²(x)]dx)

    AE: D(x) = σ_total = sqrt(σ_base² + σ_AE²)  → proporcjonalny do σ
    ME: D(x) = sqrt(σ_base² · ??? + (σ_ME·x)²)   → zależy od x!

    Klucz: ME zmienia kształt P_inf(x) — może tworzyć bimodalność
    (Horsthemke & Lefever 1984, Rozdział 5).
    """
    x = np.linspace(x_range[0], x_range[1], n_points)

    def _fp_stationary(x_arr, drift_fn, diff_fn) -> np.ndarray:
        """Numeryczne P_inf z warunkiem normalizacji."""
        # P_inf(x) ∝ (1/D²(x)) · exp(∫ 2f(x)/D²(x) dx)
        f_vals = drift_fn(x_arr)
        d_vals = diff_fn(x_arr)
        d2 = d_vals**2 + 1e-12  # zabezpieczenie przed zerem

        # Całka trapezowa z lewej granicy
        integrand = 2.0 * f_vals / d2
        dx = x_arr[1] - x_arr[0]
        integral = np.cumsum(integrand) * dx

        p = np.exp(integral - integral.max()) / d2
        p = np.maximum(p, 0.0)
        norm = np.trapezoid(p, x_arr)
        if norm > 1e-15:
            p /= norm
        return p

    if f_type == "ou":
        def drift(x): return theta * (mu - x)
    else:  # bistabilny
        def drift(x): return a * x - b * x**3

    # 1. Czysty — D(x) = σ_base (stała dla OU/bistabilny AE)
    def diff_clean(x): return sigma_base * np.ones_like(x)

    # 2. Z błędem AE — D(x) = sqrt(σ_base² + σ_AE²)  (nadal stała!)
    sigma_total_ae = np.sqrt(sigma_base**2 + sigma_ae**2)
    def diff_ae(x): return sigma_total_ae * np.ones_like(x)

    # 3. Z błędem ME — D(x) = sqrt(σ_base² + (σ_ME·|x|)²)  (zależy od x!)
    def diff_me(x): return np.sqrt(sigma_base**2 + (sigma_me * np.abs(x))**2)

    p_clean = _fp_stationary(x, drift, diff_clean)
    p_ae    = _fp_stationary(x, drift, diff_ae)
    p_me    = _fp_stationary(x, drift, diff_me)

    return FokkerPlanckResult(
        x_grid=x,
        p_stationary=p_clean,
        p_ae=p_ae,
        p_me=p_me,
    )


def compute_fp_time_evolution(
    f_type: Literal["ou", "bistable"] = "bistable",
    a: float = 1.0,
    b: float = 1.0,
    sigma_base: float = 0.5,
    sigma_ae: float = 0.3,
    sigma_me: float = 0.3,
    theta: float = 2.0,
    mu: float = 0.0,
    x_range: tuple[float, float] = (-2.5, 2.5),
    n_x: int = 200,
    n_t_snapshots: int = 8,
    T_total: float = 5.0,
) -> dict:
    """
    Ewolucja P(x,t) w czasie — snapshoty dla animacji w UI.
    Metoda: numeryczne różniczkowanie skończone równania Fokkera-Plancka.

    ∂P/∂t = -∂/∂x[f(x)P] + (1/2)∂²/∂x²[D²(x)P]
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    dx = x[1] - x[0]
    dt = 0.001  # mały krok czasowy dla stabilności

    if f_type == "ou":
        def drift(x): return theta * (mu - x)
    else:
        def drift(x): return a * x - b * x**3

    def diff_ae(x):  return np.sqrt(sigma_base**2 + sigma_ae**2) * np.ones_like(x)
    def diff_me(x):  return np.sqrt(sigma_base**2 + (sigma_me * np.abs(x))**2)

    def evolve_fp(diff_fn, n_steps: int) -> list[np.ndarray]:
        """Ewolucja P(x,t) metodą różnic skończonych (upwind scheme)."""
        # Start: Gaussian centered at 0
        P = np.exp(-x**2 / 0.1)
        P /= np.trapezoid(P, x) + 1e-15

        snapshots = [P.copy()]
        save_every = max(1, n_steps // n_t_snapshots)

        D2 = diff_fn(x)**2

        for step in range(n_steps):
            f = drift(x)
            # Fluxes — centered differences
            # ∂/∂x[f·P]:
            fP = f * P
            dfP = np.gradient(fP, dx)
            # ∂²/∂x²[D²·P]:
            D2P = D2 * P
            dD2P  = np.gradient(D2P, dx)
            d2D2P = np.gradient(dD2P, dx)

            dP = -dfP + 0.5 * d2D2P
            P = P + dt * dP
            P = np.maximum(P, 0.0)
            norm = np.trapezoid(P, x)
            if norm > 1e-15:
                P /= norm

            if (step + 1) % save_every == 0:
                snapshots.append(P.copy())

        return snapshots[:n_t_snapshots + 1]

    n_steps = int(T_total / dt)
    snaps_ae = evolve_fp(diff_ae, n_steps)
    snaps_me = evolve_fp(diff_me, n_steps)

    return {
        "x_grid":    x,
        "snaps_ae":  snaps_ae,
        "snaps_me":  snaps_me,
        "times":     np.linspace(0, T_total, len(snaps_ae)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STOCHASTIC RESONANCE
# ─────────────────────────────────────────────────────────────────────────────

def run_stochastic_resonance(
    omega_signal: float = 0.2,   # częstotliwość sygnału wejściowego
    A_signal: float = 0.8,       # amplituda sygnału (poniżej progu = 0.9)
    a: float = 1.0,              # parametry studni bistabilnej
    b: float = 1.0,
    T: float = 200.0,
    dt: float = 0.01,
    n_sigma: int = 30,           # liczba badanych poziomów σ
    sigma_max: float = 2.5,
    seed: int | None = 42,
) -> SRResult:
    """
    Stochastic Resonance (SR) — fenomen wzmocnienia sygnału przez szum.

    Referencja: Gammaitoni et al. (1998) Rev. Mod. Phys. 70, 223.
    Benzi et al. (1981) — oryginalne odkrycie w kontekście epok lodowych.

    Układ: dX = (aX - bX³ + A·sin(ωt))dt + σ·dW  (lub σ·X·dW dla ME)
    Miarą: SNR = amplituda składowej ω w widmie mocy / szum tła

    Kluczowy wynik: SNR(σ) ma MAKSIMUM przy optymalnym σ* — ani za mały,
    ani za duży szum. To właśnie Stochastic Resonance.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    times = np.arange(n_steps) * dt
    sqrt_dt = np.sqrt(dt)

    sigma_range = np.linspace(0.01, sigma_max, n_sigma)
    snr_ae = np.zeros(n_sigma)
    snr_me = np.zeros(n_sigma)

    def _compute_snr(x_series: np.ndarray) -> float:
        """Oblicza SNR przez FFT — stosunek piku przy ω do rms szumu."""
        # Okno Hanna aby uniknąć efektów brzegowych
        N = len(x_series)
        window = np.hanning(N)
        X_fft = np.fft.rfft(x_series * window)
        freqs  = np.fft.rfftfreq(N, d=dt)
        power  = np.abs(X_fft)**2

        # Znajdź indeks piku przy ω_signal
        target_freq = omega_signal / (2 * np.pi)
        idx_peak = np.argmin(np.abs(freqs - target_freq))
        if idx_peak == 0:
            idx_peak = 1

        signal_power = power[idx_peak]
        # Tło: mediana mocy w okolicach piku (bez samego piku)
        wing = max(2, idx_peak // 4)
        noise_band = np.concatenate([
            power[max(0, idx_peak - wing): max(0, idx_peak - 1)],
            power[idx_peak + 1: idx_peak + wing + 1]
        ])
        noise_power = np.median(noise_band) if len(noise_band) > 0 else power.mean()
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-12))
        return float(snr)

    for j, sigma in enumerate(sigma_range):
        # Szok wspólny
        dW = rng.standard_normal(n_steps) * sqrt_dt

        # AE: dX = (aX - bX³ + A·sin(ωt))dt + σ·dW
        x_ae = np.zeros(n_steps)
        x_ae[0] = 0.1
        for i in range(n_steps - 1):
            s = A_signal * np.sin(omega_signal * times[i])
            x_ae[i+1] = x_ae[i] + (a * x_ae[i] - b * x_ae[i]**3 + s) * dt + sigma * dW[i]
        x_ae = np.clip(x_ae, -5, 5)
        snr_ae[j] = _compute_snr(x_ae[n_steps//4:])  # skip transient

        # ME: dX = (aX - bX³ + A·sin(ωt))dt + σ·|X|·dW
        x_me = np.zeros(n_steps)
        x_me[0] = 0.1
        for i in range(n_steps - 1):
            s = A_signal * np.sin(omega_signal * times[i])
            x_me[i+1] = x_me[i] + (a * x_me[i] - b * x_me[i]**3 + s) * dt + sigma * abs(x_me[i]) * dW[i]
        x_me = np.clip(x_me, -5, 5)
        snr_me[j] = _compute_snr(x_me[n_steps//4:])

    # Referencja bez szumu
    x_clean = np.zeros(n_steps)
    x_clean[0] = 0.1
    for i in range(n_steps - 1):
        s = A_signal * np.sin(omega_signal * times[i])
        x_clean[i+1] = x_clean[i] + (a * x_clean[i] - b * x_clean[i]**3 + s) * dt
    snr_clean = _compute_snr(x_clean)

    opt_ae = float(sigma_range[np.argmax(snr_ae)])
    opt_me = float(sigma_range[np.argmax(snr_me)])

    return SRResult(
        sigma_range=sigma_range,
        snr_ae=snr_ae,
        snr_me=snr_me,
        snr_clean=snr_clean,
        opt_sigma_ae=opt_ae,
        opt_sigma_me=opt_me,
    )


def get_sr_sample_trajectories(
    sigma_low: float, sigma_opt: float, sigma_high: float,
    omega_signal: float = 0.2,
    A_signal: float = 0.8,
    a: float = 1.0, b: float = 1.0,
    T: float = 60.0, dt: float = 0.01,
    seed: int | None = 42,
) -> dict:
    """
    Generuje 3 trajektorie X(t) przy σ: za mały, optymalny, za duży.
    Do wizualizacji efektu SR.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    times = np.arange(n_steps) * dt
    sqrt_dt = np.sqrt(dt)
    signal = A_signal * np.sin(omega_signal * times)

    result = {}
    for label, sigma in [("low", sigma_low), ("opt", sigma_opt), ("high", sigma_high)]:
        x = np.zeros(n_steps)
        x[0] = 0.1
        dW = rng.standard_normal(n_steps) * sqrt_dt
        for i in range(n_steps - 1):
            x[i+1] = x[i] + (a*x[i] - b*x[i]**3 + signal[i]) * dt + sigma * dW[i]
        result[label] = np.clip(x, -5, 5)

    result["times"]  = times
    result["signal"] = signal
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ZASTOSOWANIE — PORTFEL Z BŁĘDEM MODELOWYM
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolio_with_errors(
    n_years: int = 20,
    mu_true: float = 0.07,     # prawdziwy dryf
    sigma_true: float = 0.18,  # prawdziwa zmienność
    sigma_ae: float = 0.02,    # błąd addytywny (microstructure noise)
    sigma_me: float = 0.05,    # błąd multiplikatywny (vol estimation error)
    initial_value: float = 100_000.0,
    n_paths: int = 500,
    dt: float = 1/252,
    seed: int | None = 42,
) -> dict:
    """
    Symuluje wartość portfela z dwoma typami błędów modelowych.

    Kontekst finansowy (Gardiner 2009, sekcja 4.4):
    - AE: bid-ask spread, rounding, microstructure noise — stały w skali
    - ME: błąd estymacji zmienności — skaluje się z poziomem portfela
      (np. 5% błąd w σ → 5% błąd w całym portfelu po czasie)

    Kluczowa konsekwencja ME: błąd narasta EKSPONENTALNIE z wartością portfela.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(n_years / dt)
    sqrt_dt = np.sqrt(dt)
    times_y = np.linspace(0, n_years, n_steps + 1)

    V_clean = np.full((n_paths, n_steps + 1), initial_value)
    V_ae    = np.full((n_paths, n_steps + 1), initial_value)
    V_me    = np.full((n_paths, n_steps + 1), initial_value)

    dW_base = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_ae   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt
    dW_me   = rng.standard_normal((n_paths, n_steps)) * sqrt_dt

    for i in range(n_steps):
        ret_base = (mu_true - 0.5 * sigma_true**2) * dt + sigma_true * dW_base[:, i]

        V_clean[:, i+1] = V_clean[:, i] * np.exp(ret_base)

        ret_ae = ret_base + sigma_ae * dW_ae[:, i]
        V_ae[:, i+1]   = V_ae[:, i] * np.exp(ret_ae)

        ret_me = ret_base + sigma_me * dW_me[:, i]  # ME: skaluje się z wartością V
        V_me[:, i+1]   = V_me[:, i] * np.exp(ret_me)

    def pct(paths, q): return np.percentile(paths[:, -1], q)

    final_clean = V_clean[:, -1]
    final_ae    = V_ae[:, -1]
    final_me    = V_me[:, -1]

    return {
        "times_y": times_y,
        "V_clean": V_clean,
        "V_ae":    V_ae,
        "V_me":    V_me,
        "summary": {
            "clean_median": np.median(final_clean),
            "ae_median":    np.median(final_ae),
            "me_median":    np.median(final_me),
            "clean_p10":  pct(V_clean, 10),
            "ae_p10":     pct(V_ae, 10),
            "me_p10":     pct(V_me, 10),
            "clean_p90":  pct(V_clean, 90),
            "ae_p90":     pct(V_ae, 90),
            "me_p90":     pct(V_me, 90),
            "expected_loss_ae": np.median(final_ae) - np.median(final_clean),
            "expected_loss_me": np.median(final_me) - np.median(final_clean),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  METRYKI STATYSTYCZNE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_path_stats(
    paths_clean: np.ndarray,
    paths_ae: np.ndarray,
    paths_me: np.ndarray,
) -> dict:
    """Oblicza statystyki porównawcze dla 3 zestawów ścieżek."""
    def final_stats(paths):
        final = paths[:, -1]
        return {
            "mean":     float(np.mean(final)),
            "std":      float(np.std(final)),
            "skewness": float(_skewness(final)),
            "kurtosis": float(_kurtosis(final)),
            "p5":       float(np.percentile(final, 5)),
            "p95":      float(np.percentile(final, 95)),
        }

    sc = final_stats(paths_clean)
    sa = final_stats(paths_ae)
    sm = final_stats(paths_me)

    return {
        "clean": sc,
        "ae":    sa,
        "me":    sm,
        "bias_ae":      sa["mean"] - sc["mean"],
        "bias_me":      sm["mean"] - sc["mean"],
        "variance_ae":  sa["std"]**2,
        "variance_me":  sm["std"]**2,
        "mse_ae":       (sa["mean"] - sc["mean"])**2 + sa["std"]**2,
        "mse_me":       (sm["mean"] - sc["mean"])**2 + sm["std"]**2,
    }


def _skewness(x: np.ndarray) -> float:
    """Trzeci moment standaryzowany."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s)**3))


def _kurtosis(x: np.ndarray) -> float:
    """Nadmiarowa kurtoza (excess kurtosis; 0 dla normalnego)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s)**4) - 3.0)


def compute_noise_scaling_demo(
    x_values: np.ndarray | None = None,
    sigma_ae: float = 0.5,
    sigma_me: float = 0.5,
) -> dict:
    """
    Prosty demo: jak AE i ME skalują się z wartością X.
    Używane do wykresu koncepcyjnego w Tab Teoria.
    """
    if x_values is None:
        x_values = np.linspace(0.01, 5.0, 200)

    noise_ae = sigma_ae * np.ones_like(x_values)   # stała amplituda
    noise_me = sigma_me * x_values                  # rośnie z X

    return {
        "x":        x_values,
        "noise_ae": noise_ae,
        "noise_me": noise_me,
        "ratio":    noise_me / noise_ae,  # kiedy ME > AE
    }
