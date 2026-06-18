"""
hmm_regime.py — Hidden Markov Model Regime Detection
======================================================
Implementuje Gaussian HMM dla detekcji ukrytych reżimów rynkowych.

Zamiast prostych progów (Investment Clock), HMM:
  - Automatycznie identyfikuje K reżimów z danych (Bull, Bear, Crisis, Neutral)
  - Zwraca PROBABILISTYCZNE przypisanie P(reżim | historia) — nie 0/1
  - Obsługuje przejścia stanów (transition matrix) — lepsze modelowanie persistency
  - Wyestymowane parametry: (μ_k, σ_k) per reżim → interpretowalność

Architektura:
  Obserwacje: r_t = μ_{S_t} + σ_{S_t}·ε_t,  ε_t ~ N(0,1)
  Stany ukryte: S_t ∈ {0, 1, ..., K-1}
  Przejścia: P(S_t | S_{t-1}) = A_{ij} (macierz przejść)

Referencje:
  Hamilton (1989) — "A New Approach to the Economic Analysis of Nonstationary
                    Time Series", Econometrica 57(2).
  Hamilton (1994) — "Time Series Analysis", Princeton UP, Ch. 22.
  Guidolin & Timmermann (2008) — "International Asset Allocation Under
                                  Regime Switching", Review of Financial Studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from modules.logger import setup_logger

logger = setup_logger(__name__)

# ─── Kolor per reżim (interpretacja po posortowaniu wg μ) ────────────────────
_REGIME_COLORS_BASE = [
    "#00e676",   # 0: najwyższe μ → Bull
    "#3498db",   # 1: umiarkowane  → Normal
    "#f39c12",   # 2: niskie       → Caution
    "#e74c3c",   # 3: najniższe    → Bear/Crisis
]
_REGIME_LABELS_BASE = ["Bull 🐂", "Normal 📈", "Caution ⚠️", "Bear/Crisis 💀"]


class HMMRegimeDetector:
    """
    Gaussian HMM z Baum-Welch EM oraz Viterbi dekodowaniem.

    Obsługuje 2-5 reżimów (K). Domyślnie K=3 (Bull, Neutral, Bear).

    Użycie:
        hmm = HMMRegimeDetector(n_regimes=3)
        hmm.fit(returns_series)
        regimes = hmm.predict(returns_series)          # Viterbi
        probs   = hmm.predict_proba(returns_series)    # P(S_t | data)
        summary = hmm.regime_summary()
    """

    def __init__(self, n_regimes: int = 3, n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Parametry modelu
        self._mu: np.ndarray | None = None        # (K,) — średnie per reżim
        self._sigma: np.ndarray | None = None     # (K,) — odch. std per reżim
        self._pi: np.ndarray | None = None        # (K,) — init probs
        self._A: np.ndarray | None = None         # (K, K) — transition matrix
        self._fitted = False
        self._regime_map: list[int] = []           # mapowanie indeksu → sortowany

        # Wyniki ostatniego fit
        self._last_log_likelihood: float = -np.inf
        self._fitted_returns: pd.Series | None = None

    # ─── 1. FIT (Baum-Welch EM) ───────────────────────────────────────────────

    def fit(self, returns: pd.Series | np.ndarray) -> "HMMRegimeDetector":
        """
        Estymuje parametry HMM przez EM (Baum-Welch).
        Próbuje hmmlearn najpierw (szybciej), fallback do własnej implementacji.

        Parameters
        ----------
        returns : dzienny szereg zwrotów (ułamkowe: 0.01 = 1%)
        """
        r = self._prepare(returns)
        if len(r) < 60:
            logger.warning("HMM: za mało danych (<60 obs) — używam domyślnych")
            self._init_default_params(r)
            return self

        # ── Próba użycia hmmlearn (opcjonalna) ──────────────────────────────
        fitted = self._try_hmmlearn(r)
        if not fitted:
            # Fallback: własna implementacja EM
            self._em_fit(r)

        # Posortuj reżimy wg μ (malejące — Bull najpierw)
        order = np.argsort(self._mu)[::-1]
        self._mu = self._mu[order]
        self._sigma = self._sigma[order]
        self._pi = self._pi[order]
        self._A = self._A[np.ix_(order, order)]
        self._regime_map = list(order)

        self._fitted = True
        self._fitted_returns = pd.Series(r)
        logger.info(
            f"HMM fitted K={self.n_regimes} | "
            f"μ={self._mu.round(4)} | σ={self._sigma.round(4)} | "
            f"LL={self._last_log_likelihood:.2f}"
        )
        return self

    def _try_hmmlearn(self, r: np.ndarray) -> bool:
        """Próbuje użyć hmmlearn. Zwraca True jeśli sukces."""
        try:
            from hmmlearn import hmm as hmmlearn_hmm
            X = r.reshape(-1, 1)
            model = hmmlearn_hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
                verbose=False,
            )
            model.fit(X)
            self._mu = model.means_.flatten()
            self._sigma = np.sqrt(model.covars_.flatten())
            self._pi = model.startprob_
            self._A = model.transmat_
            self._last_log_likelihood = model.score(X)
            self._hmmlearn_model = model
            return True
        except (ImportError, Exception) as e:
            logger.debug(f"hmmlearn niedostępny lub błąd: {e} — używam własnego EM")
            return False

    # ─── 2. Własna implementacja EM (fallback) ────────────────────────────────

    def _em_fit(self, r: np.ndarray) -> None:
        """Baum-Welch EM dla Gaussian HMM — własna implementacja."""
        K = self.n_regimes
        T = len(r)
        rng = np.random.default_rng(self.random_state)

        # Inicjalizacja K-means
        kmeans_idx = np.argsort(r)
        chunk = T // K
        mu = np.array([r[kmeans_idx[i * chunk:(i + 1) * chunk]].mean() for i in range(K)])
        sigma = np.full(K, r.std() * 0.7)
        pi = np.ones(K) / K
        A = np.ones((K, K)) / K

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step: Forward-Backward
            alpha, scale = self._forward(r, pi, A, mu, sigma)
            beta = self._backward(r, A, mu, sigma, scale)
            gamma, xi = self._compute_gamma_xi(r, alpha, beta, A, mu, sigma)

            # M-step: aggiornare parametri
            pi_new = gamma[0] / gamma[0].sum()
            A_new = xi.sum(axis=0) / (xi.sum(axis=0).sum(axis=1, keepdims=True) + 1e-12)
            mu_new = (gamma * r[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-12)
            sigma_new = np.sqrt(
                (gamma * (r[:, None] - mu_new[None, :]) ** 2).sum(axis=0)
                / (gamma.sum(axis=0) + 1e-12)
            )
            sigma_new = np.maximum(sigma_new, 1e-6)

            # Log-likelihood
            ll = float(np.sum(np.log(scale + 1e-300)))

            pi, A, mu, sigma = pi_new, A_new, mu_new, sigma_new

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._mu = mu
        self._sigma = sigma
        self._pi = pi
        self._A = A
        self._last_log_likelihood = prev_ll

    def _gaussian_pdf(self, r: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """B(t, k) = N(r_t | μ_k, σ_k). Returns (T, K)."""
        diff = r[:, None] - mu[None, :]
        return np.exp(-0.5 * (diff / (sigma[None, :] + 1e-10)) ** 2) / (
            np.sqrt(2 * np.pi) * sigma[None, :] + 1e-300
        )

    def _forward(self, r, pi, A, mu, sigma):
        T, K = len(r), len(mu)
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        B = self._gaussian_pdf(r, mu, sigma)

        alpha[0] = pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0] + 1e-300

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t] + 1e-300
        return alpha, scale

    def _backward(self, r, A, mu, sigma, scale):
        T, K = len(r), len(mu)
        beta = np.zeros((T, K))
        B = self._gaussian_pdf(r, mu, sigma)
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (A * B[t + 1][None, :] * beta[t + 1][None, :]).sum(axis=1)
            beta[t] /= scale[t + 1] + 1e-300
        return beta

    def _compute_gamma_xi(self, r, alpha, beta, A, mu, sigma):
        T, K = alpha.shape
        B = self._gaussian_pdf(r, mu, sigma)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = alpha[t, :, None] * A * B[t + 1, None, :] * beta[t + 1, None, :]
            xi[t] /= xi[t].sum() + 1e-300
        return gamma, xi

    def _init_default_params(self, r: np.ndarray) -> None:
        """Domyślne parametry gdy za mało danych."""
        K = self.n_regimes
        self._mu = np.linspace(r.mean() + r.std(), r.mean() - r.std(), K)
        self._sigma = np.full(K, r.std() * 0.8)
        self._pi = np.ones(K) / K
        self._A = np.full((K, K), 1 / K)
        self._fitted = True

    # ─── 3. PREDICT (Viterbi) ────────────────────────────────────────────────

    def predict(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """
        Viterbi dekodowanie — najbardziej prawdopodobna sekwencja stanów.
        Returns: array (T,) z indeksami reżimów.
        """
        if not self._fitted:
            raise RuntimeError("HMM nie został wytrenowany — wywołaj fit() najpierw")

        # Użyj hmmlearn jeśli dostępny
        if hasattr(self, "_hmmlearn_model"):
            r = self._prepare(returns).reshape(-1, 1)
            raw = self._hmmlearn_model.predict(r)
            # Przetłumacz na posortowane indeksy
            inv_map = {orig: new for new, orig in enumerate(self._regime_map)}
            return np.array([inv_map.get(s, s) for s in raw])

        r = self._prepare(returns)
        T = len(r)
        K = self.n_regimes
        B = self._gaussian_pdf(r, self._mu, self._sigma)

        # Log-domain Viterbi
        log_A = np.log(self._A + 1e-300)
        log_B = np.log(B + 1e-300)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self._pi + 1e-300) + log_B[0]

        for t in range(1, T):
            for k in range(K):
                vals = delta[t - 1] + log_A[:, k]
                psi[t, k] = np.argmax(vals)
                delta[t, k] = vals[psi[t, k]] + log_B[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """
        Posterior P(S_t | data) przez Forward-Backward.
        Returns: (T, K) — posterior probability per reżim.
        """
        if not self._fitted:
            raise RuntimeError("HMM nie został wytrenowany")

        if hasattr(self, "_hmmlearn_model"):
            r = self._prepare(returns).reshape(-1, 1)
            proba = self._hmmlearn_model.predict_proba(r)
            # Przetłumacz kolumny na posortowane indeksy
            inv_map = {orig: new for new, orig in enumerate(self._regime_map)}
            sorted_proba = np.zeros_like(proba)
            for orig, new in inv_map.items():
                if orig < proba.shape[1] and new < sorted_proba.shape[1]:
                    sorted_proba[:, new] = proba[:, orig]
            return sorted_proba

        r = self._prepare(returns)
        alpha, scale = self._forward(r, self._pi, self._A, self._mu, self._sigma)
        beta = self._backward(r, self._A, self._mu, self._sigma, scale)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    # ─── 4. Analityki i Summary ───────────────────────────────────────────────

    def regime_summary(self) -> dict:
        """Słownik z parametrami i interpretacją każdego reżimu."""
        if not self._fitted:
            return {}

        summaries = []
        labels = self._get_labels()
        colors = self._get_colors()

        for k in range(self.n_regimes):
            persistence = float(self._A[k, k])
            expected_duration = 1.0 / max(1.0 - persistence, 1e-6)
            summaries.append({
                "regime_id":         k,
                "label":             labels[k],
                "color":             colors[k],
                "mu_annual":         float(self._mu[k] * 252),
                "sigma_annual":      float(self._sigma[k] * np.sqrt(252)),
                "sharpe_proxy":      float(self._mu[k] / (self._sigma[k] + 1e-10) * np.sqrt(252)),
                "persistence":       persistence,
                "expected_duration_days": expected_duration,
                "init_prob":         float(self._pi[k]),
                "transition_to":     {
                    labels[j]: float(self._A[k, j])
                    for j in range(self.n_regimes) if j != k
                },
            })

        return {
            "regimes":          summaries,
            "n_regimes":        self.n_regimes,
            "log_likelihood":   self._last_log_likelihood,
            "transition_matrix": self._A.tolist(),
            "labels":           labels,
            "colors":           colors,
        }

    def get_current_regime(self, returns: pd.Series | np.ndarray) -> dict:
        """
        Zwraca obecny reżim (ostatnia obserwacja) z prawdopodobieństwem.
        """
        proba = self.predict_proba(returns)
        last_proba = proba[-1]
        regime_id = int(np.argmax(last_proba))
        labels = self._get_labels()
        colors = self._get_colors()
        return {
            "regime_id":        regime_id,
            "label":            labels[regime_id],
            "color":            colors[regime_id],
            "confidence":       float(last_proba[regime_id]),
            "probabilities":    {labels[k]: float(last_proba[k]) for k in range(self.n_regimes)},
            "mu_annual":        float(self._mu[regime_id] * 252),
            "sigma_annual":     float(self._sigma[regime_id] * np.sqrt(252)),
        }

    def _get_labels(self) -> list[str]:
        K = self.n_regimes
        base = _REGIME_LABELS_BASE[:K]
        if len(base) < K:
            base += [f"Regime {i}" for i in range(len(base), K)]
        return base

    def _get_colors(self) -> list[str]:
        K = self.n_regimes
        base = _REGIME_COLORS_BASE[:K]
        if len(base) < K:
            base += ["#888"] * (K - len(base))
        return base

    @staticmethod
    def _prepare(returns: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(returns, pd.Series):
            r = returns.dropna().values
        else:
            r = np.asarray(returns, dtype=float)
            r = r[~np.isnan(r)]
        return r.astype(float)


# ─── Wizualizacja Plotly ─────────────────────────────────────────────────────

def plot_hmm_regimes(
    returns: pd.Series,
    hmm: HMMRegimeDetector,
    title: str = "HMM Regime Detection — Hidden Markov Model",
) -> go.Figure:
    """
    Wykres 3-panelowy: cena + reżim posteriorny + prawdopodobieństwa stanów.
    """
    proba = hmm.predict_proba(returns)
    states = hmm.predict(returns)
    labels = hmm._get_labels()
    colors = hmm._get_colors()

    dates = returns.dropna().index
    n = min(len(dates), len(proba))
    dates = dates[:n]
    proba = proba[:n]
    states = states[:n]

    prices = (1 + returns.dropna().iloc[:n]).cumprod()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Portfel (skum.)", "Reżim HMM (Viterbi)", "P(Reżim | Dane) — Posteriorny"],
        row_heights=[0.35, 0.25, 0.40],
    )

    # Panel 1: Cena
    fig.add_trace(go.Scatter(
        x=dates, y=prices.values,
        line=dict(color="#e2e4f0", width=1.5),
        name="Wartość",
    ), row=1, col=1)

    # Kolorowanie tła wg reżimu
    for k in range(hmm.n_regimes):
        mask = states == k
        for i in np.where(np.diff(np.concatenate([[False], mask, [False]])))[0].reshape(-1, 2):
            fig.add_vrect(
                x0=dates[i[0]] if i[0] < len(dates) else dates[-1],
                x1=dates[i[1] - 1] if i[1] <= len(dates) else dates[-1],
                fillcolor=colors[k],
                opacity=0.1, layer="below", line_width=0,
                row=1, col=1,
            )

    # Panel 2: Viterbi states (scatter)
    fig.add_trace(go.Scatter(
        x=dates, y=states,
        mode="lines",
        line=dict(color="#ffea00", width=1.5),
        name="Reżim (Viterbi)",
    ), row=2, col=1)
    fig.update_yaxes(
        tickvals=list(range(hmm.n_regimes)),
        ticktext=labels,
        row=2, col=1,
    )

    # Panel 3: Posterior probabilities (stacked area)
    for k in range(hmm.n_regimes):
        fig.add_trace(go.Scatter(
            x=dates, y=proba[:, k],
            name=labels[k],
            line=dict(color=colors[k], width=1),
            stackgroup="one",
            fillcolor=colors[k].replace("#", "rgba(").rstrip(")") if "#" in colors[k] else colors[k],
        ), row=3, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e4f0")),
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.8)",
        font=dict(color="#e2e4f0", family="Inter"),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#1c1c2e")
    fig.update_yaxes(gridcolor="#1c1c2e")

    return fig


def plot_hmm_transition_matrix(hmm: HMMRegimeDetector) -> go.Figure:
    """Heatmapa macierzy przejść A_{ij}."""
    labels = hmm._get_labels()
    A = hmm._A

    fig = go.Figure(go.Heatmap(
        z=A,
        x=[f"→ {l}" for l in labels],
        y=[f"{l} →" for l in labels],
        text=np.round(A, 3),
        texttemplate="%{text}",
        colorscale="Viridis",
        zmin=0, zmax=1,
        showscale=True,
    ))
    fig.update_layout(
        title="Macierz Przejść HMM (A)",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        margin=dict(l=100, r=20, t=50, b=80),
    )
    return fig
