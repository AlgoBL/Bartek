"""
conformal_predictor.py — Conformal Prediction dla gwarantowanych przedziałów ufności
======================================================================================
Ref: Angelopoulos, A. & Bates, S. (2024) "Conformal Risk Control", ICLR 2024
     Venn-Abers calibration + split conformal w jednym module.

Gwarantuje: P(Y ∈ Ĉ(X)) ≥ 1 − α  (marginal coverage)
BEZ żadnych założeń o rozkładzie danych — czysto dystrybucyjnie agnostyczne.

Użycie (przykład z TCN):
    cp = ConformalPredictor(alpha=0.10)         # 90% gwarancja pokrycia
    cp.calibrate(cal_predictions, cal_actuals)   # na zbiorze kalibracyjnym
    lo, hi = cp.predict_interval(new_pred)       # gwarancja dla nowej obserwacji

    # Dla klasyfikacji reżimów (softmax):
    cp_clf = ConformalClassifier(alpha=0.10)
    cp_clf.calibrate(cal_probs, cal_labels)
    prediction_set = cp_clf.predict_set(new_probs)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  REGRESJA — Gwarantowane przedziały ufności
# ─────────────────────────────────────────────────────────────────────────────

class ConformalPredictor:
    """
    Split Conformal Prediction dla regresji.

    Algorytm:
    1. Na zbiorze kalibracyjnym (oddzielnym od treningu!) oblicz
       non-conformity scores: s_i = |Y_i − Ŷ_i|
    2. Oblicz empiryczny (1-α)-kwantyl: q̂ = quantile(s, ⌈(n+1)(1−α)⌉ / n)
    3. Dla nowej obserwacji: Ĉ(X) = [Ŷ − q̂, Ŷ + q̂]

    Gwarancja: P(Y ∈ Ĉ(X)) ≥ 1 − α  (dla skończonej próby kalibracyjnej)
    """

    def __init__(self, alpha: float = 0.10):
        """
        Parameters
        ----------
        alpha : float
            Poziom błędu (0.10 → 90% pokrycie gwarantowane)
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None
        self._calibrated = False

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        symmetric: bool = True,
    ) -> dict:
        """
        Kalibracja na zbiorze walidacyjnym.

        Parameters
        ----------
        predictions : array (n,) — punktowe predykcje modelu
        actuals     : array (n,) — rzeczywiste wartości
        symmetric   : bool — czy używać |ŷ - y| (True) czy signed residuals (False)

        Returns
        -------
        dict z metrykami kalibracji: coverage_empirical, q_hat, n_cal
        """
        predictions = np.asarray(predictions, dtype=float)
        actuals     = np.asarray(actuals, dtype=float)
        n = len(actuals)

        if symmetric:
            self._cal_scores = np.abs(actuals - predictions)
        else:
            self._cal_scores = actuals - predictions   # signed

        # Empiryczny kwantyl z korektą finite-sample
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = np.quantile(self._cal_scores, level)
        self._calibrated = True

        # Empiryczne pokrycie (powinno być ≥ 1-alpha)
        emp_coverage = float(np.mean(self._cal_scores <= self.q_hat))

        return {
            "q_hat":              self.q_hat,
            "n_cal":              n,
            "alpha":              self.alpha,
            "coverage_target":    1 - self.alpha,
            "coverage_empirical": emp_coverage,
            "mean_abs_error":     float(np.mean(self._cal_scores)),
        }

    def predict_interval(
        self,
        new_prediction: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Zwraca gwarantowany przedział ufności dla nowej predykcji.

        Returns
        -------
        (lower, upper) — numpy arrays
        """
        if not self._calibrated:
            raise RuntimeError("Najpierw wywołaj calibrate().")
        pred = np.asarray(new_prediction, dtype=float)
        return pred - self.q_hat, pred + self.q_hat

    def predict_interval_df(
        self,
        predictions: pd.Series | np.ndarray,
        index=None,
    ) -> pd.DataFrame:
        """Zwraca DataFrame z kolumnami: prediction, lower, upper, width."""
        preds = np.asarray(predictions, dtype=float)
        lo, hi = self.predict_interval(preds)
        return pd.DataFrame(
            {"prediction": preds, "lower": lo, "upper": hi, "width": hi - lo},
            index=index,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  KLASYFIKACJA — Prediction Sets (dla reżimów TCN/GMM)
# ─────────────────────────────────────────────────────────────────────────────

class ConformalClassifier:
    """
    Adaptive Prediction Sets (APS) dla klasyfikacji reżimów.

    Ref: Romano, Sesia & Candès (2020) "Classification with Valid and
         Adaptive Coverage" NeurIPS 2020.

    Działa z dowolnym modelem dającym prawdopodobieństwa (softmax).
    Zamiast pojedynczej klasy, zwraca ZBIÓR klas gwarantując pokrycie.

    Przykład dla reżimów: {Bull, Bear} zamiast tylko "Bull" gdy niepewność duża.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._calibrated = False
        self.classes_: Optional[list] = None

    def _nonconformity(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Oblicza APS nonconformity scores."""
        n = len(labels)
        scores = np.zeros(n)
        for i in range(n):
            # Sortuj klasy malejąco po prawdopodobieństwie
            order = np.argsort(-probs[i])
            cumsum = 0.0
            for k, cls in enumerate(order):
                cumsum += probs[i][cls]
                if cls == labels[i]:
                    # Score = kumulatywna suma do prawdziwej klasy + losowość
                    scores[i] = cumsum - probs[i][cls] * np.random.uniform(0, 1)
                    break
        return scores

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        classes: Optional[list] = None,
    ) -> dict:
        """
        Parameters
        ----------
        cal_probs  : (n, K) array — prawdopodobieństwa klas
        cal_labels : (n,) array — prawdziwe etykiety (int lub str)
        classes    : lista nazw klas (opcjonalnie)
        """
        cal_probs  = np.asarray(cal_probs,  dtype=float)
        cal_labels = np.asarray(cal_labels)
        n = len(cal_labels)
        K = cal_probs.shape[1]

        if classes is not None:
            self.classes_ = classes
            # Zamień str labels na int indices
            class_to_idx = {c: i for i, c in enumerate(classes)}
            int_labels = np.array([class_to_idx[l] for l in cal_labels])
        else:
            self.classes_ = list(range(K))
            int_labels = cal_labels.astype(int)

        scores = self._nonconformity(cal_probs, int_labels)
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, min(level, 1.0)))
        self._calibrated = True

        emp_cov = float(np.mean(scores <= self.q_hat))
        return {
            "q_hat":              self.q_hat,
            "n_cal":              n,
            "coverage_target":    1 - self.alpha,
            "coverage_empirical": emp_cov,
        }

    def predict_set(self, probs: np.ndarray) -> list:
        """
        Zwraca prediction set (listę klas) dla wektora prawdopodobieństw.

        Parameters
        ----------
        probs : (K,) array — softmax probabilities

        Returns
        -------
        list nazw klas w zestawie predykcji
        """
        if not self._calibrated:
            raise RuntimeError("Najpierw wywołaj calibrate().")
        order     = np.argsort(-probs)
        pred_set  = []
        cumsum    = 0.0
        for cls_idx in order:
            cumsum += probs[cls_idx]
            pred_set.append(self.classes_[cls_idx])
            if cumsum >= self.q_hat:
                break
        return pred_set

    def predict_set_batch(self, probs_matrix: np.ndarray) -> list[list]:
        """Batch version dla wielu obserwacji."""
        return [self.predict_set(probs_matrix[i]) for i in range(len(probs_matrix))]


# ─────────────────────────────────────────────────────────────────────────────
#  WBUDOWANA SYMULACJA KALIBRACYJNA (dla demo / testów)
# ─────────────────────────────────────────────────────────────────────────────

def demo_regression(n_cal: int = 500, alpha: float = 0.10) -> dict:
    """Szybki demo conformal prediction dla regresji."""
    np.random.seed(42)
    # Symulowany model: predykcja z szumem
    x_cal  = np.random.uniform(-3, 3, n_cal)
    y_cal  = np.sin(x_cal) + np.random.normal(0, 0.3, n_cal)
    y_pred = np.sin(x_cal) + np.random.normal(0, 0.5, n_cal)  # niedoskonały model

    cp = ConformalPredictor(alpha=alpha)
    metrics = cp.calibrate(y_pred, y_cal)

    # Test na nowych danych
    x_test = np.linspace(-3, 3, 50)
    y_test_pred = np.sin(x_test)
    lo, hi = cp.predict_interval(y_test_pred)

    # Rzeczywiste pokrycie
    y_test_true = np.sin(x_test) + np.random.normal(0, 0.3, 50)
    actual_cov  = float(np.mean((y_test_true >= lo) & (y_test_true <= hi)))

    return {
        **metrics,
        "coverage_test": actual_cov,
        "interval_width_mean": float(np.mean(hi - lo)),
        "x_test": x_test,
        "y_pred": y_test_pred,
        "lower":  lo,
        "upper":  hi,
        "y_true": y_test_true,
    }


def demo_classification(n_cal: int = 300, alpha: float = 0.10) -> dict:
    """Szybki demo conformal classification dla reżimów."""
    np.random.seed(42)
    classes = ["Bull", "Neutral", "Bear"]
    K = len(classes)

    # Symulowane prawdopodobieństwa (np. z GMM)
    true_labels  = np.random.choice(K, n_cal)
    # Model jest ok. 70% dokładny — daje wyższe prawdopodobieństwo prawdziwej klasy
    raw_probs = np.random.dirichlet([0.5] * K, n_cal)
    for i in range(n_cal):
        raw_probs[i, true_labels[i]] += 0.6
    raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)

    cp_clf = ConformalClassifier(alpha=alpha)
    metrics = cp_clf.calibrate(raw_probs, true_labels, classes=classes)

    # Przykładowe prediction sets
    test_probs   = np.random.dirichlet([1.0] * K, 10)
    pred_sets    = cp_clf.predict_set_batch(test_probs)
    avg_set_size = np.mean([len(s) for s in pred_sets])

    return {
        **metrics,
        "avg_set_size":  avg_set_size,
        "example_probs": test_probs[:3].tolist(),
        "example_sets":  pred_sets[:3],
    }
