"""
LSTM Regime Detector — Intelligent Barbell
Sequential neural network for market regime prediction with soft probabilities.
Fallback to GMM if PyTorch is not installed.
Reference: Hochreiter & Schmidhuber (1997), Lim et al. (2021) - TFT.
"""
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _LSTMNet(nn.Module if HAS_TORCH else object):
    """Simple 2-layer LSTM with fully connected head for regime classification."""
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last timestep
        return out


class LSTMRegimeDetector:
    """
    LSTM-based regime detector.
    Trains on rolling windows of market features to predict Bull/Bear/Crisis.

    Parameters
    ----------
    seq_len : lookback window size (days)
    n_regimes : number of regime classes
    hidden_size : LSTM hidden units
    n_epochs : training epochs
    """

    def __init__(
        self,
        seq_len: int = 30,
        n_regimes: int = 3,
        hidden_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ):
        self.seq_len = seq_len
        self.n_regimes = n_regimes
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.trained = False
        self._available = HAS_TORCH

    def _build_features(self, returns: pd.Series) -> np.ndarray:
        """Multi-feature array: return, rolling_vol, momentum, skew."""
        df = pd.DataFrame({"r": returns})
        df["vol"] = df["r"].rolling(21, min_periods=5).std().fillna(df["r"].std())
        df["mom"] = df["r"].rolling(63, min_periods=10).mean().fillna(0)
        df["skew"] = df["r"].rolling(63, min_periods=10).skew().fillna(0)
        return df[["r", "vol", "mom", "skew"]].values.astype(np.float32)

    def _make_sequences(self, X: np.ndarray):
        """Build (n_samples, seq_len, n_features) tensor from flat array."""
        seqs = []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len: i])
        return np.array(seqs, dtype=np.float32)

    def fit(self, returns: pd.Series, labels: np.ndarray):
        """
        Train LSTM on (returns, labels).
        Labels should come from a pre-fitted GMM for pseudo-labels.
        """
        if not HAS_TORCH:
            return  # no-op, will use GMM fallback

        X = self._build_features(returns)
        X_seq = self._make_sequences(X)  # (N-seq_len, seq_len, 4)
        y = labels[self.seq_len:]        # align labels

        n_features = X_seq.shape[2]

        self.model = _LSTMNet(
            input_size=n_features,
            hidden_size=self.hidden_size,
            n_classes=self.n_regimes,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_seq)
        y_t = torch.tensor(y.astype(np.int64))

        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.trained = True

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """
        Predict soft regime probabilities. Shape: (n_samples, n_regimes).
        For the first seq_len samples, returns uniform distribution as warmup.
        """
        if not HAS_TORCH or not self.trained:
            return None  # signal to fallback to GMM

        X = self._build_features(returns)
        X_seq = self._make_sequences(X)
        X_t = torch.tensor(X_seq)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1).numpy()

        # Pad beginning with uniform distribution
        warmup = np.ones((self.seq_len, self.n_regimes)) / self.n_regimes
        return np.vstack([warmup, proba])

    def is_available(self) -> bool:
        return HAS_TORCH


def train_lstm_on_gmm_labels(
    returns: pd.Series,
    gmm_labels: np.ndarray,
    seq_len: int = 30,
    n_regimes: int = 3,
    n_epochs: int = 50,
) -> "LSTMRegimeDetector":
    """
    Convenience function: train LSTM with GMM-generated pseudo-labels.
    Returns fitted LSTMRegimeDetector or None if PyTorch not available.
    """
    detector = LSTMRegimeDetector(seq_len=seq_len, n_regimes=n_regimes, n_epochs=n_epochs)
    if not detector.is_available():
        return None
    detector.fit(returns, gmm_labels)
    return detector
