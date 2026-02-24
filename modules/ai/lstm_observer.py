"""
Neural Regime Detector — Intelligent Barbell v3.0
Implementuje dwa modele sekwencyjne do detekcji reżimów rynkowych:

  1. TCN  — Temporal Convolutional Network (dilated causal convolutions)
             Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
             and Recurrent Networks for Sequence Modeling"
             Lepszy od LSTM: szybszy trening, brak problemu zanikającego gradientu,
             równoległy trening na całej sekwencji.

  2. LSTM — zachowany jako alternatywa / porównanie

Fallback do GMM jeśli PyTorch niedostępny.
"""
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─── Helper: base class safe for no-torch ────────────────────────────────────
_ModuleBase = nn.Module if HAS_TORCH else object


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TCN — Temporal Convolutional Network
# ══════════════════════════════════════════════════════════════════════════════

class _CausalConv1d(nn.Module if HAS_TORCH else object):
    """
    Dilated Causal Convolution block.
    Padding = (kernel-1) * dilation → left-only padding → causal.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        out = out[:, :, : x.size(2)]   # remove future-looking padding
        # LayerNorm expects (batch, seq, channels) → transpose
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return F.relu(out)


class _TCNBlock(nn.Module if HAS_TORCH else object):
    """Residual TCN block: two dilated causal convs + skip connection."""
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.dropout(self.conv1(x))
        out = self.dropout(self.conv2(out))
        return F.relu(out + residual)


class _TCNNet(nn.Module if HAS_TORCH else object):
    """
    Full TCN for regime classification.
    Architecture: projection → N residual blocks (exponential dilation) → head.

    Receptive field = 1 + 2*(kernel-1)*(2^N - 1)
    With kernel=3, N=4 blocks: RF = 1 + 2*(2)*(15) = 61 days.
    """
    def __init__(self, input_size: int, n_classes: int,
                 n_channels: int = 64, kernel_size: int = 3, n_blocks: int = 4):
        super().__init__()
        # Input projection
        self.input_proj = nn.Conv1d(input_size, n_channels, kernel_size=1)

        # Residual TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            _TCNBlock(n_channels, kernel_size, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        # Classification head (uses last time step)
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_channels // 2, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # Take last timestep → (batch, channels)
        x = x[:, :, -1]
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LSTM (original architecture, kept for comparison)
# ══════════════════════════════════════════════════════════════════════════════

class _LSTMNet(nn.Module if HAS_TORCH else object):
    """2-layer LSTM with fully connected head for regime classification."""
    def __init__(self, input_size: int, hidden_size: int,
                 n_classes: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Unified Regime Detector (TCN default, LSTM fallback)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMRegimeDetector:
    """
    Neural regime detector: TCN (default) or LSTM.
    Trains on rolling windows of market features → predicts Bull/Bear/Crisis.

    Parameters
    ----------
    seq_len    : lookback window size (days)
    n_regimes  : number of regime classes (3 = Bull/Bear/Crisis)
    model_type : 'tcn' (default) or 'lstm'
    n_epochs   : training epochs
    """

    def __init__(
        self,
        seq_len: int = 30,
        n_regimes: int = 3,
        model_type: str = "tcn",
        hidden_size: int = 64,
        n_epochs: int = 60,
        lr: float = 1e-3,
    ):
        self.seq_len    = seq_len
        self.n_regimes  = n_regimes
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_epochs   = n_epochs
        self.lr         = lr
        self.model      = None
        self.trained    = False
        self._available = HAS_TORCH

    # ── Feature engineering ──────────────────────────────────────────────
    def _build_features(self, returns: pd.Series) -> np.ndarray:
        """7-feature array: return, vol_21, vol_5, momentum_63, momentum_21,
        skew_63, realised_range."""
        df = pd.DataFrame({"r": returns})
        df["vol_21"]   = df["r"].rolling(21, min_periods=5).std().fillna(df["r"].std())
        df["vol_5"]    = df["r"].rolling(5,  min_periods=2).std().fillna(df["r"].std())
        df["mom_63"]   = df["r"].rolling(63, min_periods=10).mean().fillna(0)
        df["mom_21"]   = df["r"].rolling(21, min_periods=5).mean().fillna(0)
        df["skew_63"]  = df["r"].rolling(63, min_periods=10).skew().fillna(0)
        df["rng"]      = (df["r"] - df["r"].rolling(21).mean()) / (df["vol_21"] + 1e-8)
        return df[["r", "vol_21", "vol_5", "mom_63", "mom_21", "skew_63", "rng"]].values.astype(np.float32)

    def _make_sequences(self, X: np.ndarray):
        seqs = [X[i - self.seq_len: i] for i in range(self.seq_len, len(X))]
        return np.array(seqs, dtype=np.float32)

    # ── Build model ───────────────────────────────────────────────────────
    def _build_model(self, n_features: int):
        if self.model_type == "tcn":
            return _TCNNet(
                input_size=n_features,
                n_classes=self.n_regimes,
                n_channels=64,
                kernel_size=3,
                n_blocks=4,
            )
        else:
            return _LSTMNet(
                input_size=n_features,
                hidden_size=self.hidden_size,
                n_classes=self.n_regimes,
            )

    # ── Training ──────────────────────────────────────────────────────────
    def fit(self, returns: pd.Series, labels: np.ndarray):
        """
        Train model on (returns, labels).
        Labels typically come from a pre-fitted GMM (pseudo-labels).
        """
        if not HAS_TORCH:
            return

        X      = self._build_features(returns)
        X_seq  = self._make_sequences(X)
        y      = labels[self.seq_len:]

        n_features = X_seq.shape[2]
        self.model = self._build_model(n_features)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_seq)
        y_t = torch.tensor(y.astype(np.int64))

        self.model.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss   = criterion(logits, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        self.trained = True

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_proba(self, returns: pd.Series) -> np.ndarray | None:
        """
        Returns soft regime probabilities. Shape: (n_samples, n_regimes).
        Columns: [P(Bull), P(Bear), P(Crisis)].
        Returns None if PyTorch unavailable or model not trained.
        """
        if not HAS_TORCH or not self.trained:
            return None

        X     = self._build_features(returns)
        X_seq = self._make_sequences(X)
        X_t   = torch.tensor(X_seq)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            proba  = torch.softmax(logits, dim=1).numpy()

        warmup = np.ones((self.seq_len, self.n_regimes)) / self.n_regimes
        return np.vstack([warmup, proba])

    def is_available(self) -> bool:
        return HAS_TORCH

    def model_name(self) -> str:
        return f"{'TCN' if self.model_type == 'tcn' else 'LSTM'} (PyTorch {'✅' if HAS_TORCH else '❌'})"


# ── Convenience functions ─────────────────────────────────────────────────────

def train_regime_detector(
    returns: pd.Series,
    gmm_labels: np.ndarray,
    seq_len: int = 30,
    n_regimes: int = 3,
    n_epochs: int = 60,
    model_type: str = "tcn",
) -> "LSTMRegimeDetector | None":
    """
    Train TCN (default) or LSTM on GMM pseudo-labels.
    Returns fitted detector, or None if PyTorch not available.
    """
    detector = LSTMRegimeDetector(
        seq_len=seq_len,
        n_regimes=n_regimes,
        model_type=model_type,
        n_epochs=n_epochs,
    )
    if not detector.is_available():
        return None
    detector.fit(returns, gmm_labels)
    return detector


# Backwards-compatible alias
def train_lstm_on_gmm_labels(
    returns: pd.Series,
    gmm_labels: np.ndarray,
    seq_len: int = 30,
    n_regimes: int = 3,
    n_epochs: int = 50,
) -> "LSTMRegimeDetector | None":
    return train_regime_detector(
        returns, gmm_labels,
        seq_len=seq_len, n_regimes=n_regimes,
        n_epochs=n_epochs, model_type="tcn",
    )
