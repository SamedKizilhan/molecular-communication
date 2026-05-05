"""
ML-based detectors: MLP, ANN, CNN, LSTM (PyTorch).
Window-based: input is a sliding window of past observations.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDetector


def _build_windows(signal: np.ndarray, W: int) -> np.ndarray:
    """Build overlapping windows of size W, zero-padded at the start."""
    n = len(signal)
    X = np.zeros((n, W), dtype=np.float32)
    for i in range(n):
        start = max(0, i - W + 1)
        length = i - start + 1
        X[i, W - length:] = signal[start: i + 1]
    return X


def _train(model, X_t, y_t, epochs=30, lr=1e-3, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()
    model.eval()


class _MLPNet(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class _ANNNet(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class _CNNNet(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * W, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)


class _LSTMNet(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class _WindowDetector(BaseDetector):
    """Base for all window-based neural detectors."""

    def __init__(self, Tb: int, scale: int):
        self.W = max(10, Tb * scale)
        self.model = None
        self._mean = 0.0
        self._std = 1.0

    def _preprocess(self, signal):
        return (signal - self._mean) / (self._std + 1e-8)

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        self._mean = signal.mean()
        self._std = signal.std()
        sig = self._preprocess(signal)
        X = _build_windows(sig, self.W)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(bits, dtype=torch.float32)
        self.model = self._build_model()
        _train(self.model, X_t, y_t)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        sig = self._preprocess(signal)
        X = _build_windows(sig, self.W)
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).squeeze().numpy()
        return (probs >= 0.5).astype(int)

    def _build_model(self):
        raise NotImplementedError


class FeedforwardMLP(_WindowDetector):
    name = "Feedforward-MLP"

    def __init__(self, Tb: int = 100):
        super().__init__(Tb, scale=10)

    def _build_model(self):
        return _MLPNet(self.W)


class ANNDetector(_WindowDetector):
    name = "ANN"

    def __init__(self, Tb: int = 100):
        super().__init__(Tb, scale=10)

    def _build_model(self):
        return _ANNNet(self.W)


class CNNDetector(_WindowDetector):
    name = "CNN"

    def __init__(self, Tb: int = 100):
        super().__init__(Tb, scale=10)

    def _build_model(self):
        return _CNNNet(self.W)


class LSTMDetector(BaseDetector):
    name = "LSTM"

    def __init__(self, hidden: int = 16, seq_len: int = 10):
        self.hidden = hidden
        self.seq_len = seq_len
        self.model = None
        self._mean = 0.0
        self._std = 1.0

    def _preprocess(self, signal):
        return (signal - self._mean) / (self._std + 1e-8)

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        self._mean = signal.mean()
        self._std = signal.std()
        sig = self._preprocess(signal)
        X = _build_windows(sig, self.seq_len)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(bits, dtype=torch.float32)
        self.model = _LSTMNet(self.hidden)
        _train(self.model, X_t, y_t)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        sig = self._preprocess(signal)
        X = _build_windows(sig, self.seq_len)
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).squeeze().numpy()
        return (probs >= 0.5).astype(int)
