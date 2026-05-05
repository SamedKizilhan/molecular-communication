"""
Classical detectors:
  - FixedThreshold (Peak-Fixed)
  - AdaptiveEMA
  - MAPDetector (mismatched, static average CIR)
"""
import numpy as np
from scipy.signal import find_peaks
from .base import BaseDetector


class FixedThreshold(BaseDetector):
    name = "Peak-Fixed"

    def __init__(self):
        self.threshold = 0.5

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        # Grid search for best fixed threshold on training data
        best_acc, best_th = 0.0, 0.5
        for th in np.linspace(signal.min(), signal.max(), 200):
            acc = ((signal >= th).astype(int) == bits).mean()
            if acc > best_acc:
                best_acc, best_th = acc, th
        self.threshold = best_th

    def predict(self, signal: np.ndarray) -> np.ndarray:
        return (signal >= self.threshold).astype(int)


class AdaptiveEMA(BaseDetector):
    """Exponential Moving Average adaptive threshold (Damrath & Hoeher 2016)."""
    name = "Adaptive-EMA"

    def __init__(self, beta: float = 0.9, eta: float = 0.0):
        self.beta = beta
        self.eta = eta  # fixed offset above baseline

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        # Tune beta and eta via grid search on training data
        best_acc = 0.0
        for beta in [0.7, 0.8, 0.9, 0.95, 0.99]:
            for eta in np.linspace(0, signal.std(), 10):
                preds = self._run(signal, beta, eta)
                acc = (preds == bits).mean()
                if acc > best_acc:
                    best_acc, self.beta, self.eta = acc, beta, eta

    def _run(self, signal: np.ndarray, beta: float, eta: float) -> np.ndarray:
        I = 0.0
        preds = np.zeros(len(signal), dtype=int)
        for k, u in enumerate(signal):
            if k > 0:
                I = beta * I + (1 - beta) * signal[k - 1]
            preds[k] = int(u > eta + I)
        return preds

    def predict(self, signal: np.ndarray) -> np.ndarray:
        return self._run(signal, self.beta, self.eta)


class MAPDetector(BaseDetector):
    """
    Mismatched MAP detector using Viterbi with a static average CIR
    estimated from training data (Bilge 2025, Section III-C).
    """
    name = "MAP (Mismatched)"

    def __init__(self, memory: int = 5):
        self.memory = memory
        self.cir = None       # average CIR taps
        self.sigma = None

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        # Estimate average CIR taps by simple regression on training data
        # Build Toeplitz feature matrix for bit sequence
        n = len(bits)
        m = self.memory
        X = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if i - j >= 0:
                    X[i, j] = bits[i - j]
        # Least-squares estimate of CIR taps
        self.cir, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
        residuals = signal - X @ self.cir
        self.sigma = residuals.std() + 1e-8

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if self.cir is None:
            raise RuntimeError("Call fit() before predict()")
        n = len(signal)
        m = self.memory
        preds = np.zeros(n, dtype=int)
        # Greedy symbol-by-symbol MAP decision
        estimated_bits = []
        for k in range(n):
            # Compute expected signal for bit=0 and bit=1
            isi = sum(
                self.cir[j] * (estimated_bits[k - j - 1] if k - j - 1 >= 0 else 0)
                for j in range(1, min(m, k + 1))
            )
            mu1 = self.cir[0] + isi
            mu0 = isi
            # Likelihood ratio
            lr = ((signal[k] - mu0) ** 2 - (signal[k] - mu1) ** 2) / (2 * self.sigma ** 2)
            bit = int(lr > 0)
            preds[k] = bit
            estimated_bits.append(bit)
        return preds
