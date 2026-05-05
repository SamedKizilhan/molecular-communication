"""
Hybrid ISI-Cancel + RC detector (novel project extension).

Combines:
  1. Meng 2014 ISI Cancellation as a preprocessing filter
     (z_k = u_k - u_{k-1}) to remove the slowly-varying ISI baseline.
  2. RC-ISI (Bilge 2025) on the cleaned signal for classification.

Hypothesis: physics-informed preprocessing reduces the dynamic range
the reservoir must cover, improving accuracy at severe ISI (low Tb).
"""
import numpy as np
from sklearn.metrics import roc_curve
from .base import BaseDetector
from .reservoir import EchoStateNetwork
from ..preprocessing.isi_cancellation import isi_cancel


class HybridISICancelRC(BaseDetector):
    """ISI cancellation (Meng 2014) followed by RC-ISI (Bilge 2025)."""
    name = "Hybrid-ISICancel+RC"

    def __init__(self, val_ratio: float = 0.3, **esn_kwargs):
        self.esn = EchoStateNetwork(**esn_kwargs)
        self.val_ratio = val_ratio
        self.threshold = 0.5
        self._mean = 0.0
        self._std = 1.0

    def _process(self, signal: np.ndarray) -> np.ndarray:
        """Apply ISI cancel then z-score normalize."""
        z = isi_cancel(signal)
        return (z - self._mean) / (self._std + 1e-8)

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        z = isi_cancel(signal)
        self._mean = z.mean()
        self._std = z.std()
        norm = self._process(signal)

        split = int(len(norm) * (1 - self.val_ratio))
        self.esn.fit(norm[:split], bits[:split].astype(float))

        raw_val = self.esn.transform(norm[split:])
        bits_val = bits[split:]
        fpr, tpr, thresholds = roc_curve(bits_val, raw_val)
        fnr = 1 - tpr
        ber = 0.5 * (fpr + fnr)
        best_idx = np.argmin(ber)
        self.threshold = float(thresholds[best_idx])

        # Retrain on full set
        self.esn.fit(norm, bits.astype(float))

    def predict(self, signal: np.ndarray) -> np.ndarray:
        norm = self._process(signal)
        raw = self.esn.transform(norm)
        return (raw >= self.threshold).astype(int)
