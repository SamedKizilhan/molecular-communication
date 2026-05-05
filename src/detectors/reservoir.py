"""
Reservoir Computing detector (Echo State Network) for molecular communications.
Implements both RC and RC-ISI variants from Bilge et al. 2025.

RC-ISI uses ROC-based threshold optimization instead of fixed 0.5.
Hyperparameters from the paper: Nr=400, rho=0.7, alpha=0.3, sin=1.0, Twash=300.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve
from .base import BaseDetector


class EchoStateNetwork:
    """
    Leaky-Integrator Echo State Network (LI-ESN).
    Only the readout weights are trained; reservoir is fixed random.
    """

    def __init__(
        self,
        Nr: int = 400,
        spectral_radius: float = 0.7,
        leaky_rate: float = 0.3,
        input_scaling: float = 1.0,
        washout: int = 300,
        ridge_alpha: float = 1e-4,
        seed: int = 42,
    ):
        self.Nr = Nr
        self.rho = spectral_radius
        self.alpha = leaky_rate
        self.sin = input_scaling
        self.washout = washout
        self.ridge_alpha = ridge_alpha
        rng = np.random.default_rng(seed)

        # Random input weights Win: (Nr, 1)
        self.Win = rng.uniform(-self.sin, self.sin, (Nr, 1))

        # Random sparse reservoir weights Wres: (Nr, Nr), connectivity ~0.1
        density = 0.1
        Wres = rng.standard_normal((Nr, Nr))
        mask = rng.random((Nr, Nr)) > density
        Wres[mask] = 0.0
        # Scale to desired spectral radius
        eigvals = np.linalg.eigvals(Wres)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-10:
            Wres = Wres * (self.rho / sr)
        self.Wres = Wres

        self.readout = None  # Ridge regression model

    def _run_reservoir(self, inputs: np.ndarray) -> np.ndarray:
        """Drive reservoir with input sequence; return state matrix."""
        T = len(inputs)
        x = np.zeros(self.Nr)
        states = np.zeros((T, self.Nr))
        for k in range(T):
            u = np.array([[inputs[k]]])
            x = (1 - self.alpha) * x + self.alpha * np.tanh(
                self.Wres @ x + (self.Win @ u).ravel()
            )
            states[k] = x
        return states

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        states = self._run_reservoir(inputs)
        # Washout: cap at 20% of training data to avoid consuming all samples
        w = min(self.washout, max(1, len(inputs) // 5))
        S = states[w:]
        y = targets[w:]
        self.readout = Ridge(alpha=self.ridge_alpha)
        self.readout.fit(S, y)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        """Return raw regression output (continuous, not thresholded)."""
        states = self._run_reservoir(inputs)
        return self.readout.predict(states)


class RCDetector(BaseDetector):
    """Standard RC detector with fixed 0.5 threshold."""
    name = "RC"

    def __init__(self, **esn_kwargs):
        self.esn = EchoStateNetwork(**esn_kwargs)
        self._mean = 0.0
        self._std = 1.0

    def _normalize(self, signal):
        return (signal - self._mean) / (self._std + 1e-8)

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        self._mean = signal.mean()
        self._std = signal.std()
        self.esn.fit(self._normalize(signal), bits.astype(float))

    def predict(self, signal: np.ndarray) -> np.ndarray:
        raw = self.esn.transform(self._normalize(signal))
        return (raw >= 0.5).astype(int)


class RCISIDetector(BaseDetector):
    """
    RC-ISI detector: RC + ROC-based threshold optimization.
    The optimal threshold minimizes BER on the validation portion of
    training data.
    """
    name = "RC-ISI"

    def __init__(self, val_ratio: float = 0.3, **esn_kwargs):
        self.esn = EchoStateNetwork(**esn_kwargs)
        self.val_ratio = val_ratio
        self.threshold = 0.5
        self._mean = 0.0
        self._std = 1.0

    def _normalize(self, signal):
        return (signal - self._mean) / (self._std + 1e-8)

    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        self._mean = signal.mean()
        self._std = signal.std()
        norm = self._normalize(signal)

        # Split into train / validation for threshold optimization
        split = int(len(norm) * (1 - self.val_ratio))
        self.esn.fit(norm[:split], bits[:split].astype(float))

        # Get ROC on validation set
        raw_val = self.esn.transform(norm[split:])
        bits_val = bits[split:]
        fpr, tpr, thresholds = roc_curve(bits_val, raw_val)
        # Choose threshold that minimizes BER = 0.5*(FPR + FNR)
        fnr = 1 - tpr
        ber = 0.5 * (fpr + fnr)
        best_idx = np.argmin(ber)
        self.threshold = float(thresholds[best_idx])

        # Re-train on full set with the tuned threshold
        self.esn.fit(norm, bits.astype(float))

    def predict(self, signal: np.ndarray) -> np.ndarray:
        raw = self.esn.transform(self._normalize(signal))
        return (raw >= self.threshold).astype(int)
