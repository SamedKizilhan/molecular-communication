"""Base detector interface."""
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    name: str = "BaseDetector"

    @abstractmethod
    def fit(self, signal: np.ndarray, bits: np.ndarray) -> None:
        """Train on signal/label pairs."""

    @abstractmethod
    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Return binary predictions."""

    def score(self, signal: np.ndarray, bits: np.ndarray) -> float:
        preds = self.predict(signal)
        return float((preds == bits).mean())
