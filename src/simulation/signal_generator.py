"""
Bit sequence generation and signal utilities.
"""
import numpy as np


def generate_bits(n: int, p_one: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p_one, size=n).astype(np.int32)


def train_test_split(
    signal: np.ndarray, bits: np.ndarray, train_ratio: float = 0.7
):
    split = int(len(bits) * train_ratio)
    return (
        signal[:split], bits[:split],
        signal[split:], bits[split:],
    )
