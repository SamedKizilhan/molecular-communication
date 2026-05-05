"""
ISI Cancellation preprocessing from Meng et al. 2014.

The technique takes the difference between samples at the peak of the new
signal and the baseline (t=0 of the symbol) to suppress residual ISI from
previous symbols.  Applied to a sampled scalar signal this becomes:
    z_k = u_k - u_{k-1}   (where u_k is the end-of-symbol sample)

This removes the slowly-varying ISI baseline, leaving only the signal
component for the current bit.
"""
import numpy as np


def isi_cancel(signal: np.ndarray) -> np.ndarray:
    """
    Apply ISI cancellation: z[k] = signal[k] - signal[k-1].
    Returns array of same length; z[0] = signal[0].
    """
    out = np.empty_like(signal)
    out[0] = signal[0]
    out[1:] = signal[1:] - signal[:-1]
    return out


def normalize_zscore(signal: np.ndarray, mean: float = None, std: float = None):
    """Z-score normalize.  If mean/std not given, compute from signal."""
    if mean is None:
        mean = signal.mean()
    if std is None:
        std = signal.std() + 1e-8
    return (signal - mean) / std, mean, std
