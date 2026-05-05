"""Evaluation utilities: BER, accuracy, ROC, latency measurement."""
import time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def ber(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    return float((bits_true != bits_pred).mean())


def accuracy(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    return float((bits_true == bits_pred).mean())


def measure_latency(detector, signal: np.ndarray, n_repeats: int = 5) -> float:
    """Return per-symbol inference latency in microseconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        detector.predict(signal)
        t1 = time.perf_counter()
        times.append((t1 - t0) / len(signal) * 1e6)
    return float(np.median(times))


def roc_auc(bits_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(bits_true, scores))
    except Exception:
        return float("nan")


def benchmark_detector(
    detector,
    train_signal: np.ndarray,
    train_bits: np.ndarray,
    test_signal: np.ndarray,
    test_bits: np.ndarray,
) -> dict:
    detector.fit(train_signal, train_bits)
    preds = detector.predict(test_signal)
    latency = measure_latency(detector, test_signal)
    return {
        "name": detector.name,
        "accuracy": accuracy(test_bits, preds),
        "ber": ber(test_bits, preds),
        "latency_us": latency,
        "n_params": _count_params(detector),
    }


def _count_params(detector) -> int:
    """Count trainable parameters (works for RC and PyTorch models)."""
    # RC / Ridge readout
    if hasattr(detector, "esn") and detector.esn.readout is not None:
        r = detector.esn.readout
        n = r.coef_.size + r.intercept_.size
        return int(n)
    # PyTorch model
    if hasattr(detector, "model") and detector.model is not None:
        return int(sum(p.numel() for p in detector.model.parameters()))
    return 0
