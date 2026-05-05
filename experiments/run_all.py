"""
Main benchmark: replicates Bilge 2025 Table II and adds Hybrid-ISICancel+RC.

Usage:
    python experiments/run_all.py [--fast] [--tb 10 50 100 200]

--fast  : use fewer bits (500 instead of 2000) for quick smoke-test
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.simulation.channel import MobileChannel
from src.simulation.signal_generator import generate_bits, train_test_split
from src.detectors.classical import FixedThreshold, AdaptiveEMA, MAPDetector
from src.detectors.ml_detectors import FeedforwardMLP, ANNDetector, CNNDetector, LSTMDetector
from src.detectors.reservoir import RCDetector, RCISIDetector
from src.detectors.hybrid import HybridISICancelRC
from src.evaluation.metrics import benchmark_detector


def make_detectors(Tb: int):
    return [
        FixedThreshold(),
        AdaptiveEMA(),
        MAPDetector(memory=5),
        FeedforwardMLP(Tb=Tb),
        ANNDetector(Tb=Tb),
        CNNDetector(Tb=Tb),
        LSTMDetector(seq_len=10),
        RCDetector(),
        RCISIDetector(),
        HybridISICancelRC(),
    ]


def run_experiment(Tb: int, n_bits: int, seed: int = 42) -> list:
    print(f"\n--- Tb={Tb}s | {n_bits} bits ---")
    bits = generate_bits(n_bits, seed=seed)
    channel = MobileChannel(seed=seed)
    print("  Simulating channel...", end=" ", flush=True)
    signal = channel.simulate(bits, Tb)
    print("done.")

    tr_sig, tr_bits, te_sig, te_bits = train_test_split(signal, bits, 0.7)

    results = []
    for det in make_detectors(Tb):
        try:
            res = benchmark_detector(det, tr_sig, tr_bits, te_sig, te_bits)
            res["Tb"] = Tb
            results.append(res)
            print(f"  {det.name:30s}  acc={res['accuracy']:.3f}  "
                  f"BER={res['ber']:.3f}  lat={res['latency_us']:.2f}µs  "
                  f"params={res['n_params']}")
        except Exception as e:
            print(f"  {det.name:30s}  ERROR: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tb", nargs="+", type=int,
                        default=[10, 30, 50, 70, 90, 100, 200])
    args = parser.parse_args()

    n_bits = 500 if args.fast else 2000
    all_results = []

    for Tb in args.tb:
        all_results.extend(run_experiment(Tb, n_bits))

    df = pd.DataFrame(all_results)
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    pivot = df.pivot_table(
        index="name", columns="Tb", values="accuracy", aggfunc="mean"
    ).round(3)
    print("\nAccuracy summary:")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
