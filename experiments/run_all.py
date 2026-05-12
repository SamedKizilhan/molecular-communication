"""
Main benchmark: replicates Bilge 2025 Table II and adds Hybrid-ISICancel+RC.

Usage:
    python experiments/run_all.py [--fast] [--tb 10 50 100 200] [--smart-tx]

--fast      : use fewer bits (500 instead of 2000) for quick smoke-test
--smart-tx  : also run experiments with ISI-aware smart transmitter
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


def run_experiment(
    Tb: int,
    n_bits: int,
    seed: int = 42,
    smart_tx: bool = False,
    gamma: float = 0.8,
) -> list:
    mode = "Smart-TX" if smart_tx else "Normal"
    print(f"\n--- Tb={Tb}s | {n_bits} bits | TX={mode} ---")
    bits = generate_bits(n_bits, seed=seed)
    channel = MobileChannel(seed=seed)
    print("  Simulating channel...", end=" ", flush=True)
    signal = channel.simulate(bits, Tb, smart_tx=smart_tx, gamma=gamma)
    print("done.")

    tr_sig, tr_bits, te_sig, te_bits = train_test_split(signal, bits, 0.7)

    results = []
    for det in make_detectors(Tb):
        try:
            res = benchmark_detector(det, tr_sig, tr_bits, te_sig, te_bits)
            res["Tb"] = Tb
            res["tx_mode"] = mode
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
    parser.add_argument("--smart-tx", action="store_true",
                        help="Also run with ISI-aware smart transmitter")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Smart-TX aggressiveness (default: 0.8)")
    args = parser.parse_args()

    n_bits = 500 if args.fast else 2000
    all_results = []

    for Tb in args.tb:
        # Always run normal TX
        all_results.extend(run_experiment(Tb, n_bits))
        # Optionally run smart TX
        if args.smart_tx:
            all_results.extend(
                run_experiment(Tb, n_bits, smart_tx=True, gamma=args.gamma)
            )

    df = pd.DataFrame(all_results)
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    if args.smart_tx:
        for mode in ["Normal", "Smart-TX"]:
            sub = df[df["tx_mode"] == mode]
            pivot = sub.pivot_table(
                index="name", columns="Tb", values="accuracy", aggfunc="mean"
            ).round(3)
            print(f"\nAccuracy summary ({mode}):")
            print(pivot.to_string())
    else:
        pivot = df.pivot_table(
            index="name", columns="Tb", values="accuracy", aggfunc="mean"
        ).round(3)
        print("\nAccuracy summary:")
        print(pivot.to_string())


if __name__ == "__main__":
    main()

