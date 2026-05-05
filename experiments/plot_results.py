"""
Generate figures from results.csv:
  - Fig 1: Accuracy vs Tb (replication of Bilge 2025 Fig. 4)
  - Fig 2: Latency vs Tb (replication of Bilge 2025 Fig. 5)
  - Fig 3: BER vs Tb comparison

Usage:
    python experiments/plot_results.py [--csv path/to/results.csv]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "RC-ISI": "tab:blue",
    "RC": "tab:orange",
    "ANN": "tab:green",
    "Feedforward-MLP": "tab:red",
    "CNN": "tab:purple",
    "Adaptive-EMA": "tab:brown",
    "LSTM": "tab:pink",
    "MAP (Mismatched)": "tab:gray",
    "Peak-Fixed": "black",
    "Hybrid-ISICancel+RC": "tab:cyan",
}
MARKERS = {
    "RC-ISI": "o",
    "RC": "s",
    "ANN": "^",
    "Feedforward-MLP": "D",
    "CNN": "v",
    "Adaptive-EMA": "x",
    "LSTM": "*",
    "MAP (Mismatched)": "+",
    "Peak-Fixed": ".",
    "Hybrid-ISICancel+RC": "P",
}


def load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def fig_accuracy(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby("name"):
        grp = grp.sort_values("Tb")
        ax.plot(
            grp["Tb"], grp["accuracy"],
            label=name,
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            linewidth=1.5,
        )
    ax.set_xlabel("Symbol time $T_b$ (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection Accuracy vs Symbol Interval")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_accuracy.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_latency(df: pd.DataFrame, out_dir: str):
    ml_names = {"RC-ISI", "RC", "ANN", "Feedforward-MLP", "CNN", "LSTM", "Hybrid-ISICancel+RC"}
    sub = df[df["name"].isin(ml_names)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in sub.groupby("name"):
        grp = grp.sort_values("Tb")
        ax.semilogy(
            grp["Tb"], grp["latency_us"],
            label=name,
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            linewidth=1.5,
        )
    ax.set_xlabel("Symbol time $T_b$ (s)")
    ax.set_ylabel("µs / symbol (log scale)")
    ax.set_title("Inference Latency vs Symbol Interval")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_latency.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_ber(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby("name"):
        grp = grp.sort_values("Tb")
        ax.semilogy(
            grp["Tb"], grp["ber"].clip(1e-4),
            label=name,
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            linewidth=1.5,
        )
    ax.set_xlabel("Symbol time $T_b$ (s)")
    ax.set_ylabel("BER (log scale)")
    ax.set_title("Bit Error Rate vs Symbol Interval")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_ber.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        os.path.dirname(__file__), "results.csv"))
    args = parser.parse_args()

    df = load(args.csv)
    out_dir = os.path.dirname(args.csv)
    fig_accuracy(df, out_dir)
    fig_latency(df, out_dir)
    fig_ber(df, out_dir)


if __name__ == "__main__":
    main()
