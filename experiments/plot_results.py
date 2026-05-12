"""
Generate figures from results.csv:
  - Fig 1: Accuracy vs Tb (replication of Bilge 2025 Fig. 4)
  - Fig 2: Latency vs Tb (replication of Bilge 2025 Fig. 5)
  - Fig 3: BER vs Tb comparison
  - Fig 4: Normal vs Smart-TX accuracy (when --smart-tx data present)
  - Fig 5: Smart-TX accuracy improvement delta

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


def _plot_metric(df: pd.DataFrame, out_dir: str, metric: str, ylabel: str,
                 title: str, filename: str, log_y: bool = False,
                 subset: set = None, suffix: str = ""):
    """Generic helper for single-mode metric plots."""
    sub = df if subset is None else df[df["name"].isin(subset)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in sub.groupby("name"):
        grp = grp.sort_values("Tb")
        plot_fn = ax.semilogy if log_y else ax.plot
        y_vals = grp[metric].clip(1e-4) if log_y else grp[metric]
        plot_fn(
            grp["Tb"], y_vals,
            label=name,
            color=COLORS.get(name, None),
            marker=MARKERS.get(name, "o"),
            linewidth=1.5,
        )
    ax.set_xlabel("Symbol time $T_b$ (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title + suffix)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_accuracy(df: pd.DataFrame, out_dir: str, suffix: str = ""):
    fname = "fig_accuracy.png" if not suffix else f"fig_accuracy_{suffix}.png"
    _plot_metric(df, out_dir, "accuracy", "Accuracy",
                 "Detection Accuracy vs Symbol Interval", fname,
                 suffix=f" ({suffix})" if suffix else "")


def fig_latency(df: pd.DataFrame, out_dir: str, suffix: str = ""):
    ml_names = {"RC-ISI", "RC", "ANN", "Feedforward-MLP", "CNN", "LSTM",
                "Hybrid-ISICancel+RC"}
    fname = "fig_latency.png" if not suffix else f"fig_latency_{suffix}.png"
    _plot_metric(df, out_dir, "latency_us", "µs / symbol (log scale)",
                 "Inference Latency vs Symbol Interval", fname,
                 log_y=True, subset=ml_names,
                 suffix=f" ({suffix})" if suffix else "")


def fig_ber(df: pd.DataFrame, out_dir: str, suffix: str = ""):
    fname = "fig_ber.png" if not suffix else f"fig_ber_{suffix}.png"
    _plot_metric(df, out_dir, "ber", "BER (log scale)",
                 "Bit Error Rate vs Symbol Interval", fname, log_y=True,
                 suffix=f" ({suffix})" if suffix else "")


# ── Comparison plots (Normal vs Smart-TX) ────────────────────────────

def fig_comparison_accuracy(df: pd.DataFrame, out_dir: str):
    """Side-by-side accuracy: Normal vs Smart-TX for key detectors."""
    key_detectors = ["RC-ISI", "Hybrid-ISICancel+RC", "Adaptive-EMA",
                     "RC", "Feedforward-MLP", "LSTM"]
    sub = df[df["name"].isin(key_detectors)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, ["Normal", "Smart-TX"]):
        mode_df = sub[sub["tx_mode"] == mode]
        for name, grp in mode_df.groupby("name"):
            grp = grp.sort_values("Tb")
            ax.plot(grp["Tb"], grp["accuracy"], label=name,
                    color=COLORS.get(name), marker=MARKERS.get(name, "o"),
                    linewidth=1.5)
        ax.set_xlabel("Symbol time $T_b$ (s)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy — {mode} TX")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    fig.suptitle("Normal TX vs Smart TX — Accuracy Comparison", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_comparison_accuracy.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_delta_improvement(df: pd.DataFrame, out_dir: str):
    """Bar chart showing accuracy improvement (Smart-TX − Normal) per detector per Tb."""
    normal = df[df["tx_mode"] == "Normal"].set_index(["name", "Tb"])["accuracy"]
    smart = df[df["tx_mode"] == "Smart-TX"].set_index(["name", "Tb"])["accuracy"]
    delta = (smart - normal).reset_index()
    delta.columns = ["name", "Tb", "delta"]

    tb_values = sorted(delta["Tb"].unique())
    detectors = sorted(delta["name"].unique())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(detectors))
    width = 0.8 / len(tb_values)

    for i, tb in enumerate(tb_values):
        vals = []
        for det in detectors:
            row = delta[(delta["name"] == det) & (delta["Tb"] == tb)]
            vals.append(row["delta"].values[0] if len(row) else 0)
        bars = ax.bar(x + i * width, [v * 100 for v in vals], width,
                      label=f"Tb={tb}s", alpha=0.8)
        # Color bars: green for positive, red for negative
        for bar, v in zip(bars, vals):
            bar.set_color("tab:green" if v >= 0 else "tab:red")

    ax.set_xlabel("Detector")
    ax.set_ylabel("Accuracy Improvement (pp)")
    ax.set_title("Smart TX Improvement over Normal TX (percentage points)")
    ax.set_xticks(x + width * (len(tb_values) - 1) / 2)
    ax.set_xticklabels(detectors, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=7, title="Tb")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_delta_improvement.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_comparison_ber(df: pd.DataFrame, out_dir: str):
    """Side-by-side BER: Normal vs Smart-TX for key detectors."""
    key_detectors = ["RC-ISI", "Hybrid-ISICancel+RC", "Adaptive-EMA",
                     "RC", "Feedforward-MLP", "LSTM"]
    sub = df[df["name"].isin(key_detectors)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, ["Normal", "Smart-TX"]):
        mode_df = sub[sub["tx_mode"] == mode]
        for name, grp in mode_df.groupby("name"):
            grp = grp.sort_values("Tb")
            ax.semilogy(grp["Tb"], grp["ber"].clip(1e-4), label=name,
                        color=COLORS.get(name), marker=MARKERS.get(name, "o"),
                        linewidth=1.5)
        ax.set_xlabel("Symbol time $T_b$ (s)")
        ax.set_ylabel("BER (log scale)")
        ax.set_title(f"BER — {mode} TX")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()

    fig.suptitle("Normal TX vs Smart TX — BER Comparison", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_comparison_ber.png")
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

    has_smart_tx = "tx_mode" in df.columns and "Smart-TX" in df["tx_mode"].values

    if has_smart_tx:
        # Generate separate plots per mode
        for mode in ["Normal", "Smart-TX"]:
            mode_df = df[df["tx_mode"] == mode]
            fig_accuracy(mode_df, out_dir, suffix=mode)
            fig_ber(mode_df, out_dir, suffix=mode)
        fig_latency(df[df["tx_mode"] == "Normal"], out_dir)

        # Generate comparison plots
        fig_comparison_accuracy(df, out_dir)
        fig_comparison_ber(df, out_dir)
        fig_delta_improvement(df, out_dir)
    else:
        fig_accuracy(df, out_dir)
        fig_latency(df, out_dir)
        fig_ber(df, out_dir)


if __name__ == "__main__":
    main()

