"""Utilities for plotting training learning curves saved by train_dqn.py."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_curve(csv_path: str) -> dict:
    """Load a learning-curve CSV produced by train_dqn.save_curve().

    Expected columns: episode, win_rate, loss_rate, draw_rate
    Returns a dict of lists keyed by column name.
    """
    data = {"episode": [], "win_rate": [], "loss_rate": [], "draw_rate": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["win_rate"].append(float(row["win_rate"]))
            data["loss_rate"].append(float(row["loss_rate"]))
            data["draw_rate"].append(float(row["draw_rate"]))
    return data


def plot_learning_curve(csv_path: str, save_path: str = None, title: str = None):
    """Plot win / loss / draw rates from a training curve CSV.

    Args:
        csv_path:  path to the CSV file written by train_dqn.
        save_path: if given, save the figure to this path (PNG/PDF/…).
                   Otherwise the figure is displayed interactively.
        title:     optional plot title (defaults to the CSV filename).
    """
    data = load_curve(csv_path)
    episodes  = np.array(data["episode"])
    win_rates  = np.array(data["win_rate"])
    loss_rates = np.array(data["loss_rate"])
    draw_rates = np.array(data["draw_rate"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, win_rates,  label="Win rate",  color="tab:green")
    ax.plot(episodes, loss_rates, label="Loss rate", color="tab:red")
    ax.plot(episodes, draw_rates, label="Draw rate", color="tab:blue")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (rolling window)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title or os.path.basename(csv_path))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_phases(checkpoint_dir: str = "checkpoints", save_path: str = None):
    """Plot learning curves for all phases found in checkpoint_dir."""
    curves = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("curve_dqn_phase") and f.endswith(".csv")
    ])
    if not curves:
        print("No curve_dqn_phase*.csv files found in", checkpoint_dir)
        return

    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 5), sharey=True)
    if len(curves) == 1:
        axes = [axes]

    for ax, fname in zip(axes, curves):
        data = load_curve(os.path.join(checkpoint_dir, fname))
        episodes  = np.array(data["episode"])
        ax.plot(episodes, data["win_rate"],  label="Win",  color="tab:green")
        ax.plot(episodes, data["loss_rate"], label="Loss", color="tab:red")
        ax.plot(episodes, data["draw_rate"], label="Draw", color="tab:blue")
        ax.set_title(fname.replace("curve_", "").replace(".csv", ""))
        ax.set_xlabel("Episode")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Rate (rolling window)")
    fig.suptitle("Training learning curves — all phases")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_learning_curve(sys.argv[1], save_path=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        plot_all_phases()
