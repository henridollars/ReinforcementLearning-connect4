"""Utilities for plotting training learning curves saved by train_dqn.py."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_curve(csv_path: str) -> dict:
    """Load a learning-curve CSV produced by train_dqn.

    Expected columns: episode, win_rate, loss_rate, draw_rate, avg_reward, avg_loss
    Returns a dict of lists keyed by column name.
    """
    data = {k: [] for k in ("episode", "win_rate", "loss_rate", "draw_rate",
                             "avg_reward", "avg_loss")}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["win_rate"].append(float(row["win_rate"]))
            data["loss_rate"].append(float(row["loss_rate"]))
            data["draw_rate"].append(float(row["draw_rate"]))
            data["avg_reward"].append(float(row["avg_reward"]))
            data["avg_loss"].append(float(row["avg_loss"]))
    return data


def _add_smoothing(ax, x, y, color, label, alpha_raw=0.25, window=10):
    """Plot raw data faintly and a rolling-mean overlay."""
    ax.plot(x, y, color=color, alpha=alpha_raw)
    if len(y) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(y, kernel, mode="valid")
        x_smooth = x[window - 1:]
        ax.plot(x_smooth, smoothed, color=color, linewidth=2, label=label)
    else:
        ax.lines[-1].set_label(label)
        ax.lines[-1].set_alpha(1.0)


def plot_learning_curve(csv_path: str, save_path: str = None, title: str = None):
    """Plot win rate / avg episode reward / avg loss from a training curve CSV.

    Three vertically stacked subplots share the x-axis (episode).
    A faint raw line plus a smoothed overlay is shown for each metric.

    Args:
        csv_path:  path to the CSV file written by train_dqn.
        save_path: if given, save the figure to this path; otherwise display.
        title:     optional overall figure title.
    """
    data     = load_curve(csv_path)
    episodes = np.array(data["episode"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title or os.path.basename(csv_path), fontsize=13)

    # ── Win rate ──
    _add_smoothing(axes[0], episodes, np.array(data["win_rate"]),
                   color="tab:green", label="Win rate")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # ── Average episode reward ──
    _add_smoothing(axes[1], episodes, np.array(data["avg_reward"]),
                   color="tab:blue", label="Avg episode reward")
    axes[1].set_ylabel("Avg reward")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # ── Average loss ──
    _add_smoothing(axes[2], episodes, np.array(data["avg_loss"]),
                   color="tab:orange", label="Avg loss")
    axes[2].set_ylabel("Avg loss")
    axes[2].set_xlabel("Episode")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_phases(checkpoint_dir: str = "checkpoints", save_path: str = None):
    """Plot all three metrics across all phases in a 3-row × N-phase grid."""
    curves = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("curve_dqn_phase") and f.endswith(".csv")
    ])
    if not curves:
        print("No curve_dqn_phase*.csv files found in", checkpoint_dir)
        return

    n_phases = len(curves)
    fig, axes = plt.subplots(3, n_phases,
                             figsize=(6 * n_phases, 10),
                             sharex="col", sharey="row")
    # Ensure axes is always 2-D
    if n_phases == 1:
        axes = axes.reshape(3, 1)

    metrics = [
        ("win_rate",   "Win rate",          "tab:green",  (0, 1)),
        ("avg_reward", "Avg episode reward", "tab:blue",   None),
        ("avg_loss",   "Avg loss",           "tab:orange", None),
    ]

    for col, fname in enumerate(curves):
        data     = load_curve(os.path.join(checkpoint_dir, fname))
        episodes = np.array(data["episode"])
        phase_title = fname.replace("curve_", "").replace(".csv", "")

        for row, (key, ylabel, color, ylim) in enumerate(metrics):
            ax = axes[row, col]
            _add_smoothing(ax, episodes, np.array(data[key]), color=color, label=ylabel)
            if row == 0:
                ax.set_title(phase_title, fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Episode")
            if ylim:
                ax.set_ylim(*ylim)
            if key == "win_rate":
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            if key == "avg_reward":
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Training learning curves — all phases", fontsize=13)
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
        out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".csv", ".png")
        plot_learning_curve(sys.argv[1], save_path=out)
    else:
        plot_all_phases(save_path="checkpoints/learning_curves.png")
