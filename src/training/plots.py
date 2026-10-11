"""Headless plots generated from saved CSV metrics, including empty runs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window=25):
    """Trailing average with partial windows at the start of a series."""
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    sums = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (sums[ends] - sums[starts]) / (ends - starts)


def generate_plots(output_dir, window=25):
    output_dir = Path(output_dir)
    with (output_dir / "training_metrics.csv").open(newline="", encoding="utf-8") as stream:
        training = list(csv.DictReader(stream))
    with (output_dir / "validation_metrics.csv").open(newline="", encoding="utf-8") as stream:
        validation = list(csv.DictReader(stream))
    with (output_dir / "training_config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)

    def curve(filename, rows, x_key, y_key, title, xlabel, ylabel, smooth=False):
        usable = [row for row in rows if row.get(y_key) not in (None, "")]
        fig, axis = plt.subplots(figsize=(8, 4.5))
        try:
            if usable:
                xs = [float(row[x_key]) for row in usable]
                ys = [float(row[y_key]) for row in usable]
                axis.plot(xs, ys, alpha=0.3 if smooth else 1,
                          marker=None if smooth else ".", label="Recorded")
                if smooth:
                    axis.plot(xs, moving_average(ys, window), label=f"Trailing mean ({window} episodes)")
                    axis.legend()
            else:
                axis.text(0.5, 0.5, "No optimizer updates" if y_key == "mean_loss"
                          else "No training episodes", ha="center", va="center", transform=axis.transAxes)
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=150)
        finally:
            plt.close(fig)

    curve("training_reward.png", training, "episode", "reward", "Training episode reward",
          "Completed training episodes", "Reward (negative total waiting time, CPU time units)", True)
    curve("training_waiting_time.png", training, "episode", "mean_waiting_time",
          "Training mean waiting time", "Completed training episodes", "Waiting time per process (CPU time units)", True)
    curve("validation_waiting_time.png", validation, "episode", "mean_waiting_time",
          "Greedy validation on fixed workloads", "Completed training episodes",
          "Waiting time per process (CPU time units)")
    curve("dqn_loss.png", training, "episode", "mean_loss", "DQN loss (episode mean over optimizer updates)",
          "Completed training episodes", "Smooth L1 loss", True)
    epsilon_rows = [{"environment_steps": 0, "epsilon": config["initial_epsilon"]}, *training]
    curve("epsilon_decay.png", epsilon_rows, "environment_steps", "epsilon", "Linear exploration schedule",
          "Completed environment steps", "Epsilon (exploration probability)")
