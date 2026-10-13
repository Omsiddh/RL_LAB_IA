"""Plot logged training behavior separately from greedy validation results."""

import matplotlib.pyplot as plt
import numpy as np

from .data import POLICIES
from .style import COLORS, rolling_mean


RUNS = {POLICIES[2]: "original_10000", POLICIES[1]: "augmented_10000"}
ROOT = "runs/phase6_3b_controlled/"


def generate(data, writer):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = {}
    for policy, folder in RUNS.items():
        bundle = data["training"][policy]
        rows = bundle["validation"]
        episodes = [int(r["episode"]) for r in rows]
        waiting = [float(r["mean_waiting_time"]) for r in rows]
        best = bundle["best"]
        ax.plot(episodes, waiting, color=COLORS[policy], marker=".", linewidth=1.5, label=policy)
        ax.scatter([best["episode"]], [best["validation"]["mean_waiting_time"]], marker="*", s=170, color=COLORS[policy], edgecolor="black", linewidth=.5, zorder=4)
        offset = (14, 30) if policy == POLICIES[2] else (-185, 14)
        ax.annotate(f"Best: episode {best['episode']}\n{best['validation']['mean_waiting_time']:.6f}",
            (best["episode"], best["validation"]["mean_waiting_time"]), xytext=offset, textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": COLORS[policy]},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
        plotted[policy] = {"episodes": episodes, "waiting": waiting, "best_episode": best["episode"]}
    srtf = float(data["permutation_rows"][0]["srtf_waiting"])
    ax.axhline(srtf, linestyle="--", color=COLORS["SRTF"], label="SRTF validation reference")
    ax.set(xlabel="Completed training episodes", ylabel="CPU time units / process", title="Greedy validation waiting during training", xlim=(0, 10200))
    ax.grid()
    ax.legend(fontsize=9)
    writer.save(fig, "validation_learning_curves", "model-selection validation",
        "VALIDATION | 128 fixed workloads / 645 processes, seed 1729 | epsilon = 0; every 250 episodes\nSource: phase6_3b_controlled/{original,augmented}_10000/validation_metrics.csv; stars mark previously selected checkpoints",
        [ROOT + folder + "/validation_metrics.csv" for folder in RUNS.values()] + [ROOT + "permutation_comparison.csv"] + [ROOT + folder + "/best_model.json" for folder in RUNS.values()], plotted)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = {}
    for policy, folder in RUNS.items():
        rows = data["training"][policy]["training"]
        episodes = [int(r["episode"]) for r in rows]
        reward = [float(r["reward"]) for r in rows]
        smooth = rolling_mean(reward, 100)
        ax.plot(episodes, reward, color=COLORS[policy], alpha=.1, linewidth=.45, rasterized=True)
        ax.plot(episodes, smooth, color=COLORS[policy], linewidth=1.3, label=policy)
        plotted[policy] = {"window_episodes": 100, "episodes": episodes, "rolling_reward": smooth}
    ax.set(xlabel="Training episode", ylabel="Episode total reward (negative total waiting)", title="Training reward: exploratory rollouts", xlim=(1, 10000))
    ax.legend(fontsize=9)
    ax.grid()
    writer.save(fig, "training_reward", "training",
        "TRAINING | 10,000 episodes / run, shared seed 42 | faint: raw; solid: trailing 100-episode mean (shorter prefix)\nSource: phase6_3b_controlled/{original,augmented}_10000/training_metrics.csv; these are not greedy validation scores",
        [ROOT + folder + "/training_metrics.csv" for folder in RUNS.values()], plotted)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = {}
    for policy, folder in RUNS.items():
        rows = data["training"][policy]["optimizer"]
        updates = [int(r["optimizer_updates"]) for r in rows]
        losses = [float(r["loss"]) for r in rows]
        smooth = rolling_mean(losses, 250)
        indices = sorted(set(range(0, len(rows), 50)) | {len(rows) - 1})
        x = [updates[i] for i in indices]
        y = [smooth[i] for i in indices]
        ax.plot(x, y, color=COLORS[policy], label=policy, linewidth=1.5)
        plotted[policy] = {"optimizer_updates": x, "loss": y, "trailing_window_updates": 250, "display_stride": 50, "total_updates": len(rows)}
    ax.set(xlabel="Optimizer update", ylabel="Smooth L1 TD loss (250-update trailing mean)", title="DQN training loss", ylim=(0, None))
    ax.legend(fontsize=9)
    ax.grid()
    writer.save(fig, "optimizer_loss", "training",
        "TRAINING | 68,326 updates / run | trailing 250-update mean, displayed every 50 updates (plus final); shorter prefix\nSource: phase6_3b_controlled/{original,augmented}_10000/optimizer_diagnostics.csv; decreasing loss is not proof of convergence",
        [ROOT + folder + "/optimizer_diagnostics.csv" for folder in RUNS.values()], plotted)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    series = []
    for policy in RUNS:
        rows = data["training"][policy]["training"]
        series.append(([int(r["environment_steps"]) for r in rows], [float(r["epsilon"]) for r in rows]))
    if series[0] != series[1]:
        raise ValueError("Training epsilon curves expected to match; saved histories disagree")
    ax.plot(*series[0], color=COLORS[POLICIES[2]], linewidth=2.5, label="Original 10k")
    ax.plot(*series[1], color=COLORS[POLICIES[1]], linestyle="--", linewidth=1.5, label="Augmented 10k (same schedule)")
    ax.set(xlabel="Environment CPU steps", ylabel="Training exploration probability ε", title="Logged training epsilon schedule", ylim=(0, 1.05), xlim=(0, series[0][0][-1]))
    ax.grid()
    ax.legend(fontsize=9)
    writer.save(fig, "epsilon_schedule", "training",
        "TRAINING | 10,000 episodes / 274,300 CPU steps per run; end-of-episode values | Validation epsilon is always zero\nSource: phase6_3b_controlled/{original,augmented}_10000/training_metrics.csv",
        [ROOT + folder + "/training_metrics.csv" for folder in RUNS.values()], {"environment_steps": series[0][0], "epsilon": series[0][1], "identical_for_both_runs": True})
