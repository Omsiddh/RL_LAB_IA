"""Comparison figures from complete training histories and shared evaluations."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.training.plots import moving_average


COLORS = {"reference_5250": "#777777", "original_10000_best": "#247ba0", "augmented_10000_best": "#d65d0e"}
LABELS = {"reference_5250": "Reference (6k run)", "original_10000_best": "A: original 10k", "augmented_10000_best": "B: augmented 10k"}


def comparison_plots(root, summary):
    histories = {}
    for name, folder in (("original_10000_best", "original_10000"), ("augmented_10000_best", "augmented_10000")):
        histories[name] = {}
        for kind in ("training_metrics", "validation_metrics", "optimizer_diagnostics"):
            with (root / folder / (kind + ".csv")).open(newline="") as stream:
                histories[name][kind] = list(csv.DictReader(stream))
        histories[name]["health"] = json.loads((root / folder / "validation_health.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    try:
        for name, data in histories.items():
            color, label = COLORS[name], LABELS[name]
            validation, training, updates = (data[key] for key in ("validation_metrics", "training_metrics", "optimizer_diagnostics"))
            axes[0, 0].plot([int(r["episode"]) for r in validation], [float(r["mean_waiting_time"]) for r in validation],
                            marker=".", label=label, color=color)
            axes[0, 1].plot([int(r["episode"]) for r in training], moving_average([float(r["mean_waiting_time"]) for r in training], 200),
                            label=label, color=color)
            axes[1, 0].plot([int(r["optimizer_updates"]) for r in updates], moving_average([float(r["loss"]) for r in updates], 500),
                            label=label, color=color)
            axes[1, 1].plot([int(r["environment_steps"]) for r in training], [float(r["epsilon"]) for r in training], label=label, color=color)
        axes[0, 0].axhline(6.006201550387597, color="#238b45", linestyle="--", label="SRTF validation")
        axes[0, 0].axhline(6.586046511627907, color="#777777", linestyle=":", label="Reference best")
        axes[0, 0].set(title="Original-layout greedy validation every 250 episodes", xlabel="Training episode", ylabel="Process-weighted mean waiting")
        axes[0, 1].set(title="Training waiting (200-episode running mean)", xlabel="Training episode", ylabel="Mean waiting")
        axes[1, 0].set(title="Training loss (500-update running mean)", xlabel="Optimizer update", ylabel="Smooth L1 loss")
        axes[1, 1].set(title="Identical exploration schedules", xlabel="Environment steps", ylabel="Epsilon")
        for axis in axes.ravel():
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "learning_curves.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        for name, data in histories.items():
            updates, health = data["optimizer_diagnostics"], data["health"]
            axes[0].plot([int(r["optimizer_updates"]) for r in updates],
                moving_average([float(r["gradient_norm_before_clipping"]) for r in updates], 500), label=LABELS[name], color=COLORS[name])
            axes[1].plot([r["episode"] for r in health], [r["policy_valid_q"]["maximum_absolute"] for r in health],
                marker=".", label=LABELS[name], color=COLORS[name])
        axes[0].axhline(10, color="#777777", linestyle="--", label="Clip threshold")
        axes[0].set(title="Gradient norm before clipping (500-update mean)", xlabel="Optimizer update", ylabel="L2 norm")
        axes[1].set(title="Largest absolute valid Q during validation", xlabel="Training episode", ylabel="Absolute Q (reward units)")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "stability_diagnostics.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    try:
        original = [r for r in summary["comparison"] if r["arrangements"] == "original"]
        positions = np.arange(len(original))
        for axis, key, title in zip(axes.ravel(), ("mean_waiting_time", "mean_response_time", "mean_context_switches", "mean_preemptions"),
                ("Waiting per process", "Response per process", "Context switches per workload", "Preemptions per workload")):
            bars = axis.bar(positions, [r[key] for r in original], color=[COLORS[r["model"]] for r in original])
            axis.bar_label(bars, fmt="%.3f", padding=3)
            axis.set(xticks=positions, xticklabels=[LABELS[r["model"]] for r in original], title=title)
            axis.set_ylim(0, max(r[key] for r in original) * 1.2)
            axis.grid(axis="y", alpha=.2)
        axes[0, 0].axhline(6.006201550387597, color="#238b45", linestyle="--", label="SRTF")
        axes[0, 0].legend()
        fig.suptitle("Selected checkpoints: original Phase 5 validation workloads")
        fig.tight_layout()
        fig.savefig(root / "checkpoint_comparison.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        positions = np.arange(3)
        for offset, name in enumerate(COLORS):
            rows = [r for r in summary["by_process_count"] if r["model"] == name]
            x = positions + (offset - 1) * .25
            axes[0].bar(x, [r["action_agreement_percent"] for r in rows], .25, color=COLORS[name], label=LABELS[name])
            axes[1].bar(x, [r["mean_waiting_range"] for r in rows], .25, color=COLORS[name], label=LABELS[name])
        axes[0].set(title="Identical frozen observations and permutations", ylabel="Identity action agreement (%)", ylim=(0, 105))
        axes[1].set(title="Complete rollout permutations", ylabel="Mean within-workload waiting range")
        for axis in axes:
            axis.set(xticks=positions, xticklabels=["4", "5", "6"], xlabel="Process count")
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "permutation_consistency_comparison.png", dpi=160)
    finally:
        plt.close(fig)
