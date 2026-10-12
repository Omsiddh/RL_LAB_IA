"""Standalone figures for fixed states and full trajectories, kept separate."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(summary, sensitivity, output_dir):
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        for group, color in ((4, "#247ba0"), (5, "#845ec2"), (6, "#d65d0e")):
            for axis, key in zip(axes, ("slot_agreement", "unique_shortest_slot_selection")):
                rows = [r for r in summary[key] if r["process_count_group"] == group]
                x = [r["slot"] for r in rows]
                y = np.array([100 * r["estimate"] for r in rows])
                error = np.array([[100 * r["estimate"] - 100 * r["ci_low"] for r in rows],
                                  [100 * r["ci_high"] - 100 * r["estimate"] for r in rows]])
                axis.errorbar(x, y, yerr=np.maximum(error, 0), marker="o", capsize=3,
                              label=f"{group} processes", color=color)
                axis.set(xticks=range(6), ylim=(-2, 103), ylabel="Percent", xlabel="Destination slot (zero-based)")
                axis.grid(alpha=0.2)
        axes[0].set_title("Identity agreement by original action's destination")
        axes[1].set_title("Uniquely shortest process selected by destination")
        axes[0].legend()
        fig.suptitle("Fixed observed states: seven alternative layouts; workload bootstrap 95% CIs")
        fig.tight_layout()
        fig.savefig(output_dir / "action_agreement_by_slot.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        positions = np.arange(len(sensitivity))
        ordered = sorted(sensitivity, key=lambda row: (row["process_count"], row["waiting_range"]))
        axes[0].vlines(positions, [r["minimum_waiting"] for r in ordered],
                       [r["maximum_waiting"] for r in ordered], color="#845ec2", alpha=0.8)
        axes[0].scatter(positions, [r["original_waiting"] for r in ordered], s=9, color="#222222", label="Original")
        axes[0].set(title="DQN range across all eight arrangements", xlabel="Workloads ordered by count, then range",
                    ylabel="Mean waiting per process (CPU time units)")
        axes[0].legend()
        axes[1].boxplot([[r["waiting_range"] for r in sensitivity if r["process_count"] == n] for n in (4, 5, 6)],
                        tick_labels=["4", "5", "6"])
        axes[1].set(title="Within-workload waiting variation", xlabel="Process count", ylabel="Max minus min mean waiting")
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "waiting_time_variation.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    try:
        for policy, color in (("DQN", "#845ec2"), ("SRTF", "#238b45")):
            rows = [r for r in summary["by_arrangement"] if r["policy"] == policy and r["process_count_group"] == "all"]
            axes[0].plot([r["arrangement_id"] for r in rows], [r["mean_waiting_time"] for r in rows],
                         marker="o", label=policy, color=color)
        axes[0].set(title="Same 128 workloads at each arrangement draw", xlabel="Arrangement (0 = original)",
                    ylabel="Process-weighted mean waiting")
        axes[0].legend()
        axes[1].scatter([r["waiting_range"] for r in sensitivity],
                        [r["srtf_total_waiting_range"] / r["process_count"] for r in sensitivity], alpha=0.6)
        axes[1].set(title="Within-workload invariance control", xlabel="DQN mean waiting range", ylabel="SRTF mean waiting range")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "dqn_srtf_permutation_consistency.png", dpi=160)
    finally:
        plt.close(fig)
