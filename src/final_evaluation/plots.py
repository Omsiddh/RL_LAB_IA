"""Static held-out comparison charts, with no training or policy changes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protocol import POLICIES


COLORS = ("#276749", "#245fa5", "#dd8c25", "#8269a9", "#777777", "#b34d53")
LABELS = ("SRTF", "Augmented\nDQN 10k", "Original\nDQN 10k", "Reference\nDQN 6k", "FCFS", "RR (q=2)")


def comparison_charts(comparison, grouped, output_dir):
    for metric, title, unit in (
            ("mean_waiting_time", "Held-out mean waiting time", "CPU time units / process"),
            ("mean_turnaround_time", "Held-out mean turnaround time", "CPU time units / process"),
            ("mean_response_time", "Held-out mean response time", "CPU time units / process"),
            ("mean_context_switches", "Held-out context switches", "Switches / workload"),
            ("mean_preemptions", "Held-out preemptions", "Preemptions / workload")):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        values = [r[metric] for r in comparison]
        bars = ax.bar(LABELS, values, color=COLORS)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.17))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / (metric + ".png"), dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(LABELS[1:4], [r["srtf_percentage_waiting_gap"] for r in comparison[1:4]], color=COLORS[1:4])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(title="Held-out DQN waiting gaps above SRTF", ylabel="Percentage above SRTF waiting")
    ax.set_ylim(0, max(r["srtf_percentage_waiting_gap"] for r in comparison[1:4]) * 1.2 or 1)
    fig.tight_layout()
    fig.savefig(output_dir / "dqn_optimality_gaps.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(3)
    for i, policy in enumerate(POLICIES):
        group = sorted((r for r in grouped if r["policy"] == policy), key=lambda r: r["process_count_group"])
        if len(group) != 3:
            raise ValueError("Final process-count chart requires all three process-count groups")
        axes[0].bar(x + (i - 2.5) * .13, [r["mean_waiting_time"] for r in group], width=.13, color=COLORS[i], label=LABELS[i].replace("\n", " "))
        if 1 <= i <= 3:
            axes[1].plot((4, 5, 6), [r["srtf_percentage_waiting_gap"] for r in group], marker="o", color=COLORS[i], label=LABELS[i].replace("\n", " "))
    axes[0].set(xticks=x, xticklabels=(4, 5, 6), xlabel="Processes / workload", ylabel="Waiting / process", title="Held-out waiting by process count")
    axes[1].set(xticks=(4, 5, 6), xlabel="Processes / workload", ylabel="Percentage above SRTF waiting", title="DQN optimality gaps by process count")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "by_process_count.png", dpi=160)
    plt.close(fig)
