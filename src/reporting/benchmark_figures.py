"""Final-test comparisons from saved aggregates and paired intervals."""

import matplotlib.pyplot as plt
import numpy as np

from .data import POLICIES
from .style import COLORS


SOURCE = "runs/phase6_4_final_test/"
FOOTER = "FINAL HELD-OUT TEST | 256 workloads / 1,277 processes | Source: phase6_4_final_test/"


def generate(data, writer):
    rows = data["final"]
    for name, metric, title, unit in (
            ("final_waiting", "mean_waiting_time", "Mean waiting time", "CPU time units / process"),
            ("final_turnaround", "mean_turnaround_time", "Mean turnaround time", "CPU time units / process"),
            ("final_response", "mean_response_time", "Mean response time", "CPU time units / process"),
            ("final_context_switches", "mean_context_switches", "Mean context switches", "Switches / workload"),
            ("final_preemptions", "mean_preemptions", "Mean preemptions", "Preemptions / workload")):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        values = [float(r[metric]) for r in rows]
        bars = ax.barh(POLICIES, values, color=[COLORS[p] for p in POLICIES], height=.65)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5, fontsize=9)
        ax.set(xlabel=unit, ylabel="Scheduler", title="Final held-out test: " + title, xlim=(0, max(values) * 1.17 or 1))
        ax.grid(axis="x")
        if name == "final_waiting":
            ax.axvline(values[0], linestyle="--", color=COLORS["SRTF"], linewidth=1.2, label="SRTF waiting reference")
            ax.legend(loc="lower right", fontsize=8)
        writer.save(fig, name, "final held-out test", FOOTER + "policy_comparison.csv", [SOURCE + "policy_comparison.csv"],
                    {"policies": POLICIES, "metric": metric, "values": values})
    fig, ax = plt.subplots(figsize=(8, 4.7))
    policies = POLICIES[1:4]
    values = [float(r["srtf_percentage_waiting_gap"]) for r in rows[1:4]]
    bars = ax.bar(policies, values, color=[COLORS[p] for p in policies], width=.6)
    ax.bar_label(bars, labels=[f"{v:.3f}%" for v in values], padding=5)
    ax.axhline(0, color=COLORS["SRTF"], linewidth=1.3, label="SRTF = 0%")
    ax.set(title="Final held-out test: DQN waiting gap to SRTF", xlabel="Selected DQN checkpoint", ylabel="Waiting above SRTF (%)", ylim=(0, max(values) * 1.22))
    ax.legend(fontsize=8)
    writer.save(fig, "optimality_gaps", "final held-out test", FOOTER + "policy_comparison.csv", [SOURCE + "policy_comparison.csv"], {"policies": policies, "values_percent": values})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    points = np.arange(3)
    plotted = {}
    for i, policy in enumerate(POLICIES):
        selected = sorted((r for r in data["final_groups"] if r["policy"] == policy), key=lambda r: int(r["process_count_group"]))
        values = [float(r["mean_waiting_time"]) for r in selected]
        plotted[policy] = values
        ax.bar(points + (i - 2.5) * .13, values, width=.13, label=policy, color=COLORS[policy])
    ax.set(xticks=points, xticklabels=[f"{n} processes\n{data['test_metadata']['process_count_workloads'][str(n)]} workloads" for n in (4, 5, 6)],
           xlabel="Process count", ylabel="CPU time units / process", title="Final held-out test: waiting by process count")
    ax.grid(axis="y")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    writer.save(fig, "by_process_count", "final held-out test", FOOTER + "by_process_count.csv", [SOURCE + "by_process_count.csv"], {"process_counts": [4, 5, 6], "waiting": plotted})
    interval = next(r for r in data["intervals"] if r["primary"] == "True")
    estimate, low, high = [float(interval[k]) for k in ("difference", "ci_low", "ci_high")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o", color=COLORS[POLICIES[1]], markersize=9, capsize=7, linewidth=2)
    ax.axvline(0, linestyle="--", color="#555555")
    ax.set(yticks=[0], yticklabels=["Augmented − original\n10k DQN"], xlabel="Paired waiting difference (CPU time units / process)",
           title="Final held-out test: augmentation ablation", ylim=(-.6, .6), xlim=(min(low * 1.2, -.01), max(.03, high * 1.2)))
    ax.text(.02, .87, f"Difference {estimate:+.6f}; 95% CI [{low:+.6f}, {high:+.6f}]", transform=ax.transAxes, fontsize=9)
    ax.text(.02, .08, "Negative values favor augmentation", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="x")
    writer.save(fig, "augmentation_difference_ci", "final held-out test",
        FOOTER + "bootstrap_intervals.csv\n5,000 paired workload resamples; uncertainty across workloads, conditional on this training seed pair",
        [SOURCE + "bootstrap_intervals.csv"], {"difference": estimate, "ci_low": low, "ci_high": high, "replicates": int(interval["replicates"]), "seed": int(interval["seed"])})
