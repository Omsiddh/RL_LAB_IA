"""Stage 6.2 comparison charts from verified aggregate results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def comparison_charts(comparison, output_dir):
    specifications = [
        ("waiting_time", "Mean waiting time", "Waiting time per process (CPU time units)"),
        ("turnaround_time", "Mean turnaround time", "Turnaround time per process (CPU time units)"),
        ("response_time", "Mean response time", "Response time per process (CPU time units)"),
        ("context_switches", "Mean context switches", "Context switches per workload"),
        ("preemptions", "Mean preemptions", "Preemptions per workload"),
    ]
    for key, title, ylabel in specifications:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        try:
            values = [row["mean_" + key] for row in comparison]
            bars = axis.bar([row["policy"] for row in comparison], values,
                            color=["#7952b3", "#718096", "#238b45", "#3182bd"])
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
            axis.set(title=title + " — fixed Phase 5 validation workloads", ylabel=ylabel, xlabel="Scheduler")
            axis.set_ylim(0, max(max(values) * 1.18, 1))
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(Path(output_dir) / (key + "_comparison.png"), dpi=150)
        finally:
            plt.close(fig)
