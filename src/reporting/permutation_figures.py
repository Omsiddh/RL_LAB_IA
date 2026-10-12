"""Robustness plots clearly restricted to saved validation interventions."""

import matplotlib.pyplot as plt
import numpy as np

from .data import MODEL_NAMES
from .style import MODEL_COLORS


ROOT = "runs/phase6_3b_controlled/"
FOOTER = "VALIDATION / PERMUTATION EXPERIMENT | 128 workloads / 645 processes | Not held-out test results"


def generate(data, writer):
    model_ids = tuple(MODEL_NAMES)
    summaries = [data["permutation_models"][model]["summary"] for model in model_ids]
    sources = [ROOT + "evaluation/" + model + "/permutation_summary.json" for model in model_ids]
    for name, title, ylabel, values, low, high, details in (
        ("permutation_agreement", "Fixed-state process-identity agreement", "Agreement across non-identity layouts (%)",
         [100 * s["fixed_state"]["agreement"]["estimate"] for s in summaries],
         [100 * s["fixed_state"]["agreement"]["ci_low"] for s in summaries],
         [100 * s["fixed_state"]["agreement"]["ci_high"] for s in summaries],
         "1,969 common original states; 13,783 comparisons / model; saved workload-bootstrap 95% CIs"),
        ("permutation_waiting_range", "Within-workload waiting variation", "Mean waiting range (CPU time units / process)",
         [s["full_rollout"]["mean_waiting_range"] for s in summaries],
         [s["full_rollout"]["mean_range_ci"]["ci_low"] for s in summaries],
         [s["full_rollout"]["mean_range_ci"]["ci_high"] for s in summaries],
         "Eight complete-trajectory layouts / workload (original + seven alternatives); saved workload-bootstrap 95% CIs")):
        fig, ax = plt.subplots(figsize=(8.8, 5))
        x = np.arange(3)
        bars = ax.bar(x, values, color=[MODEL_COLORS[m] for m in model_ids], width=.6)
        ax.errorbar(x, values, yerr=[np.array(values) - low, np.array(high) - values], fmt="none", ecolor="#222222", capsize=5)
        ax.set(xticks=x, xticklabels=list(MODEL_NAMES.values()), xlabel="Selected model", ylabel=ylabel, title="Validation / permutations: " + title,
               ylim=(0, 105 if name == "permutation_agreement" else max(high) * 1.18))
        for bar, value, upper in zip(bars, values, high):
            ax.text(bar.get_x() + bar.get_width() / 2, upper + (1.5 if name == "permutation_agreement" else .025), f"{value:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y")
        writer.save(fig, name, "validation / permutation", FOOTER + "\n" + details + "; source: phase6_3b_controlled/evaluation/*/permutation_summary.json", sources,
                    {"models": model_ids, "values": values, "ci_low": low, "ci_high": high})
    if all("slots" in data["permutation_models"][model] for model in model_ids):
        fig, ax = plt.subplots(figsize=(9.2, 5))
        plotted = {}
        for model in model_ids:
            rows = [r for r in data["permutation_models"][model]["slots"] if r["process_count_group"] == "all"]
            slots = [int(r["slot"]) for r in rows]
            values, low, high = ([100 * float(r[key]) for r in rows] for key in ("estimate", "ci_low", "ci_high"))
            ax.errorbar(slots, values, yerr=[np.array(values) - low, np.array(high) - values], marker="o", capsize=3, linewidth=1.3, label=MODEL_NAMES[model], color=MODEL_COLORS[model])
            plotted[model] = {"slots": slots, "agreement_percent": values, "ci_low": low, "ci_high": high, "comparisons": [int(r["comparisons"]) for r in rows]}
        ax.set(xlabel="Destination slot of the original selected process (zero based)", ylabel="Original chosen identity retained (%)",
               title="Validation / permutations: agreement by destination slot", xticks=range(6), ylim=(0, 105))
        ax.legend(fontsize=9)
        ax.grid()
        writer.save(fig, "permutation_by_slot", "validation / permutation", FOOTER + "\nSaved workload-bootstrap 95% CIs; denominators vary by slot; slots 4/5 only exist for 5–6/6 jobs | Source: evaluation/*/slot_agreement.csv",
            [ROOT + "evaluation/" + model + "/slot_agreement.csv" for model in model_ids], plotted)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plotted = {}
    for model in model_ids:
        rows = sorted((r for r in data["permutation_groups"] if r["model"] == model), key=lambda r: int(r["process_count_group"]))
        counts = [int(r["process_count_group"]) for r in rows]
        agreement = [float(r["action_agreement_percent"]) for r in rows]
        ranges = [float(r["mean_waiting_range"]) for r in rows]
        axes[0].plot(counts, agreement, marker="o", color=MODEL_COLORS[model], label=MODEL_NAMES[model])
        axes[1].plot(counts, ranges, marker="o", color=MODEL_COLORS[model], label=MODEL_NAMES[model])
        plotted[model] = {"process_counts": counts, "agreement_percent": agreement, "mean_range": ranges}
    axes[0].set(ylabel="Fixed-state agreement (%)", ylim=(0, 105), title="Action consistency")
    axes[1].set(ylabel="Mean waiting range (CPU units / process)", ylim=(0, None), title="Trajectory variation")
    for ax in axes:
        ax.set(xlabel="Processes / workload", xticks=(4, 5, 6))
        ax.grid()
        ax.legend(fontsize=8)
    fig.suptitle("Validation / permutations: robustness by process count", fontsize=13)
    writer.save(fig, "permutation_by_process_count", "validation / permutation", FOOTER + "\n4 jobs: 41 workloads; 5 jobs: 41; 6 jobs: 46 | Source: phase6_3b_controlled/by_process_count.csv", [ROOT + "by_process_count.csv"], plotted)
