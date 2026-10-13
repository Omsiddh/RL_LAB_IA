"""Render only recorded execution intervals; never invoke a scheduling policy."""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .data import same


DISPLAY_POLICIES = ("SRTF", "Original DQN 10k", "Augmented DQN 10k")
PROCESS_COLORS = ("#4c78a8", "#f2a541", "#54a477", "#b477a7", "#df6a62", "#83a9b5")


def merge_intervals(timeline):
    """Coalesce adjacent identical (index, PID) segments, including idle."""
    merged = []
    for start, end, index, pid in timeline:
        if end <= start or (merged and start != merged[-1][1]):
            raise ValueError("Gantt timeline contains nonpositive durations, gaps, or overlap")
        if merged and merged[-1][1] == start and merged[-1][2:] == (index, pid):
            merged[-1] = (merged[-1][0], end, index, pid)
        else:
            merged.append((start, end, index, pid))
    return merged


def verify_timeline(definitions, timeline):
    runtime, completion, first = defaultdict(int), {}, {}
    cursor = 0
    for start, end, index, pid in timeline:
        if start != cursor or end <= start:
            raise ValueError("Gantt timeline is not a contiguous non-overlapping CPU history")
        cursor = end
        if index is None:
            if pid is not None:
                raise ValueError("Idle interval has a process ID")
            continue
        if not isinstance(index, int) or not 0 <= index < len(definitions) or definitions[index][0] != pid:
            raise ValueError("Gantt timeline changed a stable process index or identity")
        if start < definitions[index][1]:
            raise ValueError("Gantt timeline executes a process before its arrival")
        runtime[pid] += end - start
        first.setdefault(pid, start)
        completion[pid] = end
    if dict(runtime) != {pid: burst for pid, _, burst in definitions}:
        raise ValueError("Gantt execution durations do not equal saved CPU bursts")
    waiting = sum(completion[pid] - arrival - burst for pid, arrival, burst in definitions) / len(definitions)
    return {"runtime": dict(runtime), "total_time": cursor, "mean_waiting_time": waiting,
            "first_start": first, "completion": completion}


def verify_examples(data):
    checks = []
    for example in data["timelines"]:
        wid, definitions = example["workload_id"], example["definitions"]
        if definitions != data["test_workloads"][wid]:
            raise ValueError("Gantt case-study definitions differ from saved test workloads")
        for policy, timeline in example["timelines"].items():
            checked = verify_timeline(definitions, timeline)
            merged = merge_intervals(timeline)
            if checked != verify_timeline(definitions, merged):
                raise ValueError("Merging Gantt segments altered timing or CPU accounting")
            prefix = policy.lower().replace(" ", "_")
            same(checked["mean_waiting_time"], example["policy_metrics"][prefix + "_mean_waiting_time"], "Gantt waiting " + policy)
            for row in (r for r in data["processes"] if r["policy"] == policy and int(r["workload_id"]) == wid):
                same(checked["first_start"][row["pid"]], row["first_start_time"], "Gantt first start")
                same(checked["completion"][row["pid"]], row["completion_time"], "Gantt completion")
            checks.append({"workload_id": wid, "policy": policy, "raw_segments": len(timeline),
                "merged_segments": len(merged), "process_runtime": checked["runtime"], "timing_unchanged": True})
    return checks


def generate(data, writer):
    for example in data["timelines"]:
        wid, definitions = example["workload_id"], example["definitions"]
        colors = {pid: PROCESS_COLORS[i] for i, pid in enumerate(sorted(p[0] for p in definitions))}
        fig = plt.figure(figsize=(11.5, 7.6))
        grid = fig.add_gridspec(4, 1, height_ratios=(1.7, 1, 1, 1))
        header = fig.add_subplot(grid[0])
        header.axis("off")
        delta = example["augmented_minus_original_waiting"]
        fig.suptitle(f"Final held-out workload {wid}: augmented − original waiting = {delta:+.3f}", fontsize=14)
        table = header.table(cellText=[[index, pid, arrival, burst] for index, (pid, arrival, burst) in enumerate(definitions)],
            colLabels=["Slot", "Process", "Arrival", "CPU burst"], cellLoc="center", bbox=(.02, .03, .67, .88))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for index, (pid, _, _) in enumerate(definitions, start=1):
            rgb = to_rgb(colors[pid])
            table[index, 1].set_facecolor(tuple(.78 + .22 * c for c in rgb))
        header.legend(handles=[Patch(facecolor=colors[pid], label=pid) for pid in sorted(colors)] + [Patch(facecolor="#dddddd", hatch="///", label="Idle")],
            loc="center right", frameon=False, fontsize=9)
        axes, plotted = [], {}
        for i, policy in enumerate(DISPLAY_POLICIES):
            ax = fig.add_subplot(grid[i + 1], sharex=axes[0] if axes else None)
            axes.append(ax)
            merged = merge_intervals(example["timelines"][policy])
            for start, end, index, pid in merged:
                ax.broken_barh([(start, end - start)], (.38, .43), facecolors=colors[pid] if pid is not None else "#dddddd", edgecolors="white", linewidth=.8,
                    hatch="///" if pid is None else None)
                ax.text((start + end) / 2, .595, str(pid) if pid is not None else "idle", ha="center", va="center", fontsize=8, color="#111111")
            wait = float(example["policy_metrics"][policy.lower().replace(" ", "_") + "_mean_waiting_time"])
            ax.text(0, .04, f"Mean waiting = {wait:.6f} CPU time units / process", transform=ax.transAxes, fontsize=8)
            ax.set(yticks=[], ylim=(0, 1))
            ax.set_ylabel(policy.replace(" DQN ", "\nDQN "), rotation=0, labelpad=12, va="center", ha="right")
            ax.tick_params(axis="x", labelbottom=i == 2)
            ax.grid(axis="x", linestyle=":")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            ax.spines[["left", "top", "right"]].set_visible(False)
            plotted[policy] = {"merged_intervals": merged, "mean_waiting_time": wait}
        end = max(e[1] for policy in DISPLAY_POLICIES for e in example["timelines"][policy])
        axes[-1].set(xlabel="Simulated CPU time (intervals include start, exclude end)", xlim=(0, end))
        writer.save(fig, f"gantt_workload_{wid}", "final held-out illustrative case",
            f"FINAL HELD-OUT CASE STUDY | {len(definitions)} processes; workload {wid} of 256 (zero based); selected by extreme augmented − original waiting\nSource: phase6_4_final_test/selected_timelines.json; contiguous intervals merged; no policy execution or invented segments",
            ["runs/phase6_4_final_test/selected_timelines.json"], {"workload_id": wid, "definitions": definitions, "policies": plotted})
