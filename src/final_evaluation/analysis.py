"""Paired workload statistics; timing averages always use process totals."""

from collections import defaultdict
from statistics import median

import numpy as np

from src.benchmark.analysis import aggregate, percentage_gap
from .protocol import BOOTSTRAP_SEED, POLICIES


PRIMARY, CONTROL = POLICIES[1:3]
CONTRASTS = [(PRIMARY, p) for p in (CONTROL, "SRTF", "FCFS", "Round Robin", "Reference DQN 6k")]
CONTRASTS += [(CONTROL, "SRTF"), ("Reference DQN 6k", "SRTF")]


def bootstrap_intervals(workloads, results, seed=BOOTSTRAP_SEED, replicates=5000):
    """One index matrix per group retains all six policies in each draw."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in ("all", 4, 5, 6):
        ids = [i for i, w in enumerate(workloads) if group == "all" or len(w) == group]
        if not ids:
            continue
        counts = np.array([len(workloads[i]) for i in ids])
        samples = rng.integers(0, len(ids), size=(replicates, len(ids)))
        denominator = counts[samples].sum(axis=1)
        for left, right in CONTRASTS:
            metrics = ("waiting_time", "response_time", "context_switches", "preemptions") if (left, right) == (PRIMARY, CONTROL) else ("waiting_time",)
            for metric in metrics:
                def total(i, policy):
                    value = results[i][policy]
                    return sum(p[metric] for p in value["processes"].values()) if metric.endswith("_time") else value[metric]
                differences = np.array([total(i, left) - total(i, right) for i in ids])
                timing = metric.endswith("_time")
                boot = differences[samples].sum(axis=1) / (denominator if timing else len(ids))
                low, high = np.percentile(boot, [2.5, 97.5])
                rows.append({"group": group, "left_policy": left, "right_policy": right,
                    "metric": metric, "difference": float(differences.sum() / (counts.sum() if timing else len(ids))),
                    "ci_low": float(low), "ci_high": float(high), "replicates": replicates,
                    "seed": seed, "workload_count": len(ids), "process_count": int(counts.sum()),
                    "primary": group == "all" and (left, right) == (PRIMARY, CONTROL) and metric == "waiting_time"})
    return rows


def analyze(workloads, results):
    if not workloads or len(workloads) != len(results):
        raise ValueError("Exactly one six-policy result group is required per workload")
    processes, wide, paired, records = [], [], [], defaultdict(list)
    for wid, (workload, result) in enumerate(zip(workloads, results)):
        if set(result) != set(POLICIES):
            raise ValueError("Every workload must have all six policies")
        row = {"workload_id": wid, "process_count": len(workload)}
        pair = {"workload_id": wid, "process_count": len(workload)}
        for policy in POLICIES:
            value = result[policy]
            record = {"workload_id": wid, "process_count": len(workload),
                **{key: value[key] for key in ("mean_waiting_time", "mean_turnaround_time", "mean_response_time",
                    "context_switches", "preemptions", "total_reward", "total_time", "idle_time")}}
            records[policy].append(record)
            prefix = policy.lower().replace(" ", "_")
            row.update({prefix + "_" + k: v for k, v in record.items() if k not in ("workload_id", "process_count")})
            for index, (pid, arrival, burst) in enumerate(workload):
                processes.append({"policy": policy, "workload_id": wid, "process_count": len(workload),
                    "process_index": index, "pid": pid, "arrival_time": arrival, "burst_time": burst,
                    **value["processes"][pid]})
            wait = -value["total_reward"]
            srtf = -result["SRTF"]["total_reward"]
            pair[prefix + "_srtf_total_waiting_gap"] = wait - srtf
            pair[prefix + "_srtf_mean_waiting_gap"] = (wait - srtf) / len(workload)
            pair[prefix + "_srtf_percentage_gap"] = percentage_gap(wait, srtf)
        for left, right in CONTRASTS[:5]:
            prefix = "augmented_minus_" + right.lower().replace(" ", "_")
            difference = -result[left]["total_reward"] + result[right]["total_reward"]
            pair[prefix + "_total_waiting_difference"] = difference
            pair[prefix + "_mean_waiting_difference"] = difference / len(workload)
        difference = pair["augmented_minus_original_dqn_10k_total_waiting_difference"]
        pair["augmentation_outcome"] = "improves" if difference < 0 else "worsens" if difference > 0 else "ties"
        wide.append(row)
        paired.append(pair)
    comparison, grouped = [], []
    for group in ("all", 4, 5, 6):
        group_rows = []
        for policy in POLICIES:
            p = [row for row in processes if row["policy"] == policy and (group == "all" or row["process_count"] == group)]
            w = [row for row in records[policy] if group == "all" or row["process_count"] == group]
            if p:
                group_rows.append({"policy": policy, **({} if group == "all" else {"process_count_group": group}), **aggregate(p, w)})
        if not group_rows:
            continue
        optimal = group_rows[0]["total_waiting_time"]
        for row in group_rows:
            row["srtf_absolute_waiting_gap"] = (row["total_waiting_time"] - optimal) / row["process_count"]
            row["srtf_percentage_waiting_gap"] = percentage_gap(row["total_waiting_time"], optimal)
            policy = row["policy"]
            gaps = [-results[i][policy]["total_reward"] + results[i]["SRTF"]["total_reward"]
                    for i, w in enumerate(workloads) if group == "all" or len(w) == group]
            row["srtf_tied_workloads"] = sum(g == 0 for g in gaps)
            row["srtf_above_workloads"] = sum(g > 0 for g in gaps)
        (comparison if group == "all" else grouped).extend(group_rows)
    differences = [p["augmented_minus_original_dqn_10k_mean_waiting_difference"] for p in paired]
    best = min(range(len(workloads)), key=lambda i: (differences[i], i))
    worst = min(range(len(workloads)), key=lambda i: (-differences[i], i))
    examples = []
    for label, wid in (("best", best), ("worst", worst)):
        examples.append({"label": label, "workload_id": wid, "definitions": workloads[wid],
            "augmented_minus_original_waiting": differences[wid], "paired_metrics": paired[wid],
            "policy_metrics": wide[wid], "timelines": {p: results[wid][p]["timeline"] for p in POLICIES}})
    outcomes = {outcome: sum(p["augmentation_outcome"] == outcome for p in paired) for outcome in ("improves", "worsens", "ties")}
    summary = {"comparison": comparison, "by_process_count": grouped,
        "augmentation_workload_outcomes": outcomes,
        "augmentation_workload_outcomes_by_process_count": {str(n): {outcome: sum(p["process_count"] == n and p["augmentation_outcome"] == outcome for p in paired)
            for outcome in outcomes} for n in (4, 5, 6)},
        "workload_mean_waiting_difference_distribution": {"minimum": min(differences), "maximum": max(differences),
            "median": median(differences), "unweighted_mean": sum(differences) / len(differences)},
        "examples": [{k: v for k, v in ex.items() if k != "timelines"} for ex in examples],
        "limitations": ["One paired training seed; bootstrap measures workload sampling, not independent training-seed variability",
            "Selected checkpoints were fixed using model-selection validation, not this test collection",
            "Same workload distribution as training; no evidence for distribution shift or convergence",
            "Zero context-switch overhead; SRTF knows remaining bursts, while DQN receives normalized adapter observations",
            "Secondary and subgroup percentile intervals are descriptive and unadjusted for multiple comparisons",
            "The deterministic repeat is the same sample, not another experiment"]}
    return comparison, wide, processes, grouped, paired, examples, summary
