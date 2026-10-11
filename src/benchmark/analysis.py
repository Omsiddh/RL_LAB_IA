"""Process-weighted aggregates, paired workload gaps, and slot diagnostics."""

from collections import defaultdict
from statistics import fmean, median

from .evaluation import POLICIES


def percentage_gap(actual, reference):
    if reference == 0:
        return 0.0 if actual == 0 else None
    return 100 * (actual - reference) / reference


def aggregate(processes, workloads):
    """Use process totals for timing means; workload totals for switch means."""
    count = len(processes)
    totals = {key: sum(row[key] for row in processes)
              for key in ("waiting_time", "turnaround_time", "response_time")}
    reward = sum(row["total_reward"] for row in workloads)
    if reward != -totals["waiting_time"]:
        raise AssertionError("Aggregate reward identity failed")
    return {
        "workload_count": len(workloads), "process_count": count,
        **{"total_" + key: value for key, value in totals.items()},
        **{"mean_" + key: value / count for key, value in totals.items()},
        "mean_context_switches": fmean(row["context_switches"] for row in workloads),
        "mean_preemptions": fmean(row["preemptions"] for row in workloads),
        "total_reward": reward, "mean_reward": reward / len(workloads),
    }


def slot_diagnostics(decisions):
    groups = defaultdict(list)
    for row in decisions:
        groups[("all", row["process_index"])].append(row)
        groups[(row["process_count"], row["process_index"])].append(row)
    rows = []
    for (count, slot), group in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        unique = [row for row in group if row["unique_shortest"]]
        equivalent = [row for row in group if row["equivalent_choice_candidate"]]
        selected = sum(row["selected"] for row in equivalent)
        expected = sum(row["equivalent_choice_expected"] for row in equivalent)
        rows.append({
            "process_count_group": count, "process_index": slot,
            "ready_opportunities": len(group), "selected_count": sum(row["selected"] for row in group),
            "selection_rate_when_ready": sum(row["selected"] for row in group) / len(group),
            "unique_shortest_opportunities": len(unique),
            "unique_shortest_selected": sum(row["selected"] for row in unique),
            "unique_shortest_selection_rate": sum(row["selected"] for row in unique) / len(unique) if unique else None,
            "identical_feature_opportunities": len(equivalent),
            "identical_feature_selected": selected, "identical_feature_expected_uniform": expected,
            "identical_feature_selection_surplus": selected - expected,
        })
    return rows


def analyze_results(workloads, results, decisions):
    if not workloads or len(workloads) != len(results):
        raise ValueError("Exactly one policy-result group is required per workload")
    processes, workload_rows, policy_workloads = [], [], defaultdict(list)
    for workload_id, (workload, result) in enumerate(zip(workloads, results)):
        row = {"workload_id": workload_id, "process_count": len(workload)}
        for policy in POLICIES:
            metrics = result[policy]
            record = {"workload_id": workload_id, "process_count": len(workload),
                      **{key: metrics[key] for key in ("mean_waiting_time", "mean_turnaround_time", "mean_response_time",
                                                     "context_switches", "preemptions", "total_reward", "total_time", "idle_time")}}
            policy_workloads[policy].append(record)
            prefix = policy.lower().replace(" ", "_")
            row.update({prefix + "_" + key: value for key, value in record.items()
                        if key not in ("workload_id", "process_count")})
            for index, (pid, arrival, burst) in enumerate(workload):
                processes.append({"policy": policy, "workload_id": workload_id,
                                  "process_count": len(workload), "process_index": index,
                                  "pid": pid, "arrival_time": arrival, "burst_time": burst,
                                  **metrics["processes"][pid]})
        dqn_wait = -result["DQN"]["total_reward"]
        srtf_wait = -result["SRTF"]["total_reward"]
        row.update({"dqn_srtf_total_waiting_gap": dqn_wait - srtf_wait,
                    "dqn_srtf_mean_waiting_gap": (dqn_wait - srtf_wait) / len(workload),
                    "dqn_srtf_percentage_waiting_gap": percentage_gap(dqn_wait, srtf_wait)})
        workload_rows.append(row)
    comparison, grouped = [], []
    for policy in POLICIES:
        policy_processes = [row for row in processes if row["policy"] == policy]
        comparison.append({"policy": policy, **aggregate(policy_processes, policy_workloads[policy])})
        for count in sorted({len(workload) for workload in workloads}):
            grouped.append({"policy": policy, "process_count_group": count,
                            **aggregate([row for row in policy_processes if row["process_count"] == count],
                                        [row for row in policy_workloads[policy] if row["process_count"] == count])})
    by_policy = {row["policy"]: row for row in comparison}
    dqn, srtf = by_policy["DQN"], by_policy["SRTF"]
    gaps = [row["dqn_srtf_mean_waiting_gap"] for row in workload_rows]
    best_ids = [row["workload_id"] for row in workload_rows if row["dqn_srtf_mean_waiting_gap"] == min(gaps)]
    worst_ids = [row["workload_id"] for row in workload_rows if row["dqn_srtf_mean_waiting_gap"] == max(gaps)]

    def example(workload_id):
        return {"workload_id": workload_id, "definitions": workloads[workload_id],
                "comparison": workload_rows[workload_id],
                "timelines": {policy: results[workload_id][policy]["timeline"] for policy in POLICIES}}

    summary = {
        "comparison": comparison, "by_process_count": grouped,
        "optimality_gap": {
            "absolute_mean_waiting_gap": dqn["mean_waiting_time"] - srtf["mean_waiting_time"],
            "percentage_mean_waiting_gap": percentage_gap(dqn["total_waiting_time"], srtf["total_waiting_time"]),
            "total_waiting_gap": dqn["total_waiting_time"] - srtf["total_waiting_time"],
            "dqn_srtf_tied_workloads": sum(gap == 0 for gap in gaps),
            "dqn_above_srtf_workloads": sum(gap > 0 for gap in gaps),
            "srtf_inequality_violations": 0,
            "unweighted_mean_workload_gap": fmean(gaps), "median_workload_gap": median(gaps),
            "minimum_workload_gap": min(gaps), "maximum_workload_gap": max(gaps),
            "best_workload_ids": best_ids, "worst_workload_ids": worst_ids,
            "best_example": example(best_ids[0]), "worst_example": example(worst_ids[0]),
        },
        "slot_bias": {"statistics": slot_diagnostics(decisions),
                      "interpretation_limit": "Observational diagnostics on original slots only. Ready/selection rates are "
                                              "confounded by arrival, burst and workload size. Identical-feature choices "
                                              "condition on equal six-feature observations in the selected group. "
                                              "Unique-shortest rates exclude single-ready states. Events are dependent; "
                                              "these are descriptive counts, not significance tests or a causal bias estimate."},
    }
    return comparison, workload_rows, processes, summary
