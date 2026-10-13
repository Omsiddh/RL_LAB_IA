"""Paired, workload-cluster bootstrap summaries for the two experiments."""

from collections import defaultdict

import numpy as np


def bootstrap_ratio(numerators, denominators, seed=63021729, replicates=5000):
    """Resample whole workload contributions; never resample steps or pairs.

    Contributions can be sums/counts, paired waiting differences/process counts,
    or within-workload covariance/variance. Zero-denominator draws are omitted
    and counted explicitly. The point estimate always uses all supplied data.
    """
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if (numerators.ndim != 1 or numerators.shape != denominators.shape or not len(numerators)
            or not np.isfinite(numerators).all() or not np.isfinite(denominators).all()
            or (denominators < 0).any() or type(replicates) is not int or replicates < 1):
        raise ValueError("Expected finite paired workload contributions and positive replicates")
    total = denominators.sum()
    if total == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None,
                "workloads": len(numerators), "valid_bootstrap_draws": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerators), size=(replicates, len(numerators)))
    sums = denominators[indices].sum(axis=1)
    valid = sums > 0
    samples = numerators[indices].sum(axis=1)[valid] / sums[valid]
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(numerators.sum() / total), "ci_low": float(low), "ci_high": float(high),
            "workloads": len(numerators), "valid_bootstrap_draws": int(valid.sum())}


def agreement_summary(rows, seed, replicates):
    contributions = defaultdict(lambda: [0, 0])
    for row in rows:
        contributions[row["workload_id"]][0] += row["action_agreement"]
        contributions[row["workload_id"]][1] += 1
    values = list(contributions.values())
    if not values:
        return {"comparisons": 0, "agreement": None}
    return {"comparisons": sum(value[1] for value in values),
            "agreement": bootstrap_ratio([v[0] for v in values], [v[1] for v in values], seed, replicates),
            "disagreement_classes": {label: sum(row["preference_class"] == label for row in rows)
                                     for label in ("tied", "close", "clear")}}


def aggregate_rollouts(rows):
    process_count = sum(row["process_count"] for row in rows)
    result = {"evaluated_workload_arrangements": len(rows), "evaluated_processes": process_count}
    for key in ("waiting_time", "turnaround_time", "response_time"):
        result["mean_" + key] = sum(row["mean_" + key] * row["process_count"] for row in rows) / process_count
    for key in ("context_switches", "preemptions", "total_reward"):
        result["mean_" + key.removeprefix("total_")] = sum(row[key] for row in rows) / len(rows)
    result["total_reward"] = sum(row["total_reward"] for row in rows)
    return result


def analyze(workloads, actions, rollouts, seed=63021729, replicates=5000):
    index = {(row["workload_id"], row["policy"], row["arrangement_id"]): row for row in rollouts}
    sensitivity = []
    for workload_id, workload in enumerate(workloads):
        dqn = [index[workload_id, "DQN", arrangement] for arrangement in range(8)]
        srtf = [index[workload_id, "SRTF", arrangement] for arrangement in range(8)]
        means = np.array([row["mean_waiting_time"] for row in dqn])
        totals = np.array([row["total_waiting_time"] for row in dqn])
        sensitivity.append({
            "workload_id": workload_id, "process_count": len(workload),
            "original_waiting": float(means[0]), "permuted_mean_waiting": float(means[1:].mean()),
            "permuted_minus_original": float(means[1:].mean() - means[0]),
            "paired_total_waiting_difference": float(totals[1:].mean() - totals[0]),
            "minimum_waiting": float(means.min()), "maximum_waiting": float(means.max()),
            "waiting_range": float(np.ptp(means)), "waiting_population_std": float(means.std(ddof=0)),
            "waiting_changed": bool(np.ptp(totals) > 0),
            "better_permutations": int((totals[1:] < totals[0]).sum()),
            "worse_permutations": int((totals[1:] > totals[0]).sum()),
            "tied_permutations": int((totals[1:] == totals[0]).sum()),
            "best_arrangement_id": int(means.argmin()), "worst_arrangement_id": int(means.argmax()),
            "srtf_waiting": srtf[0]["mean_waiting_time"],
            "srtf_total_waiting_range": max(r["total_waiting_time"] for r in srtf) - min(r["total_waiting_time"] for r in srtf),
            **{f"srtf_{key}_range": max(r[key] for r in srtf) - min(r[key] for r in srtf)
               for key in ("mean_turnaround_time", "mean_response_time", "context_switches", "preemptions")},
        })
    # Identity arrangement is excluded from agreement rates (it is trivially one).
    comparisons = [row for row in actions if row["arrangement_id"] != 0]
    summary = {"fixed_state": agreement_summary(comparisons, seed, replicates),
               "full_rollout": {}, "by_process_count": [], "by_source_policy": {},
               "by_arrangement": [], "slot_agreement": [], "unique_shortest_slot_selection": [],
               "matched_shortest_slot_effects": [], "trajectory_slot_associations": []}
    summary["fixed_state"]["sampled_original_states"] = sum(row["arrangement_id"] == 0 for row in actions)
    for source in ("DQN", "SRTF"):
        summary["by_source_policy"][source] = agreement_summary(
            [row for row in comparisons if row["source_policy"] == source], seed, replicates)
    for group in ("all", 4, 5, 6):
        selected = [row for row in sensitivity if group == "all" or row["process_count"] == group]
        if not selected:
            continue
        ids = {row["workload_id"] for row in selected}
        selected_actions = [row for row in comparisons if row["workload_id"] in ids]
        pair = bootstrap_ratio([r["paired_total_waiting_difference"] for r in selected],
                               [r["process_count"] for r in selected], seed, replicates)
        changed = bootstrap_ratio([int(r["waiting_changed"]) for r in selected], [1] * len(selected), seed, replicates)
        original = aggregate_rollouts([index[i, "DQN", 0] for i in ids])
        permuted = aggregate_rollouts([index[i, "DQN", arrangement] for i in ids for arrangement in range(1, 8)])
        control = aggregate_rollouts([index[i, "SRTF", 0] for i in ids])
        agreement = agreement_summary(selected_actions, seed, replicates)["agreement"]
        group_row = {"process_count_group": group, "workloads": len(ids),
                     "dqn_original_waiting": original["mean_waiting_time"],
                     "dqn_permuted_waiting": permuted["mean_waiting_time"],
                     "srtf_waiting": control["mean_waiting_time"],
                     "paired_waiting_delta": pair["estimate"], "delta_ci_low": pair["ci_low"], "delta_ci_high": pair["ci_high"],
                     "changed_workloads_percent": 100 * changed["estimate"],
                     "mean_waiting_range": float(np.mean([r["waiting_range"] for r in selected])),
                     "maximum_waiting_range": max(r["waiting_range"] for r in selected),
                     "mean_waiting_population_std": float(np.mean([r["waiting_population_std"] for r in selected])),
                     "action_agreement_percent": 100 * agreement["estimate"],
                     "agreement_ci_low_percent": 100 * agreement["ci_low"], "agreement_ci_high_percent": 100 * agreement["ci_high"]}
        if group == "all":
            summary["full_rollout"] = {**group_row, "paired_delta": pair, "changed_workload_fraction": changed,
                "original_DQN": original, "permuted_DQN": permuted, "original_SRTF": control,
                "permuted_SRTF": aggregate_rollouts([r for r in rollouts if r["policy"] == "SRTF" and r["arrangement_id"]]),
                "changed_workloads": sum(r["waiting_changed"] for r in selected),
                "better_permutations": sum(r["better_permutations"] for r in selected),
                "worse_permutations": sum(r["worse_permutations"] for r in selected),
                "tied_permutations": sum(r["tied_permutations"] for r in selected),
                "mean_range_ci": bootstrap_ratio([r["waiting_range"] for r in selected], [1] * len(selected), seed, replicates),
                "largest_range_workload": max(selected, key=lambda r: r["waiting_range"]),
                "srtf_changed_workloads": {key: sum(r[key] > 0 for r in selected) for key in selected[0] if key.startswith("srtf_") and key.endswith("_range")}}
        else:
            summary["by_process_count"].append(group_row)
        for arrangement in range(8):
            for policy in ("DQN", "SRTF"):
                subset = [index[i, policy, arrangement] for i in ids]
                action_subset = [r for r in selected_actions if r["arrangement_id"] == arrangement]
                summary["by_arrangement"].append({"process_count_group": group, "policy": policy,
                    "arrangement_id": arrangement, **aggregate_rollouts(subset),
                    "fixed_state_agreement": (sum(r["action_agreement"] for r in action_subset) / len(action_subset)
                                              if action_subset and policy == "DQN" else None)})
        for slot in range(6 if group == "all" else group):
            subset = [r for r in selected_actions if r["original_action_new_slot"] == slot]
            stats = agreement_summary(subset, seed, replicates)
            if stats["comparisons"]:
                summary["slot_agreement"].append({"process_count_group": group, "slot": slot,
                    "comparisons": stats["comparisons"], **stats["agreement"]})
            shortest = [r for r in selected_actions if r["unique_shortest_new_slot"] == slot]
            counts = defaultdict(lambda: [0, 0])
            for row in shortest:
                counts[row["workload_id"]][0] += row["unique_shortest_selected"]
                counts[row["workload_id"]][1] += 1
            if counts:
                values = list(counts.values())
                summary["unique_shortest_slot_selection"].append({"process_count_group": group, "slot": slot,
                    "opportunities": sum(v[1] for v in values),
                    **bootstrap_ratio([v[0] for v in values], [v[1] for v in values], seed, replicates)})
        states = defaultdict(list)
        for row in selected_actions:
            if row["unique_shortest_original_index"] is not None:
                states[row["workload_id"], row["source_policy"], row["state_id"]].append(row)
        for slot in (3, 5):
            matched = defaultdict(lambda: [0.0, 0])
            for (workload_id, _, _), state_rows in states.items():
                inside = [r["unique_shortest_selected"] for r in state_rows if r["unique_shortest_new_slot"] == slot]
                outside = [r["unique_shortest_selected"] for r in state_rows if r["unique_shortest_new_slot"] != slot]
                if inside and outside:
                    matched[workload_id][0] += float(np.mean(inside) - np.mean(outside))
                    matched[workload_id][1] += 1
            if matched:
                # Include zero-contribution workloads so bootstrap unit remains the original collection/group.
                summary["matched_shortest_slot_effects"].append({"process_count_group": group, "slot": slot,
                    "matched_states": sum(v[1] for v in matched.values()),
                    "contributing_workloads": len(matched),
                    **bootstrap_ratio([matched[i][0] for i in ids], [matched[i][1] for i in ids], seed, replicates)})
            covariance, variance = [], []
            for i in ids:
                rs = [index[i, "DQN", a] for a in range(8)]
                exposure = np.array([r[f"slot_{slot}_unique_shortest_opportunities"] /
                    max(1, sum(r[f"slot_{s}_unique_shortest_opportunities"] for s in range(6))) for r in rs])
                waiting = np.array([r["mean_waiting_time"] for r in rs])
                x, y = exposure - exposure.mean(), waiting - waiting.mean()
                covariance.append(float(x @ y))
                variance.append(float(x @ x))
            if sum(variance) > 0:
                summary["trajectory_slot_associations"].append({"process_count_group": group, "slot": slot,
                    "predictor": "fraction of multi-ready unique-shortest opportunities located in this slot",
                    "outcome": "within-workload centered DQN mean waiting; slope per unit exposure fraction",
                    **bootstrap_ratio(covariance, variance, seed, replicates)})
    return summary, sensitivity
