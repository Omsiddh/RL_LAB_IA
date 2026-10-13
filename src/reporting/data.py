"""Read saved experiment records and check consistency without policy execution."""

import csv
import hashlib
import json
import math
from pathlib import Path


POLICIES = ("SRTF", "Augmented DQN 10k", "Original DQN 10k", "Reference DQN 6k", "FCFS", "Round Robin")
MODEL_NAMES = {"reference_5250": "Reference DQN 6k", "original_10000_best": "Original DQN 10k", "augmented_10000_best": "Augmented DQN 10k"}
METRICS = ("mean_waiting_time", "mean_turnaround_time", "mean_response_time", "mean_context_switches", "mean_preemptions", "total_reward", "srtf_percentage_waiting_gap")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError(f"Saved artifact consistency failure for {label}: {actual} != {expected}")


class Sources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, relative):
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 6.5 source artifact is missing: {path}. Restore the saved experiment; do not rerun policies.")
        self.hashes[relative] = sha256(path)
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def csv(self, relative):
        with self.path(relative).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


def verify_final(data):
    """Cross-check displayed means against saved process/workload totals."""
    rows, process_rows, workload_rows = data["final"], data["processes"], data["workload_rows"]
    metadata, summary = data["test_metadata"], data["final_summary"]
    if len(rows) != 6 or tuple(r["policy"] for r in rows) != POLICIES:
        raise ValueError("Final comparison must retain the recorded six-policy ordering")
    count, processes = metadata["workload_count"], metadata["process_count"]
    if count != 256 or processes != 1277 or len(workload_rows) != count or len(process_rows) != processes * 6:
        raise ValueError("Final saved collection must contain 256 workloads / 1277 processes per policy")
    definitions = data["test_workloads"]
    digest = hashlib.sha256(json.dumps(definitions, separators=(",", ":")).encode()).hexdigest()
    if len(definitions) != count or sum(map(len, definitions)) != processes or digest != metadata["workload_sha256"]:
        raise ValueError("Saved test workload definitions/counts/digest do not match their manifest")
    for n in (4, 5, 6):
        if sum(len(w) == n for w in definitions) != metadata["process_count_workloads"][str(n)]:
            raise ValueError("Saved process-count distribution does not match its manifest")
    saved = {r["policy"]: r for r in summary["comparison"]}
    canonical = {(str(wid), str(index), str(pid), str(arrival), str(burst))
                 for wid, w in enumerate(definitions) for index, (pid, arrival, burst) in enumerate(w)}
    for row in rows:
        policy, prefix = row["policy"], row["policy"].lower().replace(" ", "_")
        selected = [p for p in process_rows if p["policy"] == policy]
        identities = {(p["workload_id"], p["process_index"], p["pid"], p["arrival_time"], p["burst_time"]) for p in selected}
        if identities != canonical or len(selected) != processes:
            raise ValueError("Saved policy process identities differ from frozen definitions: " + policy)
        for key in METRICS:
            same(row[key], saved[policy][key], policy + " " + key)
        for key in ("waiting_time", "turnaround_time", "response_time"):
            total = sum(int(p[key]) for p in selected)
            same(row["mean_" + key], total / processes, policy + " weighted " + key)
            same(total, saved[policy]["total_" + key], policy + " total " + key)
        for key in ("context_switches", "preemptions"):
            same(row["mean_" + key], sum(int(w[prefix + "_" + key]) for w in workload_rows) / count, policy + " workload mean " + key)
        same(row["total_reward"], -sum(int(p["waiting_time"]) for p in selected), policy + " reward identity")
        same(row["total_reward"], sum(int(w[prefix + "_total_reward"]) for w in workload_rows), policy + " workload reward")
        same(row["workload_count"], count, policy + " count")
        same(row["process_count"], processes, policy + " processes")
    for row in data["final_groups"]:
        n, policy = int(row["process_count_group"]), row["policy"]
        group = [p for p in process_rows if p["policy"] == policy and int(p["process_count"]) == n]
        for key in ("waiting_time", "turnaround_time", "response_time"):
            same(row["mean_" + key], sum(int(p[key]) for p in group) / len(group), policy + " group " + key)
        same(row["workload_count"], metadata["process_count_workloads"][str(n)], "group count")
    for row in data["intervals"]:
        saved_ci = next(r for r in summary["bootstrap_intervals"] if str(r["group"]) == row["group"]
                        and r["left_policy"] == row["left_policy"] and r["right_policy"] == row["right_policy"] and r["metric"] == row["metric"])
        for key in ("difference", "ci_low", "ci_high", "replicates", "seed"):
            same(row[key], saved_ci[key], "saved bootstrap " + key)
    # These are explicit requested sanity checks, never plotted source values.
    for policy, expected in ((POLICIES[0], 5.826155), (POLICIES[1], 5.964761), (POLICIES[2], 6.118246)):
        if abs(float(next(r for r in rows if r["policy"] == policy)["mean_waiting_time"]) - expected) > 1e-6:
            raise ValueError("Requested source sanity check failed: " + policy)


def load_results(root):
    source = Sources(root)
    final = "runs/phase6_4_final_test/"
    controlled = "runs/phase6_3b_controlled/"
    permutation = "runs/phase6_3_permutation/"
    data = {"final": source.csv(final + "policy_comparison.csv"),
        "processes": source.csv(final + "per_process_results.csv"), "workload_rows": source.csv(final + "per_workload_comparison.csv"),
        "final_groups": source.csv(final + "by_process_count.csv"), "intervals": source.csv(final + "bootstrap_intervals.csv"),
        "final_summary": source.json(final + "benchmark_summary.json"), "test_metadata": source.json(final + "test_workload_manifest.json"),
        "test_workloads": source.json(final + "test_workloads.json"), "timelines": source.json(final + "selected_timelines.json"),
        "protocol": source.json(final + "frozen_protocol.json"), "models": source.json(final + "selected_model_manifest.json"),
        "phase5_metadata": source.json("runs/phase5_full_6000/best_model.json"),
        "phase5_validation": source.csv("runs/phase5_full_6000/validation_metrics.csv"),
        "phase5_config": source.json("runs/phase5_full_6000/training_config.json"),
        "permutation_reference": source.json(permutation + "permutation_summary.json"),
        "permutation_rows": source.csv(controlled + "permutation_comparison.csv"),
        "permutation_groups": source.csv(controlled + "by_process_count.csv"),
        "validation_comparison": source.csv(controlled + "original_validation_comparison.csv"),
        "validation_paired": source.csv(controlled + "paired_model_comparison.csv"),
        "controlled": source.json(controlled + "controlled_summary.json"),
        "training_summary": source.json(controlled + "training_comparison.json"),
        "training": {}, "permutation_models": {}, "missing_optional_data": []}
    for folder, policy in (("original_10000", POLICIES[2]), ("augmented_10000", POLICIES[1])):
        prefix = controlled + folder + "/"
        data["training"][policy] = {"training": source.csv(prefix + "training_metrics.csv"),
            "validation": source.csv(prefix + "validation_metrics.csv"), "optimizer": source.csv(prefix + "optimizer_diagnostics.csv"),
            "best": source.json(prefix + "best_model.json"), "config": source.json(prefix + "training_config.json")}
    for model in MODEL_NAMES:
        prefix = controlled + "evaluation/" + model + "/"
        data["permutation_models"][model] = {"summary": source.json(prefix + "permutation_summary.json")}
        if (source.root / (prefix + "slot_agreement.csv")).is_file():
            data["permutation_models"][model]["slots"] = source.csv(prefix + "slot_agreement.csv")
        else:
            data["missing_optional_data"].append(prefix + "slot_agreement.csv")
    verify_final(data)
    ref = data["permutation_reference"]["fixed_state"]
    same(ref["agreement"]["estimate"], data["permutation_models"]["reference_5250"]["summary"]["fixed_state"]["agreement"]["estimate"], "Stage 6.3A/B reference agreement")
    if data["controlled"]["fixed_original_states"] != 1969 or data["controlled"]["fixed_permutation_comparisons_per_model"] != 13783:
        raise ValueError("Shared-state permutation sample counts do not match the saved experiment")
    for model, bundle in data["permutation_models"].items():
        row = next(r for r in data["permutation_rows"] if r["model"] == model and r["arrangements"] == "original")
        same(row["fixed_state_agreement_percent"], 100 * bundle["summary"]["fixed_state"]["agreement"]["estimate"], model + " agreement")
        if bundle["summary"]["fixed_state"]["comparisons"] != 13783:
            raise ValueError("Models did not share the same fixed-state comparison count")
    expected_episodes = {POLICIES[2]: 9000, POLICIES[1]: 9750}
    for policy, bundle in data["training"].items():
        rows, val, best = bundle["training"], bundle["validation"], bundle["best"]
        if [int(r["episode"]) for r in rows] != list(range(1, 10001)) or [int(r["episode"]) for r in val] != list(range(0, 10001, 250)):
            raise ValueError("Saved training/validation histories are incomplete: " + policy)
        if best["episode"] != expected_episodes[policy]:
            raise ValueError("Selected checkpoint changed: " + policy)
        same(next(r for r in val if int(r["episode"]) == best["episode"])["mean_waiting_time"], best["validation"]["mean_waiting_time"], policy + " best checkpoint")
    if {e["workload_id"] for e in data["timelines"]} != {69, 201}:
        raise ValueError("The saved Gantt case studies must be workloads 69 and 201")
    data["sources"] = source
    return data
