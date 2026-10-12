"""Compare all three selected policies on identical workloads and frozen states."""

from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import greedy_evaluation
from src.benchmark.run import reconstruct_validation, write_csv, write_json
from .analysis import aggregate_rollouts, analyze, bootstrap_ratio
from .controlled import protected_inputs
from .experiment import agent_snapshot, collect_states, fixed_state_experiment, tree_digest, workload_experiment
from .run import file_digests, save_outputs
from .transforms import validate_permutation


MODEL_NAMES = ("reference_5250", "original_10000_best", "augmented_10000_best")


def collect_shared_states(workloads, reference):
    """Freeze the Stage 6.3A reference/SRTF distribution, never each new model's."""
    with greedy_evaluation(reference, 1729):
        return {(wid, source): collect_states(workload, reference, source, maximum=8)
                for wid, workload in enumerate(workloads) for source in ("DQN", "SRTF")}


def save_state_bank(path, bank):
    rows = [(wid, source, sid, state) for (wid, source), states in bank.items()
            for sid, state in enumerate(states)]
    np.savez_compressed(path, observations=np.stack([state["observation"] for _, _, _, state in rows]),
        masks=np.stack([state["mask"] for _, _, _, state in rows]),
        workload_ids=np.array([wid for wid, _, _, _ in rows]),
        source_policies=np.array([source for _, source, _, _ in rows]),
        state_ids=np.array([sid for _, _, sid, _ in rows]),
        steps=np.array([state["step"] for _, _, _, state in rows]),
        times=np.array([state["time"] for _, _, _, state in rows]))


def load_state_bank(path):
    bank = defaultdict(list)
    with np.load(path, allow_pickle=False) as stored:
        for i, wid in enumerate(stored["workload_ids"]):
            key = int(wid), str(stored["source_policies"][i])
            if int(stored["state_ids"][i]) != len(bank[key]):
                raise AssertionError("Invalid fixed-state ordering")
            bank[key].append({"observation": stored["observations"][i].copy(), "mask": stored["masks"][i].copy(),
                              "step": int(stored["steps"][i]), "time": int(stored["times"][i])})
    return dict(bank)


def csv_cells(rows):
    """Compare serialized outputs with previous artifacts, including float precision."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return list(csv.DictReader(stream))


def paired_comparison(candidate_name, reference_name, evaluated, workloads, seed=63021729, replicates=5000):
    candidate, reference = evaluated[candidate_name], evaluated[reference_name]
    rows = []
    for group in ("all", 4, 5, 6):
        ids = [i for i, w in enumerate(workloads) if group == "all" or len(w) == group]
        if not ids:
            continue
        count = [len(workloads[i]) for i in ids]
        ci = {}
        for label, data in (("candidate", candidate), ("reference", reference)):
            rollout_index = {(r["workload_id"], r["arrangement_id"]): r for r in data["rollouts"] if r["policy"] == "DQN"}
            sensitivity = {r["workload_id"]: r for r in data["sensitivity"]}
            agreements, comparisons = defaultdict(int), defaultdict(int)
            for action in data["actions"]:
                if action["arrangement_id"]:
                    agreements[action["workload_id"]] += action["action_agreement"]
                    comparisons[action["workload_id"]] += 1
            ci[label] = {"index": rollout_index, "sensitivity": sensitivity,
                         "agree": agreements, "comparisons": comparisons}
        left, right = ci["candidate"], ci["reference"]
        if any(left["comparisons"][i] != right["comparisons"][i] for i in ids):
            raise AssertionError("Models did not evaluate the same number of fixed states/layouts")
        contributions = {
            "original_waiting": ([left["index"][i, 0]["total_waiting_time"] - right["index"][i, 0]["total_waiting_time"] for i in ids], count),
            "permuted_waiting": ([sum(left["index"][i, a]["total_waiting_time"] - right["index"][i, a]["total_waiting_time"] for a in range(1, 8)) / 7 for i in ids], count),
            "fixed_state_agreement": ([left["agree"][i] - right["agree"][i] for i in ids], [left["comparisons"][i] for i in ids]),
            "waiting_range": ([left["sensitivity"][i]["waiting_range"] - right["sensitivity"][i]["waiting_range"] for i in ids], [1] * len(ids)),
            "waiting_std": ([left["sensitivity"][i]["waiting_population_std"] - right["sensitivity"][i]["waiting_population_std"] for i in ids], [1] * len(ids)),
        }
        for metric, key, process_weighted in (("original_response", "mean_response_time", True),
                ("original_context_switches", "context_switches", False), ("original_preemptions", "preemptions", False)):
            contributions[metric] = ([
                (left["index"][i, 0][key] - right["index"][i, 0][key]) * (len(workloads[i]) if process_weighted else 1)
                for i in ids], count if process_weighted else [1] * len(ids))
        for metric, (numerators, denominators) in contributions.items():
            rows.append({"candidate": candidate_name, "reference": reference_name, "process_count_group": group,
                         "metric": metric, **bootstrap_ratio(numerators, denominators, seed, replicates)})
    return rows


def evaluate_pair(root):
    root = Path(root)
    destination = root / "evaluation"
    if destination.exists():
        raise FileExistsError("Evaluation results already exist; preserve them")
    training = json.loads((root / "training_comparison.json").read_text())
    _, workloads = reconstruct_validation("runs/phase5_full_6000")
    prior_dir = Path("runs/phase6_3_permutation")
    previous = json.loads((prior_dir / "permutation_summary.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(workloads, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != previous["validation_workload_sha256"]:
        raise AssertionError("Validation collection differs from Stage 6.3A")
    raw_maps = json.loads((prior_dir / "slot_permutations.json").read_text())
    permutations = tuple(tuple(validate_permutation(p) for p in maps) for maps in raw_maps)
    if len(permutations) != 128 or any(len(ps) != 8 or len(set(ps)) != 8 or ps[0] != tuple(range(len(w)))
            or any(len(p) != len(w) for p in ps) for w, ps in zip(workloads, permutations)):
        raise AssertionError("Invalid saved Stage 6.3A mappings")
    protected = protected_inputs()
    trained_files = [p for name in ("original_10000", "augmented_10000") for p in (root / name).rglob("*") if p.is_file()]
    protected.update(file_digests(trained_files))
    destination.mkdir()
    checkpoints = {"reference_5250": Path("runs/phase5_full_6000/best_model.pt"),
                   "original_10000_best": root / "original_10000/best_model.pt",
                   "augmented_10000_best": root / "augmented_10000/best_model.pt"}
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        reference = DQNAgent()
        reference.load(checkpoints["reference_5250"])
        before_reference = agent_snapshot(reference)
        bank = collect_shared_states(workloads, reference)
        if agent_snapshot(reference) != before_reference:
            raise AssertionError("State collection changed reference model")
        save_state_bank(destination / "fixed_state_observations.npz", bank)
        reloaded = load_state_bank(destination / "fixed_state_observations.npz")
        if tree_digest(bank) != tree_digest(reloaded):
            raise AssertionError("Frozen state bank serialization changed observations")
        bank_digest, workload_digest = tree_digest(bank), tree_digest(workloads)
        evaluated, comparison, by_count = {}, [], []
        for name, checkpoint in checkpoints.items():
            print("EVALUATE " + name + " using shared reference/SRTF observations and saved Stage 6.3A maps", flush=True)
            agent = DQNAgent()
            agent.load(checkpoint)
            before = agent_snapshot(agent)
            with greedy_evaluation(agent, 1729):
                actions = fixed_state_experiment(workloads, permutations, agent, state_collection=bank)
                rollouts, processes = workload_experiment(workloads, permutations, agent)
            with greedy_evaluation(agent, 1729):
                repeated_actions = fixed_state_experiment(workloads, permutations, agent, state_collection=bank)
                repeated_rollouts, repeated_processes = workload_experiment(workloads, permutations, agent)
            if tree_digest((actions, rollouts, processes)) != tree_digest((repeated_actions, repeated_rollouts, repeated_processes)):
                raise AssertionError("Evaluation is not deterministic")
            if agent_snapshot(agent) != before or tree_digest(bank) != bank_digest or tree_digest(workloads) != workload_digest:
                raise AssertionError("Evaluation changed agent state, shared observations, or workload definitions")
            if name == "reference_5250":
                for filename, rows in (("action_equivariance.csv", actions), ("permutation_rollouts.csv", rollouts), ("per_process_results.csv", processes)):
                    with (prior_dir / filename).open(newline="") as stream:
                        if csv_cells(rows) != list(csv.DictReader(stream)):
                            raise AssertionError("Reference does not exactly reproduce Stage 6.3A: " + filename)
            summary, sensitivity = analyze(workloads, actions, rollouts)
            metadata = json.loads(checkpoint.with_suffix(".json").read_text())
            original = summary["full_rollout"]["original_DQN"]
            for key in ("mean_waiting_time", "mean_turnaround_time", "mean_reward"):
                if not math.isclose(original[key], metadata["validation"][key], rel_tol=0, abs_tol=1e-12):
                    raise AssertionError("Selected checkpoint differs from logged validation")
            summary.update({"stage": "6.3B", "model": name, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_episode": metadata["episode"], "validation_workload_sha256": fingerprint,
                "shared_state_bank_digest": bank_digest, "shared_state_sources": "episode-5250 reference DQN and SRTF only",
                "agent_state_before": before, "agent_state_after": agent_snapshot(agent),
                "checks": {"agent_state_unchanged": True, "same_frozen_state_bank": True,
                    "same_saved_permutations": True, "checkpoint_score_reproduced": True,
                    "deterministic_repeat_exact": True, "reward_identity": True, "srtf_optimality": True}})
            target = destination / name
            target.mkdir()
            save_outputs(target, summary, sensitivity, actions, rollouts, processes, workloads, permutations)
            evaluated[name] = {"summary": summary, "actions": actions, "rollouts": rollouts, "sensitivity": sensitivity}
            for arrangement, key in (("original", "original_DQN"), ("seven_alternatives", "permuted_DQN")):
                metrics = summary["full_rollout"][key]
                srtf = summary["full_rollout"]["original_SRTF"]["mean_waiting_time"]
                comparison.append({"model": name, "best_episode": metadata["episode"], "arrangements": arrangement,
                    **metrics, "srtf_waiting": srtf, "srtf_absolute_waiting_gap": metrics["mean_waiting_time"] - srtf,
                    "srtf_percentage_waiting_gap": 100 * (metrics["mean_waiting_time"] - srtf) / srtf,
                    "fixed_state_agreement_percent": summary["full_rollout"]["action_agreement_percent"],
                    "changed_workloads_percent": summary["full_rollout"]["changed_workloads_percent"],
                    "mean_waiting_range": summary["full_rollout"]["mean_waiting_range"],
                    "mean_waiting_std": summary["full_rollout"]["mean_waiting_population_std"]})
            for row in summary["by_process_count"]:
                group_metrics = {}
                for label, policy, permuted in (("original", "DQN", False), ("permuted", "DQN", True),
                                                ("srtf", "SRTF", False)):
                    grouped = aggregate_rollouts([r for r in rollouts if r["policy"] == policy
                        and r["process_count"] == row["process_count_group"]
                        and (r["arrangement_id"] > 0 if permuted else r["arrangement_id"] == 0)])
                    group_metrics.update({label + "_" + key: value for key, value in grouped.items()})
                by_count.append({"model": name, "best_episode": metadata["episode"], **row, **group_metrics,
                    "srtf_original_absolute_gap": row["dqn_original_waiting"] - row["srtf_waiting"],
                    "srtf_original_percentage_gap": 100 * (row["dqn_original_waiting"] - row["srtf_waiting"]) / row["srtf_waiting"],
                    "srtf_permuted_absolute_gap": row["dqn_permuted_waiting"] - row["srtf_waiting"],
                    "srtf_permuted_percentage_gap": 100 * (row["dqn_permuted_waiting"] - row["srtf_waiting"]) / row["srtf_waiting"]})
        paired = []
        for candidate_name, reference_name in (("augmented_10000_best", "original_10000_best"),
                ("original_10000_best", "reference_5250"), ("augmented_10000_best", "reference_5250")):
            paired.extend(paired_comparison(candidate_name, reference_name, evaluated, workloads))
        result = {"stage": "6.3B", "validation_only": True, "workload_count": 128, "process_count": 645,
            "validation_workload_sha256": fingerprint, "comparison": comparison, "by_process_count": by_count,
            "paired_model_comparison": paired, "training": training,
            "shared_state_bank_digest": bank_digest, "fixed_original_states": sum(len(states) for states in bank.values()),
            "fixed_permutation_comparisons_per_model": evaluated["reference_5250"]["summary"]["fixed_state"]["comparisons"],
            "bootstrap": {"seed": 63021729, "replicates": 5000, "confidence": .95,
                "method": "Paired percentile workload-cluster bootstrap, retaining all states/arrangements per workload",
                "limitations": "Descriptive, no multiplicity correction; workload variation conditional on these selected checkpoints, not training-seed uncertainty"},
            "checks": {"same_validation_workloads": True, "same_saved_stage6_3a_maps": True,
                "same_frozen_observations_for_all_models": True, "reference_exactly_reproduces_stage6_3a": True,
                "all_evaluations_repeat_exactly": True, "model_optimizer_replay_unchanged_during_evaluation": True,
                "srtf_waiting_invariant_and_optimal_all_arrangements": True, "all_reward_identities": True,
                "previous_sources_and_artifacts_preserved": True}, "protected_input_sha256": protected,
            "limitations": ["One paired training seed and one augmentation seed; no estimate of between-training-seed variability.",
                "Best checkpoints selected on the same original validation collection used here; results are not an unbiased final-test estimate.",
                "The common fixed-state distribution comes from the reference policy and SRTF, not the new policies' own trajectories.",
                "Replay augmentation changes coordinate coverage and transition distribution, not the number of samples or optimizer updates.",
                "Random layout effects may cancel in aggregate; trajectory slot associations are observational."]}
        if file_digests([Path(p) for p in protected]) != protected:
            raise AssertionError("Protected sources or artifacts changed during evaluation")
        write_csv(root / "original_validation_comparison.csv", [r for r in comparison if r["arrangements"] == "original"])
        write_csv(root / "permutation_comparison.csv", comparison)
        write_csv(root / "by_process_count.csv", by_count)
        write_csv(root / "paired_model_comparison.csv", paired)
        write_json(root / "controlled_summary.json", result)
        from .controlled_plots import comparison_plots
        comparison_plots(root, result)
        from .controlled_report import write_report
        write_report(root, result)
    finally:
        torch.set_num_threads(original_threads)
    print(json.dumps({"comparison": comparison, "checks": result["checks"]}, indent=2), flush=True)
    return result
