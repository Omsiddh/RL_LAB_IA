"""Run Stage 6.3A: python -m src.permutation.run --output-dir <fresh directory>."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import greedy_evaluation
from src.benchmark.run import reconstruct_validation, write_csv, write_json
from .analysis import analyze
from .experiment import agent_snapshot, fixed_state_experiment, tree_digest, workload_experiment
from .plots import plot_results
from .transforms import make_permutations


def file_digests(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def save_outputs(output_dir, summary, sensitivity, actions, rollouts, processes, workloads, permutations):
    write_csv(output_dir / "action_equivariance.csv", actions)
    write_csv(output_dir / "permutation_rollouts.csv", rollouts)
    write_csv(output_dir / "per_process_results.csv", processes)
    write_csv(output_dir / "per_workload_sensitivity.csv", sensitivity)
    for key in ("by_process_count", "by_arrangement", "slot_agreement", "unique_shortest_slot_selection",
                "matched_shortest_slot_effects", "trajectory_slot_associations"):
        if summary[key]:
            write_csv(output_dir / (key + ".csv"), summary[key])
    by_mapping = defaultdict(lambda: [0, 0])
    for row in actions:
        key = row["process_count"], row["source_policy"], row["new_slot_to_original"]
        by_mapping[key][0] += row["action_agreement"]
        by_mapping[key][1] += 1
    write_csv(output_dir / "action_agreement_by_permutation.csv", [
        {"process_count": n, "source_policy": source, "new_slot_to_original": mapping,
         "agreements": a, "comparisons": b, "agreement_fraction": a / b}
        for (n, source, mapping), (a, b) in sorted(by_mapping.items())])
    write_json(output_dir / "validation_workloads.json", workloads)
    write_json(output_dir / "slot_permutations.json", permutations)
    write_json(output_dir / "permutation_summary.json", summary)
    plot_results(summary, sensitivity, output_dir)


def run_experiment(training_dir="runs/phase5_full_6000", benchmark_dir="runs/stage6_2_benchmark",
                   output_dir="runs/phase6_3_permutation", permutation_seed=6301729,
                   bootstrap_seed=63021729, bootstrap_replicates=5000, maximum_states=8):
    training_dir, benchmark_dir, output_dir = map(Path, (training_dir, benchmark_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory; previous artifacts are preserved")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1 or type(maximum_states) is not int or maximum_states < 1:
        raise ValueError("Bootstrap replicates and maximum states must be positive integers")
    config, workloads = reconstruct_validation(training_dir)
    workload_digest = tree_digest(workloads)
    fingerprint = hashlib.sha256(json.dumps(workloads, separators=(",", ":")).encode()).hexdigest()
    previous = json.loads((benchmark_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    if fingerprint != previous["validation_workload_sha256"]:
        raise AssertionError("Validation collection differs from Stage 6.2")
    metadata = json.loads((training_dir / "best_model.json").read_text(encoding="utf-8"))
    if metadata["episode"] != 5250:
        raise ValueError("Expected unchanged best checkpoint from episode 5250")
    checkpoint = training_dir / "best_model.pt"
    protected_paths = list(training_dir.glob("*")) + list(benchmark_dir.glob("*"))
    protected_paths = [path for path in protected_paths if path.is_file()]
    for folder in ("src/env", "src/agent", "src/training", "src/baselines", "src/benchmark"):
        protected_paths.extend(Path(folder).rglob("*.py"))
    protected = file_digests(sorted(protected_paths))
    permutations = tuple(make_permutations(len(workload), permutation_seed + 1000003 * workload_id)
                         for workload_id, workload in enumerate(workloads))
    output_dir.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["torch_threads"])
        agent = DQNAgent(device="cpu")
        agent.load(checkpoint)
        if agent.optimizer_steps != metadata["optimizer_updates"] or agent.config != config["agent_config"]:
            raise AssertionError("Loaded checkpoint configuration/counters differ from Phase 5")
        before = agent_snapshot(agent)
        print("Collecting fixed DQN/SRTF states and testing eight slot layouts...", flush=True)
        with greedy_evaluation(agent, config["validation_seed"]):
            actions = fixed_state_experiment(workloads, permutations, agent, maximum_states)
            print("Replaying DQN and SRTF on 1,024 workload arrangements...", flush=True)
            rollouts, processes = workload_experiment(workloads, permutations, agent)
        if agent_snapshot(agent) != before:
            raise AssertionError("Evaluation changed model, optimizer, gradients, replay, or RNG state")
        print("Repeating both experiments to check exact deterministic results...", flush=True)
        with greedy_evaluation(agent, config["validation_seed"]):
            repeated_actions = fixed_state_experiment(workloads, permutations, agent, maximum_states)
            repeated_rollouts, repeated_processes = workload_experiment(workloads, permutations, agent)
        if tree_digest((actions, rollouts, processes)) != tree_digest((repeated_actions, repeated_rollouts, repeated_processes)):
            raise AssertionError("Permutation experiments are not deterministic")
        if agent_snapshot(agent) != before or tree_digest(workloads) != workload_digest:
            raise AssertionError("Agent state or original workloads changed")
        print("Computing paired workload bootstrap intervals and exporting results...", flush=True)
        summary, sensitivity = analyze(workloads, actions, rollouts, bootstrap_seed, bootstrap_replicates)
        for policy in ("DQN", "SRTF"):
            actual = summary["full_rollout"]["original_" + policy]
            expected = next(row for row in previous["comparison"] if row["policy"] == policy)
            for key in ("mean_waiting_time", "mean_turnaround_time", "mean_response_time", "mean_context_switches", "mean_preemptions", "total_reward"):
                if not math.isclose(actual[key], expected[key], rel_tol=0, abs_tol=1e-12):
                    raise AssertionError("Original ordering differs from Stage 6.2: " + policy + " " + key)
        for key in ("mean_waiting_time", "mean_turnaround_time", "mean_reward"):
            if not math.isclose(summary["full_rollout"]["original_DQN"][key], metadata["validation"][key], rel_tol=0, abs_tol=1e-12):
                raise AssertionError("Original DQN score differs from checkpoint validation")
        summary.update({
            "stage": "6.3A", "checkpoint": str(checkpoint.resolve()), "checkpoint_episode": 5250,
            "validation_seed": 1729, "workload_count": 128, "process_count": 645,
            "validation_workload_sha256": fingerprint, "checkpoint_sha256": protected[str(checkpoint)],
            "permutation_seed": permutation_seed, "per_workload_seed_rule": "permutation_seed + 1000003 * zero_based_workload_id",
            "arrangements_per_workload": 8, "maximum_states_per_source_per_workload": maximum_states,
            "state_sampling": "Up to maximum evenly spaced multi-ready states from each original DQN and SRTF trajectory; one single-ready fallback only if none multi-ready",
            "index_convention": "Zero-based slots/workload IDs; permutation[new_slot] = original_slot; occupied includes not-yet-arrived/completed jobs; unused slots fixed",
            "q_preferences": {"tie_tolerance": 1e-6, "close_threshold": 0.05,
                "classification": "Disagreement is tied if either top-two margin <= 1e-6; otherwise close if either <= 0.05; otherwise clear. Absolute reward-unit heuristic, not statistical equivalence.",
                "inference": "All eight layouts (including original) evaluated in one batch per fixed state"},
            "bootstrap": {"seed": bootstrap_seed, "replicates": bootstrap_replicates, "confidence": 0.95,
                "method": "Percentile paired cluster bootstrap: resample workloads with replacement, retain all within-workload states/arrangements; timing delta is a process-weighted ratio",
                "multiplicity": "Intervals are descriptive and not adjusted for multiple comparisons"},
            "agent_state_before": before, "agent_state_after": agent_snapshot(agent),
            "checks": {"exact_original_collection": True, "original_scores_reproduced": True,
                "model_optimizer_replay_rng_gradients_unchanged": True, "workloads_unchanged": True,
                "deterministic_repeat_exact": True, "srtf_total_waiting_invariant_all_1024_arrangements": True,
                "reward_identity_all_2048_rollouts": True, "srtf_optimality_all_arrangements": True,
                "protected_inputs_unchanged": True},
            "protected_input_sha256": protected,
            "limitations": ["Model-selection validation collection, one checkpoint, seven sampled alternative arrangements per workload; no final test evaluation.",
                "Fixed-state layout intervention isolates action sensitivity, not its downstream scheduling cost.",
                "Full trajectories change after decisions diverge; shortest-slot exposure associations are observational and may reflect reverse causation.",
                "SRTF may assign tied jobs differently, affecting identity metrics even when total waiting is invariant.",
                "Bootstrap intervals describe variation across this collection; sampled steps and both source policies are dependent within workloads.",
                "Matched shortest-slot effects include only states with sampled layouts both inside and outside the target slot; generalization is limited."]})
        if file_digests([Path(path) for path in protected]) != protected:
            raise AssertionError("Protected checkpoint, prior experiment artifacts, or source changed")
        save_outputs(output_dir, summary, sensitivity, actions, rollouts, processes, workloads, permutations)
    finally:
        torch.set_num_threads(previous_threads)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("runs/phase5_full_6000"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("runs/stage6_2_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase6_3_permutation"))
    parser.add_argument("--permutation-seed", type=int, default=6301729)
    parser.add_argument("--bootstrap-seed", type=int, default=63021729)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--maximum-states", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run_experiment(**vars(args))
    print(json.dumps({key: summary[key] for key in ("fixed_state", "by_process_count", "checks")}, indent=2))


if __name__ == "__main__":
    main()
