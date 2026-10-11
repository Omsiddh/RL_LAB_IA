"""Run Stage 6.2 with ``python -m src.benchmark.run``."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from src.agent import DQNAgent
from src.training.validation import Validator
from .analysis import analyze_results
from .evaluation import evaluate_workloads, weights_digest
from .plots import comparison_charts


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_validation(training_dir):
    """Use the Phase 5 reconstruction code and saved settings, unchanged."""
    config = json.loads((Path(training_dir) / "training_config.json").read_text(encoding="utf-8"))
    if config["validation_seed"] != 1729 or config["validation_workloads"] != 128:
        raise ValueError("Stage 6.2 requires the original 128-workload validation collection, seed 1729")
    workloads = Validator(config["validation_workloads"], config["validation_seed"]).workloads
    if len(workloads) != 128 or sum(map(len, workloads)) != 645:
        raise AssertionError("Phase 5 validation reconstruction must contain 128 workloads and 645 processes")
    return config, workloads


def save_outputs(output_dir, workloads, comparison, workload_rows, processes, decisions, summary):
    write_csv(output_dir / "baseline_comparison.csv", comparison)
    write_csv(output_dir / "per_workload_comparison.csv", workload_rows)
    write_csv(output_dir / "per_process_results.csv", processes)
    write_csv(output_dir / "by_process_count.csv", summary["by_process_count"])
    write_csv(output_dir / "slot_bias.csv", summary["slot_bias"]["statistics"])
    write_csv(output_dir / "dqn_decision_diagnostics.csv", decisions)
    write_json(output_dir / "validation_workloads.json", workloads)
    write_json(output_dir / "benchmark_summary.json", summary)
    comparison_charts(comparison, output_dir)


def run_benchmark(training_dir="runs/phase5_full_6000", output_dir="runs/stage6_2_benchmark"):
    training_dir, output_dir = Path(training_dir), Path(output_dir)
    config, workloads = reconstruct_validation(training_dir)
    checkpoint = training_dir / "best_model.pt"
    metadata_path = training_dir / "best_model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["episode"] != 5250:
        raise ValueError("Stage 6.2 requires the best checkpoint selected at episode 5250")
    with (training_dir / "validation_metrics.csv").open(newline="", encoding="utf-8") as stream:
        logged = next(row for row in csv.DictReader(stream) if int(row["episode"]) == metadata["episode"])
    protected_paths = [checkpoint, metadata_path, training_dir / "training_config.json",
                       training_dir / "validation_metrics.csv"]
    for folder in ("src/env", "src/agent", "src/training", "src/baselines"):
        protected_paths.extend(sorted(Path(folder).rglob("*.py")))
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_paths}
    # Fail before evaluation if artifacts would be overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["torch_threads"])
        agent = DQNAgent(device="cpu")
        agent.load(checkpoint)
        if agent.optimizer_steps != metadata["optimizer_updates"] or agent.config != config["agent_config"]:
            raise AssertionError("Checkpoint counters/configuration differ from Phase 5 metadata")
        digest = weights_digest(agent)
        results, decisions = evaluate_workloads(workloads, agent, config["validation_seed"])
        comparison, workload_rows, processes, summary = analyze_results(workloads, results, decisions)
        dqn = comparison[0]
        for key in ("mean_waiting_time", "mean_turnaround_time", "mean_reward"):
            if not math.isclose(dqn[key], metadata["validation"][key], rel_tol=0, abs_tol=1e-12):
                raise AssertionError("DQN benchmark does not reproduce checkpoint validation: " + key)
            if not math.isclose(dqn[key], float(logged[key]), rel_tol=0, abs_tol=1e-12):
                raise AssertionError("DQN benchmark differs from Phase 5 validation CSV: " + key)
        if not math.isclose(dqn["mean_waiting_time"], 6.586046511627907, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("Unexpected DQN validation waiting time")
        # Reuse the same collection object; do not reconstruct another dataset.
        repeated, repeated_decisions = evaluate_workloads(workloads, agent, config["validation_seed"])
        if repeated != results or repeated_decisions != decisions:
            raise AssertionError("Benchmark is not deterministic")
        workload_json = json.dumps(workloads, separators=(",", ":")).encode()
        summary.update({
            "stage": "6.2", "dataset_role": "Phase 5 model-selection validation; not final test",
            "checkpoint": str(checkpoint.resolve()), "checkpoint_episode": metadata["episode"],
            "validation_seed": config["validation_seed"], "workload_count": 128, "process_count": 645,
            "workload_id_convention": "Zero-based index in original Phase 5 validation collection",
            "validation_workload_sha256": hashlib.sha256(workload_json).hexdigest(),
            "dqn_epsilon": 0.0, "round_robin_quantum": 2, "device": "cpu",
            "checks": {"weights_unchanged": weights_digest(agent) == digest,
                       "optimizer_updates_unchanged": agent.optimizer_steps == metadata["optimizer_updates"],
                       "replay_buffer_empty": len(agent.replay_buffer) == 0,
                       "deterministic_repeat_exact": True, "original_indices_preserved": True,
                       "reward_consistency_all_policies": True, "srtf_inequalities_all_workloads": True,
                       "dqn_validation_score_reproduced": True},
            "protected_input_sha256": protected,
            "limitations": ["Model-selection validation data; not an unbiased final-test estimate.",
                            "One checkpoint, one fixed workload collection, zero context-switch overhead.",
                            "Slot analysis is observational and cannot establish a causal slot effect.",
                            "Percentage gaps with zero SRTF waiting are null unless both waits are zero."],
        })
        save_outputs(output_dir, workloads, comparison, workload_rows, processes, decisions, summary)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in protected.items()):
            raise AssertionError("Protected source or Phase 5 artifact changed")
        summary["checks"]["protected_inputs_unchanged"] = True
        write_json(output_dir / "benchmark_summary.json", summary)
    finally:
        torch.set_num_threads(previous_threads)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("runs/phase5_full_6000"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stage6_2_benchmark"))
    args = parser.parse_args(argv)
    summary = run_benchmark(args.training_dir, args.output_dir)
    print(json.dumps({"comparison": summary["comparison"],
                      "optimality_gap": {key: value for key, value in summary["optimality_gap"].items()
                                         if "example" not in key}, "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
