"""Execute the frozen Stage 6.4 protocol once, plus a same-collection repeat."""

import argparse
import json
from pathlib import Path

import torch

from src.benchmark.run import write_csv, write_json
from src.permutation.experiment import tree_digest
from .analysis import analyze, bootstrap_intervals
from .evaluation import evaluate_all
from .plots import comparison_charts
from .protocol import freeze_protocol, prepare_test_workloads, sha256
from .report import write_report


def verify_protected(protected):
    changed = [path for path, expected in protected.items() if not Path(path).is_file() or sha256(path) != expected]
    if changed:
        raise AssertionError("Protected source/prior artifact changed: " + ", ".join(changed))
    return len(protected)


def run_final_evaluation(output_dir="runs/phase6_4_final_test", test_results="Full test suite result is recorded separately."):
    output_dir = Path(output_dir)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        protocol, models, agents = freeze_protocol(output_dir)
        frozen_hashes = {name: sha256(output_dir / name) for name in ("frozen_protocol.json", "selected_model_manifest.json")}
        print("Protocol and model manifest frozen; generating the single predeclared test collection", flush=True)
        workloads, metadata = prepare_test_workloads(output_dir, protocol)
        print(f"Saved {len(workloads)} workloads, {metadata['process_count']} processes; beginning evaluation", flush=True)
        results, checks = evaluate_all(workloads, agents, progress=lambda message: print(message, flush=True))
        print("Repeating the same frozen collection for exact determinism verification", flush=True)
        repeated, repeat_checks = evaluate_all(workloads, agents, progress=lambda message: print(message, flush=True))
        if results != repeated or checks != repeat_checks:
            raise AssertionError("Repeated evaluation differs from original results/state checks")
        comparison, wide, processes, grouped, paired, examples, summary = analyze(workloads, results)
        intervals = bootstrap_intervals(workloads, results, protocol["statistics"]["seed"], protocol["statistics"]["replicates"])
        if intervals != bootstrap_intervals(workloads, results, protocol["statistics"]["seed"], protocol["statistics"]["replicates"]):
            raise AssertionError("Bootstrap does not reproduce deterministically")
        checks.update({"deterministic_repeat_exact": True, "deterministic_bootstrap_exact": True,
            "unique_workloads_evaluated": len(workloads), "repeat_is_same_sample": True,
            "evaluation_result_sha256": tree_digest(results), "per_process_row_count": len(processes),
            "protected_source_and_prior_artifact_count": verify_protected(protocol["source_configuration_and_prior_artifact_sha256"]),
            "protected_sources_validation_and_prior_artifacts_unchanged": True,
            "full_test_suite": test_results, "protocol_manifest_unchanged_after_evaluation": True})
        summary.update({"stage": "6.4", "dataset_role": "Final held-out test; no model selection",
            "test_collection": metadata, "selected_models": [{k: model[k] for k in ("policy", "checkpoint", "checkpoint_sha256", "selected_episode")} for model in models],
            "bootstrap_seed": protocol["statistics"]["seed"], "bootstrap_intervals": intervals,
            "checks": checks, "dqn_epsilon": 0.0, "round_robin_quantum": 2})
        for filename, rows in (("policy_comparison.csv", comparison), ("per_workload_comparison.csv", wide),
                ("per_process_results.csv", processes), ("by_process_count.csv", grouped),
                ("paired_model_comparison.csv", paired), ("bootstrap_intervals.csv", intervals)):
            write_csv(output_dir / filename, rows)
        write_json(output_dir / "selected_timelines.json", examples)
        write_json(output_dir / "benchmark_summary.json", summary)
        write_json(output_dir / "verification_report.json", checks)
        comparison_charts(comparison, grouped, output_dir)
        write_report(output_dir, summary, test_results)
        verify_protected(protocol["source_configuration_and_prior_artifact_sha256"])
        if any(sha256(output_dir / name) != digest for name, digest in frozen_hashes.items()):
            raise AssertionError("Frozen protocol or model manifest changed after evaluation")
        print(json.dumps({"comparison": comparison, "augmentation_outcomes": summary["augmentation_workload_outcomes"],
            "primary_interval": next(r for r in intervals if r["primary"]), "checks_passed": True}, indent=2), flush=True)
        return summary
    finally:
        torch.set_num_threads(threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase6_4_final_test"))
    parser.add_argument("--test-results", default="Full test suite result is recorded separately.")
    args = parser.parse_args(argv)
    run_final_evaluation(args.output_dir, args.test_results)


if __name__ == "__main__":
    main()
