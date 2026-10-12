"""Stage 6.3B fresh paired training; evaluation uses a shared frozen state bank."""

import argparse
import csv
import json
from pathlib import Path

from src.benchmark.run import write_json
from src.training.config import TrainingConfig
from .audit import audit_workloads
from .diagnostics import train_observed
from .run import file_digests


def protected_inputs():
    paths = []
    # Deliberately name only prior validation/training experiments, never held-out data.
    for folder in ("runs/phase5_full_6000", "runs/stage6_2_benchmark", "runs/phase6_3_permutation"):
        paths.extend(p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for folder in ("src/env", "src/agent", "src/baselines", "src/training"):
        paths.extend(Path(folder).rglob("*.py"))
    return file_digests(sorted(paths))


def train_pair(output_dir):
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("original_10000", "augmented_10000", "training_comparison.json")):
        raise FileExistsError("Use a fresh controlled-experiment directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_workloads()
    audit_path = output_dir / "workload_audit.json"
    if audit_path.exists():
        saved = json.loads(audit_path.read_text())
        if saved["workload_sequence_digest"] != audit["workload_sequence_digest"]:
            raise AssertionError("Saved audit differs from requested training stream")
    else:
        write_json(audit_path, audit)
    print("AUDIT: workload definitions are already shuffled once; no new workload shuffle. "
          "Replay augmentation uses its own RNG seed 630342.", flush=True)
    protected = protected_inputs()
    write_json(output_dir / "preservation_manifest.json", protected)
    summaries = {}
    for label, mode in (("original_10000", "none"), ("augmented_10000", "permutation")):
        print("START FRESH RUN " + label, flush=True)
        config = TrainingConfig(episodes=10000, replay_augmentation=mode, augmentation_seed=630342)
        summaries[label] = train_observed(config, output_dir / label)
        if (summaries[label]["workload_sequence_digest"] != audit["workload_sequence_digest"]
                or summaries[label]["environment_steps"] != audit["expected_environment_steps"]
                or summaries[label]["optimizer_updates"] != 68326):
            raise AssertionError("Workload stream, steps, or update counts differ from the controlled design")
    # Strong regression check: the new default path must reproduce the historical
    # uninterrupted 6,000-episode run, including every loss and gradient mean.
    prefixes = {}
    for name in ("training_metrics.csv", "validation_metrics.csv"):
        with (Path("runs/phase5_full_6000") / name).open(newline="") as stream:
            previous = list(csv.DictReader(stream))
        with (output_dir / "original_10000" / name).open(newline="") as stream:
            current = list(csv.DictReader(stream))
        prefixes[name] = previous == current[:len(previous)]
        if not prefixes[name]:
            raise AssertionError("Non-augmented behavior differs from historical Phase 5 prefix: " + name)
    if protected_inputs() != protected:
        raise AssertionError("Protected prior artifacts or sources changed during training")
    result = {"runs": summaries, "historical_prefix_exact": prefixes,
              "same_training_workload_stream": True, "same_environment_steps_and_update_counts": True,
              "protected_inputs_unchanged": True, "augmentation_seed": 630342}
    write_json(output_dir / "training_comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase6_3b_controlled"))
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args(argv)
    if args.phase in ("train", "all"):
        train_pair(args.output_dir)
    if args.phase in ("evaluate", "all"):
        from .controlled_evaluation import evaluate_pair
        evaluate_pair(args.output_dir)


if __name__ == "__main__":
    main()
