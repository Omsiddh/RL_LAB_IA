"""Fixed choices and manifests, saved before generating or evaluating test data."""

import csv
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import greedy_evaluation
from src.benchmark.run import reconstruct_validation, write_json
from src.env.rl_adapter import RLAdapter
from src.permutation.experiment import agent_snapshot
from src.training.validation import sample_workload


POLICIES = ("SRTF", "Augmented DQN 10k", "Original DQN 10k", "Reference DQN 6k", "FCFS", "Round Robin")
MODELS = {
    "Augmented DQN 10k": ("runs/phase6_3b_controlled/augmented_10000/best_model.pt", 9750),
    "Original DQN 10k": ("runs/phase6_3b_controlled/original_10000/best_model.pt", 9000),
    "Reference DQN 6k": ("runs/phase5_full_6000/best_model.pt", 5250),
}
TEST_SEED = 6401729
BOOTSTRAP_SEED = 64021729
EVALUATION_SEED = 640342


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workload_digest(workloads):
    return hashlib.sha256(json.dumps(workloads, separators=(",", ":")).encode()).hexdigest()


def protected_manifest(output_dir):
    """Hash current sources/configs and all existing experiment files."""
    output_dir = Path(output_dir).resolve()
    paths = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    paths += [Path(name) for name in ("README.md", "pyproject.toml", "requirements.txt")]
    paths += [p for p in Path("runs").rglob("*") if p.is_file() and "__pycache__" not in p.parts
              and not p.resolve().is_relative_to(output_dir)]
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths))}


def freeze_protocol(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Final test output directory must be new; previous artifacts are preserved")
    manifest, agents = [], {}
    # Previously recorded hashes establish the exact already-selected files.
    prior = json.loads(Path("runs/phase6_3b_controlled/controlled_summary.json").read_text())
    prior_hashes = {str(Path(path).resolve()): digest for path, digest in prior["protected_input_sha256"].items()}
    for policy, (relative, episode) in MODELS.items():
        checkpoint = Path(relative).resolve()
        metadata_path = checkpoint.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata["episode"] != episode:
            raise AssertionError("Selected checkpoint episode differs from frozen choice: " + policy)
        if sha256(checkpoint) != prior_hashes[str(checkpoint)]:
            raise AssertionError("Selected checkpoint changed since Stage 6.3B: " + policy)
        with (checkpoint.parent / "validation_metrics.csv").open(newline="") as stream:
            logged = next(row for row in csv.DictReader(stream) if int(row["episode"]) == episode)
        if not all(float(logged[key]) == metadata["validation"][key] for key in
                   ("mean_waiting_time", "mean_turnaround_time", "mean_reward")):
            raise AssertionError("Checkpoint metadata differs from its validation log")
        agent = DQNAgent(device="cpu")
        agent.load(checkpoint)
        if not all(torch.isfinite(tensor).all().item() for network in (agent.policy_network, agent.target_network)
                   for tensor in network.state_dict().values()):
            raise AssertionError("Loaded checkpoint contains non-finite weights")
        if agent.optimizer_steps != metadata["optimizer_updates"] or agent.config != metadata["configuration"]["agent_config"]:
            raise AssertionError("Checkpoint optimizer counters/configuration differ from metadata")
        # A synthetic, one-process load/mask probe, before any test data exists.
        before = agent_snapshot(agent)
        adapter = RLAdapter()
        observation = adapter.reset((("load_probe", 0, 1),))
        with greedy_evaluation(agent, EVALUATION_SEED):
            if agent.select_action(observation, adapter.valid_action_mask(), epsilon=0.0) != 0:
                raise AssertionError("Loaded greedy checkpoint failed valid-action probe")
        if agent_snapshot(agent) != before:
            raise AssertionError("Load probe changed model/optimizer/replay/RNG state")
        agents[policy] = agent
        manifest.append({"policy": policy, "role": "primary" if policy == POLICIES[1] else "equal-budget control" if policy == POLICIES[2] else "historical reference",
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "metadata_sha256": sha256(metadata_path), "selected_episode": episode,
            "metadata": metadata, "loaded_state": before, "greedy_load_probe_passed": True})
    protocol = {
        "stage": "6.4", "frozen_before_test_generation_and_evaluation": True,
        "models_fixed_by_user": True, "policies_in_execution_order": POLICIES,
        "primary_contrast": ["Augmented DQN 10k", "Original DQN 10k"],
        "secondary_waiting_contrasts": [[POLICIES[1], other] for other in ("SRTF", "FCFS", "Round Robin", "Reference DQN 6k")],
        "test_collection": {"workload_count": 256, "seed": TEST_SEED,
            "seed_origin": "No prior collection or seed found in repository source/configuration/experiment records; this single dedicated seed was chosen before evaluation",
            "generator": "src.training.validation.sample_workload",
            "sampling": "Local Random(test_seed), per workload getrandbits(64) -> existing generate_workload -> one training-style RNG shuffle",
            "process_count_range": [4, 6], "arrival_time_range": [0, 8], "burst_time_range": [1, 10],
            "duplicates": "Record duplicates across validation/test, including identities/order-independent arrival-burst configurations; never redraw"},
        "evaluation": {"epsilon": 0.0, "device": "cpu", "torch_threads": 1,
            "action_rng_seed": EVALUATION_SEED, "gradients": False, "round_robin_quantum": 2,
            "training_optimizer_replay_updates": False, "context_switch_overhead": 0,
            "repeat": "Same frozen collection and same loaded checkpoints, exact outcome/timeline equality; not a new sample"},
        "aggregation": {"timing": "Total per-process time / total processes", "switches_preemptions_reward_means": "Totals / workload count",
            "srtf_percentage_gap": "100 * (policy total waiting - SRTF total waiting) / SRTF total waiting; zero/zero = 0, positive/zero = null"},
        "statistics": {"seed": BOOTSTRAP_SEED, "replicates": 5000, "confidence": .95,
            "method": "Paired percentile workload bootstrap; a common index matrix retains all six algorithms in each draw. Recalculate summed waiting differences / summed process counts.",
            "groups": ["all", 4, 5, 6], "secondary_tradeoff_intervals": ["response_time", "context_switches", "preemptions"],
            "multiplicity": "Primary contrast declared in advance; secondary/subgroup intervals descriptive, without multiplicity correction"},
        "example_selection": "Best/worst by augmented minus original per-workload mean waiting; lowest workload ID breaks ties. Store all six timelines for those examples only.",
        "boundaries": ["No training, tuning, model reselection, or additional test seeds", "No polished Gantt figures, Stage 6.5 narrative, or Phase 7"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
    }
    protected = protected_manifest(output_dir)
    protocol["source_configuration_and_prior_artifact_sha256"] = protected
    output_dir.mkdir(parents=True, exist_ok=False)
    # These files are written once, before generating any held-out workload.
    write_json(output_dir / "selected_model_manifest.json", manifest)
    write_json(output_dir / "frozen_protocol.json", protocol)
    return protocol, manifest, agents


def prepare_test_workloads(output_dir, protocol):
    rng = random.Random(protocol["test_collection"]["seed"])
    workloads = tuple(sample_workload(rng) for _ in range(protocol["test_collection"]["workload_count"]))
    _, validation = reconstruct_validation("runs/phase5_full_6000")
    validation_exact = set(validation)
    # Attribute multiset ignores PID labels and slot ordering and catches the
    # same physical scheduling instance represented with another slot layout.
    signature = lambda w: tuple(sorted((arrival, burst) for _, arrival, burst in w))
    validation_signatures = {signature(w) for w in validation}
    exact_count = sum(w in validation_exact for w in workloads)
    configuration_count = sum(signature(w) in validation_signatures for w in workloads)
    digest = workload_digest(workloads)
    if digest == workload_digest(validation) or len(workloads) != 256:
        raise AssertionError("Test collection must be distinct from the fixed validation collection")
    metadata = {**protocol["test_collection"], "process_count": sum(map(len, workloads)),
        "process_count_workloads": {str(n): sum(len(w) == n for w in workloads) for n in (4, 5, 6)},
        "workload_sha256": digest, "validation_workload_sha256": workload_digest(validation),
        "exact_cross_split_duplicate_workloads": exact_count,
        "arrival_burst_multiset_cross_split_duplicate_workloads": configuration_count,
        "within_test_exact_duplicate_workloads": len(workloads) - len(set(workloads)),
        "within_test_arrival_burst_multiset_duplicates": len(workloads) - len({signature(w) for w in workloads}),
        "no_duplicate_filtering_or_redraw": True}
    output_dir = Path(output_dir)
    write_json(output_dir / "test_workloads.json", workloads)
    write_json(output_dir / "test_workload_manifest.json", metadata)
    loaded = tuple(tuple(tuple(p) for p in w) for w in json.loads((output_dir / "test_workloads.json").read_text()))
    if loaded != workloads or workload_digest(loaded) != digest:
        raise AssertionError("Saved test definitions did not round-trip exactly")
    return workloads, metadata
