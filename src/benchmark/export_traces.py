"""Export actual checkpoint decisions for JSON import and playback."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from main import DEFAULT_CHECKPOINT, PROJECT_ROOT, TEST_WORKLOADS
from src.agent import DQNAgent
from src.env.rl_adapter import RLAdapter
from .evaluation import evaluate_workloads, weights_digest


def json_result(result):
    """Name timeline fields explicitly for a browser importer."""
    return {**result, "timeline": [
        dict(zip(("start", "end", "process_index", "pid"), interval))
        for interval in result["timeline"]
    ]}


def export_traces(workload_ids=(0, 69, 201), checkpoint=DEFAULT_CHECKPOINT,
                  output_dir=PROJECT_ROOT / "exports/dqn_traces"):
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir)
    workloads_bytes = TEST_WORKLOADS.read_bytes()
    workloads = json.loads(workloads_bytes)
    ids = list(workload_ids)
    if (not ids or any(type(i) is not int or not 0 <= i < len(workloads) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"Choose distinct workload IDs between 0 and {len(workloads) - 1}")
    selected = tuple(tuple(tuple(process) for process in workloads[i]) for i in ids)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent = DQNAgent(device="cpu")
    agent.load(checkpoint)
    traces = []
    results, _ = evaluate_workloads(selected, agent, traces=traces)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise AssertionError("Checkpoint file changed during export")
    adapter = RLAdapter()
    environment = {
        "cpu_cores": 1, "cpu_step_duration": 1, "time_unit": "simulated CPU unit",
        "io_blocking": False, "context_switch_overhead": 0,
        "idle_time_handling": "automatically jump to the next arrival",
        "reward": "negative number of other arrived unfinished processes waiting",
        "max_processes": adapter.max_processes,
        "observation_size": 39, "action_count": 6,
        "process_features": ["ready", "completed", "normalized_original_burst",
                             "normalized_remaining_burst", "normalized_waiting",
                             "last_executed"],
        "global_features": ["normalized_time", "arrived_fraction", "completed_fraction"],
        "normalization": {"max_burst_time": adapter.max_burst_time,
                          "max_arrival_time": adapter.max_arrival_time,
                          "max_episode_duration": adapter.max_episode_duration},
    }
    model = {
        "algorithm": "DQN", "source": "pytorch_checkpoint_inference",
        "checkpoint": (checkpoint.relative_to(PROJECT_ROOT).as_posix()
                       if checkpoint.is_relative_to(PROJECT_ROOT) else checkpoint.name),
        "checkpoint_sha256": checkpoint_hash, "network_weights_sha256": weights_digest(agent),
        "architecture": [39, 64, 64, 6], "hidden_activation": "ReLU",
        "epsilon": 0.0, "decision_mode": "masked_greedy", "inference_only": True,
        "checkpoint_optimizer_updates": agent.optimizer_steps, "training_seed": agent.config["seed"],
    }
    payloads = []
    for workload_id, workload, steps, result in zip(ids, selected, traces, results):
        payload = {
            "schema_version": 1, "trace_type": "cpu_scheduling_dqn",
            "workload_id": workload_id, "model": model, "environment": environment,
            "test_workloads_file_sha256": hashlib.sha256(workloads_bytes).hexdigest(),
            "workload": [{"process_index": index, "pid": pid, "arrival_time": arrival,
                          "burst_time": burst}
                         for index, (pid, arrival, burst) in enumerate(workload)],
            "decisions": steps,
            **json_result(result["DQN"]),
            "baselines": {name: json_result(result[name])
                          for name in ("FCFS", "SRTF", "Round Robin")},
            "round_robin_quantum": 2,
        }
        # Strict JSON uses null for masked outputs, never Infinity or NaN.
        payloads.append((workload_id, json.dumps(payload, indent=2, allow_nan=False) + "\n"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for workload_id, text in payloads:
        path = output_dir / f"workload_{workload_id}.json"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-ids", type=int, nargs="+", default=[0, 69, 201])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exports/dqn_traces")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        paths = export_traces(args.workload_ids, args.checkpoint, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path}: {len(payload['decisions'])} real DQN decisions, "
              f"waiting={payload['mean_waiting_time']:.6f}, reward={payload['total_reward']}")


if __name__ == "__main__":
    main()
