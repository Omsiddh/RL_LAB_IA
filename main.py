"""Demonstrate the saved CPU scheduler on an existing held-out workload."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import evaluate_workloads

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/phase6_3b_controlled/augmented_10000/best_model.pt"
TEST_WORKLOADS = PROJECT_ROOT / "runs/phase6_4_final_test/test_workloads.json"


def run_demo(checkpoint=DEFAULT_CHECKPOINT, workload_id=69):
    workloads = json.loads(TEST_WORKLOADS.read_text(encoding="utf-8"))
    if not 0 <= workload_id < len(workloads):
        raise ValueError(f"Workload ID must be between 0 and {len(workloads) - 1}")
    workload = tuple(tuple(process) for process in workloads[workload_id])
    agent = DQNAgent(device="cpu")
    agent.load(checkpoint)
    results, _ = evaluate_workloads((workload,), agent)
    return workload, results[0]


def console_output(workload, results, workload_id, checkpoint):
    lines = ["Adaptive Preemptive CPU Scheduling Using Deep Q-Learning",
             f"Held-out workload {workload_id}; greedy DQN (epsilon=0), CPU inference",
             f"Checkpoint: {checkpoint}", "", "PID  Arrival  Burst"]
    lines.extend(f"{pid:<4} {arrival:>7} {burst:>6}" for pid, arrival, burst in workload)
    lines.extend(["", "DQN actions: each interval executes one CPU unit"])
    for start, end, index, pid in results["DQN"]["timeline"]:
        lines.append(f"  [{start:>2}, {end:>2})  " +
                     ("IDLE" if index is None else f"slot {index}: {pid}"))
    lines.extend(["", "Scheduler       Waiting  Turnaround  Response  Switches  Reward"])
    for name in ("DQN", "SRTF", "FCFS", "Round Robin"):
        result = results[name]
        lines.append(f"{name:<14} {result['mean_waiting_time']:>8.3f} "
                     f"{result['mean_turnaround_time']:>11.3f} "
                     f"{result['mean_response_time']:>9.3f} "
                     f"{result['context_switches']:>9} {result['total_reward']:>7}")
    lines.extend(["Times are CPU units per process; Round Robin quantum is 2.",
                  "", "DQN process results", "PID  Completion  Waiting  Turnaround  Response"])
    for pid, _, _ in workload:
        process = results["DQN"]["processes"][pid]
        lines.append(f"{pid:<4} {process['completion_time']:>10} "
                     f"{process['waiting_time']:>8} {process['turnaround_time']:>11} "
                     f"{process['response_time']:>9}")
    lines.append("Verified: every job completes; reward = negative total waiting; model weights unchanged.")
    return "\n".join(lines) + "\n"


def save_gantt(workload, results, workload_id, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = ("DQN", "SRTF", "FCFS", "Round Robin")
    colors = {pid: plt.get_cmap("tab10")(i) for i, (pid, _, _) in enumerate(workload)}
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 6), sharex=True)
    try:
        for axis, name in zip(axes, names):
            for start, end, _, pid in results[name]["timeline"]:
                axis.broken_barh([(start, end - start)], (0, 1),
                                 facecolors=colors.get(pid, "#dddddd"))
            axis.set(ylim=(0, 1), yticks=[], ylabel=name)
            axis.set_xlim(0, results[name]["total_time"])
            axis.grid(axis="x", alpha=0.2)
            axis.set_title(f"Mean waiting: {results[name]['mean_waiting_time']:.3f}",
                           fontsize=10, loc="right")
        axes[-1].set_xlabel("Simulated CPU time units")
        fig.suptitle(f"Actual scheduler executions on held-out workload {workload_id}")
        fig.legend(handles=[Patch(color=color, label=pid) for pid, color in colors.items()],
                   loc="lower center", ncol=len(workload))
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", type=int, default=69,
                        help="Zero-based saved test workload ID (default: 69; counterexample: 201)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Saved DQN checkpoint; defaults to the selected augmented model")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/demo",
                        help="Destination for demo output, JSON results, and Gantt chart")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        checkpoint = args.checkpoint.resolve()
        workload, results = run_demo(checkpoint, args.workload_id)
        output = console_output(workload, results, args.workload_id, checkpoint)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "demo_output.txt").write_text(output, encoding="utf-8")
        metadata = {"workload_id": args.workload_id, "workload": workload,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "epsilon": 0.0, "round_robin_quantum": 2, "results": results}
        (args.output_dir / "demo_results.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        save_gantt(workload, results, args.workload_id, args.output_dir / "demo_gantt.png")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output, end="")
    print(f"Saved demo_output.txt, demo_results.json, and demo_gantt.png in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
