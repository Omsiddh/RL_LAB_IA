"""Evaluate unchanged policies on identical workloads without learning."""

from contextlib import contextmanager
import hashlib
import random
from statistics import fmean

import numpy as np
import torch

from src.baselines import run_fcfs, run_round_robin, run_srtf
from src.env.rl_adapter import RLAdapter


POLICIES = ("DQN", "FCFS", "SRTF", "Round Robin")


def weights_digest(agent):
    digest = hashlib.sha256()
    for network in (agent.policy_network, agent.target_network):
        for name, tensor in network.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def greedy_evaluation(agent, seed):
    original_rng = agent.rng
    original_mode = agent.policy_network.training
    try:
        agent.rng = random.Random(seed)
        agent.policy_network.eval()
        with torch.no_grad():
            yield
    finally:
        agent.rng = original_rng
        agent.policy_network.train(original_mode)


def _run_dqn(workload, agent, workload_id, decisions, trace=None):
    adapter = RLAdapter()
    observation = adapter.reset(workload)
    step = 0
    while not adapter.env.is_done():
        mask = adapter.valid_action_mask()
        action = agent.select_action(observation, mask, epsilon=0.0)
        if trace is not None:
            q_values = agent.policy_network(observation).detach().cpu().tolist()
        # Diagnostic data uses only currently observed ready processes and does
        # not feed anything back into action selection.
        ready = np.flatnonzero(mask).tolist()
        remaining = {index: adapter.env.processes[index].remaining_time for index in ready}
        minimum = min(remaining.values())
        shortest = [index for index in ready if remaining[index] == minimum]
        equivalent = [index for index in ready if np.array_equal(
            observation[index * 6:index * 6 + 6], observation[action * 6:action * 6 + 6])]
        for index in ready:
            decisions.append({
                "workload_id": workload_id, "process_count": len(workload),
                "step": step, "time": adapter.env.time, "process_index": index,
                "selected": index == action, "ready_count": len(ready),
                "remaining_time": remaining[index],
                "unique_shortest": len(ready) > 1 and len(shortest) == 1 and index in shortest,
                "equivalent_choice_candidate": len(equivalent) > 1 and index in equivalent,
                "equivalent_choice_expected": 1 / len(equivalent) if len(equivalent) > 1 and index in equivalent else 0.0,
            })
        next_observation, reward, done, info = adapter.step(action)
        if trace is not None:
            trace.append({
                "step": step, "time": info["start_time"],
                "observation": observation.tolist(), "action_mask": mask.tolist(),
                "q_values": q_values,
                "masked_q_values": [value if valid else None
                                    for value, valid in zip(q_values, mask)],
                "selected_action": action, "selected_pid": info["pid"],
                "epsilon": 0.0, "exploratory": False, "reward": reward,
                "waiting_process_indices": info["waiting_indices"],
                "execution": {"start": info["start_time"], "end": info["end_time"],
                              "process_index": action, "pid": info["pid"]},
                "context_switch": info["context_switch"], "preemption": info["preemption"],
                "process_completed": info["completed"], "done": done,
                "idle_interval_after": (dict(zip(("start", "end", "process_index", "pid"),
                                                  info["idle_interval"]))
                                        if info["idle_interval"] is not None else None),
                "next_time": adapter.env.time,
                "next_observation": next_observation.tolist(),
                "next_action_mask": info["action_mask"].tolist(),
                "cumulative_reward": adapter.env.total_reward,
            })
        observation = next_observation
        step += 1
    metrics = adapter.env.get_metrics()
    processes = metrics["processes"]
    return {
        "algorithm": "DQN", "timeline": tuple(adapter.env.timeline), "processes": processes,
        **{"mean_" + key: fmean(process[key] for process in processes.values())
           for key in ("waiting_time", "turnaround_time", "response_time")},
        "context_switches": metrics["context_switches"], "preemptions": metrics["preemptions"],
        "total_reward": metrics["total_reward"], "total_time": metrics["time"],
        "idle_time": metrics["idle_time"],
    }


def evaluate_workloads(workloads, agent, validation_seed=1729, traces=None):
    """Replay the supplied immutable definitions with their original indices.

    Returns one row of complete simulator results per workload and observational
    DQN decision records. No workload generation, permutation, or training occurs.
    If supplied, traces receives one list of real DQN transitions per workload.
    """
    if not workloads or any(not workload for workload in workloads):
        raise ValueError("Benchmark requires nonempty workloads")
    definitions_before = tuple(tuple(tuple(process) for process in workload) for workload in workloads)
    before = weights_digest(agent)
    optimizer_updates = agent.optimizer_steps
    replay_size = len(agent.replay_buffer)
    replay_rng = agent.replay_buffer.rng.getstate()
    results, decisions = [], []
    with greedy_evaluation(agent, validation_seed):
        for workload_id, workload in enumerate(workloads):
            trace = [] if traces is not None else None
            result = {"DQN": _run_dqn(workload, agent, workload_id, decisions, trace),
                      "FCFS": run_fcfs(workload), "SRTF": run_srtf(workload),
                      "Round Robin": run_round_robin(workload, quantum=2)}
            for policy in POLICIES:
                metrics = result[policy]
                if set(metrics["processes"]) != {pid for pid, _, _ in workload}:
                    raise AssertionError("Policies must evaluate identical process definitions")
                waiting = sum(p["waiting_time"] for p in metrics["processes"].values())
                if metrics["total_reward"] != -waiting:
                    raise AssertionError("Reward must equal negative total waiting time")
                executed = [entry for entry in metrics["timeline"] if entry[2] is not None]
                if len(executed) != sum(burst for _, _, burst in workload):
                    raise AssertionError("CPU burst accounting failed")
                if any(workload[index][0] != pid for _, _, index, pid in executed):
                    raise AssertionError("Original process indices changed")
                optimal_waiting = sum(p["waiting_time"] for p in result["SRTF"]["processes"].values())
                if optimal_waiting > waiting:
                    raise AssertionError("SRTF waiting-time optimality inequality failed")
            results.append(result)
            if traces is not None:
                traces.append(trace)
    if weights_digest(agent) != before:
        raise AssertionError("Benchmark modified network weights")
    if (agent.optimizer_steps != optimizer_updates or len(agent.replay_buffer) != replay_size
            or agent.replay_buffer.rng.getstate() != replay_rng):
        raise AssertionError("Benchmark modified optimizer/replay state")
    if definitions_before != tuple(tuple(tuple(p) for p in w) for w in workloads):
        raise AssertionError("Benchmark modified original workloads")
    return results, decisions
