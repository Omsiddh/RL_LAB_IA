"""Read-only policy evaluation with independent completed-timeline checks."""

from contextlib import ExitStack
import random
from statistics import fmean

import numpy as np
import torch

from src.baselines import run_fcfs, run_round_robin, run_srtf
from src.benchmark.evaluation import greedy_evaluation
from src.env.rl_adapter import RLAdapter
from src.permutation.experiment import agent_snapshot, tree_digest
from .protocol import EVALUATION_SEED, MODELS, POLICIES, workload_digest


def global_rng_snapshot():
    return tree_digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def run_dqn(workload, agent):
    """Choose masked greedy actions; delegate every CPU tick to the adapter."""
    adapter = RLAdapter()
    observation = adapter.reset(workload)
    while not adapter.env.is_done():
        mask = adapter.valid_action_mask()
        action = agent.select_action(observation, mask, epsilon=0.0)
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(mask) or not mask[action]:
            raise AssertionError("DQN action violates the ready-process mask")
        observation, _, _, _ = adapter.step(action)
    metrics = adapter.env.get_metrics()
    processes = metrics["processes"]
    return {"algorithm": "DQN", "timeline": tuple(adapter.env.timeline), "processes": processes,
            **{"mean_" + key: fmean(p[key] for p in processes.values())
               for key in ("waiting_time", "turnaround_time", "response_time")},
            **{key: metrics[key] for key in ("context_switches", "preemptions", "total_reward", "idle_time")},
            "total_time": metrics["time"]}


def verify_result(workload, result):
    """Check recorded execution against immutable definitions, not policy code."""
    if set(result["processes"]) != {p[0] for p in workload}:
        raise AssertionError("Policy evaluated different process identities")
    runtime = [0] * len(workload)
    first, completion, cursor, idle = {}, {}, 0, 0
    for start, end, index, pid in result["timeline"]:
        if start != cursor or end <= start:
            raise AssertionError("Timeline has gaps, overlap, or invalid durations")
        cursor = end
        if index is None:
            if pid is not None:
                raise AssertionError("Idle timeline has a process identity")
            idle += end - start
            continue
        if not 0 <= index < len(workload) or workload[index][0] != pid:
            raise AssertionError("Stable process index/identity changed")
        if start < workload[index][1]:
            raise AssertionError("Process executed before arrival")
        if end - start != 1:
            raise AssertionError("CPU execution must use simulator unit steps")
        runtime[index] += end - start
        first.setdefault(pid, start)
        completion[pid] = end
    if runtime != [p[2] for p in workload]:
        raise AssertionError("Process CPU-burst accounting failed")
    if cursor != result["total_time"] or idle != result["idle_time"] or cursor != sum(runtime) + idle:
        raise AssertionError("Total CPU/idle time accounting failed")
    waiting = 0
    for pid, arrival, burst in workload:
        process = result["processes"][pid]
        if process["first_start_time"] != first[pid] or process["completion_time"] != completion[pid]:
            raise AssertionError("Process completion/start differs from timeline")
        if (process["turnaround_time"] != completion[pid] - arrival
                or process["turnaround_time"] != process["waiting_time"] + burst
                or process["response_time"] != first[pid] - arrival
                or process["waiting_time"] < 0):
            raise AssertionError("Process waiting/turnaround/response identity failed")
        waiting += process["waiting_time"]
    if result["total_reward"] != -waiting:
        raise AssertionError("Reward must equal negative total waiting time")
    for key in ("waiting_time", "turnaround_time", "response_time"):
        if result["mean_" + key] != fmean(p[key] for p in result["processes"].values()):
            raise AssertionError("Reported process means differ from per-process results")


def evaluate_all(workloads, agents, progress=None):
    """Use one frozen collection for six policies, restoring every agent RNG."""
    if not workloads or any(not w for w in workloads):
        raise ValueError("Evaluation requires nonempty workloads")
    if set(agents) != set(MODELS):
        raise ValueError("Exactly the three preselected DQN policies are required")
    before = {policy: agent_snapshot(agent) for policy, agent in agents.items()}
    original_rng_objects = {policy: agent.rng for policy, agent in agents.items()}
    original_replay_objects = {policy: agent.replay_buffer for policy, agent in agents.items()}
    workload_before, rng_before = workload_digest(workloads), global_rng_snapshot()
    results = []
    with ExitStack() as stack:
        for agent in agents.values():
            stack.enter_context(greedy_evaluation(agent, EVALUATION_SEED))
        for workload_id, workload in enumerate(workloads):
            result = {}
            for policy in POLICIES:
                if policy in agents:
                    value = run_dqn(workload, agents[policy])
                elif policy == "SRTF":
                    value = run_srtf(workload)
                elif policy == "FCFS":
                    value = run_fcfs(workload)
                else:
                    value = run_round_robin(workload, quantum=2)
                verify_result(workload, value)
                result[policy] = value
            optimal = -result["SRTF"]["total_reward"]
            if any(optimal > -value["total_reward"] for value in result.values()):
                raise AssertionError("SRTF waiting-time optimality inequality failed")
            results.append(result)
            if progress and (workload_id + 1) % 64 == 0:
                progress(f"Evaluated {workload_id + 1}/{len(workloads)} workloads for all six policies")
    after = {policy: agent_snapshot(agent) for policy, agent in agents.items()}
    if before != after:
        raise AssertionError("Evaluation changed model, optimizer, replay, RNG, gradients or mode")
    if any(agent.rng is not original_rng_objects[policy] or agent.replay_buffer is not original_replay_objects[policy]
           for policy, agent in agents.items()):
        raise AssertionError("Evaluation replaced a persistent RNG or replay object")
    if global_rng_snapshot() != rng_before:
        raise AssertionError("Evaluation changed global Python/NumPy/Torch RNG state")
    if workload_digest(workloads) != workload_before:
        raise AssertionError("Evaluation changed original workload definitions")
    return results, {"completed_rollouts": len(workloads) * len(POLICIES),
        "identical_workloads_and_stable_indices": True, "all_processes_complete": True,
        "no_execution_before_arrival": True, "cpu_burst_and_total_time_accounting": True,
        "reward_equals_negative_waiting_every_rollout": True, "turnaround_equals_waiting_plus_burst_every_process": True,
        "srtf_inequality_violations": 0, "all_dqn_actions_mask_valid": True,
        "agent_state_before": before, "agent_state_after": after,
        "model_optimizer_replay_rng_gradients_modes_unchanged": True,
        "global_rng_unchanged": True, "workload_sha256_before": workload_before,
        "workload_sha256_after": workload_digest(workloads)}
