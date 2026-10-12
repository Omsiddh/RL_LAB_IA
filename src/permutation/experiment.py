"""Separate fixed-state interventions from complete scheduling trajectories."""

import hashlib
import json

import numpy as np
import torch

from src.baselines import run_srtf
from src.baselines.srtf import SRTFPolicy
from src.env.rl_adapter import RLAdapter
from src.training.validation import episode_metrics
from .transforms import inverse_permutation, permute_observation, permute_workload


def tree_digest(value):
    """Value-based digest, including optimizer tensors and replay arrays."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\x00")

    visit(value)
    return digest.hexdigest()


def agent_snapshot(agent):
    return {
        "policy_weights": tree_digest(agent.policy_network.state_dict()),
        "target_weights": tree_digest(agent.target_network.state_dict()),
        "optimizer": tree_digest(agent.optimizer.state_dict()),
        "optimizer_updates": agent.optimizer_steps,
        "replay": tree_digest(list(agent.replay_buffer._transitions)),
        "replay_rng": tree_digest(agent.replay_buffer.rng.getstate()),
        "action_rng": tree_digest(agent.rng.getstate()),
        "gradients": tree_digest([parameter.grad for parameter in agent.policy_network.parameters()]),
        "policy_mode": agent.policy_network.training,
    }


def collect_states(workload, agent, source, maximum=8):
    """Evenly sample multi-ready decision states, or one single-ready fallback.

    Rollouts are original, unpermuted trajectories. SRTF selection is delegated
    to the existing policy; RLAdapter supplies observations on that trajectory.
    """
    if maximum < 1:
        raise ValueError("maximum must be positive")
    adapter = RLAdapter()
    observation = adapter.reset(workload)
    policy = SRTFPolicy()
    policy.reset()
    states = []
    step = 0
    while not adapter.env.is_done():
        mask = adapter.valid_action_mask()
        states.append({"observation": observation.copy(), "mask": mask.copy(),
                       "time": adapter.env.time, "step": step})
        if source == "DQN":
            action = agent.select_action(observation, mask, epsilon=0.0)
        elif source == "SRTF":
            ready = {index: adapter.env.processes[index] for index in adapter.env.valid_actions()}
            action = policy.select_action(ready)
        else:
            raise ValueError("State source must be DQN or SRTF")
        observation, _, _, info = adapter.step(action)
        policy.after_step(action, info["completed"])
        step += 1
    candidates = [state for state in states if state["mask"].sum() > 1]
    if not candidates:
        return states[:1]
    indices = np.linspace(0, len(candidates) - 1, min(maximum, len(candidates)), dtype=int)
    return [candidates[index] for index in indices]


def fixed_state_experiment(workloads, permutations, agent, maximum=8, close_threshold=0.05, tie_tolerance=1e-6,
                           state_collection=None):
    """Optionally evaluate a shared frozen collection, independent of this agent's rollout."""
    rows = []
    for workload_id, workload in enumerate(workloads):
        arrangements = permutations[workload_id]
        for source in ("DQN", "SRTF"):
            states = (collect_states(workload, agent, source, maximum) if state_collection is None
                      else state_collection[workload_id, source])
            for state_id, state in enumerate(states):
                transformed = [permute_observation(state["observation"], state["mask"], p) for p in arrangements]
                observations = np.stack([item[0] for item in transformed])
                masks = np.stack([item[1] for item in transformed])
                q_values = agent.policy_network(observations).detach().cpu().numpy()
                if not np.isfinite(q_values).all():
                    raise AssertionError("Non-finite fixed-state Q-values")
                masked = np.where(masks, q_values, -np.inf)
                selected = masked.argmax(axis=1)
                margins = []
                for values, mask in zip(q_values, masks):
                    valid = np.sort(values[mask])
                    margins.append(float(valid[-1] - valid[-2]) if len(valid) > 1 else None)
                original_selected = int(selected[0])
                ready = np.flatnonzero(state["mask"])
                remaining = state["observation"][ready * 6 + 3]
                shortest = ready[remaining == remaining.min()]
                unique = int(shortest[0]) if len(ready) > 1 and len(shortest) == 1 else None
                for arrangement_id, permutation in enumerate(arrangements):
                    chosen = int(selected[arrangement_id])
                    identity = permutation[chosen]
                    inverse = inverse_permutation(permutation)
                    margin = margins[arrangement_id]
                    agree = identity == original_selected
                    finite_margins = [value for value in (margins[0], margin) if value is not None]
                    minimum_margin = min(finite_margins) if finite_margins else None
                    classification = ("agreement" if agree else "tied" if minimum_margin <= tie_tolerance
                                      else "close" if minimum_margin <= close_threshold else "clear")
                    mapped_q = q_values[arrangement_id][list(inverse)][ready]
                    rows.append({
                        "workload_id": workload_id, "process_count": len(workload),
                        "source_policy": source, "state_id": state_id, "step": state["step"], "time": state["time"],
                        "ready_count": len(ready), "arrangement_id": arrangement_id,
                        "new_slot_to_original": json.dumps(permutation),
                        "original_action_index": original_selected, "original_action_pid": workload[original_selected][0],
                        "original_action_new_slot": inverse[original_selected],
                        "selected_slot": chosen, "selected_original_index": identity,
                        "selected_pid": workload[identity][0], "action_agreement": agree,
                        "original_top_two_margin": margins[0], "permuted_top_two_margin": margin,
                        "preference_class": classification,
                        "maximum_identity_q_difference": float(np.max(np.abs(mapped_q - q_values[0][ready]))),
                        "unique_shortest_original_index": unique,
                        "unique_shortest_new_slot": inverse[unique] if unique is not None else None,
                        "unique_shortest_selected": identity == unique if unique is not None else None,
                    })
    return rows


def run_dqn(workload, agent):
    adapter = RLAdapter()
    state = adapter.reset(workload)
    reward_sum = step = 0
    unique_counts = {index: [0, 0] for index in range(len(workload))}
    while not adapter.env.is_done():
        mask = adapter.valid_action_mask()
        action = agent.select_action(state, mask, epsilon=0.0)
        ready = np.flatnonzero(mask)
        remaining = state[ready * 6 + 3]
        shortest = ready[remaining == remaining.min()]
        if len(ready) > 1 and len(shortest) == 1:
            index = int(shortest[0])
            unique_counts[index][0] += 1
            unique_counts[index][1] += action == index
        state, reward, _, _ = adapter.step(action)
        reward_sum += reward
        step += 1
    episode_metrics(adapter, reward_sum, step)
    metrics = adapter.env.get_metrics()
    return {"processes": metrics["processes"], "timeline": tuple(adapter.env.timeline),
            "context_switches": metrics["context_switches"], "preemptions": metrics["preemptions"],
            "total_reward": metrics["total_reward"], "total_time": metrics["time"],
            "unique_counts": unique_counts}


def workload_experiment(workloads, permutations, agent):
    rows, process_rows = [], []
    for workload_id, workload in enumerate(workloads):
        original_indices = {pid: index for index, (pid, _, _) in enumerate(workload)}
        srtf_original_waiting = None
        for arrangement_id, permutation in enumerate(permutations[workload_id]):
            arranged = permute_workload(workload, permutation)
            for policy, result in (("DQN", run_dqn(arranged, agent)), ("SRTF", run_srtf(arranged))):
                metrics = result["processes"]
                waiting = sum(p["waiting_time"] for p in metrics.values())
                if result["total_reward"] != -waiting:
                    raise AssertionError("Reward/waiting identity failed")
                if policy == "SRTF":
                    if srtf_original_waiting is None:
                        srtf_original_waiting = waiting
                    if waiting != srtf_original_waiting:
                        raise AssertionError("SRTF total waiting changed under permutation")
                row = {"workload_id": workload_id, "process_count": len(workload), "policy": policy,
                       "arrangement_id": arrangement_id, "new_slot_to_original": json.dumps(permutation),
                       "total_waiting_time": waiting,
                       **{"mean_" + key: sum(p[key] for p in metrics.values()) / len(workload)
                          for key in ("waiting_time", "turnaround_time", "response_time")},
                       "context_switches": result["context_switches"], "preemptions": result["preemptions"],
                       "total_reward": result["total_reward"], "total_time": result["total_time"]}
                for slot in range(6):
                    counts = result.get("unique_counts", {}).get(slot, [0, 0])
                    row[f"slot_{slot}_unique_shortest_opportunities"] = counts[0]
                    row[f"slot_{slot}_unique_shortest_selected"] = counts[1]
                rows.append(row)
                for new_slot, (pid, arrival, burst) in enumerate(arranged):
                    if permutation[new_slot] != original_indices[pid]:
                        raise AssertionError("Identity mapping failed")
                    process_rows.append({"workload_id": workload_id, "policy": policy,
                                         "arrangement_id": arrangement_id, "new_slot": new_slot,
                                         "original_index": original_indices[pid], "pid": pid,
                                         "arrival_time": arrival, "burst_time": burst, **metrics[pid]})
            if rows[-2]["total_waiting_time"] < rows[-1]["total_waiting_time"]:
                raise AssertionError("DQN waiting below the SRTF control")
    return rows, process_rows
