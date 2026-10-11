"""Fixed validation workloads and isolated greedy evaluation."""

import math
import random

import torch

from src.env.rl_adapter import RLAdapter
from src.env.workload import generate_workload


def sample_workload(rng):
    """Assign slots by a random permutation independent of process attributes."""
    workload = list(generate_workload(seed=rng.getrandbits(64)))
    rng.shuffle(workload)
    return tuple(workload)


def episode_metrics(adapter, reward, steps):
    """Read completed-process metrics only after all scheduling decisions."""
    if not adapter.env.is_done() or not adapter.env.processes:
        raise ValueError("Metrics require a completed, nonempty workload")
    processes = adapter.env.processes
    waiting = sum(process.get_waiting_time() for process in processes)
    turnaround = sum(process.get_turnaround_time() for process in processes)
    if (reward != -waiting or reward != adapter.env.total_reward
            or steps != sum(process.burst_time for process in processes)):
        raise AssertionError("Scheduling reward or episode-length invariant failed")
    metrics = {
        "reward": reward, "process_count": len(processes),
        "mean_waiting_time": waiting / len(processes),
        "mean_turnaround_time": turnaround / len(processes),
        "episode_length": steps,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Episode metrics must be finite")
    return metrics


class Validator:
    def __init__(self, workload_count=128, seed=1729):
        if type(workload_count) is not int or workload_count < 1:
            raise ValueError("workload_count must be a positive integer")
        self.seed = seed
        rng = random.Random(seed)
        self.workloads = tuple(sample_workload(rng) for _ in range(workload_count))

    def evaluate(self, agent):
        """Return process-weighted means, preserving training RNG and mode.

        select_action draws from its RNG even at epsilon zero. Give it a local
        validation RNG for this call, then restore the original RNG object.
        No replay insertion, optimization, or target synchronization occurs.
        """
        training_rng = agent.rng
        training_mode = agent.policy_network.training
        total_processes = total_waiting = total_turnaround = total_reward = 0
        try:
            agent.rng = random.Random(self.seed)
            agent.policy_network.eval()
            with torch.no_grad():
                adapter = RLAdapter()
                for workload in self.workloads:
                    state = adapter.reset(workload)
                    done = False
                    reward_sum = steps = 0
                    while not done:
                        action = agent.select_action(state, adapter.valid_action_mask(), epsilon=0.0)
                        state, reward, done, _ = adapter.step(action)
                        reward_sum += reward
                        steps += 1
                    metrics = episode_metrics(adapter, reward_sum, steps)
                    total_processes += metrics["process_count"]
                    total_waiting += -reward_sum
                    total_turnaround += sum(p.get_turnaround_time() for p in adapter.env.processes)
                    total_reward += reward_sum
        finally:
            agent.rng = training_rng
            agent.policy_network.train(training_mode)
        return {
            "workload_count": len(self.workloads), "process_count": total_processes,
            "mean_waiting_time": total_waiting / total_processes,
            "mean_turnaround_time": total_turnaround / total_processes,
            "mean_reward": total_reward / len(self.workloads),
        }
