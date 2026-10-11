"""Shared simulator execution and standardized completed-run results."""

from statistics import fmean

from src.env.environment import CPUSchedulingEnv


class SchedulingPolicy:
    """Policies receive only ready process objects keyed by stable indices."""

    def reset(self):
        self.current = None

    def select_action(self, ready):
        raise NotImplementedError

    def after_step(self, action, completed):
        self.current = None if completed else action


def run_policy(workload, policy, env=None):
    """Reset and replay a workload; all CPU execution belongs to env.step.

    The optional environment permits reuse and inspection. Neither the complete
    workload nor unarrived process records are provided to the policy. The
    simulator alone handles idle jumps, completion, rewards, and switches.
    """
    if env is None:
        env = CPUSchedulingEnv()
    env.reset(workload)
    policy.reset()
    while not env.is_done():
        ready = {index: env.processes[index] for index in env.valid_actions()}
        action = policy.select_action(ready)
        _, _, _, info = env.step(action)
        policy.after_step(action, info["completed"])

    metrics = env.get_metrics()
    processes = metrics["processes"]

    def mean_metric(name):
        return fmean(process[name] for process in processes.values()) if processes else 0.0

    return {
        "algorithm": policy.name,
        # Keep the simulator's individual CPU steps and idle intervals exactly.
        "timeline": tuple(env.timeline), "processes": processes,
        "mean_waiting_time": mean_metric("waiting_time"),
        "mean_turnaround_time": mean_metric("turnaround_time"),
        "mean_response_time": mean_metric("response_time"),
        "context_switches": metrics["context_switches"],
        "preemptions": metrics["preemptions"],
        "total_reward": metrics["total_reward"],
        "total_time": metrics["time"], "idle_time": metrics["idle_time"],
    }
