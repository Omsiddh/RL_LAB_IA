"""Non-preemptive first-come, first-served scheduling."""

from .runner import SchedulingPolicy, run_policy


class FCFSPolicy(SchedulingPolicy):
    name = "FCFS"

    def select_action(self, ready):
        if self.current in ready:
            return self.current
        return min(ready, key=lambda index: (ready[index].arrival_time, index))


def run_fcfs(workload, env=None):
    """Replay a workload with arrival/index ties and execution to completion."""
    return run_policy(workload, FCFSPolicy(), env)
