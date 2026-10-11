"""Preemptive shortest-remaining-time-first scheduling."""

from .runner import SchedulingPolicy, run_policy


class SRTFPolicy(SchedulingPolicy):
    name = "SRTF"

    def select_action(self, ready):
        shortest = min(process.remaining_time for process in ready.values())
        if self.current in ready and ready[self.current].remaining_time == shortest:
            return self.current
        return min(ready, key=lambda index: (
            ready[index].remaining_time, ready[index].arrival_time, index,
        ))


def run_srtf(workload, env=None):
    """Reconsider the shortest arrived, unfinished process after each CPU step."""
    return run_policy(workload, SRTFPolicy(), env)
