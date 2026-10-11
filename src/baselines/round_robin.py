"""FIFO Round Robin with arrivals admitted before quantum-expiry requeue."""

from collections import deque

from .runner import SchedulingPolicy, run_policy


class RoundRobinPolicy(SchedulingPolicy):
    name = "Round Robin"

    def __init__(self, quantum=2):
        if type(quantum) is not int or quantum < 1:
            raise ValueError("quantum must be a positive integer")
        self.quantum = quantum

    def reset(self):
        super().reset()
        self.queue = deque()
        self.admitted = set()
        self.used = 0

    def select_action(self, ready):
        # Only already-arrived processes enter the queue. Keep old waiters in
        # place and append new arrivals chronologically, with stable-index ties.
        newcomers = sorted((index for index in ready if index not in self.admitted),
                           key=lambda index: (ready[index].arrival_time, index))
        self.queue.extend(newcomers)
        self.admitted.update(newcomers)

        if self.current is not None:
            if self.used < self.quantum:
                return self.current
            # New arrivals at this boundary have already been enqueued.
            self.queue.append(self.current)
            self.current = None

        self.current = self.queue.popleft()
        self.used = 0
        return self.current

    def after_step(self, action, completed):
        self.used += 1
        super().after_step(action, completed)
        if completed:
            self.used = 0


def run_round_robin(workload, quantum=2, env=None):
    """Replay a workload with FIFO rotation and a positive integer quantum."""
    return run_policy(workload, RoundRobinPolicy(quantum), env)
