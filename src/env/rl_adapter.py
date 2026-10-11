import numpy as np

from .environment import CPUSchedulingEnv


class RLAdapter:

    def __init__(self, env=None, max_burst_time=10, max_arrival_time=8, max_episode_duration=None):
        self.max_processes = 6
        if type(max_burst_time) is not int or max_burst_time < 1:
            raise ValueError("max_burst_time must be a positive integer")
        if type(max_arrival_time) is not int or max_arrival_time < 0:
            raise ValueError("max_arrival_time must be a nonnegative integer")
        duration_bound = max_arrival_time + self.max_processes * max_burst_time
        if max_episode_duration is None:
            max_episode_duration = duration_bound
        if (type(max_episode_duration) is not int
                or max_episode_duration < duration_bound):
            raise ValueError("max_episode_duration must cover the configured workload bounds")

        if env is None:
            env = CPUSchedulingEnv()
        self.env = env
        self.max_burst_time = max_burst_time
        self.max_arrival_time = max_arrival_time
        self.max_episode_duration = max_episode_duration
        self.last_executed_index = None

    def reset(self, workload):
        definitions = tuple(workload)
        if len(definitions) > self.max_processes:
            raise ValueError("At most six processes are supported")
        for entry in definitions:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise ValueError("Each process must be (pid, arrival_time, burst_time)")
            pid, arrival, burst = entry
            if type(arrival) is not int or not 0 <= arrival <= self.max_arrival_time:
                raise ValueError("Arrival time is outside the configured bounds")
            if type(burst) is not int or not 1 <= burst <= self.max_burst_time:
                raise ValueError("Burst time is outside the configured bounds")

        self.env.reset(definitions)
        self.last_executed_index = None
        return self.get_observation()

    def valid_action_mask(self):
        mask = np.zeros(self.max_processes, dtype=bool)
        for index in self.env.valid_actions():
            mask[index] = True
        return mask

    def get_observation(self):
        observation = np.zeros(39, dtype=np.float32)
        state = self.env.get_state()
        arrived_count = 0
        completed_count = 0
        for process in state["processes"]:
            if not process["arrived"]:
                continue
            index = process["index"]
            offset = index * 6 # Each process has 6 features to track so just how you manage pixel data in engine it converts data into 1D matrix/vector
            arrived_count += 1
            remaining = process["remaining_time"]
            burst = process["burst_time"]
            end_time = state["time"]
            if remaining > 0:
                observation[offset] = 1
            else:
                observation[offset + 1] = 1
                completed_count += 1
                end_time = process["completion_time"]

            # Elapsed time since arrival = executed time + waiting time.
            executed_time = burst - remaining 
            waiting_time = end_time - process["arrival_time"] - executed_time
            observation[offset + 2] = burst / self.max_burst_time # Normalized Burst Time
            observation[offset + 3] = remaining / self.max_burst_time # Normalized Remaining Burst Time
            observation[offset + 4] = waiting_time / self.max_episode_duration # # Normalized Waiting Time
            if index == self.last_executed_index:
                observation[offset + 5] = 1

        observation[36] = state["time"] / self.max_episode_duration
        observation[37] = arrived_count / self.max_processes
        observation[38] = completed_count / self.max_processes
        return observation

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.last_executed_index = action
        info = info.copy()
        info["action_mask"] = self.valid_action_mask().copy()
        return self.get_observation(), reward, done, info
