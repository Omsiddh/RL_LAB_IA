class Process:

    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.first_start_time = None
        self.completion_time = None

    def get_turnaround_time(self):
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    def get_waiting_time(self):
        if self.completion_time is None:
            return None
        return self.get_turnaround_time() - self.burst_time

    def get_response_time(self):
        if self.first_start_time is None:
            return None
        return self.first_start_time - self.arrival_time


class CPUSchedulingEnv:
    def __init__(self):
        self.reset([])

    def reset(self, workload):
        definitions = []
        used_pids = []
        for entry in workload:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise ValueError("Each process must be (pid, arrival_time, burst_time)")
            pid, arrival_time, burst_time = entry
            if type(pid) not in (str, int) or pid == "":
                raise ValueError("PID must be an integer or a nonempty string")
            if pid in used_pids:
                raise ValueError("PIDs must be unique")
            if type(arrival_time) is not int or arrival_time < 0:
                raise ValueError("Arrival time must be a nonnegative integer")
            if type(burst_time) is not int or burst_time <= 0:
                raise ValueError("Burst time must be a positive integer")
            used_pids.append(pid)
            definitions.append((pid, arrival_time, burst_time))

        self.workload = tuple(definitions)
        self.processes = []
        for pid, arrival_time, burst_time in self.workload:
            self.processes.append(Process(pid, arrival_time, burst_time))
        self.time = 0
        self.timeline = []
        self.context_switches = 0
        self.preemptions = 0
        self.total_reward = 0
        self.idle_time = 0
        self.previous_index = None
        self.skip_idle()
        return self.get_state()

    def is_done(self):
        for process in self.processes:
            if process.remaining_time > 0:
                return False
        return True

    def valid_actions(self):
        ready = []
        for index, process in enumerate(self.processes):
            if process.arrival_time <= self.time and process.remaining_time > 0:
                ready.append(index)
        return ready

    def get_state(self, debug=False):
        process_states = []
        for index, process in enumerate(self.processes):
            arrived = process.arrival_time <= self.time
            burst_time = None
            remaining_time = None

            if arrived or debug:
                burst_time = process.burst_time
                remaining_time = process.remaining_time
            process_states.append({
                "index": index,
                "pid": process.pid,
                "arrival_time": process.arrival_time,
                "arrived": arrived,
                "burst_time": burst_time,
                "remaining_time": remaining_time,
                "first_start_time": process.first_start_time,
                "completion_time": process.completion_time,
            })
        return {
            "time": self.time,
            "processes": process_states,
            "valid_actions": self.valid_actions(),
            "done": self.is_done(),
            "context_switches": self.context_switches,
            "preemptions": self.preemptions,
            "total_reward": self.total_reward,
            "idle_time": self.idle_time,
        }

    def skip_idle(self):
        if self.is_done() or self.valid_actions():
            return None
        next_arrival = None
        for process in self.processes:
            if process.remaining_time > 0:
                if next_arrival is None or process.arrival_time < next_arrival:
                    next_arrival = process.arrival_time

        # Entries are (start, end, process_index, pid)
        # Start is included, end is excluded, None means idle
        interval = (self.time, next_arrival, None, None)
        self.timeline.append(interval)
        self.idle_time += next_arrival - self.time
        self.time = next_arrival
        self.previous_index = None
        return interval

    def step(self, action):
        ready = self.valid_actions()

        if type(action) is not int or action not in ready:
            raise ValueError("Action must be a valid ready-process index")

        process = self.processes[action]
        switched = False
        preempted = False
        if self.previous_index is not None and self.previous_index != action:
            switched = True
            self.context_switches += 1
            previous_process = self.processes[self.previous_index]
            if previous_process.remaining_time > 0:
                preempted = True
                self.preemptions += 1

        waiting_indices = []
        for index in ready:
            if index != action:
                waiting_indices.append(index)
        reward = -len(waiting_indices)

        start_time = self.time
        if process.first_start_time is None:
            process.first_start_time = start_time
        process.remaining_time -= 1
        self.time += 1
        if process.remaining_time == 0:
            process.completion_time = self.time
        end_time = self.time
        self.timeline.append((start_time, end_time, action, process.pid))
        self.previous_index = action
        self.total_reward += reward

        idle_interval = self.skip_idle()
        next_state = self.get_state()
        info = {
            "process_index": action,
            "pid": process.pid,
            "start_time": start_time,
            "end_time": end_time,
            "waiting_indices": waiting_indices,
            "context_switch": switched,
            "preemption": preempted,
            "completed": process.remaining_time == 0,
            "idle_interval": idle_interval,
        }
        return next_state, reward, self.is_done(), info

    def get_metrics(self):
        process_metrics = {}
        for process in self.processes:
            process_metrics[process.pid] = {
                "first_start_time": process.first_start_time,
                "completion_time": process.completion_time,
                "turnaround_time": process.get_turnaround_time(),
                "waiting_time": process.get_waiting_time(),
                "response_time": process.get_response_time(),
            }
        return {
            "time": self.time,
            "context_switches": self.context_switches,
            "preemptions": self.preemptions,
            "total_reward": self.total_reward,
            "idle_time": self.idle_time,
            "processes": process_metrics,
        }
