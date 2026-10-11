"""Baseline scheduling decisions and independent simulator accounting checks."""

from collections import Counter
from copy import deepcopy
import math
import random

import pytest

from src.baselines import run_fcfs, run_round_robin, run_srtf
from src.baselines.fcfs import FCFSPolicy
from src.baselines.round_robin import RoundRobinPolicy
from src.baselines.runner import run_policy
from src.baselines.srtf import SRTFPolicy
from src.env.environment import CPUSchedulingEnv, Process
from src.env.workload import generate_workload


SCHEDULERS = [run_fcfs, run_srtf, run_round_robin]
MANUAL = (("P1", 0, 5), ("P2", 1, 2), ("P3", 3, 1))


def compact_timeline(result):
    """Merge adjacent CPU steps for readable schedule assertions only."""
    merged = []
    for start, end, index, pid in result["timeline"]:
        if merged and merged[-1][1] == start and merged[-1][2:] == (index, pid):
            merged[-1] = (merged[-1][0], end, index, pid)
        else:
            merged.append((start, end, index, pid))
    return [(pid, start, end) for start, end, _, pid in merged]


def assert_accounting(workload, result):
    counts = Counter()
    previous = None
    end_time = idle = switches = preemptions = reward = 0
    starts, completions = {}, {}
    for start, end, index, pid in result["timeline"]:
        assert start == end_time and end > start
        end_time = end
        if index is None:
            assert pid is None
            idle += end - start
            previous = None
            continue
        assert end - start == 1
        assert workload[index][0] == pid and workload[index][1] <= start
        if previous is not None and previous != index:
            switches += 1
            preemptions += counts[previous] < workload[previous][2]
        reward -= sum(arrival <= start and counts[i] < burst and i != index
                      for i, (_, arrival, burst) in enumerate(workload))
        starts.setdefault(pid, start)
        completions[pid] = end
        counts[index] += 1
        previous = index
    assert result["total_time"] == end_time
    assert result["idle_time"] == idle
    assert result["context_switches"] == switches
    assert result["preemptions"] == preemptions
    assert result["total_reward"] == reward
    assert sum(counts.values()) == sum(burst for _, _, burst in workload)
    assert result["total_time"] == sum(counts.values()) + idle
    assert set(result["processes"]) == {pid for pid, _, _ in workload}
    for index, (pid, arrival, burst) in enumerate(workload):
        assert counts[index] == burst
        metrics = result["processes"][pid]
        assert metrics == {
            "first_start_time": starts[pid], "completion_time": completions[pid],
            "waiting_time": completions[pid] - arrival - burst,
            "turnaround_time": completions[pid] - arrival,
            "response_time": starts[pid] - arrival,
        }
        assert all(value >= 0 for value in metrics.values())
    assert reward == -sum(p["waiting_time"] for p in result["processes"].values())
    for key in ("waiting_time", "turnaround_time", "response_time"):
        expected = sum(p[key] for p in result["processes"].values()) / len(workload) if workload else 0
        assert result["mean_" + key] == pytest.approx(expected)
        assert math.isfinite(result["mean_" + key])


@pytest.mark.parametrize("scheduler,schedule,waiting,turnaround,response,switches,preemptions,reward", [
    (run_fcfs, [("P1", 0, 5), ("P2", 5, 7), ("P3", 7, 8)], 8/3, 16/3, 8/3, 2, 0, -8),
    (run_srtf, [("P1", 0, 1), ("P2", 1, 3), ("P3", 3, 4), ("P1", 4, 8)], 1, 11/3, 0, 3, 1, -3),
    (run_round_robin, [("P1", 0, 2), ("P2", 2, 4), ("P1", 4, 6), ("P3", 6, 7), ("P1", 7, 8)],
     7/3, 5, 4/3, 4, 2, -7),
])
def test_manual_schedules_and_metrics(scheduler, schedule, waiting, turnaround, response,
                                    switches, preemptions, reward):
    result = scheduler(MANUAL)
    assert compact_timeline(result) == schedule
    assert result["mean_waiting_time"] == pytest.approx(waiting)
    assert result["mean_turnaround_time"] == pytest.approx(turnaround)
    assert result["mean_response_time"] == pytest.approx(response)
    assert (result["context_switches"], result["preemptions"], result["total_reward"]) == (switches, preemptions, reward)
    assert result["total_time"] == 8
    assert_accounting(MANUAL, result)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_single_process_has_no_unnecessary_switches(scheduler):
    workload = (("only", 0, 7),)
    result = scheduler(workload)
    assert compact_timeline(result) == [("only", 0, 7)]
    assert result["context_switches"] == result["preemptions"] == result["total_reward"] == 0
    assert_accounting(workload, result)


@pytest.mark.parametrize("scheduler,schedule", [
    (run_fcfs, [("Z", 0, 3), ("A", 3, 4), ("B", 4, 6)]),
    (run_srtf, [("A", 0, 1), ("B", 1, 3), ("Z", 3, 6)]),
    (run_round_robin, [("Z", 0, 2), ("A", 2, 3), ("B", 3, 5), ("Z", 5, 6)]),
])
def test_simultaneous_arrivals_keep_original_indices(scheduler, schedule):
    workload = (("Z", 0, 3), ("A", 0, 1), ("B", 0, 2))
    result = scheduler(workload)
    assert compact_timeline(result) == schedule
    assert_accounting(workload, result)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_initial_and_intermediate_idle_periods(scheduler):
    workload = (("late", 7, 2), ("early", 3, 1), ("last", 12, 1))
    result = scheduler(workload)
    assert compact_timeline(result) == [
        (None, 0, 3), ("early", 3, 4), (None, 4, 7), ("late", 7, 9),
        (None, 9, 12), ("last", 12, 13),
    ]
    assert result["context_switches"] == result["preemptions"] == 0
    assert_accounting(workload, result)


def test_fcfs_is_nonpreemptive_and_orders_ready_by_arrival():
    workload = (("later", 3, 1), ("early", 1, 5), ("middle", 2, 2))
    result = run_fcfs(workload)
    assert compact_timeline(result) == [(None, 0, 1), ("early", 1, 6), ("middle", 6, 8), ("later", 8, 9)]
    assert result["preemptions"] == 0
    assert_accounting(workload, result)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_equal_arrivals_and_bursts_break_ties_by_index_not_pid(scheduler):
    workload = (("Z", 0, 2), ("A", 0, 2), ("M", 0, 2))
    result = scheduler(workload)
    assert compact_timeline(result) == [("Z", 0, 2), ("A", 2, 4), ("M", 4, 6)]
    assert_accounting(workload, result)


def test_srtf_prefers_current_process_on_a_remaining_time_tie():
    workload = (("new", 1, 2), ("running", 0, 3))
    result = run_srtf(workload)
    assert compact_timeline(result) == [("running", 0, 3), ("new", 3, 5)]
    assert result["preemptions"] == 0
    # Explicitly exercise continuation overriding the other tie breakers.
    policy = SRTFPolicy()
    policy.reset()
    policy.current = 2
    ready = {0: Process("first", 0, 2), 2: Process("current", 1, 2)}
    assert policy.select_action(ready) == 2
    assert_accounting(workload, result)


def test_srtf_remaining_ties_use_arrival_then_index():
    policy = SRTFPolicy()
    policy.reset()
    ready = {0: Process("later", 3, 2), 3: Process("earlier_high", 1, 2),
             2: Process("earlier_low", 1, 2)}
    assert policy.select_action(ready) == 2
    policy.current = 0
    ready[0].remaining_time = 3
    assert policy.select_action(ready) == 2


def test_srtf_repeated_preemption_and_completion():
    workload = (("A", 0, 12), ("B", 1, 8), ("C", 2, 4), ("D", 3, 1))
    result = run_srtf(workload)
    assert compact_timeline(result) == [
        ("A", 0, 1), ("B", 1, 2), ("C", 2, 3), ("D", 3, 4),
        ("C", 4, 7), ("B", 7, 14), ("A", 14, 25),
    ]
    assert result["preemptions"] == 3
    assert_accounting(workload, result)


def test_rr_arrival_at_expiry_is_queued_before_expired_process():
    workload = (("A", 0, 4), ("B", 2, 1))
    result = run_round_robin(workload)
    assert compact_timeline(result) == [("A", 0, 2), ("B", 2, 3), ("A", 3, 5)]
    assert_accounting(workload, result)


def test_rr_old_waiters_precede_new_arrivals_and_requeued_process():
    workload = (("A", 0, 4), ("B", 0, 2), ("C", 1, 1), ("D", 2, 1))
    result = run_round_robin(workload)
    assert compact_timeline(result) == [("A", 0, 2), ("B", 2, 4), ("C", 4, 5), ("D", 5, 6), ("A", 6, 8)]
    assert_accounting(workload, result)


def test_rr_arrivals_during_quantum_are_chronological_with_index_ties():
    workload = (("late", 2, 1), ("runner", 0, 6), ("middle", 1, 1), ("same", 1, 1))
    result = run_round_robin(workload, quantum=4)
    assert compact_timeline(result) == [("runner", 0, 4), ("middle", 4, 5), ("same", 5, 6), ("late", 6, 7), ("runner", 7, 9)]
    assert_accounting(workload, result)


def test_rr_completion_gives_the_next_process_a_full_quantum():
    workload = (("A", 0, 1), ("B", 0, 3), ("C", 0, 2))
    result = run_round_robin(workload)
    assert compact_timeline(result) == [("A", 0, 1), ("B", 1, 3), ("C", 3, 5), ("B", 5, 6)]
    assert_accounting(workload, result)


def test_rr_quantum_one_repeated_preemptions():
    workload = (("A", 0, 3), ("B", 0, 3), ("C", 0, 3))
    result = run_round_robin(workload, quantum=1)
    assert [entry[3] for entry in result["timeline"]] == ["A", "B", "C"] * 3
    assert result["context_switches"] == 8
    assert result["preemptions"] == 6
    assert_accounting(workload, result)


def test_rr_large_quantum_matches_fcfs():
    fcfs = run_fcfs(MANUAL)
    rr = run_round_robin(MANUAL, quantum=100)
    assert {key: value for key, value in fcfs.items() if key != "algorithm"} == {
        key: value for key, value in rr.items() if key != "algorithm"
    }


@pytest.mark.parametrize("quantum", [0, -1, True, False, 1.5, "2", None])
def test_rr_rejects_invalid_quantum_before_changing_environment(quantum):
    env = CPUSchedulingEnv()
    env.reset(MANUAL)
    env.step(0)
    before = deepcopy(env.__dict__)
    with pytest.raises(ValueError, match="positive integer"):
        run_round_robin(MANUAL, quantum=quantum, env=env)
    assert env.timeline == before["timeline"]
    assert env.get_metrics()["time"] == before["time"]


def test_rr_queue_contains_each_ready_waiter_once():
    class InspectQueue(RoundRobinPolicy):
        def select_action(self, ready):
            action = super().select_action(ready)
            assert len(self.queue) == len(set(self.queue))
            assert action not in self.queue
            assert set(self.queue) | {action} == set(ready)
            assert self.admitted.issuperset(ready)
            return action

    workload = (("A", 0, 8), ("B", 0, 5), ("C", 1, 3), ("D", 4, 6))
    result = run_policy(workload, InspectQueue(2))
    assert_accounting(workload, result)


@pytest.mark.parametrize("policy_type", [FCFSPolicy, SRTFPolicy, RoundRobinPolicy])
def test_policy_only_receives_arrived_unfinished_simulator_objects(policy_type):
    env = CPUSchedulingEnv()
    calls = []

    class InspectPolicy(policy_type):
        def select_action(self, ready):
            assert set(ready) == set(env.valid_actions())
            for index, process in ready.items():
                assert process is env.processes[index]
                assert process.arrival_time <= env.time and process.remaining_time > 0
            calls.append(set(ready))
            return super().select_action(ready)

    workload = (("A", 0, 2), ("future", 10, 3), ("later", 11, 2))
    result = run_policy(workload, InspectPolicy(), env)
    assert calls[0] == {0}
    assert_accounting(workload, result)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_future_bursts_and_future_process_count_do_not_affect_earlier_actions(scheduler):
    first = (("A", 0, 7), ("B", 0, 5), ("future", 4, 1))
    second = (("A", 0, 7), ("B", 0, 5), ("future", 4, 9), ("extra", 8, 3))
    a, b = scheduler(first), scheduler(second)
    assert [row for row in a["timeline"] if row[1] <= 4] == [row for row in b["timeline"] if row[1] <= 4]


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_workload_immutability_repeatability_and_clean_reset(scheduler):
    before = deepcopy(MANUAL)
    env = CPUSchedulingEnv()
    env.reset((("discard", 0, 10),))
    env.step(0)
    old_process = env.processes[0]
    first = scheduler(MANUAL, env=env)
    assert first == scheduler(MANUAL, env=env) == scheduler(MANUAL)
    assert MANUAL == before
    assert env.processes[0] is not old_process
    assert old_process.remaining_time == 9
    assert first["timeline"] == tuple(env.timeline)
    assert first["processes"] == env.get_metrics()["processes"]
    assert_accounting(MANUAL, first)
    saved = deepcopy(first)
    scheduler((("replacement", 2, 1),), env=env)
    assert first == saved  # Results do not alias a reused environment.


def test_reused_policy_resets_rr_queue_and_quantum_state():
    policy = RoundRobinPolicy(2)
    first = run_policy(MANUAL, policy)
    run_policy((("other", 3, 1),), policy)
    assert run_policy(MANUAL, policy) == first


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_empty_workload_returns_zero_means(scheduler):
    result = scheduler(())
    assert result["timeline"] == () and result["processes"] == {}
    assert_accounting((), result)


def test_standardized_result_fields_are_shared():
    expected = {"algorithm", "timeline", "processes", "mean_waiting_time", "mean_turnaround_time",
                "mean_response_time", "context_switches", "preemptions", "total_reward", "total_time", "idle_time"}
    assert all(set(scheduler(MANUAL)) == expected for scheduler in SCHEDULERS)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_every_cpu_unit_executes_through_existing_step(scheduler):
    class TrackedEnvironment(CPUSchedulingEnv):
        def __init__(self):
            self.actions = []
            super().__init__()

        def step(self, action):
            assert type(action) is int and action in self.valid_actions()
            self.actions.append(action)
            return super().step(action)

    env = TrackedEnvironment()
    result = scheduler(MANUAL, env=env)
    assert len(env.actions) == sum(burst for _, _, burst in MANUAL)
    assert env.actions == [entry[2] for entry in result["timeline"] if entry[2] is not None]


def test_randomized_workloads_srtf_is_no_worse_than_fcfs_and_rr():
    # Fresh unit-test workloads only; no reserved final-test files are read.
    global_rng = random.getstate()
    for number in range(100):
        workload = list(generate_workload(seed=610100 + number, process_count_range=(1, 8),
                                          burst_time_range=(1, 12), arrival_time_range=(0, 25)))
        random.Random(610200 + number).shuffle(workload)
        workload = tuple(workload)
        shortest = run_srtf(workload)
        assert_accounting(workload, shortest)
        fcfs = run_fcfs(workload)
        assert_accounting(workload, fcfs)
        assert shortest["mean_waiting_time"] <= fcfs["mean_waiting_time"] + 1e-12
        for quantum in (1, 2, 5, 20):
            rr = run_round_robin(workload, quantum=quantum)
            assert_accounting(workload, rr)
            assert shortest["mean_waiting_time"] <= rr["mean_waiting_time"] + 1e-12
    assert random.getstate() == global_rng
