"""Simulator tests using ordinary functions and loops, without decorators."""

from copy import deepcopy
import random

import pytest

from src.env.environment import CPUSchedulingEnv
from src.env.workload import generate_workload


def snapshot(env):
    """Copy all episode data to check that rejected actions change nothing."""
    return deepcopy({
        "state": env.get_state(debug=True),
        "workload": env.workload,
        "timeline": env.timeline,
        "metrics": env.get_metrics(),
        "previous_index": env.previous_index,
    })


def run_actions(env, actions):
    rewards = []
    for action in actions:
        state, reward, done, info = env.step(action)
        assert state["done"] == done == env.is_done()
        assert state["valid_actions"] == env.valid_actions()
        assert info["end_time"] - info["start_time"] == 1
        rewards.append(reward)
    return rewards


def assert_completed_episode(env):
    """Check reward and timing accounting independently from the timeline."""
    assert env.is_done()
    assert env.valid_actions() == []
    total_waiting = 0
    total_burst = 0
    previous_end = 0
    for start, end, index, pid in env.timeline:
        assert start == previous_end
        assert end > start
        previous_end = end
        if index is not None:
            assert end - start == 1
            assert pid == env.processes[index].pid
    assert previous_end == env.time

    for index, process in enumerate(env.processes):
        executions = []
        waited_units = 0
        for interval in env.timeline:
            start, end, running_index, pid = interval
            if running_index == index:
                executions.append(interval)
            elif process.arrival_time <= start < process.completion_time:
                waited_units += end - start
        assert len(executions) == process.burst_time
        assert process.remaining_time == 0
        assert process.first_start_time == executions[0][0]
        assert process.completion_time == executions[-1][1]
        assert process.get_waiting_time() == waited_units
        assert process.get_response_time() == process.first_start_time - process.arrival_time
        assert process.get_turnaround_time() == process.completion_time - process.arrival_time
        total_waiting += process.get_waiting_time()
        total_burst += process.burst_time
    assert env.total_reward == -total_waiting
    assert total_burst + env.idle_time == env.time


def test_single_process():
    env = CPUSchedulingEnv()
    initial = env.reset([("P1", 0, 3)])
    assert initial["valid_actions"] == [0]
    assert env.processes[0].get_waiting_time() is None
    assert env.processes[0].get_turnaround_time() is None
    assert env.processes[0].get_response_time() is None
    assert run_actions(env, [0, 0, 0]) == [0, 0, 0]
    assert env.get_metrics()["processes"]["P1"] == {
        "first_start_time": 0, "completion_time": 3,
        "turnaround_time": 3, "waiting_time": 0, "response_time": 0,
    }
    assert env.context_switches == env.preemptions == env.idle_time == 0
    assert initial["processes"][0]["remaining_time"] == 3
    assert_completed_episode(env)


def test_all_processes_arrive_at_zero():
    env = CPUSchedulingEnv()
    env.reset([("A", 0, 2), ("B", 0, 1), ("C", 0, 2)])
    assert run_actions(env, [0, 1, 2, 0, 2]) == [-2, -2, -1, -1, 0]
    metrics = env.get_metrics()["processes"]
    assert metrics["A"]["completion_time"] == 4
    assert metrics["B"]["completion_time"] == 2
    assert metrics["C"]["completion_time"] == 5
    assert metrics["A"]["waiting_time"] == 2
    assert metrics["B"]["waiting_time"] == 1
    assert metrics["C"]["waiting_time"] == 3
    assert metrics["A"]["response_time"] == 0
    assert metrics["B"]["response_time"] == 1
    assert metrics["C"]["response_time"] == 2
    assert env.context_switches == 4
    assert env.preemptions == 2
    assert_completed_episode(env)


def test_manual_case():
    env = CPUSchedulingEnv()
    env.reset([("P1", 0, 5), ("P2", 1, 2), ("P3", 3, 1)])
    rewards = run_actions(env, [0, 1, 1, 2, 0, 0, 0, 0])
    assert rewards == [0, -1, -1, -1, 0, 0, 0, 0]
    expected = [(8, 3, 8), (3, 0, 2), (4, 0, 1)]
    for index, process in enumerate(env.processes):
        completion, waiting, turnaround = expected[index]
        assert process.completion_time == completion
        assert process.get_waiting_time() == waiting
        assert process.get_turnaround_time() == turnaround
        assert process.get_response_time() == 0
    assert env.total_reward == sum(rewards) == -3
    assert env.context_switches == 3
    assert env.preemptions == 1
    assert_completed_episode(env)


def test_staggered_arrivals_and_hidden_future_bursts():
    env = CPUSchedulingEnv()
    initial = env.reset([("late", 1, 3), ("early", 0, 2)])
    assert initial["valid_actions"] == [1]  # Indices were not arrival-sorted.
    assert initial["processes"][0]["burst_time"] is None
    assert initial["processes"][0]["remaining_time"] is None
    assert env.get_state(debug=True)["processes"][0]["burst_time"] == 3
    state, reward, done, info = env.step(1)
    assert reward == 0  # Arrival at t=1 did not wait during [0, 1).
    assert not done
    assert state["time"] == 1
    assert state["valid_actions"] == [0, 1]
    assert state["processes"][0]["burst_time"] == 3
    assert info["waiting_indices"] == []
    assert initial["processes"][0]["burst_time"] is None
    assert run_actions(env, [0, 1, 0, 0]) == [-1, -1, 0, 0]
    assert_completed_episode(env)


def test_initial_and_intermediate_idle_time():
    env = CPUSchedulingEnv()
    initial = env.reset([("late", 7, 2), ("early", 3, 1)])
    assert initial["time"] == initial["idle_time"] == 3
    assert initial["valid_actions"] == [1]
    assert env.timeline == [(0, 3, None, None)]
    state, reward, done, info = env.step(1)
    assert state["time"] == 7
    assert reward == 0 and not done
    assert state["valid_actions"] == [0]
    assert info["start_time"] == 3 and info["end_time"] == 4
    assert info["idle_interval"] == (4, 7, None, None)
    assert state["idle_time"] == 6
    assert run_actions(env, [0, 0]) == [0, 0]
    assert env.time == 9
    assert env.context_switches == env.preemptions == 0
    assert env.timeline == [
        (0, 3, None, None), (3, 4, 1, "early"), (4, 7, None, None),
        (7, 8, 0, "late"), (8, 9, 0, "late"),
    ]
    assert_completed_episode(env)


def test_preemption_versus_switch_after_completion():
    env = CPUSchedulingEnv()
    env.reset([("A", 0, 2), ("B", 0, 1)])
    state, reward, done, info = env.step(0)
    assert not info["context_switch"] and not info["preemption"]
    state, reward, done, info = env.step(1)
    assert info["context_switch"] and info["preemption"]
    assert info["completed"]
    state, reward, done, info = env.step(0)
    assert info["context_switch"] and not info["preemption"]
    assert env.context_switches == 2 and env.preemptions == 1
    assert_completed_episode(env)


def test_invalid_actions_do_not_change_environment():
    env = CPUSchedulingEnv()
    env.reset([("ready", 0, 2), ("future", 4, 1)])
    for action in [-1, 2, 99, 1, True, False, 0.0, "0", None, []]:
        before = snapshot(env)
        with pytest.raises(ValueError):
            env.step(action)
        assert snapshot(env) == before


def test_completed_and_terminal_actions_are_rejected():
    env = CPUSchedulingEnv()
    env.reset([("A", 0, 1), ("B", 0, 1)])
    env.step(0)
    before = snapshot(env)
    with pytest.raises(ValueError):
        env.step(0)
    assert snapshot(env) == before
    env.step(1)
    before = snapshot(env)
    with pytest.raises(ValueError):
        env.step(1)
    assert snapshot(env) == before


def test_reset_isolation_and_workload_replay():
    definitions = [["A", 0, 2], ["B", 0, 1]]
    env = CPUSchedulingEnv()
    initial = env.reset(definitions)
    other = CPUSchedulingEnv()
    other.reset(env.workload)
    definitions[0][2] = 99
    definitions.clear()
    assert env.workload == (("A", 0, 2), ("B", 0, 1))
    with pytest.raises(TypeError):
        env.workload[0][2] = 99
    old_process = env.processes[0]
    run_actions(env, [0, 1, 0])
    assert other.get_state() == initial
    assert env.reset(env.workload) == initial
    assert env.processes[0] is not old_process
    assert old_process.completion_time == 3
    assert env.timeline == [] and env.previous_index is None
    run_actions(env, [1, 0, 0])
    assert env.preemptions == 0
    assert_completed_episode(env)
    env.reset(env.workload)
    env.step(0)
    assert env.reset(env.workload) == initial  # Reset a partial episode too.
    state = env.reset([("new", 2, 1)])
    assert state["time"] == state["idle_time"] == 2
    assert state["context_switches"] == state["preemptions"] == state["total_reward"] == 0
    assert state["processes"][0]["pid"] == "new"


def test_returned_state_and_metrics_are_independent():
    env = CPUSchedulingEnv()
    state = env.reset([("P", 0, 2)])
    before = snapshot(env)
    state["processes"][0]["remaining_time"] = 99
    state["valid_actions"].clear()
    metrics = env.get_metrics()
    metrics["processes"]["P"]["completion_time"] = 99
    assert snapshot(env) == before


def test_empty_workload():
    env = CPUSchedulingEnv()
    state = env.reset([])
    assert state["done"] and state["valid_actions"] == []
    assert state["time"] == state["total_reward"] == state["idle_time"] == 0
    assert env.timeline == []
    with pytest.raises(ValueError):
        env.step(0)
    assert_completed_episode(env)


def test_invalid_workloads_leave_previous_episode_unchanged():
    invalid_workloads = [
        [("A", 0, 1), ("A", 2, 1)], [object()], [None], [("P", 0)],
        [("", 0, 1)], [(None, 0, 1)], [(True, 0, 1)], [("P", -1, 1)],
        [("P", 0.5, 1)], [("P", True, 1)], [("P", 0, 0)],
        [("P", 0, -1)], [("P", 0, 1.5)], [("P", 0, True)],
    ]
    env = CPUSchedulingEnv()
    env.reset([("valid", 0, 2)])
    env.step(0)
    for workload in invalid_workloads:
        before = snapshot(env)
        with pytest.raises(ValueError):
            env.reset(workload)
        assert snapshot(env) == before


def test_reproducible_immutable_workloads_and_rng_isolation():
    before = random.getstate()
    first = generate_workload(seed=42)
    assert first == generate_workload(seed=42)
    assert first != generate_workload(seed=43)
    assert random.getstate() == before
    assert isinstance(first, tuple)
    with pytest.raises(TypeError):
        first[0][1] = 99


def test_default_workload_bounds():
    for seed in range(20):
        workload = generate_workload(seed)
        assert 4 <= len(workload) <= 6
        for index, entry in enumerate(workload):
            pid, arrival, burst = entry
            assert pid == "P" + str(index + 1)
            assert type(arrival) is int and 0 <= arrival <= 8
            assert type(burst) is int and 1 <= burst <= 10


def test_configurable_workload_ranges():
    workload = generate_workload(7, process_count_range=(2, 2),
                                 burst_time_range=(12, 12), arrival_time_range=(15, 15))
    assert workload == (("P1", 15, 12), ("P2", 15, 12))


def test_invalid_workload_ranges():
    invalid_settings = [
        {"process_count_range": (0, 3)}, {"process_count_range": (6, 4)},
        {"burst_time_range": (0, 10)}, {"arrival_time_range": (-1, 8)},
        {"arrival_time_range": (2, 1)}, {"burst_time_range": (1, 2.5)},
        {"process_count_range": (True, 6)}, {"burst_time_range": (1,)},
        {"arrival_time_range": None},
    ]
    for settings in invalid_settings:
        with pytest.raises(ValueError):
            generate_workload(1, **settings)


def test_reward_identity_on_seeded_episodes():
    for seed in range(20):
        env = CPUSchedulingEnv()
        env.reset(generate_workload(seed))
        choices = random.Random(seed + 100)
        total_reward = 0
        while not env.is_done():
            assert env.valid_actions()
            action = choices.choice(env.valid_actions())
            state, reward, done, info = env.step(action)
            total_reward += reward
        assert env.total_reward == total_reward
        assert_completed_episode(env)


def explore_all_actions(env):
    """Test every possible action sequence for a small workload."""
    if env.is_done():
        assert_completed_episode(env)
        return
    assert env.valid_actions()
    for action in env.valid_actions():
        branch = deepcopy(env)
        branch.step(action)
        explore_all_actions(branch)


def test_reward_identity_on_every_small_schedule():
    for arrivals in [(0, 0, 0), (0, 1, 3), (2, 2, 8)]:
        env = CPUSchedulingEnv()
        env.reset([("A", arrivals[0], 2), ("B", arrivals[1], 2),
                   ("C", arrivals[2], 1)])
        explore_all_actions(env)
