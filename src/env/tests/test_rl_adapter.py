"""Deterministic tests for the observation wrapper, without a learning agent."""

from copy import deepcopy
import random

import numpy as np
import pytest

from src.env.environment import CPUSchedulingEnv
from src.env.rl_adapter import RLAdapter
from src.env.workload import generate_workload


def snapshot(adapter):
    return deepcopy({
        "state": adapter.env.get_state(debug=True),
        "timeline": adapter.env.timeline,
        "workload": adapter.env.workload,
        "previous_index": adapter.env.previous_index,
        "last_executed_index": adapter.last_executed_index,
    })


def test_shape_dtype_padding_and_initial_normalization():
    adapter = RLAdapter()
    observation = adapter.reset([("A", 0, 5), ("future", 4, 9)])
    assert observation.shape == (39,)
    assert observation.dtype == np.float32
    assert adapter.max_episode_duration == 68
    np.testing.assert_allclose(observation[:6], [1, 0, 0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(observation[6:36], np.zeros(30))
    np.testing.assert_allclose(observation[36:], [0, 1 / 6, 0])


def test_custom_bounds_and_consistent_normalization_across_episodes():
    adapter = RLAdapter(max_burst_time=4, max_arrival_time=2)
    assert adapter.max_episode_duration == 26
    observation = adapter.reset([("A", 2, 4)])
    np.testing.assert_allclose(observation[:6], [1, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(observation[36:], [2 / 26, 1 / 6, 0])
    observation = adapter.reset([("B", 0, 2), ("C", 1, 3)])
    assert observation[2] == 0.5
    assert adapter.max_episode_duration == 26
    custom = RLAdapter(max_burst_time=4, max_arrival_time=2, max_episode_duration=100)
    assert custom.reset([("A", 2, 4)])[36] == np.float32(0.02)


def test_waiting_flags_completion_and_last_executed():
    adapter = RLAdapter()
    adapter.reset([("A", 0, 2), ("B", 0, 1), ("C", 2, 1)])
    observation, reward, done, info = adapter.step(0)
    assert reward == -1 and not done
    np.testing.assert_allclose(observation[:6], [1, 0, 0.2, 0.1, 0, 1])
    np.testing.assert_allclose(observation[6:12], [1, 0, 0.1, 0.1, 1 / 68, 0])
    np.testing.assert_array_equal(observation[12:36], np.zeros(24))
    observation, reward, done, info = adapter.step(1)
    assert reward == -1 and not done
    np.testing.assert_allclose(observation[:6], [1, 0, 0.2, 0.1, 1 / 68, 0])
    np.testing.assert_allclose(observation[6:12], [0, 1, 0.1, 0, 1 / 68, 1])
    np.testing.assert_allclose(observation[12:18], [1, 0, 0.1, 0.1, 0, 0])
    np.testing.assert_allclose(observation[36:], [2 / 68, 3 / 6, 1 / 6])
    observation, reward, done, info = adapter.step(0)
    # B's accumulated waiting stays fixed after completion.
    assert observation[10] == np.float32(1 / 68)
    assert observation[11] == 0 and observation[5] == 1


def test_mask_and_stable_slots_with_unsorted_arrivals():
    adapter = RLAdapter()
    observation = adapter.reset([("late", 2, 1), ("early", 0, 2)])
    mask = adapter.valid_action_mask()
    assert mask.shape == (6,) and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [False, True, False, False, False, False])
    np.testing.assert_array_equal(observation[:6], np.zeros(6))
    adapter.step(1)
    observation, reward, done, info = adapter.step(1)
    assert adapter.env.processes[0].pid == "late"
    np.testing.assert_array_equal(info["action_mask"], [True, False, False, False, False, False])
    assert observation[0] == 1 and observation[7] == 1
    observation, reward, done, info = adapter.step(0)
    assert done
    assert not adapter.valid_action_mask().any()
    assert not info["action_mask"].any()
    assert not observation[:36:6].any()


def test_six_slots_can_all_be_used():
    adapter = RLAdapter()
    workload = []
    for index in range(6):
        workload.append((index, 0, 1))
    observation = adapter.reset(workload)
    assert adapter.valid_action_mask().all()
    assert observation[37] == 1
    for action in range(6):
        observation, reward, done, info = adapter.step(action)
    assert done and observation[38] == 1
    assert observation[35] == 1
    assert not adapter.valid_action_mask().any()


def test_forwarding_preserves_state_reward_done_and_simulator_info():
    env = CPUSchedulingEnv()
    adapter = RLAdapter(env=env)
    direct = CPUSchedulingEnv()
    workload = [("P1", 0, 5), ("P2", 1, 2), ("P3", 3, 1)]
    adapter.reset(workload)
    direct.reset(workload)
    total_reward = 0
    for action in [0, 1, 1, 2, 0, 0, 0, 0]:
        observation, reward, done, info = adapter.step(action)
        direct_state, direct_reward, direct_done, direct_info = direct.step(action)
        assert reward == direct_reward and done == direct_done
        assert env.get_state() == direct_state
        assert env.timeline == direct.timeline
        for key in direct_info:
            assert info[key] == direct_info[key]
        np.testing.assert_array_equal(info["action_mask"], adapter.valid_action_mask())
        total_reward += reward
    assert total_reward == env.total_reward == -3
    assert env.context_switches == 3 and env.preemptions == 1
    assert env.get_metrics() == direct.get_metrics()


def test_invalid_actions_do_not_mutate_either_object():
    adapter = RLAdapter()
    adapter.reset([("ready", 0, 2), ("future", 4, 1)])
    adapter.step(0)
    for action in [-1, 1, 2, 5, 6, 99, True, False, 0.0, "0", None, []]:
        before = snapshot(adapter)
        with pytest.raises(ValueError):
            adapter.step(action)
        assert snapshot(adapter) == before


def test_completed_and_terminal_actions_are_rejected():
    adapter = RLAdapter()
    adapter.reset([("A", 0, 1), ("B", 0, 1)])
    adapter.step(0)
    before = snapshot(adapter)
    with pytest.raises(ValueError):
        adapter.step(0)
    assert snapshot(adapter) == before
    adapter.step(1)
    before = snapshot(adapter)
    with pytest.raises(ValueError):
        adapter.step(1)
    assert snapshot(adapter) == before


def test_no_future_leakage_in_features_counts_masks_or_slot_order():
    first = RLAdapter()
    second = RLAdapter()
    # Same visible slot/history, different future counts, PIDs, bursts and arrivals.
    a = first.reset([("hidden-A", 4, 10), ("visible", 0, 4)])
    b = second.reset([("hidden-B", 6, 1), ("visible", 0, 4), ("extra", 8, 7)])
    for step in range(4):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(first.valid_action_mask(), second.valid_action_mask())
        if step < 3:
            a, reward_a, done_a, info_a = first.step(1)
            b, reward_b, done_b, info_b = second.step(1)
            assert reward_a == reward_b and done_a == done_b
            np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])


def test_idle_jumps_and_last_execution_across_idle():
    adapter = RLAdapter()
    observation = adapter.reset([("A", 3, 1), ("B", 7, 1)])
    np.testing.assert_allclose(observation[36:], [3 / 68, 1 / 6, 0])
    observation, reward, done, info = adapter.step(0)
    assert observation[36] == np.float32(7 / 68)
    assert observation[5] == 1  # Last executed survives the idle gap.
    assert observation[4] == observation[10] == 0
    assert adapter.env.previous_index is None
    assert info["idle_interval"] == (4, 7, None, None)
    assert reward == 0 and not done
    assert adapter.valid_action_mask()[1]
    adapter.step(1)
    assert adapter.env.context_switches == 0


def test_reset_clears_history_and_empty_workload_is_terminal():
    adapter = RLAdapter()
    first = adapter.reset([("A", 0, 2), ("B", 0, 1)])
    adapter.step(0)
    again = adapter.reset(adapter.env.workload)
    np.testing.assert_array_equal(first, again)
    assert adapter.last_executed_index is None
    empty = adapter.reset([])
    np.testing.assert_array_equal(empty, np.zeros(39))
    assert adapter.env.is_done()
    assert not adapter.valid_action_mask().any()


def test_invalid_workload_bounds_are_rejected_atomically():
    adapter = RLAdapter()
    adapter.reset([("A", 0, 2)])
    adapter.step(0)
    invalid = [
        [(index, 0, 1) for index in range(7)], [("A", 9, 1)],
        [("A", -1, 1)], [("A", 0, 11)], [("A", 0, 0)],
        [("A", True, 1)], [("A", 0, 1.5)], [("A", 0)],
        [("A", 0, 1), ("A", 1, 1)], [(None, 0, 1)],
    ]
    for workload in invalid:
        before = snapshot(adapter)
        with pytest.raises(ValueError):
            adapter.reset(workload)
        assert snapshot(adapter) == before


def test_invalid_configuration():
    for config in [
        {"max_burst_time": 0}, {"max_burst_time": True},
        {"max_burst_time": 1.5}, {"max_arrival_time": -1},
        {"max_arrival_time": True}, {"max_episode_duration": 67},
        {"max_episode_duration": 68.0},
    ]:
        with pytest.raises(ValueError):
            RLAdapter(**config)


def test_returned_observations_and_masks_are_independent_copies():
    adapter = RLAdapter()
    observation = adapter.reset([("A", 0, 2)])
    observation[:] = 99
    mask = adapter.valid_action_mask()
    mask[:] = False
    assert adapter.get_observation()[0] == 1
    assert adapter.valid_action_mask()[0]
    observation, reward, done, info = adapter.step(0)
    old_mask = info["action_mask"].copy()
    adapter.step(0)
    np.testing.assert_array_equal(info["action_mask"], old_mask)
    info["action_mask"][:] = True
    assert not adapter.valid_action_mask().any()


def test_reward_and_observation_invariants_on_seeded_complete_episodes():
    for seed in range(20):
        adapter = RLAdapter()
        observation = adapter.reset(generate_workload(seed))
        choices = random.Random(seed + 100)
        rewards = 0
        while not adapter.env.is_done():
            assert observation.shape == (39,) and observation.dtype == np.float32
            assert np.all(observation >= 0) and np.all(observation <= 1)
            mask = adapter.valid_action_mask()
            assert mask.any()
            # Use ordinary Python indices, matching the simulator's action API.
            actions = []
            for index in range(6):
                if mask[index]:
                    actions.append(index)
            observation, reward, done, info = adapter.step(choices.choice(actions))
            rewards += reward
            np.testing.assert_array_equal(info["action_mask"], adapter.valid_action_mask())
        total_waiting = 0
        for index, process in enumerate(adapter.env.processes):
            total_waiting += process.get_waiting_time()
            assert observation[index * 6 + 4] == np.float32(
                process.get_waiting_time() / adapter.max_episode_duration
            )
        assert rewards == adapter.env.total_reward == -total_waiting
        assert not adapter.valid_action_mask().any()
