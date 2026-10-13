"""Stage 6.3A block mappings, evaluation isolation, and paired statistics."""

from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import greedy_evaluation
from src.benchmark.run import reconstruct_validation
from src.env.rl_adapter import RLAdapter
from src.permutation.analysis import analyze, bootstrap_ratio
from src.permutation.experiment import agent_snapshot, fixed_state_experiment, tree_digest, workload_experiment
from src.permutation.run import run_experiment, save_outputs
from src.permutation.transforms import inverse_permutation, make_permutations, permute_observation, permute_workload


WORKLOADS = (
    (("A", 0, 5), ("B", 1, 2), ("C", 3, 1), ("D", 8, 2)),
    (("A", 0, 5), ("B", 0, 2), ("C", 2, 2), ("D", 8, 1), ("E", 3, 1)),
    (("A", 2, 1), ("B", 2, 1), ("C", 5, 1), ("D", 8, 1), ("E", 8, 1), ("F", 8, 1)),
)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_validation_fingerprint(tmp_path):
    (tmp_path / "training_config.json").write_text(json.dumps({"validation_seed": 1729, "validation_workloads": 128}))
    _, workloads = reconstruct_validation(tmp_path)
    assert len(workloads) == 128 and sum(map(len, workloads)) == 645
    assert hashlib.sha256(json.dumps(workloads, separators=(",", ":")).encode()).hexdigest() == (
        "7387547e5cf6bdccfcf92732be65a798c0bf0c369050ad06d93869a973f5fe8e")


@pytest.mark.parametrize("count", [4, 5, 6])
def test_distinct_seeded_permutations_and_inverse(count):
    state = random.getstate()
    permutations = make_permutations(count, 6301729)
    assert len(permutations) == len(set(permutations)) == 8
    assert permutations[0] == tuple(range(count))
    assert permutations == make_permutations(count, 6301729)
    assert permutations != make_permutations(count, 6301730)
    assert random.getstate() == state
    for permutation in permutations:
        inverse = inverse_permutation(permutation)
        assert all(permutation[inverse[i]] == i for i in range(count))
        assert all(inverse[permutation[i]] == i for i in range(count))


@pytest.mark.parametrize("bad", [(0, 0), (1, 2), (), (False, 1), (0.0, 1), tuple(range(7))])
def test_invalid_maps_rejected(bad):
    with pytest.raises(ValueError):
        inverse_permutation(bad)


def test_impossible_distinct_arrangements_rejected():
    with pytest.raises(ValueError):
        make_permutations(2, 4)


def test_complete_blocks_masks_globals_last_process_and_hidden_future():
    adapter = RLAdapter()
    workload = (("ready", 0, 5), ("future", 7, 9), ("ready2", 0, 2), ("future2", 8, 4))
    adapter.reset(workload)
    observation, _, _, _ = adapter.step(0)
    mask = adapter.valid_action_mask()
    original, original_mask = observation.copy(), mask.copy()
    permutation = (1, 2, 3, 0)
    moved, moved_mask = permute_observation(observation, mask, permutation)
    for new, old in enumerate(permutation):
        np.testing.assert_array_equal(moved[new * 6:new * 6 + 6], original[old * 6:old * 6 + 6])
        assert moved_mask[new] == original_mask[old]
    assert moved[3 * 6 + 5] == 1 and moved[:36].reshape(6, 6)[:, 5].sum() == 1
    np.testing.assert_array_equal(moved[36:], original[36:])
    np.testing.assert_array_equal(moved[24:36], original[24:36])
    for new in (0, 2):
        assert not moved_mask[new] and not moved[new * 6:new * 6 + 6].any()
    np.testing.assert_array_equal(observation, original)
    np.testing.assert_array_equal(mask, original_mask)
    restored, restored_mask = permute_observation(moved, moved_mask, inverse_permutation(permutation))
    np.testing.assert_array_equal(restored, original)
    np.testing.assert_array_equal(restored_mask, original_mask)
    # Changing unavailable future bursts does not affect the transformed state.
    other = RLAdapter()
    other.reset((("ready", 0, 5), ("future", 7, 1), ("ready2", 0, 2), ("future2", 8, 10)))
    other_observation, _, _, _ = other.step(0)
    np.testing.assert_array_equal(permute_observation(other_observation, other.valid_action_mask(), permutation)[0], moved)


def test_workload_reordering_preserves_definition_objects():
    workload = WORKLOADS[0]
    before = deepcopy(workload)
    permutation = (3, 1, 0, 2)
    moved = permute_workload(workload, permutation)
    assert workload == before
    assert all(moved[new] is workload[old] for new, old in enumerate(permutation))
    assert permute_workload(moved, inverse_permutation(permutation)) == workload
    with pytest.raises(ValueError):
        permute_workload(workload, (1, 0))


@pytest.mark.parametrize("margin,label", [(0.0, "tied"), (0.01, "close"), (1.0, "clear")])
def test_fixed_state_identity_agreement_and_margin_classes(monkeypatch, margin, label):
    import src.permutation.experiment as experiment
    agent = DQNAgent(seed=7)
    workload = (("A", 0, 4), ("B", 0, 2), ("C", 0, 6), ("D", 0, 8))
    adapter = RLAdapter()
    obs = adapter.reset(workload)
    state = {"observation": obs, "mask": adapter.valid_action_mask(), "time": 0, "step": 0}
    monkeypatch.setattr(experiment, "collect_states", lambda *args: [state])

    def forward(values):
        assert not torch.is_grad_enabled()
        return torch.tensor([margin, 0, -margin, -2 * margin, -10, -10]).repeat(len(values), 1)

    monkeypatch.setattr(agent.policy_network, "forward", forward)
    with greedy_evaluation(agent, 42):
        rows = fixed_state_experiment((workload,), (((0, 1, 2, 3), (1, 0, 2, 3)),), agent)
    assert all(r["action_agreement"] for r in rows if r["arrangement_id"] == 0)
    alternative = [r for r in rows if r["arrangement_id"] == 1]
    assert all(not r["action_agreement"] and r["selected_pid"] == "B" and r["preference_class"] == label for r in alternative)


def test_equivariant_shortest_remaining_network_agrees(monkeypatch):
    import src.permutation.experiment as experiment
    agent = DQNAgent(seed=7)
    workload = (("A", 0, 4), ("B", 0, 2), ("C", 0, 6), ("D", 0, 8))
    adapter = RLAdapter()
    observation = adapter.reset(workload)
    state = {"observation": observation, "mask": adapter.valid_action_mask(), "time": 0, "step": 0}
    monkeypatch.setattr(experiment, "collect_states", lambda *args: [state])
    monkeypatch.setattr(agent.policy_network, "forward", lambda values: -torch.as_tensor(values[:, :36].reshape(-1, 6, 6)[:, :, 3]))
    with greedy_evaluation(agent, 42):
        rows = fixed_state_experiment((workload,), (make_permutations(4, 13),), agent)
    assert all(row["action_agreement"] and row["selected_pid"] == "B" for row in rows)
    assert all(row["maximum_identity_q_difference"] == 0 for row in rows)


@pytest.fixture(scope="module")
def evaluated():
    agent = DQNAgent(seed=17)
    permutations = tuple(make_permutations(len(w), 63 + i) for i, w in enumerate(WORKLOADS))
    with greedy_evaluation(agent, 1729):
        actions = fixed_state_experiment(WORKLOADS, permutations, agent, maximum=3)
        rollouts, processes = workload_experiment(WORKLOADS, permutations, agent)
    return agent, permutations, actions, rollouts, processes


def test_isolation_including_populated_optimizer_replay_and_existing_gradients(monkeypatch):
    agent = DQNAgent(seed=2)
    parameter = next(agent.policy_network.parameters())
    agent.optimizer.state[parameter] = {"step": torch.tensor(3.), "exp_avg": torch.ones_like(parameter), "exp_avg_sq": torch.ones_like(parameter)}
    parameter.grad = torch.ones_like(parameter)
    agent.remember(np.zeros(39), 0, -1, np.zeros(39), True, np.zeros(6, dtype=bool))
    before = agent_snapshot(agent)
    original = agent.select_action

    def select(observation, mask, epsilon):
        assert epsilon == 0 and not torch.is_grad_enabled() and not agent.policy_network.training
        return original(observation, mask, epsilon)

    def forbidden(*args, **kwargs):
        pytest.fail("Analysis attempted learning/replay mutation")

    monkeypatch.setattr(agent, "select_action", select)
    for name in ("remember", "optimize_step", "sync_target_network"):
        monkeypatch.setattr(agent, name, forbidden)
    permutation = (make_permutations(4, 11),)
    with greedy_evaluation(agent, 1729):
        fixed_state_experiment(WORKLOADS[:1], permutation, agent, maximum=2)
        workload_experiment(WORKLOADS[:1], permutation, agent)
    assert agent_snapshot(agent) == before
    agent.optimizer.state[parameter]["exp_avg"].add_(1)
    assert agent_snapshot(agent)["optimizer"] != before["optimizer"]
    agent.replay_buffer._transitions[0][0][0] = 2
    assert agent_snapshot(agent)["replay"] != before["replay"]


def test_reward_srtf_invariance_identity_runtime_and_no_early_execution(evaluated):
    _, permutations, _, rollouts, processes = evaluated
    assert len(rollouts) == 3 * 8 * 2 and len(processes) == sum(map(len, WORKLOADS)) * 8 * 2
    for workload_id, workload in enumerate(WORKLOADS):
        control = [r for r in rollouts if r["workload_id"] == workload_id and r["policy"] == "SRTF"]
        assert len({r["total_waiting_time"] for r in control}) == 1
        for row in (r for r in rollouts if r["workload_id"] == workload_id):
            ps = [p for p in processes if p["workload_id"] == workload_id and p["policy"] == row["policy"] and p["arrangement_id"] == row["arrangement_id"]]
            assert row["total_reward"] == -sum(p["waiting_time"] for p in ps)
            assert row["total_waiting_time"] >= control[0]["total_waiting_time"]
            assert sum(p["turnaround_time"] - p["waiting_time"] for p in ps) == sum(w[2] for w in workload)
            for p in ps:
                assert workload[p["original_index"]] == (p["pid"], p["arrival_time"], p["burst_time"])
                assert permutations[workload_id][row["arrangement_id"]][p["new_slot"]] == p["original_index"]
                assert p["first_start_time"] >= p["arrival_time"] and p["completion_time"] <= row["total_time"]


def test_both_experiments_repeat_exactly(evaluated):
    agent, permutations, actions, rollouts, processes = evaluated
    before = agent_snapshot(agent)
    with greedy_evaluation(agent, 1729):
        repeated_actions = fixed_state_experiment(WORKLOADS, permutations, agent, maximum=3)
        repeated_rollouts, repeated_processes = workload_experiment(WORKLOADS, permutations, agent)
    assert tree_digest((actions, rollouts, processes)) == tree_digest((repeated_actions, repeated_rollouts, repeated_processes))
    assert agent_snapshot(agent) == before


def test_bootstrap_resamples_paired_workload_ratios_and_is_deterministic():
    numerators, denominators = np.array([0., 20.]), np.array([1., 2.])
    result = bootstrap_ratio(numerators, denominators, seed=13, replicates=200)
    assert result["estimate"] == pytest.approx(20 / 3)
    indices = np.random.default_rng(13).integers(0, 2, size=(200, 2))
    expected = numerators[indices].sum(1) / denominators[indices].sum(1)
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert result == bootstrap_ratio(numerators, denominators, seed=13, replicates=200)
    constant = bootstrap_ratio([2, 10], [1, 5], replicates=100)
    assert constant["estimate"] == constant["ci_low"] == constant["ci_high"] == 2
    assert bootstrap_ratio([0, 0], [0, 0])["estimate"] is None


def test_process_weighted_means_and_variation_match_raw_rows(evaluated):
    _, _, actions, rollouts, processes = evaluated
    summary, sensitivity = analyze(WORKLOADS, actions, rollouts, replicates=100)
    original = [p for p in processes if p["policy"] == "DQN" and p["arrangement_id"] == 0]
    permuted = [p for p in processes if p["policy"] == "DQN" and p["arrangement_id"] > 0]
    for label, rows in (("original_DQN", original), ("permuted_DQN", permuted)):
        assert summary["full_rollout"][label]["mean_waiting_time"] == pytest.approx(sum(r["waiting_time"] for r in rows) / len(rows))
    expected = sum(r["waiting_time"] for r in permuted) / len(permuted) - sum(r["waiting_time"] for r in original) / len(original)
    assert summary["full_rollout"]["paired_delta"]["estimate"] == pytest.approx(expected)
    for row in sensitivity:
        means = [r["mean_waiting_time"] for r in rollouts if r["workload_id"] == row["workload_id"] and r["policy"] == "DQN"]
        assert row["waiting_range"] == np.ptp(means)
        assert row["waiting_population_std"] == np.std(means, ddof=0)
    assert summary["fixed_state"]["comparisons"] == sum(r["arrangement_id"] > 0 for r in actions)
    assert {r["process_count_group"] for r in summary["by_process_count"]} == {4, 5, 6}


def test_exports_include_all_required_csvs_and_charts(tmp_path, evaluated):
    _, permutations, actions, rollouts, processes = evaluated
    summary, sensitivity = analyze(WORKLOADS, actions, rollouts, replicates=100)
    save_outputs(tmp_path, summary, sensitivity, actions, rollouts, processes, WORKLOADS, permutations)
    for name in ("action_equivariance.csv", "permutation_rollouts.csv", "per_workload_sensitivity.csv", "by_process_count.csv", "per_process_results.csv", "action_agreement_by_permutation.csv"):
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "permutation_summary.json").read_text())["fixed_state"] == summary["fixed_state"]
    for name in ("action_agreement_by_slot", "waiting_time_variation", "dqn_srtf_permutation_consistency"):
        assert (tmp_path / (name + ".png")).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_existing_directory_refused_before_reading_or_loading(tmp_path):
    sentinel = tmp_path / "previous.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="fresh"):
        run_experiment(training_dir=tmp_path / "not-read", output_dir=tmp_path)
    assert sentinel.read_text() == "keep"
