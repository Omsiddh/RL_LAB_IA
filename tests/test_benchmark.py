"""Stage 6.2 evaluation isolation, paired comparisons, and result exports."""

from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.benchmark.analysis import aggregate, analyze_results, percentage_gap
from src.benchmark.evaluation import POLICIES, evaluate_workloads, weights_digest
from src.benchmark.run import main, reconstruct_validation, run_benchmark, save_outputs


WORKLOADS = (
    (("single", 2, 3),),
    (("later", 3, 1), ("first", 0, 4)),
    (("P1", 0, 5), ("P2", 1, 2), ("P3", 3, 1)),
)


def test_reconstructs_exact_phase5_collection(tmp_path):
    path = tmp_path / "training_config.json"
    path.write_text(json.dumps({"validation_seed": 1729, "validation_workloads": 128}))
    config, workloads = reconstruct_validation(tmp_path)
    assert config["validation_seed"] == 1729
    assert isinstance(workloads, tuple)
    assert len(workloads) == 128 and sum(map(len, workloads)) == 645
    assert workloads == reconstruct_validation(tmp_path)[1]
    assert {len(workload) for workload in workloads} == {4, 5, 6}
    assert workloads[1] == (("P1", 7, 5), ("P2", 4, 5), ("P4", 0, 4), ("P3", 7, 5))


@pytest.mark.parametrize("seed,count", [(1730, 128), (1729, 127)])
def test_rejects_replacement_validation_collection(tmp_path, seed, count):
    (tmp_path / "training_config.json").write_text(json.dumps({"validation_seed": seed, "validation_workloads": count}))
    with pytest.raises(ValueError, match="original"):
        reconstruct_validation(tmp_path)


def test_all_policies_receive_identical_original_workloads(monkeypatch):
    import src.benchmark.evaluation as evaluation
    observed = {policy: [] for policy in POLICIES}
    original_dqn = evaluation._run_dqn

    def dqn(workload, *args):
        observed["DQN"].append(workload)
        return original_dqn(workload, *args)

    monkeypatch.setattr(evaluation, "_run_dqn", dqn)
    for policy, name in (("FCFS", "run_fcfs"), ("SRTF", "run_srtf"), ("Round Robin", "run_round_robin")):
        original = getattr(evaluation, name)

        def wrapper(workload, _original=original, _policy=policy, **kwargs):
            observed[_policy].append(workload)
            if _policy == "Round Robin":
                assert kwargs["quantum"] == 2
            return _original(workload, **kwargs)

        monkeypatch.setattr(evaluation, name, wrapper)
    before = deepcopy(WORKLOADS)
    results, _ = evaluate_workloads(WORKLOADS, DQNAgent(seed=1))
    assert WORKLOADS == before
    for policy in POLICIES:
        assert all(received is original for received, original in zip(observed[policy], WORKLOADS))
        for workload, result in zip(WORKLOADS, results):
            assert set(result[policy]["processes"]) == {pid for pid, _, _ in workload}
            assert all(workload[index][0] == pid for _, _, index, pid in result[policy]["timeline"] if index is not None)


@pytest.mark.parametrize("training_mode", [True, False])
def test_evaluation_is_greedy_and_preserves_model_replay_rng_and_gradients(monkeypatch, training_mode):
    agent = DQNAgent(seed=9)
    agent.policy_network.train(training_mode)
    state = np.zeros(39, dtype=np.float32)
    agent.remember(state, 0, -1, state, True, np.zeros(6, dtype=bool))
    replay = deepcopy(list(agent.replay_buffer._transitions))
    rng, replay_rng = agent.rng.getstate(), agent.replay_buffer.rng.getstate()
    digest = weights_digest(agent)
    original_select = agent.select_action

    def select(observation, mask, epsilon):
        assert epsilon == 0 and not torch.is_grad_enabled()
        assert not agent.policy_network.training
        action = original_select(observation, mask, epsilon)
        assert mask[action]
        return action

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation attempted training or replay mutation")

    monkeypatch.setattr(agent, "select_action", select)
    for name in ("remember", "optimize_step", "sync_target_network"):
        monkeypatch.setattr(agent, name, forbidden)
    first = evaluate_workloads(WORKLOADS, agent)
    assert first == evaluate_workloads(WORKLOADS, agent)
    assert weights_digest(agent) == digest and agent.optimizer_steps == 0
    assert agent.policy_network.training == training_mode
    assert agent.rng.getstate() == rng and agent.replay_buffer.rng.getstate() == replay_rng
    assert len(agent.replay_buffer) == len(replay)
    for old, new in zip(replay, agent.replay_buffer._transitions):
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)
    assert all(parameter.grad is None for parameter in agent.policy_network.parameters())


def test_rng_and_mode_restored_on_evaluation_failure(monkeypatch):
    agent = DQNAgent(seed=2)
    rng = agent.rng

    def fail(*args, **kwargs):
        raise RuntimeError("selection failure")

    monkeypatch.setattr(agent, "select_action", fail)
    with pytest.raises(RuntimeError, match="selection failure"):
        evaluate_workloads(WORKLOADS, agent)
    assert agent.rng is rng and agent.policy_network.training


def test_weight_mutation_is_detected(monkeypatch):
    agent = DQNAgent(seed=1)
    original = agent.select_action

    def mutate(observation, mask, epsilon):
        with torch.no_grad():
            agent.policy_network.layers[-1].bias.add_(0.01)
        return original(observation, mask, epsilon)

    monkeypatch.setattr(agent, "select_action", mutate)
    with pytest.raises(AssertionError, match="weights"):
        evaluate_workloads(WORKLOADS, agent)


def test_process_weighted_timing_and_workload_weighted_switches():
    processes = [dict(waiting_time=0, turnaround_time=1, response_time=0),
                 dict(waiting_time=10, turnaround_time=11, response_time=4),
                 dict(waiting_time=10, turnaround_time=11, response_time=6)]
    workloads = [dict(total_reward=0, context_switches=2, preemptions=1),
                 dict(total_reward=-20, context_switches=6, preemptions=3)]
    result = aggregate(processes, workloads)
    assert result["mean_waiting_time"] == pytest.approx(20/3)
    assert result["mean_turnaround_time"] == pytest.approx(23/3)
    assert result["mean_response_time"] == pytest.approx(10/3)
    assert result["mean_context_switches"] == 4 and result["mean_preemptions"] == 2
    assert result["total_reward"] == -20
    workloads[0]["total_reward"] = 1
    with pytest.raises(AssertionError, match="reward"):
        aggregate(processes, workloads)


def test_reward_inequalities_gaps_and_process_count_groups():
    results, decisions = evaluate_workloads(WORKLOADS, DQNAgent(seed=5))
    comparison, workloads, processes, summary = analyze_results(WORKLOADS, results, decisions)
    assert len(comparison) == 4 and len(workloads) == len(WORKLOADS)
    assert len(processes) == 4 * sum(map(len, WORKLOADS))
    for result in results:
        for policy in POLICIES:
            assert result["SRTF"]["total_reward"] >= result[policy]["total_reward"]
            assert result[policy]["total_reward"] == -sum(p["waiting_time"] for p in result[policy]["processes"].values())
    for row in workloads:
        assert row["dqn_srtf_total_waiting_gap"] >= 0
        assert row["dqn_srtf_mean_waiting_gap"] == pytest.approx(row["dqn_mean_waiting_time"] - row["srtf_mean_waiting_time"])
    assert summary["optimality_gap"]["dqn_srtf_tied_workloads"] == sum(row["dqn_srtf_total_waiting_gap"] == 0 for row in workloads)
    for group in summary["by_process_count"]:
        subset = [row for row in processes if row["policy"] == group["policy"] and row["process_count"] == group["process_count_group"]]
        assert group["mean_waiting_time"] == sum(row["waiting_time"] for row in subset) / len(subset)


def test_percentage_gaps_handle_zero_reference():
    assert percentage_gap(12, 10) == 20
    assert percentage_gap(0, 0) == 0
    assert percentage_gap(1, 0) is None


def test_slot_diagnostics_account_for_ready_opportunities_and_identical_features():
    from src.benchmark.analysis import slot_diagnostics
    decisions = [dict(process_count=4, process_index=0, selected=True, unique_shortest=False,
                      equivalent_choice_candidate=True, equivalent_choice_expected=0.5),
                 dict(process_count=4, process_index=1, selected=False, unique_shortest=False,
                      equivalent_choice_candidate=True, equivalent_choice_expected=0.5)]
    rows = {row["process_index"]: row for row in slot_diagnostics(decisions) if row["process_count_group"] == "all"}
    assert rows[0]["identical_feature_selection_surplus"] == 0.5
    assert rows[1]["identical_feature_selection_surplus"] == -0.5
    assert rows[0]["unique_shortest_selection_rate"] is None


def test_result_exports_and_all_five_charts(tmp_path):
    results, decisions = evaluate_workloads(WORKLOADS, DQNAgent(seed=8))
    comparison, workloads, processes, summary = analyze_results(WORKLOADS, results, decisions)
    save_outputs(tmp_path, WORKLOADS, comparison, workloads, processes, decisions, summary)
    for name, count in (("baseline_comparison.csv", 4), ("per_workload_comparison.csv", 3), ("per_process_results.csv", 24)):
        with (tmp_path / name).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    assert json.loads((tmp_path / "benchmark_summary.json").read_text())["comparison"] == comparison
    definitions = json.loads((tmp_path / "validation_workloads.json").read_text())
    assert definitions == [[list(process) for process in workload] for workload in WORKLOADS]
    for key in ("waiting_time", "turnaround_time", "response_time", "context_switches", "preemptions"):
        assert (tmp_path / (key + "_comparison.png")).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_empty_or_mismatched_workloads_are_rejected():
    with pytest.raises(ValueError):
        evaluate_workloads((), DQNAgent())
    with pytest.raises(ValueError):
        analyze_results(WORKLOADS, [], [])


def test_existing_output_is_preserved_before_loading_checkpoint(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    (training / "training_config.json").write_text(json.dumps({"validation_seed": 1729, "validation_workloads": 128}))
    (training / "best_model.json").write_text(json.dumps({"episode": 5250}))
    (training / "best_model.pt").write_bytes(b"never loaded")
    (training / "validation_metrics.csv").write_text("episode\n5250\n")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "prior_results.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError):
        run_benchmark(training, output)
    assert sentinel.read_text() == "preserve me"


def test_cli_passes_paths_to_benchmark(monkeypatch, tmp_path):
    calls = []

    def benchmark(training_dir, output_dir):
        calls.append((training_dir, output_dir))
        return {"comparison": [], "optimality_gap": {}, "checks": {}}

    monkeypatch.setattr("src.benchmark.run.run_benchmark", benchmark)
    main(["--training-dir", str(tmp_path / "phase5"), "--output-dir", str(tmp_path / "result")])
    assert calls == [(tmp_path / "phase5", tmp_path / "result")]
