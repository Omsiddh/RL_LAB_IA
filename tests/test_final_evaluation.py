"""Synthetic-only tests: never generate or inspect the held-out collection."""

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.baselines import run_fcfs, run_srtf
from src.env.rl_adapter import RLAdapter
from src.final_evaluation.analysis import PRIMARY, CONTROL, analyze, bootstrap_intervals
from src.final_evaluation.evaluation import evaluate_all, global_rng_snapshot, verify_result
from src.final_evaluation import protocol
from src.final_evaluation.protocol import MODELS, POLICIES, sha256, workload_digest
from src.final_evaluation.run import verify_protected
from src.permutation.experiment import agent_snapshot


@pytest.fixture
def workloads():
    # Include initial/mid-rollout idle, out-of-order identities, ties, and
    # unequal process counts so weighting mistakes cannot hide.
    return ((("C", 2, 5), ("A", 2, 2), ("D", 5, 1), ("B", 8, 2)),
            (("E", 0, 1), ("A", 4, 2), ("C", 4, 2), ("B", 4, 1), ("D", 6, 4)),
            (("F", 0, 8), ("B", 1, 3), ("E", 3, 1), ("A", 4, 1), ("D", 4, 2), ("C", 6, 1)))


@pytest.fixture
def agents():
    return {policy: DQNAgent(seed=17 + i, batch_size=1) for i, policy in enumerate(MODELS)}


def test_read_only_repeat_with_populated_optimizer_replay_and_gradients(workloads, agents, monkeypatch):
    for agent in agents.values():
        adapter = RLAdapter()
        state = adapter.reset(workloads[0])
        next_state, reward, done, info = adapter.step(0)
        agent.remember(state, 0, reward, next_state, done, info["action_mask"])
        assert agent.optimize_step() is not None  # Synthetic fixture, not any checkpoint.
    before = {p: agent_snapshot(a) for p, a in agents.items()}
    global_before = global_rng_snapshot()
    digest = workload_digest(workloads)

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation attempted training/replay/target synchronization")

    for agent in agents.values():
        monkeypatch.setattr(agent, "remember", forbidden)
        monkeypatch.setattr(agent, "optimize_step", forbidden)
        monkeypatch.setattr(agent, "sync_target_network", forbidden)
    first, checks = evaluate_all(workloads, agents)
    second, repeated_checks = evaluate_all(workloads, agents)
    assert first == second and checks == repeated_checks
    assert checks["completed_rollouts"] == 18 and checks["srtf_inequality_violations"] == 0
    assert before == {p: agent_snapshot(a) for p, a in agents.items()}
    assert global_rng_snapshot() == global_before and workload_digest(workloads) == digest


def test_each_policy_receives_the_identical_workload_objects(workloads, agents, monkeypatch):
    import src.final_evaluation.evaluation as evaluation
    observed = []
    for name in ("run_srtf", "run_fcfs", "run_round_robin", "run_dqn"):
        original = getattr(evaluation, name)
        def spy(workload, *args, _original=original, **kwargs):
            observed.append(workload)
            return _original(workload, *args, **kwargs)
        monkeypatch.setattr(evaluation, name, spy)
    evaluate_all(workloads, agents)
    assert len(observed) == 18
    assert all(observed[wid * 6 + p] is workload for wid, workload in enumerate(workloads) for p in range(6))


def test_eps_zero_no_grad_eval_and_invalid_high_q_slots_masked(workloads, agents, monkeypatch):
    for agent in agents.values():
        with torch.no_grad():
            for parameter in agent.policy_network.parameters():
                parameter.zero_()
            agent.policy_network.layers[-1].bias.copy_(torch.arange(6, dtype=torch.float32) * 100)
        original = agent.select_action
        def checked(state, mask, epsilon, _agent=agent, _original=original):
            assert epsilon == 0.0 and not torch.is_grad_enabled()
            assert not _agent.policy_network.training
            action = _original(state, mask, epsilon)
            assert mask[action]
            return action
        monkeypatch.setattr(agent, "select_action", checked)
    results, _ = evaluate_all(workloads, agents)
    for wid, workload in enumerate(workloads):
        for result in results[wid].values():
            verify_result(workload, result)


def test_invalid_dqn_action_rejected_and_modes_rng_restored(workloads, agents, monkeypatch):
    before = {p: agent_snapshot(a) for p, a in agents.items()}
    monkeypatch.setattr(agents[PRIMARY], "select_action", lambda *args, **kwargs: 5)
    with pytest.raises(AssertionError, match="mask"):
        evaluate_all(workloads, agents)
    assert before == {p: agent_snapshot(a) for p, a in agents.items()}


@pytest.mark.parametrize("fault,match", [("identity", "identities"), ("index", "index/identity"),
    ("arrival", "before arrival"), ("duration", "durations"), ("runtime", "burst accounting"),
    ("total_time", "accounting"), ("idle", "accounting"), ("completion", "completion/start"),
    ("turnaround", "turnaround/response"), ("response", "turnaround/response"),
    ("reward", "negative total waiting"), ("mean", "Reported process means")])
def test_timeline_and_metric_guards_detect_corruption(fault, match):
    workload = (("A", 2, 2), ("B", 4, 1))
    result = deepcopy(run_fcfs(workload))
    timeline = list(result["timeline"])
    if fault == "identity":
        result["processes"]["X"] = result["processes"].pop("A")
    elif fault == "index":
        timeline[1] = (*timeline[1][:2], 1, "A")
    elif fault == "arrival":
        workload = (("A", 3, 2), ("B", 4, 1))
    elif fault == "duration":
        timeline[1] = (2, 2, 0, "A")
    elif fault == "runtime":
        workload = (("A", 2, 3), ("B", 4, 1))
    elif fault in ("total_time", "idle"):
        result["total_time" if fault == "total_time" else "idle_time"] += 1
    elif fault == "completion":
        result["processes"]["A"]["completion_time"] += 1
    elif fault in ("turnaround", "response"):
        result["processes"]["A"][fault + "_time"] += 1
    elif fault == "reward":
        result["total_reward"] -= 1
    else:
        result["mean_waiting_time"] += 1
    result["timeline"] = tuple(timeline)
    with pytest.raises(AssertionError, match=match):
        verify_result(workload, result)


def test_srtf_inequality_guard(workloads, agents, monkeypatch):
    monkeypatch.setattr("src.final_evaluation.evaluation.run_srtf", run_fcfs)
    monkeypatch.setattr("src.final_evaluation.evaluation.run_fcfs", run_srtf)
    with pytest.raises(AssertionError, match="optimality"):
        evaluate_all(workloads, agents)


def test_process_weighting_counts_groups_and_examples(workloads, agents):
    results, _ = evaluate_all(workloads, agents)
    comparison, wide, processes, grouped, paired, examples, summary = analyze(workloads, results)
    assert len(comparison) == 6 and len(grouped) == 18 and len(processes) == 90
    assert len(wide) == len(paired) == 3
    for row in comparison:
        policy = row["policy"]
        totals = [r[policy]["total_reward"] for r in results]
        assert row["process_count"] == 15 and row["workload_count"] == 3
        assert row["mean_waiting_time"] == -sum(totals) / 15
        assert row["mean_reward"] == sum(totals) / 3
        assert row["srtf_tied_workloads"] + row["srtf_above_workloads"] == 3
    differences = [p["augmented_minus_original_dqn_10k_mean_waiting_difference"] for p in paired]
    assert examples[0]["augmented_minus_original_waiting"] == min(differences)
    assert examples[1]["augmented_minus_original_waiting"] == max(differences)
    assert sum(summary["augmentation_workload_outcomes"].values()) == 3
    for group in grouped:
        assert group["process_count"] == group["process_count_group"]


def test_bootstrap_recalculates_ratios_and_retains_all_policy_pairs():
    workloads = ((None,) * 4, (None,) * 6)
    results = []
    for wid, count in enumerate((4, 6)):
        results.append({policy: {"context_switches": i + wid, "preemptions": i,
            "processes": {str(j): {"waiting_time": (i + 1) * (wid + 1), "response_time": i + wid}
                          for j in range(count)}} for i, policy in enumerate(POLICIES)})
    rows = bootstrap_intervals(workloads, results, seed=19, replicates=5000)
    assert rows == bootstrap_intervals(workloads, results, seed=19, replicates=5000)
    primary = next(r for r in rows if r["primary"])
    sample = np.random.default_rng(19).integers(0, 2, size=(5000, 2))
    differences, counts = np.array([-4, -12]), np.array([4, 6])
    expected = differences[sample].sum(axis=1) / counts[sample].sum(axis=1)
    assert primary["difference"] == -1.6
    assert (primary["ci_low"], primary["ci_high"]) == tuple(np.percentile(expected, [2.5, 97.5]))
    contrasts = {(r["left_policy"], r["right_policy"]): r for r in rows if r["group"] == "all" and r["metric"] == "waiting_time"}
    # All policy totals in this fixture are proportional; common draws preserve
    # this relationship in BOTH ends of every percentile interval.
    for key, row in contrasts.items():
        factor = POLICIES.index(key[0]) - POLICIES.index(key[1])
        low, high = np.percentile(expected * -factor, [2.5, 97.5])
        assert row["ci_low"] == low and row["ci_high"] == high


def test_freeze_refuses_overwrite_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "DQNAgent", lambda *args, **kwargs: pytest.fail("Loaded checkpoint before overwrite rejection"))
    with pytest.raises(FileExistsError):
        protocol.freeze_protocol(tmp_path)


def test_freeze_saves_verified_model_and_source_manifests_before_sampling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for folder in ("src", "tests", "runs/phase6_3b_controlled"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "pyproject.toml", "requirements.txt", "src/probe.py"):
        (tmp_path / name).write_text("synthetic fixture")
    models, hashes = {}, {}
    for i, policy in enumerate(MODELS):
        folder = tmp_path / "runs" / str(i)
        folder.mkdir()
        checkpoint = folder / "best_model.pt"
        agent = DQNAgent(seed=31 + i)
        agent.save(checkpoint)
        episode = 100 + i
        validation = {"mean_waiting_time": 1.0, "mean_turnaround_time": 2.0, "mean_reward": -4.0}
        metadata = {"episode": episode, "optimizer_updates": 0,
                    "configuration": {"agent_config": agent.config}, "validation": validation}
        checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
        (folder / "validation_metrics.csv").write_text("episode,mean_waiting_time,mean_turnaround_time,mean_reward\n" + f"{episode},1.0,2.0,-4.0\n")
        models[policy] = (str(checkpoint), episode)
        hashes[str(checkpoint)] = sha256(checkpoint)
    (tmp_path / "runs/phase6_3b_controlled/controlled_summary.json").write_text(json.dumps({"protected_input_sha256": hashes}))
    monkeypatch.setattr(protocol, "MODELS", models)
    monkeypatch.setattr(protocol, "sample_workload", lambda rng: pytest.fail("Sampled test data before protocol was frozen"))
    output = tmp_path / "runs/new_final"
    frozen, manifest, loaded = protocol.freeze_protocol(output)
    assert (output / "frozen_protocol.json").exists()
    assert (output / "selected_model_manifest.json").exists()
    assert not (output / "test_workloads.json").exists()
    assert len(manifest) == len(loaded) == 3
    assert all(m["greedy_load_probe_passed"] and m["checkpoint_sha256"] == hashes[m["checkpoint"]] for m in manifest)
    assert verify_protected(frozen["source_configuration_and_prior_artifact_sha256"]) > 0


def test_protected_hash_detects_changes_or_removal(tmp_path):
    path = tmp_path / "prior.csv"
    path.write_text("original")
    frozen = {str(path): sha256(path)}
    assert verify_protected(frozen) == 1
    path.write_text("changed")
    with pytest.raises(AssertionError, match="Protected"):
        verify_protected(frozen)
    path.unlink()
    with pytest.raises(AssertionError, match="Protected"):
        verify_protected(frozen)


def test_collection_sampling_saves_duplicates_without_redraw(tmp_path, monkeypatch):
    # Inject a toy collection and a non-test seed; do not call the real sampler.
    workload = (("A", 0, 2), ("B", 1, 1), ("C", 2, 1), ("D", 2, 2))
    calls = []
    def sampler(rng):
        calls.append(rng.getstate())
        rng.random()
        return workload
    monkeypatch.setattr(protocol, "sample_workload", sampler)
    monkeypatch.setattr(protocol, "reconstruct_validation", lambda _: ({}, (tuple(reversed(workload)),)))
    rng_before = random.getstate()
    generated, metadata = protocol.prepare_test_workloads(tmp_path, {"test_collection": {"seed": 23, "workload_count": 256}})
    assert len(calls) == 256 and len(generated) == 256
    assert metadata["within_test_exact_duplicate_workloads"] == 255
    assert metadata["arrival_burst_multiset_cross_split_duplicate_workloads"] == 256
    assert metadata["exact_cross_split_duplicate_workloads"] == 0
    assert random.getstate() == rng_before
    assert metadata["workload_sha256"] == workload_digest(generated)
    assert json.loads((tmp_path / "test_workloads.json").read_text())[0] == [list(p) for p in workload]


def test_agent_mutation_is_detected(workloads, agents, monkeypatch):
    original = agents[PRIMARY].select_action
    def mutating(*args, **kwargs):
        agents[PRIMARY].optimizer_steps += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(agents[PRIMARY], "select_action", mutating)
    with pytest.raises(AssertionError, match="changed model"):
        evaluate_all(workloads, agents)


def test_missing_policy_or_empty_collection_rejected(workloads, agents):
    with pytest.raises(ValueError):
        evaluate_all((), agents)
    with pytest.raises(ValueError):
        evaluate_all(workloads, {PRIMARY: agents[PRIMARY]})


def test_csv_json_report_and_all_charts_roundtrip_synthetic_results(tmp_path, workloads, agents):
    from src.benchmark.run import write_csv, write_json
    from src.final_evaluation.plots import comparison_charts
    from src.final_evaluation.report import write_report
    results, checks = evaluate_all(workloads, agents)
    comparison, wide, processes, grouped, paired, examples, summary = analyze(workloads, results)
    intervals = bootstrap_intervals(workloads, results, seed=19, replicates=5000)
    for name, rows in (("policy_comparison", comparison), ("per_workload_comparison", wide),
                       ("per_process_results", processes), ("by_process_count", grouped),
                       ("paired_model_comparison", paired), ("bootstrap_intervals", intervals)):
        write_csv(tmp_path / (name + ".csv"), rows)
    summary.update({"bootstrap_intervals": intervals, "bootstrap_seed": 19, "checks": checks, "selected_models": [],
        "test_collection": {"seed": 23, "process_count": 15, "process_count_workloads": {"4": 1, "5": 1, "6": 1},
            "workload_sha256": workload_digest(workloads), "validation_workload_sha256": "synthetic",
            "exact_cross_split_duplicate_workloads": 0, "arrival_burst_multiset_cross_split_duplicate_workloads": 0,
            "within_test_exact_duplicate_workloads": 0, "within_test_arrival_burst_multiset_duplicates": 0}})
    write_json(tmp_path / "benchmark_summary.json", summary)
    write_json(tmp_path / "selected_timelines.json", examples)
    assert json.loads((tmp_path / "benchmark_summary.json").read_text())["comparison"] == comparison
    comparison_charts(comparison, grouped, tmp_path)
    write_report(tmp_path, summary, "Synthetic artifact smoke test")
    assert len(list(tmp_path.glob("*.png"))) == 7
    assert "Synthetic artifact smoke test" in (tmp_path / "analysis_report.md").read_text()
