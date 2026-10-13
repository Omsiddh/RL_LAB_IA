"""Verify real model outputs, transition replay, and saved evaluation agreement."""

import csv
import hashlib
import json

import numpy as np
import pytest
import torch

from main import DEFAULT_CHECKPOINT, PROJECT_ROOT
from src.agent import DQNAgent
from src.benchmark.evaluation import evaluate_workloads, weights_digest
from src.benchmark.export_traces import export_traces
from src.env.rl_adapter import RLAdapter


def verify_decisions(workload, decisions, agent):
    adapter = RLAdapter()
    observation = adapter.reset(workload)
    reward_sum = 0
    for number, decision in enumerate(decisions):
        assert decision["step"] == number
        assert decision["time"] == adapter.env.time
        np.testing.assert_array_equal(decision["observation"], observation)
        assert decision["action_mask"] == adapter.valid_action_mask().tolist()
        with torch.no_grad():
            actual_q = agent.policy_network(observation).cpu().tolist()
        assert decision["q_values"] == actual_q
        valid = adapter.env.valid_actions()
        assert decision["selected_action"] == max(valid, key=lambda i: actual_q[i])
        assert decision["masked_q_values"] == [
            value if i in valid else None for i, value in enumerate(actual_q)]
        for index in range(6):
            if index >= len(workload) or workload[index][1] > adapter.env.time:
                assert decision["observation"][index * 6:index * 6 + 6] == [0] * 6
        observation, reward, done, info = adapter.step(decision["selected_action"])
        reward_sum += reward
        assert decision["reward"] == reward == -(len(valid) - 1)
        assert decision["selected_pid"] == info["pid"]
        assert decision["execution"] == {
            "start": info["start_time"], "end": info["end_time"],
            "process_index": info["process_index"], "pid": info["pid"]}
        assert decision["context_switch"] == info["context_switch"]
        assert decision["preemption"] == info["preemption"]
        assert decision["process_completed"] == info["completed"]
        idle = (dict(zip(("start", "end", "process_index", "pid"), info["idle_interval"]))
                if info["idle_interval"] is not None else None)
        assert decision["idle_interval_after"] == idle
        assert decision["next_time"] == adapter.env.time
        np.testing.assert_array_equal(decision["next_observation"], observation)
        assert decision["next_action_mask"] == info["action_mask"].tolist()
        assert decision["done"] == done
        assert decision["cumulative_reward"] == reward_sum
    assert adapter.env.is_done()
    assert decisions[-1]["next_action_mask"] == [False] * 6
    assert len(decisions) == sum(burst for _, _, burst in workload)


def test_trace_collection_is_passive_and_handles_idle_jumps():
    workloads = ((("first", 2, 1), ("later", 5, 2)),)
    agent = DQNAgent(seed=9)
    before_weights, before_rng = weights_digest(agent), agent.rng.getstate()
    without_traces = evaluate_workloads(workloads, agent)
    traces = []
    assert evaluate_workloads(workloads, agent, traces=traces) == without_traces
    assert weights_digest(agent) == before_weights and agent.rng.getstate() == before_rng
    assert agent.optimizer_steps == 0 and len(agent.replay_buffer) == 0
    assert all(parameter.grad is None for parameter in agent.policy_network.parameters())
    verify_decisions(workloads[0], traces[0], agent)
    assert traces[0][0]["time"] == 2
    assert traces[0][0]["execution"]["end"] == 3
    assert traces[0][0]["idle_interval_after"] == {
        "start": 3, "end": 5, "process_index": None, "pid": None}
    assert traces[0][0]["next_time"] == 5


def test_exported_checkpoint_traces_match_frozen_evaluation(tmp_path):
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Saved checkpoint is required for the inference regression")
    digest_before = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
    paths = export_traces(output_dir=tmp_path)
    assert [path.name for path in paths] == ["workload_0.json", "workload_69.json", "workload_201.json"]
    agent = DQNAgent(device="cpu")
    agent.load(DEFAULT_CHECKPOINT)
    final = PROJECT_ROOT / "runs/phase6_4_final_test"
    with (final / "per_workload_comparison.csv").open(newline="") as stream:
        recorded = {int(row["workload_id"]): row for row in csv.DictReader(stream)}
    cases = {case["workload_id"]: case for case in
             json.loads((final / "selected_timelines.json").read_text(encoding="utf-8"))}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Infinity" not in text and "NaN" not in text
        payload = json.loads(text)
        assert payload["model"]["checkpoint_sha256"] == digest_before
        assert payload["model"]["source"] == "pytorch_checkpoint_inference"
        assert payload["model"]["architecture"] == [39, 64, 64, 6]
        workload = tuple((p["pid"], p["arrival_time"], p["burst_time"]) for p in payload["workload"])
        verify_decisions(workload, payload["decisions"], agent)
        row = recorded[payload["workload_id"]]
        for metric in ("mean_waiting_time", "mean_turnaround_time", "mean_response_time",
                       "context_switches", "preemptions", "total_reward", "total_time", "idle_time"):
            assert payload[metric] == pytest.approx(float(row["augmented_dqn_10k_" + metric]))
        assert payload["total_reward"] == -sum(p["waiting_time"] for p in payload["processes"].values())
        if payload["workload_id"] in cases:
            expected = cases[payload["workload_id"]]["timelines"]
            as_arrays = lambda timeline: [[t["start"], t["end"], t["process_index"], t["pid"]]
                                          for t in timeline]
            assert as_arrays(payload["timeline"]) == expected["Augmented DQN 10k"]
            for name in ("FCFS", "SRTF", "Round Robin"):
                assert as_arrays(payload["baselines"][name]["timeline"]) == expected[name]
    assert hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest() == digest_before


@pytest.mark.parametrize("ids", [[], [-1], [256], [69, 69], [True]])
def test_invalid_ids_fail_before_model_loading_or_output(tmp_path, monkeypatch, ids):
    import src.benchmark.export_traces as exporter
    collection = tmp_path / "workloads.json"
    collection.write_text(json.dumps([[['P1', 0, 1]]] * 256))
    monkeypatch.setattr(exporter, "TEST_WORKLOADS", collection)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="distinct workload IDs"):
        export_traces(ids, checkpoint=tmp_path / "missing.pt", output_dir=output)
    assert not output.exists()
