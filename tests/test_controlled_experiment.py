"""Fair frozen-state comparisons, paired estimates, and observer isolation."""

from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.benchmark.evaluation import greedy_evaluation
from src.permutation.audit import audit_workloads
from src.permutation.controlled_evaluation import collect_shared_states, load_state_bank, paired_comparison, save_state_bank
from src.permutation.diagnostics import train_observed
from src.permutation.experiment import fixed_state_experiment, tree_digest
from src.permutation.transforms import make_permutations
from src.training.config import TrainingConfig
from src.training.train import Trainer


def test_audit_reuses_existing_sampling_without_affecting_rng():
    before = random.getstate()
    audit = audit_workloads(200, 42)
    assert audit == audit_workloads(200, 42) and random.getstate() == before
    assert sum(audit["process_count_workloads"].values()) == 200
    assert all(row["occupied_workloads"] == 200 for row in audit["slot_attributes"][:4])
    assert audit["slot_attributes"][5]["occupied_workloads"] == audit["process_count_workloads"][6]
    assert sum(r["observed"] for r in audit["pid_placements"]) == sum(n * count for n, count in audit["process_count_workloads"].items())


def test_models_use_identical_saved_observations_without_collecting_own_rollouts(tmp_path, monkeypatch):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        workloads = ((("A", 0, 5), ("B", 1, 2), ("C", 3, 1), ("D", 0, 2)),)
        bank = collect_shared_states(workloads, DQNAgent(seed=1))
        save_state_bank(tmp_path / "states.npz", bank)
        restored = load_state_bank(tmp_path / "states.npz")
        digest = tree_digest(bank)
        assert tree_digest(restored) == digest

        def forbidden(*args, **kwargs):
            pytest.fail("A candidate model collected its own observation distribution")

        monkeypatch.setattr("src.permutation.experiment.collect_states", forbidden)
        permutations = (make_permutations(4, 63),)
        observed = []
        for seed in (2, 3):
            agent = DQNAgent(seed=seed)
            inputs = []
            hook = agent.policy_network.register_forward_pre_hook(lambda module, args: inputs.append(np.asarray(args[0]).copy()))
            with greedy_evaluation(agent, 1729):
                rows = fixed_state_experiment(workloads, permutations, agent, state_collection=restored)
            hook.remove()
            observed.append(inputs)
            assert len(rows) == sum(map(len, bank.values())) * 8
        assert tree_digest(observed[0]) == tree_digest(observed[1])
        assert tree_digest(restored) == digest
    finally:
        torch.set_num_threads(original_threads)


def test_paired_model_comparison_weights_processes_and_retains_state_pairs():
    workloads = ((0,) * 4, (0,) * 6)
    evaluated = {}
    for name, totals, agreement_count in (("reference", [8, 24], [2, 4]), ("candidate", [4, 30], [5, 5])):
        evaluated[name] = {"rollouts": [], "actions": [], "sensitivity": []}
        for wid, total in enumerate(totals):
            for arrangement in range(8):
                evaluated[name]["rollouts"].append({"workload_id": wid, "arrangement_id": arrangement,
                    "policy": "DQN", "total_waiting_time": total, "mean_response_time": total / len(workloads[wid]),
                    "context_switches": 2, "preemptions": 1})
            for a in range(7):
                evaluated[name]["actions"].append({"workload_id": wid, "arrangement_id": a + 1,
                                                    "action_agreement": a < agreement_count[wid]})
            evaluated[name]["sensitivity"].append({"workload_id": wid, "waiting_range": 0, "waiting_population_std": 0})
    rows = paired_comparison("candidate", "reference", evaluated, workloads, replicates=100)
    overall = {r["metric"]: r for r in rows if r["process_count_group"] == "all"}
    assert overall["original_waiting"]["estimate"] == pytest.approx(.2)
    assert overall["permuted_waiting"]["estimate"] == pytest.approx(.2)
    assert overall["fixed_state_agreement"]["estimate"] == pytest.approx(4 / 14)
    assert overall["waiting_range"]["estimate"] == 0
    assert overall["original_waiting"]["workloads"] == 2


def test_diagnostics_do_not_change_training_or_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.train.generate_plots", lambda path: None)
    config = TrainingConfig(episodes=3, validation_workloads=2, validation_interval=1,
        replay_warmup=5, agent_options={"batch_size": 2}, replay_augmentation="permutation")
    plain = Trainer(config)
    plain.run(tmp_path / "plain", verbose=False)
    summary = train_observed(config, tmp_path / "observed")
    for name in ("training_metrics.csv", "validation_metrics.csv"):
        assert (tmp_path / "plain" / name).read_bytes() == (tmp_path / "observed" / name).read_bytes()
    assert summary["optimizer_updates"] == plain.agent.optimizer_steps
    assert summary["checkpoint_scores_verified"] and summary["nonfinite_loss_gradient_q_count"] == 0
