"""Training orchestration, isolation, checkpoint selection, and integration."""

from copy import deepcopy
import csv
import json
import random

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.env.rl_adapter import RLAdapter
from src.training.config import TrainingConfig
from src.training.plots import generate_plots, moving_average
from src.training.train import Trainer, main
from src.training.validation import Validator, episode_metrics, sample_workload


@pytest.fixture
def fast_plots(monkeypatch):
    # Plot generation has separate integration coverage below.
    monkeypatch.setattr("src.training.train.generate_plots", lambda path: None)


def small_config(**overrides):
    settings = dict(episodes=3, validation_workloads=2, validation_interval=2,
                    replay_warmup=5, optimization_interval=3, seed=7,
                    agent_options={"batch_size": 2, "target_sync_interval": 2})
    settings.update(overrides)
    return TrainingConfig(**settings)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_training_defaults_reuse_dqn_configuration():
    config = TrainingConfig()
    assert (config.episodes, config.replay_warmup, config.optimization_interval) == (6000, 1000, 4)
    assert (config.validation_interval, config.validation_workloads) == (250, 128)
    trainer = Trainer(config)
    assert trainer.agent.config == DQNAgent(seed=config.seed).config
    assert config.agent_options == {}
    assert len(trainer.validator.workloads) == 128


def test_linear_epsilon_uses_global_steps_and_clamps():
    config = TrainingConfig()
    assert config.epsilon_at(0) == 1
    assert config.epsilon_at(50000) == pytest.approx(0.525)
    assert config.epsilon_at(100000) == 0.05
    assert config.epsilon_at(200000) == 0.05
    custom = TrainingConfig(initial_epsilon=0.7, minimum_epsilon=0.2, epsilon_decay_steps=10)
    assert custom.epsilon_at(5) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        config.epsilon_at(-1)


@pytest.mark.parametrize("options", [
    {"episodes": -1}, {"episodes": True}, {"replay_warmup": -1},
    {"optimization_interval": 0}, {"validation_interval": 0}, {"validation_workloads": 0},
    {"epsilon_decay_steps": 0}, {"initial_epsilon": float("nan")},
    {"minimum_epsilon": 1.1}, {"initial_epsilon": 0.1, "minimum_epsilon": 0.2},
    {"seed": None}, {"validation_seed": True}, {"torch_threads": 0},
    {"agent_options": {"seed": 3}},
])
def test_invalid_training_configuration(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options)


def test_impossible_warmup_is_rejected():
    with pytest.raises(ValueError, match="capacity"):
        Trainer(small_config(replay_warmup=9, agent_options={"replay_capacity": 8, "batch_size": 2}))


def test_workloads_are_reproducible_permuted_and_within_bounds(monkeypatch):
    first, second = random.Random(8), random.Random(8)
    workloads = [sample_workload(first) for _ in range(20)]
    assert workloads == [sample_workload(second) for _ in range(20)]
    for workload in workloads:
        assert 4 <= len(workload) <= 6
        assert all(0 <= arrival <= 8 and 1 <= burst <= 10 for _, arrival, burst in workload)
    fixed = tuple((f"P{i}", i, 6 - i) for i in range(6))
    monkeypatch.setattr("src.training.validation.generate_workload", lambda seed: fixed)
    permutations = [sample_workload(first) for _ in range(20)]
    assert len(set(permutations)) > 1
    assert all(set(workload) == set(fixed) for workload in permutations)


def test_validator_fixed_collection_does_not_touch_global_rngs():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    first = Validator(8, seed=31)
    assert first.workloads == Validator(8, seed=31).workloads
    assert first.workloads != Validator(8, seed=32).workloads
    assert random.getstate() == python_state
    assert_tree_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_validation_is_greedy_and_preserves_all_training_state(monkeypatch):
    trainer = Trainer(small_config())
    agent = trainer.agent
    state = np.zeros(39, dtype=np.float32)
    for _ in range(2):
        agent.remember(state, 0, -2, state, True, np.zeros(6, dtype=bool))
    agent.optimize_step()
    weights = deepcopy([agent.policy_network.state_dict(), agent.target_network.state_dict()])
    optimizer = deepcopy(agent.optimizer.state_dict())
    gradients = [p.grad.clone() for p in agent.policy_network.parameters()]
    replay = deepcopy(list(agent.replay_buffer._transitions))
    rng = agent.rng
    rng_state = rng.getstate()
    replay_rng_state = agent.replay_buffer.rng.getstate()
    workload_rng_state = trainer.training_rng.getstate()
    updates = agent.optimizer_steps
    original_select = agent.select_action
    calls = []

    def greedy_only(observation, mask, epsilon):
        assert epsilon == 0
        assert not torch.is_grad_enabled()
        assert not agent.policy_network.training
        action = original_select(observation, mask, epsilon)
        assert mask[action]
        calls.append(action)
        return action

    monkeypatch.setattr(agent, "select_action", greedy_only)
    with torch.enable_grad():
        metrics = trainer.validator.evaluate(agent)
    assert calls
    assert metrics == trainer.validator.evaluate(agent)
    assert_tree_equal(weights, [agent.policy_network.state_dict(), agent.target_network.state_dict()])
    assert_tree_equal(optimizer, agent.optimizer.state_dict())
    assert_tree_equal(gradients, [p.grad for p in agent.policy_network.parameters()])
    assert_tree_equal(replay, list(agent.replay_buffer._transitions))
    assert agent.rng is rng and rng.getstate() == rng_state
    assert agent.replay_buffer.rng.getstate() == replay_rng_state
    assert trainer.training_rng.getstate() == workload_rng_state
    assert agent.optimizer_steps == updates
    assert trainer.environment_steps == 0
    assert trainer.config.epsilon_at(trainer.environment_steps) == 1.0
    assert agent.policy_network.training


def test_validation_restores_rng_and_mode_after_exception(monkeypatch):
    agent = DQNAgent(seed=1)
    agent.policy_network.eval()
    rng = agent.rng

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(agent, "select_action", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        Validator(1).evaluate(agent)
    assert agent.rng is rng
    assert not agent.policy_network.training


def test_validation_means_are_weighted_by_process_count():
    validator = Validator(1)
    validator.workloads = ((("A", 0, 1),), (("B", 0, 1), ("C", 0, 1)))
    result = validator.evaluate(DQNAgent(seed=0))
    assert result["process_count"] == 3
    assert result["mean_waiting_time"] == pytest.approx(1 / 3)
    assert result["mean_turnaround_time"] == pytest.approx(4 / 3)
    assert result["mean_reward"] == -0.5


def test_warmup_frequency_masks_and_target_sync(monkeypatch, tmp_path, fast_plots):
    workload = tuple((f"P{i}", 0, 2) for i in range(4))
    monkeypatch.setattr("src.training.train.sample_workload", lambda rng: workload)
    trainer = Trainer(small_config())
    original_optimize = trainer.agent.optimize_step
    original_sync = trainer.agent.sync_target_network
    original_select = trainer.agent.select_action
    optimize_at, sync_at, epsilons = [], [], []

    def optimize():
        optimize_at.append(trainer.environment_steps)
        assert len(trainer.agent.replay_buffer) >= 5
        return original_optimize()

    def sync():
        sync_at.append(trainer.agent.optimizer_steps)
        original_sync()

    def select(observation, mask, epsilon):
        action = original_select(observation, mask, epsilon)
        assert mask[action]
        if epsilon > 0:
            epsilons.append(epsilon)
        return action

    monkeypatch.setattr(trainer.agent, "optimize_step", optimize)
    monkeypatch.setattr(trainer.agent, "sync_target_network", sync)
    monkeypatch.setattr(trainer.agent, "select_action", select)
    summary = trainer.run(tmp_path, verbose=False)
    assert optimize_at == [6, 9, 12, 15, 18, 21, 24]
    assert sync_at == [2, 4, 6]  # Only DQNAgent performs synchronization.
    assert summary["environment_steps"] == 24 and summary["optimizer_updates"] == 7
    assert epsilons == [trainer.config.epsilon_at(step) for step in range(24)]
    assert [row["episode"] for row in trainer.validation_metrics] == [0, 2, 3]
    batch = trainer.agent.replay_buffer.sample(24)
    assert np.all(batch["next_action_masks"][batch["dones"]] == False)
    assert np.all(batch["next_action_masks"][~batch["dones"]].any(axis=1))
    for row in trainer.training_metrics:
        assert row["reward"] == -row["mean_waiting_time"] * row["process_count"]
        assert row["mean_turnaround_time"] - row["mean_waiting_time"] == 2
        assert all(np.isfinite(value) for value in row.values() if value != "")


def test_replay_batch_requirement_also_gates_optimization(monkeypatch, tmp_path, fast_plots):
    monkeypatch.setattr("src.training.train.sample_workload", lambda rng: (("P", 0, 3),))
    trainer = Trainer(small_config(episodes=1, replay_warmup=0,
                                  optimization_interval=1, agent_options={"batch_size": 4}))
    trainer.run(tmp_path, verbose=False)
    assert len(trainer.agent.replay_buffer) == 3
    assert trainer.agent.optimizer_steps == 0
    assert trainer.training_metrics[0]["mean_loss"] == ""


def test_policy_input_hides_future_processes_in_training(monkeypatch, tmp_path, fast_plots):
    workload = (("future", 8, 10), ("ready", 0, 10), ("later", 7, 3), ("last", 6, 4))
    monkeypatch.setattr("src.training.train.sample_workload", lambda rng: workload)
    trainer = Trainer(small_config(episodes=1))
    original = trainer.agent.select_action
    calls = []

    def inspect(observation, action_mask, epsilon):
        if epsilon > 0 and not calls:
            # Only the arrived slot is populated. Neither future bursts nor
            # the total process count reaches policy selection.
            assert observation.shape == (39,)
            for index in (0, 2, 3, 4, 5):
                np.testing.assert_array_equal(observation[index * 6:index * 6 + 6], np.zeros(6))
            assert observation[37] == pytest.approx(1 / 6)
            np.testing.assert_array_equal(action_mask, [False, True, False, False, False, False])
            calls.append(True)
        return original(observation, action_mask, epsilon)

    monkeypatch.setattr(trainer.agent, "select_action", inspect)
    trainer.run(tmp_path, verbose=False)
    assert calls


def test_training_is_reproducible_and_independent_of_validation_frequency(tmp_path, fast_plots):
    first = Trainer(small_config(validation_interval=1))
    second = Trainer(small_config(validation_interval=3))
    torch_rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    first.run(tmp_path / "first", verbose=False)
    second.run(tmp_path / "second", verbose=False)
    assert first.training_metrics == second.training_metrics
    assert_tree_equal(first.agent.policy_network.state_dict(), second.agent.policy_network.state_dict())
    assert_tree_equal(list(first.agent.replay_buffer._transitions), list(second.agent.replay_buffer._transitions))
    assert first.training_rng.getstate() == second.training_rng.getstate()
    assert first.agent.rng.getstate() == second.agent.rng.getstate()
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_best_checkpoint_contains_best_validation_weights(tmp_path, monkeypatch, fast_plots):
    trainer = Trainer(small_config(episodes=2, validation_interval=1))
    scores = iter([5.0, 2.0, 4.0])
    evaluated_weights = []

    def evaluate(agent):
        evaluated_weights.append(deepcopy(agent.policy_network.state_dict()))
        waiting = next(scores)
        return {"mean_waiting_time": waiting, "mean_turnaround_time": waiting + 3,
                "mean_reward": -waiting * 4, "workload_count": 1, "process_count": 4}

    monkeypatch.setattr(trainer.validator, "evaluate", evaluate)
    trainer.run(tmp_path, verbose=False)
    restored = DQNAgent()
    restored.load(tmp_path / "best_model.pt")
    assert_tree_equal(restored.policy_network.state_dict(), evaluated_weights[1])
    assert len(restored.replay_buffer) == 0
    restored.load(tmp_path / "final_model.pt")
    assert_tree_equal(restored.policy_network.state_dict(), evaluated_weights[2])
    metadata = json.loads((tmp_path / "best_model.json").read_text())
    assert metadata["episode"] == 1
    assert metadata["validation"]["mean_waiting_time"] == 2
    assert metadata["environment_steps"] == metadata["epsilon_schedule_position"]
    assert metadata["epsilon"] == trainer.config.epsilon_at(metadata["environment_steps"])
    assert metadata["optimizer_updates"] > 0
    assert metadata["seed"] == trainer.config.seed
    assert not metadata["exact_resume_supported"]
    assert not metadata["replay_experiences_saved"]
    with (tmp_path / "validation_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert min(float(row["mean_waiting_time"]) for row in rows) == 2
    assert [row["is_best"] for row in rows] == ["True", "True", "False"]
    with (tmp_path / "training_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    config = json.loads((tmp_path / "training_config.json").read_text())
    assert config["agent_config"] == trainer.agent.config


def test_completed_metrics_and_reward_invariant():
    adapter = RLAdapter()
    adapter.reset((("A", 0, 2), ("B", 0, 1)))
    with pytest.raises(ValueError):
        episode_metrics(adapter, 0, 0)
    reward = sum(adapter.step(action)[1] for action in [0, 0, 1])
    assert reward == -2
    metrics = episode_metrics(adapter, reward, 3)
    assert metrics["mean_waiting_time"] == 1
    assert metrics["mean_turnaround_time"] == 2.5
    with pytest.raises(AssertionError):
        episode_metrics(adapter, reward + 1, 3)


@pytest.mark.parametrize("episodes,warmup", [(0, 1000), (1, 1000), (1, 0)])
def test_plots_handle_empty_missing_and_short_loss_data(tmp_path, episodes, warmup):
    trainer = Trainer(small_config(episodes=episodes, replay_warmup=warmup))
    trainer.run(tmp_path, verbose=False)
    for filename in ("training_reward.png", "training_waiting_time.png", "validation_waiting_time.png",
                     "dqn_loss.png", "epsilon_decay.png"):
        assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # The same plotting entry point can regenerate figures from saved logs.
    generate_plots(tmp_path, window=50)
    np.testing.assert_allclose(moving_average([1, 3, 8], window=2), [1, 2, 5.5])
    assert moving_average([]).size == 0
    assert moving_average([4], 50).tolist() == [4]


def test_output_artifacts_are_not_overwritten(tmp_path, fast_plots):
    trainer = Trainer(small_config(episodes=0))
    trainer.run(tmp_path, verbose=False)
    before = (tmp_path / "best_model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        Trainer(small_config()).run(tmp_path, verbose=False)
    assert (tmp_path / "best_model.pt").read_bytes() == before
    with pytest.raises(RuntimeError):
        trainer.run(tmp_path / "again", verbose=False)


def test_cli_smoke_and_explicit_episode_override(monkeypatch, tmp_path):
    calls = []

    def run(trainer, output_dir):
        calls.append((trainer.config, output_dir))
        return {"episodes": trainer.config.episodes}

    monkeypatch.setattr(Trainer, "run", run)
    main(["--smoke", "--seed", "12", "--output-dir", str(tmp_path)])
    assert calls[-1][0].episodes == 20 and calls[-1][0].seed == 12
    main(["--smoke", "--episodes", "5"])
    assert calls[-1][0].episodes == 5
    main([])
    assert calls[-1][0].episodes == 6000
