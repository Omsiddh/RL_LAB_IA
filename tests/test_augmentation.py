"""Replay-coordinate augmentation and preservation of the disabled path."""

from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from src.env.rl_adapter import RLAdapter
from src.permutation.experiment import tree_digest
from src.permutation.transforms import inverse_permutation
from src.training.augmentation import augment_transition, permute_transition
from src.training.config import TrainingConfig
from src.training.train import Trainer


def transition():
    adapter = RLAdapter()
    state = adapter.reset((("A", 0, 5), ("future", 7, 8), ("B", 0, 3), ("late", 1, 1)))
    next_state, reward, done, info = adapter.step(2)
    return state, 2, reward, next_state, done, info["action_mask"]


def test_same_block_map_for_both_states_action_mask_globals_reward_and_flag():
    original = transition()
    before = tree_digest(original)
    permutation = (1, 3, 0, 2)
    moved = permute_transition(*original, permutation)
    assert moved[1] == 3 and moved[2] == original[2] and moved[4] is original[4]
    for new, old in enumerate(permutation):
        for endpoint in (0, 3):
            np.testing.assert_array_equal(moved[endpoint][new * 6:new * 6 + 6], original[endpoint][old * 6:old * 6 + 6])
        assert moved[5][new] == original[5][old]
    for endpoint in (0, 3):
        np.testing.assert_array_equal(moved[endpoint][24:], original[endpoint][24:])
    assert moved[3][3 * 6 + 5] == 1  # Last executed B moved from slot 2 to 3.
    assert not moved[0][:6].any() and not moved[3][:6].any() and not moved[5][0]
    assert not moved[0][6:12].any() and moved[3][6] == 1  # Arrival between endpoints.
    restored = permute_transition(*moved, inverse_permutation(permutation))
    assert tree_digest(restored) == before == tree_digest(original)


def test_terminal_all_false_mask_and_empty_slots_preserved():
    state, action, reward, next_state, _, _ = transition()
    terminal = (state, action, reward, next_state, True, np.zeros(6, dtype=bool))
    moved = permute_transition(*terminal, (3, 2, 0, 1))
    assert moved[4] is True and not moved[5].any()
    assert not moved[0][24:36].any() and not moved[3][24:36].any()


def test_separate_rng_is_deterministic_and_does_not_touch_other_streams():
    first, repeated, different = random.Random(630342), random.Random(630342), random.Random(630343)
    python_before, numpy_before = random.getstate(), tree_digest(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    original = transition()
    one = [tree_digest(augment_transition(*original, 4, first)) for _ in range(20)]
    two = [tree_digest(augment_transition(*original, 4, repeated)) for _ in range(20)]
    other = [tree_digest(augment_transition(*original, 4, different)) for _ in range(20)]
    assert one == two and one != other and len(set(one)) > 1
    assert random.getstate() == python_before and tree_digest(np.random.get_state()) == numpy_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


@pytest.mark.parametrize("options", [{"replay_augmentation": "shuffle"}, {"augmentation_seed": None}, {"augmentation_seed": True}])
def test_invalid_augmentation_config(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options)


def test_default_path_never_uses_augmentation_and_seed_is_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.train.generate_plots", lambda path: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled augmentation path called the transform")

    monkeypatch.setattr("src.training.train.augment_transition", forbidden)
    config = TrainingConfig(episodes=3, validation_workloads=2, validation_interval=3,
                            replay_warmup=5, agent_options={"batch_size": 2})
    original = Trainer(config)
    other = Trainer(replace(config, augmentation_seed=999))
    original.run(tmp_path / "original", verbose=False)
    other.run(tmp_path / "other", verbose=False)
    assert original.training_metrics == other.training_metrics
    assert original.validation_metrics == other.validation_metrics
    assert tree_digest(original.agent.optimizer.state_dict()) == tree_digest(other.agent.optimizer.state_dict())
    assert tree_digest(original.agent.policy_network.state_dict()) == tree_digest(other.agent.policy_network.state_dict())
    assert tree_digest(list(original.agent.replay_buffer._transitions)) == tree_digest(list(other.agent.replay_buffer._transitions))
    assert original.augmentation_rng is other.augmentation_rng is None


def test_replay_integration_uses_one_transform_per_live_step_and_same_workloads(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.train.generate_plots", lambda path: None)
    calls = []

    def fixed(state, action, reward, next_state, done, mask, count, rng):
        permutation = tuple(reversed(range(count)))
        result = permute_transition(state, action, reward, next_state, done, mask, permutation)
        calls.append(((state, action, reward, next_state, done, mask), result, count))
        return result

    monkeypatch.setattr("src.training.train.augment_transition", fixed)
    config = TrainingConfig(episodes=2, validation_workloads=2, validation_interval=2, replay_warmup=1000)
    original, augmented = Trainer(config), Trainer(replace(config, replay_augmentation="permutation"))
    original.run(tmp_path / "original", verbose=False)
    augmented.run(tmp_path / "augmented", verbose=False)
    # Before warmup there is no learning, so live trajectories/actions must match.
    assert original.training_metrics == augmented.training_metrics
    assert original.training_rng.getstate() == augmented.training_rng.getstate()
    assert original.agent.rng.getstate() == augmented.agent.rng.getstate()
    assert len(calls) == original.environment_steps == len(original.agent.replay_buffer) == len(augmented.agent.replay_buffer)
    for live, (observed, moved, count), replay in zip(original.agent.replay_buffer._transitions, calls, augmented.agent.replay_buffer._transitions):
        for a, b in zip(live, observed):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(moved, replay):
            np.testing.assert_array_equal(a, b)
        assert replay[0][replay[1] * 6] == 1  # Remapped action indexes a ready state block.
    assert augmented.agent.optimizer_steps == original.agent.optimizer_steps == 0


def test_augmented_training_repeatability_counts_and_validation_rng_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.train.generate_plots", lambda path: None)
    config = TrainingConfig(episodes=3, validation_workloads=2, validation_interval=1,
        replay_warmup=5, agent_options={"batch_size": 2}, replay_augmentation="permutation")
    first, second = Trainer(config), Trainer(replace(config, validation_interval=3))
    first.run(tmp_path / "first", verbose=False)
    second.run(tmp_path / "second", verbose=False)
    assert first.training_metrics == second.training_metrics
    assert first.augmentation_rng.getstate() == second.augmentation_rng.getstate()
    assert first.training_rng.getstate() == second.training_rng.getstate()
    assert tree_digest(first.agent.policy_network.state_dict()) == tree_digest(second.agent.policy_network.state_dict())
    expected = len([s for s in range(1, first.environment_steps + 1) if s >= 5 and s % 4 == 0])
    assert first.agent.optimizer_steps == second.agent.optimizer_steps == expected
    assert len(first.agent.replay_buffer) == first.environment_steps
    assert first.agent.config == Trainer(replace(config, replay_augmentation="none")).agent.config
