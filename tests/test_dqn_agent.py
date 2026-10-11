"""Controlled unit tests plus one untrained scheduling integration episode."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from src.agent import DQNAgent, QNetwork, ReplayBuffer
from src.env.rl_adapter import RLAdapter
from src.env.workload import generate_workload


def constant_q(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))


def add_terminal(agent, reward=-2):
    state = np.zeros(39, dtype=np.float32)
    agent.remember(state, 0, reward, state, True, np.zeros(6, dtype=bool))


def equal_weights(first, second):
    return all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_network_architecture_shapes_and_raw_outputs():
    network = QNetwork()
    linear = []
    for layer in network.layers:
        if isinstance(layer, nn.Linear):
            linear.append((layer.in_features, layer.out_features))
    assert linear == [(39, 64), (64, 64), (64, 6)]
    assert sum(isinstance(layer, nn.ReLU) for layer in network.layers) == 2
    assert network(np.zeros(39)).shape == (6,)
    assert network(torch.zeros(5, 39)).shape == (5, 6)
    constant_q(network, [-3, 2, 8, 0, -1, 4])
    torch.testing.assert_close(network(np.zeros(39)), torch.tensor([-3., 2., 8., 0., -1., 4.]))


def test_network_rejects_bad_observations():
    network = QNetwork()
    for observation in [np.zeros(38), np.zeros((2, 3, 39)), np.full(39, np.nan)]:
        with pytest.raises(ValueError):
            network(observation)


def test_exploitation_masks_invalid_high_q_and_handles_negative_q():
    agent = DQNAgent(seed=1)
    constant_q(agent.policy_network, [999, -9, 1000, -2, 900, 800])
    mask = np.array([False, True, False, True, False, False])
    for attempt in range(10):
        assert agent.select_action(np.zeros(39), mask, epsilon=0) == 3
    # Selection does not permanently change or mask the network output.
    assert agent.policy_network(np.zeros(39))[2].item() == 1000


def test_seeded_exploration_is_reproducible_and_uses_only_valid_actions():
    first = DQNAgent(seed=42)
    second = DQNAgent(seed=42)
    assert equal_weights(first.policy_network, second.policy_network)
    mask = np.array([False, True, False, True, False, True])
    a = []
    b = []
    for attempt in range(600):
        a.append(first.select_action(np.zeros(39), mask, epsilon=1))
        b.append(second.select_action(np.zeros(39), mask, epsilon=1))
    assert a == b
    assert set(a) == {1, 3, 5}
    # Broad deterministic sanity check, not a statistical performance claim.
    for action in [1, 3, 5]:
        assert 140 < a.count(action) < 260


def test_single_valid_action_and_empty_mask():
    agent = DQNAgent(seed=3)
    mask = np.array([False, False, True, False, False, False])
    for epsilon in [0, 0.5, 1]:
        assert agent.select_action(np.zeros(39), mask, epsilon) == 2
    with pytest.raises(ValueError, match="no valid actions"):
        agent.select_action(np.zeros(39), np.zeros(6, dtype=bool))


def test_action_selection_input_errors():
    agent = DQNAgent()
    for epsilon in [-0.1, 1.1, np.nan, True, "0"]:
        with pytest.raises(ValueError, match="epsilon"):
            agent.select_action(np.zeros(39), np.ones(6, dtype=bool), epsilon)
    for mask in [np.ones(5, dtype=bool), [1] * 6]:
        with pytest.raises(ValueError, match="mask"):
            agent.select_action(np.zeros(39), mask)


def test_replay_fifo_capacity_and_batch_dtypes():
    buffer = ReplayBuffer(capacity=3, seed=1)
    for number in range(5):
        state = np.full(39, number, dtype=np.float32)
        buffer.add(state, number % 6, -number, state, True, np.zeros(6, dtype=bool))
    assert len(buffer) == 3
    batch = buffer.sample(3)
    assert set(batch["states"][:, 0]) == {2, 3, 4}
    assert batch["states"].shape == batch["next_states"].shape == (3, 39)
    assert batch["states"].dtype == batch["rewards"].dtype == np.float32
    assert batch["actions"].dtype == np.int64
    assert batch["dones"].dtype == batch["next_action_masks"].dtype == np.bool_
    assert batch["next_action_masks"].shape == (3, 6)


def test_replay_copies_inputs_and_sampled_arrays():
    buffer = ReplayBuffer(2)
    state = np.zeros(39, dtype=np.float32)
    next_state = np.ones(39, dtype=np.float32)
    mask = np.array([True, False, False, False, False, False])
    buffer.add(state, 0, -1, next_state, False, mask)
    state[:] = 99
    next_state[:] = 99
    mask[:] = False
    batch = buffer.sample(1)
    np.testing.assert_array_equal(batch["states"], np.zeros((1, 39)))
    np.testing.assert_array_equal(batch["next_states"], np.ones((1, 39)))
    assert batch["next_action_masks"][0, 0]
    batch["states"][:] = 22
    batch["next_states"][:] = 22
    batch["next_action_masks"][:] = False
    assert buffer.sample(1)["states"][0, 0] == 0
    assert buffer.sample(1)["next_states"][0, 0] == 1
    assert buffer.sample(1)["next_action_masks"][0, 0]


def test_replay_seeded_sampling_without_replacement():
    first = ReplayBuffer(10, seed=4)
    second = ReplayBuffer(10, seed=4)
    for index in range(10):
        for buffer in [first, second]:
            state = np.full(39, index, dtype=np.float32)
            buffer.add(state, 0, -1, state, True, np.zeros(6, dtype=bool))
    for attempt in range(3):
        a = first.sample(5)
        b = second.sample(5)
        np.testing.assert_array_equal(a["states"], b["states"])
        assert len(set(a["states"][:, 0])) == 5


def test_replay_rejects_invalid_transitions_atomically():
    buffer = ReplayBuffer(2)
    valid = [np.zeros(39), 0, -1, np.zeros(39), False, np.ones(6, dtype=bool)]
    invalid = [(0, np.zeros(38)), (0, np.full(39, np.nan)), (1, 6), (1, True),
               (2, float("inf")), (2, "bad"), (3, np.zeros(40)), (4, 1),
               (5, np.ones(6)), (5, np.zeros(6, dtype=bool))]
    for position, value in invalid:
        transition = valid.copy()
        transition[position] = value
        with pytest.raises(ValueError):
            buffer.add(*transition)
        assert len(buffer) == 0
    for size in [0, 1, -1, True]:
        with pytest.raises(ValueError):
            buffer.sample(size)
    with pytest.raises(ValueError):
        ReplayBuffer(0)


def test_masked_bellman_targets_and_terminal_empty_masks():
    agent = DQNAgent(gamma=0.5, batch_size=3, seed=2)
    constant_q(agent.target_network, [1000, -4, 900, -2, 800, 700])
    state = np.zeros(39)
    agent.remember(state, 0, -3, state, False,
                   np.array([False, True, False, True, False, False]))
    agent.remember(state, 1, -7, state, True, np.zeros(6, dtype=bool))
    # done takes precedence even if a caller provides a nonempty terminal mask.
    agent.remember(state, 2, -5, state, True, np.ones(6, dtype=bool))
    batch = agent.replay_buffer.sample(3)
    targets = agent._compute_targets(batch)
    assert not targets.requires_grad and torch.isfinite(targets).all()
    for index, action in enumerate(batch["actions"]):
        assert targets[index].item() == {0: -4, 1: -7, 2: -5}[action]


def test_default_gamma_and_all_terminal_update():
    agent = DQNAgent(batch_size=1, seed=1)
    assert agent.gamma == 1.0
    constant_q(agent.policy_network, [0] * 6)
    constant_q(agent.target_network, [999] * 6)
    add_terminal(agent, -2)
    result = agent.optimize_step()
    assert result["loss"] == pytest.approx(1.5)  # Smooth L1 at error 2.
    assert result["optimizer_steps"] == 1


def test_policy_updates_target_isolation_and_gradient_clipping():
    agent = DQNAgent(batch_size=1, max_gradient_norm=0.05, seed=2)
    assert equal_weights(agent.policy_network, agent.target_network)
    assert not any(p.requires_grad for p in agent.target_network.parameters())
    assert not agent.target_network.training
    policy_before = deepcopy(agent.policy_network.state_dict())
    target_before = deepcopy(agent.target_network.state_dict())
    add_terminal(agent, -100)
    result = agent.optimize_step()
    assert np.isfinite(result["loss"])
    assert any(not torch.equal(policy_before[k], v) for k, v in agent.policy_network.state_dict().items())
    for key, value in agent.target_network.state_dict().items():
        assert torch.equal(target_before[key], value)
    assert all(p.grad is None for p in agent.target_network.parameters())
    norm = torch.sqrt(sum(p.grad.square().sum() for p in agent.policy_network.parameters()))
    assert norm.item() <= 0.05001
    policy_ids = {id(p) for p in agent.policy_network.parameters()}
    assert {id(p) for group in agent.optimizer.param_groups for p in group["params"]} == policy_ids


def test_warmup_does_not_count_and_target_sync_uses_optimizer_updates():
    agent = DQNAgent(batch_size=2, target_sync_interval=2, seed=1)
    assert agent.optimize_step() is None and agent.optimizer_steps == 0
    add_terminal(agent)
    assert agent.optimize_step() is None and agent.optimizer_steps == 0
    add_terminal(agent)
    assert agent.optimizer_steps == 0
    agent.optimize_step()
    assert agent.optimizer_steps == 1
    assert not equal_weights(agent.policy_network, agent.target_network)
    agent.optimize_step()
    assert agent.optimizer_steps == 2
    assert equal_weights(agent.policy_network, agent.target_network)
    agent.optimize_step()
    assert not equal_weights(agent.policy_network, agent.target_network)
    agent.sync_target_network()
    assert equal_weights(agent.policy_network, agent.target_network)


def test_seeded_updates_are_reproducible_and_do_not_reseed_global_torch():
    global_state = torch.random.get_rng_state().clone()
    first = DQNAgent(batch_size=1, seed=8)
    second = DQNAgent(batch_size=1, seed=8)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    for agent in [first, second]:
        add_terminal(agent)
    assert first.optimize_step() == second.optimize_step()
    assert equal_weights(first.policy_network, second.policy_network)


def test_save_load_restores_networks_optimizer_config_counter_and_rng(tmp_path):
    original = DQNAgent(batch_size=1, target_sync_interval=3, gamma=0.9, seed=12)
    add_terminal(original)
    original.optimize_step()
    path = tmp_path / "agent.pt"
    original.save(path)
    restored = DQNAgent(seed=999)
    restored.load(path)
    assert restored.config == original.config
    assert restored.optimizer_steps == 1
    assert len(restored.replay_buffer) == 0
    assert equal_weights(original.policy_network, restored.policy_network)
    assert equal_weights(original.target_network, restored.target_network)
    assert not any(p.requires_grad for p in restored.target_network.parameters())
    assert restored.optimizer.state_dict()["param_groups"] == original.optimizer.state_dict()["param_groups"]
    mask = np.ones(6, dtype=bool)
    for attempt in range(10):
        assert original.select_action(np.zeros(39), mask, 1) == restored.select_action(np.zeros(39), mask, 1)
    assert original.select_action(np.zeros(39), mask, 0) == restored.select_action(np.zeros(39), mask, 0)
    # Equal subsequent updates prove Adam moments were restored, not just weights.
    add_terminal(restored)
    original.optimize_step()
    restored.optimize_step()
    assert equal_weights(original.policy_network, restored.policy_network)


def test_invalid_configuration_and_unavailable_cuda():
    for config in [{"learning_rate": 0}, {"gamma": 1.1}, {"gamma": float("nan")},
                   {"batch_size": 0}, {"replay_capacity": 1}, {"target_sync_interval": True},
                   {"max_gradient_norm": -1}, {"seed": "bad"}, {"device": "meta"}]:
        with pytest.raises(ValueError):
            DQNAgent(**config)
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="unavailable"):
            DQNAgent(device="cuda")


def test_cuda_update_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")
    agent = DQNAgent(device="cuda", batch_size=1, seed=1)
    add_terminal(agent)
    assert agent.optimize_step()["optimizer_steps"] == 1
    assert next(agent.policy_network.parameters()).is_cuda


def test_untrained_agent_completes_generated_episode_using_valid_actions():
    adapter = RLAdapter()
    workload = generate_workload(seed=19)
    observation = adapter.reset(workload)
    agent = DQNAgent(seed=19)
    reward_sum = 0
    steps = 0
    done = False
    while not done:
        mask = adapter.valid_action_mask()
        action = agent.select_action(observation, mask, epsilon=0)
        assert mask[action]
        next_observation, reward, done, info = adapter.step(action)
        agent.remember(observation, action, reward, next_observation, done, info["action_mask"])
        observation = next_observation
        reward_sum += reward
        steps += 1
        assert steps <= 60
    assert steps == sum(entry[2] for entry in workload)
    assert len(agent.replay_buffer) == steps
    assert reward_sum == -sum(p.get_waiting_time() for p in adapter.env.processes)
    assert not adapter.valid_action_mask().any()
    assert agent.optimizer_steps == 0  # Integration only; no learning occurred.
