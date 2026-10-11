"""Masked DQN action selection and individual optimizer updates."""

import math
from numbers import Real
import random

import torch
from torch import nn

from .network import QNetwork
from .replay_buffer import ReplayBuffer, copy_mask, copy_observation


class DQNAgent:
    def __init__(self, learning_rate=0.0005, gamma=1.0, batch_size=64, replay_capacity=20000, target_sync_interval=250, max_gradient_norm=10.0, device="cpu", seed=None):
        
        for name, value in [("learning_rate", learning_rate),
                            ("max_gradient_norm", max_gradient_norm)]:
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be a finite positive number") 
        if not isinstance(gamma, Real) or isinstance(gamma, bool) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        for name, value in [("batch_size", batch_size), ("replay_capacity", replay_capacity),
                            ("target_sync_interval", target_sync_interval)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if batch_size > replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.config = {
            "learning_rate": learning_rate, "gamma": gamma, "batch_size": batch_size,
            "replay_capacity": replay_capacity, "target_sync_interval": target_sync_interval,
            "max_gradient_norm": max_gradient_norm, "seed": seed,
        }
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.max_gradient_norm = max_gradient_norm
        self.rng = random.Random(seed)
        self.replay_buffer = ReplayBuffer(replay_capacity, seed)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            self.policy_network = QNetwork().to(self.device)
            self.target_network = QNetwork().to(self.device)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.sync_target_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer_steps = 0

    def select_action(self, observation, action_mask, epsilon=0.0):
        observation = copy_observation(observation)
        mask = copy_mask(action_mask)
        if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        valid = []
        for index in range(6):
            if mask[index]:
                valid.append(index)
        if not valid:
            raise ValueError("Cannot select an action when no valid actions exist")
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        with torch.no_grad():
            q_values = self.policy_network(observation)
            mask_tensor = torch.as_tensor(mask, device=self.device)
            return q_values.masked_fill(~mask_tensor, -torch.inf).argmax().item()

    def remember(self, state, action, reward, next_state, done, next_action_mask):
        self.replay_buffer.add(state, action, reward, next_state, done, next_action_mask)

    def _compute_targets(self, batch):
        with torch.no_grad():
            rewards = torch.as_tensor(batch["rewards"], device=self.device)
            dones = torch.as_tensor(batch["dones"], device=self.device)
            future = torch.zeros_like(rewards)
            nonterminal = ~dones
            if nonterminal.any():
                states = torch.as_tensor(batch["next_states"], device=self.device)
                masks = torch.as_tensor(batch["next_action_masks"], device=self.device)
                q_values = self.target_network(states[nonterminal])
                future[nonterminal] = q_values.masked_fill(
                    ~masks[nonterminal], -torch.inf
                ).max(dim=1).values
            return rewards + self.gamma * future

    def optimize_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        predicted = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self._compute_targets(batch)
        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.max_gradient_norm
        )
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.optimizer_steps % self.target_sync_interval == 0:
            self.sync_target_network()
        return {"loss": loss.item(), "gradient_norm": gradient_norm.item(),
                "optimizer_steps": self.optimizer_steps}

    def sync_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, path):
        torch.save({
            "config": self.config,
            "policy": self.policy_network.state_dict(),
            "target": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "action_rng": self.rng.getstate(),
            "replay_rng": self.replay_buffer.rng.getstate(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        restored = DQNAgent(device=self.device, **checkpoint["config"])
        restored.policy_network.load_state_dict(checkpoint["policy"])
        restored.target_network.load_state_dict(checkpoint["target"])
        restored.optimizer.load_state_dict(checkpoint["optimizer"])
        restored.optimizer_steps = checkpoint["optimizer_steps"]
        restored.rng.setstate(checkpoint["action_rng"])
        restored.replay_buffer.rng.setstate(checkpoint["replay_rng"])
        self.__dict__.update(restored.__dict__)
