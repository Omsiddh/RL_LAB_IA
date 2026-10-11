"""Training controls; optimizer defaults remain owned by DQNAgent."""

from dataclasses import asdict, dataclass, field
import math
from numbers import Real


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 6000
    initial_epsilon: float = 1.0
    minimum_epsilon: float = 0.05
    epsilon_decay_steps: int = 100000
    replay_warmup: int = 1000
    optimization_interval: int = 4
    validation_interval: int = 250
    validation_workloads: int = 128
    seed: int = 42
    validation_seed: int = 1729
    device: str = "cpu"
    torch_threads: int = 1
    # Only overrides are passed to DQNAgent; no duplicate optimizer defaults.
    agent_options: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("episodes", "replay_warmup"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        for name in ("epsilon_decay_steps", "optimization_interval",
                     "validation_interval", "validation_workloads", "torch_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("initial_epsilon", "minimum_epsilon"):
            value = getattr(self, name)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(name + " must be finite and between 0 and 1")
        if self.minimum_epsilon > self.initial_epsilon:
            raise ValueError("minimum_epsilon must not exceed initial_epsilon")
        for name in ("seed", "validation_seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + " must be an integer")
        if not isinstance(self.agent_options, dict):
            raise ValueError("agent_options must be a dictionary of DQN overrides")
        if {"seed", "device"} & self.agent_options.keys():
            raise ValueError("Set seed and device on TrainingConfig")
        object.__setattr__(self, "agent_options", dict(self.agent_options))

    def epsilon_at(self, environment_steps):
        """Epsilon for the next action after this many completed steps."""
        if type(environment_steps) is not int or environment_steps < 0:
            raise ValueError("environment_steps must be a nonnegative integer")
        if environment_steps >= self.epsilon_decay_steps:
            return self.minimum_epsilon
        fraction = environment_steps / self.epsilon_decay_steps
        return max(self.minimum_epsilon, self.initial_epsilon
                   - fraction * (self.initial_epsilon - self.minimum_epsilon))

    def to_dict(self):
        return asdict(self)
