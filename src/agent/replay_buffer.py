from collections import deque
import math
from numbers import Real
import random

import numpy as np


def copy_observation(state):
    try:
        values = np.array(state, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("State must contain 39 numeric values") from error
    if values.shape != (39,) or not np.isfinite(values).all():
        raise ValueError("State must have shape (39,) and finite values")
    return values


def copy_mask(mask):
    values = np.asarray(mask)
    if values.shape != (6,) or values.dtype != np.bool_:
        raise ValueError("Action mask must contain exactly six Boolean values")
    return values.copy()


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity = capacity
        self._transitions = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self._transitions)

    def add(self, state, action, reward, next_state, done, next_action_mask):
        state = copy_observation(state)
        next_state = copy_observation(next_state)
        mask = copy_mask(next_action_mask)
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Action must be a Python integer from 0 to 5")
        if not isinstance(reward, Real) or isinstance(reward, bool) or not math.isfinite(reward):
            raise ValueError("Reward must be a finite number")
        if not np.isfinite(np.float32(reward)):
            raise ValueError("Reward must fit in float32")
        if type(done) not in (bool, np.bool_):
            raise ValueError("done must be Boolean")
        if not done and not mask.any():
            raise ValueError("A non-terminal next state must have a valid action")
        self._transitions.append((state, action, float(reward), next_state, bool(done), mask))

    def sample(self, batch_size):
        if type(batch_size) is not int or not 1 <= batch_size <= len(self):
            raise ValueError("Batch size must be positive and no larger than the buffer")
        transitions = self.rng.sample(list(self._transitions), batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*transitions)
        return {
            "states": np.stack(states),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.stack(next_states),
            "dones": np.array(dones, dtype=bool),
            "next_action_masks": np.stack(masks),
        }
