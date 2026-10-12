"""Explicit new-slot-to-original-slot maps and lossless observation transforms."""

import math
import random

import numpy as np


def validate_permutation(permutation):
    permutation = tuple(permutation)
    if not 1 <= len(permutation) <= 6 or set(permutation) != set(range(len(permutation))):
        raise ValueError("Permutation must contain each occupied index exactly once (one to six slots)")
    if any(type(index) is not int for index in permutation):
        raise ValueError("Permutation indices must be Python integers")
    return permutation


def inverse_permutation(permutation):
    permutation = validate_permutation(permutation)
    inverse = [0] * len(permutation)
    for new_slot, original_slot in enumerate(permutation):
        inverse[original_slot] = new_slot
    return tuple(inverse)


def make_permutations(process_count, seed, arrangements=8):
    if type(process_count) is not int or not 1 <= process_count <= 6:
        raise ValueError("process_count must be between one and six")
    if type(arrangements) is not int or not 1 <= arrangements <= math.factorial(process_count):
        raise ValueError("Cannot produce the requested number of distinct arrangements")
    rng = random.Random(seed)
    permutations = [tuple(range(process_count))]
    seen = set(permutations)
    while len(permutations) < arrangements:
        permutation = tuple(rng.sample(range(process_count), process_count))
        if permutation not in seen:
            seen.add(permutation)
            permutations.append(permutation)
    return tuple(permutations)


def permute_observation(observation, mask, permutation):
    """Move occupied six-feature blocks together, leaving globals/unused slots.

    permutation[new_slot] gives the original slot. Future processes remain
    indistinguishable from unused slots in the observation: their zero blocks
    and false mask entries move without inserting workload metadata.
    """
    permutation = validate_permutation(permutation)
    observation = np.asarray(observation)
    mask = np.asarray(mask)
    if observation.shape != (39,) or mask.shape != (6,) or mask.dtype != np.bool_:
        raise ValueError("Expected a 39-feature observation and six-entry Boolean mask")
    transformed, transformed_mask = observation.copy(), mask.copy()
    blocks = observation[:36].reshape(6, 6)
    transformed[:6 * len(permutation)] = blocks[list(permutation)].reshape(-1)
    transformed_mask[:len(permutation)] = mask[list(permutation)]
    return transformed, transformed_mask


def permute_workload(workload, permutation):
    permutation = validate_permutation(permutation)
    if len(workload) != len(permutation):
        raise ValueError("Workload and permutation sizes must match")
    return tuple(workload[index] for index in permutation)
