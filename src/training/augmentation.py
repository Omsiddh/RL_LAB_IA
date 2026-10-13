"""Replay-only coordinate transforms; live simulator actions stay unchanged."""

from src.permutation.transforms import inverse_permutation, permute_observation


def permute_transition(state, action, reward, next_state, done, next_action_mask, permutation):
    """Use one new-slot -> old-slot map for both endpoints and the action.

    No workload attributes are accepted. The occupied count/map is metadata for
    moving zero future blocks, not an additional network input. Unused slots
    remain fixed. The returned tuple replaces one transition, never adds one.
    """
    inverse = inverse_permutation(permutation)
    if type(action) is not int or not 0 <= action < len(inverse):
        raise ValueError("Action must index an occupied workload slot")
    moved_state, _ = permute_observation(state, next_action_mask, permutation)
    moved_next, moved_mask = permute_observation(next_state, next_action_mask, permutation)
    return moved_state, inverse[action], reward, moved_next, done, moved_mask


def augment_transition(state, action, reward, next_state, done, next_action_mask, process_count, rng):
    if type(process_count) is not int or not 1 <= process_count <= 6:
        raise ValueError("process_count must be between one and six")
    permutation = tuple(rng.sample(range(process_count), process_count))
    return permute_transition(state, action, reward, next_state, done, next_action_mask, permutation)
