"""Audit the existing workload RNG stream without changing its sampler."""

from collections import Counter
import random

from src.training.validation import sample_workload
from .experiment import tree_digest


def audit_workloads(episodes=10000, seed=42):
    rng = random.Random(seed)
    counts = Counter()
    placement = Counter()
    attributes = {slot: [0, 0, 0] for slot in range(6)}
    definitions = []
    for _ in range(episodes):
        workload = sample_workload(rng)
        definitions.append(workload)
        counts[len(workload)] += 1
        for slot, (pid, arrival, burst) in enumerate(workload):
            placement[len(workload), slot, pid] += 1
            values = attributes[slot]
            values[0] += 1
            values[1] += arrival
            values[2] += burst
    return {
        "episodes": episodes, "training_seed": seed,
        "existing_randomization": "sample_workload draws a 64-bit seed, calls generate_workload with a local Random, then shuffles the resulting definitions once with the training RNG",
        "generator": "PID order P1..Pn before shuffle; count uniform 4..6; arrival iid uniform 0..8; burst iid uniform 1..10",
        "interpretation": "PID-to-slot assignment is uniform conditional on n in the sampler. Finite counts fluctuate. Slots 0..3 always occupied, slot 4 only for n>=5, slot 5 only for n=6. Ready-state/replay exposure also depends on arrivals and policy; it need not be uniform.",
        "decision": "Do not add a workload shuffle. Test independent replay-coordinate augmentation, replacing each stored transition by one permuted version.",
        "process_count_workloads": dict(sorted(counts.items())),
        "workload_sequence_digest": tree_digest(tuple(definitions)),
        "expected_environment_steps": sum(burst for w in definitions for _, _, burst in w),
        "slot_attributes": [{"slot": slot, "occupied_workloads": value[0],
            "mean_arrival": value[1] / value[0], "mean_burst": value[2] / value[0]}
            for slot, value in attributes.items()],
        "pid_placements": [{"process_count": n, "slot": slot, "pid": pid, "observed": value,
            "expected": counts[n] / n} for (n, slot, pid), value in sorted(placement.items())],
    }
