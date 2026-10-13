# Stage 6.3A executed results

The unchanged best checkpoint, `runs/phase5_full_6000/best_model.pt`, selected
at episode 5250, shows substantial sensitivity to slot layout. The aggregate
original-versus-permuted waiting difference is small and its confidence
interval includes zero; these results do not establish overall deterioration
under random rearrangements.

## Inputs and verification

- Exact Phase 5 validation collection: seed 1729, 128 workloads, 645 processes.
  SHA-256: `7387547e5cf6bdccfcf92732be65a798c0bf0c369050ad06d93869a973f5fe8e`.
- Checkpoint SHA-256: `70cda1270c41eb0137d10d7ac295e575e127af98b77fe575facd39c3b4dd83a5`.
- Eight arrangements per workload: original plus seven distinct permutations;
  permutation seed 6301729, per-workload seed = seed + 1000003 * workload ID.
- DQN epsilon 0; CPU; no gradients, training, optimization, or replay insertion.
- Both complete experiments repeated exactly, including all per-process metrics.
- Full optimizer/model/replay/RNG/gradient snapshots unchanged. Protected
  simulator, adapter, baseline, agent, training sources and prior results unchanged.
- Full test suite: **175 passed, 1 skipped** in 8.04 seconds. The skip is the
  existing CUDA-availability test. This includes 24 new permutation tests.
- No reserved final workloads accessed. No augmentation implemented.

## Fixed-state experiment

Up to eight evenly spaced multi-ready states per original rollout/source were
sampled, with a single-ready fallback if necessary. The two sources are the
original DQN and original SRTF trajectories. Complete six-feature blocks,
including the last-executed flag, move with their masks; globals remain fixed.

There are 1,969 sampled original states and 13,783 comparisons against the
seven alternatives. Identity agreement is **68.98%**, with workload-bootstrap
95% CI **[66.99%, 70.95%]**. Disagreement is **31.02%**. DQN-source agreement is
69.70% [67.49%, 71.91%]; SRTF-source agreement is 68.25% [66.07%, 70.34%].
These are DQN choices on both sources' observations, not SRTF action agreement.

Among 4,276 disagreements, 3,747 have both top-two Q margins above 0.05;
529 have at least one margin <= 0.05, and none are tied within 1e-6.
Thus 87.63% of disagreements have clear preferences under the stated heuristic.
The cutoffs are absolute Q/reward units, not statistical equivalence criteria.

| Destination slot (zero-based) | Original chosen identity retained | Uniquely shortest identity selected |
| --- | ---: | ---: |
| 0 | 87.22% | 97.04% |
| 1 | 81.87% | 92.79% |
| 2 | 76.04% | 88.28% |
| 3 | 34.65% | 37.00% |
| 4 | 75.20% | 87.42% |
| 5 | 46.58% | 57.05% |

Slot agreement conditions on the destination of the original chosen identity.
Shortest-process selection conditions on the destination of that unique
shortest identity. These columns have different denominators. Slot 5 only
exists for six-process workloads; slot 4 exists for five and six processes.

Matched states with the same unique-shortest identity in target versus other
slots show selection-rate differences of **-52.85 percentage points** for
slot 3 [95% CI -55.75, -50.02; 1,487 matched states] and **-26.81 points** for
slot 5 [-32.22, -21.50; 467 states]. Other slots may also rearrange, so these
interventions demonstrate layout sensitivity rather than a fully isolated
single-slot mechanism. Only states with both target and other placements
among the seven sampled alternatives contribute.

## Complete trajectory experiment

There are 1,024 workload arrangements and 2,048 DQN/SRTF rollouts, with 10,320
per-process result rows. Original arrangement scores exactly reproduce Stage 6.2
and the DQN checkpoint validation score.

| Policy / arrangements | Waiting | Turnaround | Response | Switches/workload | Preemptions/workload | Reward/workload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN original | 6.586047 | 12.000000 | 5.055814 | 5.101563 | 1.156250 | -33.187500 |
| DQN seven alternatives pooled | 6.602436 | 12.016390 | 5.151274 | 5.087054 | 1.141741 | -33.270089 |
| SRTF original | 6.006202 | 11.420155 | 5.314729 | 4.601563 | 0.656250 | -30.265625 |
| SRTF seven alternatives pooled | 6.006202 | 11.420155 | 5.314729 | 4.601563 | 0.656250 | -30.265625 |

Timing means are process weighted. Pooling seven alternatives repeats each
workload seven times; reward totals therefore differ by sample count:
DQN original -4,248, alternatives -29,810; SRTF original -3,874,
alternatives -27,118. Every rollout satisfies reward = negative total waiting.

DQN paired waiting difference (alternatives minus original) is **+0.016390**
CPU time units per process, 95% workload-bootstrap CI **[-0.117396, +0.137551]**.
Across alternatives, 320 improve on their own original, 356 worsen, and 220 tie.
Waiting changes on **125/128 workloads (97.66%; CI 94.53%-100%)**. Mean
within-workload range is **1.373568** [1.232943, 1.522799]; mean population
standard deviation over eight arrangements is **0.473396**. Maximum range is
**4.5** on workload 81: original/worst 20.333333, best 15.833333, SRTF 15.666667.

| Processes | Workloads | Fixed-state agreement | DQN original waiting | DQN alternative waiting | SRTF waiting | Paired delta [95% CI] | Changed workloads | Mean range |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 41 | 72.76% | 4.231707 | 4.357143 | 3.969512 | +0.125436 [-0.017422, +0.256098] | 92.68% | 0.957317 |
| 5 | 41 | 71.54% | 5.843902 | 5.893380 | 5.439024 | +0.049477 [-0.102456, +0.199321] | 100% | 1.278049 |
| 6 | 46 | 63.70% | 8.536232 | 8.463251 | 7.637681 | -0.072981 [-0.358178, +0.186348] | 100% | 1.829710 |

Mean waiting by arrangement draw 0-7 is respectively:
6.586047, 6.612403, 6.607752, 6.567442, 6.572093, 6.551938, 6.651163,
6.654264. Each workload has its own permutation at each draw number;
`action_agreement_by_permutation.csv` instead groups actual slot maps.

SRTF total waiting and reward remain invariant on every arrangement of every
workload. Its workload mean turnaround, response, switches, and preemptions
also remain invariant in this collection. Tie breaking can still reassign
per-process timing: **18 SRTF processes across 9 workloads** have different
identity-level waiting times under permutations. Thus aggregate invariance
does not imply identity-level scheduling invariance.

## Slot exposure during full trajectories

Within-workload centered association slopes between mean waiting and the
fraction of unique-shortest opportunities located in slot 3 or 5 are positive:
slot 3 **1.7175** [1.4106, 2.0667], slot 5 **0.7941** [0.1331, 1.6344], in
waiting units per unit exposure fraction. A 0.1 higher fraction corresponds
to slopes of 0.1718 and 0.0794 waiting units respectively.

These are observational associations. Delaying a process can itself increase
its observed exposure, and subsequent trajectories depend on earlier choices.
They do not establish the causal waiting-time cost of a particular slot.
The fixed-state comparisons provide the direct evidence of action sensitivity.

## Interpretation and limits

Evidence supports substantial DQN slot sensitivity, especially in slots 3 and
5, rather than Q ties alone. Effects on individual workloads are large despite
nearly cancelling effects on the aggregate mean. There is no evidence here
that random permutations uniformly worsen performance, or that DQN outperforms
SRTF on waiting time: SRTF waiting is <= DQN on all 1,024 arrangements.

Permutation augmentation appears worth testing in a separately authorized
training experiment, preferably with a predeclared evaluation protocol. Its
benefit is a hypothesis, not demonstrated by this analysis. No augmentation,
training, hyperparameter changes, or architecture changes were made.

All CIs use 5,000 paired percentile bootstrap draws, seed 63021729, resampling
workloads and retaining their dependent states/arrangements. CIs are descriptive
and unadjusted for multiple comparisons. This is one checkpoint, sampled layouts,
representative states, and a model-selection validation collection; conclusions
are not an unbiased final-test estimate or proof of convergence/generalization.

Figures: [fixed-state slots](action_agreement_by_slot.png),
[waiting variation](waiting_time_variation.png),
[DQN/SRTF consistency](dqn_srtf_permutation_consistency.png).
