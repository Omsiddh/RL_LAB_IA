# Stage 6.4 held-out evaluation

Augmentation reduced held-out waiting versus the equal-budget original in this fixed seed pair; the paired workload interval excludes zero.

This is a technical evaluation record. No checkpoints were trained, tuned, or reselected after test inspection; Stage 6.5 and Phase 7 were not implemented.

## Frozen inputs

No prior saved final collection or test seed was found. Dedicated seed 6401729 was fixed before generation: 256 workloads, 1277 processes, existing count 4..6 / arrival 0..8 / burst 1..10 distribution and training's single slot shuffle. All policies replay the exact same immutable collection.

Workload SHA-256: `9ec79309a0c84bd948f8dc3611bd3753aa2b1b24fc00cb26e75dbfb22c7a8f0b`. Counts: {'4': 90, '5': 79, '6': 87}. Validation digest: `7387547e5cf6bdccfcf92732be65a798c0bf0c369050ad06d93869a973f5fe8e`.

Cross-split duplicates: 0 exact workloads, 0 arrival/burst multisets ignoring identity/order. Within-test duplicates: 0 exact, 0 arrival/burst multisets. No redraw/filtering occurred.

[Frozen protocol](frozen_protocol.json), [checkpoint manifest](selected_model_manifest.json), [workload definitions](test_workloads.json), [workload metadata](test_workload_manifest.json). Both manifests were saved before test generation. Source/configuration and every preexisting experiment artifact were hashed and verified unchanged.

- Augmented DQN 10k: episode 9750, `D:\RL_LAB_IA\runs\phase6_3b_controlled\augmented_10000\best_model.pt`; SHA-256 `e124abab3d2f9350496f732fa97b5b2334374702b1103b8b713cacc6c781fb4f`.
- Original DQN 10k: episode 9000, `D:\RL_LAB_IA\runs\phase6_3b_controlled\original_10000\best_model.pt`; SHA-256 `c47c60a29ac97ad642dc69d2c44111a73382a5661abf9e1c6f79ff52c1ca2439`.
- Reference DQN 6k: episode 5250, `D:\RL_LAB_IA\runs\phase5_full_6000\best_model.pt`; SHA-256 `70cda1270c41eb0137d10d7ac295e575e127af98b77fe575facd39c3b4dd83a5`.

## Six-policy comparison

Times are process weighted; switches, preemptions, and mean reward are workload weighted. RR quantum = 2. Greedy DQN epsilon = 0 on CPU, with gradients disabled.

| Policy | Waiting | Turnaround | Response | Switches | Preemptions | Total reward | Mean reward | SRTF gap | Gap % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SRTF | 5.826155 | 11.255286 | 5.132341 | 4.621094 | 0.703125 | -7440 | -29.062500 | 0.000000 | 0.000000 |
| Augmented DQN 10k | 5.964761 | 11.393892 | 4.712608 | 4.968750 | 1.050781 | -7617 | -29.753906 | 0.138606 | 2.379032 |
| Original DQN 10k | 6.118246 | 11.547377 | 4.671887 | 5.082031 | 1.164062 | -7813 | -30.519531 | 0.292091 | 5.013441 |
| Reference DQN 6k | 6.517619 | 11.946750 | 4.881754 | 5.152344 | 1.234375 | -8323 | -32.511719 | 0.691464 | 11.868280 |
| FCFS | 8.429131 | 13.858262 | 8.429131 | 3.917969 | 0.000000 | -10764 | -42.046875 | 2.602976 | 44.677419 |
| Round Robin | 11.218481 | 16.647612 | 2.451840 | 13.046875 | 9.128906 | -14326 | -55.960938 | 5.392326 | 92.553763 |

Reward equals negative total waiting on every rollout. SRTF is no worse on waiting for every workload and every policy.

| DQN | Ties SRTF (workloads) | Waiting above SRTF (workloads) |
| --- | --- | --- |
| Augmented DQN 10k | 169 | 87 |
| Original DQN 10k | 133 | 123 |
| Reference DQN 6k | 85 | 171 |

## Paired comparisons

Augmented versus original workload outcomes: {'improves': 90, 'worsens': 36, 'ties': 130}. Negative timing differences favor augmentation; response/switch/preemption differences use the same sign convention.

| Left | Right | Metric | Difference | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| Augmented DQN 10k | Original DQN 10k | waiting_time | -0.153485 | -0.207905 | -0.102263 |
| Augmented DQN 10k | Original DQN 10k | response_time | 0.040720 | -0.149158 | 0.226080 |
| Augmented DQN 10k | Original DQN 10k | context_switches | -0.113281 | -0.273438 | 0.046875 |
| Augmented DQN 10k | Original DQN 10k | preemptions | -0.113281 | -0.273438 | 0.046875 |
| Augmented DQN 10k | SRTF | waiting_time | 0.138606 | 0.108919 | 0.173041 |
| Augmented DQN 10k | FCFS | waiting_time | -2.464370 | -2.695157 | -2.248218 |
| Augmented DQN 10k | Round Robin | waiting_time | -5.253720 | -5.601670 | -4.921065 |
| Augmented DQN 10k | Reference DQN 6k | waiting_time | -0.552858 | -0.653828 | -0.454315 |
| Original DQN 10k | SRTF | waiting_time | 0.292091 | 0.241515 | 0.347353 |
| Reference DQN 6k | SRTF | waiting_time | 0.691464 | 0.595861 | 0.793814 |

Intervals use 5,000 paired percentile workload resamples, seed 64021729. Every draw retains all six paired results; timing estimates recompute summed process times divided by summed process counts. These measure workload sampling uncertainty conditional on fixed trained checkpoints, not variation across independent training seeds. Secondary and subgroup intervals are unadjusted for multiplicity.

## Process-count groups

| Processes | Policy | Workloads | Waiting | Response | Switches | Preemptions | SRTF gap % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | SRTF | 90 | 4.027778 | 3.388889 | 3.544444 | 0.655556 | 0.000000 |
| 4 | Augmented DQN 10k | 90 | 4.075000 | 3.177778 | 3.633333 | 0.744444 | 1.172414 |
| 4 | Original DQN 10k | 90 | 4.163889 | 2.938889 | 3.944444 | 1.055556 | 3.379310 |
| 4 | Reference DQN 6k | 90 | 4.386111 | 2.991667 | 3.888889 | 1.000000 | 8.896552 |
| 4 | FCFS | 90 | 5.727778 | 5.727778 | 2.888889 | 0.000000 | 42.206897 |
| 4 | Round Robin | 90 | 7.744444 | 1.550000 | 9.733333 | 6.844444 | 92.275862 |
| 5 | SRTF | 79 | 5.825316 | 5.215190 | 4.569620 | 0.620253 | 0.000000 |
| 5 | Augmented DQN 10k | 79 | 5.896203 | 4.810127 | 4.924051 | 0.974684 | 1.216862 |
| 5 | Original DQN 10k | 79 | 6.050633 | 4.797468 | 5.088608 | 1.139241 | 3.867884 |
| 5 | Reference DQN 6k | 79 | 6.445570 | 4.881013 | 5.063291 | 1.113924 | 10.647545 |
| 5 | FCFS | 79 | 8.141772 | 8.141772 | 3.949367 | 0.000000 | 39.765319 |
| 5 | Round Robin | 79 | 11.200000 | 2.324051 | 13.518987 | 9.569620 | 92.264233 |
| 6 | SRTF | 87 | 7.067050 | 6.272031 | 5.781609 | 0.827586 | 0.000000 |
| 6 | Augmented DQN 10k | 87 | 7.319923 | 5.697318 | 6.390805 | 1.436782 | 3.578205 |
| 6 | Original DQN 10k | 87 | 7.517241 | 5.772031 | 6.252874 | 1.298851 | 6.370290 |
| 6 | Reference DQN 6k | 87 | 8.042146 | 6.185824 | 6.540230 | 1.586207 | 13.797777 |
| 6 | FCFS | 87 | 10.509579 | 10.509579 | 4.954023 | 0.000000 | 48.712388 |
| 6 | Round Robin | 87 | 13.628352 | 3.170498 | 16.045977 | 11.091954 | 92.843589 |

Group turnaround, rewards, totals, and SRTF tie counts are in [by_process_count.csv](by_process_count.csv); subgroup intervals are in [bootstrap_intervals.csv](bootstrap_intervals.csv).

## Selected examples

Workload IDs are zero based. Selection uses augmented minus original mean waiting; lowest ID breaks ties. Unit-step timelines for all six policies are stored in [selected_timelines.json](selected_timelines.json) for later Gantt rendering.

- Best: workload 69, definitions `(('P5', 6, 10), ('P4', 8, 2), ('P2', 6, 10), ('P1', 2, 10), ('P3', 7, 8))`, augmented minus original waiting = -2.000000; augmented 10.200000, original 12.200000, SRTF 10.200000.
- Worst: workload 201, definitions `(('P3', 2, 9), ('P4', 5, 7), ('P5', 0, 4), ('P6', 6, 6), ('P2', 5, 10), ('P1', 2, 5))`, augmented minus original waiting = 1.000000; augmented 11.166667, original 10.166667, SRTF 10.166667.

## Verification and limits

214 passed, 1 skipped in 14.60s. Command: .\.venv\Scripts\python.exe -m pytest -q. The skip is the existing CUDA-availability test. Includes 26 new synthetic-only final-evaluation tests.

The complete six-policy evaluation was repeated on the same frozen workloads, with exact per-process, per-workload, and timeline equality. Bootstrap outputs also reproduce exactly. Runtime assertions check arrivals, completion, process identity/index, CPU/idle accounting, reward/waiting identity, turnaround identity, action masks, SRTF inequalities, immutable definitions, and unchanged full model/optimizer/replay/RNG/gradient snapshots. Global Python/NumPy/Torch RNG states and prior source/artifact hashes remain unchanged. [Verification details](verification_report.json).

- One paired training seed; bootstrap measures workload sampling, not independent training-seed variability
- Selected checkpoints were fixed using model-selection validation, not this test collection
- Same workload distribution as training; no evidence for distribution shift or convergence
- Zero context-switch overhead; SRTF knows remaining bursts, while DQN receives normalized adapter observations
- Secondary and subgroup percentile intervals are descriptive and unadjusted for multiple comparisons
- The deterministic repeat is the same sample, not another experiment

Augmentation reduced held-out waiting versus the equal-budget original in this fixed seed pair; the paired workload interval excludes zero. This supports a held-out benefit on the training distribution for this checkpoint pair. It does not establish reliable improvement across training seeds or convergence.

Charts: [waiting](mean_waiting_time.png), [turnaround](mean_turnaround_time.png), [response](mean_response_time.png), [switches](mean_context_switches.png), [preemptions](mean_preemptions.png), [DQN optimality gaps](dqn_optimality_gaps.png), [process-count groups](by_process_count.png).
