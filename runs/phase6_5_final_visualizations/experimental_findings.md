# Stage 6.5 experimental findings

This is a technical findings and figure companion, not the final academic report. It presents existing experiments only. No scheduling policies, held-out evaluation, workload generation, checkpoint loading, training, tuning, or model reselection occurred during reporting.

## 1. Experimental objective

Compare selected deep Q-learning schedulers with classical CPU scheduling baselines on waiting time, quantify replay-permutation augmentation at equal training budget, and describe permutation sensitivity and timing tradeoffs. The declared primary held-out contrast is augmented 10k minus original 10k; comparisons with SRTF, FCFS, RR, and the historical 6k model are secondary.

## 2. Simulator and workload assumptions

The simulator models one CPU, integer CPU-time steps, perfect knowledge of arrived processes' burst lengths, preemption, no I/O, and zero context-switch overhead. Future-job properties are hidden from the policy until arrival. Workloads have 4–6 processes, arrivals uniformly sampled in 0–8, and bursts in 1–10. The existing sampler performs one seeded slot shuffle per workload. Stable indices and process identities are preserved.

Final-test seed **6401729** was fixed before test generation because no earlier test seed or collection was recorded. The saved collection has **256 workloads / 1277 processes**. Workload counts for 4/5/6 processes are respectively 90/79/87.

Saved test SHA-256: `9ec79309a0c84bd948f8dc3611bd3753aa2b1b24fc00cb26e75dbfb22c7a8f0b`. Exact cross-split duplicates: 0; arrival/burst-multiset cross-split duplicates: 0. No duplicate filtering or redrawing occurred. [Exact definitions](../phase6_4_final_test/test_workloads.json), [frozen protocol](../phase6_4_final_test/frozen_protocol.json).

## 3. Models and classical scheduling baselines

FCFS is non-preemptive, SRTF selects shortest remaining arrived work, and Round Robin uses FIFO with quantum 2. All use the same tested simulator. Under these zero-overhead assumptions SRTF minimizes mean waiting (equivalently total completion/turnaround) for this single-core preemptive objective; DQN is not expected to improve on that oracle policy for waiting.

The following checkpoints were selected on original-layout validation before held-out testing and were not changed in Stage 6.5:

| Role | Model | Selected episode | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| primary | Augmented DQN 10k | 9750 | `e124abab3d2f9350496f732fa97b5b2334374702b1103b8b713cacc6c781fb4f` |
| equal-budget control | Original DQN 10k | 9000 | `c47c60a29ac97ad642dc69d2c44111a73382a5661abf9e1c6f79ff52c1ca2439` |
| historical reference | Reference DQN 6k | 5250 | `70cda1270c41eb0137d10d7ac295e575e127af98b77fe575facd39c3b4dd83a5` |

[Full selected-model metadata](../phase6_4_final_test/selected_model_manifest.json). Stage 6.5 reads checkpoint sidecars and recorded hashes; it does not load model weights.

## 4. Training and validation methodology

The historical Phase 5 run trained for 6000 episodes and selected its reference at episode 5250. Stage 6.3B trained two fresh, uninterrupted 10,000-episode models with shared training seed 42 and identical workloads. Replay augmentation used a separate RNG seed 630342. Greedy epsilon-zero validation occurred every 250 episodes on the same seed-1729 128-workload / 645-process collection.

Architecture and optimizer hyperparameters were held fixed. Recorded settings: learning_rate=0.0005, gamma=1.0, batch_size=64, replay_capacity=20000, target_sync_interval=250, max_gradient_norm=10.0.

| Training run | Environment steps | Optimizer updates | Best episode | Best validation waiting | Final validation waiting |
| --- | --- | --- | --- | --- | --- |
| original_10000 | 274300 | 68326 | 9000 | 6.303876 | 6.328682 |
| augmented_10000 | 274300 | 68326 | 9750 | 6.113178 | 6.156589 |

[Greedy validation waiting curves](figures/validation_learning_curves.png) ([SVG](figures/validation_learning_curves.svg)); [exploratory training rewards](figures/training_reward.png) ([SVG](figures/training_reward.svg)); [optimizer TD loss](figures/optimizer_loss.png) ([SVG](figures/optimizer_loss.svg)); [logged training epsilon](figures/epsilon_schedule.png) ([SVG](figures/epsilon_schedule.svg)).

The reward panel shows raw episode totals and a trailing 100-episode arithmetic mean, with shorter initial windows. These rewards depend on workload size and exploratory behavior; they are not validation scores. Loss uses a trailing 250-optimizer-update mean, displayed every 50 updates plus the final update. Epsilon shows end-of-episode observations against CPU steps; both schedules match exactly.

| Model | Validation minimum, episodes 8000–10000 | Maximum | Final |
| --- | --- | --- | --- |
| Original DQN 10k | 6.303876 | 6.558140 | 6.328682 |
| Augmented DQN 10k | 6.113178 | 6.322481 | 6.156589 |

Both histories continue to fluctuate late in training, and final scores exceed the selected best scores. Finite-sample validation, nonstationary replay/targets, and ongoing parameter updates can contribute; these logs alone do not isolate a cause. Lower TD loss and improved validation do not prove mathematical convergence or a permanent plateau.

## 5. Final held-out comparison

All six policies were tested on identical workloads with epsilon zero for DQN. Timing means divide summed process times by total processes. Switches/preemptions divide workload totals by workload count. The table below copies recorded values; no policies are rerun.

| Policy | Average waiting | Average turnaround | Average response | Switches/workload | Preemptions/workload | Total reward | Waiting gap to SRTF (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SRTF | 5.826155 | 11.255286 | 5.132341 | 4.621094 | 0.703125 | -7440 | 0.000000 |
| Augmented DQN 10k | 5.964761 | 11.393892 | 4.712608 | 4.968750 | 1.050781 | -7617 | 2.379032 |
| Original DQN 10k | 6.118246 | 11.547377 | 4.671887 | 5.082031 | 1.164062 | -7813 | 5.013441 |
| Reference DQN 6k | 6.517619 | 11.946750 | 4.881754 | 5.152344 | 1.234375 | -8323 | 11.868280 |
| FCFS | 8.429131 | 13.858262 | 8.429131 | 3.917969 | 0.000000 | -10764 | 44.677419 |
| Round Robin | 11.218481 | 16.647612 | 2.451840 | 13.046875 | 9.128906 | -14326 | 92.553763 |

[Machine-readable consolidated table](consolidated_benchmark.csv). [waiting](figures/final_waiting.png) ([SVG](figures/final_waiting.svg)); [turnaround](figures/final_turnaround.png) ([SVG](figures/final_turnaround.svg)); [response](figures/final_response.png) ([SVG](figures/final_response.svg)); [switches](figures/final_context_switches.png) ([SVG](figures/final_context_switches.svg)); [preemptions](figures/final_preemptions.png) ([SVG](figures/final_preemptions.svg)).

Recorded total reward equals negative total waiting for every rollout. The saved per-process and per-workload files independently reproduce all displayed means and rewards. RR has the shortest response but substantially greater waiting and switching. Compared with SRTF, each DQN has lower response time here and higher waiting; this is a tradeoff, not overall superiority.

## 6. SRTF optimality-gap analysis

| DQN | Absolute waiting gap | Gap (%) | Workloads tying SRTF | Workloads above SRTF |
| --- | --- | --- | --- | --- |
| Augmented DQN 10k | 0.138606 | 2.379032 | 169 | 87 |
| Original DQN 10k | 0.292091 | 5.013441 | 133 | 123 |
| Reference DQN 6k | 0.691464 | 11.868280 | 85 | 171 |

Augmented waiting is 5.964761, versus SRTF 5.826155; its gap is 2.379032%. It ties SRTF on 169/256 workloads. SRTF is no worse on waiting for every saved workload. [DQN optimality gaps](figures/optimality_gaps.png) ([SVG](figures/optimality_gaps.svg)).

## 7. Replay permutation augmentation ablation

Training already randomized occupied workload slots. Augmentation therefore did not add a second workload shuffle: it transformed one replay transition's coordinates before insertion. A shared occupied-slot permutation moves whole six-feature blocks in state and next state, remaps the action and next-action mask, and preserves globals, reward, done flag, hidden future-job observations, padding, and the one-transition-per-step budget.

The primary held-out waiting contrast is **-0.153485**, 95% paired workload-bootstrap CI **[-0.207905, -0.102263]**. [Existing paired confidence interval](figures/augmentation_difference_ci.png) ([SVG](figures/augmentation_difference_ci.svg)).

Across workloads augmentation improves 90, worsens 36, and ties 130 original-DQN results. Equal budgets control workload sequence, environment steps, updates, architecture, hyperparameters, and validation; selected checkpoints occur at different episodes.

| Augmented minus comparator | Metric | Difference | 95% low | 95% high |
| --- | --- | --- | --- | --- |
| Original DQN 10k | waiting_time | -0.153485 | -0.207905 | -0.102263 |
| Original DQN 10k | response_time | 0.040720 | -0.149158 | 0.226080 |
| Original DQN 10k | context_switches | -0.113281 | -0.273438 | 0.046875 |
| Original DQN 10k | preemptions | -0.113281 | -0.273438 | 0.046875 |
| SRTF | waiting_time | 0.138606 | 0.108919 | 0.173041 |
| FCFS | waiting_time | -2.464370 | -2.695157 | -2.248218 |
| Round Robin | waiting_time | -5.253720 | -5.601670 | -4.921065 |
| Reference DQN 6k | waiting_time | -0.552858 | -0.653828 | -0.454315 |

Response increases slightly versus the equal-budget original, while switches and preemptions decrease slightly. Their intervals include zero; a reliable improvement in these secondary metrics is not established. The primary waiting interval excludes zero and supports a same-distribution benefit for this fixed checkpoint pair, not a universal training advantage.

## 8. Permutation consistency analysis

These are **VALIDATION / PERMUTATION EXPERIMENTS**, not held-out test estimates. All models share 1,969 frozen states from reference-DQN and SRTF trajectories and 13,783 non-identity comparisons each. Full-trajectory variability uses the original plus seven fixed alternative layouts per workload. Identity agreement and trajectory waiting range answer different questions.

| Model | Identity agreement (%) | Mean waiting range | Mean population std | Changed workloads (%) |
| --- | --- | --- | --- | --- |
| Reference DQN 6k | 68.976275 | 1.373568 | 0.473396 | 97.656250 |
| Original DQN 10k | 82.021331 | 0.747917 | 0.262444 | 96.093750 |
| Augmented DQN 10k | 87.948923 | 0.409245 | 0.149142 | 81.250000 |

[Fixed-state agreement](figures/permutation_agreement.png) ([SVG](figures/permutation_agreement.svg)); [within-workload waiting variation](figures/permutation_waiting_range.png) ([SVG](figures/permutation_waiting_range.svg)); [robustness by process count](figures/permutation_by_process_count.png) ([SVG](figures/permutation_by_process_count.svg)).

[Conditional agreement by destination slot](figures/permutation_by_slot.png) ([SVG](figures/permutation_by_slot.svg)) conditions on the destination of the originally chosen process. Slot denominators differ; slots 4 and 5 occur only for 5–6 and 6 jobs. Slot 3/5 weakness in the reference and improvements in later models describe layout sensitivity, not an isolated causal slot mechanism.

Stage 6.3A's reference pooled alternatives-minus-original waiting delta is +0.016390, CI [-0.117396, +0.137551]. Thus aggregate cancellation can coexist with strong per-workload sensitivity. Augmentation reduces sensitivity in the saved comparisons, but agreement remains below 100% and waiting still varies: exact process-slot permutation equivariance is not established.

## 9. Gantt-chart case studies

Case selection was recorded during Stage 6.4: minimum and maximum augmented-minus-original per-workload mean waiting, with lowest workload ID breaking ties. These extremes illustrate mechanisms and failure cases; they are not representative averages. Charts use actual process IDs, shared CPU axes, original slot/arrival/burst tables, consistent per-process colors, and saved mean waiting. Only adjacent identical segments are merged, including idle intervals.

### Workload 69: best augmented-relative case

| Original slot | Process | Arrival | Burst |
| --- | --- | --- | --- |
| 0 | P5 | 6 | 10 |
| 1 | P4 | 8 | 2 |
| 2 | P2 | 6 | 10 |
| 3 | P1 | 2 | 10 |
| 4 | P3 | 7 | 8 |

| Scheduler | Mean waiting |
| --- | --- |
| SRTF | 10.200000 |
| Original DQN 10k | 12.200000 |
| Augmented DQN 10k | 10.200000 |

Augmented minus original = -2.000000 CPU units / process. [Aligned recorded timelines for workload 69](figures/gantt_workload_69.png) ([SVG](figures/gantt_workload_69.svg)).

### Workload 201: worst augmented-relative case

| Original slot | Process | Arrival | Burst |
| --- | --- | --- | --- |
| 0 | P3 | 2 | 9 |
| 1 | P4 | 5 | 7 |
| 2 | P5 | 0 | 4 |
| 3 | P6 | 6 | 6 |
| 4 | P2 | 5 | 10 |
| 5 | P1 | 2 | 5 |

| Scheduler | Mean waiting |
| --- | --- |
| SRTF | 10.166667 |
| Original DQN 10k | 10.166667 |
| Augmented DQN 10k | 11.166667 |

Augmented minus original = +1.000000 CPU units / process. [Aligned recorded timelines for workload 201](figures/gantt_workload_201.png) ([SVG](figures/gantt_workload_201.svg)).

Every recorded process runtime equals its burst; arrivals, stable indices, first-start/completion times, and mean waiting agree with saved results. Merged and raw timelines preserve identical durations and timing. No simulated segments were fabricated or executed.

## 10. Process-count effects

[All-policy held-out waiting by process count](figures/by_process_count.png) ([SVG](figures/by_process_count.svg)).

| Processes | Workloads | SRTF waiting | Augmented waiting | Original waiting | Reference waiting | Augmented gap (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 90 | 4.027778 | 4.075000 | 4.163889 | 4.386111 | 1.172414 |
| 5 | 79 | 5.825316 | 5.896203 | 6.050633 | 6.445570 | 1.216862 |
| 6 | 87 | 7.067050 | 7.319923 | 7.517241 | 8.042146 | 3.578205 |

Augmented waiting is lower than original in each held-out process-count group, while six-process workloads retain the largest DQN/SRTF gaps. Validation/permutation group plots separately show reduced consistency and larger waiting ranges with more occupied slots; these are descriptive effects, not proof of a specific causal mechanism.

## 11. Statistical uncertainty and limitations

Held-out intervals copy the existing 5000-draw paired percentile workload bootstrap, seed 64021729. Whole workloads retain all paired schedulers, and each timing draw recalculates summed process-time differences / summed process counts. This reporting phase does not draw new bootstrap samples. Validation/permutation intervals likewise reuse saved statistics, with model-selection bias and dependent states/layouts retained at workload level.

- One paired DQN training seed: intervals quantify sampled-workload uncertainty, not independent training-seed variation.
- The held-out test follows the same simulated distribution as training; it does not test distribution shift.
- Arrived burst lengths are perfectly known. Single-core CPU, zero switch overhead, and no I/O limit external applicability.
- SRTF is optimal for waiting under these assumptions; DQN does not outperform SRTF on this objective.
- Greedy validation selected the checkpoints before held-out testing; best and final scores differ and late fluctuations remain.
- Secondary/subgroup intervals have no multiple-comparison adjustment. Extreme Gantt examples are selected illustrations.
- There is no proof of convergence, exact slot equivariance, real OS execution, or deployment.

## 12. Final conclusions

For the fixed checkpoint pair, replay augmentation reduces held-out waiting from 6.118246 to 5.964761, with a paired interval excluding zero. Saved shared-state and trajectory interventions also show greater permutation consistency and smaller within-workload waiting variation. These observations support a same-distribution benefit for the evaluated models.

SRTF remains the waiting-time reference at 5.826155; augmented DQN retains a 2.379% gap. RR trades greater waiting and switching for faster response. The results do not establish cross-seed reliability, broader-distribution generalization, convergence, or overall DQN superiority.

All figures have 300-DPI PNG and deterministic SVG exports. [Figure/source manifest](figure_manifest.json) records exact plotted values, source SHA-256 hashes, export hashes, and verification. Stage 6.5 adds presentation only; Phase 7 and the final academic report remain outside this work.
