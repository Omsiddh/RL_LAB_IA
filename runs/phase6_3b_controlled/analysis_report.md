# Stage 6.3B controlled training and replay augmentation

Augmentation improved selected-checkpoint original-layout validation waiting in this paired experiment; its workload-bootstrap delta interval excludes zero.

This uses one paired training seed (42), one augmentation seed (630342), and model-selection validation data. Workload bootstrap intervals do not measure training-seed variability or remove checkpoint-selection bias. No final test workloads were used.

## Audit before implementation

`generate_workload` uses a local RNG to sample count 4..6, arrivals 0..8, and bursts 1..10. Definitions initially have PID order P1..Pn. Phase 5 `sample_workload` already draws a fresh 64-bit seed and performs one independent shuffle using the training workload RNG. No second workload shuffle was introduced.

PID placements are uniform in the sampler conditional on process count, with finite-sample fluctuations. Padding causes unequal unconditional slot occupancy; policy-dependent ready/replay exposure need not be uniform. Slot 3 has the same occupancy as 0..2, so its previously observed sensitivity cannot be attributed simply to padding.

| Slot | Occupied / 10,000 | Mean arrival | Mean burst |
| --- | --- | --- | --- |
| 0 | 10000 | 4.0210 | 5.4918 |
| 1 | 10000 | 3.9552 | 5.4790 |
| 2 | 10000 | 3.9971 | 5.5090 |
| 3 | 10000 | 3.9973 | 5.5075 |
| 4 | 6565 | 4.0331 | 5.5085 |
| 5 | 3288 | 4.0335 | 5.5547 |

Workload counts: 4 processes = 3,435; 5 = 3,277; 6 = 3,288. Conditional PID/slot cell counts are 821..897 (expected 858.75), 613..688 (655.4), and 503..584 (548). See [workload_audit.json](workload_audit.json) for all cells and the workload stream digest.

## Implementation and controls

Added `src/training/augmentation.py`. `TrainingConfig` and the training CLI expose `replay_augmentation=none|permutation` and `augmentation_seed`; the default is none. Before each replay insertion, enabled augmentation uniformly samples a permutation of occupied workload slots, including hidden future jobs. It moves whole six-feature blocks at both transition endpoints, remaps the action with the inverse map, and moves the next-action mask. Globals, reward, terminal flag, unused slots, and live simulator actions remain unchanged. One transformed transition replaces one original transition; there are no additional samples.

The disabled branch retains the old `remember` call. The new A run reproduces all 6,000 historical training rows and all original validation rows exactly, including loss/gradient means. Both new runs use identical full workload definitions/order, 274,300 environment steps, 68,326 optimizer updates, replay capacity 20,000, batch size 64, and unchanged 39->64->64->6 DQN, optimizer, reward, epsilon schedule, warmup, and synchronization settings.

Both runs began from fresh identical seed-42 initialization and ran uninterrupted for 10,000 episodes. Neither loaded the earlier checkpoint for training. Greedy validation uses the exact seed-1729 128-workload / 645-process collection at episode 0 and every 250 episodes. Strictly lower process-weighted original-layout waiting selects each best checkpoint. Best and final checkpoint files were independently reloaded and reproduced their logged scores.

New analysis modules audit sampling, observe training without extra updates, freeze/serialize common states, evaluate the selected models, compute paired comparisons, and generate charts/reports. The existing fixed-state evaluator gained an optional shared-collection argument; its default behavior remains unchanged. Simulator, RL adapter, DQN architecture, reward, and baseline sources were untouched. Existing checkpoints and experiment artifacts were hash-verified unchanged.

## Training statistics and checkpoints

| Run | Best episode | Best waiting | Final waiting | Updates | Mean loss | Max loss | Max pre-clip norm | Clip count | Max valid abs Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_10000 | 9000 | 6.303876 | 6.328682 | 68326 | 1.466648 | 8.501230 | 23.267 | 1001 | 167.140 |
| augmented_10000 | 9750 | 6.113178 | 6.156589 | 68326 | 1.542725 | 7.385961 | 20.944 | 690 | 153.190 |

Both runs have zero non-finite losses, gradient norms, validation Q-values, or weights detected. Gradient clipping statistics include actual post-clipping norms; floating-point norms can be slightly above the threshold of 10. Maxima span training, not just the selected checkpoint. Full optimizer-level diagnostics and validation Q summaries are saved for both runs.

Best/final artifacts and complete histories:

- [A best](original_10000/best_model.pt), [A final](original_10000/final_model.pt), [A training CSV](original_10000/training_metrics.csv), [A validation CSV](original_10000/validation_metrics.csv), [A diagnostics](original_10000/optimizer_diagnostics.csv).
- [B best](augmented_10000/best_model.pt), [B final](augmented_10000/final_model.pt), [B training CSV](augmented_10000/training_metrics.csv), [B validation CSV](augmented_10000/validation_metrics.csv), [B diagnostics](augmented_10000/optimizer_diagnostics.csv).

## Original validation performance

| Selected model | Episode | Waiting | Turnaround | Response | Switches/workload | Preemptions/workload | Total reward | Gap to SRTF | Gap % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 5250 | 6.586047 | 12.000000 | 5.055814 | 5.101562 | 1.156250 | -4248 | 0.579845 | 9.654 |
| A: original 10k | 9000 | 6.303876 | 11.717829 | 4.875969 | 5.101562 | 1.156250 | -4066 | 0.297674 | 4.956 |
| B: augmented 10k | 9750 | 6.113178 | 11.527132 | 5.027907 | 4.890625 | 0.945312 | -3943 | 0.106977 | 1.781 |

SRTF waiting = 6.006201550387597. Timing means are process weighted; switches and preemptions are workload weighted. Reward equals negative total waiting on every rollout. SRTF waiting is invariant and <= each DQN on every tested arrangement.

## Shared fixed-state and full-trajectory permutations

All three models use the same 1,969 frozen original states, producing 13,783 non-identity comparisons per model. Sources are the episode-5250 reference DQN and SRTF original trajectories. The observations are saved in [fixed_state_observations.npz](evaluation/fixed_state_observations.npz), without pickle, and reloaded exactly. No model collects its own distinct observations for the action-agreement comparison.

Every model uses the original Stage 6.3A per-workload maps: original plus seven fixed alternatives. The reference reproduces Stage 6.3A action, workload, and per-process CSVs exactly. Evaluation repeats exactly and preserves full model/optimizer/replay/RNG/gradient state.

| Selected model | Fixed-state agreement % | Alternative waiting | Alternative response | Alternative switches | Alternative preemptions | Mean waiting range | Mean population std | Changed workloads % | Alternative gap % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 68.976275 | 6.602436 | 5.151274 | 5.087054 | 1.141741 | 1.373568 | 0.473396 | 97.656250 | 9.926986 |
| A: original 10k | 82.021331 | 6.310963 | 5.027243 | 5.024554 | 1.079241 | 0.747917 | 0.262444 | 96.093750 | 5.074121 |
| B: augmented 10k | 87.948923 | 6.150609 | 4.955703 | 4.992188 | 1.046875 | 0.409245 | 0.149142 | 81.250000 | 2.404307 |

Ranges and population standard deviations use all eight complete-trajectory arrangements per workload. Agreement excludes identity layouts. Fixed-state actions and complete trajectory performance answer different questions; subsequent states differ when policy decisions diverge.

## Results by process count

| Model | Processes | Original waiting | Alternative waiting | SRTF waiting | Agreement % | Mean range | Original gap % | Alternative gap % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 4 | 4.231707 | 4.357143 | 3.969512 | 72.760918 | 0.957317 | 6.605223 | 9.765196 |
| Reference | 5 | 5.843902 | 5.893380 | 5.439024 | 71.537456 | 1.278049 | 7.443946 | 8.353619 |
| Reference | 6 | 8.536232 | 8.463251 | 7.637681 | 63.701830 | 1.829710 | 11.764706 | 10.809162 |
| A: original 10k | 4 | 4.134146 | 4.140244 | 3.969512 | 86.454478 | 0.475610 | 4.147465 | 4.301075 |
| A: original 10k | 5 | 5.614634 | 5.615331 | 5.439024 | 83.471254 | 0.497561 | 3.228700 | 3.241512 |
| A: original 10k | 6 | 8.105072 | 8.117495 | 7.637681 | 77.228494 | 1.213768 | 6.119545 | 6.282190 |
| B: augmented 10k | 4 | 4.048780 | 4.041812 | 3.969512 | 90.796941 | 0.286585 | 1.996928 | 1.821374 |
| B: augmented 10k | 5 | 5.507317 | 5.515679 | 5.439024 | 89.960801 | 0.263415 | 1.255605 | 1.409353 |
| B: augmented 10k | 6 | 7.789855 | 7.875259 | 7.637681 | 83.904243 | 0.648551 | 1.992410 | 3.110599 |

| Model | Processes | Original response | Alternative response | Original switches | Alternative switches | Original preemptions | Alternative preemptions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 4 | 3.091463 | 3.085366 | 3.804878 | 3.790941 | 0.902439 | 0.888502 |
| Reference | 5 | 4.614634 | 4.459233 | 4.780488 | 4.996516 | 0.878049 | 1.094077 |
| Reference | 6 | 6.550725 | 6.892857 | 6.543478 | 6.322981 | 1.630435 | 1.409938 |
| A: original 10k | 4 | 3.085366 | 3.102787 | 3.731707 | 3.773519 | 0.829268 | 0.871080 |
| A: original 10k | 5 | 4.112195 | 4.344948 | 5.121951 | 4.947735 | 1.219512 | 1.045296 |
| A: original 10k | 6 | 6.507246 | 6.677536 | 6.304348 | 6.208075 | 1.391304 | 1.295031 |
| B: augmented 10k | 4 | 3.030488 | 3.050523 | 3.731707 | 3.738676 | 0.829268 | 0.836237 |
| B: augmented 10k | 5 | 4.385366 | 4.300348 | 4.780488 | 4.947735 | 0.878049 | 1.045296 |
| B: augmented 10k | 6 | 6.692029 | 6.574534 | 6.021739 | 6.149068 | 1.108696 | 1.236025 |

The process-count CSV additionally records original/alternative turnaround, rewards, and all SRTF control metrics.

## Paired B minus A comparisons

| Metric | Difference | 95% CI low | 95% CI high |
| --- | --- | --- | --- |
| original_waiting | -0.190698 | -0.262920 | -0.122101 |
| permuted_waiting | -0.160354 | -0.196621 | -0.126433 |
| fixed_state_agreement | 0.059276 | 0.042093 | 0.075907 |
| waiting_range | -0.338672 | -0.422656 | -0.258063 |
| waiting_std | -0.113302 | -0.139564 | -0.088112 |
| original_response | 0.151938 | -0.098096 | 0.415547 |
| original_context_switches | -0.210938 | -0.367188 | -0.054688 |
| original_preemptions | -0.210938 | -0.367188 | -0.054688 |

Agreement differences are fractions (multiply by 100 for percentage points); negative waiting/range/std differences favor augmentation. Confidence intervals use 5,000 paired percentile bootstrap draws, seed 63021729, resampling whole workloads and retaining their states/arrangements. Timing differences use process-weighted ratios. Additional A/reference and B/reference contrasts and process-count groups are in [paired_model_comparison.csv](paired_model_comparison.csv).

## Complete validation history

| Episode | A waiting | A best | B waiting | B best | Training epsilon | Updates (both) | A interval loss | B interval loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9.066667 | True | 9.066667 | True | 1.00000 | 0 |  |  |
| 250 | 8.269767 | True | 7.838760 | True | 0.93341 | 1503 | 0.450574 | 0.455594 |
| 500 | 7.434109 | True | 6.868217 | True | 0.86837 | 3215 | 1.369248 | 1.354217 |
| 750 | 7.175194 | True | 6.831008 | True | 0.80287 | 4938 | 2.237011 | 2.208823 |
| 1000 | 7.258915 | False | 7.079070 | False | 0.73752 | 6658 | 2.919115 | 2.888599 |
| 1250 | 7.581395 | False | 7.703876 | False | 0.67254 | 8368 | 3.235789 | 3.200101 |
| 1500 | 7.680620 | False | 7.981395 | False | 0.60645 | 10107 | 3.123458 | 3.099684 |
| 1750 | 7.863566 | False | 7.982946 | False | 0.54111 | 11827 | 2.947440 | 2.929426 |
| 2000 | 7.924031 | False | 7.862016 | False | 0.47596 | 13541 | 2.598718 | 2.540304 |
| 2250 | 8.032558 | False | 7.454264 | False | 0.41009 | 15275 | 2.453647 | 2.320373 |
| 2500 | 7.804651 | False | 7.508527 | False | 0.34560 | 16972 | 2.224053 | 2.244239 |
| 2750 | 7.651163 | False | 7.367442 | False | 0.28218 | 18641 | 2.049921 | 2.257089 |
| 3000 | 7.593798 | False | 7.452713 | False | 0.21706 | 20354 | 1.848682 | 2.202317 |
| 3250 | 7.506977 | False | 7.285271 | False | 0.15366 | 22023 | 1.674224 | 2.061389 |
| 3500 | 7.533333 | False | 7.106977 | False | 0.08896 | 23725 | 1.619439 | 1.975275 |
| 3750 | 7.232558 | False | 7.204651 | False | 0.05000 | 25514 | 1.474166 | 1.803634 |
| 4000 | 7.316279 | False | 6.956589 | False | 0.05000 | 27218 | 1.356729 | 1.687745 |
| 4250 | 7.170543 | True | 7.013953 | False | 0.05000 | 28974 | 1.252823 | 1.612017 |
| 4500 | 7.203101 | False | 7.125581 | False | 0.05000 | 30688 | 1.197042 | 1.619064 |
| 4750 | 7.110078 | True | 6.719380 | True | 0.05000 | 32404 | 1.103672 | 1.554969 |
| 5000 | 6.981395 | True | 6.638760 | True | 0.05000 | 34195 | 1.064946 | 1.489430 |
| 5250 | 6.586047 | True | 6.658915 | False | 0.05000 | 35907 | 1.068747 | 1.462688 |
| 5500 | 6.806202 | False | 6.741085 | False | 0.05000 | 37656 | 1.040880 | 1.427604 |
| 5750 | 6.714729 | False | 6.496124 | True | 0.05000 | 39360 | 1.055188 | 1.347411 |
| 6000 | 6.643411 | False | 6.559690 | False | 0.05000 | 41062 | 1.082488 | 1.259713 |
| 6250 | 6.641860 | False | 6.434109 | True | 0.05000 | 42777 | 1.098565 | 1.178340 |
| 6500 | 6.511628 | True | 6.443411 | False | 0.05000 | 44480 | 1.051970 | 1.061199 |
| 6750 | 6.480620 | True | 6.406202 | True | 0.05000 | 46218 | 1.028443 | 1.040537 |
| 7000 | 6.648062 | False | 6.350388 | True | 0.05000 | 47970 | 1.066537 | 1.037270 |
| 7250 | 6.581395 | False | 6.328682 | True | 0.05000 | 49643 | 1.065990 | 0.985284 |
| 7500 | 6.444961 | True | 6.339535 | False | 0.05000 | 51313 | 1.054082 | 0.916049 |
| 7750 | 6.539535 | False | 6.232558 | True | 0.05000 | 53019 | 1.046126 | 0.885612 |
| 8000 | 6.558140 | False | 6.322481 | False | 0.05000 | 54726 | 1.060515 | 0.863182 |
| 8250 | 6.556589 | False | 6.221705 | True | 0.05000 | 56364 | 1.059215 | 0.861744 |
| 8500 | 6.494574 | False | 6.249612 | False | 0.05000 | 58088 | 1.006101 | 0.827975 |
| 8750 | 6.457364 | False | 6.189147 | True | 0.05000 | 59744 | 0.917557 | 0.793685 |
| 9000 | 6.303876 | True | 6.127132 | True | 0.05000 | 61437 | 0.892499 | 0.792164 |
| 9250 | 6.324031 | False | 6.155039 | False | 0.05000 | 63199 | 0.908070 | 0.821186 |
| 9500 | 6.314729 | False | 6.155039 | False | 0.05000 | 64930 | 0.929745 | 0.812726 |
| 9750 | 6.367442 | False | 6.113178 | True | 0.05000 | 66652 | 0.946088 | 0.805489 |
| 10000 | 6.328682 | False | 6.156589 | False | 0.05000 | 68326 | 0.929443 | 0.815259 |

Validation epsilon is always zero. Training epsilon and update counts are identical at each checkpoint because the workload sequence and CPU burst accounting match. Both CSV histories additionally contain turnaround and reward at every validation checkpoint.

## Interpretation

Augmentation improved selected-checkpoint original-layout validation waiting in this paired experiment; its workload-bootstrap delta interval excludes zero.

B minus A fixed-state agreement = +5.928 percentage points, and mean waiting range difference = -0.338672. These directly describe consistency on the shared observations and sampled layouts. A change in aggregate mean should not be substituted for these sensitivity measurements.

The augmentation intervention changes replay coordinate coverage while holding workload generation, architecture, hyperparameters, environment steps, optimizer updates, and validation fixed. It does not establish which individual slot mechanism explains any improvement. The selected checkpoints may occur at different episodes. One seed pair is insufficient to establish a generally reliable training advantage or convergence, and the bootstrap intervals are unadjusted for multiple comparisons. No Phase 6.4 work or final-test evaluation was performed.

Figures: [learning curves](learning_curves.png), [stability](stability_diagnostics.png), [checkpoint metrics](checkpoint_comparison.png), [permutation consistency](permutation_consistency_comparison.png).

## Test results

188 passed, 1 skipped. Command: `.\.venv\Scripts\python.exe -m pytest -q`. The skip is the existing CUDA availability test.
