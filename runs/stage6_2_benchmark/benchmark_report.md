# Stage 6.2 initial benchmark and optimality gap

Evaluated the unchanged episode-5250 checkpoint `runs/phase5_full_6000/best_model.pt`
with epsilon 0 against the unchanged FCFS, SRTF, and Round Robin (quantum 2).
The saved Phase 5 configuration and existing `Validator` reconstruct exactly
128 validation workloads / 645 processes using seed 1729. Original workload
order and process indices were preserved. This is model-selection validation,
not an unbiased final test; reserved final test workloads were not accessed.

## Comparison

Timing means are process-weighted CPU time units. Switches and preemptions are
means per workload. Total reward equals negative summed process waiting time.

| Policy | Waiting | Turnaround | Response | Context switches | Preemptions | Total reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 6.586047 | 12.000000 | 5.055814 | 5.101562 | 1.156250 | -4248 |
| FCFS | 8.401550 | 13.815504 | 8.401550 | 3.945312 | 0.000000 | -5419 |
| SRTF | 6.006202 | 11.420155 | 5.314729 | 4.601562 | 0.656250 | -3874 |
| Round Robin, q=2 | 11.372093 | 16.786047 | 2.479070 | 12.960938 | 9.015625 | -7335 |

DQN waiting reproduces the Phase 5 reported value exactly:
**6.586046511627907**. Turnaround 12.0 and mean workload reward -33.1875 also
match the checkpoint metadata and episode-5250 validation CSV row.

## Optimality gap and paired results

- DQN minus SRTF process-weighted mean waiting: **0.57984496124031** CPU time units.
- Relative to SRTF: **9.654104284976768%** greater waiting.
- Total extra waiting across 645 processes: **374** CPU time units.
- SRTF waiting <= every other policy on **all 128 workloads**, with zero violations.
- DQN ties SRTF on **40 workloads (31.25%)** and is worse on **88**. It never beats SRTF on waiting.
- Unweighted mean per-workload mean gap: **0.5365885416666667**; median **0.29166666666666663**.
- DQN beats FCFS on waiting in **105** workloads, ties in **11**, and loses in **12**.

DQN improves aggregate waiting over FCFS and RR, while SRTF remains better on
waiting and turnaround. DQN response is slightly faster than SRTF on aggregate;
RR has the fastest response and the highest switching/preemption counts. These
results do not demonstrate overall DQN superiority.

## Results by process count

All timing means in this table are waiting time per process. Full waiting,
turnaround, response, switch, preemption, reward, and count results for every
policy/group are in `by_process_count.csv` and `benchmark_summary.json`.

| Processes per workload | Workloads | Processes | DQN | FCFS | SRTF | RR q=2 | DQN–SRTF gap | Gap % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 41 | 164 | 4.231707 | 5.652439 | 3.969512 | 7.597561 | 0.262195 | 6.6052% |
| 5 | 41 | 205 | 5.843902 | 7.682927 | 5.439024 | 10.326829 | 0.404878 | 7.4439% |
| 6 | 46 | 276 | 8.536232 | 10.568841 | 7.637681 | 14.391304 | 0.898551 | 11.7647% |

The DQN waiting gap grows with process count in this collection.

## Best and worst DQN-relative-to-SRTF workloads

IDs are zero-based positions in the original validation collection. All 40 tied
workloads are listed in the summary JSON. One representative best workload is
**ID 1**, with definitions in original slot order:

| Slot | PID | Arrival | Burst |
| ---: | --- | ---: | ---: |
| 0 | P1 | 7 | 5 |
| 1 | P2 | 4 | 5 |
| 2 | P4 | 0 | 4 |
| 3 | P3 | 7 | 5 |

DQN and SRTF both have mean waiting **2.25**, gap **0**.

The worst workload by excess mean waiting is **ID 81**:

| Slot | PID | Arrival | Burst |
| ---: | --- | ---: | ---: |
| 0 | P1 | 4 | 10 |
| 1 | P4 | 6 | 10 |
| 2 | P2 | 8 | 9 |
| 3 | P6 | 1 | 8 |
| 4 | P3 | 4 | 5 |
| 5 | P5 | 0 | 9 |

DQN mean waiting **20.333333**, SRTF **15.666667**: excess **4.666667** per
process, **28** total, or **29.7872%**. DQN is also worse than FCFS (**17.333333**)
on this workload. Full timelines for both example workloads are retained in
`benchmark_summary.json`; all per-process outcomes are in `per_process_results.csv`.

## Process-slot diagnostic

The DQN exhibits a marked **slot-associated error pattern** on the original
validation slots. When a slot is uniquely shortest among multiple ready jobs:

| Slot (zero-based) | Selected / opportunities | Selection rate |
| ---: | ---: | ---: |
| 0 | 433 / 436 | 99.31% |
| 1 | 430 / 444 | 96.85% |
| 2 | 393 / 420 | 93.57% |
| 3 | 93 / 244 | 38.11% |
| 4 | 273 / 281 | 97.15% |
| 5 | 166 / 261 | 63.60% |

Slots 3 and 5 account for **246 of 298** skipped uniquely-shortest opportunities
(82.55%). The slot-3 pattern persists within each process-count group: selected
49.23% for four processes, 37.36% for five, and 30.68% for six. Among six-process
workloads, slots 0/1/2/4 were selected 94.01–99.36% of the time when uniquely
shortest, compared with slot 3 at 30.68% and slot 5 at 63.60%.

This is consistent with uneven slot behavior and merits future investigation,
but does not establish a causal slot effect: ready-process feature distributions
and state histories differ, and repeated decisions within a workload are dependent.
Only **nine** observed selections involved multiple ready slots with exactly
identical six-feature observations, too few for a strong matched-feature claim.
No workloads or indices were permuted to perform this analysis.

## Verification and implementation

- Complete suite: **151 passed, 1 skipped** (CUDA unavailable); all previous tests pass.
- The complete benchmark was repeated using the same reconstructed collection
  object. Every result, timeline, and DQN decision record matched exactly.
- Both network digests remained unchanged, optimizer updates remained 35907,
  replay stayed empty, and no optimization or training was performed.
- Workload identity, stable indices, burst accounting, reward identities, and
  per-workload SRTF inequalities passed runtime checks.
- Checkpoint, metadata, validation CSV/config, simulator, adapter, agent, training,
  and baseline source files retained their original SHA-256 hashes.

Created `src/benchmark/{__init__,evaluation,analysis,plots,run}.py` and
`tests/test_benchmark.py`; updated README documentation. Scheduling and learning
implementations were unchanged. Stage 6.3, tuning, further training, and final
test evaluation were not implemented.

## Outputs

The four required CSV/JSON outputs and five comparison charts are in this
directory. Supporting files contain the exact workload definitions, results by
process count, slot summaries, and per-decision diagnostics. The comparison CSV
has 4 policy rows, the paired workload CSV 128 rows, and the process CSV 2580 rows.
