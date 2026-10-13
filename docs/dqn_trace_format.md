# Importing real DQN decision traces

Each file in `exports/dqn_traces/` contains one workload executed by the saved
PyTorch checkpoint. Workloads 69 and 201 are the favorable and unfavorable
augmentation cases from the final evaluation; workload 0 is another held-out
example. These are recorded greedy inference decisions, with no training during
export. The exporter uses the same evaluation engine as the Python demo.

From the repository root, generate or refresh the files:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark.export_traces --workload-ids 0 69 201
```

Use `--checkpoint PATH` to export another model and `--output-dir PATH` to choose
a different destination. Selected `workload_ID.json` files in that destination
are replaced. Checkpoints and saved evaluation records are read only.

## Import contract

The root object has `schema_version: 1` and `trace_type: "cpu_scheduling_dqn"`.
An interactive application can import the file and play back `decisions` in order.

| Field | Meaning |
|---|---|
| `workload_id` | Original, zero-based final-test ID |
| `workload` | Objects with `process_index`, `pid`, `arrival_time`, `burst_time` |
| `model` | Checkpoint hash, actual architecture, greedy mode, training seed |
| `environment` | Simulator assumptions, normalization, observation feature order |
| `decisions` | One transition for each executed CPU unit |
| `timeline` | Full DQN execution, including idle intervals |
| `processes` | Final per-PID completion, waiting, turnaround, and response times |
| `mean_waiting_time`, `total_reward`, etc. | Final DQN metrics |
| `baselines` | Actual FCFS, SRTF, and Round Robin timelines and final metrics |

Each decision contains:

- `step`, `time`: execution step and clock immediately before the decision.
- `observation`: the actual 39 floats supplied to the network before execution.
- `action_mask`: six booleans; only true entries are selectable.
- `q_values`: six raw network predictions from this checkpoint.
- `masked_q_values`: the same values, with invalid outputs represented by JSON
  `null`. No nonstandard `Infinity` or `NaN` values are emitted.
- `selected_action`, `selected_pid`: stable process index and selected identity.
- `epsilon: 0`, `exploratory: false`: these are greedy evaluation decisions.
- `reward`, `waiting_process_indices`, `cumulative_reward`: actual simulator reward.
- `execution`: `{start, end, process_index, pid}` for the executed CPU unit.
- `context_switch`, `preemption`, `process_completed`: separate simulator events.
- `next_time`, `next_observation`, `next_action_mask`, `done`: state after execution
  and any automatic idle jump.
- `idle_interval_after`: an idle interval object, or `null`.

All intervals include their start and exclude their end. A null `process_index`
and null `pid` identify idle time. Every busy interval is exactly one unit.
`next_time` can exceed `execution.end` if the simulator skips an idle gap. The
full timeline also includes idle time before the first decision.

## Playback and the decision inspector

Display process slot `i` using `observation.slice(i * 6, i * 6 + 6)`. Within a
slot the feature order is readiness, completion, original burst/10, remaining
burst/10, accumulated waiting/68, and last-executed flag. Elements 36–38 contain
time/68, arrived count/6, and completed count/6. Future and unused slots are zero.
The full workload is supplied for the workload editor; its future details were
not supplied to the DQN observation.

Show the six actual Q-values and mask before applying the transition. Select
the highest raw Q-value among valid actions; first index wins ties. Q-values
estimate future return and are not probabilities, burst predictions, or a
guarantee of optimality. Then reveal the recorded action and reward.

Use `execution`, `idle_interval_after`, and `next_time` for playback. Use the
final metrics only after completion, or label them as final recorded outcomes.
Provisional metrics can be calculated from the executed prefix using the same
workload. The sum of recorded rewards must equal negative total final waiting.

Validate observation lengths, mask lengths, selected actions, time continuity,
and termination before accepting an import. Terminal masks contain six false
entries. Editing a process or choosing a different workload invalidates the
imported DQN trace: reset playback and request a matching trace. Display it as
**Recorded trained DQN inference**, since importing JSON does not execute or
train the neural network in the browser.
