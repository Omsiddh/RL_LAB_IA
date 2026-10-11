# Adaptive Preemptive CPU Scheduling - Phases 2, 3 and 4

A single-core simulator with one-unit scheduling decisions, no I/O blocking,
and zero context-switch overhead. Runtime dependencies are standard-library
only for the simulator; the Phase 3 adapter uses NumPy. Pytest is needed for
tests. The Phase 4 DQN components use PyTorch. Requires Python 3.11 or newer.

- `src/env/environment.py`: ordinary `Process` and `CPUSchedulingEnv` classes,
  immutable workload tuples, dictionary states, execution/idle timeline,
  rewards and timing metrics.
- `src/env/workload.py`: seeded random generation. Defaults are 4–6 processes,
  bursts of 1–10, and arrivals at 0–8; all generation bounds are inclusive and
  configurable. Explicit workloads can use other counts and bounds.
- `src/env/tests/test_environment.py`: deterministic examples, input/reset
  isolation, workload reproducibility, and reward/timing accounting checks.
- `requirements.txt`: NumPy, PyTorch and pytest; `pyproject.toml` configures test discovery.
- `src/env/__init__.py`: exposes the environment, process models and generator.

From the repository root:

Create a local virtual environment (do not commit it):

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on macOS/Linux:

```sh
source .venv/bin/activate
```

Then install dependencies and run the tests:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

```python
from src.env import CPUSchedulingEnv, generate_workload

workload = (
    ("P1", 0, 5),  # PID, arrival time, original burst time
    ("P2", 1, 2),
    ("P3", 3, 1),
)
env = CPUSchedulingEnv()
state = env.reset(workload)
for action in [0, 1, 1, 2, 0, 0, 0, 0]:
    state, reward, done, info = env.step(action)
assert env.total_reward == -3
metrics = env.get_metrics()

random_workload = generate_workload(seed=42)
custom_workload = generate_workload(
    seed=42, process_count_range=(5, 5),
    burst_time_range=(1, 7), arrival_time_range=(0, 4),
)
```

Actions index the original workload order, never arrival-sorted order. Obtain
eligible indices from `env.valid_actions()` or `state["valid_actions"]`. Invalid
actions raise `ValueError` without changing episode data. Empty workloads are
immediately terminal. Reset creates fresh runtime records without changing the
immutable definitions, so the same workload can be replayed in multiple envs.

Reset skips initial idle time, and each step skips any subsequent idle gap.
Timeline entries are `(start, end, process_index, pid)` tuples. Each interval
includes its start time and excludes its end time. Idle intervals have
`process_index=None` and `pid=None` and add no reward. A returned
nonterminal state always has valid actions. `info["end_time"]` is the end of the
executed unit; `state["time"]` includes any following idle jump.

A context switch changes processes between consecutive busy units. A true
preemption additionally requires the previous process to remain unfinished.
Initial dispatch and dispatch after an idle gap do not count as switches.
Each reward is the negative count of other arrived, unfinished processes
waiting during that unit. At completion, total reward equals negative total
waiting time. Timing metrics are per PID; completion, turnaround and waiting
times remain `None` until completion, and response time until first execution.

Default state snapshots hide burst and remaining times for future arrivals.
`env.get_state(debug=True)` and internal runtime records expose complete data
for debugging. The Phase 3 adapter below provides numerical RL observations.
The Phase 4 section adds DQN components. Full training, baseline policies and
visualization are deferred to future phases.

## How to explain the code

There are two classes. `Process` stores one process's data. `CPUSchedulingEnv`
stores the processes, clock, timeline and totals. They use ordinary constructors
(`__init__`), methods, loops and conditions, without decorators or type hints.

A workload is a tuple of tuples. For example, `("P1", 0, 5)` means P1 arrives
at time 0 and needs 5 CPU units. Tuples cannot have their elements replaced,
so the original definitions remain unchanged. Reset creates new `Process`
objects from these definitions. Only their `remaining_time` changes during
execution; their original `burst_time` stays the same.

`reset(workload)` checks the input and clears the previous episode.
`valid_actions()` finds processes that have arrived and are unfinished.
`is_done()` checks whether every process has finished. `skip_idle()` advances
the clock to the next arrival when nothing can run.

`step(action)` checks the chosen index, counts any switch or preemption,
counts the other waiting processes, runs one CPU unit, records completion,
and skips any idle gap. It returns a state dictionary, the reward, a boolean
completion flag, and an information dictionary. Reading `state["time"]`
gets the clock value. Returned dictionaries are separate copies, so changing
them cannot change the simulator.

The process timing methods use these formulas:

- `get_turnaround_time()`: completion time minus arrival time.
- `get_waiting_time()`: turnaround time minus original burst time.
- `get_response_time()`: first start time minus arrival time.

These methods return `None` until the required timestamps are available.
`get_metrics()` collects these values for each PID and adds the episode totals.

Every other ready process waits exactly one unit during a step. The reward is
the negative number of these processes. Adding rewards over a completed
episode therefore gives the negative of the sum of all waiting times.

`type(action) is int` is a validation check, not a conversion. It rejects
fractions, strings and booleans as actions. `tuple(workload)` deliberately
creates the immutable workload container. `str(index + 1)` turns a process
number into text so the generator can create names such as P1 and P2.

Tests are ordinary functions with `assert` checks. Repeated cases use loops
instead of pytest decorators. `pytest.raises(ValueError)` checks that invalid
input is rejected. `deepcopy` gives tests independent copies when checking
state isolation or exploring alternative action sequences.

## Phase 3: RL observation adapter

`src/env/rl_adapter.py` wraps the existing simulator with `RLAdapter`.
`src/env/tests/test_rl_adapter.py` checks its numerical observations, masks,
normalization, future-data hiding and preservation of simulator behavior.
The adapter uses ordinary methods without decorators or type hints.

```python
from src.env.rl_adapter import RLAdapter

adapter = RLAdapter(max_burst_time=10, max_arrival_time=8)
observation = adapter.reset((("P1", 0, 5), ("P2", 1, 2)))
mask = adapter.valid_action_mask()
observation, reward, done, info = adapter.step(0)
next_mask = info["action_mask"]
```

Observations are NumPy `float32` arrays with shape `(39,)`. Each of six slots
contains, in order: ready flag, completed flag, original burst divided by the
maximum burst, remaining burst divided by the maximum burst, accumulated
waiting divided by the maximum duration, and last-executed flag. The final
three values are time divided by maximum duration, arrived count divided by
six, and completed count divided by six.

Unarrived and unused slots both contain six zeros. No PID, arrival timestamp,
future burst or total workload count is included in the observation. Slot
indices follow input order, matching the simulator without any sorting.
The adapter reads the simulator's ordinary state; internal `adapter.env`
records remain available for debugging and should not be passed to an agent.

The six-element Boolean mask marks only arrived, unfinished slots as valid.
`info["action_mask"]` is an independent copy of the next-state mask. Terminal
masks contain only False. Invalid actions raise before either object's runtime
data changes. As in Phase 2, actions must be ordinary Python integer indices.

Normalization stays fixed across resets. The default duration is calculated
as `max_arrival_time + 6 * max_burst_time`, which gives 68 with default bounds.
An explicit `max_episode_duration` may be supplied if it is at least this
bound. Workloads outside the configured bounds, or with over six processes,
are rejected before the current episode is changed.

For an arrived process, accumulated waiting is elapsed time since arrival
minus executed CPU time. Elapsed time stops at completion, so completed
processes do not keep accumulating waiting. The adapter retains its own last
executed index across idle gaps; the simulator's previous index still clears
after idle for context-switch counting. Rewards are forwarded unchanged.

No DQN, neural network, replay buffer, baseline policy, training or
visualization is implemented in Phase 3.

## Phase 4: DQN components

- `src/agent/network.py`: `QNetwork`, with fully connected layers
  39 -> 64 -> 64 -> 6, ReLU between hidden layers and raw Q-value outputs.
  It accepts one observation or a batch. Masks do not modify network outputs.
- `src/agent/replay_buffer.py`: `ReplayBuffer`, a fixed-capacity FIFO buffer.
  It copies input arrays and returns independent sampled batches. A local
  seeded random generator samples without replacement.
- `src/agent/dqn_agent.py`: `DQNAgent`, providing masked epsilon-greedy
  actions, replay insertion, single optimizer updates, target synchronization,
  and checkpoint save/load.
- `src/agent/__init__.py`: exposes these three components for importing.
- `tests/test_dqn_agent.py`: controlled unit tests and one untrained-agent
  scheduling smoke test. Test discovery includes both this directory and the
  original environment tests.

```python
from src.agent import DQNAgent
from src.env.rl_adapter import RLAdapter

adapter = RLAdapter()
agent = DQNAgent(seed=42)
state = adapter.reset((("P1", 0, 2), ("P2", 1, 1)))
action = agent.select_action(state, adapter.valid_action_mask(), epsilon=1.0)
next_state, reward, done, info = adapter.step(action)
agent.remember(state, action, reward, next_state, done, info["action_mask"])
result = agent.optimize_step()  # None until at least batch_size entries exist.
```

This example shows one interaction, not an episode training loop. The agent
receives only observations, masks, actions, rewards and completion flags. It
does not inspect simulator-private process data or modify rewards.

Defaults are learning rate 0.0005, gamma 1.0, batch size 64, replay capacity
20000, target synchronization every 250 successful optimizer steps, and
gradient clipping at norm 10. Configure them with `learning_rate`, `gamma`,
`batch_size`, `replay_capacity`, `target_sync_interval`, and
`max_gradient_norm`. Adam and Smooth L1 (Huber) loss are used. Computation
defaults to `device="cpu"`; `device="cuda"` requires an available CUDA device
and a compatible PyTorch installation. CUDA requests fail explicitly when
unavailable. Seeds reproduce CPU initialization, exploration and replay
sampling; identical results across different hardware are not guaranteed.

`select_action(observation, action_mask, epsilon)` returns an ordinary Python
index. Epsilon 0 chooses the highest-Q valid action, and epsilon 1 samples
uniformly among valid actions. Empty masks and malformed inputs are rejected.
`remember(state, action, reward, next_state, done, next_action_mask)` stores
a transition. Non-terminal next states must have a nonempty valid-action mask.

`optimize_step()` computes the selected policy Q-values and targets
`reward + gamma * max_valid_target_Q`. Terminal future values are zero; their
empty masks never enter the maximum. Targets have no gradients, and only
policy parameters belong to the optimizer. The target starts as a copy of the
policy and is frozen. `optimizer_steps` counts successful optimizer updates,
independently of environment steps. Successful calls return loss, gradient norm
before clipping, and the update count. `sync_target_network()` also allows an
explicit copy of policy weights to the target.

`agent.save("agent.pt")` saves both networks, optimizer state, configuration,
optimizer update count and action/replay RNG states. `agent.load("agent.pt")`
restores them on the receiving agent's device. Replay experiences are excluded;
loading starts with an empty buffer, so this is not a full training-session
checkpoint. Target isolation and the next synchronization boundary are retained.

Run the complete suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The smoke test completes a generated episode with an untrained network and no
optimizer updates. It verifies integration only, not scheduling quality.
Full training, evaluation baselines, visualization and hyperparameter search
remain outside Phase 4.

## Phase 5: reproducible training and greedy validation

The training pipeline reuses `DQNAgent` for action selection, replay insertion,
optimization, target synchronization, and checkpoint save/load. It does not
implement scheduling baselines or final test-set comparisons.

- `src/training/config.py`: validated training configuration and linear epsilon schedule.
- `src/training/train.py`: training orchestration, CSV logs, checkpoints, and CLI.
- `src/training/validation.py`: fixed validation workloads and isolated greedy evaluation.
- `src/training/plots.py`: headless PNG plots generated from saved logs.
- `tests/test_training.py`: timing, isolation, reproducibility, metrics, and artifact tests.

Install dependencies into the project environment, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.training.train --smoke --seed 42 --output-dir runs/phase5_smoke
.\.venv\Scripts\python.exe -m src.training.train --episodes 500 --seed 42 --output-dir runs/phase5_diagnostic
```

`--smoke` selects 20 episodes; an explicit `--episodes` overrides it. Without
either option the CLI uses the 6000-episode default. A full experiment is an
explicit next step, not something run by the smoke or diagnostic commands.
Choose a fresh output directory for every run; existing training artifacts are
protected from overwriting. Generated files under `runs/` are ignored by Git.

Default controls:

| Control | Default |
| --- | --- |
| Training episodes | 6000 |
| Initial / minimum epsilon | 1.0 / 0.05 |
| Linear decay duration | 100000 environment steps |
| Replay warm-up | 1000 transitions |
| Optimization interval | 4 environment steps |
| Validation interval | 250 completed episodes |
| Fixed validation workloads | 128 |
| Training / validation seed | 42 / 1729 |
| Device / PyTorch CPU threads | cpu / 1 |
| Learning rate / gamma | 0.0005 / 1.0 (DQN defaults) |
| Batch size / replay capacity | 64 / 20000 (DQN defaults) |
| Target synchronization | Every 250 optimizer updates (DQN default) |

CLI flags also expose epsilon controls, warm-up, optimization interval,
validation interval/count/seed, device, and CPU thread count. Programmatic
configuration can override DQN settings without duplicating their defaults:

```python
from src.training.config import TrainingConfig
from src.training.train import Trainer

config = TrainingConfig(episodes=100, seed=42,
                        agent_options={"learning_rate": 0.0005, "batch_size": 64})
summary = Trainer(config).run("runs/custom")
```

Each episode generates a workload with 4–6 processes, arrival times 0–8, and
bursts 1–10. Input order is shuffled independently of process attributes to
randomize slot assignments. Policy selection receives only the adapter's
39-feature observation and six-action mask. Unarrived process features remain
zero; neither total process count nor future arrival/burst information is
passed to the policy. Completed-process records are read afterward for logging.
Episode reward must equal negative total process waiting time, and episode
length must equal total burst time; the pipeline checks these invariants.

Epsilon is computed before each action from the number of completed global
environment steps. It reaches exactly 0.05 at step 100000 and remains there.
After inserting a transition and incrementing the step counter, optimization
runs when replay contains at least both the warm-up and batch-size requirements
and the global step is divisible by the optimization interval. With defaults,
the first update is at step 1000. Only `DQNAgent.optimize_step()` synchronizes
the target network. Loss and pre-clipping gradient norm are averaged over the
successful optimizer updates in each episode; episodes with no updates have
empty CSV cells for these statistics.

Validation workloads are generated once using a dedicated seed, independent of
training workload generation, exploration, and replay sampling. Every evaluation
uses this same collection and slot assignments. Evaluation runs before training,
at the validation interval, and at the end when it falls between intervals.
It uses epsilon 0, evaluation mode, and disabled gradients, with an isolated
action RNG. It restores training mode and RNG afterward, including on failure,
and performs no optimization, replay insertion, or target synchronization.
Waiting and turnaround means divide totals by the total number of processes
across all validation workloads, rather than averaging per-workload means.
No test workloads are used for checkpoint selection.

Every output directory contains:

- `training_metrics.csv`: episode reward, process count, waiting/turnaround means,
  epsilon at episode start/end, episode length, environment and optimizer counters,
  and optimizer statistics.
- `validation_metrics.csv`: evaluation episode/counters, process-weighted means,
  mean workload reward, and whether the result improved the best score.
- `training_config.json`: training controls, resolved DQN configuration, seeds,
  and Python/NumPy/PyTorch versions.
- `best_model.pt`: checkpoint with the lowest validation waiting time; ties keep
  the earlier checkpoint. The untrained policy is eligible.
- `final_model.pt`: final policy saved separately, regardless of its score.
- `best_model.json` and `final_model.json`: corresponding episode, environment-step
  count, optimizer-update count, epsilon schedule position/value, seed, full
  configuration, validation metrics, and resume limitations.
- `training_reward.png`, `training_waiting_time.png`, `validation_waiting_time.png`,
  `dqn_loss.png`, and `epsilon_decay.png`. Training and loss plots show trailing
  25-episode averages with partial windows at the beginning. Empty training or
  loss series produce an explanatory figure instead of an error.

Regenerate plots from the saved CSV files:

```python
from src.training.plots import generate_plots
generate_plots("runs/phase5_diagnostic", window=25)
```

Load a saved model for evaluation using the existing agent API:

```python
from src.agent import DQNAgent
from src.training.validation import Validator

agent = DQNAgent()
agent.load("runs/phase5_diagnostic/best_model.pt")
metrics = Validator(workload_count=128, seed=1729).evaluate(agent)
```

These checkpoints support model evaluation and retain Adam state and optimizer
counts. They are **not full interrupted-session resume checkpoints**: replay
experiences are excluded, `DQNAgent.load()` starts with empty replay, and the
trainer does not restore workload RNG or orchestration state. Exact resume
equivalence is not claimed and no resume CLI is provided. Reproducibility is
tested on CPU with the same software, seed, and thread configuration; identical
results across hardware or library versions are not guaranteed. The trainer
temporarily enables deterministic PyTorch algorithms and sets the configured
CPU thread count, restoring both settings afterward.

A 20-episode smoke run usually ends before the 1000-transition warm-up, so zero
optimizer updates and an empty loss curve are expected. A 500-episode diagnostic
also covers only an early portion of the 100000-step epsilon schedule; it is a
pipeline/learning diagnostic, not evidence of convergence or baseline superiority.

## Stage 6.1: traditional scheduling baselines

`src/baselines/` implements FCFS, SRTF, and FIFO Round Robin using the existing
`CPUSchedulingEnv`. Every CPU time unit executes through `env.step(action)`;
the simulator owns timing, idle jumps, rewards, completion, switch/preemption
counts, and per-process metrics. The shared runner passes each policy only
arrived, unfinished simulator process objects, keyed by original workload index.
Policies receive neither the full workload nor future process records.

```python
from src.baselines import run_fcfs, run_srtf, run_round_robin

workload = (("P1", 0, 5), ("P2", 1, 2), ("P3", 3, 1))
fcfs = run_fcfs(workload)
srtf = run_srtf(workload)
rr = run_round_robin(workload, quantum=2)
print(rr["mean_waiting_time"])  # 7 / 3
```

Every call resets a clean simulator episode and leaves the input workload
unchanged. An optional `env=` permits reuse or inspection of an existing
simulator, which is reset before replay. All three functions return the same
dictionary fields:

| Field | Meaning |
| --- | --- |
| `algorithm` | FCFS, SRTF, or Round Robin |
| `timeline` | Tuple of simulator `(start, end, index, pid)` entries; idle entries have `index=pid=None` |
| `processes` | Simulator metrics keyed by PID: first start, completion, waiting, turnaround, response |
| `mean_waiting_time`, `mean_turnaround_time`, `mean_response_time` | Arithmetic means across processes, in CPU time units |
| `context_switches`, `preemptions` | Simulator counts, with no execution-time overhead |
| `total_reward` | Simulator reward; equals negative total process waiting time |
| `total_time`, `idle_time` | Final simulated time from zero, and total idle time |

The timeline preserves individual one-unit execution steps, plus simulator idle
intervals. Consecutive execution steps are combined below only for readability.
Returned metrics and timeline remain independent of later simulator resets.
Empty workloads return an empty timeline/metrics and zero means and counters.

- FCFS runs each selected process to completion. It chooses the earliest arrival,
  breaking ties by original workload index.
- SRTF reconsiders remaining time after every CPU unit. It continues the current
  process when tied for shortest; otherwise arrival time and original index break ties.
- Round Robin uses a FIFO queue and defaults to a two-unit quantum. Arrivals
  observed during each slice enter chronologically, with index ties. At expiry,
  new arrivals enter before the unfinished running process is requeued; existing
  waiters retain their position. Completion starts the next process with a fresh
  quantum. A sole ready process continues without a simulator context switch.
  Quantum must be a positive Python integer; Boolean/fractional values are rejected.

Manual workload schedules:

| Policy | Schedule | Mean waiting | Mean turnaround | Mean response | Switches | Preemptions | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCFS | P1 [0,5), P2 [5,7), P3 [7,8) | 8/3 | 16/3 | 8/3 | 2 | 0 | -8 |
| SRTF | P1 [0,1), P2 [1,3), P3 [3,4), P1 [4,8) | 1 | 11/3 | 0 | 3 | 1 | -3 |
| RR, q=2 | P1 [0,2), P2 [2,4), P1 [4,6), P3 [6,7), P1 [7,8) | 7/3 | 5 | 4/3 | 4 | 2 | -7 |

All three finish at time 8. Times are integer and intervals exclude their end.
These policies assume the simulator's single CPU, known bursts after arrival,
zero switching overhead, and no I/O blocking. SRTF minimizes average waiting
under these assumptions; it does not necessarily minimize response time or
switch count. The baseline package has no six-process adapter restriction.

`tests/test_baselines.py` checks the manual examples, ties, idle periods,
preemptions, quantum and queue behavior, completion, accounting, immutability,
repeatability, clean resets, and the shared result schema. It also uses 100
fresh synthetic unit-test workloads to check SRTF against FCFS and Round Robin
at quanta 1, 2, 5, and 20. It does not read reserved final test workloads.
Run the complete suite with `.\.venv\Scripts\python.exe -m pytest -q`.

Stage 6.1 adds no DQN comparison, convergence analysis, new training experiment,
hyperparameter tuning, final test evaluation, or final visualizations.

## Stage 6.2: initial validation benchmark and optimality gap

`src/benchmark/` evaluates the saved episode-5250 DQN best checkpoint against
FCFS, SRTF, and Round Robin (quantum 2). It reconstructs the **same** Phase 5
validation collection through `Validator`, using the saved configuration and
seed 1729. Reconstruction must yield exactly **128 workloads / 645 processes**.
It preserves input order, process indices, and the immutable definitions; there
is no new validation collection, slot permutation, or reserved test-set access.

```powershell
.\.venv\Scripts\python.exe -m src.benchmark.run --training-dir runs/phase5_full_6000 --output-dir runs/stage6_2_benchmark
```

Choose a fresh output directory; an existing directory is never overwritten.
The modules are `evaluation.py` (greedy replay and invariants), `analysis.py`
(paired gaps, aggregates, and slot diagnostics), `plots.py` (comparison charts),
and `run.py` (checkpoint/configuration verification, CLI, and export).
`tests/test_benchmark.py` covers isolation, determinism, workload identity,
weight preservation, means, reward consistency, optimality, exports, and charts.

The benchmark uses epsilon zero, evaluation mode, disabled gradients, and a
dedicated temporary action RNG. It restores the original mode/RNG and checks
both networks, optimizer counters, and replay state. It does not train or call
the optimizer. Baselines use their unchanged Stage 6.1 functions. All four
policies execute through the existing simulator or adapter.

Timing means divide summed per-process times by the number of processes;
switch/preemption means divide totals by workload count. Each workload checks
that SRTF waiting is no greater than any other policy, reward equals negative
total waiting, and CPU bursts/indices are preserved. DQN aggregate waiting,
turnaround, and mean reward must match both the checkpoint sidecar and the
episode-5250 validation CSV row. A full repeated replay of the same collection
must produce identical results and decision records. Protected source and
Phase 5 input files are checked by SHA-256 before and after benchmarking.

Required outputs:

- `baseline_comparison.csv`: four policy rows with process-weighted timing means,
  total times/reward, mean workload reward, and mean switches/preemptions.
- `per_workload_comparison.csv`: 128 paired rows with all four policies' metrics
  and DQN minus SRTF absolute/percentage waiting gaps.
- `per_process_results.csv`: 2580 rows with policy, zero-based workload ID,
  original process index, PID, arrival/burst, and simulator process metrics.
- `benchmark_summary.json`: aggregates, results by process count, optimality-gap
  statistics, best/worst examples and timelines, slot diagnostics, provenance,
  checks, and limitations.
- `waiting_time_comparison.png`, `turnaround_time_comparison.png`,
  `response_time_comparison.png`, `context_switches_comparison.png`, and
  `preemptions_comparison.png`.

Supporting exports are `by_process_count.csv` (all metrics for 4/5/6 processes),
`slot_bias.csv` (overall and stratified counts), `dqn_decision_diagnostics.csv`
(one row per ready process per DQN decision), and `validation_workloads.json`
(the exact reconstructed definitions in original order). Workload IDs start at
zero, matching the order in `Validator.workloads`.

The aggregate percentage gap is
`100 * (DQN total waiting - SRTF total waiting) / SRTF total waiting`.
Per-workload percentage gaps use the corresponding workload totals; when SRTF
waiting is zero, the percentage is zero if DQN also waits zero, otherwise null.
The process-weighted aggregate gap differs from an unweighted average of
per-workload mean gaps when process counts vary.

Slot analysis measures choices when a ready process is uniquely shortest,
excluding single-ready states, and choices among exactly equal six-feature
slot observations. Raw ready/selection rates are also recorded, stratified by
process count. These are observational, dependent decision counts with unequal
workload/feature distributions; they are not a causal slot effect or a formal
significance test. No alternative slot assignment is used.

These results use data previously used for model selection, so they are an
initial validation benchmark rather than an unbiased final test. Stage 6.2
does not tune hyperparameters, train a new model, implement Stage 6.3, or access
the reserved final workloads.
