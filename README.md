# Adaptive Preemptive CPU Scheduling Using Deep Q-Learning

A trained Deep Q-Network (DQN) selects a ready process for each simulated CPU
time unit. The project minimizes mean waiting time and compares the learned
scheduler with FCFS, SRTF, and Round Robin using the same verified simulator.

This single-core simulation uses 4–6 processes, arrivals at 0–8, bursts of 1–10,
preemption, no I/O blocking, and zero context-switch cost. Burst lengths are known
after arrival; future job properties are hidden from the agent.

## Repository structure

```text
RL_LAB_IA/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
├── tests/
├── docs/
├── exports/
├── web/
└── runs/
```

All paths below are relative to this repository root.

## Interactive React website

From the repository root:

```powershell
cd web
npm install
npm run dev
```

Open **http://127.0.0.1:5173/**. Node.js and npm are required; the website runs
entirely in the browser after its assets load.

- **Laboratory:** edit process IDs, arrivals and bursts; add/remove jobs; generate
  seeded workloads; play, pause, step, reset or manually choose the next job.
- **Compare:** view FCFS, SRTF, configurable Round Robin and the trained DQN on
  the same workload, with Gantt charts and final metrics. Export JSON or SVG.
- **Project results:** inspect the recorded 256-workload evaluation, paired
  augmentation results and measured validation curves.
- **Q-learning:** train a separate tabular teaching agent on the basic workload,
  inspect its Q-table, and run its learned policy.

Recorded presets **69**, **201** and **0** replay exact PyTorch decision traces.
Custom workloads execute the actual exported DQN weights in JavaScript; browser
floating-point calculations can differ slightly from PyTorch. Future job data
stays hidden in the DQN observation. The tabular teaching agent is separate from
the saved project DQN.

To refresh browser weights, traces and saved experiment assets, first return to
the repository root and run:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark.export_web_assets
```

Website verification and production build, from `web/`:

```powershell
npm test
npm run build
```

See [web/README.md](web/README.md) for a short demonstration sequence.

## Run the demonstration

Run all commands from the **repository root**, the directory containing
`main.py`, `README.md`, `src/`, and `runs/`. Complete the setup below once
before running the demonstration.

```powershell
.\.venv\Scripts\python.exe main.py
```

With an activated environment, use `python main.py`.

The default demo loads the saved augmented checkpoint and replays held-out
workload **69** with greedy actions (epsilon zero). It prints the workload, every
DQN action, a four-scheduler comparison, and individual process metrics. It writes:

- `runs/demo/demo_output.txt`: actual execution trace and comparison table.
- `runs/demo/demo_results.json`: metrics, unit timelines, checkpoint SHA-256.
- `runs/demo/demo_gantt.png`: actual executions for all four schedulers.

Expected mean waiting times in CPU units per process:

| DQN | SRTF | FCFS | Round Robin (quantum 2) |
|---:|---:|---:|---:|
| 10.200 | 10.200 | 15.800 | 19.400 |

DQN reward is **−51**, equal to negative total waiting. Existing evaluation code
checks completion, burst accounting, reward consistency, valid actions, and
unchanged network weights. Repeating the demo replaces only its three output files.

Run the counterexample to see a workload where DQN has greater waiting time:

```powershell
.\.venv\Scripts\python.exe main.py --workload-id 201 --output-dir runs/demo_counterexample
```

DQN waiting is **11.167** and SRTF waiting is **10.167** here. Workload IDs are
zero based, from 0 to 255. These examples illustrate behavior; aggregate claims
use the complete 256-workload evaluation.

Use `--checkpoint PATH` for another saved model and `--output-dir PATH` for a
different destination. Show all options with `python main.py --help`.

## Export real DQN decisions for an interactive demo

The importable JSON examples are in `exports/dqn_traces/`: workloads **0, 69,
and 201** from the final test collection. They contain actual observations,
action masks, Q-values, greedy actions, rewards, transitions, and full timelines
from the saved augmented DQN checkpoint. Each file also contains real FCFS,
SRTF, and Round Robin timelines and final metrics for the same workload.

Generate or refresh them from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark.export_traces --workload-ids 0 69 201
```

Use `--checkpoint PATH` for another model and `--output-dir PATH` for another
destination. Upload a `workload_ID.json` file to the interactive demo and use
its `decisions` for recorded playback. Label this **Recorded trained DQN
inference**. Changing the workload requires a new matching trace.

See [the JSON import contract](docs/dqn_trace_format.md) for the field names,
observation layout, action masking, and idle-time handling. The export performs
inference only and preserves the checkpoint and saved evaluation records.

## Setup

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
```

For macOS/Linux, create the environment with `python3 -m venv .venv`, activate it
with `source .venv/bin/activate`, and use `python` for the remaining commands.

The trained demonstration uses these repository-relative files:

- `runs/phase6_3b_controlled/augmented_10000/best_model.pt`
- `runs/phase6_4_final_test/test_workloads.json`

`runs/` is excluded from Git. A source-only clone needs the saved experiment
files at these paths before running the trained demonstration. The
checkpoint-specific test skips when the checkpoint is absent; simulator and
agent tests remain available.

## Reinforcement learning formulation

| Component | Definition |
|---|---|
| Environment | Discrete-time, single-core CPU scheduling simulator |
| Observation | 39 float32 values: six features per process slot, three global values |
| Actions | Six stable process indices; only arrived, unfinished processes are valid |
| Reward | Negative number of other ready processes waiting during the CPU unit |
| Policy | Masked epsilon-greedy training; masked greedy inference |
| Discount | 1.0 for a finite, undiscounted waiting-time objective |
| Termination | All processes complete; idle gaps advance automatically |

Each slot contains readiness, completion, original burst/10, remaining burst/10,
accumulated waiting/68, and a last-executed flag. Global values are time/68,
arrived count/6, and completed count/6. Unarrived and unused slots are zero.
Input order determines stable indices. Masks restrict action selection and
next-state bootstrapping. Terminal transitions have no bootstrap.

Every other ready process waits one unit per step, so
`episode return = -sum(process waiting times)`.
Waiting = completion − arrival − burst; turnaround = completion − arrival;
response = first execution − arrival. Timing means divide summed process times
by process count. Switch/preemption means divide totals by workload count.
The observation hides future arrivals and partially observes the simulator state.

## Agent and training

The network is `39 → 64 ReLU → 64 ReLU → 6`, producing Q-values.
Standard DQN uses uniform replay, a target network, Adam, Smooth L1 loss, and
gradient clipping. Nonterminal targets are
`reward + gamma * max_valid Q_target(next_observation, action)`.
Terminal targets equal reward. Only the policy network receives gradients.

| Setting | Value |
|---|---|
| Learning rate / discount | 0.0005 / 1.0 |
| Batch size / replay capacity | 64 / 20,000 |
| Replay warm-up | 1,000 transitions |
| Optimization | Every four CPU execution steps |
| Target synchronization | Every 250 optimizer updates |
| Gradient norm limit | 10 |
| Exploration | Linear 1.0 → 0.05 over 100,000 steps |
| Validation | Greedy evaluation every 250 episodes |
| Validation collection | 128 workloads, 645 processes; seed 1729 |
| Controlled experiment | 10,000 episodes per model; training seed 42 |
| Selected augmented model | Episode 9,750; validation waiting 6.113178 |

The saved experiment investigated slot sensitivity by consistently permuting
current/next process blocks, action indices, and masks in replay transitions.
Rewards, global features, terminal flags, and padding were retained. Original
and augmented runs used identical workload streams and training budgets.

The current trainer implements **standard replay DQN**. The separate controlled
augmentation experiment driver is not included in this source tree. Its frozen
checkpoints, configuration, histories, and diagnostics are supplied as evidence.
The demonstration reproduces inference from those checkpoints.

Run a short standard-training check in a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --smoke --seed 42 --output-dir runs/new_smoke
```

The 20-episode smoke run ends before replay warm-up and makes no optimizer updates;
it verifies execution and artifact generation, not learning.

Run fresh standard DQN training:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --episodes 6000 --seed 42 --output-dir runs/new_training
```

Training writes CSV histories, configuration, best/final checkpoints, and plots.
Choose a fresh directory. Best checkpoints minimize validation waiting.
Checkpoints support inference; exact training resumption is unsupported because
replay experiences and trainer/workload RNG state are not saved.

## Held-out results

The frozen final collection contains **256 workloads / 1,277 processes**, seed
**6401729**. All policies replay identical jobs and indices. Models were selected
using validation before final test evaluation.

| Scheduler | Waiting | Turnaround | Response | Waiting gap to SRTF |
|---|---:|---:|---:|---:|
| SRTF | 5.826 | 11.255 | 5.132 | 0.00% |
| Augmented DQN, 10k | 5.965 | 11.394 | 4.713 | 2.38% |
| Original DQN, 10k | 6.118 | 11.547 | 4.672 | 5.01% |
| Earlier DQN, 6k | 6.518 | 11.947 | 4.882 | 11.87% |
| FCFS | 8.429 | 13.858 | 8.429 | 44.68% |
| Round Robin, quantum 2 | 11.218 | 16.648 | 2.452 | 92.55% |

Times are CPU units per process. Augmentation reduced waiting by **0.153485**
against the equal-budget original, with paired 95% workload-bootstrap interval
**[−0.207905, −0.102263]**. It improved 90 workloads, worsened 36, and tied 130.
The augmented model tied SRTF on 169/256 workloads.

SRTF remains the optimal waiting-time reference under the known-burst,
preemptive, zero-overhead assumptions. Round Robin has lower response time but
more waiting and switching. The interval measures workload uncertainty for fixed
checkpoints, not variation across independent training seeds. Falling TD loss
and fluctuating validation curves do not prove convergence.

## Source and saved evidence

| Location | Purpose |
|---|---|
| `main.py` | Saved-checkpoint demonstration and actual Gantt chart |
| `src/env/` | Simulator, generator, observation adapter |
| `src/agent/` | Network, replay, masked DQN, checkpoint loading |
| `src/training/` | Standard DQN training and fixed validation |
| `src/baselines/` | FCFS, SRTF, Round Robin, shared simulator runner |
| `src/benchmark/` | Isolated greedy evaluation and validation benchmark |
| `src/benchmark/export_traces.py` | Actual DQN transition and Q-value JSON export |
| `tests/`, `src/env/tests/` | Implementation checks and demo regression |
| `exports/dqn_traces/` | Importable recorded DQN examples |
| `docs/dqn_trace_format.md` | JSON contract for interactive trace playback |
| `runs/phase6_3b_controlled/` | Controlled histories and selected checkpoints |
| `runs/phase6_4_final_test/` | Frozen workloads, metrics, intervals, verification |
| `runs/phase6_5_final_visualizations/` | Evaluation figures and source manifest |
| `runs/demo/` | Actual demonstration output |

Historical artifact directory names are retained to match saved provenance.
Old absolute paths in historical manifests identify the original experiment
machine; the demonstration resolves inputs from this project.

## Limitations and viva preparation

Results apply to a narrow synthetic distribution, one CPU, known arrived bursts,
zero switch cost, and one paired training seed. There is no distribution-shift
test. The network supports at most six processes. Replay augmentation improves
measured slot robustness without enforcing exact permutation equivariance.

Be ready to explain the reward/waiting identity, the 39 features, action masks,
the DQN target, replay and target networks, exploration, greedy evaluation,
checkpoint selection, SRTF's role, both demo cases, and the bootstrap limitation.

Source repository: https://github.com/Omsiddh/RL_LAB_IA
