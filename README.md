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
