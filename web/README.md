# CPU Scheduling Laboratory

A React website for demonstrating the project's CPU scheduler and measured results.

## Start

Run from this directory:

```powershell
npm install
npm run dev
```

Open http://127.0.0.1:5173/. Use `npm run build` for production output in `dist/`
and `npm run preview` to serve that output locally. Opening `index.html` directly
does not start the application.

## Demonstration sequence

1. Choose **Basic preemption**, select SRTF and click **Step +1**. Inspect the
   ready queue, reward, true preemptions and Gantt chart. Run to completion:
   mean waiting is 1.000, compared with FCFS 2.667 and Round Robin (q=2) 2.333.
2. Change an arrival or burst, add a process, then open **Compare** to see all
   four policies on your custom workload. DQN uses the actual exported weights.
3. Choose **Recorded workload 69** to replay exact saved checkpoint decisions:
   DQN mean waiting 10.200, total reward −51. Choose **201** for a case where
   DQN is worse than SRTF: DQN waiting 11.167, total reward −67.
4. Open **Project results** for the real 256-workload benchmark. Explain the
   augmented DQN's improvement over the original and its remaining gap to SRTF.
5. Open **Q-learning**, train the tabular agent and apply its learned policy.
   This small teaching experiment trains on the basic workload; it is separate
   from the project's saved DQN.

Workload JSON can be an array of `{pid, arrival_time, burst_time}` objects.
Recorded DQN trace JSON must match the schema described in
`../docs/dqn_trace_format.md`; imports are validated by replaying the schedule.

## Data and verification

`public/data/model.json` contains the actual 39→64→64→6 network weights and
checkpoint SHA-256. `workload_*.json` are PyTorch decision records. `results.json`
contains the saved experiment measurements and validation curves. Regenerate
them from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark.export_web_assets
```

`npm test` checks hand-computed schedules, Python trace parity for all three
examples, real Q-value parity, idle intervals, validation, and tabular learning.
Browser inference has small floating-point differences; recorded presets replay
the exact Python decisions. No Python server is required to demonstrate the site.
