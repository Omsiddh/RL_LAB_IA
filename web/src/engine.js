export const ALGORITHMS = ['FCFS', 'SRTF', 'Round Robin', 'DQN', 'Manual', 'Q-learning'];
export const COLORS = ['#2866ad', '#c26c26', '#25816c', '#865ca6', '#bc4665', '#77702a'];
export const BASIC = [
  { pid: 'P1', arrival_time: 0, burst_time: 5 },
  { pid: 'P2', arrival_time: 1, burst_time: 2 },
  { pid: 'P3', arrival_time: 3, burst_time: 1 },
];
export const PRESETS = {
  basic: BASIC,
  simultaneous: [8, 2, 5, 1].map((burst, i) => ({ pid: `P${i + 1}`, arrival_time: 0, burst_time: burst })),
  dynamic: [
    { pid: 'P1', arrival_time: 0, burst_time: 10 },
    { pid: 'P2', arrival_time: 2, burst_time: 2 },
    { pid: 'P3', arrival_time: 4, burst_time: 1 },
    { pid: 'P4', arrival_time: 6, burst_time: 3 },
  ],
  contention: [9, 2, 8, 3, 6, 1].map((burst, i) => ({ pid: `P${i + 1}`, arrival_time: i % 3, burst_time: burst })),
};

export function validateWorkload(workload) {
  if (!Array.isArray(workload) || workload.length < 1 || workload.length > 6) return 'Use between one and six processes.';
  const ids = new Set();
  for (const p of workload) {
    if (typeof p.pid !== 'string' || !p.pid.trim() || p.pid.length > 24 || ids.has(p.pid)) return 'Process IDs must be unique, nonempty strings of at most 24 characters.';
    if (!Number.isInteger(p.arrival_time) || p.arrival_time < 0 || p.arrival_time > 8) return 'Arrival times must be integers from 0 to 8.';
    if (!Number.isInteger(p.burst_time) || p.burst_time < 1 || p.burst_time > 10) return 'CPU bursts must be integers from 1 to 10.';
    ids.add(p.pid);
  }
  return null;
}

export function sameWorkload(a, b) {
  return a.length === b.length && a.every((p, i) => p.pid === b[i].pid && p.arrival_time === b[i].arrival_time && p.burst_time === b[i].burst_time);
}

export function rng(seed) {
  let state = Number(seed) >>> 0;
  return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
}

export function randomWorkload(count, seed) {
  const random = rng(seed);
  return Array.from({ length: count }, (_, i) => ({ pid: `P${i + 1}`, arrival_time: Math.floor(random() * 9), burst_time: 1 + Math.floor(random() * 10) }));
}

export function readyIndices(state) {
  return state.processes.flatMap((p, i) => p.arrival_time <= state.time && p.remaining > 0 ? [i] : []);
}

function skipIdle(state) {
  if (state.processes.every(p => p.remaining === 0) || readyIndices(state).length) return null;
  const end = Math.min(...state.processes.filter(p => p.remaining > 0).map(p => p.arrival_time));
  const interval = { start: state.time, end, process_index: null, pid: null };
  state.timeline.push(interval);
  state.idle += end - state.time;
  state.time = end;
  state.previous = null;
  return interval;
}

export function initialState(workload) {
  const error = validateWorkload(workload);
  if (error) throw new Error(error);
  const state = {
    time: 0, processes: workload.map(p => ({ ...p, remaining: p.burst_time, waiting: 0, first: null, completion: null })),
    timeline: [], decisions: [], reward: 0, switches: 0, preemptions: 0, idle: 0,
    previous: null, lastExecuted: null, current: null, queue: [], admitted: [], quantumUsed: 0, done: false,
  };
  skipIdle(state);
  return state;
}

export function observation(state) {
  const values = Array(39).fill(0);
  let arrived = 0, completed = 0;
  state.processes.forEach((p, i) => {
    if (p.arrival_time > state.time) return;
    arrived++;
    if (!p.remaining) completed++;
    const elapsed = (p.completion ?? state.time) - p.arrival_time;
    values.splice(i * 6, 6, Number(p.remaining > 0), Number(p.remaining === 0),
      Math.fround(p.burst_time / 10), Math.fround(p.remaining / 10),
      Math.fround((elapsed - (p.burst_time - p.remaining)) / 68), Number(state.lastExecuted === i));
  });
  values[36] = Math.fround(state.time / 68);
  values[37] = Math.fround(arrived / 6);
  values[38] = Math.fround(completed / 6);
  return values;
}

export function predictQ(model, input) {
  if (!model?.layers || model.source !== 'exported_pytorch_checkpoint') throw new Error('Load the actual trained model weights first.');
  let values = input;
  model.layers.forEach((layer, index) => {
    values = layer.weight.map((weights, row) => {
      let total = layer.bias[row];
      for (let column = 0; column < weights.length; column++) total += weights[column] * values[column];
      return Math.fround(index < model.layers.length - 1 ? Math.max(0, total) : total);
    });
  });
  return values;
}

export function greedy(valid, values) {
  return valid.reduce((best, index) => values[index] > values[best] ? index : best, valid[0]);
}

export function qState(state) {
  return `${state.time}|${state.processes.map(p => p.arrival_time > state.time ? '_' : p.remaining).join(',')}`;
}

export function step(state, algorithm, { quantum = 2, model, trace, manual, table } = {}) {
  if (state.done) return state;
  const next = { ...state, processes: state.processes.map(p => ({ ...p })), timeline: [...state.timeline], decisions: [...state.decisions], queue: [...state.queue], admitted: [...state.admitted] };
  const ready = readyIndices(next);
  const obs = observation(next);
  const mask = Array.from({ length: 6 }, (_, i) => ready.includes(i));
  let action, qValues = null, reason, queueBefore = [];
  const arrivalOrder = [...ready].sort((a, b) => next.processes[a].arrival_time - next.processes[b].arrival_time || a - b);
  if (algorithm === 'FCFS') {
    action = ready.includes(next.current) ? next.current : arrivalOrder[0];
    reason = ready.includes(next.current) ? 'Continue the running process until completion.' : 'Dispatch the earliest-arrived ready process; ties use stable slot order.';
  } else if (algorithm === 'SRTF') {
    const minimum = Math.min(...ready.map(i => next.processes[i].remaining));
    action = ready.includes(next.current) && next.processes[next.current].remaining === minimum ? next.current
      : arrivalOrder.find(i => next.processes[i].remaining === minimum);
    reason = `Choose the shortest remaining ready job (${minimum} CPU units). Continue the current job when tied.`;
  } else if (algorithm === 'Round Robin') {
    const newcomers = arrivalOrder.filter(i => !next.admitted.includes(i));
    next.queue.push(...newcomers);
    next.admitted.push(...newcomers);
    if (ready.includes(next.current) && next.quantumUsed < quantum) action = next.current;
    else {
      if (ready.includes(next.current)) next.queue.push(next.current);
      next.current = next.queue.shift();
      next.quantumUsed = 0;
      action = next.current;
    }
    queueBefore = [...next.queue];
    reason = `FIFO dispatch; quantum ${quantum}, executing unit ${next.quantumUsed + 1}. New arrivals enter before quantum-expiry requeue.`;
  } else if (algorithm === 'DQN') {
    if (trace) {
      const record = trace.decisions[next.decisions.length];
      if (!record || record.time !== next.time) throw new Error('The imported trace does not match this simulation.');
      action = record.selected_action;
      qValues = record.q_values;
      reason = 'Recorded greedy decision from the actual PyTorch checkpoint.';
    } else {
      qValues = predictQ(model, obs);
      action = greedy(ready, qValues);
      reason = 'Run the exported trained 39 → 64 → 64 → 6 network and select the highest valid Q-value.';
    }
  } else if (algorithm === 'Manual') {
    action = manual;
    reason = 'You chose this ready process for one CPU unit.';
  } else if (algorithm === 'Q-learning') {
    qValues = table?.[qState(next)] ?? Array(6).fill(0);
    action = greedy(ready, qValues);
    reason = table?.[qState(next)] ? 'Greedy action from the learned Q-table row.' : 'Unseen state: zero-initialized Q-values; ties use the first valid slot.';
  } else throw new Error('Unknown scheduling algorithm.');
  if (!ready.includes(action)) throw new Error('Choose an arrived, unfinished process.');
  const p = next.processes[action];
  const contextSwitch = next.previous !== null && next.previous !== action;
  const preemption = contextSwitch && next.processes[next.previous].remaining > 0;
  const reward = -(ready.length - 1);
  ready.forEach(i => { if (i !== action) next.processes[i].waiting++; });
  if (p.first === null) p.first = next.time;
  const execution = { start: next.time, end: next.time + 1, process_index: action, pid: p.pid };
  p.remaining--;
  next.time++;
  if (!p.remaining) p.completion = next.time;
  next.switches += Number(contextSwitch);
  next.preemptions += Number(preemption);
  next.reward += reward;
  next.timeline.push(execution);
  next.previous = next.lastExecuted = action;
  next.current = p.remaining ? action : null;
  if (algorithm === 'Round Robin') next.quantumUsed = p.remaining ? next.quantumUsed + 1 : 0;
  next.done = next.processes.every(process => process.remaining === 0);
  const idleAfter = skipIdle(next);
  next.decisions.push({ step: next.decisions.length, time: execution.start, observation: obs, action_mask: mask,
    q_values: qValues, selected_action: action, selected_pid: p.pid, reward, cumulative_reward: next.reward,
    execution, context_switch: contextSwitch, preemption, process_completed: !p.remaining,
    idle_interval_after: idleAfter, next_time: next.time, done: next.done, reason, queue: queueBefore, quantum_used: next.quantumUsed });
  return next;
}

export function run(workload, algorithm, options = {}) {
  let state = initialState(workload);
  while (!state.done) state = step(state, algorithm, options);
  const totalWaiting = state.processes.reduce((sum, p) => sum + p.waiting, 0);
  if (state.reward !== -totalWaiting) throw new Error('Reward accounting failed.');
  return state;
}

export function metrics(state) {
  const count = state.processes.length;
  const finished = state.processes.filter(p => p.completion !== null);
  const started = state.processes.filter(p => p.first !== null);
  const mean = (items, selector) => items.length ? items.reduce((sum, p) => sum + selector(p), 0) / items.length : null;
  return {
    waiting: state.processes.reduce((sum, p) => sum + p.waiting, 0) / count,
    turnaround: mean(finished, p => p.completion - p.arrival_time),
    response: mean(started, p => p.first - p.arrival_time),
    completed: finished.length, switches: state.switches, preemptions: state.preemptions,
    reward: state.reward, utilization: state.time ? (state.time - state.idle) / state.time * 100 : 0,
  };
}

export function mergeTimeline(timeline) {
  const merged = [];
  for (const interval of timeline) {
    const last = merged.at(-1);
    if (last && last.process_index === interval.process_index && last.end === interval.start) last.end = interval.end;
    else merged.push({ ...interval });
  }
  return merged;
}

export function validateTrace(payload) {
  if (payload?.schema_version !== 1 || payload.trace_type !== 'cpu_scheduling_dqn') throw new Error('Import a version 1 DQN trace exported by this project.');
  const error = validateWorkload(payload.workload);
  if (error) throw new Error(error);
  if (!Array.isArray(payload.decisions) || payload.decisions.length !== payload.workload.reduce((sum, p) => sum + p.burst_time, 0)) throw new Error('Trace must contain one decision for every CPU execution unit.');
  let state = initialState(payload.workload);
  for (const [index, record] of payload.decisions.entries()) {
    const obs = observation(state), ready = readyIndices(state);
    if (record.step !== index) throw new Error('Trace decision indices must be consecutive.');
    if (!Array.isArray(record.observation) || record.observation.length !== 39 || !record.observation.every((v, i) => Number.isFinite(v) && Math.abs(v - obs[i]) < 1e-6)) throw new Error('Trace observation does not match the workload.');
    if (!Array.isArray(record.q_values) || record.q_values.length !== 6 || !record.q_values.every(Number.isFinite)) throw new Error('Trace Q-values must be six finite numbers.');
    if (!Array.isArray(record.action_mask) || record.action_mask.length !== 6 || !record.action_mask.every((valid, i) => valid === ready.includes(i))) throw new Error('Trace action mask is invalid.');
    if (record.selected_action !== greedy(ready, record.q_values)) throw new Error('Trace action is not a masked greedy decision.');
    state = step(state, 'DQN', { trace: payload });
    const actual = state.decisions.at(-1);
    for (const key of ['reward', 'next_time', 'done', 'context_switch', 'preemption', 'selected_pid']) {
      if (record[key] !== actual[key]) throw new Error(`Trace ${key} is inconsistent.`);
    }
    if (!record.execution || ['start', 'end', 'process_index', 'pid'].some(key => record.execution[key] !== actual.execution[key])) throw new Error('Trace execution interval is inconsistent.');
    if (!Array.isArray(record.next_action_mask) || record.next_action_mask.length !== 6 || !record.next_action_mask.every((valid, i) => valid === readyIndices(state).includes(i))) throw new Error('Trace next-state action mask is inconsistent.');
    const nextObs = observation(state);
    if (!Array.isArray(record.next_observation) || record.next_observation.length !== 39 || !record.next_observation.every((value, i) => Number.isFinite(value) && Math.abs(value - nextObs[i]) < 1e-6)) throw new Error('Trace next observation is inconsistent.');
    const actualIdle = actual.idle_interval_after;
    if ((record.idle_interval_after === null) !== (actualIdle === null) || (actualIdle && ['start', 'end', 'process_index', 'pid'].some(key => record.idle_interval_after[key] !== actualIdle[key]))) throw new Error('Trace idle interval is inconsistent.');
  }
  if (!state.done || payload.total_reward !== state.reward || Math.abs(payload.mean_waiting_time - metrics(state).waiting) > 1e-8) throw new Error('Trace final metrics are inconsistent.');
  if (!Array.isArray(payload.timeline) || payload.timeline.length !== state.timeline.length || !payload.timeline.every((entry, i) => ['start', 'end', 'process_index', 'pid'].every(key => entry[key] === state.timeline[i][key]))) throw new Error('Trace timeline is inconsistent.');
  return payload;
}

export async function trainTabular({ episodes, alpha, gamma, epsilon, seed, onProgress }) {
  const table = {}, random = rng(seed), curve = [];
  const validate = episode => { const score = metrics(run(BASIC, 'Q-learning', { table })).waiting; curve.push({ episode, waiting: score }); };
  validate(0);
  for (let episode = 1; episode <= episodes; episode++) {
    let state = initialState(BASIC);
    const exploration = Math.max(0.05, epsilon * (1 - episode / episodes));
    while (!state.done) {
      const key = qState(state), valid = readyIndices(state);
      const row = table[key] ??= Array(6).fill(0);
      const action = random() < exploration ? valid[Math.floor(random() * valid.length)] : greedy(valid, row);
      const next = step(state, 'Manual', { manual: action });
      const nextKey = qState(next), nextValid = readyIndices(next);
      const nextRow = table[nextKey] ?? Array(6).fill(0);
      const target = next.decisions.at(-1).reward + (next.done ? 0 : gamma * nextRow[greedy(nextValid, nextRow)]);
      row[action] += alpha * (target - row[action]);
      state = next;
    }
    if (episode % 50 === 0 || episode === episodes) {
      validate(episode); onProgress?.({ episode, episodes });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  return { table, curve, episodes, alpha, gamma, epsilon, seed, workload: BASIC };
}
