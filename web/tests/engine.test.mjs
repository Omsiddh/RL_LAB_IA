import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { BASIC, initialState, metrics, observation, predictQ, randomWorkload, readyIndices, run, step, trainTabular, validateTrace, validateWorkload } from '../src/engine.js';

const data = name => JSON.parse(fs.readFileSync(new URL(`../public/data/${name}.json`, import.meta.url), 'utf8'));
const model = data('model');
const keys = ['start', 'end', 'process_index', 'pid'];
const close = (actual, expected, tolerance = 1e-7) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);

test('hand-computed basic baseline schedules and reward identity', () => {
  const expected = { FCFS: [8 / 3, -8], SRTF: [1, -3], 'Round Robin': [7 / 3, -7] };
  for (const [algorithm, [waiting, reward]] of Object.entries(expected)) {
    const state = run(BASIC, algorithm, { quantum: 2 });
    close(metrics(state).waiting, waiting);
    assert.equal(state.reward, reward);
    assert.equal(state.time, 8);
    assert.equal(state.timeline.filter(e => e.pid !== null).length, 8);
  }
});

test('browser baselines reproduce all three Python example timelines', () => {
  for (const id of [0, 69, 201]) {
    const trace = data(`workload_${id}`);
    validateTrace(trace);
    for (const algorithm of ['FCFS', 'SRTF', 'Round Robin']) {
      const state = run(trace.workload, algorithm, { quantum: 2 });
      assert.deepEqual(state.timeline, trace.baselines[algorithm].timeline);
      close(metrics(state).waiting, trace.baselines[algorithm].mean_waiting_time);
      assert.equal(state.switches, trace.baselines[algorithm].context_switches);
      assert.equal(state.preemptions, trace.baselines[algorithm].preemptions);
    }
  }
});

test('exported trained network predictions match PyTorch within float tolerance', () => {
  let largest = 0;
  for (const id of [0, 69, 201]) {
    const trace = data(`workload_${id}`);
    assert.equal(model.checkpoint_sha256, trace.model.checkpoint_sha256);
    for (const decision of trace.decisions) {
      const q = predictQ(model, decision.observation);
      q.forEach((value, i) => { largest = Math.max(largest, Math.abs(value - decision.q_values[i])); close(value, decision.q_values[i], 0.0001); });
    }
    const state = run(trace.workload, 'DQN', { model });
    assert.deepEqual(state.timeline, trace.timeline);
    assert.equal(state.reward, trace.total_reward);
  }
  console.log(`Maximum browser/PyTorch Q-value error: ${largest}`);
});

test('recorded DQN playback reproduces actual choices, rewards, and final metrics', () => {
  for (const id of [69, 201]) {
    const trace = data(`workload_${id}`), state = run(trace.workload, 'DQN', { trace });
    assert.deepEqual(state.timeline, trace.timeline);
    assert.equal(state.reward, trace.total_reward);
    close(metrics(state).waiting, trace.mean_waiting_time);
    state.decisions.forEach((decision, i) => {
      assert.equal(decision.selected_action, trace.decisions[i].selected_action);
      assert.deepEqual(decision.q_values, trace.decisions[i].q_values);
    });
  }
});

test('one step executes one busy unit and handles initial and subsequent idle gaps', () => {
  const workload = [{ pid: 'A', arrival_time: 2, burst_time: 1 }, { pid: 'B', arrival_time: 5, burst_time: 1 }];
  const start = initialState(workload), next = step(start, 'FCFS');
  assert.equal(start.time, 2);
  assert.equal(start.decisions.length, 0);
  assert.equal(next.time, 5);
  assert.deepEqual(next.decisions[0].execution, { start: 2, end: 3, process_index: 0, pid: 'A' });
  assert.deepEqual(next.decisions[0].idle_interval_after, { start: 3, end: 5, process_index: null, pid: null });
  assert.equal(next.switches, 0);
  const final = step(next, 'FCFS');
  assert.equal(final.switches, 0);
  assert.equal(final.reward, 0);
  assert.equal(metrics(final).waiting, 0);
});

test('invalid manual choices and edited trace content are rejected', () => {
  assert.throws(() => step(initialState(BASIC), 'Manual', { manual: 2 }), /arrived/);
  const invalid = data('workload_69');
  invalid.decisions[0].reward--;
  assert.throws(() => validateTrace(invalid), /reward/);
  const masked = data('workload_69');
  masked.decisions[0].action_mask[0] = true;
  assert.throws(() => validateTrace(masked), /mask/);
  const terminal = data('workload_201');
  terminal.decisions.at(-1).next_action_mask[0] = true;
  assert.throws(() => validateTrace(terminal), /next-state action mask/);
  const interval = data('workload_69');
  interval.timeline[0].end++;
  assert.throws(() => validateTrace(interval), /timeline/);
});

test('future process slots are zero and terminal observations are valid', () => {
  const state = initialState(BASIC), obs = observation(state);
  assert.equal(obs.length, 39);
  assert.deepEqual(obs.slice(6, 36), Array(30).fill(0));
  assert.deepEqual(readyIndices(state), [0]);
  const final = run(BASIC, 'SRTF');
  assert.deepEqual(readyIndices(final), []);
  close(observation(final)[38], Math.fround(3 / 6));
});

test('custom seeded workloads are repeatable and validated', () => {
  assert.deepEqual(randomWorkload(6, 42), randomWorkload(6, 42));
  assert.equal(validateWorkload(randomWorkload(6, 42)), null);
  assert.notEqual(validateWorkload([{ pid: 'X', arrival_time: 0, burst_time: 0 }]), null);
  assert.notEqual(validateWorkload([{ pid: 'X', arrival_time: 0, burst_time: 1 }, { pid: 'X', arrival_time: 1, burst_time: 2 }]), null);
});

test('real tabular Bellman updates learn the fixed basic example', async () => {
  const learned = await trainTabular({ episodes: 2000, alpha: 0.2, gamma: 1, epsilon: 1, seed: 42 });
  assert.ok(Object.keys(learned.table).length > 5);
  assert.ok(Object.values(learned.table).some(row => row.some(value => value < 0)));
  assert.ok(learned.curve.at(-1).waiting < learned.curve[0].waiting);
  close(learned.curve.at(-1).waiting, 1);
  assert.equal(learned.episodes, 2000);
});
