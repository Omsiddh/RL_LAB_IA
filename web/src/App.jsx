
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ALGORITHMS, BASIC, COLORS, PRESETS, initialState, mergeTimeline, metrics, observation, randomWorkload, readyIndices, run, sameWorkload, step, trainTabular, validateTrace, validateWorkload } from './engine.js';

const number = (value, precision = 3) => value == null ? '—' : Number(value).toFixed(precision);
const copy = workload => workload.map(({ pid, arrival_time, burst_time }) => ({ pid, arrival_time, burst_time }));
const definitions = ['Ready', 'Completed', 'Burst / 10', 'Remaining / 10', 'Waiting / 68', 'Last run'];

function download(name, data, type = 'application/json') {
  const content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Section({ title, detail, children, className = '' }) {
  return <section className={`panel ${className}`}><div className="panel-heading"><h2>{title}</h2>{detail && <span>{detail}</span>}</div>{children}</section>;
}

function Timeline({ state, label, onInspect, compact = false }) {
  const end = Math.max(8, state?.time ?? 0), scale = 25, left = 12;
  const width = Math.max(450, end * scale + left * 2);
  const intervals = mergeTimeline(state?.timeline ?? []);
  const ticks = Array.from({ length: Math.floor(end / (end > 32 ? 4 : 2)) + 1 }, (_, i) => i * (end > 32 ? 4 : 2));
  const pattern = `idle-${label.replace(/[^a-z0-9]/gi, '')}`;
  return <div className={`timeline-scroll ${compact ? 'compact' : ''}`}>
    <svg className="timeline" width={width} height={compact ? 69 : 94} viewBox={`0 0 ${width} ${compact ? 69 : 94}`} role="img" aria-label={`${label} Gantt chart`}>
      <defs><pattern id={pattern} width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#e8e9eb" /><path d="M-1 1L7 9M1-1L9 7" stroke="#bdc4cc" strokeWidth="1" /></pattern></defs>
      {ticks.map(t => <g key={t}><line x1={left + t * scale} x2={left + t * scale} y1="7" y2="45" stroke="#e1e5e9" /><text x={left + t * scale} y="62" textAnchor="middle" fontSize="11" fill="#677382" fontFamily="monospace">{t}</text></g>)}
      {intervals.map((segment, index) => <g key={index} className={onInspect ? 'gantt-segment' : ''} onClick={() => onInspect?.(segment)} tabIndex={onInspect ? 0 : undefined} onKeyDown={event => { if (event.key === 'Enter') onInspect?.(segment); }} role={onInspect ? 'button' : undefined} aria-label={`${segment.pid ?? 'Idle'} from ${segment.start} to ${segment.end}`}>
        <title>{segment.pid ?? 'Idle'} [{segment.start}, {segment.end}) · {segment.end - segment.start} CPU units</title>
        <rect x={left + segment.start * scale} y="12" width={(segment.end - segment.start) * scale} height="30" fill={segment.process_index === null ? `url(#${pattern})` : COLORS[segment.process_index]} stroke="#fff" strokeWidth="1" rx="1" />
        {(segment.end - segment.start) > 1 && <text x={left + (segment.start + segment.end) / 2 * scale} y="31" textAnchor="middle" fill={segment.pid === null ? '#526170' : '#fff'} fontSize="11" fontFamily="monospace">{segment.pid ?? 'IDLE'}</text>}
      </g>)}
      {state?.decisions?.filter(d => d.preemption).map(d => <path key={d.time} d={`M${left + d.time * scale - 4} 4h8l-4 6z`} fill="#252e37"><title>Preemption at t={d.time}</title></path>)}
      {!intervals.length && <text x="18" y="31" fill="#81909e" fontSize="12">Step or play to execute the first CPU unit.</text>}
      {!compact && <text x={width / 2} y="85" textAnchor="middle" fill="#677382" fontSize="11">Simulated CPU time · intervals include start and exclude end</text>}
    </svg>
  </div>;
}

function Curve({ series, labels, colors = ['#876447', '#2866ad'] }) {
  const all = series.flat();
  if (!all.length) return <p className="empty">Train the agent to obtain a measured learning curve.</p>;
  const maxX = Math.max(1, ...all.map(p => p.episode)), minY = Math.floor(Math.min(...all.map(p => p.waiting)) * 2) / 2;
  const maxY = Math.max(minY + 1, Math.ceil(Math.max(...all.map(p => p.waiting)) * 2) / 2);
  const x = value => 47 + value / maxX * 560, y = value => 183 - (value - minY) / (maxY - minY) * 153;
  return <><svg className="curve" viewBox="0 0 640 228" role="img" aria-label="Greedy validation waiting versus training episodes">
    {Array.from({ length: 5 }, (_, i) => minY + (maxY - minY) * i / 4).map(value => <g key={value}><line x1="47" x2="607" y1={y(value)} y2={y(value)} stroke="#e4e8ed" /><text x="40" y={y(value) + 4} textAnchor="end" fill="#667483" fontSize="11">{number(value, 1)}</text></g>)}
    {Array.from({ length: 5 }, (_, i) => maxX * i / 4).map(value => <text key={value} x={x(value)} y="204" textAnchor="middle" fill="#667483" fontSize="11">{Math.round(value)}</text>)}
    {series.map((points, i) => <g key={i}><polyline points={points.map(p => `${x(p.episode)},${y(p.waiting)}`).join(' ')} fill="none" stroke={colors[i]} strokeWidth="2" />{points.map(p => <circle key={p.episode} cx={x(p.episode)} cy={y(p.waiting)} r="2.5" fill={colors[i]}><title>Episode {p.episode}: waiting {number(p.waiting, 6)}</title></circle>)}</g>)}
    <text x="327" y="226" textAnchor="middle" fontSize="11" fill="#667483">Training episodes</text>
  </svg><div className="legend">{labels.map((label, i) => <span key={label}><i style={{ background: colors[i] }} />{label}</span>)}</div></>;
}

function Legend({ workload }) {
  return <div className="legend">{workload.map((p, i) => <span key={i}><i style={{ background: COLORS[i] }} />{p.pid || `Slot ${i}`}</span>)}<span><i className="idle-key" />Idle</span><span>▼ true preemption</span></div>;
}

function ProcessEditor({ workload, simulation, preset, setPreset, change, add, remove, count, setCount, seed, setSeed, generate, importWorkload }) {
  const inputRef = useRef(null);
  return <Section title="Workload" detail={`${workload.length} / 6 slots`}>
    <label className="field-label">Preset<select aria-label="Workload preset" value={preset} onChange={event => setPreset(event.target.value)}>
      <option value="basic">Basic preemption</option><option value="simultaneous">Simultaneous arrivals</option><option value="dynamic">Dynamic arrivals</option><option value="contention">Heavy contention</option>
      <option value="69">Recorded workload 69</option><option value="201">Recorded workload 201</option><option value="0">Recorded workload 0</option><option value="custom">Custom workload</option><option value="imported">Imported trace / workload</option>
    </select></label>
    <div className="table-scroll"><table className="editor"><thead><tr><th>Process</th><th>Arrival</th><th>Burst</th><th>Left</th><th /></tr></thead><tbody>{workload.map((p, i) => <tr key={i}>
      <td><div className="pid-input"><i style={{ background: COLORS[i] }} /><input aria-label={`Process ${i + 1} ID`} value={p.pid} maxLength={24} onChange={event => change(i, 'pid', event.target.value)} /></div></td>
      <td><input aria-label={`Process ${i + 1} arrival`} type="number" min="0" max="8" value={p.arrival_time} onChange={event => change(i, 'arrival_time', Number(event.target.value))} /></td>
      <td><input aria-label={`Process ${i + 1} burst`} type="number" min="1" max="10" value={p.burst_time} onChange={event => change(i, 'burst_time', Number(event.target.value))} /></td>
      <td className="mono">{simulation?.processes[i]?.remaining ?? p.burst_time}</td>
      <td><button className="remove" aria-label={`Remove process ${i + 1}`} disabled={workload.length === 1} onClick={() => remove(i)}>×</button></td>
    </tr>)}</tbody></table></div>
    <div className="editor-actions"><button disabled={workload.length === 6} onClick={add}>+ Add process</button><button onClick={() => download('workload.json', workload)}>Export JSON</button></div>
    <div className="generator"><label>Jobs<input aria-label="Random workload process count" type="number" min="1" max="6" value={count} onChange={event => setCount(Number(event.target.value))} /></label><label>Seed<input aria-label="Random workload seed" type="number" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label><button disabled={!Number.isInteger(count) || count < 1 || count > 6 || !Number.isInteger(seed)} onClick={generate}>Generate</button></div>
    <input ref={inputRef} className="file-input" type="file" accept=".json,application/json" onChange={importWorkload} /><button className="full-width" onClick={() => inputRef.current.click()}>Import workload JSON</button>
    <p className="hint">Arrivals 0–8 · bursts 1–10 · one CPU. Editing a task resets the simulation.</p>
  </Section>;
}

function Inspector({ decision, workload, algorithm, traceActive }) {
  const [expanded, setExpanded] = useState(false);
  return <Section title="Decision inspector" detail={decision ? `t = ${decision.time}` : 'No decision yet'}>
    {!decision ? <p className="empty">Execute a CPU unit to inspect the action, state and reward.</p> : <>
      <div className="decision-choice"><span className="eyebrow">{algorithm === 'DQN' ? (traceActive ? 'Recorded checkpoint decision' : 'Exported network inference') : algorithm}</span><strong style={{ color: COLORS[decision.selected_action] }}>{decision.selected_pid}</strong><span className="mono">slot {decision.selected_action} · [{decision.execution.start}, {decision.execution.end})</span></div>
      <p className="reason">{decision.reason}</p>
      <dl className="compact-metrics"><div><dt>Reward</dt><dd>{decision.reward}</dd></div><div><dt>Switch</dt><dd>{decision.context_switch ? 'Yes' : 'No'}</dd></div><div><dt>Preempt</dt><dd>{decision.preemption ? 'Yes' : 'No'}</dd></div></dl>
      {decision.idle_interval_after && <p className="idle-notice">Idle jump: {decision.idle_interval_after.start} → {decision.idle_interval_after.end}. No waiting reward is added.</p>}
      {algorithm === 'Round Robin' && <p className="hint">Queue after dispatch: {decision.queue.map(i => workload[i].pid).join(' → ') || 'empty'}</p>}
      {decision.q_values && <><table className="q-table"><thead><tr><th>Slot / job</th><th>Valid</th><th>Q-value</th></tr></thead><tbody>{decision.q_values.map((value, i) => <tr key={i} className={i === decision.selected_action ? 'selected-row' : !decision.action_mask[i] ? 'invalid-row' : ''}><td>{i} / {workload[i]?.pid ?? 'unused'}</td><td>{decision.action_mask[i] ? '✓' : '—'}</td><td className="mono">{number(value, 5)}</td></tr>)}</tbody></table><p className="hint">{algorithm === 'DQN' ? 'Actual network predictions; invalid slots cannot be selected. Q-values estimate future return, not probabilities.' : 'Values from the tabular agent. A zero row can mean an unseen state.'}</p></>}
      <button className="text-button" onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide' : 'Inspect'} 39-feature observation</button>
      {expanded && <div className="observation"><div className="table-scroll"><table><thead><tr><th>Slot</th>{definitions.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{Array.from({ length: 6 }, (_, i) => <tr key={i}><td>{i}</td>{decision.observation.slice(i * 6, i * 6 + 6).map((v, j) => <td key={j}>{number(v, 3)}</td>)}</tr>)}</tbody></table></div><p className="mono hint">Globals: [{decision.observation.slice(36).map(v => number(v)).join(', ')}]</p><p className="hint">Future jobs and padding are zero. Time / 68, arrived / 6, completed / 6.</p></div>}
    </>}
  </Section>;
}

function Comparison({ comparisons, workload, quantum, mode, scope = 'full', simulation }) {
  const [pair, setPair] = useState(['SRTF', 'DQN']);
  const [inspected, setInspected] = useState(null);
  const [synced, setSynced] = useState(false);
  const svgRef = useRef(null);
  const names = Object.keys(comparisons);
  const optimal = comparisons.SRTF && metrics(comparisons.SRTF).waiting;
  const effectiveB = comparisons[pair[1]] ? pair[1] : names[0];
  const a = comparisons[pair[0]], b = comparisons[effectiveB];
  const difference = a && b ? metrics(b).waiting - metrics(a).waiting : null;
  function exportSvg() {
    const lanes = names.map((name, row) => `<text x="10" y="${row * 65 + 27}" font-family="monospace" font-size="12">${name}</text>` + mergeTimeline(comparisons[name].timeline).map(s => `<rect x="${135 + s.start * 18}" y="${row * 65 + 10}" width="${(s.end - s.start) * 18}" height="28" fill="${s.process_index === null ? '#dce1e6' : COLORS[s.process_index]}"/>`).join('')).join('');
    const end = Math.max(...names.map(name => comparisons[name].time));
    const ticks = Array.from({ length: Math.floor(end / 2) + 1 }, (_, i) => `<text x="${135 + i * 36}" y="${names.length * 65 + 12}" font-size="11" font-family="monospace">${i * 2}</text>`).join('');
    const legend = workload.map((p, i) => `<rect x="${15 + i * 110}" y="${names.length * 65 + 28}" width="12" height="12" fill="${COLORS[i]}"/><text x="${33 + i * 110}" y="${names.length * 65 + 39}" font-size="11">${p.pid.replace(/[&<>"']/g, '')}</text>`).join('');
    download('algorithm_comparison.svg', `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.max(780, end * 18 + 160)}" height="${names.length * 65 + 65}"><rect width="100%" height="100%" fill="white"/>${lanes}${ticks}${legend}</svg>`, 'image/svg+xml');
  }
  return <><Section title="Same workload. Different policies." detail={scope === 'full' ? 'Complete executions' : 'Current workload'}>
    <div className="comparison-toolbar"><span>DQN: {mode}</span><label><input type="checkbox" checked={synced} onChange={event => setSynced(event.target.checked)} /> Sync charts to live time</label><button onClick={() => download('algorithm_comparison.json', { workload, round_robin_quantum: quantum, dqn_mode: mode, results: comparisons })}>Export results JSON</button><button onClick={exportSvg}>Export Gantt SVG</button></div>
    <div ref={svgRef}>{names.map(name => {
      const result = comparisons[name], values = metrics(result);
      const view = synced && simulation ? { ...result, timeline: result.timeline.filter(s => s.start < simulation.time).map(s => ({ ...s, end: Math.min(s.end, simulation.time) })), decisions: result.decisions.filter(d => d.time < simulation.time), time: simulation.time } : result;
      return <div className="comparison-lane" key={name}><div><strong>{name}</strong><span className="mono">waiting {number(values.waiting)}</span></div><Timeline state={view} label={name} compact onInspect={interval => setInspected({ name, ...interval })} /></div>;
    })}</div>
    {inspected && <p className="inspection mono">{inspected.name}: {inspected.pid ?? 'IDLE'} [{inspected.start}, {inspected.end}), duration {inspected.end - inspected.start}</p>}
    <Legend workload={workload} />
    <div className="table-scroll"><table className="comparison-table"><thead><tr><th>Algorithm</th><th>Waiting</th><th>Turnaround</th><th>Response</th><th>Switches</th><th>Preemptions</th><th>Reward</th><th>SRTF gap</th></tr></thead><tbody>{names.map(name => {
      const values = metrics(comparisons[name]);
      return <tr key={name} className={name === 'DQN' ? 'highlight-row' : ''}><th>{name}</th><td>{number(values.waiting)}</td><td>{number(values.turnaround)}</td><td>{number(values.response)}</td><td>{values.switches}</td><td>{values.preemptions}</td><td>{values.reward}</td><td>{optimal ? `${number((values.waiting - optimal) / optimal * 100, 2)}%` : values.waiting === 0 ? '0.00%' : 'n/a'}</td></tr>;
    })}</tbody></table></div>
    <p className="hint">Times are CPU units per process. Round Robin quantum = {quantum}. All policies use the same execution engine. SRTF is the optimal waiting reference under known bursts and zero switch cost.</p>
  </Section><Section title="Compare two policies" detail="B − A mean waiting">
    <div className="pair"><label>A<select value={pair[0]} onChange={e => setPair([e.target.value, pair[1]])}>{names.map(name => <option key={name}>{name}</option>)}</select></label><label>B<select value={effectiveB} onChange={e => setPair([pair[0], e.target.value])}>{names.map(name => <option key={name}>{name}</option>)}</select></label><div className="pair-result"><strong>{number(difference)}</strong><span>CPU units / process · negative favors B</span></div></div>
  </Section></>;
}

function SavedResults({ data }) {
  if (!data) return <Section title="Saved experiment results"><p className="empty">Loading verified experiment data…</p></Section>;
  const max = Math.max(...data.policies.map(p => p.mean_waiting_time));
  return <div className="results-page"><div className="page-intro"><span className="eyebrow">Measured Python experiments</span><h1>Results from the trained schedulers</h1><p>256 frozen workloads · 1,277 processes · test seed {data.test_seed}. These figures are independent of the editable live workload.</p></div>
    <div className="results-grid"><Section title="Held-out mean waiting time" detail="Lower is better"><div className="bar-chart">{data.policies.map(row => <div className="bar-row" key={row.policy}><span>{row.policy}</span><div><i style={{ width: `${row.mean_waiting_time / max * 100}%`, background: row.policy.includes('Augmented') ? '#2866ad' : '#8a98a7' }} /></div><strong>{number(row.mean_waiting_time)}</strong></div>)}</div><p className="hint">CPU time units per process; identical workloads for every policy.</p></Section>
      <Section title="What augmentation changed" detail="One paired training seed"><div className="study-number">{number(data.augmentation_waiting_difference, 6)}</div><p>Augmented minus original DQN waiting</p><p className="mono">95% CI [{data.augmentation_waiting_interval.map(v => number(v, 6)).join(', ')}]</p><div className="outcomes"><div><strong>{data.improved}</strong><span>improved</span></div><div><strong>{data.worsened}</strong><span>worsened</span></div><div><strong>{data.tied}</strong><span>tied</span></div></div><p className="hint">The interval measures workload sampling uncertainty for fixed checkpoints. It does not establish improvement across independent training seeds.</p></Section></div>
    <Section title="Complete benchmark" detail="Recorded final evaluation"><div className="table-scroll"><table><thead><tr><th>Scheduler</th><th>Waiting</th><th>Turnaround</th><th>Response</th><th>Switches</th><th>Preemptions</th><th>SRTF gap</th></tr></thead><tbody>{data.policies.map(row => <tr key={row.policy} className={row.policy.includes('Augmented') ? 'highlight-row' : ''}><th>{row.policy}</th>{['mean_waiting_time', 'mean_turnaround_time', 'mean_response_time', 'mean_context_switches', 'mean_preemptions'].map(key => <td key={key}>{number(row[key])}</td>)}<td>{number(row.srtf_percentage_waiting_gap, 2)}%</td></tr>)}</tbody></table></div></Section>
    <Section title="Greedy validation throughout training" detail="128 fixed validation workloads"><Curve series={[data.learning_curves.original_10000, data.learning_curves.augmented_10000]} labels={['Original DQN', 'Augmented DQN']} /><p className="hint">Checkpoints were selected on validation before held-out testing. Fluctuating performance and lower training loss do not prove mathematical convergence.</p></Section>
    <Section title="Scope of the result"><p className="body-copy">Single core, synthetic arrivals, known bursts after arrival, no I/O and zero switching overhead. The augmented DQN is 2.38% above SRTF on mean waiting. Round Robin responds earlier but waits and switches more. Broader distributions and independent seeds remain future work.</p></Section>
  </div>;
}

function Learning({ learned, setLearned, apply }) {
  const [settings, setSettings] = useState({ episodes: 2000, alpha: 0.2, gamma: 1, epsilon: 1, seed: 42 });
  const [progress, setProgress] = useState(null), [error, setError] = useState('');
  const valid = Number.isInteger(settings.episodes) && settings.episodes > 0 && settings.episodes <= 10000 && settings.alpha > 0 && settings.alpha <= 1 && settings.gamma >= 0 && settings.gamma <= 1 && settings.epsilon >= 0 && settings.epsilon <= 1 && Number.isInteger(settings.seed);
  async function train() {
    setError(''); setProgress({ episode: 0, episodes: settings.episodes });
    try { setLearned(await trainTabular({ ...settings, onProgress: setProgress })); } catch (e) { setError(e.message); } finally { setProgress(null); }
  }
  return <div className="learning-page"><div className="page-intro"><span className="eyebrow">Actual tabular Q-learning</span><h1>Learn a scheduling policy</h1><p>This teaching agent trains on the fixed three-process basic workload. Its Q-table is tied to those processes and does not automatically generalize to custom tasks.</p></div>
    <div className="results-grid"><Section title="Training controls" detail="Runs in your browser"><div className="learning-form">{Object.entries(settings).map(([key, value]) => <label key={key}>{key}<input type="number" aria-label={`Training ${key}`} step={['alpha', 'gamma', 'epsilon'].includes(key) ? 0.05 : 1} value={value} disabled={!!progress} onChange={event => setSettings({ ...settings, [key]: Number(event.target.value) })} /></label>)}</div><button className="primary" disabled={!!progress || !valid} onClick={train}>{progress ? `Training ${progress.episode} / ${progress.episodes}` : 'Train Q-learning agent'}</button>{progress && <progress max={progress.episodes} value={progress.episode} />}{error && <p className="error">{error}</p>}<p className="hint">State: time and arrived-process remaining times. Actions: valid process indices. Reward: negative count of waiting jobs. Updates use the Bellman target after every CPU unit.</p><div className="formula">Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,a′) − Q(s,a)]</div><button disabled={!learned || !!progress} className="full-width" onClick={apply}>Use learned policy in the laboratory</button></Section>
      <Section title="Measured greedy waiting" detail="Basic workload"><Curve series={[learned?.curve ?? []]} labels={['Q-learning validation']} colors={['#2866ad']} />{learned && <div className="learning-summary"><strong>{number(learned.curve.at(-1).waiting)}</strong><span>mean waiting after {learned.episodes} training episodes</span><span>{Object.keys(learned.table).length} learned state rows</span></div>}<p className="hint">The SRTF reference waiting for this workload is 1.000. A low score reflects this fixed example, not generalization to new workloads.</p></Section></div>
    {learned && <Section title="Learned Q-table" detail="First 20 state rows"><div className="table-scroll"><table><thead><tr><th>State (time | remaining)</th><th>P1</th><th>P2</th><th>P3</th></tr></thead><tbody>{Object.entries(learned.table).slice(0, 20).map(([key, values]) => <tr key={key}><th className="mono">{key}</th>{values.slice(0, 3).map((value, i) => <td key={i}>{number(value, 5)}</td>)}</tr>)}</tbody></table></div><button onClick={() => download('q_learning_training.json', learned)}>Export learned Q-table and measured curve</button></Section>}
  </div>;
}

export default function App() {
  const [tab, setTab] = useState('lab'), [algorithm, setAlgorithm] = useState('SRTF');
  const [workload, setWorkload] = useState(() => copy(BASIC)), [preset, setPresetName] = useState('basic');
  const [simulation, setSimulation] = useState(() => initialState(BASIC));
  const [playing, setPlaying] = useState(false), [speed, setSpeed] = useState(500), [quantum, setQuantum] = useState(2);
  const [count, setCount] = useState(4), [seed, setSeed] = useState(42), [selectedStep, setSelectedStep] = useState(null);
  const [assets, setAssets] = useState({ model: null, results: null, traces: {} }), [loadError, setLoadError] = useState(''), [reload, setReload] = useState(0);
  const [trace, setTrace] = useState(null), [learned, setLearned] = useState(null), [notice, setNotice] = useState('');
  const traceInput = useRef(null);
  const error = validateWorkload(workload);
  const traceActive = trace && sameWorkload(workload, trace.workload) ? trace : null;
  const qAvailable = learned && sameWorkload(workload, BASIC);
  const mode = traceActive ? `Recorded trace · workload ${traceActive.workload_id}` : 'Actual exported network · browser inference';
  const options = { quantum, model: assets.model, trace: traceActive, table: learned?.table };
  useEffect(() => {
    const controller = new AbortController();
    setLoadError('');
    const read = async name => { const response = await fetch(`./data/${name}.json`, { signal: controller.signal }); if (!response.ok) throw new Error(`Unable to load ${name}. Run the app using its web server.`); return response.json(); };
    Promise.all([read('model'), read('results'), ...[0, 69, 201].map(id => read(`workload_${id}`))]).then(([model, results, ...traces]) => {
      traces.forEach(validateTrace);
      setAssets({ model, results, traces: Object.fromEntries(traces.map(t => [t.workload_id, t])) });
    }).catch(e => { if (e.name !== 'AbortError') setLoadError(e.message); });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setSimulation(current => current && !current.done ? step(current, algorithm, { quantum, model: assets.model, trace: traceActive, table: learned?.table }) : current);
      setSelectedStep(null);
    }, speed);
    return () => clearInterval(timer);
  }, [playing, speed, algorithm, quantum, assets.model, traceActive, learned]);
  useEffect(() => { if (simulation?.done) setPlaying(false); }, [simulation?.done]);
  const comparisons = useMemo(() => {
    if (error) return {};
    const results = Object.fromEntries(['FCFS', 'SRTF', 'Round Robin'].map(name => [name, run(workload, name, { quantum })]));
    if (assets.model || traceActive) results.DQN = run(workload, 'DQN', { model: assets.model, trace: traceActive });
    if (qAvailable) results['Q-learning'] = run(workload, 'Q-learning', { table: learned.table });
    if (algorithm === 'Manual' && simulation?.done) results.Manual = simulation;
    return results;
  }, [workload, quantum, assets.model, traceActive, learned, qAvailable, error, algorithm, simulation?.done]);
  function reset(nextWorkload = workload) { setPlaying(false); setSelectedStep(null); setSimulation(validateWorkload(nextWorkload) ? null : initialState(nextWorkload)); }
  function loadWorkload(nextWorkload, name = 'custom', nextTrace = null) {
    setWorkload(copy(nextWorkload)); setPresetName(name); setTrace(nextTrace); reset(nextWorkload); setNotice('');
    if (algorithm === 'Q-learning' && !sameWorkload(nextWorkload, BASIC)) setAlgorithm('SRTF');
  }
  function selectPreset(name) {
    if (name === 'custom' || name === 'imported') return;
    if (PRESETS[name]) loadWorkload(PRESETS[name], name);
    else if (assets.traces[name]) { loadWorkload(assets.traces[name].workload, name, assets.traces[name]); setAlgorithm('DQN'); }
    else setNotice('The trained trace is still loading.');
  }
  function advance(manual) { if (simulation && !simulation.done) { setSimulation(step(simulation, algorithm, { ...options, manual })); setSelectedStep(null); } }
  function finish() { setPlaying(false); let value = simulation; while (value && !value.done) value = step(value, algorithm, options); setSimulation(value); setSelectedStep(null); }
  async function importJson(event, kind) {
    const file = event.target.files[0]; event.target.value = ''; if (!file) return;
    try {
      if (file.size > 4 * 1024 * 1024) throw new Error('Use a JSON file smaller than 4 MB.');
      const data = JSON.parse(await file.text());
      if (kind === 'trace') { validateTrace(data); loadWorkload(data.workload, 'imported', data); setAlgorithm('DQN'); setTab('lab'); setNotice(`Imported recorded DQN workload ${data.workload_id}.`); }
      else { const values = Array.isArray(data) ? data : data.workload; const problem = validateWorkload(values); if (problem) throw new Error(problem); loadWorkload(values, 'imported'); }
    } catch (e) { setNotice(`Import failed: ${e.message}`); }
  }
  const values = simulation ? metrics(simulation) : null;
  const decision = simulation?.decisions[selectedStep ?? simulation.decisions.length - 1];
  const ready = simulation ? readyIndices(simulation) : [];
  const last = simulation?.decisions.at(-1);
  const blocked = !!error || !simulation || simulation.done || (algorithm === 'DQN' && !assets.model && !traceActive);
  return <div className="app-shell">
    <header className="app-header"><div className="brand"><span className="brand-mark">CPU</span><div><strong>Scheduling Laboratory</strong><span>Adaptive preemptive scheduling with deep Q-learning</span></div></div><nav aria-label="Application sections">{[['lab', 'Laboratory'], ['compare', 'Compare'], ['results', 'Project results'], ['learning', 'Q-learning']].map(([name, label]) => <button key={name} className={tab === name ? 'active' : ''} onClick={() => { setTab(name); setPlaying(false); }}>{label}</button>)}</nav><span className="environment-tag">1 CORE / 6 SLOTS</span></header>
    <main>
      {loadError && <div className="notice error">{loadError} <button onClick={() => setReload(reload + 1)}>Retry data load</button></div>}
      {notice && <div className={`notice ${notice.startsWith('Import failed') ? 'error' : ''}`}>{notice}<button className="remove" aria-label="Dismiss notification" onClick={() => setNotice('')}>×</button></div>}
      {tab === 'lab' && <><div className="toolbar"><label>Algorithm<select aria-label="Scheduling algorithm" value={algorithm} onChange={event => { setAlgorithm(event.target.value); reset(); }}>
        {ALGORITHMS.map(name => <option key={name} disabled={name === 'DQN' && !assets.model && !traceActive || name === 'Q-learning' && !qAvailable}>{name}</option>)}</select></label>
        <div className="playback"><button className="primary" disabled={blocked || algorithm === 'Manual'} onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button><button disabled={blocked || algorithm === 'Manual'} onClick={() => { setPlaying(false); advance(); }}>Step +1</button><button disabled={blocked || algorithm === 'Manual'} onClick={finish}>Run to completion</button><button onClick={() => reset()}>Reset</button></div>
        <label className="speed">Step interval<input aria-label="Playback interval" type="range" min="100" max="1500" step="100" value={speed} onChange={e => setSpeed(Number(e.target.value))} /><span className="mono">{speed} ms</span></label><label className="quantum">RR quantum<input aria-label="Round Robin quantum" type="number" min="1" max="10" value={quantum} onChange={e => { const v = Number(e.target.value); if (Number.isInteger(v) && v > 0 && v <= 10) { setQuantum(v); reset(); } }} /></label></div>
        {error && <div className="notice error">{error}</div>}
        <div className="laboratory-grid"><aside><ProcessEditor workload={workload} simulation={simulation} preset={preset} setPreset={selectPreset}
          change={(index, field, value) => loadWorkload(workload.map((p, i) => i === index ? { ...p, [field]: value } : p))}
          add={() => { const taken = new Set(workload.map(p => p.pid)); let id = 1; while (taken.has(`P${id}`)) id++; loadWorkload([...workload, { pid: `P${id}`, arrival_time: 0, burst_time: 3 }]); }}
          remove={index => loadWorkload(workload.filter((_, i) => i !== index))} count={count} setCount={setCount} seed={seed} setSeed={setSeed} generate={() => loadWorkload(randomWorkload(count, seed))} importWorkload={e => importJson(e, 'workload')} />
          <Section title="Trained DQN" detail="Actual project checkpoint"><div className="network"><span>39</span><i>→</i><span>64</span><i>→</i><span>64</span><i>→</i><span>6</span></div><p className="hint">Two ReLU layers. Six raw Q-values. Invalid outputs are masked.</p><input ref={traceInput} className="file-input" type="file" accept=".json,application/json" onChange={e => importJson(e, 'trace')} /><button className="full-width" onClick={() => traceInput.current.click()}>Import recorded DQN trace</button><p className="hint mono hash">Checkpoint {assets.model?.checkpoint_sha256.slice(0, 16) ?? 'loading'}…</p></Section>
        </aside><div className="simulation-column"><Section title="Single-core CPU" detail={simulation?.done ? 'Workload complete' : playing ? 'Executing' : 'Paused'}><div className="cpu"><div className="clock"><span>SIMULATED TIME</span><strong>{simulation?.time ?? '—'}</strong><span>CPU units</span></div><div className="cpu-slot" style={{ borderColor: last ? COLORS[last.selected_action] : '#b9c5d1' }}><span>{last ? 'LAST EXECUTED UNIT' : 'READY TO DISPATCH'}</span><strong style={{ color: last ? COLORS[last.selected_action] : '#536272' }}>{last?.selected_pid ?? '—'}</strong><span className="mono">{last ? `[${last.execution.start}, ${last.execution.end})` : 'Select a policy and step'}</span></div></div>
          <div className="ready-heading"><span>Ready queue</span><span>{ready.length} available</span></div><div className="ready-queue">{ready.length ? ready.map(i => <button key={i} style={{ borderLeftColor: COLORS[i] }} disabled={algorithm !== 'Manual'} onClick={() => advance(i)} title={algorithm === 'Manual' ? `Execute ${workload[i].pid} for one unit` : 'Arrived and unfinished'}><strong>{workload[i].pid}</strong><span>{simulation.processes[i].remaining} left</span></button>) : <span className="hint">{simulation?.done ? 'All processes completed.' : 'No valid workload.'}</span>}</div>{algorithm === 'Manual' && <p className="hint">Click a ready process to execute exactly one CPU unit.</p>}
          <div className="process-statuses">{simulation?.processes.map((p, i) => <span key={i} className={p.remaining === 0 ? 'completed' : ''}><i style={{ background: COLORS[i] }} />{p.pid}<em>{!p.remaining ? 'Completed' : p.arrival_time > simulation.time ? 'Not arrived' : last?.selected_action === i ? 'Running / last selected' : 'Ready'}</em></span>)}</div></Section>
          <Section title="Execution timeline" detail={algorithm === 'DQN' ? mode : algorithm}><Timeline state={simulation} label="Live" onInspect={segment => { if (segment.process_index === null) setNotice(`Idle interval [${segment.start}, ${segment.end}).`); else { setSelectedStep(simulation.decisions.findIndex(d => d.time === segment.start)); setPlaying(false); } }} /><Legend workload={workload} />{simulation?.decisions.length > 0 && <label className="inspect-step">Inspect executed unit<select aria-label="Inspect decision" value={selectedStep ?? simulation.decisions.length - 1} onChange={e => { setSelectedStep(Number(e.target.value)); setPlaying(false); }}>{simulation.decisions.map((d, i) => <option value={i} key={i}>t={d.time} · {d.selected_pid} · reward {d.reward}</option>)}</select></label>}</Section>
          <Section title="Live metrics" detail={simulation?.done ? 'Final outcomes' : 'Provisional'}><div className="live-metrics">{[['Waiting', number(values?.waiting)], ['Turnaround', number(values?.turnaround)], ['Response', number(values?.response)], ['Completed', `${values?.completed ?? 0}/${workload.length}`], ['CPU utilization', `${number(values?.utilization, 1)}%`], ['Switches', values?.switches ?? 0], ['Preemptions', values?.preemptions ?? 0], ['Total reward', values?.reward ?? 0]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><p className="hint">During execution: waiting is accrued waiting / all jobs; turnaround averages completed jobs; response averages started jobs. All become final when the workload completes.</p></Section>
          {simulation?.done && <Section title="Final process outcomes"><div className="table-scroll"><table><thead><tr><th>Process</th><th>Completion</th><th>Waiting</th><th>Turnaround</th><th>Response</th></tr></thead><tbody>{simulation.processes.map(p => <tr key={p.pid}><th>{p.pid}</th><td>{p.completion}</td><td>{p.waiting}</td><td>{p.completion - p.arrival_time}</td><td>{p.first - p.arrival_time}</td></tr>)}</tbody></table></div></Section>}
        </div><aside><Inspector decision={decision} workload={workload} algorithm={algorithm} traceActive={!!traceActive} /><Section title="Scheduling notes"><details><summary>Preemption and context switches</summary><p>A switch changes jobs between busy units. A true preemption also leaves the previous job unfinished. Dispatch after idle is not counted as a switch.</p></details><details><summary>Reward and waiting</summary><p>Each other ready job waits one unit, so reward is its negative count. At completion, total return equals negative total waiting.</p></details><details><summary>Recorded vs live DQN</summary><p>Recorded presets replay exact PyTorch decisions. Custom workloads run the real exported weights in JavaScript. Floating-point arithmetic can differ slightly from PyTorch; the browser implementation is checked against the recorded examples.</p></details></Section></aside></div>
        {Object.keys(comparisons).length > 0 && <div className="below-lab"><Comparison comparisons={comparisons} workload={workload} quantum={quantum} mode={mode} simulation={simulation} /></div>}
      </>}
      {tab === 'compare' && <><div className="page-intro"><span className="eyebrow">Controlled comparison</span><h1>One workload, different decisions</h1><p>Edit the processes in the Laboratory tab. Every comparison below replays that same workload from a fresh state.</p></div>{error ? <p className="notice error">{error}</p> : <Comparison comparisons={comparisons} workload={workload} quantum={quantum} mode={mode} simulation={simulation} />}</>}
      {tab === 'results' && <SavedResults data={assets.results} />}
      {tab === 'learning' && <Learning learned={learned} setLearned={setLearned} apply={() => { loadWorkload(BASIC, 'basic'); setAlgorithm('Q-learning'); setTab('lab'); }} />}
    </main><footer><span>CPU Scheduling Laboratory · single-core educational simulator</span><a href="https://github.com/Omsiddh/RL_LAB_IA" target="_blank" rel="noreferrer">Project source ↗</a><span>Known bursts after arrival · no I/O · zero switch cost</span></footer>
  </div>;
}
