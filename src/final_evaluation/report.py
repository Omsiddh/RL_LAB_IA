"""Concise Stage 6.4 technical record; polished Gantts remain out of scope."""

def table(rows, columns):
    def format_value(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(label for label, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(format_value(row[key]) for _, key in columns) + " |" for row in rows]
    return "\n".join(lines)


def write_report(output_dir, summary, test_results):
    metadata = summary["test_collection"]
    primary = next(r for r in summary["bootstrap_intervals"] if r["primary"])
    conclusion = ("Augmentation reduced held-out waiting versus the equal-budget original in this fixed seed pair; the paired workload interval excludes zero."
                  if primary["ci_high"] < 0 else "The paired workload interval does not establish reduced held-out waiting from augmentation."
                  if primary["ci_low"] <= 0 else "Augmentation increased held-out waiting in this fixed seed pair; the paired workload interval excludes zero.")
    generalization = ("This supports a held-out benefit on the training distribution for this checkpoint pair."
                      if primary["ci_high"] < 0 else "These held-out results do not establish an augmentation benefit for this checkpoint pair.")
    lines = ["# Stage 6.4 held-out evaluation", "", conclusion, "",
        "This is a technical evaluation record. No checkpoints were trained, tuned, or reselected after test inspection; Stage 6.5 and Phase 7 were not implemented.", "",
        "## Frozen inputs", "",
        f"No prior saved final collection or test seed was found. Dedicated seed {metadata['seed']} was fixed before generation: 256 workloads, {metadata['process_count']} processes, existing count 4..6 / arrival 0..8 / burst 1..10 distribution and training's single slot shuffle. All policies replay the exact same immutable collection.", "",
        f"Workload SHA-256: `{metadata['workload_sha256']}`. Counts: {metadata['process_count_workloads']}. Validation digest: `{metadata['validation_workload_sha256']}`.", "",
        f"Cross-split duplicates: {metadata['exact_cross_split_duplicate_workloads']} exact workloads, {metadata['arrival_burst_multiset_cross_split_duplicate_workloads']} arrival/burst multisets ignoring identity/order. Within-test duplicates: {metadata['within_test_exact_duplicate_workloads']} exact, {metadata['within_test_arrival_burst_multiset_duplicates']} arrival/burst multisets. No redraw/filtering occurred.", "",
        "[Frozen protocol](frozen_protocol.json), [checkpoint manifest](selected_model_manifest.json), [workload definitions](test_workloads.json), [workload metadata](test_workload_manifest.json). Both manifests were saved before test generation. Source/configuration and every preexisting experiment artifact were hashed and verified unchanged.", ""]
    for model in summary["selected_models"]:
        lines.append(f"- {model['policy']}: episode {model['selected_episode']}, `{model['checkpoint']}`; SHA-256 `{model['checkpoint_sha256']}`.")
    lines += ["", "## Six-policy comparison", "",
        "Times are process weighted; switches, preemptions, and mean reward are workload weighted. RR quantum = 2. Greedy DQN epsilon = 0 on CPU, with gradients disabled.", "",
        table(summary["comparison"], [("Policy", "policy"), ("Waiting", "mean_waiting_time"), ("Turnaround", "mean_turnaround_time"), ("Response", "mean_response_time"), ("Switches", "mean_context_switches"), ("Preemptions", "mean_preemptions"), ("Total reward", "total_reward"), ("Mean reward", "mean_reward"), ("SRTF gap", "srtf_absolute_waiting_gap"), ("Gap %", "srtf_percentage_waiting_gap")]), "",
        "Reward equals negative total waiting on every rollout. SRTF is no worse on waiting for every workload and every policy.", "",
        table(summary["comparison"][1:4], [("DQN", "policy"), ("Ties SRTF (workloads)", "srtf_tied_workloads"), ("Waiting above SRTF (workloads)", "srtf_above_workloads")]), "",
        "## Paired comparisons", "",
        f"Augmented versus original workload outcomes: {summary['augmentation_workload_outcomes']}. Negative timing differences favor augmentation; response/switch/preemption differences use the same sign convention.", "",
        table([r for r in summary["bootstrap_intervals"] if r["group"] == "all"], [("Left", "left_policy"), ("Right", "right_policy"), ("Metric", "metric"), ("Difference", "difference"), ("95% low", "ci_low"), ("95% high", "ci_high")]), "",
        f"Intervals use 5,000 paired percentile workload resamples, seed {summary['bootstrap_seed']}. Every draw retains all six paired results; timing estimates recompute summed process times divided by summed process counts. These measure workload sampling uncertainty conditional on fixed trained checkpoints, not variation across independent training seeds. Secondary and subgroup intervals are unadjusted for multiplicity.", "",
        "## Process-count groups", "",
        table(summary["by_process_count"], [("Processes", "process_count_group"), ("Policy", "policy"), ("Workloads", "workload_count"), ("Waiting", "mean_waiting_time"), ("Response", "mean_response_time"), ("Switches", "mean_context_switches"), ("Preemptions", "mean_preemptions"), ("SRTF gap %", "srtf_percentage_waiting_gap")]), "",
        "Group turnaround, rewards, totals, and SRTF tie counts are in [by_process_count.csv](by_process_count.csv); subgroup intervals are in [bootstrap_intervals.csv](bootstrap_intervals.csv).", "",
        "## Selected examples", "",
        "Workload IDs are zero based. Selection uses augmented minus original mean waiting; lowest ID breaks ties. Unit-step timelines for all six policies are stored in [selected_timelines.json](selected_timelines.json) for later Gantt rendering.", ""]
    for example in summary["examples"]:
        wid = example["workload_id"]
        metrics = example["policy_metrics"]
        lines += [f"- {example['label'].capitalize()}: workload {wid}, definitions `{example['definitions']}`, augmented minus original waiting = {example['augmented_minus_original_waiting']:.6f}; augmented {metrics['augmented_dqn_10k_mean_waiting_time']:.6f}, original {metrics['original_dqn_10k_mean_waiting_time']:.6f}, SRTF {metrics['srtf_mean_waiting_time']:.6f}."]
    lines += ["", "## Verification and limits", "", test_results, "",
        "The complete six-policy evaluation was repeated on the same frozen workloads, with exact per-process, per-workload, and timeline equality. Bootstrap outputs also reproduce exactly. Runtime assertions check arrivals, completion, process identity/index, CPU/idle accounting, reward/waiting identity, turnaround identity, action masks, SRTF inequalities, immutable definitions, and unchanged full model/optimizer/replay/RNG/gradient snapshots. Global Python/NumPy/Torch RNG states and prior source/artifact hashes remain unchanged. [Verification details](verification_report.json).", ""]
    lines += ["- " + limitation for limitation in summary["limitations"]]
    lines += ["", conclusion + " " + generalization + " It does not establish reliable improvement across training seeds or convergence.", "",
        "Charts: [waiting](mean_waiting_time.png), [turnaround](mean_turnaround_time.png), [response](mean_response_time.png), [switches](mean_context_switches.png), [preemptions](mean_preemptions.png), [DQN optimality gaps](dqn_optimality_gaps.png), [process-count groups](by_process_count.png).", ""]
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
