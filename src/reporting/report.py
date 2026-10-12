"""Build a technical findings document exclusively from recorded results."""

from .data import METRICS, MODEL_NAMES, POLICIES


HEADERS = ("Policy", "Average waiting", "Average turnaround", "Average response", "Switches/workload", "Preemptions/workload", "Total reward", "Waiting gap to SRTF (%)")


def table(headers, rows):
    def display(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(display(v) for v in row) + " |" for row in rows])


def consolidated_rows(data):
    return [{"policy": row["policy"], **{key: row[key] for key in METRICS}} for row in data["final"]]


def findings(data):
    final = {r["policy"]: r for r in data["final"]}
    augmented, original, srtf = (final[p] for p in POLICIES[1:3] + POLICIES[:1])
    summary, controlled, metadata = data["final_summary"], data["controlled"], data["test_metadata"]
    ci = next(r for r in data["intervals"] if r["primary"] == "True")
    contrasts = [r for r in data["intervals"] if r["group"] == "all"]
    config = data["training"][POLICIES[1]]["config"]
    agent_config = data["training"][POLICIES[1]]["best"]["configuration"]["agent_config"]
    figure = lambda name, label: f"[{label}](figures/{name}.png) ([SVG](figures/{name}.svg))"
    lines = ["# Stage 6.5 experimental findings", "",
        "This is a technical findings and figure companion, not the final academic report. It presents existing experiments only. No scheduling policies, held-out evaluation, workload generation, checkpoint loading, training, tuning, or model reselection occurred during reporting.", "",
        "## 1. Experimental objective", "",
        "Compare selected deep Q-learning schedulers with classical CPU scheduling baselines on waiting time, quantify replay-permutation augmentation at equal training budget, and describe permutation sensitivity and timing tradeoffs. The declared primary held-out contrast is augmented 10k minus original 10k; comparisons with SRTF, FCFS, RR, and the historical 6k model are secondary.", "",
        "## 2. Simulator and workload assumptions", "",
        "The simulator models one CPU, integer CPU-time steps, perfect knowledge of arrived processes' burst lengths, preemption, no I/O, and zero context-switch overhead. Future-job properties are hidden from the policy until arrival. Workloads have 4–6 processes, arrivals uniformly sampled in 0–8, and bursts in 1–10. The existing sampler performs one seeded slot shuffle per workload. Stable indices and process identities are preserved.", "",
        f"Final-test seed **{metadata['seed']}** was fixed before test generation because no earlier test seed or collection was recorded. The saved collection has **{metadata['workload_count']} workloads / {metadata['process_count']} processes**. Workload counts for 4/5/6 processes are respectively " + "/".join(str(metadata["process_count_workloads"][str(n)]) for n in (4, 5, 6)) + ".", "",
        f"Saved test SHA-256: `{metadata['workload_sha256']}`. Exact cross-split duplicates: {metadata['exact_cross_split_duplicate_workloads']}; arrival/burst-multiset cross-split duplicates: {metadata['arrival_burst_multiset_cross_split_duplicate_workloads']}. No duplicate filtering or redrawing occurred. [Exact definitions](../phase6_4_final_test/test_workloads.json), [frozen protocol](../phase6_4_final_test/frozen_protocol.json).", "",
        "## 3. Models and classical scheduling baselines", "",
        "FCFS is non-preemptive, SRTF selects shortest remaining arrived work, and Round Robin uses FIFO with quantum 2. All use the same tested simulator. Under these zero-overhead assumptions SRTF minimizes mean waiting (equivalently total completion/turnaround) for this single-core preemptive objective; DQN is not expected to improve on that oracle policy for waiting.", "",
        "The following checkpoints were selected on original-layout validation before held-out testing and were not changed in Stage 6.5:", "",
        table(["Role", "Model", "Selected episode", "Checkpoint SHA-256"], [[m["role"], m["policy"], m["selected_episode"], "`" + m["checkpoint_sha256"] + "`"] for m in data["models"]]), "",
        "[Full selected-model metadata](../phase6_4_final_test/selected_model_manifest.json). Stage 6.5 reads checkpoint sidecars and recorded hashes; it does not load model weights.", "",
        "## 4. Training and validation methodology", "",
        f"The historical Phase 5 run trained for {data['phase5_config']['episodes']} episodes and selected its reference at episode {data['phase5_metadata']['episode']}. Stage 6.3B trained two fresh, uninterrupted 10,000-episode models with shared training seed {config['seed']} and identical workloads. Replay augmentation used a separate RNG seed {config['augmentation_seed']}. Greedy epsilon-zero validation occurred every {config['validation_interval']} episodes on the same seed-{config['validation_seed']} 128-workload / 645-process collection.", "",
        "Architecture and optimizer hyperparameters were held fixed. Recorded settings: " + ", ".join(f"{key}={agent_config[key]}" for key in ("learning_rate", "gamma", "batch_size", "replay_capacity", "target_sync_interval", "max_gradient_norm")) + ".", "",
        table(["Training run", "Environment steps", "Optimizer updates", "Best episode", "Best validation waiting", "Final validation waiting"],
            [[name, run["environment_steps"], run["optimizer_updates"], run["best_validation_episode"], run["best_validation_mean_waiting_time"], run["final_validation_mean_waiting_time"]]
             for name, run in data["training_summary"]["runs"].items()]), "",
        figure("validation_learning_curves", "Greedy validation waiting curves") + "; " + figure("training_reward", "exploratory training rewards") + "; " + figure("optimizer_loss", "optimizer TD loss") + "; " + figure("epsilon_schedule", "logged training epsilon") + ".", "",
        "The reward panel shows raw episode totals and a trailing 100-episode arithmetic mean, with shorter initial windows. These rewards depend on workload size and exploratory behavior; they are not validation scores. Loss uses a trailing 250-optimizer-update mean, displayed every 50 updates plus the final update. Epsilon shows end-of-episode observations against CPU steps; both schedules match exactly.", ""]
    late = []
    for policy, bundle in data["training"].items():
        values = [float(r["mean_waiting_time"]) for r in bundle["validation"] if int(r["episode"]) >= 8000]
        late.append([policy, min(values), max(values), float(bundle["validation"][-1]["mean_waiting_time"])])
    lines += [table(["Model", "Validation minimum, episodes 8000–10000", "Maximum", "Final"], late), "",
        "Both histories continue to fluctuate late in training, and final scores exceed the selected best scores. Finite-sample validation, nonstationary replay/targets, and ongoing parameter updates can contribute; these logs alone do not isolate a cause. Lower TD loss and improved validation do not prove mathematical convergence or a permanent plateau.", "",
        "## 5. Final held-out comparison", "",
        "All six policies were tested on identical workloads with epsilon zero for DQN. Timing means divide summed process times by total processes. Switches/preemptions divide workload totals by workload count. The table below copies recorded values; no policies are rerun.", "",
        table(HEADERS, [[r["policy"], *[float(r[key]) if key != "total_reward" else int(r[key]) for key in METRICS]] for r in data["final"]]), "",
        "[Machine-readable consolidated table](consolidated_benchmark.csv). " + "; ".join(figure(name, label) for name, label in
            (("final_waiting", "waiting"), ("final_turnaround", "turnaround"), ("final_response", "response"), ("final_context_switches", "switches"), ("final_preemptions", "preemptions"))) + ".", "",
        "Recorded total reward equals negative total waiting for every rollout. The saved per-process and per-workload files independently reproduce all displayed means and rewards. RR has the shortest response but substantially greater waiting and switching. Compared with SRTF, each DQN has lower response time here and higher waiting; this is a tradeoff, not overall superiority.", "",
        "## 6. SRTF optimality-gap analysis", "",
        table(["DQN", "Absolute waiting gap", "Gap (%)", "Workloads tying SRTF", "Workloads above SRTF"],
            [[r["policy"], float(r["srtf_absolute_waiting_gap"]), float(r["srtf_percentage_waiting_gap"]), int(r["srtf_tied_workloads"]), int(r["srtf_above_workloads"])] for r in data["final"][1:4]]), "",
        f"Augmented waiting is {float(augmented['mean_waiting_time']):.6f}, versus SRTF {float(srtf['mean_waiting_time']):.6f}; its gap is {float(augmented['srtf_percentage_waiting_gap']):.6f}%. It ties SRTF on {augmented['srtf_tied_workloads']}/{metadata['workload_count']} workloads. SRTF is no worse on waiting for every saved workload. " + figure("optimality_gaps", "DQN optimality gaps") + ".", "",
        "## 7. Replay permutation augmentation ablation", "",
        "Training already randomized occupied workload slots. Augmentation therefore did not add a second workload shuffle: it transformed one replay transition's coordinates before insertion. A shared occupied-slot permutation moves whole six-feature blocks in state and next state, remaps the action and next-action mask, and preserves globals, reward, done flag, hidden future-job observations, padding, and the one-transition-per-step budget.", "",
        f"The primary held-out waiting contrast is **{float(ci['difference']):+.6f}**, 95% paired workload-bootstrap CI **[{float(ci['ci_low']):+.6f}, {float(ci['ci_high']):+.6f}]**. " + figure("augmentation_difference_ci", "Existing paired confidence interval") + ".", "",
        f"Across workloads augmentation improves {summary['augmentation_workload_outcomes']['improves']}, worsens {summary['augmentation_workload_outcomes']['worsens']}, and ties {summary['augmentation_workload_outcomes']['ties']} original-DQN results. Equal budgets control workload sequence, environment steps, updates, architecture, hyperparameters, and validation; selected checkpoints occur at different episodes.", "",
        table(["Augmented minus comparator", "Metric", "Difference", "95% low", "95% high"],
            [[r["right_policy"], r["metric"], float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])]
             for r in contrasts if r["left_policy"] == POLICIES[1]]), "",
        "Response increases slightly versus the equal-budget original, while switches and preemptions decrease slightly. Their intervals include zero; a reliable improvement in these secondary metrics is not established. The primary waiting interval excludes zero and supports a same-distribution benefit for this fixed checkpoint pair, not a universal training advantage.", "",
        "## 8. Permutation consistency analysis", "",
        f"These are **VALIDATION / PERMUTATION EXPERIMENTS**, not held-out test estimates. All models share {controlled['fixed_original_states']:,} frozen states from reference-DQN and SRTF trajectories and {controlled['fixed_permutation_comparisons_per_model']:,} non-identity comparisons each. Full-trajectory variability uses the original plus seven fixed alternative layouts per workload. Identity agreement and trajectory waiting range answer different questions.", ""]
    permutation = [r for r in data["permutation_rows"] if r["arrangements"] == "original"]
    lines += [table(["Model", "Identity agreement (%)", "Mean waiting range", "Mean population std", "Changed workloads (%)"],
        [[MODEL_NAMES[r["model"]], *[float(r[key]) for key in ("fixed_state_agreement_percent", "mean_waiting_range", "mean_waiting_std", "changed_workloads_percent")]] for r in permutation]), "",
        figure("permutation_agreement", "Fixed-state agreement") + "; " + figure("permutation_waiting_range", "within-workload waiting variation") + "; " + figure("permutation_by_process_count", "robustness by process count") + ".", ""]
    if not data["missing_optional_data"]:
        lines += [figure("permutation_by_slot", "Conditional agreement by destination slot") + " conditions on the destination of the originally chosen process. Slot denominators differ; slots 4 and 5 occur only for 5–6 and 6 jobs. Slot 3/5 weakness in the reference and improvements in later models describe layout sensitivity, not an isolated causal slot mechanism.", ""]
    else:
        lines += ["Some optional slot-level sources are unavailable; no slot statistic was invented. See the manifest's missing_optional_data field.", ""]
    ref_full = data["permutation_reference"]["full_rollout"]
    lines += [f"Stage 6.3A's reference pooled alternatives-minus-original waiting delta is {ref_full['paired_delta']['estimate']:+.6f}, CI [{ref_full['paired_delta']['ci_low']:+.6f}, {ref_full['paired_delta']['ci_high']:+.6f}]. Thus aggregate cancellation can coexist with strong per-workload sensitivity. Augmentation reduces sensitivity in the saved comparisons, but agreement remains below 100% and waiting still varies: exact process-slot permutation equivariance is not established.", "",
        "## 9. Gantt-chart case studies", "",
        "Case selection was recorded during Stage 6.4: minimum and maximum augmented-minus-original per-workload mean waiting, with lowest workload ID breaking ties. These extremes illustrate mechanisms and failure cases; they are not representative averages. Charts use actual process IDs, shared CPU axes, original slot/arrival/burst tables, consistent per-process colors, and saved mean waiting. Only adjacent identical segments are merged, including idle intervals.", ""]
    for example in data["timelines"]:
        wid = example["workload_id"]
        lines += [f"### Workload {wid}: {example['label']} augmented-relative case", "",
            table(["Original slot", "Process", "Arrival", "Burst"], [[i, *p] for i, p in enumerate(example["definitions"])]), "",
            table(["Scheduler", "Mean waiting"], [[policy, float(example["policy_metrics"][policy.lower().replace(" ", "_") + "_mean_waiting_time"])]
                for policy in ("SRTF", "Original DQN 10k", "Augmented DQN 10k")]), "",
            f"Augmented minus original = {example['augmented_minus_original_waiting']:+.6f} CPU units / process. " + figure(f"gantt_workload_{wid}", f"Aligned recorded timelines for workload {wid}") + ".", ""]
    lines += ["Every recorded process runtime equals its burst; arrivals, stable indices, first-start/completion times, and mean waiting agree with saved results. Merged and raw timelines preserve identical durations and timing. No simulated segments were fabricated or executed.", "",
        "## 10. Process-count effects", "",
        figure("by_process_count", "All-policy held-out waiting by process count") + ".", "",
        table(["Processes", "Workloads", "SRTF waiting", "Augmented waiting", "Original waiting", "Reference waiting", "Augmented gap (%)"],
            [[n, metadata["process_count_workloads"][str(n)], *[float(next(r for r in data["final_groups"] if r["policy"] == p and int(r["process_count_group"]) == n)["mean_waiting_time"]) for p in (POLICIES[0], POLICIES[1], POLICIES[2], POLICIES[3])],
              float(next(r for r in data["final_groups"] if r["policy"] == POLICIES[1] and int(r["process_count_group"]) == n)["srtf_percentage_waiting_gap"])] for n in (4, 5, 6)]), "",
        "Augmented waiting is lower than original in each held-out process-count group, while six-process workloads retain the largest DQN/SRTF gaps. Validation/permutation group plots separately show reduced consistency and larger waiting ranges with more occupied slots; these are descriptive effects, not proof of a specific causal mechanism.", "",
        "## 11. Statistical uncertainty and limitations", "",
        f"Held-out intervals copy the existing {ci['replicates']}-draw paired percentile workload bootstrap, seed {ci['seed']}. Whole workloads retain all paired schedulers, and each timing draw recalculates summed process-time differences / summed process counts. This reporting phase does not draw new bootstrap samples. Validation/permutation intervals likewise reuse saved statistics, with model-selection bias and dependent states/layouts retained at workload level.", "",
        "- One paired DQN training seed: intervals quantify sampled-workload uncertainty, not independent training-seed variation.",
        "- The held-out test follows the same simulated distribution as training; it does not test distribution shift.",
        "- Arrived burst lengths are perfectly known. Single-core CPU, zero switch overhead, and no I/O limit external applicability.",
        "- SRTF is optimal for waiting under these assumptions; DQN does not outperform SRTF on this objective.",
        "- Greedy validation selected the checkpoints before held-out testing; best and final scores differ and late fluctuations remain.",
        "- Secondary/subgroup intervals have no multiple-comparison adjustment. Extreme Gantt examples are selected illustrations.",
        "- There is no proof of convergence, exact slot equivariance, real OS execution, or deployment.", "",
        "## 12. Final conclusions", "",
        f"For the fixed checkpoint pair, replay augmentation reduces held-out waiting from {float(original['mean_waiting_time']):.6f} to {float(augmented['mean_waiting_time']):.6f}, with a paired interval excluding zero. Saved shared-state and trajectory interventions also show greater permutation consistency and smaller within-workload waiting variation. These observations support a same-distribution benefit for the evaluated models.", "",
        f"SRTF remains the waiting-time reference at {float(srtf['mean_waiting_time']):.6f}; augmented DQN retains a {float(augmented['srtf_percentage_waiting_gap']):.3f}% gap. RR trades greater waiting and switching for faster response. The results do not establish cross-seed reliability, broader-distribution generalization, convergence, or overall DQN superiority.", "",
        "All figures have 300-DPI PNG and deterministic SVG exports. [Figure/source manifest](figure_manifest.json) records exact plotted values, source SHA-256 hashes, export hashes, and verification. Stage 6.5 adds presentation only; Phase 7 and the final academic report remain outside this work.", ""]
    return "\n".join(lines)
