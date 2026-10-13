"""Render the executed controlled experiment without assuming improvement."""

import csv
import json


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write_report(root, summary):
    audit = json.loads((root / "workload_audit.json").read_text())
    original = [r for r in summary["comparison"] if r["arrangements"] == "original"]
    alternative = [r for r in summary["comparison"] if r["arrangements"] != "original"]
    paired = [r for r in summary["paired_model_comparison"] if r["candidate"] == "augmented_10000_best"
              and r["reference"] == "original_10000_best" and r["process_count_group"] == "all"]
    names = {"reference_5250": "Reference", "original_10000_best": "A: original 10k", "augmented_10000_best": "B: augmented 10k"}
    wait_delta = next(r for r in paired if r["metric"] == "original_waiting")
    agreement_delta = next(r for r in paired if r["metric"] == "fixed_state_agreement")
    range_delta = next(r for r in paired if r["metric"] == "waiting_range")
    if wait_delta["ci_high"] < 0:
        conclusion = "Augmentation improved selected-checkpoint original-layout validation waiting in this paired experiment; its workload-bootstrap delta interval excludes zero."
    elif wait_delta["ci_low"] > 0:
        conclusion = "Augmentation worsened selected-checkpoint original-layout validation waiting in this paired experiment; its workload-bootstrap delta interval excludes zero."
    else:
        conclusion = "The original-layout validation waiting difference has a workload-bootstrap interval spanning zero; this experiment does not establish an aggregate waiting benefit from augmentation."
    lines = ["# Stage 6.3B controlled training and replay augmentation", "", conclusion, "",
        "This uses one paired training seed (42), one augmentation seed (630342), and model-selection validation data. Workload bootstrap intervals do not measure training-seed variability or remove checkpoint-selection bias. No final test workloads were used.", "",
        "## Audit before implementation", "",
        "`generate_workload` uses a local RNG to sample count 4..6, arrivals 0..8, and bursts 1..10. Definitions initially have PID order P1..Pn. Phase 5 `sample_workload` already draws a fresh 64-bit seed and performs one independent shuffle using the training workload RNG. No second workload shuffle was introduced.", "",
        "PID placements are uniform in the sampler conditional on process count, with finite-sample fluctuations. Padding causes unequal unconditional slot occupancy; policy-dependent ready/replay exposure need not be uniform. Slot 3 has the same occupancy as 0..2, so its previously observed sensitivity cannot be attributed simply to padding.", "",
        table(["Slot", "Occupied / 10,000", "Mean arrival", "Mean burst"],
              [[r["slot"], r["occupied_workloads"], f'{r["mean_arrival"]:.4f}', f'{r["mean_burst"]:.4f}'] for r in audit["slot_attributes"]]), "",
        "Workload counts: 4 processes = 3,435; 5 = 3,277; 6 = 3,288. Conditional PID/slot cell counts are 821..897 (expected 858.75), 613..688 (655.4), and 503..584 (548). See [workload_audit.json](workload_audit.json) for all cells and the workload stream digest.", "",
        "## Implementation and controls", "",
        "Added `src/training/augmentation.py`. `TrainingConfig` and the training CLI expose `replay_augmentation=none|permutation` and `augmentation_seed`; the default is none. Before each replay insertion, enabled augmentation uniformly samples a permutation of occupied workload slots, including hidden future jobs. It moves whole six-feature blocks at both transition endpoints, remaps the action with the inverse map, and moves the next-action mask. Globals, reward, terminal flag, unused slots, and live simulator actions remain unchanged. One transformed transition replaces one original transition; there are no additional samples.", "",
        "The disabled branch retains the old `remember` call. The new A run reproduces all 6,000 historical training rows and all original validation rows exactly, including loss/gradient means. Both new runs use identical full workload definitions/order, 274,300 environment steps, 68,326 optimizer updates, replay capacity 20,000, batch size 64, and unchanged 39->64->64->6 DQN, optimizer, reward, epsilon schedule, warmup, and synchronization settings.", "",
        "Both runs began from fresh identical seed-42 initialization and ran uninterrupted for 10,000 episodes. Neither loaded the earlier checkpoint for training. Greedy validation uses the exact seed-1729 128-workload / 645-process collection at episode 0 and every 250 episodes. Strictly lower process-weighted original-layout waiting selects each best checkpoint. Best and final checkpoint files were independently reloaded and reproduced their logged scores.", "",
        "New analysis modules audit sampling, observe training without extra updates, freeze/serialize common states, evaluate the selected models, compute paired comparisons, and generate charts/reports. The existing fixed-state evaluator gained an optional shared-collection argument; its default behavior remains unchanged. Simulator, RL adapter, DQN architecture, reward, and baseline sources were untouched. Existing checkpoints and experiment artifacts were hash-verified unchanged.", "",
        "## Training statistics and checkpoints", ""]
    stats = []
    for folder, run in summary["training"]["runs"].items():
        stats.append([folder, run["best_validation_episode"], f'{run["best_validation_mean_waiting_time"]:.6f}',
            f'{run["final_validation_mean_waiting_time"]:.6f}', run["optimizer_updates"], f'{run["mean_loss"]:.6f}',
            f'{run["maximum_loss"]:.6f}', f'{run["maximum_gradient_norm_before_clipping"]:.3f}', run["clipping_count"],
            f'{run["maximum_valid_q_magnitude"]:.3f}'])
    lines += [table(["Run", "Best episode", "Best waiting", "Final waiting", "Updates", "Mean loss", "Max loss", "Max pre-clip norm", "Clip count", "Max valid abs Q"], stats), "",
        "Both runs have zero non-finite losses, gradient norms, validation Q-values, or weights detected. Gradient clipping statistics include actual post-clipping norms; floating-point norms can be slightly above the threshold of 10. Maxima span training, not just the selected checkpoint. Full optimizer-level diagnostics and validation Q summaries are saved for both runs.", "",
        "Best/final artifacts and complete histories:", "",
        "- [A best](original_10000/best_model.pt), [A final](original_10000/final_model.pt), [A training CSV](original_10000/training_metrics.csv), [A validation CSV](original_10000/validation_metrics.csv), [A diagnostics](original_10000/optimizer_diagnostics.csv).",
        "- [B best](augmented_10000/best_model.pt), [B final](augmented_10000/final_model.pt), [B training CSV](augmented_10000/training_metrics.csv), [B validation CSV](augmented_10000/validation_metrics.csv), [B diagnostics](augmented_10000/optimizer_diagnostics.csv).", "",
        "## Original validation performance", "",
        table(["Selected model", "Episode", "Waiting", "Turnaround", "Response", "Switches/workload", "Preemptions/workload", "Total reward", "Gap to SRTF", "Gap %"],
            [[names[r["model"]], r["best_episode"], *[f'{r[k]:.6f}' for k in ("mean_waiting_time", "mean_turnaround_time", "mean_response_time", "mean_context_switches", "mean_preemptions")],
              r["total_reward"], f'{r["srtf_absolute_waiting_gap"]:.6f}', f'{r["srtf_percentage_waiting_gap"]:.3f}'] for r in original]), "",
        "SRTF waiting = 6.006201550387597. Timing means are process weighted; switches and preemptions are workload weighted. Reward equals negative total waiting on every rollout. SRTF waiting is invariant and <= each DQN on every tested arrangement.", "",
        "## Shared fixed-state and full-trajectory permutations", "",
        f'All three models use the same {summary["fixed_original_states"]:,} frozen original states, producing {summary["fixed_permutation_comparisons_per_model"]:,} non-identity comparisons per model. Sources are the episode-5250 reference DQN and SRTF original trajectories. The observations are saved in [fixed_state_observations.npz](evaluation/fixed_state_observations.npz), without pickle, and reloaded exactly. No model collects its own distinct observations for the action-agreement comparison.', "",
        "Every model uses the original Stage 6.3A per-workload maps: original plus seven fixed alternatives. The reference reproduces Stage 6.3A action, workload, and per-process CSVs exactly. Evaluation repeats exactly and preserves full model/optimizer/replay/RNG/gradient state.", "",
        table(["Selected model", "Fixed-state agreement %", "Alternative waiting", "Alternative response", "Alternative switches", "Alternative preemptions", "Mean waiting range", "Mean population std", "Changed workloads %", "Alternative gap %"],
            [[names[r["model"]], *[f'{r[k]:.6f}' for k in ("fixed_state_agreement_percent", "mean_waiting_time", "mean_response_time", "mean_context_switches", "mean_preemptions", "mean_waiting_range", "mean_waiting_std", "changed_workloads_percent", "srtf_percentage_waiting_gap")]] for r in alternative]), "",
        "Ranges and population standard deviations use all eight complete-trajectory arrangements per workload. Agreement excludes identity layouts. Fixed-state actions and complete trajectory performance answer different questions; subsequent states differ when policy decisions diverge.", "",
        "## Results by process count", "",
        table(["Model", "Processes", "Original waiting", "Alternative waiting", "SRTF waiting", "Agreement %", "Mean range", "Original gap %", "Alternative gap %"],
            [[names[r["model"]], r["process_count_group"], *[f'{r[k]:.6f}' for k in ("dqn_original_waiting", "dqn_permuted_waiting", "srtf_waiting", "action_agreement_percent", "mean_waiting_range", "srtf_original_percentage_gap", "srtf_permuted_percentage_gap")]] for r in summary["by_process_count"]]), "",
        table(["Model", "Processes", "Original response", "Alternative response", "Original switches", "Alternative switches", "Original preemptions", "Alternative preemptions"],
            [[names[r["model"]], r["process_count_group"], *[f'{r[k]:.6f}' for k in ("original_mean_response_time", "permuted_mean_response_time", "original_mean_context_switches", "permuted_mean_context_switches", "original_mean_preemptions", "permuted_mean_preemptions")]] for r in summary["by_process_count"]]), "",
        "The process-count CSV additionally records original/alternative turnaround, rewards, and all SRTF control metrics.", "",
        "## Paired B minus A comparisons", "",
        table(["Metric", "Difference", "95% CI low", "95% CI high"],
              [[r["metric"], f'{r["estimate"]:.6f}', f'{r["ci_low"]:.6f}', f'{r["ci_high"]:.6f}'] for r in paired]), "",
        "Agreement differences are fractions (multiply by 100 for percentage points); negative waiting/range/std differences favor augmentation. Confidence intervals use 5,000 paired percentile bootstrap draws, seed 63021729, resampling whole workloads and retaining their states/arrangements. Timing differences use process-weighted ratios. Additional A/reference and B/reference contrasts and process-count groups are in [paired_model_comparison.csv](paired_model_comparison.csv).", "",
        "## Complete validation history", ""]
    histories = {}
    health = {}
    for folder in ("original_10000", "augmented_10000"):
        with (root / folder / "validation_metrics.csv").open(newline="") as stream:
            histories[folder] = list(csv.DictReader(stream))
        health[folder] = json.loads((root / folder / "validation_health.json").read_text())
    history_rows = []
    for i, a in enumerate(histories["original_10000"]):
        b = histories["augmented_10000"][i]
        ha, hb = health["original_10000"][i], health["augmented_10000"][i]
        history_rows.append([a["episode"], f'{float(a["mean_waiting_time"]):.6f}', a["is_best"],
            f'{float(b["mean_waiting_time"]):.6f}', b["is_best"],
            f'{ha["training_epsilon"]:.5f}', a["optimizer_updates"],
            f'{ha["interval_mean_loss"]:.6f}' if ha["interval_mean_loss"] is not None else "",
            f'{hb["interval_mean_loss"]:.6f}' if hb["interval_mean_loss"] is not None else ""])
    lines += [table(["Episode", "A waiting", "A best", "B waiting", "B best", "Training epsilon", "Updates (both)", "A interval loss", "B interval loss"], history_rows), "",
        "Validation epsilon is always zero. Training epsilon and update counts are identical at each checkpoint because the workload sequence and CPU burst accounting match. Both CSV histories additionally contain turnaround and reward at every validation checkpoint.", "",
        "## Interpretation", "", conclusion, "",
        f'B minus A fixed-state agreement = {100 * agreement_delta["estimate"]:+.3f} percentage points, and mean waiting range difference = {range_delta["estimate"]:+.6f}. These directly describe consistency on the shared observations and sampled layouts. A change in aggregate mean should not be substituted for these sensitivity measurements.', "",
        "The augmentation intervention changes replay coordinate coverage while holding workload generation, architecture, hyperparameters, environment steps, optimizer updates, and validation fixed. It does not establish which individual slot mechanism explains any improvement. The selected checkpoints may occur at different episodes. One seed pair is insufficient to establish a generally reliable training advantage or convergence, and the bootstrap intervals are unadjusted for multiple comparisons. No Phase 6.4 work or final-test evaluation was performed.", "",
        "Figures: [learning curves](learning_curves.png), [stability](stability_diagnostics.png), [checkpoint metrics](checkpoint_comparison.png), [permutation consistency](permutation_consistency_comparison.png).", ""]
    test_file = root / "test_results.json"
    if test_file.exists():
        test = json.loads(test_file.read_text())
        lines += ["## Test results", "", f'{test["passed"]} passed, {test["skipped"]} skipped. Command: `{test["command"]}`. The skip is the existing CUDA availability test.', ""]
    (root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
