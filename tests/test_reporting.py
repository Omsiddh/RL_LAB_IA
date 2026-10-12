"""Saved-data reporting tests; no scheduler, model, or workload generation."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from src.reporting import benchmark_figures, gantt
from src.reporting.data import METRICS, POLICIES, Sources, sha256, verify_final
from src.reporting.report import consolidated_rows
from src.reporting.run import run_reporting, verify_protected
from src.reporting.style import FigureWriter, configure_style, rolling_mean


@pytest.fixture
def saved_data():
    # Pure records with unequal cluster sizes. These are unit-test inputs,
    # not regenerated test workloads or scheduler outputs.
    sizes = [4] * 90 + [5] * 79 + [6] * 87
    definitions = [[[f"P{i}", 0, 1] for i in range(n)] for n in sizes]
    process_rows, workload_rows, rows, groups, summary_rows = [], [], [], [], []
    totals = [7440, 7617, 7813, 8323, 10764, 14326]
    for wid, n in enumerate(sizes):
        workload_rows.append({"workload_id": str(wid), "process_count": str(n)})
    for policy, total in zip(POLICIES, totals):
        quotient, remainder = divmod(total, 1277)
        selected, rank = [], 0
        prefix = policy.lower().replace(" ", "_")
        for wid, definition in enumerate(definitions):
            waiting = 0
            for index, (pid, arrival, burst) in enumerate(definition):
                wait = quotient + (rank < remainder)
                p = {"policy": policy, "workload_id": str(wid), "process_count": str(len(definition)), "process_index": str(index),
                    "pid": pid, "arrival_time": str(arrival), "burst_time": str(burst), "waiting_time": str(wait),
                    "turnaround_time": str(wait + burst), "response_time": str(wait)}
                selected.append(p)
                waiting += wait
                rank += 1
            workload_rows[wid].update({prefix + "_total_reward": str(-waiting), prefix + "_context_switches": "0", prefix + "_preemptions": "0"})
        numeric = {"policy": policy, "workload_count": 256, "process_count": 1277, "total_waiting_time": total,
            "total_turnaround_time": total + 1277, "total_response_time": total,
            "mean_waiting_time": total / 1277, "mean_turnaround_time": (total + 1277) / 1277, "mean_response_time": total / 1277,
            "mean_context_switches": 0, "mean_preemptions": 0, "total_reward": -total,
            "srtf_percentage_waiting_gap": 100 * (total - totals[0]) / totals[0]}
        rows.append({k: str(v) for k, v in numeric.items()})
        summary_rows.append(numeric)
        process_rows.extend(selected)
        for n in (4, 5, 6):
            group = [p for p in selected if int(p["process_count"]) == n]
            groups.append({"policy": policy, "process_count_group": str(n), "workload_count": str(sizes.count(n)),
                **{"mean_" + k: str(sum(int(p[k]) for p in group) / len(group)) for k in ("waiting_time", "turnaround_time", "response_time")}})
    interval = {"group": "all", "left_policy": POLICIES[1], "right_policy": POLICIES[2], "metric": "waiting_time",
        "difference": (totals[1] - totals[2]) / 1277, "ci_low": -.21, "ci_high": -.1, "replicates": 5000, "seed": 19, "primary": True}
    return {"final": rows, "processes": process_rows, "workload_rows": workload_rows,
        "final_groups": groups, "final_summary": {"comparison": summary_rows, "bootstrap_intervals": [interval]},
        "test_metadata": {"workload_count": 256, "process_count": 1277, "process_count_workloads": {str(n): sizes.count(n) for n in (4, 5, 6)},
            "workload_sha256": hashlib.sha256(json.dumps(definitions, separators=(",", ":")).encode()).hexdigest()},
        "test_workloads": definitions, "intervals": [{k: str(v) for k, v in interval.items()}]}


def test_saved_means_totals_groups_and_ci_crosscheck(saved_data):
    verify_final(saved_data)
    rows = consolidated_rows(saved_data)
    assert tuple(r["policy"] for r in rows) == POLICIES
    assert tuple(rows[0]) == ("policy", *METRICS)
    assert rows[1]["mean_waiting_time"] == saved_data["final"][1]["mean_waiting_time"]


@pytest.mark.parametrize("corruption", ("process_mean", "reward", "identity", "workload_count", "digest", "ci", "group"))
def test_saved_data_corruption_is_rejected(saved_data, corruption):
    if corruption == "process_mean":
        saved_data["processes"][0]["waiting_time"] = "999"
    elif corruption == "reward":
        saved_data["workload_rows"][0]["srtf_total_reward"] = "999"
    elif corruption == "identity":
        saved_data["processes"][0]["pid"] = "changed"
    elif corruption == "workload_count":
        saved_data["test_metadata"]["workload_count"] = 255
    elif corruption == "digest":
        saved_data["test_metadata"]["workload_sha256"] = "changed"
    elif corruption == "ci":
        saved_data["intervals"][0]["ci_low"] = "-99"
    else:
        saved_data["final_groups"][0]["mean_waiting_time"] = "999"
    with pytest.raises(ValueError):
        verify_final(saved_data)


def test_missing_source_has_actionable_error_without_creating_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="Restore the saved experiment; do not rerun policies"):
        Sources(tmp_path).csv("missing.csv")
    with pytest.raises(FileNotFoundError, match="source artifact is missing"):
        run_reporting(root=tmp_path, output_dir=tmp_path / "new_report")
    assert not (tmp_path / "new_report").exists()


def test_output_overwrite_rejected_before_sources_are_read(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reporting.run.load_results", lambda *args: pytest.fail("Read sources before rejecting overwrite"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_reporting(root=tmp_path, output_dir=tmp_path)


def test_gantt_merges_only_adjacent_same_identity_and_preserves_runtime():
    definitions = [["A", 1, 3], ["B", 2, 2]]
    timeline = [[0, 1, None, None], [1, 2, 0, "A"], [2, 3, 0, "A"],
                [3, 4, 1, "B"], [4, 5, 1, "B"], [5, 6, 0, "A"]]
    original = deepcopy(timeline)
    merged = gantt.merge_intervals(timeline)
    assert merged == [(0, 1, None, None), (1, 3, 0, "A"), (3, 5, 1, "B"), (5, 6, 0, "A")]
    assert gantt.verify_timeline(definitions, timeline) == gantt.verify_timeline(definitions, merged)
    assert timeline == original


@pytest.mark.parametrize("fault", ("burst", "arrival", "index", "gap", "zero_duration"))
def test_gantt_rejects_invalid_saved_segments(fault):
    definitions = [["A", 1, 2]]
    timeline = [[0, 1, None, None], [1, 2, 0, "A"], [2, 3, 0, "A"]]
    if fault == "burst":
        definitions[0][2] = 3
    elif fault == "arrival":
        definitions[0][1] = 2
    elif fault == "index":
        timeline[-1][2] = 1
    elif fault == "gap":
        timeline[-1][0] = 1
    else:
        timeline[-1][1] = 2
    with pytest.raises(ValueError):
        gantt.verify_timeline(definitions, timeline)


def test_rolling_mean_is_trailing_and_ignores_missing_prefixes():
    actual = rolling_mean([np.nan, 2, 4, 8], 2)
    assert np.isnan(actual[0]) and actual[1:] == [2, 3, 6]
    assert rolling_mean([1, 2, 3], 10) == [1, 1.5, 2]
    with pytest.raises(ValueError):
        rolling_mean([1], 0)


def test_benchmark_figures_copy_values_and_saved_ci_at_300_dpi(saved_data, tmp_path):
    configure_style()
    sources = {"runs/phase6_4_final_test/" + name: "synthetic" for name in ("policy_comparison.csv", "by_process_count.csv", "bootstrap_intervals.csv")}
    writer = FigureWriter(tmp_path, sources)
    benchmark_figures.generate(saved_data, writer)
    assert len(writer.figures) == 8
    ci = next(r for r in writer.figures if r["name"] == "augmentation_difference_ci")["plotted_data"]
    assert ci["ci_low"] == float(saved_data["intervals"][0]["ci_low"])
    assert ci["ci_high"] == float(saved_data["intervals"][0]["ci_high"])
    assert ci["difference"] == float(saved_data["intervals"][0]["difference"])
    for record in writer.figures:
        assert record["deterministic_export_verified"]
        with Image.open(tmp_path / (record["name"] + ".png")) as picture:
            assert picture.info["dpi"] == pytest.approx((300, 300), abs=.01)
        assert (tmp_path / (record["name"] + ".svg")).exists()


def test_independent_figure_builds_produce_identical_png_and_svg(tmp_path):
    configure_style()
    rendered = []
    for i in range(2):
        folder = tmp_path / str(i)
        folder.mkdir()
        writer = FigureWriter(folder, {"toy.csv": "synthetic"})
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [2, 4, 3], label="Saved toy records")
        ax.set(xlabel="Time", ylabel="Recorded value", title="Deterministic fixture")
        ax.legend()
        writer.save(fig, "fixture", "synthetic test", "Source: toy.csv", ["toy.csv"], {"x": [1, 2, 3], "y": [2, 4, 3]})
        rendered.append(writer.figures[0]["files"])
    assert rendered[0] == rendered[1]


def test_reporting_imports_cannot_execute_agents_simulators_or_workload_generation():
    forbidden = ("src.agent", "src.env", "src.baselines", "src.training", "src.permutation", "src.final_evaluation", "torch")
    for path in (Path(__file__).resolve().parents[1] / "src/reporting").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(forbidden) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden)


def test_input_preservation_check_detects_changed_artifact(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text("original")
    frozen = {path.name: sha256(path)}
    verify_protected(tmp_path, frozen)
    path.write_text("changed")
    with pytest.raises(AssertionError, match="protected input artifacts"):
        verify_protected(tmp_path, frozen)
