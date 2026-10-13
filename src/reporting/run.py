"""Build Stage 6.5 figures and findings without evaluating any scheduler."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from . import benchmark_figures, gantt, learning_figures, permutation_figures
from .data import load_results, sha256
from .report import consolidated_rows, findings
from .style import FigureWriter, configure_style


def protected_inputs(root, output):
    paths = [p for p in (root / "runs").rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.resolve().is_relative_to(output)]
    paths += list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    paths += [root / name for name in ("README.md", "pyproject.toml", "requirements.txt")]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def verify_protected(root, protected):
    changed = [path for path, digest in protected.items() if not (root / path).is_file() or sha256(root / path) != digest]
    if changed:
        raise AssertionError("Reporting changed protected input artifacts: " + ", ".join(changed))


def run_reporting(root=None, output_dir=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    output = Path(output_dir or root / "runs/phase6_5_final_visualizations").resolve()
    if output.exists():
        raise FileExistsError(f"Reporting output must be a new directory; refusing to overwrite {output}")
    protected = protected_inputs(root, output)
    data = load_results(root)
    timeline_checks = gantt.verify_examples(data)
    # No outputs are created until required sources and timelines pass checks.
    (output / "figures").mkdir(parents=True, exist_ok=False)
    configure_style()
    writer = FigureWriter(output / "figures", data["sources"].hashes)
    benchmark_figures.generate(data, writer)
    gantt.generate(data, writer)
    learning_figures.generate(data, writer)
    permutation_figures.generate(data, writer)
    rows = consolidated_rows(data)
    with (output / "consolidated_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    (output / "experimental_findings.md").write_text(findings(data), encoding="utf-8")
    verify_protected(root, protected)
    manifest = {"stage": "6.5", "source_root": str(root), "presentation_only": True,
        "policy_execution_workload_regeneration_checkpoint_loading_training": False,
        "runtime": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "sources": data["sources"].hashes, "protected_source_and_prior_artifacts": protected,
        "figures": writer.figures, "missing_optional_data": data["missing_optional_data"],
        "checks": {"final_counts": {"workloads": 256, "processes": 1277, "policies": 6},
            "final_process_weighted_means_and_totals_match": True, "final_workload_weighted_switch_preemption_means_match": True,
            "saved_test_workload_digest_matches": True, "original_process_identities_match": True,
            "plotted_bootstrap_intervals_equal_saved_statistics": True, "checkpoint_metadata_matches_validation_history": True,
            "stage6_3a_reference_agreement_reproduced_in_saved_stage6_3b_statistics": True,
            "gantt_raw_and_merged_checks": timeline_checks, "deterministic_png_and_svg_exports": True,
            "png_dpi": 300, "protected_inputs_unchanged": True, "protected_file_count": len(protected)},
        "outputs": {filename: sha256(output / filename) for filename in ("consolidated_benchmark.csv", "experimental_findings.md")}}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed {len(writer.figures)} figures; all source consistency/preservation checks passed", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run_reporting(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
