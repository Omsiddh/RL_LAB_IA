"""Refresh the React laboratory's real checkpoint, traces, and experiment data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from main import DEFAULT_CHECKPOINT, PROJECT_ROOT
from src.agent import DQNAgent
from .export_traces import export_traces


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "web/public/data")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    agent = DQNAgent(device="cpu")
    agent.load(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = [layer for layer in agent.policy_network.layers if isinstance(layer, torch.nn.Linear)]
    write_json(args.output_dir / "model.json", {
        "source": "exported_pytorch_checkpoint", "architecture": [39, 64, 64, 6],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_seed": agent.config["seed"], "gamma": agent.gamma,
        "normalization": {"burst": 10, "duration": 68, "processes": 6},
        "layers": [{"weight": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist()} for layer in layers],
    })
    export_traces((0, 69, 201), args.checkpoint, args.output_dir)
    final = PROJECT_ROOT / "runs/phase6_4_final_test"
    with (final / "policy_comparison.csv").open(newline="") as stream:
        policies = [{key: value if key == "policy" else float(value)
                     for key, value in row.items()} for row in csv.DictReader(stream)]
    curves = {}
    for name in ("original_10000", "augmented_10000"):
        path = PROJECT_ROOT / "runs/phase6_3b_controlled" / name / "validation_metrics.csv"
        with path.open(newline="") as stream:
            curves[name] = [{"episode": int(row["episode"]),
                             "waiting": float(row["mean_waiting_time"])}
                            for row in csv.DictReader(stream)]
    write_json(args.output_dir / "results.json", {
        "source": "saved_final_evaluation", "workloads": 256, "processes": 1277,
        "test_seed": 6401729, "policies": policies, "learning_curves": curves,
        "augmentation_waiting_difference": -0.153485,
        "augmentation_waiting_interval": [-0.207905, -0.102263],
        "improved": 90, "worsened": 36, "tied": 130,
    })
    print(f"Exported actual network weights, three traces, and saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
