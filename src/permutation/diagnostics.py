"""Read-only observers around Trainer optimization and greedy validation."""

import csv
import json
import math
from statistics import fmean
import time

import numpy as np
import torch

from src.agent import DQNAgent
from src.benchmark.run import write_csv, write_json
from src.training.train import Trainer
from .experiment import tree_digest


class QStatistics:
    def __init__(self):
        self.count = self.nonfinite = 0
        self.minimum, self.maximum = math.inf, -math.inf
        self.absolute_sum = self.maximum_absolute = 0.0

    def add(self, tensor):
        values = tensor.detach().cpu().numpy().astype(np.float64).ravel()
        self.nonfinite += int((~np.isfinite(values)).sum())
        values = values[np.isfinite(values)]
        if values.size:
            self.count += int(values.size)
            self.minimum = min(self.minimum, float(values.min()))
            self.maximum = max(self.maximum, float(values.max()))
            self.absolute_sum += float(np.abs(values).sum())
            self.maximum_absolute = max(self.maximum_absolute, float(np.abs(values).max()))

    def result(self):
        return {"count": self.count, "nonfinite": self.nonfinite,
            "minimum": self.minimum if self.count else None, "maximum": self.maximum if self.count else None,
            "mean_absolute": self.absolute_sum / self.count if self.count else None,
            "maximum_absolute": self.maximum_absolute}


def train_observed(config, output_dir):
    """Train from initialization, observing but not changing update behavior."""
    output_dir.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(config)
    updates, health = [], []
    original_optimize = trainer.agent.optimize_step
    original_evaluate = trainer.validator.evaluate
    interval_start = 0
    workloads = []
    original_episode = trainer._train_episode

    def observe_episode(adapter):
        original_reset = adapter.reset

        def record(workload):
            workloads.append(workload)
            return original_reset(workload)

        adapter.reset = record
        try:
            return original_episode(adapter)
        finally:
            adapter.reset = original_reset

    trainer._train_episode = observe_episode
    fields = ["episode", "environment_steps", "optimizer_updates", "loss",
              "gradient_norm_before_clipping", "gradient_norm_after_clipping", "clipping_applied"]
    with (output_dir / "optimizer_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def observe_optimize():
            result = original_optimize()
            if result is None:
                return result
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm()
                for p in trainer.agent.policy_network.parameters() if p.grad is not None])).item()
            row = {"episode": trainer.episode, "environment_steps": trainer.environment_steps,
                "optimizer_updates": result["optimizer_steps"], "loss": result["loss"],
                "gradient_norm_before_clipping": result["gradient_norm"], "gradient_norm_after_clipping": norm,
                "clipping_applied": result["gradient_norm"] > trainer.agent.max_gradient_norm}
            writer.writerow(row)
            updates.append(row)
            if not all(math.isfinite(row[key]) for key in ("loss", "gradient_norm_before_clipping", "gradient_norm_after_clipping")):
                stream.flush()
                raise FloatingPointError("Non-finite optimizer diagnostic")
            return result

        def observe_evaluate(agent):
            nonlocal interval_start
            policy_all, policy_valid, target_all = QStatistics(), QStatistics(), QStatistics()
            original_select = agent.select_action

            def inspect_action(observation, mask, epsilon):
                with torch.no_grad():
                    values = agent.policy_network(observation)
                    policy_all.add(values)
                    policy_valid.add(values[torch.as_tensor(mask, device=agent.device)])
                    target_all.add(agent.target_network(observation))
                return original_select(observation, mask, epsilon)

            agent.select_action = inspect_action
            try:
                result = original_evaluate(agent)
            finally:
                agent.select_action = original_select
            interval = updates[interval_start:]
            interval_start = len(updates)
            finite = all(torch.isfinite(p).all().item() for network in (agent.policy_network, agent.target_network)
                         for p in network.parameters())
            row = {"episode": trainer.episode, "environment_steps": trainer.environment_steps,
                "optimizer_updates": agent.optimizer_steps, "training_epsilon": config.epsilon_at(trainer.environment_steps),
                **result, "interval_mean_loss": fmean(r["loss"] for r in interval) if interval else None,
                "interval_maximum_loss": max((r["loss"] for r in interval), default=None),
                "interval_maximum_gradient_norm": max((r["gradient_norm_before_clipping"] for r in interval), default=None),
                "interval_clipping_count": sum(r["clipping_applied"] for r in interval),
                "policy_all_q": policy_all.result(), "policy_valid_q": policy_valid.result(),
                "target_all_q": target_all.result(), "weights_finite": finite}
            health.append(row)
            write_json(output_dir / "validation_health.json", health)
            stream.flush()
            if not finite or any(s.nonfinite for s in (policy_all, policy_valid, target_all)):
                raise FloatingPointError("Non-finite validation Q-values or network weights")
            print(f"Health episode={trainer.episode} loss={row['interval_mean_loss']} "
                  f"max_valid_abs_Q={policy_valid.maximum_absolute:.3f} clipped={row['interval_clipping_count']}", flush=True)
            return result

        trainer.agent.optimize_step = observe_optimize
        trainer.validator.evaluate = observe_evaluate
        started = time.perf_counter()
        result = trainer.run(output_dir, verbose=True)
        elapsed = time.perf_counter() - started
    # Load the files rather than trusting in-memory final weights or sidecars.
    verification = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.torch_threads)
        for name in ("best_model", "final_model"):
            metadata = json.loads((output_dir / (name + ".json")).read_text())
            restored = DQNAgent()
            restored.load(output_dir / (name + ".pt"))
            actual = original_evaluate(restored)
            logged = next(row for row in trainer.validation_metrics if row["episode"] == metadata["episode"])
            if (any(actual[key] != logged[key] for key in actual)
                    or metadata["validation"] != logged
                    or restored.optimizer_steps != logged["optimizer_updates"]):
                raise AssertionError("Saved checkpoint does not reproduce its logged validation")
            verification.append({"checkpoint": name, "episode": metadata["episode"], **actual, "exact_match": True})
    finally:
        torch.set_num_threads(previous_threads)
    write_csv(output_dir / "checkpoint_verification.csv", verification)
    summary = {**result, "elapsed_seconds": elapsed, "configuration": config.to_dict(),
        "workload_sequence_digest": tree_digest(tuple(workloads)),
        "nonfinite_loss_gradient_q_count": 0,
        "mean_loss": fmean(row["loss"] for row in updates), "maximum_loss": max(row["loss"] for row in updates),
        "maximum_gradient_norm_before_clipping": max(row["gradient_norm_before_clipping"] for row in updates),
        "maximum_gradient_norm_after_clipping": max(row["gradient_norm_after_clipping"] for row in updates),
        "clipping_count": sum(row["clipping_applied"] for row in updates),
        "maximum_valid_q_magnitude": max(row["policy_valid_q"]["maximum_absolute"] for row in health),
        "final_replay_size": len(trainer.agent.replay_buffer), "checkpoint_scores_verified": True}
    write_json(output_dir / "experiment_summary.json", summary)
    return summary
