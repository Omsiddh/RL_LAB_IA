"""Run with ``python -m src.training.train --smoke --output-dir runs/smoke``."""

import argparse
import csv
import json
from pathlib import Path
import platform
import random
from statistics import fmean

import numpy as np
import torch

from src.agent import DQNAgent
from src.env.rl_adapter import RLAdapter
from .config import TrainingConfig
from .plots import generate_plots
from .validation import Validator, episode_metrics, sample_workload


TRAINING_FIELDS = [
    "episode", "environment_steps", "reward", "process_count", "mean_waiting_time",
    "mean_turnaround_time", "epsilon_start", "epsilon", "episode_length",
    "episode_optimizer_updates", "optimizer_updates", "mean_loss", "mean_gradient_norm",
]
VALIDATION_FIELDS = [
    "episode", "environment_steps", "optimizer_updates", "epsilon", "workload_count",
    "process_count", "mean_waiting_time", "mean_turnaround_time", "mean_reward", "is_best",
]


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Trainer:
    def __init__(self, config=None):
        self.config = config if config is not None else TrainingConfig()
        self.agent = DQNAgent(seed=self.config.seed, device=self.config.device,
                              **self.config.agent_options)
        if self.config.replay_warmup > self.agent.replay_buffer.capacity:
            raise ValueError("replay_warmup must not exceed replay capacity")
        self.training_rng = random.Random(self.config.seed)
        self.validator = Validator(self.config.validation_workloads, self.config.validation_seed)
        self.environment_steps = 0
        self.episode = 0
        self.training_metrics = []
        self.validation_metrics = []
        self.best_validation = None
        self._started = False

    def _configuration(self):
        return {
            **self.config.to_dict(), "agent_config": self.agent.config,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": str(torch.__version__), "device": str(self.agent.device),
                        "deterministic_algorithms": True},
        }

    def _save_checkpoint(self, output_dir, name, validation):
        self.agent.save(output_dir / (name + ".pt"))
        write_json(output_dir / (name + ".json"), {
            "episode": self.episode, "environment_steps": self.environment_steps,
            "optimizer_updates": self.agent.optimizer_steps,
            "epsilon_schedule_position": self.environment_steps,
            "epsilon": self.config.epsilon_at(self.environment_steps), "seed": self.config.seed,
            "configuration": self._configuration(), "validation": validation,
            "checkpoint_purpose": "Trained model evaluation; optimizer state is also included",
            "replay_experiences_saved": False, "exact_resume_supported": False,
            "resume_limitation": "Replay is empty on load. Training orchestration and workload RNG "
                                 "are not restored by DQNAgent.load; exact resume is unsupported.",
        })

    def _validate(self, output_dir, writer, stream, verbose):
        result = {
            "episode": self.episode, "environment_steps": self.environment_steps,
            "optimizer_updates": self.agent.optimizer_steps, "epsilon": 0.0,
            **self.validator.evaluate(self.agent),
        }
        result["is_best"] = (self.best_validation is None or
                             result["mean_waiting_time"] < self.best_validation["mean_waiting_time"])
        if result["is_best"]:
            self.best_validation = result.copy()
            self._save_checkpoint(output_dir, "best_model", result)
        self.validation_metrics.append(result)
        writer.writerow(result)
        stream.flush()
        if verbose:
            print(f"Validation episode={self.episode} waiting={result['mean_waiting_time']:.4f} "
                  f"turnaround={result['mean_turnaround_time']:.4f} best={result['is_best']}", flush=True)

    def _train_episode(self, adapter):
        state = adapter.reset(sample_workload(self.training_rng))
        epsilon_start = self.config.epsilon_at(self.environment_steps)
        reward_sum = steps = 0
        losses, gradient_norms = [], []
        done = False
        while not done:
            epsilon = self.config.epsilon_at(self.environment_steps)
            action = self.agent.select_action(state, adapter.valid_action_mask(), epsilon=epsilon)
            next_state, reward, done, info = adapter.step(action)
            self.agent.remember(state, action, reward, next_state, done, info["action_mask"])
            self.environment_steps += 1
            # The global clock includes warm-up. The first eligible update is
            # a multiple of the interval with warm-up and batch size satisfied.
            if (len(self.agent.replay_buffer) >= max(self.config.replay_warmup, self.agent.batch_size)
                    and self.environment_steps % self.config.optimization_interval == 0):
                statistics = self.agent.optimize_step()
                losses.append(statistics["loss"])
                gradient_norms.append(statistics["gradient_norm"])
            state = next_state
            reward_sum += reward
            steps += 1
        return {
            "episode": self.episode, "environment_steps": self.environment_steps,
            **episode_metrics(adapter, reward_sum, steps), "epsilon_start": epsilon_start,
            "epsilon": self.config.epsilon_at(self.environment_steps),
            "episode_optimizer_updates": len(losses), "optimizer_updates": self.agent.optimizer_steps,
            # Missing statistics are empty cells, never misleading zero losses or NaNs.
            "mean_loss": fmean(losses) if losses else "",
            "mean_gradient_norm": fmean(gradient_norms) if gradient_norms else "",
        }

    def run(self, output_dir, verbose=True):
        if self._started:
            raise RuntimeError("Use a new Trainer for each run")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reserved = ["training_metrics.csv", "validation_metrics.csv", "training_config.json",
                    "best_model.pt", "final_model.pt", "best_model.json", "final_model.json",
                    "training_reward.png", "training_waiting_time.png", "validation_waiting_time.png",
                    "dqn_loss.png", "epsilon_decay.png"]
        if any((output_dir / name).exists() for name in reserved):
            raise FileExistsError("Output directory already contains training artifacts; choose a fresh directory")
        self._started = True
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.set_num_threads(self.config.torch_threads)
            torch.use_deterministic_algorithms(True)
            write_json(output_dir / "training_config.json", self._configuration())
            with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as training_stream, \
                    (output_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as validation_stream:
                training_writer = csv.DictWriter(training_stream, fieldnames=TRAINING_FIELDS)
                validation_writer = csv.DictWriter(validation_stream, fieldnames=VALIDATION_FIELDS)
                training_writer.writeheader()
                validation_writer.writeheader()
                self._validate(output_dir, validation_writer, validation_stream, verbose)
                adapter = RLAdapter()
                for self.episode in range(1, self.config.episodes + 1):
                    row = self._train_episode(adapter)
                    self.training_metrics.append(row)
                    training_writer.writerow(row)
                    training_stream.flush()
                    if verbose and (self.episode % 25 == 0 or self.episode == self.config.episodes):
                        print(f"Training episode={self.episode} steps={self.environment_steps} "
                              f"updates={self.agent.optimizer_steps} epsilon={row['epsilon']:.4f}", flush=True)
                    if self.episode % self.config.validation_interval == 0:
                        self._validate(output_dir, validation_writer, validation_stream, verbose)
                # Also evaluate the final policy when it falls between intervals.
                if self.validation_metrics[-1]["episode"] != self.episode:
                    self._validate(output_dir, validation_writer, validation_stream, verbose)
                self._save_checkpoint(output_dir, "final_model", self.validation_metrics[-1])
            generate_plots(output_dir)
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        return {
            "episodes": self.episode, "environment_steps": self.environment_steps,
            "optimizer_updates": self.agent.optimizer_steps,
            "epsilon": self.config.epsilon_at(self.environment_steps),
            "best_validation_episode": self.best_validation["episode"],
            "best_validation_mean_waiting_time": self.best_validation["mean_waiting_time"],
            "final_validation_mean_waiting_time": self.validation_metrics[-1]["mean_waiting_time"],
            "output_dir": str(output_dir.resolve()),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use 20 episodes unless --episodes is supplied")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/training"))
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-workloads", type=int, default=128)
    parser.add_argument("--replay-warmup", type=int, default=1000)
    parser.add_argument("--optimization-interval", type=int, default=4)
    parser.add_argument("--initial-epsilon", type=float, default=1.0)
    parser.add_argument("--minimum-epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    episodes = args.episodes if args.episodes is not None else (20 if args.smoke else 6000)
    config_values = vars(args).copy()
    config_values.pop("smoke")
    config_values.pop("output_dir")
    config_values["episodes"] = episodes
    summary = Trainer(TrainingConfig(**config_values)).run(args.output_dir)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
