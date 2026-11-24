import re
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from profiling import profiler

CHECKPOINT_NAME = re.compile(r"steps_(\d+)_ticks_\d+\.pt")


def load_latest_checkpoint(model, directory: str) -> bool:
    """Restore policy weights and step count from the newest checkpoint in `directory`.

    Returns False, leaving `model` untouched, if there is no checkpoint.
    """
    found = []
    for path in Path(directory).glob("steps_*.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match:
            found.append((int(match[1]), path))
    if not found:
        return False

    _, latest = max(found)
    checkpoint = torch.load(latest, map_location="cpu")
    model.policy.load_state_dict(checkpoint["policy_state_dict"])
    model.num_timesteps = checkpoint["num_timesteps"]
    print(f"Resumed from {latest} at {model.num_timesteps} steps")
    return True


class BestSurvivalCheckpoint(BaseCallback):
    """Every `every` env steps, save the policy that survived the most ticks so far."""

    def __init__(self, directory: str, every: int = 4096):
        super().__init__()
        self.directory = Path(directory)
        self.every = every
        self.best_ticks = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.last_bucket = 0

    def _on_training_start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episode_ticks = np.zeros(self.training_env.num_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        self.episode_ticks += 1
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            if self.episode_ticks[i] > self.best_ticks:
                self.best_ticks = int(self.episode_ticks[i])
                with profiler.phase("checkpoint/copy_best"):
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.policy.state_dict().items()
                    }
            self.episode_ticks[i] = 0

        bucket = self.num_timesteps // self.every
        if bucket > self.last_bucket:
            self.last_bucket = bucket
            self._save(bucket * self.every)
        return True

    def _save(self, step: int) -> None:
        if self.best_state is None:
            return
        with profiler.phase("checkpoint/save"):
            torch.save(
                {
                    "num_timesteps": step,
                    "best_ticks": self.best_ticks,
                    "policy_state_dict": self.best_state,
                },
                self.directory / f"steps_{step:08d}_ticks_{self.best_ticks}.pt",
            )
