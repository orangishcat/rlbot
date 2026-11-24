from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


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
        torch.save(
            {
                "num_timesteps": step,
                "best_ticks": self.best_ticks,
                "policy_state_dict": self.best_state,
            },
            self.directory / f"steps_{step:08d}_ticks_{self.best_ticks}.pt",
        )
