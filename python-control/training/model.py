import gymnasium as gym
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn

from profiling import profiler

from .globals import (
    GLOBAL_FEATURES,
    MAX_PLAYERS,
    OBSTACLE_FEATURES,
    OBSTACLE_SLOTS,
    PLAYER_FEATURES,
)
from .json_log import JsonlOutputFormat

# width of the vector handed to the policy / LSTM
FEATURES_DIM = 128
# width of the MLP hidden layers
MLP_HIDDEN = 256


class GRUWithCellState(nn.GRU):
    """nn.GRU that takes and returns the (h, c) tuple sb3_contrib's recurrent PPO expects.

    A GRU has no cell state, so c is passed through untouched (it stays zero).
    """

    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor]):
        h, c = hx
        output, h_n = super().forward(input, h)
        return output, (h_n, c)


class GdPolicy(RecurrentActorCriticPolicy):
    """Recurrent policy for the Dict observation, with one MultiBinary fix.

    torch>=2.15 no longer upcasts the uint8 action buffer inside
    Bernoulli.log_prob (binary_cross_entropy_with_logits), so PPO's
    evaluate_actions raises "result type Float can't be cast ... Char".
    Cast to float, which is what the logits are.

    The recurrent layers are GRUs rather than LSTMs: sb3_contrib hardcodes nn.LSTM,
    so they are swapped in here and the optimizer is rebuilt over the new parameters.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.lstm_actor = GRUWithCellState(
            self.features_dim,
            self.lstm_output_dim,
            num_layers=self.lstm_actor.num_layers,
        )
        if self.lstm_critic is not None:
            self.lstm_critic = GRUWithCellState(
                self.features_dim,
                self.lstm_output_dim,
                num_layers=self.lstm_critic.num_layers,
            )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, *args, **kwargs):
        with profiler.phase("rollout/policy_forward"):
            return super().forward(*args, **kwargs)

    def evaluate_actions(self, obs, actions, *args, **kwargs):
        if (
            isinstance(self.action_space, gym.spaces.MultiBinary)
            and not actions.is_floating_point()
        ):
            actions = actions.float()
        with profiler.phase("train/evaluate_actions"):
            return super().evaluate_actions(obs, actions, *args, **kwargs)


class GdFeatureExtractor(BaseFeaturesExtractor):
    """MLP over the OBSTACLE_SLOTS nearest obstacles, the players and the globals.

    Obstacles arrive nearest first, so each slot keeps a fixed meaning. Absent
    slots are zero rows with present = 0.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = FEATURES_DIM,
        hidden: int = MLP_HIDDEN,
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = (
            OBSTACLE_SLOTS * OBSTACLE_FEATURES
            + MAX_PLAYERS * PLAYER_FEATURES
            + GLOBAL_FEATURES
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        flat = torch.cat(
            [
                obs["obstacles"].flatten(1),  # (B, OBSTACLE_SLOTS * OBSTACLE_FEATURES)
                obs["players"].flatten(1),  # (B, MAX_PLAYERS * PLAYER_FEATURES)
                obs["globals"],  # (B, GLOBAL_FEATURES)
            ],
            dim=1,
        )
        return self.mlp(flat)


class GdRecurrentPPO(RecurrentPPO):
    """RecurrentPPO that times each phase of the training loop with `profiler`.

    `train/total` is flushed one dump late: SB3 dumps logs before `train()` runs,
    the same lag as the `train/*` metrics.
    """

    def learn(self, *args, **kwargs):
        profiler.reset()  # drop env checks and setup done before learning
        return super().learn(*args, **kwargs)

    def collect_rollouts(self, *args, **kwargs):
        with profiler.phase("rollout/total"):
            return super().collect_rollouts(*args, **kwargs)

    def train(self) -> None:
        with profiler.phase("train/total"):
            super().train()

    def dump_logs(self, iteration: int = 0) -> None:
        profiler.flush(self.logger)
        super().dump_logs(iteration)


def create_model(env: gym.Env):
    ppo = GdRecurrentPPO(
        GdPolicy,
        env,
        learning_rate=0.01,
        ent_coef=0.05,
        policy_kwargs={
            "features_extractor_class": GdFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        },
        device="cpu",
    )
    ppo.set_logger(
        Logger(
            "training/results",
            [
                JsonlOutputFormat("training/results/metrics.jsonl"),
            ],
        )
    )
    return ppo
