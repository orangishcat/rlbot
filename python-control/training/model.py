import gymnasium as gym
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn
from torch_geometric.nn.models.meta import MetaLayer
from torch_geometric.utils import scatter

from profiling import profiler

from .globals import FACE_FEATURES, GLOBAL_FEATURES, MAX_PLAYERS, PLAYER_FEATURES
from .json_log import JsonlOutputFormat

# node: rel x1, rel y1, rel x2, rel y2, type
NODE_FEATURES = 4 + 1
# edge: rel x1, rel y1, rel x2, rel y2, distance, type (0 face-face, 1 face-player, 2 player-player)
EDGE_FEATURES = 4 + 1 + 1
# width of the vector handed to the policy / LSTM
FEATURES_DIM = 128
# internal width of the GNN MLPs
GNN_HIDDEN = 32
# message passing rounds
GNN_ROUNDS = 2
# incoming edges per node: each node hears from its nearest neighbours only
K_NEIGHBORS = 8

# observation layout, from the comments in globals.py: the coordinates come right
# after the one-hot blocks, and `present` is always the last column
FACE_COORDS = FACE_FEATURES - 5  # x1, y1, x2, y2, present
FACE_PRESENT = FACE_FEATURES - 1
PLAYER_Y = (
    PLAYER_FEATURES - 6
)  # y, vel_x, vel_y, gravity_flipped, is_touching_ground, present
PLAYER_PRESENT = PLAYER_FEATURES - 1

assert FACE_COORDS + 4 == FACE_PRESENT, "face coord/present layout changed"
assert PLAYER_Y + 5 == PLAYER_PRESENT, "player y/present layout changed"


# Implementation of GNN from https://arxiv.org/abs/1806.01261
# by Pytorch Geometric at https://github.com/pyg-team/pytorch_geometric/blob/master/test/nn/models/test_meta.py
class EdgeModel(torch.nn.Module):
    def __init__(self, dim_e, dim_v, dim_u, hidden=GNN_HIDDEN):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim_v + dim_e + dim_u, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_e),
        )

    def forward(
        self,
        src: Tensor,
        dst: Tensor,
        edge_attr: Tensor | None,
        u: Tensor | None,
        batch: Tensor | None,
    ) -> Tensor:
        assert edge_attr is not None
        assert u is not None
        assert batch is not None
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, dim_e, dim_v, dim_u, hidden=GNN_HIDDEN):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(dim_v + dim_e, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_v),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(2 * dim_v + dim_u, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_v),
        )

    def forward(
        self,
        node_attr: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        u: Tensor | None,
        batch: Tensor | None,
    ) -> Tensor:
        assert edge_attr is not None
        assert u is not None
        assert batch is not None
        row = edge_index[0]
        col = edge_index[1]
        out = torch.cat([node_attr[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=node_attr.size(0), reduce="mean")
        out = torch.cat([node_attr, out, u[batch]], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, dim_e, dim_v, dim_u, hidden=GNN_HIDDEN):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(dim_u + dim_v, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_u),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        u: Tensor | None,
        batch: Tensor | None,
    ) -> Tensor:
        assert u is not None
        assert batch is not None
        out = torch.cat(
            [
                u,
                scatter(x, batch, dim=0, dim_size=u.size(0), reduce="mean"),
            ],
            dim=1,
        )
        return self.global_mlp(out)


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
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = FEATURES_DIM,
        gnn_hidden: int = GNN_HIDDEN,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.gnn = MetaLayer(
            EdgeModel(EDGE_FEATURES, NODE_FEATURES, GLOBAL_FEATURES, gnn_hidden),
            NodeModel(EDGE_FEATURES, NODE_FEATURES, GLOBAL_FEATURES, gnn_hidden),
            GlobalModel(EDGE_FEATURES, NODE_FEATURES, GLOBAL_FEATURES, gnn_hidden),
        )
        # readout: global state followed by one block per player slot
        self.out = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES + NODE_FEATURES * MAX_PLAYERS, features_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _nodes(faces: Tensor, players: Tensor) -> tuple[Tensor, Tensor]:
        """Padded node features (B, MAX_PLAYERS + FACE_SLOTS, 5) and their present mask.

        Players come first in each sample so their slots are stable.
        """
        p_y = players[:, :, PLAYER_Y]
        zeros = torch.zeros_like(p_y)
        # x is treated as zero, so a player is a point at (0, y)
        p_nodes = torch.stack([zeros, p_y, zeros, p_y, torch.ones_like(p_y)], dim=2)

        f_coords = faces[:, :, FACE_COORDS : FACE_COORDS + 4]
        f_nodes = torch.cat([f_coords, torch.zeros_like(f_coords[:, :, :1])], dim=2)

        nodes = torch.cat([p_nodes, f_nodes], dim=1)
        present = torch.cat(
            [players[:, :, PLAYER_PRESENT] > 0.5, faces[:, :, FACE_PRESENT] > 0.5],
            dim=1,
        )
        return nodes, present

    @staticmethod
    def _graph(faces: Tensor, players: Tensor):
        """Batch -> (x, edge_index, edge_attr, batch, player_rows) over present nodes only.

        Nodes are compacted across the batch. Each node receives from its K_NEIGHBORS
        nearest other present nodes in the same sample. `player_rows` maps each player
        slot to a node row, -1 if absent.
        """
        nodes, present = GdFeatureExtractor._nodes(faces, players)
        B, N = present.shape
        device = nodes.device

        flat_present = present.reshape(-1)
        # padded flat index -> compact node row
        compact = torch.cumsum(flat_present, 0) - 1
        x = nodes.reshape(-1, NODE_FEATURES)[flat_present]
        batch = torch.arange(B, device=device).repeat_interleave(N)[flat_present]
        is_player = (torch.arange(N, device=device) < MAX_PLAYERS).repeat(B)[
            flat_present
        ]

        xy = nodes[:, :, :2]
        dist = torch.cdist(xy, xy)
        dist = dist.masked_fill(~present.unsqueeze(1), float("inf"))
        dist = dist.masked_fill(
            torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0), float("inf")
        )
        k = min(K_NEIGHBORS, N - 1)
        vals, src = dist.topk(k, dim=2, largest=False)  # (B, N, k)
        dst = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
        keep = (present.unsqueeze(2) & torch.isfinite(vals)).reshape(-1)
        src = compact[(src + offset).reshape(-1)[keep]]
        dst = compact[(dst + offset).reshape(-1)[keep]]
        edge_index = torch.stack([src, dst])

        rel = x[dst, :4] - x[src, :4]
        edge_dist = (x[dst, :2] - x[src, :2]).norm(dim=1, keepdim=True)
        edge_type = is_player[src].to(x.dtype) + is_player[dst].to(x.dtype)
        edge_attr = torch.cat([rel, edge_dist, edge_type.unsqueeze(1)], dim=1)

        slots = torch.arange(MAX_PLAYERS, device=device)
        player_rows = torch.where(
            present[:, :MAX_PLAYERS],
            compact[(torch.arange(B, device=device) * N).unsqueeze(1) + slots],
            -1,
        )
        return x, edge_index, edge_attr, batch, player_rows

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        faces: Tensor = obs["faces"]  # (B, FACE_SLOTS, FACE_FEATURES)
        players: Tensor = obs["players"]  # (B, MAX_PLAYERS, PLAYER_FEATURES)
        u: Tensor = obs["globals"]  # (B, GLOBAL_FEATURES)
        batch_size = faces.shape[0]

        x, edge_index, edge_attr, batch, player_rows = self._graph(faces, players)

        for _ in range(GNN_ROUNDS):
            x, edge_attr, u = self.gnn(x, edge_index, edge_attr, u, batch)

        valid = player_rows >= 0
        if x.shape[0]:
            per_player = x[player_rows.clamp(min=0)] * valid.unsqueeze(-1)
        else:
            per_player = x.new_zeros((batch_size, MAX_PLAYERS, NODE_FEATURES))

        return self.out(torch.cat([u, per_player.flatten(1)], dim=1))


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
        device="mps",
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
