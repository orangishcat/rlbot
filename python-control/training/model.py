import gymnasium as gym
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn
from torch_geometric.nn.models.meta import MetaLayer
from torch_geometric.utils import scatter

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
PLAYER_Y = PLAYER_FEATURES - 5  # y, vel_x, vel_y, gravity_flipped, present
PLAYER_PRESENT = PLAYER_FEATURES - 1

assert FACE_COORDS + 4 == FACE_PRESENT, "face coord/present layout changed"
assert PLAYER_Y + 4 == PLAYER_PRESENT, "player y/present layout changed"


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


class GdPolicy(RecurrentActorCriticPolicy):
    """Recurrent policy for the Dict observation, with one MultiBinary fix.

    torch>=2.15 no longer upcasts the uint8 action buffer inside
    Bernoulli.log_prob (binary_cross_entropy_with_logits), so PPO's
    evaluate_actions raises "result type Float can't be cast ... Char".
    Cast to float, which is what the logits are.
    """

    def evaluate_actions(self, obs, actions, *args, **kwargs):
        if (
            isinstance(self.action_space, gym.spaces.MultiBinary)
            and not actions.is_floating_point()
        ):
            actions = actions.float()
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
    def _graph(faces: Tensor, players: Tensor):
        """One sample -> (x, edge_index, edge_attr, player_rows).

        Only present entities become nodes. Players come first so their rows are
        stable; `player_rows` maps each player slot to a node row, -1 if absent.
        """
        present_players = players[:, PLAYER_PRESENT] > 0.5
        present_faces = faces[:, FACE_PRESENT] > 0.5

        p_y = players[present_players, PLAYER_Y]
        zeros = torch.zeros_like(p_y)
        # x is treated as zero, so a player is a point at (0, y)
        p_nodes = torch.stack([zeros, p_y, zeros, p_y, torch.ones_like(p_y)], dim=1)

        f_coords = faces[present_faces, FACE_COORDS : FACE_COORDS + 4]
        f_nodes = torch.cat(
            [
                f_coords,
                torch.zeros(
                    (f_coords.shape[0], 1), dtype=f_coords.dtype, device=f_coords.device
                ),
            ],
            dim=1,
        )

        x = torch.cat([p_nodes, f_nodes], dim=0)
        n = x.shape[0]
        n_players = p_nodes.shape[0]

        slots = torch.nonzero(present_players, as_tuple=False).flatten()
        player_rows = torch.full((MAX_PLAYERS,), -1, dtype=torch.long, device=x.device)
        player_rows[slots] = torch.arange(n_players, device=x.device)

        # each node receives from its K_NEIGHBORS nearest other nodes (by position)
        k = min(K_NEIGHBORS, n - 1)
        if k > 0:
            dist_matrix = torch.cdist(x[:, :2], x[:, :2])
            dist_matrix.fill_diagonal_(float("inf"))
            src = dist_matrix.topk(k, dim=1, largest=False).indices.flatten()
            dst = torch.arange(n, device=x.device).repeat_interleave(k)
        else:
            src = torch.zeros(0, dtype=torch.long, device=x.device)
            dst = torch.zeros(0, dtype=torch.long, device=x.device)

        rel = x[dst, :4] - x[src, :4]
        dist = (x[dst, :2] - x[src, :2]).norm(dim=1, keepdim=True)
        edge_type = (src < n_players).to(x.dtype) + (dst < n_players).to(x.dtype)
        edge_attr = torch.cat([rel, dist, edge_type.unsqueeze(1)], dim=1)

        return x, torch.stack([src, dst]), edge_attr, player_rows

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        faces: Tensor = obs["faces"]  # (B, FACE_SLOTS, FACE_FEATURES)
        players: Tensor = obs["players"]  # (B, MAX_PLAYERS, PLAYER_FEATURES)
        u: Tensor = obs["globals"]  # (B, GLOBAL_FEATURES)
        batch_size = faces.shape[0]

        xs, edge_indices, edge_attrs, batches = [], [], [], []
        player_rows = torch.full(
            (batch_size, MAX_PLAYERS), -1, dtype=torch.long, device=faces.device
        )
        offset = 0
        for b in range(batch_size):
            x, edge_index, edge_attr, rows = self._graph(faces[b], players[b])
            xs.append(x)
            edge_indices.append(edge_index + offset)
            edge_attrs.append(edge_attr)
            batches.append(
                torch.full((x.shape[0],), b, dtype=torch.long, device=x.device)
            )
            player_rows[b] = torch.where(rows >= 0, rows + offset, -1)
            offset += x.shape[0]

        x = torch.cat(xs, 0)
        edge_index = torch.cat(edge_indices, 1)
        edge_attr = torch.cat(edge_attrs, 0)
        batch = torch.cat(batches, 0)

        for _ in range(GNN_ROUNDS):
            x, edge_attr, u = self.gnn(x, edge_index, edge_attr, u, batch)

        valid = player_rows >= 0
        if x.shape[0]:
            per_player = x[player_rows.clamp(min=0)] * valid.unsqueeze(-1)
        else:
            per_player = x.new_zeros((batch_size, MAX_PLAYERS, NODE_FEATURES))

        return self.out(torch.cat([u, per_player.flatten(1)], dim=1))


def create_model(env: gym.Env):
    ppo = RecurrentPPO(
        GdPolicy,
        env,
        learning_rate=0.01,
        policy_kwargs={
            "features_extractor_class": GdFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        },
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
