import gymnasium as gym
import torch
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn
from torch_geometric.nn.models.meta import MetaLayer
from torch_geometric.utils import scatter

EDGE_FEATURES = 6  # rel x1, rel y1, rel x2, rel y2, distance, type
NODE_FEATURES = 5  # rel x1, rel y1, rel x2, rel y2, type
GLOBAL_FEATURES = 1  # attempt secs
HIDDEN_FEATURES = 128
OUT_FEATURES = 2


# Implementation of GNN from https://arxiv.org/abs/1806.01261
# by Pytorch Geometric at https://github.com/pyg-team/pytorch_geometric/blob/master/test/nn/models/test_meta.py
class EdgeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                2 * NODE_FEATURES + EDGE_FEATURES + GLOBAL_FEATURES, HIDDEN_FEATURES
            ),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, OUT_FEATURES),
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
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(NODE_FEATURES + OUT_FEATURES, HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(
                NODE_FEATURES + HIDDEN_FEATURES + GLOBAL_FEATURES, HIDDEN_FEATURES
            ),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES),
        )

    def forward(
        self,
        x: Tensor,
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
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce="mean")
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(
                GLOBAL_FEATURES + HIDDEN_FEATURES + GLOBAL_FEATURES, HIDDEN_FEATURES
            ),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES),
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
                scatter(x, batch, dim=0, reduce="mean"),
            ],
            dim=1,
        )
        return self.global_mlp(out)


class GdFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, 128)
        self.feature_enc = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())


def create_model(env: gym.Env):
    ppo = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        learning_rate=1e-5,
        policy_kwargs={
            "features_extractor_class": GdFeatureExtractor,
        },
    )
    ppo.set_logger(
        Logger("training/results", [HumanOutputFormat("training/results/training.log")])
    )
    return ppo
