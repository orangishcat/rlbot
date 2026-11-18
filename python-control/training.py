from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from bridge import CMD_RESTART, GdBridge

FACE_SLOTS = 1000

# ---------- KEEP IN SYNC WITH BRIDGE ------------

# one hot for FACE_FLAT, FACE_CIRCLE, CAT_SOLID, CAT_HAZARD, CAT_HAZARD types (5 features)
# floats x1, y1, x2, y2   (x2 = radius and y2 = 0 for circular faces) (4 features)
# y2 = 0 is probably fine, but could consider another architecture to get rid of unused parameter
# boolean present (1 feature); since the model must receive inputs with fixed shape, some faces will be blank
FACE_FEATURES = 10

# one hot for gamemode CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER, SWING (7 features)
# floats for x, y, x velocity, y velocity (4 features)
# boolean for gravity_flipped (1 feature)
# boolean for present (for player 2; not present unless in dual)  (1 feature)
PLAYER_FEATURES = 13
MAX_PLAYERS = 2

# ------------------------------------------------


def onehot(enum: IntEnum):
    return np.eye(len(type(enum)), enum)


# custom env documentation: https://gymnasium.farama.org/introduction/create_custom_env/
class GdEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "players": gym.spaces.Box(
                -np.inf, np.inf, shape=(MAX_PLAYERS, PLAYER_FEATURES)
            ),
            "faces": gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(
                    FACE_SLOTS,
                    FACE_FEATURES,
                ),
            ),
        }
    )
    action_space = gym.spaces.MultiBinary(MAX_PLAYERS)  # key held for each player
    bridge = GdBridge()

    def __init__(self) -> None:
        super().__init__()
        self.bridge.connect()
        assert self.bridge.connected, (
            "GD bridge connection failed. Is GD open and in a level?"
        )

    def _get_obs(self):
        return {
            "players": np.fromiter(
                (
                    np.concat(
                        (
                            onehot(p.gamemode),
                            [p.x, p.y, p.vel_x, p.vel_y, p.size_x, p.size_y],
                        )
                    )
                    for p in (self.bridge.p1, self.bridge.p2)
                ),
                dtype=np.float32,
            ),
            "faces": np.fromiter(
                (
                    np.concat(
                        (onehot(f.kind), onehot(f.category), [f.x1, f.y1, f.x2, f.y2])
                    )
                    for f in self.bridge.faces
                ),
                dtype=np.float32,
            ),
        }

    def _get_info(self):
        return {"progress": self.bridge.attempt_ms / 1000}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.bridge.wait_cmd(CMD_RESTART)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action): ...


gym.register(id="GeometryDash", entry_point=GdEnv, max_episode_steps=100)  # pyright: ignore[reportArgumentType]


def main():
    gd_env = gym.make("GeometryDash")
    assert check_env(gd_env), "GD environment is invalid"


if __name__ == "main":
    main()
