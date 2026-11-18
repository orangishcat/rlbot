import gymnasium as gym
import numpy as np

import bridge
from bridge import FaceCategory, FaceKind, GameMode, GdBridge

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
        # remember to poll
        return {
            "players": np.fromiter(
                (
                    np.concat(
                        (
                            np.eye(len(GameMode))[p.gamemode],
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
                        (
                            np.eye(len(FaceKind))[f.kind],
                            np.eye(len(FaceCategory))[f.category],
                        )
                    )
                    for f in self.bridge.faces
                ),
                dtype=np.float32,
            ),
        }


gym.register(id="GeometryDash", entry_point=GdEnv, max_episode_steps=100)  # pyright: ignore[reportArgumentType]


def main():
    gd_env = gym.make("GeometryDash")


if __name__ == "main":
    main()
