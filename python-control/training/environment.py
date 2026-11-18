from typing import Any

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from shapely import LineString, box
from shapely.geometry import Point

from bridge import CMD_RESTART, FACE_FLAT, Face, GdBridge

from .globals import *


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
        self._last_inputs = [False for _ in range(MAX_PLAYERS)]
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
        self.bridge.set_step_mode(True)
        self.bridge.wait_cmd(CMD_RESTART)

        return self._get_obs(), self._get_info()

    @staticmethod
    def dist_betw(face: Face, player_points: list[float]):
        player_box = box(*player_points)  # pyright: ignore
        if face.kind == FACE_FLAT:
            return LineString([face.x1, face.y1, face.x2, face.y2]).distance(player_box)
        return player_box.distance(Point(face.x1, face.y1).buffer(face.y1))

    def step(self, action: list[bool]):
        reward = self.bridge.attempt_ms / 1000
        if not self._last_inputs[0] and action[0]:
            reward -= 0.01  # encourage less clicks

        player_coordinates = [
            [
                p.x - p.size_x / 2,
                p.y - p.size_y / 2,
                p.x + p.size_x / 2,
                p.y + p.size_y / 2,
            ]
            for p in [self.bridge.p1, self.bridge.p2]
            if p.present
        ]

        # minimum distance between either player hitbox and the closest point on the closest face
        min_distance = min(
            (
                min(
                    (
                        GdEnv.dist_betw(face, player_points)
                        for player_points in player_coordinates
                    ),
                )
                for face in self.bridge.faces
            ),
        )

        # reward clearance of up to 100 pixels, scaled appropriately
        reward += min(min_distance, 100) * 2e-6

        self._last_inputs = action
        self.bridge.set_inputs(*action)
        self.bridge.wait_steps(1)

        return (
            self._get_obs(),
            reward,
            self.bridge.player_destroyed,
            False,
            self._get_info(),
        )


def init():
    gym.register(id="GeometryDash", entry_point=GdEnv, max_episode_steps=1000)  # pyright: ignore[reportArgumentType]

    gd_env = gym.make("GeometryDash")
    check_env(gd_env)
    return gd_env
