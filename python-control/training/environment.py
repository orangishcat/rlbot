from typing import Any

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from shapely import LineString, box
from shapely.geometry import Point

from bridge import (
    CMD_RESTART,
    FACE_FLAT,
    MAX_FACES,
    FaceCategory,
    FaceKind,
    GameMode,
    GdBridge,
)

from .globals import *

assert FACE_SLOTS == MAX_FACES, "training FACE_SLOTS must match bridge MAX_FACES"
assert FACE_FEATURES == len(FaceKind) + len(FaceCategory) + 5, (
    "kind one-hot + category one-hot + x1, y1, x2, y2 + present"
)
assert PLAYER_FEATURES == len(GameMode) + 3 + 2, (
    "gamemode one-hot + y, vel_x, vel_y + gravity_flipped, present"
)


# custom env documentation: https://gymnasium.farama.org/introduction/create_custom_env/
class GdEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "players": gym.spaces.Box(-1e5, 1e5, shape=(MAX_PLAYERS, PLAYER_FEATURES)),
            "faces": gym.spaces.Box(
                -1e5,
                1e5,
                shape=(
                    FACE_SLOTS,
                    FACE_FEATURES,
                ),
            ),
            "globals": gym.spaces.Box(
                0.0, np.inf, shape=(GLOBAL_FEATURES,), dtype=np.float32
            ),
        }
    )
    action_space = gym.spaces.MultiBinary(MAX_PLAYERS)  # key held for each player
    bridge = GdBridge()

    def __init__(self) -> None:
        super().__init__()
        print("Waiting for bridge connection...")
        self.bridge.connect()
        assert self.bridge.connected, (
            "GD bridge connection failed. Is GD open and in a level?"
        )

    def _get_obs(self):
        px, py = self.bridge.p1.x, self.bridge.p1.y
        return {
            "players": self._player_obs(py),
            "faces": self._face_obs(px, py),
            "globals": np.array(
                [self.bridge.attempt_ms / 1000], dtype=np.float32
            ),
        }

    def _player_obs(self, py: float) -> np.ndarray:
        return np.array(
            [
                np.concatenate(
                    (
                        onehot(p.gamemode),
                        [p.y - py, p.vel_x, p.vel_y],
                        [p.gravity_flipped, p.present],
                    )
                )
                for p in (self.bridge.p1, self.bridge.p2)
            ],
            dtype=np.float32,
        )

    def _face_obs(self, px: float, py: float) -> np.ndarray:
        # fixed shape: unused rows are all zero, i.e. present = 0
        out = np.zeros((FACE_SLOTS, FACE_FEATURES), dtype=np.float32)
        faces = self.bridge.faces
        n = len(faces)
        if not n:
            return out
        rows = np.arange(n)
        kind_end = len(FaceKind)
        coords = kind_end + len(FaceCategory)
        out[rows, faces["kind"]] = 1
        out[rows, kind_end + faces["category"]] = 1
        out[:n, coords] = faces["x1"] - px
        out[:n, coords + 1] = faces["y1"] - py
        out[:n, coords + 2] = faces["x2"] - px
        out[:n, coords + 3] = faces["y2"] - py
        out[:n, FACE_FEATURES - 1] = 1  # present
        return out

    def _get_info(self):
        return {"progress": self.bridge.attempt_ms / 1000}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.bridge.set_step_mode(True)
        self.bridge.wait_cmd(CMD_RESTART)

        return self._get_obs(), self._get_info()

    @staticmethod
    def dist_betw(face: np.void, player_points: list[float]):
        player_box = box(*player_points)  # pyright: ignore
        if face["kind"] == FACE_FLAT:
            return LineString(
                [
                    (float(face["x1"]), float(face["y1"])),
                    (float(face["x2"]), float(face["y2"])),
                ]
            ).distance(player_box)
        return player_box.distance(
            Point(float(face["x1"]), float(face["y1"])).buffer(float(face["x2"]))
        )

    def step(self, action: list[bool]):
        reward = self.bridge.attempt_ms / 1000
        if action[0]:
            reward -= 0.001  # encourage less clicks

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
        # default=inf covers empty levels (no faces) and frames where both players are absent
        min_distance = min(
            (
                min(
                    (
                        GdEnv.dist_betw(face, player_points)
                        for player_points in player_coordinates
                    ),
                    default=float("inf"),
                )
                for face in self.bridge.faces
            ),
            default=float("inf"),
        )

        # reward clearance of up to 100 pixels, scaled appropriately
        reward += min(min_distance, 100) * 2e-6

        self.bridge.set_inputs(*action)
        self.bridge.wait_steps(1)

        return (
            self._get_obs(),
            reward,
            self.bridge.player_destroyed,
            False,
            self._get_info(),
        )


def create_env():
    gym.register(id="GeometryDash", entry_point=GdEnv, max_episode_steps=1000)  # pyright: ignore[reportArgumentType]

    gd_env = gym.make("GeometryDash")
    check_env(gd_env.unwrapped)
    return gd_env
