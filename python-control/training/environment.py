from typing import Any

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from shapely import box
from shapely.geometry import Point

from bridge import (
    CMD_RESTART,
    MAX_OBSTACLES,
    OBSTACLE_RECT,
    GameMode,
    GdBridge,
    ObstacleCategory,
    ObstacleKind,
)
from profiling import profiler

from .globals import *

# seconds of game time per tick; matches the step size in geode-bridge/src/main.cpp
TICK_SECONDS = 1 / 30

assert OBSTACLE_SLOTS == MAX_OBSTACLES, (
    "training OBSTACLE_SLOTS must match bridge MAX_OBSTACLES"
)
assert OBSTACLE_FEATURES == len(ObstacleKind) + len(ObstacleCategory) + 5, (
    "kind one-hot + category one-hot + x, y, w, h + present"
)
assert PLAYER_FEATURES == len(GameMode) + 3 + 3, (
    "gamemode one-hot + y, vel_x, vel_y + gravity_flipped, is_touching_ground, present"
)


# custom env documentation: https://gymnasium.farama.org/introduction/create_custom_env/
class GdEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "players": gym.spaces.Box(-1e5, 1e5, shape=(MAX_PLAYERS, PLAYER_FEATURES)),
            "obstacles": gym.spaces.Box(
                -1e5,
                1e5,
                shape=(
                    OBSTACLE_SLOTS,
                    OBSTACLE_FEATURES,
                ),
            ),
            "globals": gym.spaces.Box(
                0.0, 1e5, shape=(GLOBAL_FEATURES,), dtype=np.float32
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
            "obstacles": self._obstacle_obs(px, py),
            "globals": np.array([self.bridge.attempt_ms / 1000], dtype=np.float32),
        }

    def _player_obs(self, py: float) -> np.ndarray:
        return np.array(
            [
                np.concatenate(
                    (
                        onehot(p.gamemode),
                        [p.y - py, p.vel_x, p.vel_y],
                        [
                            p.gravity_flipped,
                            p.is_touching_ground,
                            p.present,
                        ],
                    )
                )
                for p in (self.bridge.p1, self.bridge.p2)
            ],
            dtype=np.float32,
        )

    def _obstacle_obs(self, px: float, py: float) -> np.ndarray:
        # fixed shape: unused rows are all zero, i.e. present = 0
        out = np.zeros((OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=np.float32)
        obstacles = self.bridge.obstacles
        n = len(obstacles)
        if not n:
            return out
        rows = np.arange(n)
        kind_end = len(ObstacleKind)
        coords = kind_end + len(ObstacleCategory)
        out[rows, obstacles["kind"]] = 1
        out[rows, kind_end + obstacles["category"]] = 1
        out[:n, coords] = obstacles["x"] - px
        out[:n, coords + 1] = obstacles["y"] - py
        out[:n, coords + 2] = obstacles["w"]
        out[:n, coords + 3] = obstacles["h"]
        out[:n, OBSTACLE_FEATURES - 1] = 1  # present
        return out

    def _get_info(self):
        return {"progress": self.bridge.attempt_ms / 1000}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        with profiler.phase("env/reset"):
            self.bridge.set_step_mode(True)
            self.bridge.wait_cmd(CMD_RESTART)
            return self._get_obs(), self._get_info()

    @staticmethod
    def dist_betw(obstacle: np.void, player_points: list[float]):
        player_box = box(*player_points)  # pyright: ignore
        x, y, w, h = (float(obstacle[k]) for k in ("x", "y", "w", "h"))
        if obstacle["kind"] == OBSTACLE_RECT:
            rect = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)  # pyright: ignore
            return rect.distance(player_box)
        return player_box.distance(Point(x, y).buffer(w))

    def step(self, action: list[bool]):
        with profiler.phase("env/reward"):
            reward = TICK_SECONDS
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

            # minimum distance between either player hitbox and the closest obstacle
            # default=inf covers empty levels (no obstacles) and frames where both players are absent
            min_distance = min(
                (
                    min(
                        (
                            GdEnv.dist_betw(obstacle, player_points)
                            for player_points in player_coordinates
                        ),
                        default=float("inf"),
                    )
                    for obstacle in self.bridge.obstacles
                ),
                default=float("inf"),
            )

            # reward clearance of up to 100 pixels, scaled appropriately
            reward += min(min_distance, 100) * 2e-6

        with profiler.phase("env/bridge_step"):
            self.bridge.set_inputs(*action)
            self.bridge.wait_steps(1)

        with profiler.phase("env/observe"):
            obs = self._get_obs()

        return (
            obs,
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
