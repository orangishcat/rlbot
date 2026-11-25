from enum import IntEnum

import numpy as np

# ---------- KEEP IN SYNC WITH BRIDGE ------------

OBSTACLE_SLOTS = 32

# one hot for OBSTACLE_RECT, OBSTACLE_CIRCLE, CAT_SOLID, CAT_HAZARD, CAT_INTERACT types (5 features)
# floats x, y (relative to player), w, h   (w = radius and h = 0 for circles) (4 features)
# boolean present (1 feature); since the model must receive inputs with fixed shape, some slots will be blank
OBSTACLE_FEATURES = 10

# one hot for gamemode CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER, SWING (8 features)
# floats for y, x velocity, y velocity (3 features) (no x because player x is treated as zero)
# boolean for gravity_flipped (1 feature)
# boolean for is_touching_ground: present and bottom edge at y <= 0 (1 feature)
# boolean for present (for player 2; not present unless in dual)  (1 feature)
PLAYER_FEATURES = 14
MAX_PLAYERS = 2

# attempt seconds; not sent by the bridge, read from attempt_ms
GLOBAL_FEATURES = 1

# ------------------------------------------------


def onehot(enum: IntEnum) -> np.ndarray:
    return np.eye(len(type(enum)))[enum]
