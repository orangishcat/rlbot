from enum import IntEnum

import numpy as np

# ---------- KEEP IN SYNC WITH BRIDGE ------------

FACE_SLOTS = 1000

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
