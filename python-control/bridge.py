"""Shared-memory bridge reader/writer.

MIRRORS geode-bridge/src/Shared.hpp -- keep in sync.
"""

import collections
import os
import struct
import time

BRIDGE_PATH = os.path.expanduser(
    "~/Library/Application Support/GeometryDash/rlbot_bridge.bin"
)

MAGIC = 0x52424447  # "GDBR"
VERSION = 2
MAX_FACES = 1000

# HitFace.kind
FACE_FLAT = 0  # line segment (x1, y1) - (x2, y2)
FACE_CIRCLE = 1  # circle at (x1, y1) with radius x2

# HitFace.category
CAT_SOLID = 0  # collideable, harmless (blocks, slopes)
CAT_HAZARD = 1  # kills the player (spikes, saws)
CAT_INTERACT = 2  # interactible (portals, pads, orbs, coins)

GAMEMODE_NAMES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing")

# header: magic, version, state_seq, input_seq, two_player, face_count
HEADER = struct.Struct("<IIQQII")
HEADER_SIZE = HEADER.size  # 32
PLAYER = struct.Struct("<8BBBH3f")  # 24 bytes
FACE = struct.Struct("<BB2x4f")  # 20 bytes, pad skipped
FACES_OFFSET = HEADER_SIZE + PLAYER.size * 2  # 80
INPUTS_OFFSET = FACES_OFFSET + FACE.size * MAX_FACES
TOTAL_SIZE = INPUTS_OFFSET + 8

INPUTS = struct.Struct("<BB6x")

# (kind, category, x1, y1, x2, y2)
Face = collections.namedtuple(
    "Face", "kind category x1 y1 x2 y2", defaults=(0, 0, 0.0, 0.0, 0.0, 0.0)
)


class PlayerState:
    __slots__ = ("gamemode", "gravity_flipped", "present", "x", "y", "angle")

    def __init__(self):
        self.gamemode = {name: False for name in GAMEMODE_NAMES}
        self.gravity_flipped = False
        self.present = False
        self.x = self.y = self.angle = 0.0

    @property
    def gamemode_name(self):
        for name in GAMEMODE_NAMES:
            if self.gamemode[name]:
                return name
        return None


class Bridge:
    """Reads game state written by the Geode mod, writes inputs for it."""

    def __init__(self):
        self._mm = None
        self._last_state_seq = -1
        self.p1 = PlayerState()
        self.p2 = PlayerState()
        self.two_player = False
        self.faces = []  # list of Face, nearest hitbox faces
        self.state_fresh = False

    # -- connection -----------------------------------------------------

    def connect(self, timeout=30.0):
        """Wait until the mod has created and initialized the shared memory."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._open():
                    return True
            except FileNotFoundError:
                pass
            time.sleep(0.2)
        return False

    def _open(self):
        fd = os.open(BRIDGE_PATH, os.O_RDWR)
        try:
            import mmap

            self._mm = mmap.mmap(fd, TOTAL_SIZE)
        finally:
            os.close(fd)
        magic, version = HEADER.unpack_from(self._mm)[:2]
        if magic != MAGIC or version != VERSION:
            self._mm.close()
            self._mm = None
            return False
        return True

    @property
    def connected(self):
        return self._mm is not None

    # -- reading --------------------------------------------------------

    def poll(self):
        """Read one stable state snapshot. Returns True if new data arrived."""
        if not self._mm:
            return False
        magic, version, seq, _, two, face_count = HEADER.unpack_from(self._mm)
        if magic != MAGIC or version != VERSION:
            self.state_fresh = False
            return False
        if seq == self._last_state_seq:
            self.state_fresh = False
            return False
        # read snapshot; if the writer bumped the seq mid-read, retry once
        snap = bytes(self._mm)
        if HEADER.unpack_from(snap)[2] != seq:
            return self.poll()
        self._last_state_seq = seq
        self.state_fresh = True
        self.two_player = bool(two)

        self._read_player(self.p1, HEADER_SIZE)
        self._read_player(self.p2, HEADER_SIZE + PLAYER.size)

        self.faces = []
        for i in range(min(face_count, MAX_FACES)):
            self.faces.append(
                Face(*FACE.unpack_from(snap, FACES_OFFSET + i * FACE.size))
            )
        return True

    def _read_player(self, player: PlayerState, offset: int):
        vals = PLAYER.unpack_from(self._mm, offset)
        player.gamemode = {
            GAMEMODE_NAMES[i]: bool(vals[i]) for i in range(len(GAMEMODE_NAMES))
        }
        player.gravity_flipped = bool(vals[8])
        player.present = bool(vals[9])
        player.x, player.y, player.angle = vals[11], vals[12], vals[13]

    # -- writing --------------------------------------------------------

    def set_inputs(self, p1_held: bool, p2_held: bool):
        """Write hold inputs and bump the sequence counter."""
        if not self._mm:
            return
        INPUTS.pack_into(
            self._mm, INPUTS_OFFSET, 1 if p1_held else 0, 1 if p2_held else 0
        )
        # re-read header, bump input_seq
        magic, version, state_seq, input_seq, two, count = HEADER.unpack_from(self._mm)
        HEADER.pack_into(
            self._mm, 0, magic, version, state_seq, input_seq + 1, two, count
        )

    def close(self):
        self.set_inputs(False, False)
        if self._mm:
            self._mm.close()
            self._mm = None
