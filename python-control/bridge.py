"""Shared-memory bridge reader/writer.

MIRRORS geode-bridge/src/Shared.hpp -- keep in sync.
"""

import os
import struct
import time
import warnings
from enum import IntEnum

import numpy as np

from profiling import profiler

BRIDGE_PATH = os.path.expanduser(
    "~/Library/Application Support/GeometryDash/rlbot_bridge.bin"
)

MAGIC = 0x52424447  # "GDBR"
VERSION = 1
MAX_FACES = 256


class FaceKind(IntEnum):
    FLAT = 0  # line segment (x1, y1) - (x2, y2)
    CIRCLE = 1  # circle at (x1, y1) with radius x2


class FaceCategory(IntEnum):
    SOLID = 0  # collideable, harmless (blocks, slopes)
    HAZARD = 1  # kills the player (spikes, saws)
    INTERACT = 2  # interactible (portals, pads, orbs, coins)


FACE_FLAT = FaceKind.FLAT
FACE_CIRCLE = FaceKind.CIRCLE
CAT_SOLID = FaceCategory.SOLID
CAT_HAZARD = FaceCategory.HAZARD
CAT_INTERACT = FaceCategory.INTERACT


class GameMode(IntEnum):
    CUBE = 0
    SHIP = 1
    BALL = 2
    UFO = 3
    WAVE = 4
    ROBOT = 5
    SPIDER = 6
    SWING = 7


# header: magic, version, state_seq, input_seq, two_player, face_count, attempt_ms
HEADER = struct.Struct("<IIQQIII")
HEADER_SIZE = HEADER.size  # 36
PLAYER = struct.Struct(
    "<BBBx6f"
)  # 28 bytes: gamemode, gravity, present, pad, x, y, size_x, size_y, vel_x, vel_y
FACE = struct.Struct("<BB2x4f")  # 20 bytes, pad skipped

# numpy mirror of FACE, given explicit offsets to match the C layout
FACE_DTYPE = np.dtype(
    {
        "names": ["kind", "category", "x1", "y1", "x2", "y2"],
        "formats": ["u1", "u1", "<f4", "<f4", "<f4", "<f4"],
        "offsets": [0, 1, 4, 8, 12, 16],
    }
)
assert FACE_DTYPE.itemsize == FACE.size

FACES_OFFSET = HEADER_SIZE + PLAYER.size * 2  # 92
INPUTS_OFFSET = FACES_OFFSET + FACE.size * MAX_FACES
TOTAL_SIZE = INPUTS_OFFSET + 8

INPUTS = struct.Struct("<BB")

# -- frame stepping control (after the inputs, 4 bytes of padding) --
BYTE = struct.Struct("<B")
U32 = struct.Struct("<I")
STEP_MODE_OFFSET = INPUTS_OFFSET + 2
PLAYER_DESTROYED_OFFSET = INPUTS_OFFSET + 3
STEP_REQ_OFFSET = INPUTS_OFFSET + 8
STEP_DONE_OFFSET = INPUTS_OFFSET + 12
TOTAL_SIZE = STEP_DONE_OFFSET + 4

# -- command channel (after stepDone) --
# commands
CMD_RESTART = 1
CMD_OPEN_LEVEL = 2
# status codes
CMD_OK = 0
CMD_NOT_FOUND = 1  # no level with that name
CMD_NO_LEVEL = 2  # not inside a playable level
CMD_UNKNOWN = 3  # unrecognized command id
CMD_NAME_LEN = 64  # includes NUL terminator
CMD_REQ_OFFSET = STEP_DONE_OFFSET + 4  # INPUTS_OFFSET + 16
CMD_DONE_OFFSET = CMD_REQ_OFFSET + 4
CMD_OFFSET = CMD_DONE_OFFSET + 4
CMD_STATUS_OFFSET = CMD_OFFSET + 1
CMD_NAME_OFFSET = CMD_STATUS_OFFSET + 3
TOTAL_SIZE = CMD_NAME_OFFSET + CMD_NAME_LEN


class PlayerState:
    __slots__ = (
        "gamemode",
        "gravity_flipped",
        "present",
        "x",
        "y",
        "size_x",
        "size_y",
        "vel_x",
        "vel_y",
    )

    def __init__(self):
        self.gamemode = GameMode.CUBE
        self.gravity_flipped = False
        self.present = False
        self.x = self.y = 0.0
        self.size_x = self.size_y = 0.0
        self.vel_x = self.vel_y = 0.0

    @property
    def is_touching_ground(self) -> bool:
        return self.present and self.y - self.size_y / 2 <= 91


class GdBridge:
    """Reads game state written by the Geode mod, writes inputs for it."""

    def __init__(self):
        self._mm = None
        self._last_state_seq = -1
        self.p1 = PlayerState()
        self.p2 = PlayerState()
        self.two_player = False
        self.attempt_ms = 0
        self._faces = np.zeros(MAX_FACES, dtype=FACE_DTYPE)
        self.face_count = 0
        self.state_fresh = False
        self.step_mode = False
        self.player_destroyed = False
        self.step_req = 0
        self.step_done = 0
        self.cmd_req = 0
        self.cmd_done = 0
        self.cmd_status = CMD_OK

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
            print("Magic or version does not match")
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
        with profiler.phase("bridge/poll"):
            if not self._mm:
                return False
            magic, version, seq, _, two, face_count, attempt_ms = HEADER.unpack_from(
                self._mm
            )
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
            self.attempt_ms = attempt_ms
            self.step_mode = bool(snap[STEP_MODE_OFFSET])
            self.player_destroyed = bool(snap[PLAYER_DESTROYED_OFFSET])
            self.step_req = U32.unpack_from(snap, STEP_REQ_OFFSET)[0]
            self.step_done = U32.unpack_from(snap, STEP_DONE_OFFSET)[0]
            self.cmd_req = U32.unpack_from(snap, CMD_REQ_OFFSET)[0]
            self.cmd_done = U32.unpack_from(snap, CMD_DONE_OFFSET)[0]
            self.cmd_status = snap[CMD_STATUS_OFFSET]

            self._read_player(self.p1, HEADER_SIZE)
            self._read_player(self.p2, HEADER_SIZE + PLAYER.size)

            self.face_count = min(face_count, MAX_FACES)
            if self.face_count:
                self._faces[: self.face_count] = np.frombuffer(
                    snap, FACE_DTYPE, self.face_count, FACES_OFFSET
                )
            return True

    @property
    def faces(self) -> np.ndarray:
        """This frame's face records, in shared-memory layout.

        A view into a reused buffer: read it before the next poll, do not retain it.
        """
        return self._faces[: self.face_count]

    def _read_player(self, player: PlayerState, offset: int):
        assert self._mm is not None
        gamemode, gravity, present, x, y, size_x, size_y, vel_x, vel_y = (
            PLAYER.unpack_from(self._mm, offset)
        )
        player.gamemode = GameMode(gamemode)
        player.gravity_flipped = bool(gravity)
        player.present = bool(present)
        player.x, player.y = x, y
        player.size_x, player.size_y = size_x, size_y
        player.vel_x, player.vel_y = vel_x, vel_y

    # -- writing --------------------------------------------------------

    def set_inputs(self, p1_held: bool, p2_held: bool):
        """Write hold inputs and bump the sequence counter."""
        if not self._mm:
            return
        INPUTS.pack_into(
            self._mm, INPUTS_OFFSET, 1 if p1_held else 0, 1 if p2_held else 0
        )
        # re-read header, bump input_seq
        magic, version, state_seq, input_seq, two, count, attempt_ms = (
            HEADER.unpack_from(self._mm)
        )
        HEADER.pack_into(
            self._mm,
            0,
            magic,
            version,
            state_seq,
            input_seq + 1,
            two,
            count,
            attempt_ms,
        )

    def set_step_mode(self, on: bool):
        """Tell the mod whether frame stepping is active. Enabling it makes
        the mod restart the level and freeze its own clock until stepped."""
        if not self._mm:
            return
        BYTE.pack_into(self._mm, STEP_MODE_OFFSET, 1 if on else 0)

    def request_steps(self, n: int):
        """Request n physics ticks (only has an effect while step mode is on)."""
        if not self._mm:
            return None
        req = U32.unpack_from(self._mm, STEP_REQ_OFFSET)[0]
        U32.pack_into(self._mm, STEP_REQ_OFFSET, target := (req + n))
        return target

    def wait_steps(self, n: int, timeout: float = 5.0) -> bool:
        """Request n physics ticks and block until the mod has executed them.
        Only completes while step mode is on. True if all n finished in time."""
        if not self._mm:
            return False
        with profiler.phase("bridge/wait_steps"):
            target = self.request_steps(n)
            assert target is not None
            start = time.perf_counter()
            while time.perf_counter() - start < timeout:
                self.poll()
                if self.step_done >= target:
                    return True
        warnings.warn("Waiting for steps timed out")
        return False

    def request_command(self, cmd: int, name: str = "") -> int | None:
        """Send a command to the mod (CMD_RESTART or CMD_OPEN_LEVEL).

        Returns the request number; cmd_done == that number once the mod acks it.
        """
        if not self._mm:
            return None
        raw = name.encode()[: CMD_NAME_LEN - 1]
        self._mm[CMD_NAME_OFFSET : CMD_NAME_OFFSET + CMD_NAME_LEN] = raw.ljust(
            CMD_NAME_LEN, b"\0"
        )
        BYTE.pack_into(self._mm, CMD_OFFSET, cmd)
        BYTE.pack_into(self._mm, CMD_STATUS_OFFSET, CMD_OK)
        req = U32.unpack_from(self._mm, CMD_REQ_OFFSET)[0] + 1
        U32.pack_into(self._mm, CMD_REQ_OFFSET, req)
        return req

    def wait_cmd(
        self, cmd: int, name: str = "", timeout: float = 5.0, log: bool = False
    ) -> bool:
        """Send a command and block until the mod acks it. True if it succeeded."""
        req = self.request_command(cmd, name)
        start = time.perf_counter()
        while (elapsed := time.perf_counter() - start) < timeout:
            self.poll()
            if self.cmd_done == req:
                if log:
                    print(f"Command {cmd} took {elapsed * 1e3:.2f}ms")
                return self.cmd_status == CMD_OK
        warnings.warn("Waiting for command execution timed out")
        return False

    def close(self):
        self.set_inputs(False, False)
        if self._mm:
            self._mm.close()
            self._mm = None
