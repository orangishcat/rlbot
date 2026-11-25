#pragma once

// Shared memory layout for the Geode <-> Python bridge.
// MIRRORED IN python-control/bridge.py -- keep in sync.
//
// C++ writes game state (players + obstacles), Python writes inputs.
// Each side bumps its own sequence counter after every write so the
// reader can detect complete (non-torn) writes.

#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bridge {

constexpr uint32_t MAGIC = 0x52424447; // "GDBR" little-endian
constexpr uint32_t VERSION = 3;
constexpr int MAX_OBSTACLES = 32; // nearest obstacles sent per tick
// rolling per-tick timing samples (nanoseconds) written by the frame stepper
constexpr int TICK_SAMPLES = 64;

// Obstacle::kind (rotation is ignored: rects are axis-aligned)
inline constexpr uint8_t OBSTACLE_RECT = 0;   // rect centred on (x, y), size (w, h)
inline constexpr uint8_t OBSTACLE_CIRCLE = 1; // circle centred on (x, y), radius w

// Obstacle::category
inline constexpr uint8_t CAT_SOLID =
    0; // collideable, harmless (blocks, slopes)
inline constexpr uint8_t CAT_HAZARD = 1; // kills the player (spikes, saws)
inline constexpr uint8_t CAT_INTERACT =
    2; // interactible (portals, pads, orbs, coins)

// command channel ids (Python -> C++)
inline constexpr uint8_t CMD_RESTART = 1;    // restart the current attempt
inline constexpr uint8_t CMD_OPEN_LEVEL = 2; // open a saved level by name
// command status codes (C++ -> Python)
inline constexpr uint8_t CMD_STATUS_OK = 0;
inline constexpr uint8_t CMD_STATUS_NOT_FOUND = 1; // no level with that name
inline constexpr uint8_t CMD_STATUS_NO_LEVEL = 2; // not inside a playable level
inline constexpr uint8_t CMD_STATUS_UNKNOWN = 3;  // unrecognized command id
inline constexpr int CMD_NAME_LEN = 64;           // includes NUL terminator

enum class GameMode : uint8_t {
  Cube,
  Ship,
  Ball,
  Ufo,
  Wave,
  Robot,
  Spider,
  Swing,
};

struct PlayerState {
  GameMode gamemode;
  uint8_t gravityFlipped; // 1 = upside down
  uint8_t present;        // 0 = nonexistent (e.g. dual off)
  uint8_t _pad;
  float x, y;  // player center (getPosition)
  float sizeX; // hitbox size, from the player's oriented box
  float sizeY;
  float velX; // player velocity (getCurrentXVelocity / getYVelocity)
  float velY;
}; // 28 bytes

struct Obstacle {
  uint8_t kind;     // OBSTACLE_RECT or OBSTACLE_CIRCLE
  uint8_t category; // CAT_SOLID, CAT_HAZARD or CAT_INTERACT
  uint16_t _pad;
  float x, y; // center
  float w, h; // rect: width, height / circle: w = radius, h unused (0)
}; // 20 bytes

struct SharedLayout {
  uint32_t magic;
  uint32_t version;
  uint64_t stateSeq;  // C++ -> Python, bumped after each state write
  uint64_t inputSeq;  // Python -> C++, bumped after each input write
  uint32_t twoPlayer; // level is 2 player
  uint32_t obstacleCount;
  uint32_t attemptMs; // time since current attempt started
  PlayerState p1;
  PlayerState p2;
  Obstacle obstacles[MAX_OBSTACLES]; // nearest hitboxes of interactible/collideable
                                     // objects, nearest first
  uint8_t p1Held;
  uint8_t p2Held;
  // -- frame stepping control --
  uint8_t stepMode;        // Python -> C++: 1 while frame stepping is active
  uint8_t playerDestroyed; // C++ -> Python: 1 since the last attempt reset
  uint8_t _pad[4];
  uint32_t stepReq;  // Python -> C++: total physics ticks requested
  uint32_t stepDone; // C++ -> Python: total physics ticks simulated
  // -- command channel --
  uint32_t cmdReq;   // Python -> C++: total commands requested
  uint32_t cmdDone;  // C++ -> Python: total commands processed
  uint8_t cmd;       // last requested command id (CMD_*)
  uint8_t cmdStatus; // result of the last processed command (CMD_STATUS_*)
  uint8_t _pad2[2];
  char cmdName[CMD_NAME_LEN]; // NUL-terminated level name for CMD_OPEN_LEVEL
  // -- diagnostics (C++ -> Python) --
  uint32_t objectCount;     // level objects cached in the geometry index
  uint32_t obstacleCandidates; // objects considered in the last window scan
  uint32_t indexSeq;        // bumped whenever the geometry index is rebuilt
  uint32_t tickSeq;         // ring cursor; total ticks sampled by the stepper
  uint32_t physicsNs[TICK_SAMPLES]; // GJBaseGameLayer::update duration
  uint32_t writeNs[TICK_SAMPLES];   // writeState duration
  uint32_t pumpReturns;     // times the stepper handed the frame back
  uint32_t ticksThisSession; // ticks simulated since step mode was enabled
  float nearestDist;        // player hitbox -> nearest emitted obstacle, -1 = none
  uint32_t indexBuildNs;    // duration of the last geometry index rebuild
};

inline constexpr size_t OFFSET_OBSTACLES = offsetof(SharedLayout, obstacles);
inline constexpr size_t OFFSET_INPUTS = offsetof(SharedLayout, p1Held);
inline constexpr size_t TOTAL_SIZE = sizeof(SharedLayout);
static_assert(sizeof(PlayerState) == 28);
static_assert(sizeof(Obstacle) == 20);
// keep in sync with python-control/bridge.py offsets
static_assert(offsetof(SharedLayout, stepMode) == OFFSET_INPUTS + 2);
static_assert(offsetof(SharedLayout, playerDestroyed) == OFFSET_INPUTS + 3);
static_assert(offsetof(SharedLayout, stepReq) == OFFSET_INPUTS + 8);
static_assert(offsetof(SharedLayout, stepDone) == OFFSET_INPUTS + 12);
static_assert(offsetof(SharedLayout, cmdReq) == OFFSET_INPUTS + 16);
static_assert(offsetof(SharedLayout, cmdDone) == OFFSET_INPUTS + 20);
static_assert(offsetof(SharedLayout, cmd) == OFFSET_INPUTS + 24);
static_assert(offsetof(SharedLayout, cmdStatus) == OFFSET_INPUTS + 25);
static_assert(offsetof(SharedLayout, cmdName) == OFFSET_INPUTS + 28);
// diagnostics block, mirrored in python-control/bridge.py
inline constexpr size_t OFFSET_DIAG = OFFSET_INPUTS + 28 + CMD_NAME_LEN;
static_assert(offsetof(SharedLayout, objectCount) == OFFSET_DIAG);
static_assert(offsetof(SharedLayout, obstacleCandidates) == OFFSET_DIAG + 4);
static_assert(offsetof(SharedLayout, indexSeq) == OFFSET_DIAG + 8);
static_assert(offsetof(SharedLayout, tickSeq) == OFFSET_DIAG + 12);
static_assert(offsetof(SharedLayout, physicsNs) == OFFSET_DIAG + 16);
static_assert(offsetof(SharedLayout, writeNs) ==
              OFFSET_DIAG + 16 + 4 * TICK_SAMPLES);
static_assert(offsetof(SharedLayout, pumpReturns) ==
              OFFSET_DIAG + 16 + 8 * TICK_SAMPLES);
static_assert(offsetof(SharedLayout, ticksThisSession) ==
              OFFSET_DIAG + 20 + 8 * TICK_SAMPLES);
static_assert(offsetof(SharedLayout, nearestDist) ==
              OFFSET_DIAG + 24 + 8 * TICK_SAMPLES);
static_assert(offsetof(SharedLayout, indexBuildNs) ==
              OFFSET_DIAG + 28 + 8 * TICK_SAMPLES);

inline std::string bridgePath() {
  // Must match python-control/bridge.py BRIDGE_PATH
  return std::string(getenv("HOME") ? getenv("HOME") : "/tmp") +
         "/Library/Application Support/GeometryDash/rlbot_bridge.bin";
}

class SharedMemory {
public:
  bool ensure() {
    if (m_data)
      return true;
    int fd = ::open(bridgePath().c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0)
      return false;
    if (ftruncate(fd, TOTAL_SIZE) != 0) {
      ::close(fd);
      return false;
    }
    void *ptr =
        mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED)
      return false;
    auto *d = static_cast<SharedLayout *>(ptr);
    if (d->magic != MAGIC || d->version != VERSION) {
      *d = SharedLayout{};
      d->magic = MAGIC;
      d->version = VERSION;
    }
    m_data = d;
    return true;
  }

  SharedLayout *data() { return m_data; }
  bool ready() const { return m_data != nullptr; }

private:
  SharedLayout *m_data = nullptr;
};

} // namespace bridge
