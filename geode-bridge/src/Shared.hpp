#pragma once

// Shared memory layout for the Geode <-> Python bridge.
// MIRRORED IN python-control/bridge.py -- keep in sync.
//
// C++ writes game state (players + faces), Python writes inputs.
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
constexpr uint32_t VERSION = 1;
constexpr int MAX_FACES = 1000;

// cube, ship, ball, ufo, wave, robot, spider, swing
// all false -> gamemode nonexistent
inline constexpr const char *GAMEMODE_NAMES[8] = {
    "cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"};

struct PlayerState {
  uint8_t gamemode[8];
  uint8_t gravityFlipped; // 1 = upside down
  uint8_t present;        // 0 = nonexistent (e.g. dual off)
  uint16_t _pad;
  float x, y, angle;
}; // 24 bytes

struct Face {
  float x1, y1; // start of face
  float x2, y2; // end of face
}; // 16 bytes

struct SharedLayout {
  uint32_t magic;
  uint32_t version;
  uint64_t stateSeq;  // C++ -> Python, bumped after each state write
  uint64_t inputSeq;  // Python -> C++, bumped after each input write
  uint32_t twoPlayer; // level is 2 player
  uint32_t faceCount;
  PlayerState p1;
  PlayerState p2;
  Face faces[MAX_FACES]; // nearest 100 faces to the player
  uint8_t p1Held;
  uint8_t p2Held;
  uint8_t _pad[6];
};

inline constexpr size_t OFFSET_FACES = offsetof(SharedLayout, faces);
inline constexpr size_t OFFSET_INPUTS = offsetof(SharedLayout, p1Held);
inline constexpr size_t TOTAL_SIZE = sizeof(SharedLayout);
static_assert(sizeof(PlayerState) == 24);
static_assert(sizeof(Face) == 16);

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
