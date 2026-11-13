#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Shared.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

using namespace geode::prelude;

static bridge::SharedMemory g_shm;
static bool g_shmFailed = false;

// last input state we applied, so we only queue on change (255 = unknown)
static uint8_t g_lastP1 = 255;
static uint8_t g_lastP2 = 255;
static std::chrono::steady_clock::time_point g_levelStart;
static bool g_levelStartSet = false;

static double inputTimestamp() {
  if (!g_levelStartSet) {
    g_levelStart = std::chrono::steady_clock::now();
    g_levelStartSet = true;
  }
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - g_levelStart)
      .count();
}

static void fillPlayer(bridge::PlayerState &out, PlayerObject *player,
                       bool present) {
  out = bridge::PlayerState{};
  out.present = present ? 1 : 0;
  if (!present || !player)
    return;
  out.gamemode[0] =
      !(player->m_isShip || player->m_isBall || player->m_isBird ||
        player->m_isDart || player->m_isRobot || player->m_isSpider ||
        player->m_isSwing);
  out.gamemode[1] = player->m_isShip;
  out.gamemode[2] = player->m_isBall;
  out.gamemode[3] = player->m_isBird; // ufo
  out.gamemode[4] = player->m_isDart; // wave
  out.gamemode[5] = player->m_isRobot;
  out.gamemode[6] = player->m_isSpider;
  out.gamemode[7] = player->m_isSwing;
  out.gravityFlipped = player->m_isUpsideDown;
  auto pos = player->getPosition();
  out.x = pos.x;
  out.y = pos.y;
  out.angle = player->getRotation();
}

static void writeState(PlayLayer *pl) {
  auto *d = g_shm.data();
  fillPlayer(d->p1, pl->m_player1, true);

  bool dual = pl->m_gameState.m_isDualMode;
  fillPlayer(d->p2, pl->m_player2, dual);

  d->twoPlayer =
      (pl->m_levelSettings && pl->m_levelSettings->m_twoPlayerMode) ? 1 : 0;

  // nearest faces: hitbox rect edges of nearby objects
  auto p1 = pl->m_player1->getPosition();
  auto p2pos = pl->m_player2->getPosition();

  struct Tmp {
    bridge::Face face;
    float dist;
  };
  static std::vector<Tmp> tmp;
  tmp.clear();

  for (auto *go : CCArrayExt<GameObject *>(pl->m_objects)) {
    auto rect = go->getObjectRect();
    if (rect.size.width <= 0.f && rect.size.height <= 0.f)
      continue;

    float distX = 1e18f;
    float distY = 1e18f;
    for (auto pp : {p1, p2pos}) {
      float dx =
          std::max(0.f, std::max(rect.getMinX() - pp.x, pp.x - rect.getMaxX()));
      float dy =
          std::max(0.f, std::max(rect.getMinY() - pp.y, pp.y - rect.getMaxY()));
      distX = std::min(distX, std::abs(dx));
      distY = std::min(distY, std::abs(dy));
    }
    if (distX > 800.f || distY > 500 || rect.getMaxX() < p1.x)
      continue;

    float minX = rect.getMinX(), maxX = rect.getMaxX();
    float minY = rect.getMinY(), maxY = rect.getMaxY();
    tmp.push_back({{minX, minY, maxX, minY}, distX}); // bottom
    tmp.push_back({{maxX, minY, maxX, maxY}, distX}); // right
    tmp.push_back({{maxX, maxY, minX, maxY}, distX}); // top
    tmp.push_back({{minX, maxY, minX, minY}, distX}); // left
  }

  size_t count = std::min<size_t>(tmp.size(), bridge::MAX_FACES);
  std::nth_element(tmp.begin(), tmp.begin() + count, tmp.end(),
                   [](const Tmp &a, const Tmp &b) { return a.dist < b.dist; });
  for (size_t i = 0; i < count; ++i)
    d->faces[i] = tmp[i].face;

  // release-style barrier so Python never sees a torn write
  std::atomic_thread_fence(std::memory_order_release);
  d->faceCount = static_cast<uint32_t>(count);
  d->stateSeq++;
}

static void applyInputs(PlayLayer *pl) {
  auto *d = g_shm.data();
  uint8_t p1 = d->p1Held ? 1 : 0;
  uint8_t p2 = d->p2Held ? 1 : 0;
  if (p1 != g_lastP1) {
    pl->queueButton(static_cast<int>(PlayerButton::Jump), d->p1Held != 0, false,
                    inputTimestamp());
    g_lastP1 = p1;
  }
  if (p2 != g_lastP2) {
    pl->queueButton(static_cast<int>(PlayerButton::Jump), d->p2Held != 0, true,
                    inputTimestamp());
    g_lastP2 = p2;
  }
}

class $modify(BridgePlayLayer, PlayLayer) {
  void postUpdate(float dt) {
    if (!g_shm.ready()) {
      if (!g_shmFailed) {
        g_shmFailed = !g_shm.ensure();
        if (g_shmFailed)
          log::error("rlbot: failed to open shared memory at {}",
                     bridge::bridgePath());
      }
    }

    if (g_shm.ready()) {
      applyInputs(this);
      g_levelStartSet = true;
    }

    PlayLayer::postUpdate(dt);

    if (g_shm.ready()) {
      writeState(this);
    }
  }
};
