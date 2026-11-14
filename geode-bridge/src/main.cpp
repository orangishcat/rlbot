#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Shared.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace geode::prelude;

static bridge::SharedMemory g_shm;
static bool g_shmFailed = false;

// last input state we applied, so we only queue on change (255 = unknown)
static uint8_t g_lastP1 = 255;
static uint8_t g_lastP2 = 255;
// step mode as seen by the mod, to detect off -> on transitions
static bool g_lastStepMode = false;
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

static bridge::GameMode gameModeOf(PlayerObject *player) {
  if (player->m_isShip)
    return bridge::GameMode::Ship;
  if (player->m_isBall)
    return bridge::GameMode::Ball;
  if (player->m_isBird)
    return bridge::GameMode::Ufo;
  if (player->m_isDart)
    return bridge::GameMode::Wave;
  if (player->m_isRobot)
    return bridge::GameMode::Robot;
  if (player->m_isSpider)
    return bridge::GameMode::Spider;
  if (player->m_isSwing)
    return bridge::GameMode::Swing;
  return bridge::GameMode::Cube;
}

static void fillPlayer(bridge::PlayerState &out, PlayerObject *player,
                       bool present) {
  out = bridge::PlayerState{};
  out.present = present ? 1 : 0;
  if (!present || !player)
    return;
  out.gamemode = gameModeOf(player);
  out.gravityFlipped = player->m_isUpsideDown;
  auto pos = player->getPosition();
  out.x = pos.x;
  out.y = pos.y;
  out.angle = player->getRotation();
}

// category for a gameplay object; 255 = not interactible/collideable, skip.
static uint8_t categoryForType(GameObjectType t) {
  switch (t) {
  case GameObjectType::Hazard:
  case GameObjectType::AnimatedHazard:
    return bridge::CAT_HAZARD;

  // portals
  case GameObjectType::InverseGravityPortal:
  case GameObjectType::NormalGravityPortal:
  case GameObjectType::ShipPortal:
  case GameObjectType::CubePortal:
  case GameObjectType::InverseMirrorPortal:
  case GameObjectType::NormalMirrorPortal:
  case GameObjectType::BallPortal:
  case GameObjectType::RegularSizePortal:
  case GameObjectType::MiniSizePortal:
  case GameObjectType::UfoPortal:
  case GameObjectType::DualPortal:
  case GameObjectType::SoloPortal:
  case GameObjectType::WavePortal:
  case GameObjectType::RobotPortal:
  case GameObjectType::TeleportPortal:
  case GameObjectType::SpiderPortal:
  case GameObjectType::SwingPortal:
  case GameObjectType::GravityTogglePortal:
  // pads
  case GameObjectType::YellowJumpPad:
  case GameObjectType::PinkJumpPad:
  case GameObjectType::GravityPad:
  case GameObjectType::RedJumpPad:
  case GameObjectType::SpiderPad:
  // rings/orbs
  case GameObjectType::YellowJumpRing:
  case GameObjectType::PinkJumpRing:
  case GameObjectType::GravityRing:
  case GameObjectType::RedJumpRing:
  case GameObjectType::GreenRing:
  case GameObjectType::DropRing:
  case GameObjectType::CustomRing:
  case GameObjectType::DashRing:
  case GameObjectType::GravityDashRing:
  case GameObjectType::SpiderOrb:
  case GameObjectType::TeleportOrb:
  // collectibles
  case GameObjectType::SecretCoin:
  case GameObjectType::Collectible:
  case GameObjectType::UserCoin:
    return bridge::CAT_INTERACT;

  // purely visual / non-physical
  case GameObjectType::Decoration:
  case GameObjectType::Modifier:
  case GameObjectType::EnterEffectObject:
    return 255;

  // Solid, Slope, Breakable, CollisionObject, Special and anything else
  default:
    return bridge::CAT_SOLID;
  }
}

static void writeState(PlayLayer *pl) {
  auto *d = g_shm.data();
  fillPlayer(d->p1, pl->m_player1, true);

  bool dual = pl->m_gameState.m_isDualMode;
  fillPlayer(d->p2, pl->m_player2, dual);

  d->twoPlayer =
      (pl->m_levelSettings && pl->m_levelSettings->m_twoPlayerMode) ? 1 : 0;
  d->attemptMs = static_cast<uint32_t>(pl->m_attemptTime * 1000.0);
  // set by destroyPlayer, cleared by the game's object reset
  d->playerDestroyed = pl->m_player1->m_isDead ? 1 : 0;

  // nearest faces: real hitbox faces (flat segments or circles) of nearby
  // interactible/collideable objects
  auto p1 = pl->m_player1->getPosition();
  auto p2pos = pl->m_player2->getPosition();

  struct Tmp {
    std::vector<bridge::HitFace> faces;
    float dist;
  };
  static std::vector<Tmp> tmp;
  tmp.clear();

  for (auto *go : CCArrayExt<GameObject *>(pl->m_objects)) {
    if (go->m_isDisabled || go->m_isGroupDisabled || go->m_isGroupDisabledTemp)
      continue;

    uint8_t cat = categoryForType(go->getType());
    if (cat == 255)
      continue;

    // circular hitboxes are the in-game hitbox for round objects
    // (orbs, saws, ...); everything else uses the hitbox rect
    float radius = go->getObjectRadius();
    float minX, minY, maxX, maxY, cx = 0.f, cy = 0.f;
    if (radius > 0.f) {
      auto c = go->getPosition();
      cx = c.x;
      cy = c.y;
      minX = cx - radius;
      maxX = cx + radius;
      minY = cy - radius;
      maxY = cy + radius;
    } else {
      auto rect = go->getObjectRect(); // gameplay hitbox rect, world space
      if (rect.size.width <= 0.f && rect.size.height <= 0.f)
        continue;
      minX = rect.getMinX();
      maxX = rect.getMaxX();
      minY = rect.getMinY();
      maxY = rect.getMaxY();
    }

    float distX = 1e18f;
    float distY = 1e18f;
    for (auto pp : {p1, p2pos}) {
      float dx = std::max(0.f, std::max(minX - pp.x, pp.x - maxX));
      float dy = std::max(0.f, std::max(minY - pp.y, pp.y - maxY));
      distX = std::min(distX, std::abs(dx));
      distY = std::min(distY, std::abs(dy));
    }
    if (distX > 800.f || distY > 500 || maxX < p1.x)
      continue;

    std::vector<bridge::HitFace> objectFaces;
    if (radius > 0.f) {
      objectFaces.push_back({bridge::FACE_CIRCLE, cat, 0, cx, cy, radius, 0.f});
    } else {
      std::vector<cocos2d::CCPoint> vertices;
      if (go->getType() == GameObjectType::Slope) {
        // Slopes are triangular hitboxes. GD stores their orientation in
        // m_slopeDirection / m_slopeUphill and computes the diagonal with
        // slopeYPos(); OBB2D only describes the enclosing rectangle.
        bool floorTop = go->slopeFloorTop();
        bool wallLeft = go->slopeWallLeft();
        float floorY = floorTop ? maxY : minY;
        float wallX = wallLeft ? minX : maxX;
        float tipY = static_cast<float>(go->slopeYPos(wallX));
        if (tipY < minY || tipY > maxY || tipY == floorY)
          tipY = floorTop ? minY : maxY;
        vertices = {{minX, floorY}, {maxX, floorY}, {wallX, tipY}};
      } else {
        // Preserve GD's oriented rectangular hitbox for other objects.
        go->updateOrientedBox();
        auto *obb = go->getOrientedBox();
        if (obb) {
          for (auto &point : obb->m_corners)
            vertices.push_back(point);
        } else {
          vertices = {{minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}};
        }
      }
      // The shared face list is edge-based, so any polygon can be sent as
      // consecutive vertex pairs without changing the Python protocol.
      for (size_t i = 0; i < vertices.size(); ++i) {
        auto &a = vertices[i];
        auto &b = vertices[(i + 1) % vertices.size()];
        objectFaces.push_back({bridge::FACE_FLAT, cat, 0, a.x, a.y, b.x, b.y});
      }
    }
    if (!objectFaces.empty())
      tmp.push_back({std::move(objectFaces), distX});
  }

  std::sort(tmp.begin(), tmp.end(),
            [](const Tmp &a, const Tmp &b) { return a.dist < b.dist; });
  size_t count = 0;
  for (const auto &object : tmp) {
    if (count + object.faces.size() > bridge::MAX_FACES)
      continue;
    for (const auto &face : object.faces)
      d->faces[count++] = face;
  }

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
    if (g_shm.ready() && g_shm.data()->stepMode)
      return; // frame stepping owns physics (see BridgePhysics::update)

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

// Frame-stepping gate. While stepMode is set, the game's own clock is blocked
// and every tick Python requested via stepReq is simulated with a fixed dt of
// 1/240 (TPS-240 equivalent), with state written back after each tick so
// Python can read the new hitboxes before requesting the next one.
class $modify(BridgePhysics, GJBaseGameLayer) {
  void update(float dt) {
    auto *pl = typeinfo_cast<PlayLayer *>(this);
    if (!pl || !g_shm.ready()) {
      GJBaseGameLayer::update(dt);
      return;
    }
    auto *d = g_shm.data();

    if (d->stepMode != g_lastStepMode) {
      if (d->stepMode) {
        // entering step mode: restart the attempt so Python starts clean
        pl->resetLevel();
        d->stepReq = 0;
        d->stepDone = 0;
        g_lastP1 = g_lastP2 = 255;
        writeState(pl);
      }
      g_lastStepMode = d->stepMode;
    }

    if (!d->stepMode) {
      GJBaseGameLayer::update(dt);
      return;
    }

    // drain requested ticks; capped per render frame so a runaway request
    // count can't freeze the game (the rest drains over the next frames)
    uint32_t guard = 0;
    while (d->stepDone != d->stepReq && guard++ < 4096) {
      applyInputs(pl);
      GJBaseGameLayer::update(1.f / 240.f); // one physics tick
      d->stepDone++;
      writeState(pl);
    }
  }
};
