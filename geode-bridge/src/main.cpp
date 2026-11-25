#include <Geode/Geode.hpp>
#include <Geode/binding/GameLevelManager.hpp>
#include <Geode/binding/LocalLevelManager.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Shared.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

using namespace geode::prelude;

static bridge::SharedMemory g_shm;
static bool g_shmFailed = false;

// last input state we applied, so we only queue on change (255 = unknown)
static uint8_t g_lastP1 = 255;
static uint8_t g_lastP2 = 255;
// last command request handled, so a deferred open isn't re-queued every frame
static uint32_t g_lastCmdReq = 0;
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

static float dist(const cocos2d::CCPoint &a, const cocos2d::CCPoint &b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
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
  // hitbox size from the player's own oriented box (the game keeps it
  // current); consecutive corner distances are the box's side lengths and
  // are rotation-independent
  if (auto *obb = player->getOrientedBox()) {
    auto &c = obb->m_corners;
    out.sizeX = dist(c[0], c[1]);
    out.sizeY = dist(c[1], c[2]);
  }
  if (out.sizeX <= 0.f || out.sizeY <= 0.f) {
    // fallback: GD's 30x30 base hitbox scaled by the icon (mini = 0.6)
    out.sizeX = 30.f * player->getScaleX();
    out.sizeY = 30.f * player->getScaleY();
  }
  out.velX = player->getCurrentXVelocity();
  out.velY = player->getYVelocity();
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

  // collideable, harmless
  case GameObjectType::Solid:
  case GameObjectType::Breakable:
  case GameObjectType::Slope:
  case GameObjectType::CollisionObject:
  case GameObjectType::Special:
    return bridge::CAT_SOLID;

  // purely visual / non-physical
  case GameObjectType::Decoration:
  case GameObjectType::Modifier:
  case GameObjectType::EnterEffectObject:
    return 255;

  // Unknown types are not assumed solid: a decoration misclassified as a
  // hitbox floods the obstacle list and feeds the model phantom geometry.
  default:
    return 255;
  }
}

// ---------- geometry index ----------
//
// Scanning every object in the level on every tick dominates the per-tick
// cost on long levels. Cache the gameplay objects once per level, sorted by
// their x extent, and per tick visit only the slice that can be observed.
// Objects can move, so the slice is padded by kMoveSlack and the live rects
// of the visited objects are still read every tick.

namespace {

constexpr float kTickDt = 1.f / 30.f;
// observation window: an object farther than this from the player is not sent
constexpr float kKeepAhead = 800.f;
constexpr float kKeepVertical = 500.f;
// how far the window scan reaches beyond the observation window, so objects
// that moved since the index was built are still considered
constexpr float kMoveSlack = 400.f;
// keep the app alive / let queued main-thread work run
constexpr std::chrono::milliseconds kPumpIdleBudget(250);
constexpr std::chrono::milliseconds kPumpMaxBudget(1000);
constexpr std::chrono::microseconds kIdleSleep(50);
constexpr float kFar = 1e18f;

struct IndexEntry {
  GameObject *go;
  float minX; // bounds when the index was built; only used for window lookup
  uint8_t category;
};

struct Candidate {
  bridge::Obstacle obstacle;
  float distance;
};

struct Box {
  float minX, minY, maxX, maxY;
};

std::vector<IndexEntry> g_index;
const PlayLayer *g_indexOwner = nullptr;
unsigned g_indexObjectCount = 0;

std::vector<Candidate> g_candidates; // reused between ticks

Box playerBox(const bridge::PlayerState &p) {
  Box b;
  b.minX = p.x - p.sizeX * 0.5f;
  b.maxX = p.x + p.sizeX * 0.5f;
  b.minY = p.y - p.sizeY * 0.5f;
  b.maxY = p.y + p.sizeY * 0.5f;
  return b;
}

float pointBoxDistance(float px, float py, const Box &b) {
  float dx = std::max(0.f, std::max(b.minX - px, px - b.maxX));
  float dy = std::max(0.f, std::max(b.minY - py, py - b.maxY));
  return std::sqrt(dx * dx + dy * dy);
}

// Exact distance between two axis-aligned boxes (0 when they overlap).
float boxBoxDistance(const Box &a, const Box &b) {
  float dx = std::max(0.f, std::max(a.minX - b.maxX, b.minX - a.maxX));
  float dy = std::max(0.f, std::max(a.minY - b.maxY, b.minY - a.maxY));
  return std::sqrt(dx * dx + dy * dy);
}

float obstacleBoxDistance(const bridge::Obstacle &o, const Box &b) {
  if (o.kind == bridge::OBSTACLE_CIRCLE)
    return std::max(0.f, pointBoxDistance(o.x, o.y, b) - o.w);
  return boxBoxDistance(
      Box{o.x - o.w * 0.5f, o.y - o.h * 0.5f, o.x + o.w * 0.5f, o.y + o.h * 0.5f},
      b);
}

void buildObjectIndex(PlayLayer *pl) {
  auto start = std::chrono::steady_clock::now();
  g_index.clear();
  g_indexOwner = pl;
  g_indexObjectCount = pl->m_objects ? pl->m_objects->count() : 0;
  if (pl->m_objects) {
    g_index.reserve(g_indexObjectCount);
    for (auto *go : CCArrayExt<GameObject *>(pl->m_objects)) {
      if (!go)
        continue;
      uint8_t category = categoryForType(go->getType());
      if (category == 255)
        continue;
      float radius = go->getObjectRadius();
      float minX;
      if (radius > 0.f) {
        minX = go->getPositionX() - radius;
      } else {
        auto rect = go->getObjectRect();
        if (rect.size.width <= 0.f && rect.size.height <= 0.f)
          continue;
        minX = rect.getMinX();
      }
      g_index.push_back({go, minX, category});
    }
  }
  std::sort(
      g_index.begin(), g_index.end(),
      [](const IndexEntry &a, const IndexEntry &b) { return a.minX < b.minX; });
  if (auto *d = g_shm.data()) {
    d->objectCount = static_cast<uint32_t>(g_index.size());
    d->indexBuildNs = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    d->indexSeq++;
  }
}

bool indexValid(const PlayLayer *pl) {
  return g_indexOwner == pl && pl->m_objects &&
         static_cast<unsigned>(pl->m_objects->count()) == g_indexObjectCount;
}

void recordTick(std::chrono::nanoseconds physics,
                std::chrono::nanoseconds write) {
  auto *d = g_shm.data();
  if (!d)
    return;
  uint32_t slot = d->tickSeq % bridge::TICK_SAMPLES;
  d->physicsNs[slot] = static_cast<uint32_t>(physics.count());
  d->writeNs[slot] = static_cast<uint32_t>(write.count());
  d->tickSeq++;
}

} // namespace

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

  if (!indexValid(pl))
    buildObjectIndex(pl);

  // nearest obstacles: hitboxes (rects or circles) of nearby
  // interactible/collideable objects
  auto p1 = pl->m_player1->getPosition();
  auto p2pos = pl->m_player2->getPosition();
  Box box1 = playerBox(d->p1);
  Box box2 = dual ? playerBox(d->p2) : Box{};

  g_candidates.clear();

  // window of the x-sorted index; nearest-first selection happens below
  float scanLo = p1.x - kMoveSlack;
  float scanHi = p1.x + kKeepAhead + kMoveSlack;
  auto it =
      std::lower_bound(g_index.begin(), g_index.end(), scanLo,
                       [](const IndexEntry &e, float v) { return e.minX < v; });
  for (; it != g_index.end() && it->minX <= scanHi; ++it) {
    GameObject *go = it->go;
    if (go->m_isDisabled || go->m_isGroupDisabled || go->m_isGroupDisabledTemp)
      continue;

    bridge::Obstacle obstacle{};
    float minX, minY, maxX, maxY;
    // circular hitboxes are the in-game hitbox for round objects (orbs, saws,
    // ...); everything else uses the axis-aligned hitbox rect
    float radius = go->getObjectRadius();
    if (radius > 0.f) {
      auto c = go->getPosition();
      minX = c.x - radius;
      maxX = c.x + radius;
      minY = c.y - radius;
      maxY = c.y + radius;
      obstacle = {bridge::OBSTACLE_CIRCLE, it->category, 0,
                  c.x,                     c.y,          radius, 0.f};
    } else {
      auto rect = go->getObjectRect(); // live rect: objects can move
      if (rect.size.width <= 0.f && rect.size.height <= 0.f)
        continue;
      minX = rect.getMinX();
      maxX = rect.getMaxX();
      minY = rect.getMinY();
      maxY = rect.getMaxY();
      obstacle = {bridge::OBSTACLE_RECT, it->category, 0, rect.getMidX(),
                  rect.getMidY(), rect.size.width, rect.size.height};
    }

    float distX = 1e18f;
    float distY = 1e18f;
    float nearestSq = kFar;
    for (const auto &pp : {p1, p2pos}) {
      float dx = std::max(0.f, std::max(minX - pp.x, pp.x - maxX));
      float dy = std::max(0.f, std::max(minY - pp.y, pp.y - maxY));
      distX = std::min(distX, std::abs(dx));
      distY = std::min(distY, std::abs(dy));
      nearestSq = std::min(nearestSq, dx * dx + dy * dy);
    }
    // only player 1 defines what is "behind"; p2 is a second reference point
    if (distX > kKeepAhead || distY > kKeepVertical || maxX < p1.x)
      continue;

    g_candidates.push_back({obstacle, std::sqrt(nearestSq)});
  }

  d->obstacleCandidates = static_cast<uint32_t>(g_candidates.size());

  std::sort(g_candidates.begin(), g_candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.distance < b.distance;
            });
  size_t count = 0;
  float nearestDist = -1.f;
  for (const auto &candidate : g_candidates) {
    if (count == bridge::MAX_OBSTACLES)
      break;
    const auto &obstacle = candidate.obstacle;
    d->obstacles[count++] = obstacle;
    float dist = obstacleBoxDistance(obstacle, box1);
    if (dual)
      dist = std::min(dist, obstacleBoxDistance(obstacle, box2));
    if (nearestDist < 0.f || dist < nearestDist)
      nearestDist = dist;
  }
  d->nearestDist = nearestDist;

  // release-style barrier so Python never sees a torn write
  std::atomic_thread_fence(std::memory_order_release);
  d->obstacleCount = static_cast<uint32_t>(count);
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

static std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

// Case-insensitive lookup of a saved level by name: local levels first, then
// cached/downloaded levels.
static GJGameLevel *findLevelByName(const std::string &query) {
  std::string q = toLower(query);
  auto *llm = LocalLevelManager::sharedState();
  if (llm) {
    for (auto *lvl : CCArrayExt<GJGameLevel *>(llm->m_localLevels)) {
      if (lvl && toLower(lvl->m_levelName) == q)
        return lvl;
    }
  }
  auto *glm = GameLevelManager::sharedState();
  if (glm) {
    CCDictElement *el = nullptr;
    CCDictElement *next = nullptr;
    HASH_ITER(hh, glm->m_onlineLevels->m_pElements, el, next) {
      auto *lvl = typeinfo_cast<GJGameLevel *>(el->getObject());
      if (lvl && toLower(lvl->m_levelName) == q)
        return lvl;
    }
  }
  return nullptr;
}

// Must run on the main thread (called via queueInMainThread): opening a level
// replaces the current scene.
static void executeOpenLevel(const std::string &name, uint32_t req) {
  auto *d = g_shm.data();
  auto *lvl = findLevelByName(name);
  if (!lvl) {
    d->cmdStatus = bridge::CMD_STATUS_NOT_FOUND;
    d->cmdDone = req;
    return;
  }
  auto *pl = PlayLayer::create(lvl, false);
  if (!pl) {
    d->cmdStatus = bridge::CMD_STATUS_NO_LEVEL;
    d->cmdDone = req;
    return;
  }
  auto *scene = CCScene::create();
  scene->addChild(pl);
  CCDirector::sharedDirector()->replaceScene(scene);
  g_indexOwner = nullptr; // objects of the old level are gone
  // fresh attempt state: re-queue held inputs and reset the input clock
  g_lastP1 = g_lastP2 = 255;
  g_levelStartSet = false;
  d->cmdStatus = bridge::CMD_STATUS_OK;
  d->cmdDone = req;
}

enum class CommandResult { None, Handled, Deferred };

// Drain the command channel. Needs pl for restarts; CMD_OPEN_LEVEL works
// without one (e.g. from the editor). Deferred means the command is queued on
// the main thread and only lands after the current frame returns.
static CommandResult handleCommands(PlayLayer *pl) {
  auto *d = g_shm.data();
  if (!d || d->cmdDone == d->cmdReq || d->cmdReq == g_lastCmdReq)
    return CommandResult::None;
  g_lastCmdReq = d->cmdReq;

  uint8_t cmd = d->cmd;
  std::string name(d->cmdName, strnlen(d->cmdName, bridge::CMD_NAME_LEN));
  uint32_t req = d->cmdReq;

  if (cmd == bridge::CMD_RESTART) {
    if (pl) {
      pl->resetLevel();
      g_lastP1 = g_lastP2 = 255;
      d->cmdStatus = bridge::CMD_STATUS_OK;
      writeState(pl);
    } else {
      d->cmdStatus = bridge::CMD_STATUS_NO_LEVEL;
    }
    d->cmdDone = req;
    return CommandResult::Handled;
  }

  if (cmd == bridge::CMD_OPEN_LEVEL) {
    // scene changes must not happen mid-update
    Loader::get()->queueInMainThread(
        [name, req] { executeOpenLevel(name, req); });
    return CommandResult::Deferred;
  }

  d->cmdStatus = bridge::CMD_STATUS_UNKNOWN;
  d->cmdDone = req;
  return CommandResult::Handled;
}

class $modify(BridgePlayLayer, PlayLayer) {
  // losing focus pauses the level; a paused level never ticks, so stepping
  // would hang until the window is refocused. Manual pauses still apply.
  void pauseGame(bool unfocused) {
    if (unfocused && g_shm.ready() && g_shm.data()->stepMode)
      return;
    PlayLayer::pauseGame(unfocused);
  }

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

// Run requested ticks inside a single director frame. The game's own clock is
// blocked while stepMode is set: every tick Python requests through stepReq is
// simulated with a fixed dt and written back before stepDone advances, so
// Python can read the new hitboxes before requesting the next one. Staying
// inside update() between requests removes the render-frame rate limit; the
// budget only bounds how long the app goes without drawing or pumping events.
static void pumpSteps(PlayLayer *pl) {
  auto *d = g_shm.data();
  auto idleDeadline = std::chrono::steady_clock::now() + kPumpIdleBudget;
  auto hardDeadline = std::chrono::steady_clock::now() + kPumpMaxBudget;

  while (true) {
    // scene changes are queued on the main thread and need the frame to end
    if (handleCommands(pl) == CommandResult::Deferred)
      break;
    if (!d->stepMode)
      break;

    if (d->stepDone != d->stepReq) {
      applyInputs(pl);
      auto t0 = std::chrono::steady_clock::now();
      pl->GJBaseGameLayer::update(kTickDt); // one physics tick
      auto t1 = std::chrono::steady_clock::now();
      writeState(pl);
      auto t2 = std::chrono::steady_clock::now();
      recordTick(t1 - t0, t2 - t1);
      d->stepDone++;
      d->ticksThisSession++;
      idleDeadline = std::chrono::steady_clock::now() + kPumpIdleBudget;
      continue;
    }

    auto now = std::chrono::steady_clock::now();
    if (now >= idleDeadline || now >= hardDeadline)
      break;
    // yield the core while Python computes; the sleep bounds the latency
    std::this_thread::sleep_for(kIdleSleep);
  }
  d->pumpReturns++;
}

class $modify(BridgePhysics, GJBaseGameLayer) {
  void update(float dt) {
    auto *pl = typeinfo_cast<PlayLayer *>(this);
    if (!g_shm.ready()) {
      GJBaseGameLayer::update(dt);
      return;
    }
    auto *d = g_shm.data();

    handleCommands(pl);

    if (!pl) {
      GJBaseGameLayer::update(dt);
      return;
    }

    if (d->stepMode != g_lastStepMode) {
      if (d->stepMode) {
        // entering step mode: restart the attempt so Python starts clean
        pl->resetLevel();
        d->stepReq = 0;
        d->stepDone = 0;
        d->tickSeq = 0;
        d->ticksThisSession = 0;
        d->pumpReturns = 0;
        d->nearestDist = -1.f;
        g_lastP1 = g_lastP2 = 255;
        g_indexOwner = nullptr; // object pointers change with the restart
        writeState(pl);
      }
      g_lastStepMode = d->stepMode;
    }

    if (!d->stepMode) {
      GJBaseGameLayer::update(dt);
      return;
    }

    pumpSteps(pl);
  }
};
