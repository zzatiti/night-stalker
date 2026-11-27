"""First-person horror engine: the abandoned school.

Continuous player, hidden keys, two kinds of hiding places, and a ghost whose brain is the
HMM belief tracker (belief.py) plus a memory of how YOU like to hide (ghostmemory.py).

Coordinates: x = column, y = row (floats; a cell is 1x1 with its centre at +0.5).
Grid cells used by the AI are (row, col) tuples, as in world.py.

Ghost brains
    "hmm"        HMM tracking + learns your hiding habits across rounds   (default)
    "hmm_nomem"  the same HMM ghost but it forgets everything each round  (for comparison)
    "patrol"     baseline: wanders, chases on sight, checks spots uniformly at random
    "omniscient" cheater: always knows exactly where you are, even when hidden
"""
import math
import random
from collections import deque

import numpy as np

import school
from belief import BeliefTracker
from ghostmemory import HidingMemory
from school import FLOOR
from world import astar, visible_mask

# ---- tuning knobs ---------------------------------------------------------
WALK, SPRINT, SNEAK = 2.4, 4.2, 1.3        # player speed, cells / second
SPRINT_SECONDS = 5.0
GHOST_HUNT, GHOST_CHASE = 2.3, 3.1         # ghost speed: searching / has you in sight
VISION_LIGHT_ON, VISION_LIGHT_OFF = 9, 5   # how far the ghost sees you (cells)
NOISE = {"sprint": (0.7, 15), "walk": (0.2, 7), "sneak": (0.0, 0), "idle": (0.0, 0)}
NOISE_SIGMA = 2.0
SCENT_PERIOD, SCENT_SIGMA = 3.0, 6.0       # blurry "prey sense" (blocked while hiding)
SMOOTH_STEPS, DIST_SCALE = 3, 14.0
CATCH_DIST = 0.8
HIDE_BLOCK = 1.2                           # can't hide this soon after the ghost saw you
GRACE = 10.0
TICK = 0.1
PLAYER_R = 0.22
KEYS_TOTAL = 3
SEARCH_TIME = 1.7                          # seconds to search a container
REACH, FACING = 1.9, 0.6                   # interaction distance / cos(angle) of the view cone
INSPECT_TIME, FIND_PROB = 1.8, 0.9         # ghost checks a hiding spot; chance it spots you
PLAN_LEN, RANDOM_CHECK_P = 4, 0.03         # spots per search / chance of a random check per tick
HOP_RATE, HOP_RATE_CHASE, LAND_FRAC = 1.5, 2.1, 0.3   # pocong hops per second; fraction of a hop spent landing
FREEZE_P = 0.012                           # per tick: the ghost suddenly stops and stands still (unsettling)
KEY_KIND_WEIGHTS = {"bookshelf": 3, "storage": 2, "desk": 3, "table": 2, "cupboard": 2}
BEAM_HALF_ANGLE, BEAM_RANGE = math.radians(13), 9.0   # your flashlight beam, as the ghost experiences it
NOTHING = ["Just old books.", "Dust and a dead moth.", "Nothing but dried glue and chalk.",
           "Empty. Something skittered inside.", "Only mouldy paper."]

_SCHOOL = None
_CACHE = {"vis": {}, "dist": {}, "smooth": None}


def _data():
    global _SCHOOL
    if _SCHOOL is None:
        _SCHOOL = school.build()
    return _SCHOOL


class Game:
    def __init__(self, brain="hmm", seed=None, memory=None, round_no=1):
        self.rng = random.Random(seed)
        self.brain = brain
        d = _data()
        self.H, self.W = school.H, school.W
        self.grid, self.solid, self.tall = d["grid"], d["solid"], d["tall"]
        self.lights = d["lights"]
        self.door = d["door"]
        self.objects = [dict(o) for o in d["objects"]]
        self.hide_spots = [o for o in self.objects if o["hide_type"]]
        self.px, self.py = d["player"]
        self.gx, self.gy = d["ghost"]
        self.free_cells = [(r, c) for r in range(self.H) for c in range(self.W) if not self.solid[r, c]]
        self.memory = memory if memory is not None else HidingMemory()
        self.round = round_no

        self.angle = 0.0
        self.stamina, self.exhausted = 1.0, False
        self.flashlight = True
        self.mode, self.moving = "idle", False
        self.hidden, self.hide_obj, self.hide_kind, self.hide_exit = False, None, None, None
        self.hide_blend = 0.0
        self.round_hides, self._cur_hide = [], None   # what the ghost will learn when the round ends
        self.searching = None
        self.keys_total, self.keys_got = KEYS_TOTAL, 0
        self._place_keys()

        self.tracker = BeliefTracker(self.solid) if brain.startswith("hmm") else None
        self.path, self.target = [], None
        self.plan, self.inspect, self.checked = [], None, {}
        self.last_seen, self.patrol_goal = None, None
        self.last_seen_cell, self._planned_for = None, None
        self.ghost_sees, self.ghost_state, self.ghost_lit = False, "SEARCHING", False
        self.ghost_moving = False
        self.last_seen_t = -999.0
        self.pending_noises = []
        self.last_noise = self.last_scent = None
        self._scent_clock = self._whisper_clock = self._pstep = 0.0
        self.hop_clock, self.freeze_until = 0.0, 0.0
        self.events = []

        self.t, self._acc = 0.0, 0.0
        self.status = "playing"            # playing | caught | escaped
        self.message, self.message_until = "", 0.0
        self.say("Round %d. Find %d keys hidden around the school, then reach the exit door." %
                 (self.round, self.keys_total), 8)

    # ------------------------------------------------------------- helpers
    @property
    def pcell(self):
        return int(self.py), int(self.px)

    @property
    def gcell(self):
        return int(self.gy), int(self.gx)

    @property
    def threat(self):
        return max(0.0, 1.0 - math.hypot(self.gx - self.px, self.gy - self.py) / 12.0)

    def say(self, text, seconds=3.0):
        self.message, self.message_until = text, self.t + seconds

    def _blocked(self, x, y):
        R = PLAYER_R
        for sx in (-1, 1):
            for sy in (-1, 1):
                c = min(max(int(x + sx * R), 0), self.W - 1)
                r = min(max(int(y + sy * R), 0), self.H - 1)
                if self.solid[r, c]:
                    return True
        return False

    def _visible(self, cell, radius):
        key = (cell, radius)
        if key not in _CACHE["vis"]:
            _CACHE["vis"][key] = visible_mask(self.tall, cell, radius)
        return _CACHE["vis"][key]

    def _dist_map(self, cell):
        if cell not in _CACHE["dist"]:
            dist = np.full((self.H, self.W), 1e6)
            dist[cell] = 0
            queue = deque([cell])
            while queue:
                r, c = queue.popleft()
                for dr, dc in school.DIRS:
                    n = (r + dr, c + dc)
                    if not self.solid[n] and dist[n] > dist[r, c] + 1:
                        dist[n] = dist[r, c] + 1
                        queue.append(n)
            _CACHE["dist"][cell] = dist
        return _CACHE["dist"][cell]

    def _smooth(self):
        if _CACHE["smooth"] is None:
            _CACHE["smooth"] = np.linalg.matrix_power(self.tracker.T, SMOOTH_STEPS)
        return _CACHE["smooth"]

    def _noisy(self, cell, sigma=NOISE_SIGMA):
        for _ in range(10):
            r = round(cell[0] + self.rng.gauss(0, sigma))
            c = round(cell[1] + self.rng.gauss(0, sigma))
            if 0 <= r < self.H and 0 <= c < self.W and not self.solid[r, c]:
                return (r, c)
        return cell

    def _emit_noise(self, cell, sigma=NOISE_SIGMA):
        self.pending_noises.append((cell, sigma))
        self.last_noise = (cell, self.t)

    def ghost_dist(self):
        return math.hypot(self.gx - self.px, self.gy - self.py)

    # --------------------------------------------------------- key placement
    def _place_keys(self):
        """Hide the keys in random containers the player can actually reach."""
        reach = self._dist_map((int(self.py), int(self.px)))
        ghost_d = self._dist_map((int(self.gy), int(self.gx)))
        pool = {}
        for o in self.objects:
            if not o["can_search"]:
                continue
            fd = min(reach[r, c] for r, c in o["fronts"]) if o["fronts"] else 1e6
            gd = min(ghost_d[r, c] for r, c in o["fronts"]) if o["fronts"] else 0
            if fd < 1e5 and fd >= 8 and gd >= 6:
                pool.setdefault(o["kind"], []).append(o)
        chosen = []
        for _ in range(self.keys_total):
            for _attempt in range(60):
                kinds = [k for k in pool if pool[k]]
                kind = self.rng.choices(kinds, [KEY_KIND_WEIGHTS[k] for k in kinds])[0]
                o = self.rng.choice(pool[kind])
                far = all(math.hypot(o["centre"][0] - c["centre"][0], o["centre"][1] - c["centre"][1]) >= 7
                          for c in chosen)
                if o not in chosen and (far or _attempt > 40):
                    chosen.append(o)
                    break
        for o in chosen:
            o["has_key"] = True
        self.key_objects = chosen

    # -------------------------------------------------------- interactions
    def facing_object(self):
        best = None
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        for o in self.objects:
            dx, dy = o["centre"][0] - self.px, o["centre"][1] - self.py
            dist = math.hypot(dx, dy)
            if dist <= REACH and dist > 1e-6 and (dx * ca + dy * sa) / dist >= FACING:
                if best is None or dist < best[0]:
                    best = (dist, o)
        return best[1] if best else None

    def prompt(self):
        """Text for the on-screen cue when something can be interacted with."""
        if self.status != "playing":
            return ""
        if self.hidden:
            return "[E] Come out"
        if self.searching:
            return "Searching..."
        o = self.facing_object()
        if o is None:
            return ""
        if o["kind"] == "cupboard":
            return "[E] Hide in cupboard   [Q] Search it" if not o["searched"] else "[E] Hide in cupboard"
        if o["kind"] in ("desk", "table"):
            return "[E] Hide under %s   [Q] Search %s" % (o["kind"], "drawer" if o["kind"] == "desk" else "table")
        if o["searched"]:
            return ""
        return "[E] Search %s" % o["kind"]

    def interact(self):                    # E
        if self.status != "playing":
            return
        if self.hidden:
            self._leave_hiding()
            return
        o = self.facing_object()
        if o is None:
            return
        if o["hide_type"]:
            self._hide(o)
        elif o["can_search"]:
            self._start_search(o)

    def search_key(self):                  # Q (desk drawers and tables)
        if self.status != "playing" or self.hidden:
            return
        o = self.facing_object()
        if o is not None and o["can_search"]:
            self._start_search(o)

    def _start_search(self, o):
        if o["searched"]:
            self.say("You already searched this.", 2)
            return
        self.searching = {"obj": o, "t": 0.0, "noise": 0.0}
        self.events.append("search")

    def _finish_search(self):
        o = self.searching["obj"]
        self.searching = None
        o["searched"] = True
        if o["has_key"]:
            o["has_key"] = False
            self.keys_got += 1
            self.events.append("key")
            self._emit_noise(self._noisy(self.pcell, 1.2))     # the key clinks loudly
            left = self.keys_total - self.keys_got
            self.say("You found a key! (%d/%d)  It clinked loudly..." % (self.keys_got, self.keys_total) if left
                     else "That was the last key. Get to the exit door!", 4)
        else:
            self.events.append("nothing")
            self.say(self.rng.choice(NOTHING), 2.5)

    def _hide(self, o):
        if self.t - self.last_seen_t < HIDE_BLOCK:
            self.say("It is watching you - you can't hide!", 2)
            return
        kind = o["hide_type"]
        app = min(o["fronts"], key=lambda rc: math.hypot(rc[1] + .5 - self.px, rc[0] + .5 - self.py))
        self.hide_exit = (app[1] + 0.5, app[0] + 0.5)
        cx, cy = o["centre"]
        self.angle = math.atan2(self.hide_exit[1] - cy, self.hide_exit[0] - cx)
        if kind == "desk":
            self.px, self.py = cx, cy
        else:
            self.px, self.py = self.hide_exit
        self.hidden, self.hide_obj, self.hide_kind = True, o, kind
        self.searching = None
        # The ghost cannot see you slip in, so this goes into its memory only when the round ends.
        self._cur_hide = {"kind": kind, "id": o["id"], "found": False}
        self.round_hides.append(self._cur_hide)
        self.events.append("hide_" + kind)
        if self.rng.random() < 0.6 and self.ghost_dist() < 9:
            self._emit_noise(self._noisy(self.pcell, 1.0))     # a creak: the ghost may hear it
        self.say("Hiding%s... stay quiet. [E] to come out." % (" in the cupboard" if kind == "cupboard"
                                                              else " under the desk"), 3)

    def _leave_hiding(self):
        if self.hide_kind == "desk":
            self.px, self.py = self.hide_exit
        self.hidden = False
        self.events.append("unhide")

    # ---------------------------------------------------------------- update
    def update(self, dt, forward=0.0, strafe=0.0, turn=0.0, sprint=False, crouch=False):
        if self.status != "playing":
            return
        dt = min(dt, 0.05)
        self.t += dt
        self.angle = (self.angle + turn) % (2 * math.pi)
        target_blend = 1.0 if self.hidden else 0.0
        self.hide_blend += max(-1.0, min(1.0, (target_blend - self.hide_blend) * 100)) * dt / 0.45
        self.hide_blend = min(1.0, max(0.0, self.hide_blend))

        self.moving, self.mode = False, "idle"
        if not self.hidden:
            wants = forward != 0 or strafe != 0
            can_sprint = sprint and wants and not crouch and not self.exhausted
            if can_sprint:
                speed, self.mode = SPRINT, "sprint"
                self.stamina = max(0.0, self.stamina - dt / SPRINT_SECONDS)
                self.exhausted = self.exhausted or self.stamina == 0.0
            else:
                speed, self.mode = (SNEAK, "sneak") if crouch else (WALK, "walk")
                self.stamina = min(1.0, self.stamina + dt * 0.15)
                if self.exhausted and self.stamina >= 0.3:
                    self.exhausted = False
            if wants:
                ca, sa = math.cos(self.angle), math.sin(self.angle)
                vx, vy = forward * ca - strafe * sa, forward * sa + strafe * ca
                norm = math.hypot(vx, vy)
                dx, dy = vx / norm * speed * dt, vy / norm * speed * dt
                ox, oy = self.px, self.py
                if not self._blocked(self.px + dx, self.py):
                    self.px += dx
                if not self._blocked(self.px, self.py + dy):
                    self.py += dy
                moved = math.hypot(self.px - ox, self.py - oy)
                self.moving = moved > 1e-4
                self._pstep += moved
                if self._pstep > {"sprint": 1.1, "walk": 0.9, "sneak": 0.6}[self.mode]:
                    self._pstep = 0.0
                    self.events.append("pstep_" + self.mode)
            if not self.moving:
                self.mode = "idle"
            self._update_search(dt, wants)
            self._door_check()

        self._acc += dt
        while self._acc >= TICK and self.status == "playing":
            self._acc -= TICK
            self._ai_tick()
        if self.t >= GRACE and self.status == "playing":
            self._ghost_move(dt)
        if (self.status == "playing" and not self.hidden and self.ghost_dist() < CATCH_DIST):
            self._finish("caught", "It got you.")
        self._ambience()

    def _update_search(self, dt, wants_move):
        s = self.searching
        if not s:
            return
        o = s["obj"]
        dx, dy = o["centre"][0] - self.px, o["centre"][1] - self.py
        dist = math.hypot(dx, dy)
        cos = (dx * math.cos(self.angle) + dy * math.sin(self.angle)) / max(dist, 1e-6)
        if wants_move or dist > REACH + 0.3 or cos < 0.4:
            self.searching = None
            self.say("You stopped searching.", 1.5)
            return
        s["t"] += dt
        s["noise"] += dt
        if s["noise"] >= 0.5:                      # rummaging is not silent
            s["noise"] = 0.0
            self.events.append("rummage")
            if self.rng.random() < 0.35 and self.ghost_dist() < 8:
                self._emit_noise(self._noisy(self.pcell, 1.0))
        if s["t"] >= SEARCH_TIME:
            self._finish_search()

    def _door_check(self):
        if self.door and math.hypot(self.door[0] - self.px, self.door[1] - self.py) < 1.3:
            if self.keys_got == self.keys_total:
                self._finish("escaped", "You escaped!")
            else:
                self.say("The exit is locked. Keys: %d/%d" % (self.keys_got, self.keys_total), 2)

    def _finish(self, status, text):
        if self.status != "playing":
            return
        self.status = status
        self.message, self.message_until = text, 1e9
        for h in self.round_hides:               # now the ghost "reviews" how you hid this round
            self.memory.record_hide(h["kind"], h["id"], self.round)
            self.memory.record_outcome(h["kind"], h["id"], h["found"])
        self.memory.end_round()                  # decay old habits, then the next round starts
        self.events.append("scare" if status == "caught" else "win")

    def _ambience(self):
        if self.rng.random() < 0.0035:
            self.events.append("ambient")
        self._whisper_clock += 1 / 30
        if 0.15 < self.threat < 0.95 and not self.ghost_sees and self._whisper_clock > 3.5 + 3 * (1 - self.threat):
            self._whisper_clock = 0.0
            self.events.append("whisper")

    # ----------------------------------------------------------------- ghost
    def _ai_tick(self):
        pcell, gcell = self.pcell, self.gcell
        radius = VISION_LIGHT_ON if self.flashlight else VISION_LIGHT_OFF
        vis = self._visible(gcell, radius)
        self.ghost_lit = self._beam_on_ghost()
        # it spots you if you are in view, or if you shine your flashlight straight at it
        seen = (not self.hidden) and (bool(vis[pcell[0] * self.W + pcell[1]]) or self.ghost_lit)
        self.ghost_sees = seen
        if seen:
            self.last_seen_t, self.last_seen_cell = self.t, pcell
            self.plan, self.inspect = [], None
        elif not self.inspect and self.t > GRACE and self.rng.random() < FREEZE_P:
            self.freeze_until = self.t + self.rng.uniform(0.4, 1.3)     # it just... stops, and listens

        prob, radius_hear = NOISE[self.mode] if self.moving else (0.0, 0)
        if prob and self.rng.random() < prob and self.ghost_dist() <= radius_hear:
            self._emit_noise(self._noisy(pcell))
        self._scent_clock += TICK
        if self._scent_clock >= SCENT_PERIOD and not self.hidden:
            self._scent_clock = 0.0
            cell = self._noisy(pcell, SCENT_SIGMA)
            self.pending_noises.append((cell, SCENT_SIGMA))
            self.last_scent = (cell, self.t)

        tr = self.tracker
        if tr is not None:
            tr.predict()
            for cell, sigma in self.pending_noises:
                tr.observe_noise(cell, sigma)
            if seen:
                tr.observe_seen(pcell)
            else:
                tr.observe_not_seen(vis)
        self.pending_noises.clear()

        if self.inspect:
            self._tick_inspect()
            self.path = []
        else:
            self._maybe_plan(seen)
            if self.plan:
                o = self.plan[0]
                app = self._approach(o)
                if gcell == app:
                    self._begin_inspect(o)
                    self.path = []
                else:
                    self.target = app
                    self.path = astar(self.solid, gcell, app)
            else:
                self.target = self._hunt_target(seen, pcell, gcell, vis)
                self.path = astar(self.solid, gcell, self.target) if self.target and self.target != gcell else []

        if self.inspect or self.plan:
            self.ghost_state = "INVESTIGATING"
        elif seen:
            self.ghost_state = "CHASE"
        elif tr is not None:
            self.ghost_state = "TRACKING" if tr.belief.max() > 0.2 else "SEARCHING"
        else:
            self.ghost_state = "WANDERING"

    def _hunt_target(self, seen, pcell, gcell, vis):
        if self.brain == "omniscient":
            return pcell
        if self.tracker is None:                                  # patrol baseline
            if seen:
                self.last_seen = pcell
                return pcell
            if self.last_seen is not None:
                if gcell == self.last_seen:
                    self.last_seen = None
                else:
                    return self.last_seen
            if self.patrol_goal is None or gcell == self.patrol_goal:
                self.patrol_goal = self.rng.choice(self.free_cells)
            return self.patrol_goal
        # HMM: head for the unseen area holding the most smoothed probability mass
        mass = (self._smooth() @ self.tracker.belief).reshape(self.H, self.W)
        score = mass * np.exp(-self._dist_map(gcell) / DIST_SCALE)
        score[vis.reshape(self.H, self.W)] = 0.0
        score[self.solid] = 0.0
        if score.max() <= 0:
            return self.tracker.most_likely_cell(near=gcell)
        r, c = np.unravel_index(int(score.argmax()), score.shape)
        return int(r), int(c)

    def _beam_on_ghost(self):
        """True while your flashlight beam is pointing at the ghost (it notices you, from far)."""
        if not self.flashlight or self.hidden:
            return False
        dx, dy = self.gx - self.px, self.gy - self.py
        dist = math.hypot(dx, dy)
        if dist > BEAM_RANGE or dist < 1e-6:
            return False
        diff = (math.atan2(dy, dx) - self.angle + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > BEAM_HALF_ANGLE:
            return False
        a, b = self.pcell, self.gcell
        return bool(self._visible(a, int(BEAM_RANGE) + 1)[b[0] * self.W + b[1]])

    # ---- searching hiding places --------------------------------------
    def _approach(self, o):
        dist = self._dist_map(self.gcell)
        return min(o["fronts"], key=lambda rc: dist[rc])

    def _use_memory(self):
        return self.brain == "hmm"

    def _prior(self):
        return self.memory.type_prior() if self._use_memory() else {"cupboard": 0.5, "desk": 0.5}

    def _maybe_plan(self, seen):
        if seen or self.plan or self.t < GRACE:
            return
        if self.brain == "omniscient":
            if self.hidden and self.hide_obj is not None:
                self.plan = [self.hide_obj]
            return
        since = self.t - self.last_seen_t
        if 0.8 <= since <= 12 and self._planned_for != self.last_seen_t and self.last_seen_cell:
            self._planned_for = self.last_seen_t           # it just lost you: check likely hiding places
            self._make_plan(self.last_seen_cell)
            return
        near = [o for o in self.hide_spots if self.checked.get(o["id"], -1) < self.t
                and math.hypot(o["centre"][0] - self.gx, o["centre"][1] - self.gy) < 3.5]
        if near and self.rng.random() < RANDOM_CHECK_P:     # idle curiosity
            prior = self._prior()
            o = self.rng.choice(near)
            if self.rng.random() < min(1.0, 2 * prior[o["hide_type"]]):
                self.plan = [o]

    def _make_plan(self, centre):
        spots = [o for o in self.hide_spots if self.checked.get(o["id"], -1) < self.t]
        if not spots:
            return
        prior = self._prior()
        gdist = self._dist_map(self.gcell)
        if self.tracker is not None:
            mass = self._smooth() @ self.tracker.belief
        else:
            cdist = self._dist_map(centre)
        local, totals = {}, {}
        for o in spots:
            if self.tracker is not None:
                m = sum(mass[r * self.W + c] for r, c in o["fronts"])
            else:
                m = math.exp(-min(cdist[r, c] for r, c in o["fronts"]) / 6.0)
            gd = min(gdist[r, c] for r, c in o["fronts"])
            bonus = self.memory.spot_bonus(o["id"]) if self._use_memory() else 1.0
            local[o["id"]] = (m + 1e-3) * bonus * math.exp(-gd / 30.0)
            totals[o["hide_type"]] = totals.get(o["hide_type"], 0.0) + local[o["id"]]
        weight = {o["id"]: prior[o["hide_type"]] * local[o["id"]] / totals[o["hide_type"]] for o in spots}
        plan, pool = [], list(spots)
        for _ in range(PLAN_LEN):
            if not pool:
                break
            x, acc = self.rng.random() * sum(weight[o["id"]] for o in pool), 0.0
            for o in pool:
                acc += weight[o["id"]]
                if x <= acc:
                    plan.append(o)
                    pool.remove(o)
                    break
        self.plan = plan

    def _begin_inspect(self, o):
        self.inspect = {"obj": o, "start": self.t, "until": self.t + INSPECT_TIME + self.rng.uniform(0, 0.8)}
        self.events.append("inspect")

    def _tick_inspect(self):
        ins = self.inspect
        if self.t < ins["until"]:
            return
        o = ins["obj"]
        self.inspect = None
        if self.plan and self.plan[0] is o:
            self.plan.pop(0)
        self.checked[o["id"]] = self.t + 25
        if self.hidden and self.hide_obj is o:
            if self.rng.random() < FIND_PROB:
                self._cur_hide["found"] = True
                text = "It tore open the cupboard." if self.hide_kind == "cupboard" else "It looked under the desk."
                self._finish("caught", text)
            else:
                self.say("It stared right at you... and moved on.", 3)

    @property
    def ghost_phase(self):
        """0..1 position within the current hop (0 = just landed)."""
        return self.hop_clock % 1.0

    def _ghost_move(self, dt):
        """A pocong hops: it only travels while airborne and pauses on every landing."""
        sees = self.ghost_sees
        frozen = self.t < self.freeze_until
        phase = self.ghost_phase
        hopping = (bool(self.path) and not frozen) or phase > 0.0     # always finish a hop in progress
        self.ghost_moving = hopping
        if not hopping:
            return
        before = self.hop_clock
        self.hop_clock += dt * (HOP_RATE_CHASE if sees else HOP_RATE)
        if int(self.hop_clock) > int(before):                          # landed
            self.events.append("gstep")
            if not self.path or frozen:
                self.hop_clock = float(int(self.hop_clock))            # come to rest exactly on the landing
        if not self.path or self.ghost_phase < LAND_FRAC:              # on the ground: it only travels in the air
            return
        remaining = (GHOST_CHASE if sees else GHOST_HUNT) / (1.0 - LAND_FRAC) * dt
        while remaining > 0 and self.path:
            tx, ty = self.path[0][1] + 0.5, self.path[0][0] + 0.5
            d = math.hypot(tx - self.gx, ty - self.gy)
            if d <= remaining:
                self.gx, self.gy = tx, ty
                remaining -= d
                self.path.pop(0)
            else:
                self.gx += (tx - self.gx) / d * remaining
                self.gy += (ty - self.gy) / d * remaining
                remaining = 0

    # ----------------------------------------------------------------- debug
    def debug_info(self):
        info = {"state": self.ghost_state, "brain": self.brain, "round": self.round,
                "memory": self.memory.summary(),
                "plan": [o["kind"] for o in self.plan], "inspecting": bool(self.inspect)}
        if self.tracker is not None:
            info["belief_at_player"] = self.tracker.prob_at(self.pcell)
            info["max_belief"] = float(self.tracker.belief.max())
            info["entropy"] = self.tracker.entropy()
        return info

    def next_round(self):
        """A fresh round (new random key spots) that keeps the ghost's memory."""
        return Game(self.brain, memory=self.memory, round_no=self.round + 1)


def warmup():
    """Pre-compute the slow one-off tables (call once while a loading screen is shown)."""
    g = Game("hmm", seed=0)
    g._smooth()
    g._visible(g.gcell, VISION_LIGHT_ON)
    return g
