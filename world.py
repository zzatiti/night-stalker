"""Game world, search utilities, and three enemy brains.

Enemy types
-----------
"patrol"     : baseline. Wanders between random waypoints, chases only when it
               actually sees the player, then checks the last seen position.
"hmm"        : the stalker. Maintains a belief over the player's cell (see
               belief.py) and walks A* toward the most likely cell.
"omniscient" : upper bound / cheater. A* straight to the true player position.

The world never gives the "hmm" or "patrol" enemy the player's position except
through the sensors in _enemy_turn (sight and noise).
"""
import heapq
import random

import numpy as np

from belief import BeliefTracker

HALL = "#" + "." * 28 + "#"
LAYOUT = [
    "##############################",
    "#P........#.........#........#",
    "#..###....#....#....#...##...#",
    "#..###....#....#....#........#",
    HALL,
    "#####.#########.#########.####",
    HALL,
    "#.......##.........##........#",
    HALL,
    "###.###########.###########.##",
    "#.........#.........#........#",
    "#..##.....#....###..#..#.....#",
    "#..##.....#....###..#..#.....#",
    "#.........#....S....#........#",
    "#" + "." * 27 + "E#",
    "##############################",
]
assert len({len(r) for r in LAYOUT}) == 1, "layout rows must have equal length"

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# movement speed in cells per tick, and (noise probability per step, hearing radius)
SPEEDS = {"sneak": 0.35, "walk": 0.6, "sprint": 1.0}
NOISE = {"sneak": (0.0, 0), "walk": (0.25, 6), "sprint": (1.0, 12)}


# ---------------------------------------------------------------- search tools
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(walls, start, goal):
    """Shortest 4-neighbour path. Returns the cells after `start`, up to goal."""
    if start == goal:
        return []
    h, w = walls.shape
    heap = [(manhattan(start, goal), 0, start)]
    came = {start: None}
    best_g = {start: 0}
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == goal:
            break
        if g > best_g[cur]:
            continue
        for dr, dc in DIRS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and not walls[nxt]:
                ng = g + 1
                if ng < best_g.get(nxt, 1e9):
                    best_g[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(heap, (ng + manhattan(nxt, goal), ng, nxt))
    if goal not in came:
        return []
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = came[cur]
    return path[::-1]


def line_of_sight(walls, a, b):
    """Bresenham line; blocked if any cell strictly between a and b is a wall."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dc - dr
    r, c = r0, c0
    while (r, c) != (r1, c1):
        if (r, c) != (r0, c0) and walls[r, c]:
            return False
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return True


def visible_mask(walls, pos, radius):
    """Flat boolean mask of free cells within `radius` that `pos` can see."""
    h, w = walls.shape
    mask = np.zeros(h * w, dtype=bool)
    r0, c0 = pos
    for r in range(max(0, r0 - radius), min(h, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(w, c0 + radius + 1)):
            if walls[r, c] or (r - r0) ** 2 + (c - c0) ** 2 > radius * radius:
                continue
            if line_of_sight(walls, pos, (r, c)):
                mask[r * w + c] = True
    return mask


# ----------------------------------------------------------------------- world
class World:
    def __init__(self, enemy="hmm", seed=None, enemy_speed=0.85, vision=8, noise_sigma=2.0,
                 enemy_start=None):
        self.rng = random.Random(seed)
        self.enemy_type = enemy
        self.enemy_speed = enemy_speed
        self.vision = vision
        self.noise_sigma = noise_sigma

        h, w = len(LAYOUT), len(LAYOUT[0])
        self.walls = np.zeros((h, w), dtype=bool)
        for r, row in enumerate(LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == "P":
                    self.player = (r, c)
                elif ch == "S":
                    self.enemy = (r, c)
                elif ch == "E":
                    self.exit = (r, c)
        self.free_cells = [(r, c) for r in range(h) for c in range(w) if not self.walls[r, c]]
        if enemy_start is not None:  # used by experiments to randomize the scenario
            self.enemy = enemy_start

        self.tracker = BeliefTracker(self.walls) if enemy == "hmm" else None
        self.tick = 0
        self.status = "playing"  # playing | caught | escaped
        self.last_noise = None  # (reported cell, tick) for drawing
        self.last_seen = None
        self.patrol_goal = None
        self._p_acc = 0.0
        self._e_acc = 0.0
        self._vis_cache = {}

    # -- player -------------------------------------------------------------
    def step(self, direction=(0, 0), mode="walk"):
        """Advance one tick. direction = (drow, dcol), mode in SPEEDS."""
        if self.status != "playing":
            return self.status
        self.tick += 1
        moved = self._move_player(direction, mode)
        if self.player == self.exit:
            self.status = "escaped"
            return self.status
        self._enemy_turn(self._make_noise(mode, moved))
        if manhattan(self.player, self.enemy) <= 1:
            self.status = "caught"
        return self.status

    def _move_player(self, direction, mode):
        if direction == (0, 0):
            return False
        self._p_acc += SPEEDS[mode]
        if self._p_acc < 1.0:
            return False
        self._p_acc -= 1.0
        nxt = (self.player[0] + direction[0], self.player[1] + direction[1])
        if self.walls[nxt]:
            return False
        self.player = nxt
        return True

    def _make_noise(self, mode, moved):
        """Return the location the stalker *hears* (noisy), or None."""
        if not moved:
            return None
        prob, radius = NOISE[mode]
        if self.rng.random() >= prob:
            return None
        if np.hypot(self.player[0] - self.enemy[0], self.player[1] - self.enemy[1]) > radius:
            return None
        h, w = self.walls.shape
        for _ in range(10):
            r = round(self.player[0] + self.rng.gauss(0, self.noise_sigma))
            c = round(self.player[1] + self.rng.gauss(0, self.noise_sigma))
            if 0 <= r < h and 0 <= c < w and not self.walls[r, c]:
                self.last_noise = ((r, c), self.tick)
                return (r, c)
        return self.player

    # -- enemy --------------------------------------------------------------
    def _visible(self, pos):
        if pos not in self._vis_cache:
            self._vis_cache[pos] = visible_mask(self.walls, pos, self.vision)
        return self._vis_cache[pos]

    def _enemy_turn(self, noise_obs):
        w = self.walls.shape[1]
        vis = self._visible(self.enemy)
        seen = bool(vis[self.player[0] * w + self.player[1]])
        if seen:
            self.last_seen = self.player

        if self.enemy_type == "hmm":
            t = self.tracker
            t.predict()
            if noise_obs is not None:
                t.observe_noise(noise_obs, self.noise_sigma)
            if seen:
                t.observe_seen(self.player)
            else:
                t.observe_not_seen(vis)
            target = t.most_likely_cell(near=self.enemy)
        elif self.enemy_type == "omniscient":
            target = self.player
        else:  # patrol
            target = self._patrol_target(seen)
        self._move_enemy(target)

    def _patrol_target(self, seen):
        if seen:
            return self.player
        if self.last_seen is not None:
            if self.enemy == self.last_seen:
                self.last_seen = None
            else:
                return self.last_seen
        if self.patrol_goal is None or self.enemy == self.patrol_goal:
            self.patrol_goal = self.rng.choice(self.free_cells)
        return self.patrol_goal

    def _move_enemy(self, target):
        self._e_acc += self.enemy_speed
        if self._e_acc < 1.0:
            return
        self._e_acc -= 1.0
        path = astar(self.walls, self.enemy, target)
        if path:
            self.enemy = path[0]
