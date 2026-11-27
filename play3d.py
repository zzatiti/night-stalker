"""THE SCHOOL - first-person horror with an AI ghost that learns how you hide.

Run:   py -3.13 play3d.py                (Windows)      python3 play3d.py   (Mac / Linux)
       py -3.13 play3d.py --quality high      (low | med | high - the game also lowers it by itself if it lags)
       py -3.13 play3d.py --photo               (flat reference-sheet ghost instead of the 3D model)

Controls
    W A S D / arrows   move            Mouse / Left-Right   look  (M = toggle mouse)
    Shift              sprint (loud)   Ctrl or C            sneak (quiet)
    F                  flashlight on / off  (a beam also shows the ghost where you are!)
    E                  interact: search a shelf / hide in a cupboard or under a desk / come out
    Q                  search the drawer of a desk or the top of a table
    B                  AI view (the ghost's belief map + what it remembers about you)
    P pause   [ ] brightness   F3 fps   R restart / next round   N new game (ghost forgets you)
    1 2 3 4            ghost brain: HMM+memory | patrol | omniscient | HMM without memory     Esc quit

Goal: find the 3 keys hidden in the school (bookshelves, storage shelves, desk drawers, tables, cupboards),
then reach the exit door at the east end of the main hall. Hide when it hunts you - but it is learning
WHERE you like to hide.
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import pygame

import school
import textures
from audio import Audio
from game3d import GRACE, SEARCH_TIME, Game, warmup
from school import BOARD, BOOKSHELF, CUPBOARD, EXIT, FLOOR, LOCKER, STORAGE, WALL, WINDOW

QUALITY = {"low": (320, 180), "med": (480, 270), "high": (640, 360)}
ORDER = ["low", "med", "high"]
WIN_W, WIN_H = 1280, 720
PLANE_T = (16 / 9) / 2.0 * 1.0            # half-length of the camera plane for a 16:9 picture (square pixels)
TURN_SPEED, MOUSE_SENS = 2.2, 0.0021
LS = 6                                     # light-map samples per cell
LEVELS = 4                                 # mip levels for textures
MODEL_VIEWS = 24                           # angles baked by bake_pocong.py
AMBIENT = np.array([0.020, 0.023, 0.032], np.float32)
FLASH_COL = np.array([1.00, 0.93, 0.78], np.float32)
BRAINS = {pygame.K_1: "hmm", pygame.K_2: "patrol", pygame.K_3: "omniscient", pygame.K_4: "hmm_nomem"}
TEX_NAMES = ["wall0", "wall1", "wall2", "locker", "bookshelf", "storage", "cupboard",
             "board0", "board1", "window", "exit_locked", "exit_open"]
TID = {n: i for i, n in enumerate(TEX_NAMES)}


# ------------------------------------------------------------- small helpers
def mip_stack(arr, levels=LEVELS):
    """arr: (n,256,256,3) uint8 -> (levels, n, 256, 256, 3) uint8; level k is pre-filtered (2^k times blurrier)."""
    out = [arr]
    cur = arr.astype(np.float32)
    for k in range(1, levels):
        n, h, w, c = cur.shape
        cur = cur.reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4))
        up = np.repeat(np.repeat(cur, 2 ** k, 1), 2 ** k, 2)
        out.append(up.astype(np.uint8))
    return np.stack(out)


def chain_from_rgba(rgba):
    """Straight uint8 RGBA -> mip chain of (premultiplied float rgb, alpha) like textures.load_reference_pocong()."""
    a = rgba[..., 3].astype(np.float32) / 255.0
    pm = rgba[..., :3].astype(np.float32) * a[..., None]
    chain = [(pm, a)]
    while min(pm.shape[:2]) >= 16:
        h, w = (pm.shape[0] // 2) * 2, (pm.shape[1] // 2) * 2
        pm = pm[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
        a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
        chain.append((pm, a))
    return chain


def box_blur(m, k):
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        p = np.pad(m, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float32)
        z = np.zeros_like(np.take(c, [0], axis=axis))
        c = np.concatenate([z, c], axis=axis)
        n = m.shape[axis]
        hi = np.take(c, np.arange(2 * k + 1, 2 * k + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        m = (hi - lo) / (2 * k + 1)
    return m


class LightField:
    """All the lamps baked into small soft patches; per frame they are just added together (with flicker)."""

    def __init__(self, lights, H, W):
        self.H, self.W = H, W
        self.shape = (H * LS, W * LS)
        self.patches = []
        for L in lights:
            r = L["radius"]
            if r <= 0:
                self.patches.append(None)
                continue
            x0, x1 = max(0, int((L["x"] - r) * LS)), min(W * LS, int((L["x"] + r) * LS) + 1)
            y0, y1 = max(0, int((L["y"] - r) * LS)), min(H * LS, int((L["y"] + r) * LS) + 1)
            xs, ys = (np.arange(x0, x1) + 0.5) / LS, (np.arange(y0, y1) + 0.5) / LS
            d = np.hypot(xs[None, :] - L["x"], ys[:, None] - L["y"])
            core = np.clip(1.0 - d / r, 0, 1) ** 1.7 * (1.0 + 0.9 / (1.0 + (d / 1.2) ** 2))
            vis = L["vis"].astype(np.float32).repeat(LS, 0).repeat(LS, 1)[y0:y1, x0:x1]
            vis = box_blur(vis, LS // 2 + 1)
            self.patches.append((y0, y1, x0, x1, (core * vis).astype(np.float32)))
        self.buf = np.zeros(self.shape + (3,), np.float32)
        rr = np.random.default_rng(5)
        self.rnd = rr.random(997).astype(np.float32)

    def factor(self, L, t):
        k, ph = L["kind"], L["phase"]
        if k == "flicker":
            n = self.rnd[int(t * 11 + ph * 17) % 997]
            if n < 0.17:
                return 0.04 + 0.4 * n
            if n < 0.23:
                return 1.55
            return 0.82 + 0.18 * math.sin(t * 2.3 + ph)
        if k == "dim":
            n = self.rnd[int(t * 3 + ph * 11) % 997]
            return (0.9 + 0.1 * math.sin(t * 0.8 + ph)) * (0.35 if n < 0.03 else 1.0)
        if k == "red":
            return 0.45 + 0.55 * math.sin(t * 1.9 + ph) ** 2
        if k == "moon":
            return 0.9 + 0.1 * math.sin(t * 0.3 + ph)
        return 0.0

    def build(self, lights, t, px, py, gain):
        b = self.buf
        b.fill(0.0)
        for L, p in zip(lights, self.patches):
            if p is None or abs(L["x"] - px) > 16 or abs(L["y"] - py) > 16:
                continue
            f = self.factor(L, t) * L["strength"] * gain
            if f <= 0.001:
                continue
            y0, y1, x0, x1, patch = p
            b[y0:y1, x0:x1] += patch[..., None] * (L["colour"] * f)
        return b


# ---------------------------------------------------------------------- view
class View:
    def __init__(self, quality="med", gain=0.62, use_model=True):
        self.gain = gain
        self.rng = np.random.default_rng(3)
        self.pyrng = random.Random(9)
        T = textures.load_or_build()
        walls = np.stack([T[n] for n in TEX_NAMES])
        self.wall_stack = mip_stack(walls).reshape(-1, 3)
        self.floor_stack = mip_stack(np.stack([T["floor%d" % i] for i in range(3)])).reshape(-1, 3)
        self.ceil_stack = mip_stack(np.stack([T["ceil%d" % i] for i in range(2)])).reshape(-1, 3)
        self.n_wall, self.n_floor, self.n_ceil = len(TEX_NAMES), 3, 2
        self.desk_chain, self.table_chain = chain_from_rgba(T["desk"]), chain_from_rgba(T["table"])
        self.glint_chain = chain_from_rgba(T["glint"])
        self.ghost_chains = textures.load_reference_pocong()
        self.model_chains = self._load_model_views() if use_model else None
        self.ghost_surfs = []
        for pm, a in [c[0] for c in self.ghost_chains]:
            rgb = np.clip(pm / np.maximum(a[..., None], 1e-3), 0, 255).astype(np.uint8)
            s = pygame.Surface((a.shape[1], a.shape[0]), pygame.SRCALPHA)
            pygame.surfarray.pixels3d(s)[:] = rgb.swapaxes(0, 1)
            pygame.surfarray.pixels_alpha(s)[:] = (a * 255).astype(np.uint8).swapaxes(0, 1)
            self.ghost_surfs.append(s)
        try:
            self.scare_surf = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocong_scare.png")).convert()
        except Exception:
            self.scare_surf = self.ghost_surfs[5]
        self.scare_front = None
        try:                                                  # your own jump-scare picture (pocong_front.png), if present
            self.scare_front = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocong_front.png")).convert()
        except Exception:
            pass
        self.scare3d = []
        try:                                                  # 3D head close-ups baked by bake_pocong.py
            sp = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocong_scare3d.png")).convert_alpha()
            n = 5
            fw = sp.get_width() // n
            self.scare3d = [sp.subsurface((i * fw, 0, fw, sp.get_height())).copy() for i in range(n)]
        except Exception:
            pass
        H, W = school.H, school.W
        self.H, self.W = H, W
        # which wall texture goes in which cell (variety in the plain walls)
        ct = np.zeros((H, W), np.int32)
        for r in range(H):
            for c in range(W):
                h = (r * 7 + c * 13 + (r * c) % 5) % 17
                ct[r, c] = TID["wall1"] if h == 0 else TID["wall2"] if h in (3, 8) else TID["wall0"]
        self.cell_tex_base = ct
        self.floor_var = ((np.arange(H)[:, None] * 5 + np.arange(W)[None, :] * 3 + (np.arange(H)[:, None] * np.arange(W)[None, :]) % 4) % 3).astype(np.int32)
        self.ceil_var = ((np.arange(H)[:, None] + np.arange(W)[None, :]) % 2).astype(np.int32)
        self.lf = None
        self.font = pygame.font.SysFont("georgia,timesnewroman,serif", 26)
        self.small = pygame.font.SysFont("consolas,couriernew,monospace", 17)
        self.tiny = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.big = pygame.font.SysFont("georgia,timesnewroman,serif", 84, bold=True)
        self.mid = pygame.font.SysFont("georgia,timesnewroman,serif", 36)
        self.screen_surf = None
        self.beam_a = 0.0
        self.pose = {"lean": 0.0, "idx": 5, "mirror": False, "hop": -1, "next": 0.0, "away": False, "last": None}
        self.caught_at = None
        self.flash_gain = 1.0
        self.toast = ("", 0.0)
        self.set_quality(quality)

    def _load_model_views(self):
        """Views of the 3D pocong baked by bake_pocong.py (24 angles side by side), as mip chains."""
        try:
            surf = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocong_views.png"))
        except Exception:
            return None
        rgb = pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(np.float32)
        al = pygame.surfarray.array_alpha(surf).swapaxes(0, 1).astype(np.float32) / 255.0
        fw = rgb.shape[1] // MODEL_VIEWS
        chains = []
        for i in range(MODEL_VIEWS):
            r, a = rgb[:, i * fw:(i + 1) * fw], al[:, i * fw:(i + 1) * fw]
            chain = [(r * a[..., None], a)]
            pm, aa = chain[0]
            while min(pm.shape[:2]) >= 48:
                h, w = (pm.shape[0] // 2) * 2, (pm.shape[1] // 2) * 2
                pm = pm[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
                aa = aa[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
                chain.append((pm, aa))
            chains.append(chain)
        return chains

    # ----------------------------------------------------------- resolution
    def set_quality(self, q):
        self.quality = q
        self.RW, self.RH = QUALITY[q]
        RW, RH = self.RW, self.RH
        self.cam = (2.0 * (np.arange(RW) + 0.5) / RW - 1.0).astype(np.float32)
        self.xs = np.arange(RW, dtype=np.float32)
        self.ys = np.arange(RH, dtype=np.float32)
        yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
        r2 = ((xx - RW / 2) / (RW / 2)) ** 2 + ((yy - RH / 2) / (RH / 2)) ** 2
        self.vignette = np.clip(1.08 - 0.62 * r2, 0.08, 1.0).astype(np.float32)
        self.edge = np.clip(r2 - 0.25, 0, 1).astype(np.float32)
        self.grain = [(self.rng.standard_normal((RH, RW)) * 3.2).astype(np.float32) for _ in range(4)]
        self.dust = np.stack([self.rng.random(60), self.rng.random(60), self.rng.random(60) * 6.28], 1)
        rowsf = np.arange(RH // 2, dtype=np.float32) + 0.5
        self.den = rowsf                                     # distance from horizon, in pixels, for the rows below / above it
        wood = (self.rng.random((RH, RW // 4 + 1)) * 0.5 + 0.5).astype(np.float32)
        wood = np.repeat(wood, 4, 1)[:, :RW]
        self.wood = (wood[..., None] * np.array([46, 28, 17], np.float32)[None, None, :])
        self.surf = pygame.Surface((RW, RH))

    # ---------------------------------------------------------------- render
    def render(self, game, dt):
        RW, RH = self.RW, self.RH
        H, W = self.H, self.W
        if self.lf is None:
            self.lf = LightField(game.lights, H, W)
        t = game.t
        px, py, a = game.px, game.py, game.angle
        desk_b = game.hide_blend if game.hide_kind == "desk" else 0.0
        cup_b = game.hide_blend if game.hide_kind == "cupboard" else 0.0
        h = 0.5 - 0.23 * desk_b
        flash_on = game.flashlight and not game.hidden and game.status != "caught"

        # ---- rays (vectorised DDA)
        dirx, diry = math.cos(a), math.sin(a)
        rdx = dirx + (-diry * PLANE_T) * self.cam
        rdy = diry + (dirx * PLANE_T) * self.cam
        rdx = np.where(np.abs(rdx) < 1e-9, 1e-9, rdx)
        rdy = np.where(np.abs(rdy) < 1e-9, 1e-9, rdy)
        mapx = np.full(RW, int(px))
        mapy = np.full(RW, int(py))
        ddx, ddy = np.abs(1.0 / rdx), np.abs(1.0 / rdy)
        stx, sty = np.where(rdx < 0, -1, 1), np.where(rdy < 0, -1, 1)
        sdx = np.where(rdx < 0, (px - mapx) * ddx, (mapx + 1 - px) * ddx)
        sdy = np.where(rdy < 0, (py - mapy) * ddy, (mapy + 1 - py) * ddy)
        hit = np.zeros(RW, bool)
        side = np.zeros(RW, np.int8)
        grid = game.grid
        for _ in range(80):
            go_x = (sdx < sdy) & ~hit
            go_y = (sdx >= sdy) & ~hit
            sdx = np.where(go_x, sdx + ddx, sdx)
            mapx = np.where(go_x, mapx + stx, mapx)
            sdy = np.where(go_y, sdy + ddy, sdy)
            mapy = np.where(go_y, mapy + sty, mapy)
            side = np.where(go_x, 0, np.where(go_y, 1, side))
            hit = hit | (grid[mapy, mapx] != FLOOR)
            if hit.all():
                break
        perp = np.where(side == 0, (mapx - px + (1 - stx) / 2) / rdx, (mapy - py + (1 - sty) / 2) / rdy)
        perp = np.maximum(perp, 0.05).astype(np.float32)
        hx, hy = px + perp * rdx, py + perp * rdy
        wallx = np.where(side == 0, hy, hx)
        wallx = wallx - np.floor(wallx)
        texx = (wallx * 256).astype(np.int32)
        texx = np.where((side == 0) & (rdx > 0), 255 - texx, texx)
        texx = np.where((side == 1) & (rdy < 0), 255 - texx, texx)
        texx = np.clip(texx, 0, 255)
        tile = grid[mapy, mapx]
        unlocked = game.keys_got == game.keys_total
        tid = self.cell_tex_base[mapy, mapx].copy()
        tid[tile == LOCKER] = TID["locker"]
        tid[tile == BOOKSHELF] = TID["bookshelf"]
        tid[tile == STORAGE] = TID["storage"]
        tid[tile == CUPBOARD] = TID["cupboard"]
        tid[tile == WINDOW] = TID["window"]
        tid[tile == BOARD] = TID["board0"] + ((mapx + mapy) % 2)[tile == BOARD]
        tid[tile == EXIT] = TID["exit_open"] if unlocked else TID["exit_locked"]

        lh = (RH / perp).astype(np.float32)
        y = self.ys[:, None] + 0.5
        top = RH / 2 - lh * (1 - h)
        rel = (y - top[None, :]) / lh[None, :]
        inside = (rel >= 0) & (rel < 1)
        texy = np.clip((rel * 256).astype(np.int32), 0, 255)
        wl = np.clip(np.floor(np.log2(np.maximum(256.0 / lh, 1.0))).astype(np.int32), 0, LEVELS - 1)
        widx = ((wl[None, :] * self.n_wall + tid[None, :]) * 256 + texy) * 256 + texx[None, :]
        wallpix = np.take(self.wall_stack, widx, axis=0).astype(np.float32)

        # ---- floor and ceiling (floor casting, height-aware)
        half = RH // 2
        d_f = np.minimum(h * RH / self.den, 60.0).astype(np.float32)
        d_c = np.minimum((1 - h) * RH / self.den, 60.0)[::-1].astype(np.float32)      # top row = farthest
        lvl_f = np.clip(np.floor(np.log2(np.maximum(256.0 * d_f * 2 * PLANE_T / RW, 1.0))).astype(np.int32), 0, LEVELS - 1)
        lvl_c = np.clip(np.floor(np.log2(np.maximum(256.0 * d_c * 2 * PLANE_T / RW, 1.0))).astype(np.int32), 0, LEVELS - 1)
        lm = self.lf.build(game.lights, t, px, py, self.gain)
        LH, LW = lm.shape[:2]

        def plane(dist, lvl, stack, nvar, var):
            fx = px + dist[:, None] * rdx[None, :]
            fy = py + dist[:, None] * rdy[None, :]
            tx = (fx * 256).astype(np.int32) & 255
            ty = (fy * 256).astype(np.int32) & 255
            cx = np.clip(fx.astype(np.int32), 0, W - 1)
            cy = np.clip(fy.astype(np.int32), 0, H - 1)
            idx = ((lvl[:, None] * nvar + var[cy, cx]) * 256 + ty) * 256 + tx
            tex = np.take(stack, idx, axis=0).astype(np.float32)
            li = lm[np.clip((fy * LS).astype(np.int32), 0, LH - 1), np.clip((fx * LS).astype(np.int32), 0, LW - 1)]
            return tex, li

        ftex, fli = plane(d_f, lvl_f, self.floor_stack, self.n_floor, self.floor_var)
        ctex, cli = plane(d_c, lvl_c, self.ceil_stack, self.n_ceil, self.ceil_var)
        tex = np.concatenate([ctex, ftex], 0)
        light = np.concatenate([cli * 0.55, fli], 0)
        light = np.where(inside[..., None], 0.0, light)

        # light on the walls: sampled just in front of the wall surface, one value per column
        sxp = hx - np.where(side == 0, stx * 0.18, 0.0)
        syp = hy - np.where(side == 1, sty * 0.18, 0.0)
        wcol = lm[np.clip((syp * LS).astype(np.int32), 0, LH - 1), np.clip((sxp * LS).astype(np.int32), 0, LW - 1)]
        wcol = wcol * np.where(side == 1, 0.82, 1.0)[:, None]
        wcol = wcol + (tile == WINDOW)[:, None] * np.array([0.10, 0.15, 0.30], np.float32)
        wcol = wcol + ((tile == EXIT) & unlocked)[:, None] * np.array([0.05, 0.40, 0.20], np.float32)
        grad = (1.18 - 0.45 * np.clip(rel, 0, 1)).astype(np.float32)           # lamps hang from the ceiling
        light = light + np.where(inside[..., None], wcol[None, :, :] * grad[..., None], 0.0)
        tex = np.where(inside[..., None], wallpix, tex)

        depth = np.empty((RH, RW), np.float32)
        depth[:half] = d_c[:, None]
        depth[half:] = d_f[:, None]
        depth = np.where(inside, perp[None, :], depth)

        # ---- flashlight: a soft cone that lags a little behind your aim, with falloff over distance
        beam = None
        fl_gain = 0.0
        if flash_on:
            diff = (a - self.beam_a + math.pi) % (2 * math.pi) - math.pi
            self.beam_a = (self.beam_a + diff * min(1.0, dt * 10.0)) % (2 * math.pi)
            lag = (a - self.beam_a + math.pi) % (2 * math.pi) - math.pi
            cx = RW / 2 - max(-0.2, min(0.2, lag)) * (RW / 2) / PLANE_T
            bob = 0.0
            if game.moving:
                bob = math.sin(t * (9 if game.mode == "sprint" else 6)) * (0.012 if game.mode != "sneak" else 0.004) * RH
            cy = RH * 0.54 + bob
            sg = 0.15 * RW
            bx = np.exp(-((self.xs - cx) / sg) ** 2 * 0.5)
            by = np.exp(-((self.ys - cy) / (sg * 1.05)) ** 2 * 0.5)
            bx2 = np.exp(-((self.xs - cx) / (sg * 2.4)) ** 2 * 0.5)
            by2 = np.exp(-((self.ys - cy) / (sg * 2.4)) ** 2 * 0.5)
            beam = (0.8 * np.outer(by, bx) + 0.2 * np.outer(by2, bx2)).astype(np.float32)
            # unsteady when the ghost is close, and now and then it stutters
            if game.threat > 0.4 and self.pyrng.random() < game.threat * 0.2:
                self.flash_gain = self.pyrng.uniform(0.15, 0.7)
            else:
                self.flash_gain += (1.0 - self.flash_gain) * min(1.0, dt * 8)
            fl_gain = 2.0 * self.flash_gain
            fall = 1.0 / (1.0 + (depth / 3.6) ** 2) * np.clip(1.0 - depth / 17.0, 0, 1)
            F = beam * fall * fl_gain
            light = light + F[..., None] * FLASH_COL
        else:
            self.beam_a = a
        light = light + AMBIENT
        frame = tex * light
        if flash_on:                                     # light scattered by dust and fog in the beam
            haze = beam * (1.0 - np.exp(-depth / 7.0)) * (0.045 * 255 * self.flash_gain)
            frame += haze[..., None] * np.array([0.95, 0.93, 0.85], np.float32)
        frame += (1.0 - np.exp(-depth / 22.0))[..., None] * np.array([3.0, 4.0, 6.0], np.float32)

        # ---- sprites: lamps, desks and tables, key glints, the ghost
        ca, sa = dirx, diry
        unit_ref = RH
        sprites = []
        for L in game.lights:
            if L["kind"] == "moon":
                continue
            sx_, sy_ = L["x"] - px, L["y"] - py
            d = ca * sx_ + sa * sy_
            if 0.3 < d < 16 and abs((ca * sy_ - sa * sx_) / d) < PLANE_T * 1.1:
                sprites.append((d, "lamp", L))
        for o in game.objects:
            if o["kind"] in ("desk", "table"):
                sx_, sy_ = o["centre"][0] - px, o["centre"][1] - py
                d = ca * sx_ + sa * sy_
                if 0.25 < d < 15 and abs((ca * sy_ - sa * sx_) / d) < PLANE_T * 1.2 + 0.25 / d:
                    if not (game.hidden and game.hide_obj is o):
                        sprites.append((d, "furn", o))
            if o["can_search"] and not o["searched"] and not game.hidden:
                sx_, sy_ = o["centre"][0] - px, o["centre"][1] - py
                dist = math.hypot(sx_, sy_)
                if dist < 3.4 and (sx_ * ca + sy_ * sa) / max(dist, 1e-6) > 0.3:
                    sprites.append((max(dist - 0.35, 0.3) * 0.999, "glint", o))
        gdx, gdy = game.gx - px, game.gy - py
        gd = ca * gdx + sa * gdy
        if gd > 0.15 and game.status != "caught":
            sprites.append((gd, "ghost", None))
        sprites.sort(key=lambda s: -s[0])
        for d, kind, obj in sprites:
            if kind == "lamp":
                self._lamp(frame, perp, d, obj, game, h)
            elif kind == "furn":
                self._furniture(frame, perp, d, obj, game, h, lm, beam, fl_gain)
            elif kind == "glint":
                self._glint(frame, perp, game, obj, h, beam, fl_gain, t)
            else:
                self._ghost(frame, perp, d, game, h, lm, beam, fl_gain, dt)

        # ---- dust specks floating in the beam
        if flash_on:
            for i in range(len(self.dust)):
                x0, y0, ph = self.dust[i]
                x = int((x0 * 1.3 - 0.15 + 0.02 * math.sin(t * 0.5 + ph)) * RW) % RW
                yy = int(((y0 + t * 0.012 * (1 + (i % 3))) % 1.0) * RH)
                b = beam[yy, x] * (0.4 + 0.6 * math.sin(t * 1.7 + ph) ** 2)
                if b > 0.08:
                    frame[yy, x] += 150 * b * self.flash_gain
                    if i % 4 == 0 and x + 1 < RW:
                        frame[yy, x + 1] += 70 * b

        # ---- hiding: cupboard doors / the underside of a desk
        if cup_b > 0.0:
            frame = self._cupboard_view(frame, game, cup_b, t)
        if desk_b > 0.0:
            frame = self._desk_view(frame, desk_b)

        frame *= self.vignette[..., None] * (1.0 - 0.25 * desk_b - 0.15 * cup_b)
        if game.threat > 0.35 and game.status == "playing":
            pulse = 0.55 + 0.45 * math.sin(t * (5 + 8 * game.threat))
            frame[..., 0] += 110 * (game.threat - 0.35) ** 1.3 * pulse * self.edge
        frame += self.grain[int(t * 24) % 4][..., None]
        return np.clip(frame, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- sprites
    def _project(self, game, wx, wy, depth, h):
        a = game.angle
        sx, sy = wx - game.px, wy - game.py
        trans = (math.cos(a) * sy - math.sin(a) * sx) / PLANE_T
        screen_x = self.RW / 2 * (1 + trans / depth)
        unit = self.RH / depth
        floor_y = self.RH / 2 + unit * h
        return screen_x, unit, floor_y

    def _composite(self, frame, perp, depth, chain, screen_x, bottom, height_px, light3, region_light=None,
                   mirror=False, squash_x=1.0, extra=None, shear=0.0, glow=False):
        RW, RH = self.RW, self.RH
        if height_px < 1.5:
            return
        lvl = 0
        for i, (pm, al) in enumerate(chain):
            if al.shape[0] >= height_px:
                lvl = i
        pm, al = chain[lvl]
        th, tw = al.shape
        sw = height_px * tw / th * squash_x
        top = bottom - height_px
        pad = int(abs(shear) * height_px) + 1
        x0, y0 = int(screen_x - sw / 2) - pad, int(top)
        x1, y1 = int(screen_x + sw / 2) + 1 + pad, int(bottom) + 1
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, RW), max(y0, 0), min(y1, RH)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        rows = np.arange(cy0, cy1) + 0.5
        shift = shear * (bottom - rows)                              # leaning: higher rows slide sideways
        ux = (np.arange(cx0, cx1)[None, :] + 0.5 - (screen_x - sw / 2) - shift[:, None]) / sw * tw
        inb = (ux >= 0) & (ux < tw)
        u = np.clip(ux.astype(np.int32), 0, tw - 1)
        if mirror:
            u = tw - 1 - u
        v = np.clip(((rows - top) / height_px * th).astype(np.int32), 0, th - 1)
        vis = (perp[cx0:cx1] > depth)
        if not vis.any():
            return
        alpha = al[v[:, None], u] * inb * vis[None, :]
        if alpha.max() < 0.01:
            return
        col = pm[v[:, None], u]
        region = frame[cy0:cy1, cx0:cx1]
        lit = light3[None, None, :] if region_light is None else region_light(cy0, cy1, cx0, cx1)
        add = col * lit
        if glow:                                                      # the red eyes burn in the dark
            r_, g_, b_ = col[..., 0], col[..., 1], col[..., 2]
            eye = np.clip((r_ - 1.5 * np.maximum(g_, b_) - 25 * alpha) / (90.0 * np.maximum(alpha, 0.05)), 0, 1) * (alpha > 0.5)
            add = add + eye[..., None] * np.array([255.0, 30.0, 20.0], np.float32) * 0.9 * alpha[..., None]
        if extra is not None:
            add = add + extra[..., None] * alpha[..., None] if extra.shape == alpha.shape else add
        region[:] = region * (1.0 - alpha[..., None]) + add

    def _flash_region(self, beam, depth, gain):
        if beam is None or gain <= 0:
            return None
        fall = 1.0 / (1.0 + (depth / 3.6) ** 2) * max(0.0, 1.0 - depth / 17.0) * gain

        def fn(cy0, cy1, cx0, cx1, base):
            return base[None, None, :] + (beam[cy0:cy1, cx0:cx1] * fall)[..., None] * FLASH_COL
        return fn

    def _sample(self, lm, x, y):
        LH, LW = lm.shape[:2]
        return lm[min(LH - 1, max(0, int(y * LS))), min(LW - 1, max(0, int(x * LS)))] + AMBIENT

    def _furniture(self, frame, perp, d, o, game, h, lm, beam, fl_gain):
        sx, unit, floor_y = self._project(game, o["centre"][0], o["centre"][1], d, h)
        base = self._sample(lm, *o["centre"])
        fn = self._flash_region(beam, d, fl_gain)
        region = (lambda a, b, c, e: fn(a, b, c, e, base)) if fn else None
        chain = self.desk_chain if o["kind"] == "desk" else self.table_chain
        scale = 0.58 if o["kind"] == "desk" else 0.52
        self._composite(frame, perp, d, chain, sx, floor_y, unit * scale, base, region)

    def _glint(self, frame, perp, game, o, h, beam, fl_gain, t):
        cx, cy = o["centre"]
        dx, dy = game.px - cx, game.py - cy
        n = math.hypot(dx, dy) or 1.0
        wx, wy = cx + dx / n * 0.55, cy + dy / n * 0.55
        a = game.angle
        d = math.cos(a) * (wx - game.px) + math.sin(a) * (wy - game.py)
        if d < 0.25:
            return
        sx, unit, floor_y = self._project(game, wx, wy, d, h)
        key = o["has_key"]
        if not key and (o["id"] * 7) % 5 == 0:
            return
        pulse = ((math.sin(t * (2.1 if key else 1.5) + o["id"] * 1.7) + 1) * 0.5) ** (5 if key else 3)
        k = (0.55 if key else 0.30) * pulse + 0.04
        if k < 0.06:
            return
        size = unit * (0.05 + 0.04 * pulse)
        bright = np.array([1.0, 0.93, 0.72], np.float32) * k * (1.0 + (0.7 if beam is not None else 0.0))
        self._composite(frame, perp, d, self.glint_chain, sx, floor_y - unit * 0.28 + size / 2, size, bright)

    def _lamp(self, frame, perp, d, L, game, h):
        RW, RH = self.RW, self.RH
        sx, unit, _ = self._project(game, L["x"], L["y"], d, h)
        ytop = int(RH / 2 - unit * (1 - h))
        half_w = max(1, int(unit * 0.26))
        x0, x1 = int(sx) - half_w, int(sx) + half_w + 1
        if x1 <= 0 or x0 >= RW:
            return
        cxc = min(RW - 1, max(0, int(sx)))
        if perp[cxc] <= d:
            return
        th = max(1, int(unit * 0.045))
        y0, y1 = max(0, ytop), min(RH, ytop + th)
        x0, x1 = max(0, x0), min(RW, x1)
        if y0 >= y1 or x0 >= x1:
            return
        f = self.lf.factor(L, game.t)
        if L["kind"] == "dead":
            frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * 0.4 + np.array([16, 16, 18], np.float32)
            if half_w > 3:                                 # a snapped tube hanging off one end
                yy1 = min(RH, y1 + th * 2)
                frame[y1:yy1, x1 - 3:x1] = np.array([18, 18, 20], np.float32)
        else:
            col = L["colour"] * 255.0 * (0.25 + 0.9 * min(f, 1.5))
            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1] * 0.3, col)

    def _ghost(self, frame, perp, d, game, h, lm, beam, fl_gain, dt):
        P = self.pose
        t = game.t
        hop = int(game.hop_clock)
        # ---- which way is it facing?  (it turns like a body, so walking around it shows its sides and back)
        to_p = math.atan2(game.py - game.gy, game.px - game.gx)
        if game.inspect:
            ox, oy = game.inspect["obj"]["centre"]
            want = math.atan2(oy - game.gy, ox - game.gx)
        elif game.ghost_sees:
            want = to_p
        elif game.path:
            want = math.atan2(game.path[0][0] + 0.5 - game.gy, game.path[0][1] + 0.5 - game.gx)
        elif game.ghost_dist() < 9:                          # standing still: slowly, unnaturally, it turns to look at you
            want = to_p + 0.35 * math.sin(t * 0.7) + (0.5 if (int(t * 0.4) % 5 == 0) else 0.0)
        else:
            want = P.get("yaw", to_p)
        P.setdefault("yaw", want)
        diff = (want - P["yaw"] + math.pi) % (2 * math.pi) - math.pi
        P["yaw"] += max(-5.0 * dt, min(5.0 * dt, diff * 6.0 * dt + math.copysign(min(abs(diff), 0.4 * dt), diff)))
        # relative yaw: 0 = faces you, pi = turned away
        rel = (P["yaw"] - to_p + math.pi) % (2 * math.pi) - math.pi
        fx, fy = math.cos(P["yaw"]), math.sin(P["yaw"])
        rx, ry = -math.sin(game.angle), math.cos(game.angle)          # screen-right vector
        faces_left = (fx * rx + fy * ry) < 0                          # the model's face points to the left on the sheet
        ar = abs(rel)
        screen_s = math.atan2(fx * rx + fy * ry, fx * math.cos(to_p) + fy * math.sin(to_p))   # 0 = face to you, +90deg = to your right
        if ar < 0.39:
            idx, mir = 5, False
        elif ar < 1.18:
            idx, mir = 0, not faces_left
        elif ar < 1.96:
            idx, mir = 3, not faces_left
        else:
            idx, mir = 4, False
        lean = 0.0
        if game.ghost_moving:
            if hop != P["hop"]:
                P["hop"] = hop
                P["lean"] = self.pyrng.uniform(-0.12, 0.12)
                P["blur"] = game.ghost_sees and self.pyrng.random() < 0.3    # an occasional smeared frame while it rushes you
            if P.get("blur") and ar < 1.0:
                idx = 2
        elif t > P["next"]:                                  # standing still: twitchy, never quite settled
            P["next"] = t + self.pyrng.choice([0.12, 0.2, 0.9, 1.6, 2.6])
            P["lean"] = self.pyrng.choice([0.0, 0.0, 0.07, -0.09, 0.16, -0.16])
        if game.inspect:
            lean = 0.30 * (1 if mir else -1) * min(1.0, (t - game.inspect["start"]) * 1.5)
        P["idx"], P["mirror"] = idx, mir
        use3d = self.model_chains is not None
        if use3d:
            mi = int(round(screen_s / (2 * math.pi / MODEL_VIEWS))) % MODEL_VIEWS
        sx, unit, floor_y = self._project(game, game.gx, game.gy, d, h)
        ph = game.ghost_phase
        lift = 0.0
        sq_x, sq_y = 1.0, 1.0
        if game.ghost_moving:
            if ph >= 0.3:
                air = math.sin(math.pi * (ph - 0.3) / 0.7)
                lift = 0.17 * air
                sq_x, sq_y = 1.0 - 0.03 * air, 1.0 + 0.04 * air
            else:
                k = 1.0 - ph / 0.3
                sq_x, sq_y = 1.0 + 0.08 * k, 1.0 - 0.07 * k
        else:
            sq_y = 1.0 + 0.012 * math.sin(t * 1.3)
        chain = self.model_chains[mi] if use3d else self.ghost_chains[P["idx"]]
        scale = 0.86
        height = unit * scale * sq_y
        bottom = floor_y - unit * lift
        base = self._sample(lm, game.gx, game.gy)
        near = max(0.0, 1.0 - d / 2.6)
        base = base + near * np.array([0.045, 0.06, 0.09], np.float32)     # a cold pallor when it is right in front of you
        fn = self._flash_region(beam, d, fl_gain)
        region = (lambda a, b, c, e: fn(a, b, c, e, base)) if fn else None
        lean = lean or P.get("lean", 0.0)
        self._composite(frame, perp, d, chain, sx, bottom, height, base, region, mirror=P["mirror"] and not use3d, squash_x=sq_x, glow=use3d,
                        shear=lean * (-1 if P["mirror"] and not use3d else 1))

    # -------------------------------------------------------- hiding overlays
    def _cupboard_view(self, frame, game, b, t):
        RW, RH = self.RW, self.RH
        x = self.xs
        centre = RW / 2
        open_w = (1.0 - b) * centre
        slat = ((x.astype(np.int32) // 6) % 3 != 0)                         # slats with thin gaps
        slat = slat.astype(np.float32)
        cover = np.where(np.abs(x + 0.5 - centre) <= open_w, 0.0, slat)       # doors swing shut from the sides
        edge = np.clip((np.abs(x + 0.5 - centre) - open_w) / 6.0, 0, 1)
        cover = np.maximum(cover, np.where(np.abs(x + 0.5 - centre) > open_w, 1.0, 0.0) * 0 + cover * edge)
        if game.inspect and game.hide_obj is game.inspect["obj"]:             # it is rattling the door
            shake = int(self.pyrng.uniform(-3, 3))
            cover = np.roll(cover, shake)
        leak = float(np.clip(frame.mean() / 70.0, 0.0, 1.0))
        woodlit = self.wood * (0.35 + 0.9 * leak)
        c = cover[None, :, None]
        return frame * (1.0 - c * 0.97) + woodlit * c

    def _desk_view(self, frame, b):
        RW, RH = self.RW, self.RH
        y = self.ys[:, None]
        band = (0.17 * RH * b)
        under = np.clip((band - y) / 5.0, 0, 1)                               # underside of the desk top
        x = self.xs[None, :]
        legs = np.clip((np.abs(x + 0.5 - RW / 2) - (RW * (0.5 - 0.06 * b))) / 3.0, 0, 1) * (y > band * 0.5)
        m = np.clip(under + legs * 0.9, 0, 1)[..., None]
        leak = float(np.clip(frame.mean() / 55.0, 0.0, 1.0))
        planks = 0.8 + 0.2 * np.sin(x * 0.35)[..., None]
        return frame * (1.0 - m * 0.96) + (self.wood * planks * (0.55 + 0.9 * leak)) * m

    # ------------------------------------------------------------------- HUD
    def text(self, screen, text, pos, color=(225, 225, 225), font=None, center=False):
        font = font or self.small
        img = font.render(text, True, color)
        sh = font.render(text, True, (0, 0, 0))
        if center:
            pos = (pos[0] - img.get_width() // 2, pos[1])
        screen.blit(sh, (pos[0] + 2, pos[1] + 2))
        screen.blit(img, pos)

    def draw(self, screen, game, show_ai, fps, dt, paused=False, show_fps=False):
        img = self.render(game, dt)
        pygame.surfarray.blit_array(self.surf, img.swapaxes(0, 1))
        pygame.transform.smoothscale(self.surf, screen.get_size(), screen)
        W, H = screen.get_size()
        if game.status == "playing":
            self._hud(screen, game, W, H)
        if show_ai:
            self._ai_panel(screen, game, W, H)
        if game.status == "caught":
            self._caught(screen, game, W, H)
        elif game.status == "escaped":
            self._escaped(screen, game, W, H)
        if paused:
            shade = pygame.Surface((W, H), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            screen.blit(shade, (0, 0))
            self.text(screen, "PAUSED", (W // 2, H // 2 - 30), (230, 230, 230), self.mid, True)
        if show_fps:
            self.text(screen, "%d fps  %s %dx%d" % (fps, self.quality, self.RW, self.RH), (W - 190, H - 26), (170, 170, 170), self.tiny)
        if self.toast[1] > time.time():
            self.text(screen, self.toast[0], (W // 2, H - 150), (200, 200, 200), self.small, True)

    def _hud(self, screen, game, W, H):
        t = game.t
        # keys
        for i in range(game.keys_total):
            got = i < game.keys_got
            col = (235, 200, 80) if got else (70, 66, 60)
            x = 26 + i * 40
            pygame.draw.circle(screen, col, (x, 30), 8, 3)
            pygame.draw.line(screen, col, (x + 8, 30), (x + 24, 30), 3)
            pygame.draw.line(screen, col, (x + 20, 30), (x + 20, 37), 3)
        self.text(screen, "ROUND %d" % game.round, (26, 52), (150, 150, 150), self.tiny)
        self.text(screen, "FLASHLIGHT %s" % ("ON" if game.flashlight else "OFF"), (26, 70), (190, 190, 150) if game.flashlight else (110, 110, 110), self.tiny)
        # stamina
        w = 170
        pygame.draw.rect(screen, (14, 14, 14), (26, H - 38, w + 4, 12))
        pygame.draw.rect(screen, (190, 80, 70) if game.exhausted else (150, 175, 150), (28, H - 36, int(w * game.stamina), 8))
        if t < 14:
            self.text(screen, "WASD move   mouse look   F flashlight   E interact / hide   Q search drawers   Shift sprint   Ctrl sneak   B AI view",
                      (W // 2, H - 22), (140, 140, 140), self.tiny, True)
        if not game.hidden:
            cx, cy = W // 2, H // 2
            pr = game.prompt()
            col = (230, 230, 230) if pr else (120, 120, 120)
            pygame.draw.circle(screen, col, (cx, cy), 3 if pr else 2)
            if game.searching:
                frac = min(1.0, game.searching["t"] / SEARCH_TIME)
                pygame.draw.arc(screen, (230, 220, 170), (cx - 22, cy - 22, 44, 44), math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 3)
                pr = "Searching..."
            if pr:
                self.text(screen, pr, (W // 2, H // 2 + 110), (225, 225, 215), self.small, True)
        else:
            self.text(screen, "[E] come out        you are hidden - stay still", (W // 2, H - 130), (190, 190, 190), self.small, True)
        if game.message and t < game.message_until:
            self.text(screen, game.message, (W // 2, H - 90), (232, 232, 224), self.font, True)

    def _ai_panel(self, screen, game, W, H):
        cell = 6
        ox, oy = W - game.W * cell - 16, 16
        panel = pygame.Surface((game.W * cell, game.H * cell), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 190))
        for r in range(game.H):
            for c in range(game.W):
                tl = game.grid[r, c]
                if tl != FLOOR:
                    pygame.draw.rect(panel, (80, 80, 96), (c * cell, r * cell, cell, cell))
                elif game.solid[r, c]:
                    pygame.draw.rect(panel, (70, 52, 36), (c * cell + 1, r * cell + 1, cell - 2, cell - 2))
        if game.tracker is not None:
            g = game.tracker.grid()
            peak = max(float(g.max()), 1e-9)
            if peak > 4.0 / max(1, int((~game.solid).sum())):           # skip while it has no idea at all
                for r, c in zip(*np.nonzero(g > 0.04 * peak)):
                    pygame.draw.rect(panel, (255, 40, 40, int(30 + 200 * (g[r, c] / peak) ** 1.5)), (c * cell, r * cell, cell, cell))
        for o in game.objects:
            if o["hide_type"]:
                col = (255, 190, 90) if o["hide_type"] == "cupboard" else (120, 200, 255)
                pygame.draw.rect(panel, col, (o["cell"][1] * cell + 1, o["cell"][0] * cell + 1, cell - 2, cell - 2), 1)
        pygame.draw.circle(panel, (255, 70, 70), (int(game.gx * cell), int(game.gy * cell)), 4)
        p = (int(game.px * cell), int(game.py * cell))
        pygame.draw.circle(panel, (90, 170, 255), p, 3)
        screen.blit(panel, (ox, oy))
        ox = W - 440
        info = game.debug_info()
        mem = info["memory"]
        y = oy + game.H * cell + 8
        self.text(screen, "GHOST  brain: %s   state: %s" % (info["brain"], info["state"]), (ox, y), (240, 200, 200), self.tiny)
        if "belief_at_player" in info:
            self.text(screen, "belief on your cell %.2f   peak %.2f   entropy %.1f" %
                      (info["belief_at_player"], info["max_belief"], info["entropy"]), (ox, y + 18), (230, 200, 200), self.tiny)
        if game.brain == "hmm":
            self.text(screen, "WHAT IT REMEMBERS ABOUT YOU (%d rounds so far)" % mem["rounds"], (ox, y + 44), (255, 230, 160), self.tiny)
            for i, (k, colr) in enumerate((("cupboard", (255, 190, 90)), ("desk", (120, 200, 255)))):
                p_ = mem["prior"][k]
                yy = y + 64 + i * 22
                pygame.draw.rect(screen, (30, 30, 30), (ox, yy, 160, 14))
                pygame.draw.rect(screen, colr, (ox, yy, int(160 * p_), 14))
                self.text(screen, "%s %d%%  (hid %d, found %d)" % (k, p_ * 100, mem["hides"][k], mem["found"][k]),
                          (ox + 168, yy - 1), (225, 225, 225), self.tiny)
            self.text(screen, "recent hides: %s" % (", ".join(mem["recent"]) or "none yet"), (ox, y + 112), (190, 190, 190), self.tiny)
            if info["plan"]:
                self.text(screen, "about to check: %s" % ", ".join(info["plan"]), (ox, y + 130), (255, 150, 150), self.tiny)

    def _banner(self, screen, W, H, title, color, lines):
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 165))
        screen.blit(shade, (0, 0))
        self.text(screen, title, (W // 2, H // 2 - 130), color, self.big, True)
        for i, ln in enumerate(lines):
            self.text(screen, ln, (W // 2, H // 2 - 10 + 30 * i), (215, 215, 205), self.small, True)

    def _learned_lines(self, game):
        if game.brain != "hmm":
            return []
        m = game.memory.summary()
        pr = m["prior"]
        hid = [h for h in game.round_hides]
        out = []
        if hid:
            out.append("This round you hid in: %s." % ", ".join(h["kind"] for h in hid))
        else:
            out.append("You never hid this round.")
        out.append("The ghost now expects you in a cupboard %d%% / under a desk %d%% of the time." %
                   (pr["cupboard"] * 100, pr["desk"] * 100))
        return out

    def _escaped(self, screen, game, W, H):
        lines = self._learned_lines(game) + ["", "R  next round (the ghost keeps its memory)     N  new game     Esc  quit"]
        self._banner(screen, W, H, "YOU ESCAPED", (130, 220, 150), lines)

    def _caught(self, screen, game, W, H):
        if self.caught_at is None:
            self.caught_at = time.time()
        age = time.time() - self.caught_at
        if self.scare_front is not None:
            img = self.scare_front
            iw, ih = img.get_size()
            a = min(age, 1.4)
            ease = 1 - (1 - min(1.0, a / 0.28)) ** 3                   # lunges in fast, then creeps closer
            z = 1.05 + 1.35 * ease + 0.25 * (a / 1.4)
            fxp, fyp = 0.508, 0.30                                    # where its face is in the picture
            sw, sh = iw / z, ih / z
            jx = math.sin(a * 61) * 0.012 * (1 - a / 1.4) * iw
            jy = math.cos(a * 53) * 0.012 * (1 - a / 1.4) * ih
            sx = min(max(fxp * iw - 0.5 * sw + jx, 0), iw - sw)
            sy = min(max(fyp * ih - 0.38 * sh + jy, 0), ih - sh)
            view = img.subsurface((int(sx), int(sy), int(sw), int(sh)))
            pygame.transform.smoothscale(view, (W, H), screen)
            flick = 0.55 + 0.45 * (math.sin(a * 47) > -0.2)            # the picture stutters like a failing light
            if flick < 1.0 and a < 1.0:
                dark = pygame.Surface((W, H))
                dark.set_alpha(int(150 * (1 - flick)))
                screen.blit(dark, (0, 0))
            if a < 0.25:
                red = pygame.Surface((W, H))
                red.fill((200, 0, 0))
                red.set_alpha(int(150 * (1 - a / 0.25)))
                screen.blit(red, (0, 0))
            if age >= 1.4:
                lines = [game.message] + self._learned_lines(game) + ["", "R  next round (the ghost keeps its memory)     N  new game     Esc  quit"]
                self._banner(screen, W, H, "YOU WERE CAUGHT", (210, 50, 50), lines)
            return
        if age < 1.4:
            ease = min(1.0, age / 0.32)
            ease = 1 - (1 - ease) ** 3
            fl = pygame.Surface((W, H))
            fl.fill((50 + int(60 * math.sin(age * 38) ** 2), 0, 0))
            screen.blit(fl, (0, 0))
            if self.scare3d:                                  # the real 3D head lunges at you, snapping between angles
                surf = self.scare3d[[2, 1, 0, 1, 2, 3, 4, 3][int(age * 16) % 8]]
                hh = int(H * (0.55 + 1.15 * ease) * (1 + 0.025 * math.sin(age * 31)))
                ww = int(hh * surf.get_width() / surf.get_height())
                wob = int(math.sin(age * 57) * 9 * (1 - age / 1.4)), int(math.cos(age * 49) * 7 * (1 - age / 1.4))
                img = pygame.transform.smoothscale(surf, (ww, hh))
                img.fill((205, 190, 190, 255), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(img, (int(W * 0.5 - ww * 0.40) + wob[0], int(H * 0.52 - hh * 0.60) + wob[1]))
                vig = pygame.Surface((W, H), pygame.SRCALPHA)
                pygame.draw.rect(vig, (60, 0, 0, 70), (0, 0, W, H))
                screen.blit(vig, (0, 0))
            else:
                surf = self.scare_surf
                hh = int(H * 1.0 * min(1.0, 0.45 + age * 1.2))
                ww = int(hh * surf.get_width() / surf.get_height())
                screen.blit(pygame.transform.smoothscale(surf, (ww, hh)), (W // 2 - ww // 2, H - hh))
        else:
            lines = [game.message] + self._learned_lines(game) + ["", "R  next round (the ghost keeps its memory)     N  new game     Esc  quit"]
            self._banner(screen, W, H, "YOU WERE CAUGHT", (210, 50, 50), lines)


# --------------------------------------------------------------------- main
def title_screen(screen, view, W, H):
    screen.fill((4, 4, 6))
    view.text(screen, "THE SCHOOL", (W // 2, H // 2 - 170), (190, 175, 150), view.big, True)
    view.text(screen, "Something is still teaching here.", (W // 2, H // 2 - 70), (130, 120, 110), view.mid, True)
    for i, ln in enumerate(["Find the 3 keys hidden in the school, then reach the exit door at the east end of the main hall.",
                            "The ghost hunts by sound and by what it remembers of you. Hide in cupboards or under desks.",
                            "It learns where you like to hide.", "",
                            "WASD move | mouse look | F flashlight | E interact / hide | Q search | Shift sprint | Ctrl sneak",
                            "", "press any key"]):
        view.text(screen, ln, (W // 2, H // 2 + 10 + 28 * i), (170, 165, 155), view.small, True)
    pygame.display.flip()


def main(brain="hmm", quality="med", gain=0.62, model=True, windowed_size=(WIN_W, WIN_H), max_frames=None):
    pygame.init()
    screen = pygame.display.set_mode(windowed_size)
    pygame.display.set_caption("THE SCHOOL")
    W, H = screen.get_size()
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont("georgia,serif", 30)
    screen.blit(font.render("Loading the school...", True, (170, 160, 150)), (W // 2 - 120, H // 2))
    pygame.display.flip()
    view = View(quality, gain, model)
    game = warmup()
    audio = Audio()
    game = Game(brain)
    title_screen(screen, view, W, H)
    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        pygame.time.wait(30)
    show_ai = paused = show_fps = False
    mouse_look = True
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()
    clock = pygame.time.Clock()
    slow, fast, frames, auto = 0, 0, 0, True
    view.beam_a = game.angle
    while max_frames is None or frames < max_frames:
        dt = clock.tick(60) / 1000.0
        turn = 0.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type != pygame.KEYDOWN:
                continue
            k = ev.key
            if k == pygame.K_e:
                game.interact()
            elif k == pygame.K_q:
                game.search_key()
            elif k == pygame.K_f and game.status == "playing":
                game.flashlight = not game.flashlight
                audio.click()
            elif k == pygame.K_b:
                show_ai = not show_ai
            elif k == pygame.K_p:
                paused = not paused
            elif k == pygame.K_F3:
                show_fps = not show_fps
            elif k == pygame.K_LEFTBRACKET or k == pygame.K_RIGHTBRACKET:
                view.gain = max(0.3, min(2.0, view.gain + (0.08 if k == pygame.K_RIGHTBRACKET else -0.08)))
                view.toast = ("brightness %d%%" % (view.gain / 0.62 * 100), time.time() + 1.2)
            elif k == pygame.K_m:
                mouse_look = not mouse_look
                pygame.event.set_grab(mouse_look)
                pygame.mouse.set_visible(not mouse_look)
                pygame.mouse.get_rel()
            elif k == pygame.K_n or k in BRAINS:
                brain = BRAINS.get(k, game.brain)
                game = Game(brain)                            # new game: fresh memory
                view.caught_at = None
                view.beam_a = game.angle
            elif k == pygame.K_r:
                if game.status == "playing":
                    game = Game(game.brain, memory=game.memory, round_no=game.round)    # restart this round
                else:
                    game = game.next_round()
                view.caught_at = None
                view.beam_a = game.angle
        keys = pygame.key.get_pressed()
        if not paused:
            forward = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
            strafe = keys[pygame.K_d] - keys[pygame.K_a]
            turn += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * TURN_SPEED * dt
            if mouse_look:
                turn += pygame.mouse.get_rel()[0] * MOUSE_SENS
            game.update(dt, forward, strafe, turn,
                        sprint=keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
                        crouch=keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_c])
            for e in game.events:
                audio.handle(e, game)
            game.events.clear()
            audio.update(game, dt)
        t0 = time.time()
        view.draw(screen, game, show_ai, clock.get_fps(), dt, paused, show_fps)
        pygame.display.flip()
        cost = time.time() - t0
        if auto and not paused and frames > 30:              # adapt the resolution to the machine
            slow = slow + 1 if cost > 0.05 else max(0, slow - 1)
            fast = fast + 1 if cost < 0.017 else 0
            i = ORDER.index(view.quality)
            if slow > 40 and i > 0:
                view.set_quality(ORDER[i - 1])
                slow = 0
            elif fast > 400 and i < 2 and view.quality != "high" and False:
                view.set_quality(ORDER[i + 1])
        frames += 1
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["hmm", "patrol", "omniscient", "hmm_nomem"], default="hmm")
    ap.add_argument("--quality", choices=ORDER, default="med")
    ap.add_argument("--brightness", type=float, default=0.62)
    ap.add_argument("--photo", action="store_true", help="use the flat reference-sheet sprites instead of the 3D model")
    ap.add_argument("--window", default="%dx%d" % (WIN_W, WIN_H))
    a = ap.parse_args()
    ws = tuple(int(v) for v in a.window.lower().split("x"))
    main(a.brain, a.quality, a.brightness, not a.photo, windowed_size=ws)
