"""Procedural art for the abandoned school: 256px textures and the pocong ghost.

Everything is generated with numpy (no image files needed) and cached in
textures_cache.npz next to this file, so only the first launch takes a few seconds.

The ghost is a pocong (Indonesian shroud ghost): a body wrapped in a dirty burial cloth, tied
above the head and at the feet, with the arms bound inside. Eight poses are drawn (upright,
tilted, leaning, bowing, head dropped), which the game flips between to make it move unsettlingly.
"""
import math
import os

import numpy as np
import pygame

S = 256
VERSION = "school-v5"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures_cache.npz")
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)


# ------------------------------------------------------------------ helpers
def u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def col(c):
    return np.array(c, np.float32)


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def noise(rng, cells, size=S):
    """Tileable value noise in [0, 1]."""
    g = rng.random((cells, cells)).astype(np.float32)
    xs = np.arange(size) * cells / size
    i0 = np.floor(xs).astype(int) % cells
    i1 = (i0 + 1) % cells
    f = xs - np.floor(xs)
    f = (f * f * (3 - 2 * f)).astype(np.float32)
    row = g[:, i0] * (1 - f) + g[:, i1] * f
    return row[i0, :] * (1 - f)[:, None] + row[i1, :] * f[:, None]


def fbm(rng, base=4, octaves=5, size=S, persistence=0.55):
    total, amp, norm = np.zeros((size, size), np.float32), 1.0, 0.0
    for o in range(octaves):
        total += amp * noise(rng, min(base * 2 ** o, size), size)
        norm += amp
        amp *= persistence
    return total / norm


def paint(img, mask, color, alpha=1.0):
    a = mask.astype(np.float32) * alpha
    if a.ndim == 2:
        a = a[..., None]
    img[:] = img * (1 - a) + col(color) * a


def box(x0, y0, x1, y1):
    return (XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)


def grain(img, rng, amt=4.0):
    img += rng.normal(0, amt, img.shape).astype(np.float32)


def seg(img, p0, p1, color, width=1.0, alpha=1.0):
    (x0, y0), (x1, y1) = p0, p1
    pad = int(width) + 2
    xa, xb = int(max(min(x0, x1) - pad, 0)), int(min(max(x0, x1) + pad + 1, S))
    ya, yb = int(max(min(y0, y1) - pad, 0)), int(min(max(y0, y1) + pad + 1, S))
    if xa >= xb or ya >= yb:
        return
    py, px = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy + 1e-6), 0, 1)
    d = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    a = (np.clip(width * 0.5 + 0.5 - d, 0, 1) * alpha)[..., None]
    roi = img[ya:yb, xa:xb]
    roi[:] = roi * (1 - a) + col(color) * a


def crack(img, rng, x, y, ang, length, color=(16, 14, 12), width=1.6, alpha=0.85, branch=0.12):
    pts = [(x, y)]
    for _ in range(int(length / 4)):
        ang += rng.normal(0, 0.35)
        x, y = x + math.cos(ang) * 4, y + math.sin(ang) * 4
        pts.append((x, y))
        if branch and rng.random() < branch:
            crack(img, rng, x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), length * 0.4,
                  color, width * 0.7, alpha, 0)
    for a, b in zip(pts[:-1], pts[1:]):
        seg(img, a, b, color, width, alpha)


def text_alpha(txt, size, angle=0.0):
    surf = pygame.font.Font(None, size).render(txt, True, (255, 255, 255))
    if angle:
        surf = pygame.transform.rotate(surf, angle)
    return pygame.surfarray.array3d(surf)[..., 0].T.astype(np.float32) / 255.0


def put_text(img, a, x, y, color, strength=1.0):
    h, w = a.shape
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + w, S), min(int(y) + h, S)
    if x0 >= x1 or y0 >= y1:
        return
    sub = a[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)][..., None] * strength
    roi = img[y0:y1, x0:x1]
    roi[:] = roi * (1 - sub) + col(color) * sub


# ----------------------------------------------------------------- walls
def plaster(rng, upper=(158, 160, 140), wain=(56, 82, 74), split=150, damage=1.0):
    n1, n2, n3 = fbm(rng, 3, 5), fbm(rng, 5, 4), fbm(rng, 2, 4)
    up = col(upper)[None, None, :] * (0.70 + 0.55 * n1)[..., None]
    low = col(wain)[None, None, :] * (0.72 + 0.55 * n2)[..., None] * (0.92 + 0.08 * np.sin(XX * 0.37 + 4 * n3))[..., None]
    is_low = YY >= split
    img = np.where(is_low[..., None], low, up).astype(np.float32)
    paint(img, (YY >= split - 2) & (YY < split + 3), (22, 32, 28), 0.9)
    paint(img, (YY >= split - 5) & (YY < split - 2), (205, 208, 192), 0.30)
    img *= (1 - 0.30 * smoothstep(60, 0, YY))[..., None]
    img *= (1 - 0.40 * smoothstep(196, 256, YY))[..., None]
    m = fbm(rng, 4, 3)
    thr = 0.66 - 0.05 * damage
    paint(img, (m > thr) & ~is_low, (198, 186, 150), 0.8)
    paint(img, (np.abs(m - thr) < 0.012) & ~is_low, (68, 60, 46), 0.7)
    stain = smoothstep(0.52, 0.70, fbm(rng, 2, 4)) * (YY < split)
    img *= (1 - 0.30 * stain)[..., None]
    paint(img, stain, (104, 84, 46), 0.22)
    paint(img, smoothstep(0.56, 0.68, fbm(rng, 6, 4)) * smoothstep(190, 256, YY), (16, 24, 18), 0.65)
    for _ in range(16):
        x, y = rng.uniform(0, S), rng.uniform(split + 10, S - 10)
        seg(img, (x, y), (x + rng.uniform(6, 26), y + rng.uniform(-8, 8)), (150, 160, 150), 1.2, 0.22)
    for _ in range(int(2 * damage) + 1):
        crack(img, rng, rng.uniform(20, 230), rng.uniform(0, 40), rng.uniform(0.9, 2.2), rng.uniform(60, 120))
    grain(img, rng, 4)
    return img


def poster_wall(rng):
    img = plaster(rng, damage=0.6)
    x0, y0, x1, y1 = 72, 26, 184, 132
    paper = np.empty((S, S, 3), np.float32)
    paper[:] = (206, 198, 168)
    paper *= (0.8 + 0.3 * fbm(rng, 4, 4))[..., None]
    mask = box(x0, y0, x1, y1) & ~((XX - x0) + (y1 - YY) < 30)         # torn bottom-left corner
    paint(img, mask, paper, 0.97)
    paint(img, box(x0 + 8, y0 + 8, x1 - 8, y0 + 34), (130, 46, 40), 0.8)
    for i in range(6):
        paint(img, box(x0 + 10, y0 + 46 + i * 11, x1 - 10 - rng.integers(0, 40), y0 + 51 + i * 11), (84, 80, 74), 0.55)
    put_text(img, text_alpha("DO NOT", 30), x0 + 14, y0 + 11, (225, 215, 190), 0.9)
    paint(img, ((XX - (x0 + 6)) ** 2 + (YY - (y0 + 4)) ** 2) < 12, (180, 40, 36), 1.0)   # pins
    paint(img, ((XX - (x1 - 6)) ** 2 + (YY - (y0 + 4)) ** 2) < 12, (180, 40, 36), 1.0)
    img *= 1 - 0.18 * smoothstep(0.5, 0.75, fbm(rng, 3, 4))[..., None] * mask[..., None]
    return img


def damaged_wall(rng):
    img = plaster(rng, damage=2.2)
    patch = smoothstep(0.58, 0.64, fbm(rng, 3, 4)) * (YY < 150)
    bricks = np.empty((S, S, 3), np.float32)
    row = (YY // 24).astype(int)
    bx = (XX + (row % 2) * 32) % 64
    mortar = (bx < 4) | ((YY % 24) < 4)
    bricks[:] = (118, 58, 44)
    bricks *= (0.7 + 0.5 * fbm(rng, 10, 3))[..., None]
    bricks[mortar] = (140, 134, 124)
    paint(img, patch, bricks, 0.95)
    paint(img, (np.abs(patch - 0.5) < 0.2) & (patch > 0), (50, 44, 38), 0.4)
    return img


# ------------------------------------------------------ tall furniture tiles
def locker(rng):
    img = np.empty((S, S, 3), np.float32)
    img[:] = (66, 82, 96)
    img *= (0.80 + 0.35 * fbm(rng, 4, 4))[..., None]
    for door in (0, 128):
        d = box(door + 4, 5, door + 124, 251)
        tint = rng.uniform(0.9, 1.1)
        paint(img, d, col((74, 92, 108)) * tint, 0.95)
        paint(img, box(door + 4, 5, door + 124, 9), (130, 150, 165), 0.4)             # top edge light
        for k in range(6):                                                             # vents
            paint(img, box(door + 30, 24 + k * 7, door + 94, 27 + k * 7), (10, 12, 14), 0.9)
            paint(img, box(door + 30, 190 + k * 7, door + 94, 193 + k * 7), (10, 12, 14), 0.9)
        paint(img, box(door + 96, 110, door + 104, 150), (14, 16, 18), 0.9)            # handle
        paint(img, box(door + 98, 112, door + 100, 148), (150, 160, 165), 0.5)
        paint(img, box(door + 44, 74, door + 84, 92), (170, 168, 150), 0.85)           # number plate
        put_text(img, text_alpha(str(int(rng.integers(100, 999))), 24), door + 50, 76, (30, 28, 24), 0.9)
        for _ in range(3):                                                             # dents
            cx, cy, r = door + rng.uniform(20, 108), rng.uniform(40, 230), rng.uniform(10, 26)
            dd = np.hypot(XX - cx, YY - cy)
            paint(img, np.clip(1 - dd / r, 0, 1), (20, 24, 28), 0.30)
            paint(img, np.clip(1 - np.hypot(XX - cx + 5, YY - cy + 5) / r, 0, 1), (150, 165, 175), 0.18)
        for _ in range(3):                                                             # rust drips
            x = door + rng.uniform(25, 100)
            y0 = rng.choice([30, 195])
            paint(img, box(x, y0, x + rng.uniform(2, 5), y0 + rng.uniform(25, 70)), (120, 62, 32), 0.45)
    ajar = rng.integers(0, 2)
    if ajar:
        paint(img, box(236, 5, 252, 251), (4, 4, 5), 0.95)
        paint(img, box(233, 5, 236, 251), (150, 160, 168), 0.3)
    paint(img, box(0, 0, 4, S) | box(124, 0, 132, S) | box(252, 0, 256, S) | box(0, 0, S, 5) | box(0, 251, S, 256),
          (8, 10, 12), 0.95)
    for _ in range(14):
        x, y = rng.uniform(5, 250), rng.uniform(10, 245)
        seg(img, (x, y), (x + rng.uniform(-22, 22), y + rng.uniform(-6, 6)), (170, 180, 185), 1.0, 0.18)
    img *= (1 - 0.35 * smoothstep(200, 256, YY))[..., None]
    grain(img, rng, 3)
    return img


def bookshelf(rng):
    img = np.empty((S, S, 3), np.float32)
    img[:] = (58, 38, 26)
    img *= (0.75 + 0.45 * fbm(rng, 5, 4))[..., None]
    palette = [(112, 42, 38), (44, 58, 90), (52, 84, 62), (152, 122, 62), (92, 66, 46), (104, 104, 108),
               (182, 172, 142), (70, 46, 72)]
    boards = [60, 124, 188, 250]
    top = 6
    for b in boards:
        paint(img, box(0, top, S, b - 7), (12, 8, 6), 0.95)                            # shelf interior
        x = 6
        while x < S - 10:
            if rng.random() < 0.12:
                x += rng.integers(8, 20)                                               # gap
                continue
            w = int(rng.integers(8, 22))
            h = int(rng.integers(32, b - top - 12))
            c = col(palette[rng.integers(len(palette))]) * rng.uniform(0.6, 1.0)
            body = box(x, b - 7 - h, x + w, b - 7)
            shade = 0.65 + 0.45 * np.sin(np.clip((XX - x) / w, 0, 1) * math.pi)
            paint(img, body, c[None, None, :] * shade[..., None], 0.97)
            paint(img, box(x, b - 7 - h + 6, x + w, b - 7 - h + 9), (190, 160, 80), 0.5)
            paint(img, box(x, b - 7 - h + 14, x + w, b - 7 - h + 16), (190, 160, 80), 0.35)
            paint(img, box(x + 1, b - 7 - h, x + 3, b - 7), (230, 225, 210), 0.12)
            x += w + int(rng.integers(0, 2))
        paint(img, box(0, b - 7, S, b + 1), (86, 58, 38), 0.98)
        paint(img, box(0, b - 7, S, b - 5), (160, 124, 90), 0.5)
        top = b + 1
    paint(img, box(0, 0, 7, S) | box(S - 7, 0, S, S) | box(0, 0, S, 6), (50, 32, 22), 0.98)
    for _ in range(26):                                                                # cobwebs
        a = rng.uniform(0, math.pi / 2)
        seg(img, (S - 7, 6), (S - 7 - 70 * math.cos(a), 6 + 70 * math.sin(a)), (210, 210, 205), 0.8, 0.22)
    img *= ((1 - 0.3 * fbm(rng, 3, 4)) * 0.5 + 0.5)[..., None]
    grain(img, rng, 3)
    return img


def storage(rng):
    img = np.empty((S, S, 3), np.float32)
    img[:] = (38, 40, 42)
    img *= (0.7 + 0.5 * fbm(rng, 4, 4))[..., None]
    for y in (86, 172, 250):
        paint(img, box(0, y, S, y + 8), (86, 94, 100), 0.98)
        paint(img, box(0, y, S, y + 2), (150, 158, 164), 0.5)
    paint(img, box(0, 0, 9, S) | box(S - 9, 0, S, S), (70, 78, 86), 0.98)
    for top, bot in ((10, 86), (94, 172), (180, 250)):
        x = 14
        while x < S - 40:
            kind = rng.integers(0, 3)
            w = int(rng.integers(34, 70))
            h = int(rng.integers(24, bot - top - 4))
            if kind == 0:                                                              # cardboard box
                paint(img, box(x, bot - h, x + w, bot), col((150, 118, 76)) * rng.uniform(0.6, 0.95), 0.97)
                paint(img, box(x, bot - h, x + w, bot - h + 5), (190, 160, 110), 0.5)
                paint(img, box(x + w // 2 - 4, bot - h, x + w // 2 + 4, bot), (176, 160, 120), 0.45)
            elif kind == 1:                                                            # paint bucket
                w = min(w, 44)
                paint(img, box(x, bot - h, x + w, bot), col((84, 90, 96)) * rng.uniform(0.7, 1.0), 0.97)
                paint(img, box(x, bot - h, x + w, bot - h + 4), (140, 146, 150), 0.6)
                paint(img, box(x + 4, bot - h + 12, x + w - 4, bot - h + 22), (150, 40, 34), 0.5)
            else:                                                                      # jars / bottles
                for j in range(int(rng.integers(2, 5))):
                    jx = x + j * 13
                    jh = int(rng.integers(20, 40))
                    paint(img, box(jx, bot - jh, jx + 9, bot), (70, 96, 88), 0.8)
                    paint(img, box(jx + 2, bot - jh, jx + 6, bot - jh + 4), (170, 170, 150), 0.7)
            x += w + int(rng.integers(4, 22))
    img *= (1 - 0.35 * fbm(rng, 3, 4))[..., None] * 0.6 + 0.4
    grain(img, rng, 3)
    return img


def cupboard(rng):
    img = np.empty((S, S, 3), np.float32)
    grain_f = 1 + 0.10 * np.sin(XX * 1.3 + 8 * noise(rng, 6) + 0.02 * YY)
    base = col((92, 64, 40))[None, None, :] * (0.7 + 0.5 * fbm(rng, 3, 4))[..., None] * grain_f[..., None]
    img[:] = base
    paint(img, box(0, 0, 12, S) | box(S - 12, 0, S, S) | box(0, 0, S, 12) | box(0, S - 12, S, S), (46, 30, 20), 0.97)
    for x0 in (22, 138):
        for y0, y1 in ((22, 112), (122, 236)):
            panel = box(x0, y0, x0 + 96, y1)
            paint(img, panel, (70, 46, 28), 0.55)
            paint(img, box(x0 + 8, y0 + 8, x0 + 88, y1 - 8), base * 1.05, 0.9)
            paint(img, box(x0, y0, x0 + 96, y0 + 3) | box(x0, y0, x0 + 3, y1), (150, 118, 84), 0.4)
            paint(img, box(x0, y1 - 3, x0 + 96, y1) | box(x0 + 93, y0, x0 + 96, y1), (24, 14, 8), 0.5)
    paint(img, box(127, 12, 129, S - 12), (14, 8, 5), 0.95)
    for hx in (112, 134):
        paint(img, box(hx, 108, hx + 5, 154), (150, 120, 60), 0.95)
        paint(img, box(hx + 1, 110, hx + 2, 152), (230, 200, 120), 0.5)
    paint(img, box(122, 160, 134, 172), (20, 14, 8), 0.9)                              # keyhole
    pe = smoothstep(0.60, 0.68, fbm(rng, 5, 4))
    paint(img, pe, (30, 20, 14), 0.5)
    for _ in range(18):
        x, y = rng.uniform(14, 240), rng.uniform(14, 240)
        seg(img, (x, y), (x + rng.uniform(-24, 24), y + rng.uniform(-30, 30)), (200, 170, 130), 1.0, 0.16)
    img *= (1 - 0.35 * smoothstep(210, 256, YY))[..., None]
    grain(img, rng, 3)
    return img


def board(rng, message="DON'T TURN AROUND"):
    img = np.empty((S, S, 3), np.float32)
    slate = col((26, 40, 34))[None, None, :] * (0.8 + 0.4 * fbm(rng, 4, 4))[..., None]
    img[:] = slate
    smear = smoothstep(0.50, 0.75, fbm(rng, 3, 4))
    paint(img, smear, (150, 158, 150), 0.18)
    for _ in range(9):                                                                  # eraser sweeps
        y = rng.uniform(20, 200)
        seg(img, (rng.uniform(10, 80), y), (rng.uniform(150, 240), y + rng.uniform(-10, 10)), (160, 168, 160), 22, 0.07)
    put_text(img, text_alpha(message, 58, 3), 14, 58, (222, 226, 214), 0.85)
    put_text(img, text_alpha("2 + 2 = ?", 30, -2), 26, 152, (200, 206, 196), 0.5)
    put_text(img, text_alpha("H E L P", 26, 1), 150, 160, (200, 206, 196), 0.45)
    for _ in range(40):
        x, y = rng.uniform(10, 245), rng.uniform(40, 190)
        seg(img, (x, y), (x + rng.uniform(-6, 6), y + rng.uniform(-3, 3)), (210, 214, 204), 0.9, 0.22)
    wood = col((106, 76, 46))[None, None, :] * (0.7 + 0.5 * fbm(rng, 6, 3))[..., None]
    paint(img, box(0, 0, S, 10) | box(0, 0, 10, S) | box(S - 10, 0, S, S), wood, 0.98)
    paint(img, box(0, 214, S, S), wood * 0.9, 0.98)
    paint(img, box(0, 214, S, 218), (170, 130, 90), 0.5)
    for k in range(4):
        paint(img, box(30 + k * 38, 208, 42 + k * 38, 214), (220, 220, 210), 0.8)
    grain(img, rng, 3)
    return img


def window(rng):
    img = plaster(rng, damage=1.0)
    x0, y0, x1, y1 = 34, 22, 222, 192
    sky = np.empty((S, S, 3), np.float32)
    glow = np.clip(1 - np.hypot(XX - 150, YY - 70) / 120, 0, 1)
    sky[:] = (20, 30, 56)
    sky += glow[..., None] * np.array([120, 130, 150], np.float32)
    sky += (fbm(rng, 4, 3) * 18)[..., None]
    paint(img, box(x0, y0, x1, y1), sky, 1.0)
    paint(img, np.clip(1 - np.hypot(XX - 168, YY - 62) / 17, 0, 1) > 0.15, (225, 232, 245), 0.9)  # the moon
    for (a, b) in [((x0 + 20, y0 + 120), (x0 + 70, y0 + 60)), ((x0 + 70, y0 + 60), (x0 + 60, y0 + 20)),
                   ((x0 + 70, y0 + 60), (x0 + 110, y0 + 70))]:
        seg(img, a, b, (225, 235, 250), 1.2, 0.7)                                       # cracked glass
    mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
    frame = (210, 206, 190)
    paint(img, box(x0 - 8, y0 - 8, x1 + 8, y0) | box(x0 - 8, y1, x1 + 8, y1 + 8) | box(x0 - 8, y0, x0, y1) |
          box(x1, y0, x1 + 8, y1) | box(mid_x - 4, y0, mid_x + 4, y1) | box(x0, mid_y - 4, x1, mid_y + 4), frame, 0.97)
    paint(img, smoothstep(0.55, 0.7, fbm(rng, 6, 4)) * (box(x0 - 8, y0 - 8, x1 + 8, y1 + 8)), (60, 52, 40), 0.5)
    paint(img, box(x0 - 14, y1 + 8, x1 + 14, y1 + 18), (190, 186, 170), 0.95)           # sill
    paint(img, box(x0 - 14, y1 + 18, x1 + 14, y1 + 22), (50, 46, 40), 0.7)
    seg(img, (x0 + 90, y0 + 4), (x0 + 170, y1 - 4), (84, 60, 40), 14, 0.95)             # nailed plank
    for tx in (x0 + 98, x0 + 160):
        paint(img, np.hypot(XX - tx, YY - (y0 + 4 + (tx - x0 - 90) * 0.85)) < 3, (30, 26, 24), 0.9)
    curtain = box(14, y0 - 12, x0 + 38 + 14 * np.sin(YY * 0.09), 215)
    folds = 0.7 + 0.3 * np.sin(XX * 0.5)
    paint(img, curtain, col((88, 28, 30))[None, None, :] * folds[..., None], 0.9)
    paint(img, curtain & (YY > 170 + 30 * noise(rng, 8)), (88, 28, 30), 0.0)
    grain(img, rng, 3)
    return img


def door(rng, unlocked):
    img = np.empty((S, S, 3), np.float32)
    steel = col((70, 78, 74))[None, None, :] * (0.75 + 0.4 * fbm(rng, 4, 4))[..., None]
    img[:] = steel
    paint(img, box(0, 0, 6, S) | box(S - 6, 0, S, S) | box(0, 0, S, 6) | box(125, 0, 131, S), (22, 26, 24), 0.95)
    for d in (0, 131):
        glass = box(d + 24, 70, d + 100, 130)
        paint(img, glass, (20, 40, 34) if not unlocked else (150, 255, 190), 0.95)
        paint(img, glass, (20, 20, 20), 0.0)
        seg(img, (d + 24, 100), (d + 100, 100), (40, 50, 46), 1.2, 0.6)
        seg(img, (d + 62, 70), (d + 62, 130), (40, 50, 46), 1.2, 0.6)
    paint(img, box(8, 152, 248, 164), (150, 156, 150), 0.95)                            # push bar
    paint(img, box(8, 152, 248, 155), (220, 226, 220), 0.5)
    paint(img, smoothstep(0.55, 0.70, fbm(rng, 5, 4)), (110, 60, 34), 0.35)             # rust
    sign = box(86, 12, 170, 52)
    paint(img, sign, (40, 150, 70) if unlocked else (22, 44, 28), 0.98)
    paint(img, sign, (220, 255, 225) if unlocked else (60, 90, 66), 0.0)
    put_text(img, text_alpha("EXIT", 42), 98, 16, (235, 255, 235) if unlocked else (70, 110, 76), 1.0)
    if unlocked:
        glow = np.clip(1 - np.hypot(XX - 128, YY - 32) / 100, 0, 1)
        img += glow[..., None] * np.array([20, 90, 40], np.float32)
        paint(img, box(124, 0, 132, S), (190, 255, 210), 0.75)                          # light through the gap
    else:
        for k in range(14):                                                             # chain
            cx = 14 + k * 17.2
            cy = 158 + 8 * math.sin(k * 0.55)
            for (ex, ey) in ((6, 0), (0, 4)):
                paint(img, np.hypot((XX - cx) / (7 if ex else 3), (YY - cy) / (3 if ex else 6)) < 1.0, (150, 150, 146), 0.9)
        paint(img, box(112, 170, 144, 206), (150, 30, 28), 0.97)                        # padlock
        paint(img, box(112, 170, 144, 174), (220, 70, 60), 0.6)
        seg(img, (118, 170), (118, 152), (170, 170, 170), 5, 0.95)
        seg(img, (138, 170), (138, 152), (170, 170, 170), 5, 0.95)
        seg(img, (118, 152), (138, 152), (170, 170, 170), 5, 0.95)
    img *= (1 - 0.3 * smoothstep(200, 256, YY))[..., None]
    grain(img, rng, 3)
    return img


# -------------------------------------------------- low furniture (RGBA, v=0 top)
def desk(rng, table=False):
    rgba = np.zeros((S, S, 4), np.float32)
    wood = col((116, 86, 54) if not table else (82, 56, 36))
    top_h = 40 if not table else 34
    apron = 62 if not table else 78
    img = rgba[..., :3]
    img[:] = wood[None, None, :] * (0.7 + 0.5 * fbm(rng, 5, 4))[..., None]
    alpha = np.zeros((S, S), np.float32)
    alpha[YY < apron] = 1.0
    paint(img, box(0, 0, S, 5), (190, 156, 108), 0.7)                                   # worn top lip
    paint(img, box(0, top_h - 3, S, top_h + 4), (36, 24, 14), 0.8)
    paint(img, box(0, top_h + 4, S, apron), (52, 36, 22), 0.5)
    for _ in range(18):
        x, y = rng.uniform(4, 250), rng.uniform(4, top_h - 4)
        seg(img, (x, y), (x + rng.uniform(-30, 30), y + rng.uniform(-3, 3)), (200, 170, 130), 1.0, 0.25)
    put_text(img, text_alpha(rng.choice(["JK", "M+S", "X", "2B"]), 22), rng.uniform(40, 190), 8, (50, 34, 22), 0.55)
    if table:
        paint(img, box(52, top_h + 8, 204, apron - 6), (96, 68, 42), 0.95)              # drawer
        paint(img, box(116, top_h + 24, 140, top_h + 30), (170, 140, 70), 0.95)
    legs = [(10, 34), (S - 34, S - 10)] if table else [(14, 26), (S - 26, S - 14)]
    for a, b in legs:
        alpha[:, a:b] = 1.0
        paint(img, box(a, 0, b, S), col((86, 90, 94)) if not table else col((68, 48, 30)), 0.97)
        paint(img, box(a, apron, a + 3, S), (170, 176, 180), 0.45)
    if not table:
        alpha[(YY > 150) & (YY < 160) & (XX > 26) & (XX < S - 26)] = 1.0
        paint(img, box(26, 150, S - 26, 160), (74, 78, 82), 0.95)
        paint(img, box(26, 150, S - 26, 153), (150, 156, 160), 0.4)
    shadow = np.clip((1 - (YY - apron) / 70.0), 0, 1) * (YY >= apron) * (alpha < 0.5)
    alpha = np.maximum(alpha, shadow * 0.55)
    paint(img, shadow > 0, (6, 5, 4), 0.9 * (alpha < 0.99))
    grain(img, rng, 3)
    rgba[..., 3] = alpha * 255.0
    return rgba


def floor_tile(rng, variant):
    img = np.empty((S, S, 3), np.float32)
    checker = ((XX // 128).astype(int) + (YY // 128).astype(int)) % 2
    light, dark = col((148, 140, 118)), col((84, 90, 84))
    img[:] = np.where(checker[..., None] == 0, light, dark)
    img *= (0.65 + 0.6 * fbm(rng, 4, 5))[..., None]
    paint(img, (XX % 128 < 3) | (YY % 128 < 3), (30, 28, 24), 0.85)
    for _ in range(26):                                                                 # scuffs
        x, y = rng.uniform(0, S), rng.uniform(0, S)
        seg(img, (x, y), (x + rng.uniform(-30, 30), y + rng.uniform(-30, 30)), (190, 184, 168), 1.4, 0.16)
    stain = smoothstep(0.60, 0.74, fbm(rng, 3, 4))
    paint(img, stain, (52, 40, 26), 0.55)
    if variant == 1:                                                                    # cracked
        for _ in range(3):
            crack(img, rng, rng.uniform(0, S), rng.uniform(0, S), rng.uniform(0, 6.28), rng.uniform(90, 170))
    elif variant == 2:                                                                  # debris
        for _ in range(26):
            x, y = rng.uniform(4, 250), rng.uniform(4, 250)
            a = rng.uniform(0, 3.14)
            seg(img, (x, y), (x + 9 * math.cos(a), y + 9 * math.sin(a)), (205, 200, 184), rng.uniform(2, 5), 0.8)
        wet = smoothstep(0.64, 0.72, fbm(rng, 3, 3))
        paint(img, wet, (22, 26, 28), 0.7)
        paint(img, np.clip(wet * (np.sin(XX * 0.07) * 0.5 + 0.5), 0, 1) > 0.5, (110, 120, 128), 0.22)
    grain(img, rng, 4)
    return img


def ceiling(rng, variant):
    img = np.empty((S, S, 3), np.float32)
    img[:] = (112, 110, 104)
    img *= (0.7 + 0.5 * fbm(rng, 4, 4))[..., None]
    pits = rng.random((S, S)) < 0.04
    paint(img, pits, (40, 38, 36), 0.7)
    paint(img, (XX % 128 < 3) | (YY % 128 < 3), (42, 40, 38), 0.9)
    paint(img, smoothstep(0.55, 0.72, fbm(rng, 3, 4)), (96, 76, 40), 0.45)
    if variant == 1:
        hole = box(150, 150, 230, 230)
        paint(img, hole, (10, 10, 12), 0.95)
        for k in range(3):
            seg(img, (160 + k * 24, 150), (150 + k * 30, 205), (120, 120, 120), 3, 0.8)
        seg(img, (150, 190), (230, 186), (90, 92, 94), 9, 0.9)
    grain(img, rng, 3)
    return img


def glint_sprite():
    a = np.zeros((32, 32), np.float32)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    d = np.hypot(xx - 15.5, yy - 15.5)
    a = np.clip(1 - d / 14, 0, 1) ** 2
    a = np.maximum(a, np.clip(1 - np.abs(xx - 15.5) / 1.2, 0, 1) * np.clip(1 - np.abs(yy - 15.5) / 14, 0, 1))
    a = np.maximum(a, np.clip(1 - np.abs(yy - 15.5) / 1.2, 0, 1) * np.clip(1 - np.abs(xx - 15.5) / 14, 0, 1))
    return u8(np.dstack([np.full((32, 32), 255), np.full((32, 32), 235), np.full((32, 32), 160), a * 255]))


# ------------------------------------------------------------- the pocong
POCONG = 256
POSES = [  # (lean, neck) in radians; positive leans to the right of the picture
    (0.00, 0.00),    # 0 upright
    (0.10, 0.24),    # 1 slightly tilted right, head cocked
    (-0.12, -0.27),  # 2 slightly tilted left
    (0.50, 0.30),    # 3 leaning forward
    (1.05, 0.45),    # 4 bowing
    (-1.55, -0.55),  # 5 deep bow, head hanging low
    (-0.75, -0.20),  # 6 bent at the waist
    (0.03, 1.00),    # 7 head dropped
]


def _sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def make_pocong(rng, lean, neck):
    CW = CH = POCONG
    L = 176.0
    N = 190
    s = np.linspace(0, 1, N).astype(np.float32)
    prof_s = [0, .03, .07, .12, .30, .46, .62, .74, .785, .83, .885, .94, .975, 1.0]
    prof_w = [11, 10, 6.5, 10.5, 15.5, 18.5, 19.5, 15, 9, 14.5, 18, 14.5, 6.5, 4]
    hw = np.interp(s, prof_s, prof_w).astype(np.float32)
    ang = lean * _sm((s - 0.36) / 0.30) + neck * _sm((s - 0.70) / 0.25)
    ds = L / (N - 1)
    sx = 128 + np.cumsum(np.sin(ang) * ds) - np.sin(ang[0]) * ds
    sy = (CH - 14) + np.cumsum(-np.cos(ang) * ds) + np.cos(ang[0]) * ds
    Tx, Ty = np.sin(ang), -np.cos(ang)
    Nx, Ny = np.cos(ang), np.sin(ang)

    rgb = np.zeros((CH, CW, 3), np.float32)
    alpha = np.zeros((CH, CW), np.float32)

    def over(new_rgb, new_a):
        nonlocal rgb, alpha
        a = new_a[..., None]
        rgb = rgb * (1 - a) + new_rgb * a
        alpha = alpha + new_a * (1 - alpha)

    cloth = col((190, 183, 164))
    big = fbm(rng, 4, 5)
    fine = fbm(rng, 10, 4)

    # --- the tied-off cloth ends: two at the head knot, two frills at the feet
    ex, ey, ea = sx[-1], sy[-1], ang[-1]
    ends = [(ex, ey, ea + 0.62, 34, 6.0, 0.30), (ex, ey, ea - 0.95, 24, 5.0, 0.40),
            (sx[0], sy[0], 1.35, 20, 5.0, 0.25), (sx[0], sy[0], -1.35, 16, 4.5, 0.25)]
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    for (x0, y0, a0, ln, w0, droop) in ends:
        k = np.linspace(0, 1, 24)
        px = x0 + np.sin(a0) * ln * k
        py = y0 - np.cos(a0) * ln * k + droop * ln * k ** 2 * 1.4
        d2 = (xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2
        i = d2.argmin(-1)
        d = np.sqrt(d2.min(-1))
        w = w0 * (1 - 0.7 * k[i]) + 0.6
        a = np.clip(w - d + 0.5, 0, 1)
        shade = 0.70 + 0.35 * (1 - np.clip(d / w, 0, 1))
        over(cloth[None, None, :] * shade[..., None] * (0.8 + 0.3 * fine[..., None]), a)

    # --- the body: nearest point on the spine gives (position along, sideways offset)
    body_a = np.zeros((CH, CW), np.float32)
    body_rgb = np.zeros((CH, CW, 3), np.float32)
    pts = np.stack([sx, sy], -1).astype(np.float32)
    ld = np.array([-0.50, -0.55, 0.66], np.float32)
    ld /= np.linalg.norm(ld)
    for r0 in range(0, CH, 16):
        px = xx[r0:r0 + 16, :, None] - sx[None, None, :]
        py = yy[r0:r0 + 16, :, None] - sy[None, None, :]
        d2 = px * px + py * py
        i = d2.argmin(-1)
        vx = np.take_along_axis(px, i[..., None], -1)[..., 0]
        vy = np.take_along_axis(py, i[..., None], -1)[..., 0]
        sl = vx * Nx[i] + vy * Ny[i]
        along = vx * Tx[i] + vy * Ty[i]
        u = sl / hw[i]
        sv = s[i]
        inside = (np.abs(u) < 1.0) & ~((i == 0) & (along < -2)) & ~((i == N - 1) & (along > 2))
        edge = np.clip((1 - np.abs(u)) * hw[i] / 1.4, 0, 1)
        a = np.where(inside, edge, 0.0).astype(np.float32)

        uu = np.clip(u, -1, 1)
        nz = np.sqrt(np.clip(1 - uu * uu, 0, 1))
        nxv, nyv = uu * Nx[i], uu * Ny[i]
        diff = np.clip(nxv * ld[0] + nyv * ld[1] + nz * ld[2], 0, 1)
        shade = 0.30 + 0.85 * diff
        shade *= 1 - 0.38 * np.abs(uu) ** 3
        fy = np.clip((sv * 255).astype(int), 0, 255)
        fx = np.clip(((uu * 0.5 + 0.5) * 255).astype(int), 0, 255)
        nb, nf = big[fy, fx], fine[fy, fx]
        folds = np.abs(np.sin((uu * 0.5 + 0.5) * math.pi * 7 + 5 * nb)) ** 3
        shade *= 1 - 0.22 * folds
        for tc, tw in ((0.075, 0.012), (0.785, 0.012), (0.972, 0.010)):                # cinched tie lines
            band = np.exp(-((sv - tc) / tw) ** 2)
            hl = np.exp(-((sv - tc - 0.016) / 0.008) ** 2)
            pleat = np.exp(-((sv - tc) / 0.05) ** 2) * (0.5 + 0.5 * np.cos(uu * math.pi * 8))
            shade *= (1 - 0.62 * band) * (1 - 0.22 * pleat) * (1 + 0.20 * hl)
        for side in (-1, 1):                                                           # dark hollows under the cloth
            eye = ((uu - side * 0.40) / 0.17) ** 2 + ((sv - 0.885) / 0.021) ** 2
            shade *= 1 - 0.55 * np.exp(-eye * 1.6)
        shade *= 1 - 0.28 * np.exp(-(((sv - 0.853) / 0.008) ** 2)) * (np.abs(uu) < 0.3)
        shade *= 1 + 0.10 * np.exp(-((uu / 0.07) ** 2)) * ((sv > 0.84) & (sv < 0.90))
        c = cloth[None, None, :] * np.ones_like(shade)[..., None]
        stain = _sm((nb - 0.56) / 0.14)
        c = c * (1 - 0.5 * stain[..., None]) + col((126, 104, 74)) * 0.5 * stain[..., None]
        dirt = np.clip((0.30 - sv) / 0.30, 0, 1) * (0.5 + 0.5 * nf)
        c = c * (1 - 0.55 * dirt[..., None]) + col((88, 74, 56)) * 0.55 * dirt[..., None]
        body_rgb[r0:r0 + 16] = c * (shade * (0.85 + 0.3 * nf))[..., None]
        body_a[r0:r0 + 16] = a
    over(body_rgb, body_a)
    return rgb, alpha


def build_all():
    pygame.font.init()
    rng = np.random.default_rng(2026)
    T = {}
    T["wall0"] = u8(plaster(rng))
    T["wall1"] = u8(poster_wall(rng))
    T["wall2"] = u8(damaged_wall(rng))
    T["locker"] = u8(locker(rng))
    T["bookshelf"] = u8(bookshelf(rng))
    T["storage"] = u8(storage(rng))
    T["cupboard"] = u8(cupboard(rng))
    T["board0"] = u8(board(rng, "DON'T TURN AROUND"))
    T["board1"] = u8(board(rng, "IT IS STILL HERE"))
    T["window"] = u8(window(rng))
    T["exit_locked"] = u8(door(rng, False))
    T["exit_open"] = u8(door(rng, True))
    T["desk"] = u8(desk(rng, False))
    T["table"] = u8(desk(rng, True))
    for v in range(3):
        T["floor%d" % v] = u8(floor_tile(rng, v))
    for v in range(2):
        T["ceil%d" % v] = u8(ceiling(rng, v))
    T["glint"] = glint_sprite()
    for k, (lean, neck) in enumerate(POSES):
        rgb, a = make_pocong(rng, lean, neck)
        T["pocong_rgb%d" % k] = u8(rgb)
        T["pocong_a%d" % k] = u8(a * 255)
    return T


def load_or_build(verbose=False):
    try:
        data = np.load(CACHE, allow_pickle=False)
        if str(data["version"]) == VERSION:
            return {k: data[k] for k in data.files if k != "version"}
    except Exception:
        pass
    if verbose:
        print("Building textures (first run only)...")
    T = build_all()
    try:
        np.savez_compressed(CACHE, version=np.array(VERSION), **T)
    except Exception:
        pass
    return T


SHEET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocong_sheet.png")
POCONG_FRAMES = 6


def load_reference_pocong():
    """The pocong from the reference photo: a list of mip chains, one per pose.

    Each chain is a list of (premultiplied_rgb float32 HxWx3, alpha float32 HxW), biggest first, each level
    half the size of the one before (area-averaged, so a far-away ghost does not shimmer).
    Falls back to the procedural poses if pocong_sheet.png is missing.
    """
    frames = []
    try:
        surf = pygame.image.load(SHEET)
        arr = pygame.surfarray.array3d(surf).swapaxes(0, 1).astype(np.float32)          # (h, w, 3)
        alp = pygame.surfarray.array_alpha(surf).swapaxes(0, 1).astype(np.float32) / 255.0
        fw = arr.shape[1] // POCONG_FRAMES
        for i in range(POCONG_FRAMES):
            frames.append((arr[:, i * fw:(i + 1) * fw], alp[:, i * fw:(i + 1) * fw]))
    except Exception:
        T = load_or_build()
        k = 0
        while "pocong_rgb%d" % k in T:
            frames.append((T["pocong_rgb%d" % k].astype(np.float32), T["pocong_a%d" % k].astype(np.float32) / 255.0))
            k += 1
    chains = []
    for rgb, a in frames:
        pm, al = rgb * a[..., None], a
        chain = [(pm, al)]
        while min(pm.shape[:2]) >= 48:
            h, w = (pm.shape[0] // 2) * 2, (pm.shape[1] // 2) * 2
            pm = pm[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
            al = al[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
            chain.append((pm, al))
        chains.append(chain)
    return chains
