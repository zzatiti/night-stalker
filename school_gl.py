"""Turn horror_school.glb into a playable level (grid + objects + doors) for the AI game.

World units: the model is in centimetres.  One logic cell = 50 cm (CPM = 2 cells per metre).
The result is cached in level_cache.pkl so the (slow) conversion only runs once.
"""
import collections, os, pickle, re
from collections import deque
import numpy as np
from world import visible_mask

HERE = os.path.dirname(os.path.abspath(__file__))
GLB = os.path.join(HERE, "horror_school.glb")
CACHE = os.path.join(HERE, "level_cache.pkl")
CELL, CPM = 50.0, 2.0
X0, Z0 = -3320.0, -2280.0
XMAX, ZMAX = -1170.0, 1450.0
W, H = int(np.ceil((XMAX - X0) / CELL)) + 1, int(np.ceil((ZMAX - Z0) / CELL)) + 1
FLOOR, WALL = 0, 1
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
HIDE_TYPE = {"cupboard": "cupboard", "desk": "desk"}
CAN_SEARCH = {"bookshelf", "storage", "desk", "cupboard", "bed"}


def _cat(n):
    m = re.match(r"[A-Za-z]+", n)
    return m.group(0).lower() if m else "garbled"


def _is_door(s, a):
    sm = sorted([s[0], s[2]])
    return 200 <= s[1] <= 260 and 85 <= sm[1] <= 135 and sm[0] <= 14 and a[1] < 25


def convert(glb=GLB, res=10.0, log=print):
    from glbscene import Scene
    sc = Scene(glb)
    nx, nz = int((XMAX - X0) / res) + 2, int((ZMAX - Z0) / res) + 2
    occ = np.zeros((nz, nx), np.int32)
    nodes, doors = [], []
    for ni, name, mi, M in sc.instances():
        allp = []
        prims = list(sc.mesh_prims(mi))
        for pos, nrm, uv, idx, mat in prims:
            allp.append((np.c_[pos, np.ones(len(pos))] @ M.T)[:, :3])
        if not allp:
            continue
        P = np.concatenate(allp)
        a, b = P.min(0), P.max(0)
        s = b - a
        if _is_door(s, a):
            doors.append({"node": ni, "lo": a, "hi": b})
            continue
        nodes.append((ni, _cat(name), a, b))
        if b[1] < 30 or a[1] > 190 or not (b[1] > 205):   # only tall structure forms walls
            continue
        for p, (pos, nrm, uv, idx, mat) in zip(allp, prims):
            T = p[idx]
            nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            vert = np.abs(nn[:, 1]) / (np.linalg.norm(nn, axis=1) + 1e-9) < 0.35
            ok = vert & (T[:, :, 1].max(1) > 30) & (T[:, :, 1].min(1) < 190)
            for t in T[ok]:
                for u, v in ((0, 1), (1, 2), (2, 0)):
                    p0, p1 = t[u][[0, 2]], t[v][[0, 2]]
                    n = int(np.linalg.norm(p1 - p0) / (res * 0.5)) + 2
                    q = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
                    ix = ((q[:, 0] - X0) / res).astype(int)
                    iz = ((q[:, 1] - Z0) / res).astype(int)
                    m = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
                    occ[iz[m], ix[m]] += 1
    log("walls rasterised;", len(doors), "doors;", len(nodes), "nodes")

    # ---- furniture groups: union nearby low nodes
    items = [x for x in nodes if (x[3] - x[2]).max() <= 260 and x[2][1] <= 150]
    A = np.array([x[2] for x in items])
    B = np.array([x[3] for x in items])
    par = list(range(len(items)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(len(items)):
        ov = np.all((A <= B[i] + 6) & (B >= A[i] - 6), axis=1)
        for j in np.nonzero(ov)[0]:
            if j > i:
                par[find(j)] = find(i)
    G = collections.defaultdict(list)
    for i in range(len(items)):
        G[find(i)].append(i)
    groups = []
    for v in G.values():
        groups.append({"lo": A[v].min(0), "hi": B[v].max(0),
                       "kinds": dict(collections.Counter(items[i][1] for i in v))})
    # ---- bring everything down to the logic grid
    wall = np.zeros((H, W), bool)
    k = int(CELL / res)
    for r in range(H):
        for c in range(W):
            blk = occ[r * k:(r + 1) * k, c * k:(c + 1) * k]
            if blk.size and blk.sum() > 0:
                wall[r, c] = True
    wall[0, :] = wall[-1, :] = True
    wall[:, 0] = wall[:, -1] = True
    return {"wall": wall, "doors": doors, "groups": groups}


def _cells_in(lo, hi, shrink=0.0):
    c0 = int((lo[0] + shrink - X0) // CELL); c1 = int((hi[0] - shrink - X0) // CELL)
    r0 = int((lo[2] + shrink - Z0) // CELL); r1 = int((hi[2] - shrink - Z0) // CELL)
    return [(r, c) for r in range(max(r0, 0), min(r1, H - 1) + 1) for c in range(max(c0, 0), min(c1, W - 1) + 1)]


def to_cell(x, z):
    return ((x - X0) / CELL, (z - Z0) / CELL)


def to_world(cx, cy):
    return (X0 + cx * CELL, Z0 + cy * CELL)


def _bfs(solid, start):
    d = np.full(solid.shape, 1e6)
    d[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and not solid[n] and d[n] > d[r, c] + 1:
                d[n] = d[r, c] + 1
                q.append(n)
    return d


def build():
    if os.path.exists(CACHE):
        raw = pickle.load(open(CACHE, "rb"))
    else:
        raw = convert()
        pickle.dump(raw, open(CACHE, "wb"))
    wall = raw["wall"].copy()
    solid = wall.copy()

    # ---- doors: carve a clean opening, remember the cells that the closed door fills
    doors = []
    for i, d in enumerate(raw["doors"]):
        lo, hi = d["lo"], d["hi"]
        along_z = (hi[2] - lo[2]) > (hi[0] - lo[0])
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        if along_z:
            cells = _cells_in((cx - 35, 0, lo[2] + 12), (cx + 35, 0, hi[2] - 12))
            band = _cells_in((cx - 60, 0, lo[2] + 12), (cx + 60, 0, hi[2] - 12))
        else:
            cells = _cells_in((lo[0] + 12, 0, cz - 35), (hi[0] - 12, 0, cz + 35))
            band = _cells_in((lo[0] + 12, 0, cz - 60), (hi[0] - 12, 0, cz + 60))
        for rc in band:
            wall[rc] = False
            solid[rc] = False
        door_cells = [rc for rc in cells if True]
        doors.append({"id": i, "node": d["node"], "lo": lo, "hi": hi, "along_z": along_z,
                      "cells": door_cells, "band": band,
                      "hinge": (cx, lo[2]) if along_z else (lo[0], cz),
                      "centre": to_cell(cx, cz), "angle": 0.0, "target": 0.0, "dir": 1.0})
    tall = wall.copy()
    # ---- furniture
    objects = []
    for g in raw["groups"]:
        lo, hi, kinds = g["lo"], g["hi"], g["kinds"]
        h = hi[1]
        sx, sz = hi[0] - lo[0], hi[2] - lo[2]
        cells = _cells_in(lo, hi, shrink=6.0)
        if not cells or h < 25:
            continue
        kind = None
        if "cabinet" in kinds:
            if h >= 150 and min(sx, sz) <= 200:
                kind = "cupboard"
            elif h >= 150:
                kind = "bookshelf"
            else:
                kind = "storage"
        elif "bed" in kinds:
            kind = "bed"
        elif ("table" in kinds or "desk" in kinds) and h <= 140:
            kind = "desk"
        for rc in cells:
            solid[rc] = True
            if kind in ("cupboard", "bookshelf") or h > 200:
                tall[rc] = True
        if kind is None:
            continue
        seg = 100.0 if kind == "desk" else 95.0
        axis = 0 if sx >= sz else 2
        n = max(1, int(round((hi[axis] - lo[axis]) / seg)))
        for k_ in range(n):
            plo, phi = lo.copy(), hi.copy()
            plo[axis] = lo[axis] + (hi[axis] - lo[axis]) * k_ / n
            phi[axis] = lo[axis] + (hi[axis] - lo[axis]) * (k_ + 1) / n
            pc = _cells_in(plo, phi, shrink=4.0) or cells[:1]
            cxm, czm = (plo[0] + phi[0]) / 2, (plo[2] + phi[2]) / 2
            objects.append({"kind": kind, "cells": pc, "centre": to_cell(cxm, czm), "lo": plo, "hi": phi, "h": h})
    # door cells stay passable for the AI; the game closes/opens them dynamically
    for dr in doors:
        for rc in dr["band"]:
            solid[rc] = False
    # ---- starting positions
    sr, sc_ = int((1250 - Z0) // CELL), int((-2250 - X0) // CELL)
    start = (sr, sc_)
    assert not solid[start], "start blocked"
    reach = _bfs(solid, start)
    free = np.argwhere(reach < 1e5)
    far = free[np.argsort(-reach[tuple(free.T)])[:60]]
    gcell = tuple(far[len(far) // 2])
    # exit: north end of the corridor
    er, ec = int((-2150 - Z0) // CELL), int((-2250 - X0) // CELL)
    exit_cell = (er, ec)
    if solid[exit_cell] or reach[exit_cell] > 1e5:
        cand = [tuple(x) for x in free if abs(x[0] - er) < 6]
        exit_cell = min(cand, key=lambda x: abs(x[0] - er) + abs(x[1] - ec))
    # objects the player can really stand next to: free reachable cells within ~1.2 m of the object's edge
    keep = []
    for o in objects:
        cand = {}
        cs = o["cells"]
        r0 = min(r for r, _ in cs) - 4; r1 = max(r for r, _ in cs) + 4
        c0 = min(c for _, c in cs) - 4; c1 = max(c for _, c in cs) + 4
        for r in range(max(r0, 0), min(r1, H - 1) + 1):
            for c in range(max(c0, 0), min(c1, W - 1) + 1):
                if solid[r, c] or reach[r, c] > 1e5 or (r, c) in cs:
                    continue
                d = min(np.hypot(rr - r, cc - c) for rr, cc in cs) - 0.0
                if d <= 2.6:
                    cand[(r, c)] = d
        if not cand or min(cand.values()) > 2.0:
            continue
        o["fronts"] = sorted(cand, key=cand.get)[:8]
        keep.append(o)
    out = []
    for o in keep:
        out.append({"id": len(out), "kind": o["kind"], "cell": (int(o["centre"][1]), int(o["centre"][0])),
                    "centre": o["centre"], "fronts": o["fronts"], "hide_type": HIDE_TYPE.get(o["kind"]),
                    "can_search": o["kind"] in CAN_SEARCH, "has_key": False, "searched": False,
                    "lo": o["lo"], "hi": o["hi"], "h": o["h"], "cells": o["cells"]})
    return {"grid": tall.astype(np.int8), "solid": solid, "tall": tall, "objects": out, "lights": [],
            "player": (sc_ + 0.5, sr + 0.5), "ghost": (gcell[1] + 0.5, gcell[0] + 0.5),
            "door": (exit_cell[1] + 0.5, exit_cell[0] + 0.5), "doors": doors}


if __name__ == "__main__":
    import time
    t = time.time()
    d = build()
    print("built in %.1fs" % (time.time() - t), "grid", d["solid"].shape, "free", int((~d["solid"]).sum()))
    print("objects", collections.Counter(o["kind"] for o in d["objects"]), "doors", len(d["doors"]))
