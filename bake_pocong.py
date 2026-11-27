"""Bake a 3D pocong model into 24 viewing angles -> pocong_views.png  (the game's --model option).

The game draws flat sprites, so a 3D model is shown by pre-rendering it from every 15 degrees; the game
picks the view that matches the angle you are looking at the ghost from, so it truly turns as you walk around it.

Run once (about a minute):    py -3.13 bake_pocong.py            (uses pocong.glb, else pocong.obj)
Needs only numpy + pygame.   GLB: base-colour + normal map textures are used.   OBJ: vertex colours.
Frame 0 = the face looks straight at you, then it turns toward your right.
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pocong_views.png")
NV = int(os.environ.get("NV", 24))      # views
H_PX = 760                              # final frame height
SS = 2                                  # supersampling for smooth edges
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ------------------------------------------------------------------ loaders
def load_glb(path):
    d = open(path, "rb").read()
    off, js, bn = 12, None, None
    while off < len(d):
        cl, ct = struct.unpack("<II", d[off:off + 8])
        body = d[off + 8:off + 8 + cl]
        if ct == 0x4E4F534A:
            js = json.loads(body)
        elif ct == 0x004E4942:
            bn = body
        off += 8 + cl

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        n, c = a["count"], NC[a["type"]]
        dt = np.dtype(CT[a["componentType"]])
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", c * dt.itemsize)
        raw = np.frombuffer(bn, np.uint8, count=(n - 1) * stride + c * dt.itemsize, offset=start)
        arr = np.lib.stride_tricks.as_strided(raw, (n, c * dt.itemsize), (stride, 1)).copy().view(dt).reshape(n, c)
        return arr.astype(np.float32) if dt != np.uint32 else arr.astype(np.int64)

    prim = js["meshes"][0]["primitives"][0]
    at = prim["attributes"]
    V, N = acc(at["POSITION"]), acc(at["NORMAL"])
    T = acc(at["TANGENT"]) if "TANGENT" in at else None
    UV = acc(at["TEXCOORD_0"])
    F = acc(prim["indices"]).reshape(-1, 3)

    def image(ti):
        if ti is None:
            return None
        src = js["images"][js["textures"][ti]["source"]]
        bv = js["bufferViews"][src["bufferView"]]
        raw = bn[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        s = pygame.image.load(io.BytesIO(raw))
        return pygame.surfarray.array3d(s).swapaxes(0, 1).astype(np.float32) / 255.0

    mat = js["materials"][prim.get("material", 0)]
    base = image(mat["pbrMetallicRoughness"].get("baseColorTexture", {}).get("index"))
    nmap = image(mat.get("normalTexture", {}).get("index"))
    return {"V": V, "N": N, "T": T, "UV": UV, "F": F.astype(np.int32), "base": base, "nmap": nmap}


def load_obj(path):
    V, C, F = [], [], []
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
            C.append([float(x) for x in p[4:7]] if len(p) >= 7 else [0.57, 0.52, 0.46])
        elif p[0] == "f":
            ids = [int(x.split("/")[0]) - 1 for x in p[1:]]
            for i in range(1, len(ids) - 1):
                F.append([ids[0], ids[i], ids[i + 1]])
    V, F = np.array(V, np.float32), np.array(F, np.int32)
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    return {"V": V, "N": N, "T": None, "UV": None, "F": F, "base": None, "nmap": None, "C": np.array(C, np.float32)}


def sample(img, uv):
    """bilinear, wrapping; uv in glTF convention (v down)."""
    h, w = img.shape[:2]
    x = (uv[..., 0] % 1.0) * w - 0.5
    y = (uv[..., 1] % 1.0) * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0, x1, y0, y1 = x0 % w, (x0 + 1) % w, y0 % h, (y0 + 1) % h
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


# ------------------------------------------------------------------ orient
def orient(M):
    """Find 'up' (the longest axis, the end holding the dark face is the head) and the direction the face points."""
    V = M["V"]
    ext = V.max(0) - V.min(0)
    ax = int(np.argmax(ext))
    # per-vertex albedo to locate the face: dark vertices
    if M["base"] is not None:
        alb = sample(M["base"], M["UV"])
    else:
        alb = M["C"]
    dark = (alb[:, 0] > 0.5) & (alb[:, 1] < 0.3) & (alb[:, 2] < 0.3)          # the glowing red eyes mark the face
    if dark.sum() < 3:
        dark = alb.mean(1) < 0.16
    other = [i for i in range(3) if i != ax]
    sign = 1.0
    if dark.any():
        mid = (V[:, ax].max() + V[:, ax].min()) / 2
        sign = 1.0 if V[dark, ax].mean() > mid else -1.0
    up = np.zeros(3, np.float32)
    up[ax] = sign
    # horizontal face direction: mean offset of the dark vertices that sit in the upper part
    h = (V[:, ax] * sign - (V[:, ax] * sign).min()) / ext[ax]
    sel = dark
    c = V[:, other].mean(0)
    fd = (V[sel][:, other] - c).mean(0) if sel.any() else np.array([0.0, -1.0])
    if np.linalg.norm(fd) < 1e-6:
        fd = np.array([0.0, -1.0])
    fd /= np.linalg.norm(fd)
    face = np.zeros(3, np.float32)
    face[other] = fd
    return up, face, alb


def to_camera(A, up, face):
    """Rotate so 'up' -> +y and the model's face -> +z (toward the viewer); right-handed."""
    z = face - up * float(face @ up)
    z /= np.linalg.norm(z)
    y = up / np.linalg.norm(up)
    x = np.cross(y, z)
    R = np.stack([x, y, z], 0)               # rows: camera axes in model coords
    return A @ R.T


# -------------------------------------------------------------------- bake
def main():
    src = None
    for name in ("pocong.glb", "pocong.obj"):
        if os.path.exists(os.path.join(HERE, name)):
            src = os.path.join(HERE, name)
            break
    if src is None:
        sys.exit("put pocong.glb (or pocong.obj) next to this script")
    pygame.init()
    M = load_glb(src) if src.endswith(".glb") else load_obj(src)
    print("model:", os.path.basename(src), len(M["V"]), "vertices", len(M["F"]), "triangles")
    up, face, alb = orient(M)
    print("up axis", up, "face points", face)
    P0 = to_camera(M["V"], up, face)
    N0 = to_camera(M["N"], up, face)
    T0 = to_camera(M["T"][:, :3], up, face) if M["T"] is not None else None
    F = M["F"]
    ymin, ymax = float(P0[:, 1].min()), float(P0[:, 1].max())
    radius = float(np.abs(P0[:, [0, 2]]).max())
    frames, closeups = [], []
    # job = (turn angle, pixels per model unit, lowest y, highest y, supersampling, half width in model units, list to fill)
    jobs = []
    sc_full = H_PX * SS / (ymax - ymin)
    for k in range(NV):
        jobs.append((2 * math.pi * k / NV, sc_full, ymin, ymax, SS, radius * 1.12, frames))
    top = ymax - 0.27 * (ymax - ymin)                          # the head and shoulders, for the jump scare
    sc_close = 980.0 / (ymax - top)
    for deg in (-24, -12, 0, 12, 24):
        jobs.append((math.radians(deg), sc_close, top, ymax, 1, 0.36 * (ymax - ymin) * 0.27 * 1.0 + 0.0, closeups))
    for k, (s, scale, y_lo, y_hi, ss, half_w, dest) in enumerate(jobs):
        Wp = int(2 * half_w * scale)
        Hp = int((y_hi - y_lo) * scale)
        SSj = ss
        def rot(A):
            r = np.hypot(A[:, 0], A[:, 2])
            ps = np.arctan2(A[:, 0], A[:, 2]) + s
            return np.stack([r * np.sin(ps), A[:, 1], r * np.cos(ps)], 1)
        P, Nn = rot(P0), rot(N0)
        Tn = rot(T0) if T0 is not None else None
        sx = Wp / 2 + P[:, 0] * scale
        sy = Hp - (P[:, 1] - y_lo) * scale
        zb = np.full((Hp, Wp), -1e9, np.float32)
        idx = np.full((Hp, Wp), -1, np.int32)
        b1 = np.zeros((Hp, Wp), np.float32)
        b2 = np.zeros((Hp, Wp), np.float32)
        fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        front = np.nonzero(fn[:, 2] != 0)[0]
        double_sided = M["base"] is not None
        if not double_sided:
            front = np.nonzero(fn[:, 2] > 0)[0]
        for t in front:
            i0, i1, i2 = F[t]
            x0, y0, x1, y1, x2, y2 = sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2]
            xa, xb = max(int(min(x0, x1, x2)), 0), min(int(max(x0, x1, x2)) + 2, Wp)
            ya, yb = max(int(min(y0, y1, y2)), 0), min(int(max(y0, y1, y2)) + 2, Hp)
            if xa >= xb or ya >= yb:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(ya, yb) + 0.5)
            l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
            l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
            if not inside.any():
                continue
            z = l0 * P[i0, 2] + l1 * P[i1, 2] + l2 * P[i2, 2]
            sub = zb[ya:yb, xa:xb]
            m = inside & (z > sub)
            sub[m] = z[m]
            idx[ya:yb, xa:xb][m] = t
            b1[ya:yb, xa:xb][m] = l1[m]
            b2[ya:yb, xa:xb][m] = l2[m]
        cov = idx >= 0
        t = np.where(cov, idx, 0)
        w0, w1, w2 = (1 - b1 - b2)[..., None], b1[..., None], b2[..., None]
        i0, i1, i2 = F[t, 0], F[t, 1], F[t, 2]
        nrm = Nn[i0] * w0 + Nn[i1] * w1 + Nn[i2] * w2
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        if M["base"] is not None:
            uv = M["UV"][i0] * w0 + M["UV"][i1] * w1 + M["UV"][i2] * w2
            col = sample(M["base"], uv)
            if M["nmap"] is not None and Tn is not None:
                tn = sample(M["nmap"], uv) * 2 - 1
                tan = Tn[i0] * w0 + Tn[i1] * w1 + Tn[i2] * w2
                tan = tan - nrm * (tan * nrm).sum(-1, keepdims=True)
                tan /= np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-6)
                sgn = M["T"][i0, 3][..., None]
                bit = np.cross(nrm, tan) * sgn
                nrm = tan * tn[..., 0:1] + bit * tn[..., 1:2] + nrm * tn[..., 2:3]
                nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
            # a surface seen from behind (double sided cloth): flip toward the viewer
            nrm = np.where(nrm[..., 2:3] < 0, -nrm, nrm)
        else:
            col = M["C"][i0] * w0 + M["C"][i1] * w1 + M["C"][i2] * w2
        L = np.array([0.30, 0.38, 0.88], np.float32)
        L /= np.linalg.norm(L)
        diff = np.clip((nrm * L).sum(-1), 0, 1)
        rim = np.clip(1 - nrm[..., 2], 0, 1) ** 2
        yy = (P0[i0, 1] * w0[..., 0] + P0[i1, 1] * w1[..., 0] + P0[i2, 1] * w2[..., 0] - ymin) / (ymax - ymin)
        ao = 0.70 + 0.30 * np.clip(yy * 3.0, 0, 1)
        shade = (0.34 + 0.80 * diff) * ao
        rgb = np.clip(col * shade[..., None] * 1.15 + 0.04 * rim[..., None], 0, 1)
        rgb[~cov] = 0
        alpha = cov.astype(np.float32)
        pm = rgb * alpha[..., None]
        h2, w2_ = Hp // SSj * SSj, Wp // SSj * SSj
        pm = pm[:h2, :w2_].reshape(h2 // SSj, SSj, w2_ // SSj, SSj, 3).mean((1, 3))
        al = alpha[:h2, :w2_].reshape(h2 // SSj, SSj, w2_ // SSj, SSj).mean((1, 3))
        o = np.zeros(al.shape + (4,), np.uint8)
        o[..., :3] = np.clip(pm / np.maximum(al[..., None], 1e-3) * 255, 0, 255)
        o[..., 3] = (al * 255).astype(np.uint8)
        dest.append(o)
        print("view %d/%d" % (k + 1, len(jobs)), flush=True)
    fh = frames[0].shape[0]
    sheet = np.concatenate(frames, 1)
    surf = pygame.Surface((sheet.shape[1], fh), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:] = sheet[..., :3].swapaxes(0, 1)
    pygame.surfarray.pixels_alpha(surf)[:] = sheet[..., 3].swapaxes(0, 1)
    pygame.image.save(surf, OUT)
    print("saved", OUT, "frame", frames[0].shape[1], "x", fh)
    cw = closeups[0].shape[1]
    cs = np.concatenate(closeups, 1)
    surf = pygame.Surface((cs.shape[1], cs.shape[0]), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:] = cs[..., :3].swapaxes(0, 1)
    pygame.surfarray.pixels_alpha(surf)[:] = cs[..., 3].swapaxes(0, 1)
    pygame.image.save(surf, os.path.join(HERE, "pocong_scare3d.png"))
    print("saved pocong_scare3d.png  (5 close-ups of the head, frame %d x %d)" % (cw, cs.shape[0]))


if __name__ == "__main__":
    main()
