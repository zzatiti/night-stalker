"""Load a whole GLB level: every mesh instance in world space, grouped by material, plus textures. numpy only (+pygame to decode PNG/JPG)."""
import io
import json
import struct

import numpy as np

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
                     [0, 0, 0, 1]], np.float64)


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in n:
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = n["scale"]
        M = S
    if "rotation" in n:
        M = _quat_mat(n["rotation"]) @ M
    if "translation" in n:
        T = np.eye(4)
        T[:3, 3] = n["translation"]
        M = T @ M
    return M


class Scene:
    def __init__(self, path):
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
        self.js, self.bn = js, bn
        self._acc_cache = {}

    def acc(self, i):
        if i in self._acc_cache:
            return self._acc_cache[i]
        js = self.js
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        n, c = a["count"], NC[a["type"]]
        dt = np.dtype(CT[a["componentType"]])
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", c * dt.itemsize)
        raw = np.frombuffer(self.bn, np.uint8, count=(n - 1) * stride + c * dt.itemsize, offset=start)
        arr = np.lib.stride_tricks.as_strided(raw, (n, c * dt.itemsize), (stride, 1)).copy().view(dt).reshape(n, c)
        if a.get("normalized"):
            arr = arr.astype(np.float32) / np.iinfo(dt).max
        self._acc_cache[i] = arr
        return arr

    def instances(self):
        """Yield (node_index, name, mesh_index, world 4x4) for every node carrying a mesh."""
        js = self.js
        roots = js["scenes"][js.get("scene", 0)]["nodes"]
        stack = [(r, np.eye(4)) for r in roots]
        while stack:
            ni, P = stack.pop()
            n = js["nodes"][ni]
            M = P @ node_matrix(n)
            if "mesh" in n:
                yield ni, n.get("name", ""), n["mesh"], M
            for ch in n.get("children", []):
                stack.append((ch, M))

    def mesh_prims(self, mi):
        for p in self.js["meshes"][mi]["primitives"]:
            if p.get("mode", 4) != 4:
                continue
            at = p["attributes"]
            pos = self.acc(at["POSITION"]).astype(np.float64)
            nrm = self.acc(at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None
            uv = self.acc(at["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in at else np.zeros((len(pos), 2), np.float32)
            idx = self.acc(p["indices"]).reshape(-1).astype(np.int64) if "indices" in p else np.arange(len(pos))
            yield pos, nrm, uv, idx.reshape(-1, 3), p.get("material", -1)

    def material(self, mi):
        if mi < 0:
            return {"color": (0.6, 0.6, 0.6, 1.0), "tex": None, "double": False}
        m = self.js["materials"][mi]
        pbr = m.get("pbrMetallicRoughness", {})
        tex = pbr.get("baseColorTexture", {}).get("index")
        return {"color": tuple(pbr.get("baseColorFactor", (1, 1, 1, 1))), "tex": tex, "double": m.get("doubleSided", False),
                "name": m.get("name", ""), "alpha": m.get("alphaMode", "OPAQUE"), "emissive": m.get("emissiveFactor", [0, 0, 0])}

    def image(self, ti):
        import pygame
        js = self.js
        src = js["images"][js["textures"][ti]["source"]]
        bv = js["bufferViews"][src["bufferView"]]
        raw = self.bn[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        s = pygame.image.load(io.BytesIO(raw))
        return pygame.surfarray.array3d(s).swapaxes(0, 1)
