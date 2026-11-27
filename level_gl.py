"""OpenGL renderer for the GLB school: batches by material, per-room flickering lights + a lagging flashlight cone."""
import math
import numpy as np
import moderngl
from glbscene import Scene

LIGHTING = """
uniform vec3 eye; uniform vec3 ldir; uniform float torch; uniform float amb; uniform float gain;
uniform vec3 lpos[24]; uniform vec3 lcol[24]; uniform vec4 lbox[24]; uniform int nl;
vec3 shade(vec3 albedo, vec3 wp, vec3 n){
  vec3 d = eye - wp; float dist = length(d); vec3 L = d / max(dist, 0.001);
  if (dot(n, L) < 0.0) n = -n;
  vec3 light = vec3(amb, amb * 1.05, amb * 1.35);
  for (int i = 0; i < nl; i++) {
    vec4 b = lbox[i];
    if (wp.x > b.x - 15.0 && wp.x < b.y + 15.0 && wp.z > b.z - 15.0 && wp.z < b.w + 15.0) {
      vec3 q = lpos[i] - wp; float r = length(q);
      float a = max(dot(n, q / r), 0.12) / (1.0 + r * r * 0.000045);
      light += lcol[i] * a;
    }
  }
  float cs = dot(-L, ldir);
  float cone = smoothstep(0.90, 0.985, cs);
  float halo = smoothstep(0.55, 0.95, cs) * 0.10;
  float fall = 1.0 / (1.0 + dist * dist * 0.000010) * clamp(1.0 - dist / 3000.0, 0.0, 1.0);
  light += vec3(1.0, 0.94, 0.80) * torch * (cone + halo) * fall * max(dot(n, L), 0.25) * 3.2;
  vec3 o = albedo * light * gain;
  o = o / (1.0 + o * 0.75);
  float fog = exp(-dist * 0.0009);
  return mix(vec3(0.003, 0.004, 0.007), o, fog);
}
"""
VS = """#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
out vec3 wp; out vec3 wn; out vec2 uv;
void main(){ vec4 w = model * vec4(in_pos, 1.0); wp = w.xyz; wn = mat3(model) * in_nrm; uv = in_uv; gl_Position = mvp * w; }"""
FS = """#version 330
uniform sampler2D tex; uniform vec4 base; uniform float use_tex; uniform vec3 emis;
in vec3 wp; in vec3 wn; in vec2 uv; out vec4 col;
""" + LIGHTING + """
void main(){
  vec4 c = base; if (use_tex > 0.5) c *= texture(tex, uv);
  if (c.a < 0.35) discard;
  vec3 o = shade(c.rgb, wp, normalize(wn)) + emis;
  col = vec4(pow(o, vec3(0.92)), 1.0);
}"""
GVS = """#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col;
out vec3 wp; out vec3 wn; out vec3 vc;
void main(){ vec4 w = model * vec4(in_pos, 1.0); wp = w.xyz; wn = mat3(model) * in_nrm; vc = in_col; gl_Position = mvp * w; }"""
GFS = """#version 330
uniform float pulse;
in vec3 wp; in vec3 wn; in vec3 vc; out vec4 col;
""" + LIGHTING + """
void main(){
  vec3 n = normalize(wn);
  bool eyes = vc.r > 0.45 && vc.g < 0.30 && vc.b < 0.30;
  vec3 o = shade(vc * 1.15, wp, n) + vec3(0.05, 0.05, 0.06) * vc;      // a faint pallor so it never quite disappears
  if (eyes) o = vec3(1.3, 0.05, 0.03) * pulse;
  col = vec4(pow(o, vec3(0.92)), 1.0);
}"""
QVS = """#version 330
in vec2 p; out vec2 uv; void main(){ uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }"""
QFS = """#version 330
uniform sampler2D hud; uniform float vig; uniform vec2 res; uniform float time;
in vec2 uv; out vec4 col;
float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + time) * 43758.5453); }
void main(){
  vec4 t = texture(hud, vec2(uv.x, 1.0 - uv.y));
  vec2 q = uv - 0.5; float v = clamp(dot(q, q) * 2.4 * vig, 0.0, 0.9);
  float g = (h(uv * res) - 0.5) * 0.05;                    // film grain
  float a = 1.0 - (1.0 - t.a) * (1.0 - v);
  col = vec4(t.rgb * t.a + vec3(g, g, g), a);
}"""


class Level:
    def __init__(self, ctx, path, door_nodes=()):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.gprog = ctx.program(vertex_shader=GVS, fragment_shader=GFS)
        self.qprog = ctx.program(vertex_shader=QVS, fragment_shader=QFS)
        self.quad = ctx.vertex_array(self.qprog, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes()), "2f", "p")])
        sc = Scene(path)
        door_nodes = set(door_nodes)
        groups, doorgeo, textures = {}, {}, {}
        for ni, name, mi, M in sc.instances():
            for pos, nrm, uv, idx, mat in sc.mesh_prims(mi):
                P = (np.c_[pos, np.ones(len(pos))] @ M.T)[:, :3]
                if nrm is None:
                    nrm = np.zeros_like(pos); nrm[:, 1] = 1
                N = nrm @ np.linalg.inv(M[:3, :3]).T
                N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
                item = (P.astype(np.float32), N.astype(np.float32), uv, idx.astype(np.uint32))
                if ni in door_nodes:
                    doorgeo.setdefault(ni, []).append((mat, item))
                else:
                    groups.setdefault(mat, []).append(item)

        def tex_for(m):
            if m["tex"] is None:
                return None
            if m["tex"] not in textures:
                try:
                    im = sc.image(m["tex"]); h, w = im.shape[:2]
                    if im.shape[2] == 3:
                        im = np.dstack([im, np.full((h, w, 1), 255, np.uint8)])
                    t = ctx.texture((w, h), 4, np.ascontiguousarray(im[::-1]).tobytes())
                    t.build_mipmaps(); t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); t.anisotropy = 8.0
                    textures[m["tex"]] = t
                except Exception:
                    textures[m["tex"]] = None
            return textures[m["tex"]]

        def make(lst, mat):
            off, ps, ns, us, ix = 0, [], [], [], []
            for P, N, U, I in lst:
                ps.append(P); ns.append(N); us.append(U); ix.append(I + off); off += len(P)
            P, N, U = np.concatenate(ps), np.concatenate(ns), np.concatenate(us)
            I = np.concatenate(ix).reshape(-1)
            m = sc.material(mat)
            vb = ctx.buffer(np.hstack([P, N, U]).astype(np.float32).tobytes())
            ib = ctx.buffer(I.tobytes())
            vao = ctx.vertex_array(self.prog, [(vb, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], ib)
            return dict(vao=vao, tex=tex_for(m), color=tuple(m["color"]), emis=tuple(np.array(m["emissive"][:3]) * 0.6),
                        lo=P.min(0), hi=P.max(0), tris=len(I) // 3)
        self.batches = [make(lst, mat) for mat, lst in groups.items()]
        self.doors = {}
        for ni, lst in doorgeo.items():
            by = {}
            for mat, it in lst:
                by.setdefault(mat, []).append(it)
            self.doors[ni] = [make(l, mat) for mat, l in by.items()]
        self.extra = []
        self.tris = sum(b["tris"] for b in self.batches)

    def add_box(self, lo, hi, color, emis):
        lo, hi = np.array(lo, np.float32), np.array(hi, np.float32)
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        idx = np.array([[f[0], f[1], f[2], f[0], f[2], f[3]] for f in faces], np.uint32).reshape(-1)
        nrm = c - (lo + hi) / 2
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        vb = self.ctx.buffer(np.hstack([c, nrm, np.zeros((8, 2), np.float32)]).astype(np.float32).tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vb, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], self.ctx.buffer(idx.tobytes()))
        b = dict(vao=vao, tex=None, color=color, emis=emis, lo=lo, hi=hi)
        self.extra.append(b)
        return b

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _set(pr, k, v):
        if k in pr:
            pr[k].value = v

    def begin(self, prog, mvp, eye, ldir, torch, amb, gain, lights):
        pr = prog
        pr["mvp"].write(np.ascontiguousarray(mvp.T, np.float32).tobytes())
        for k, v in (("eye", tuple(eye)), ("ldir", tuple(ldir)), ("torch", torch), ("amb", amb), ("gain", gain), ("nl", len(lights))):
            self._set(pr, k, v)
        n = len(lights)
        pad = 24 - n
        pr["lpos"].write(np.array([l["pos"] for l in lights] + [(0, 0, 0)] * pad, np.float32).tobytes())
        pr["lcol"].write(np.array([np.array(l["col"]) * l["now"] for l in lights] + [(0, 0, 0)] * pad, np.float32).tobytes())
        pr["lbox"].write(np.array([l["box"] for l in lights] + [(0, 0, 0, 0)] * pad, np.float32).tobytes())

    def draw_scene(self, mvp, eye, ldir, torch, amb, gain, lights, door_models):
        pr = self.prog
        self.begin(pr, mvp, eye, ldir, torch, amb, gain, lights)
        ident = np.eye(4, dtype=np.float32).tobytes()
        pr["model"].write(ident)
        self._draw(self.batches)
        self._draw(self.extra)
        for ni, bs in self.doors.items():
            pr["model"].write(np.ascontiguousarray(door_models.get(ni, np.eye(4)).T, np.float32).tobytes())
            self._draw(bs)

    def _draw(self, batches):
        pr = self.prog
        for b in batches:
            if b.get("hidden"):
                continue
            pr["base"].value = b["color"]
            pr["emis"].value = b["emis"]
            if b["tex"] is not None:
                b["tex"].use(0); pr["use_tex"].value = 1.0
            else:
                pr["use_tex"].value = 0.0
            b["vao"].render()

    def composite_hud(self, tex, vig, res, t):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(1)
        self.qprog["hud"].value = 1
        self.qprog["vig"].value = vig
        self.qprog["res"].value = res
        self.qprog["time"].value = t % 7.0
        self.quad.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)


class GhostModel:
    """The user's pocong.glb, drawn as a real 3D mesh (vertex colours baked from its texture)."""
    def __init__(self, ctx, level, path, height=178.0):
        import bake_pocong as bp
        M = bp.load_glb(path)
        up, face, alb = bp.orient(M)
        P = bp.to_camera(M["V"], up, face)
        N = bp.to_camera(M["N"], up, face)
        ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
        s = height / (ymax - ymin)
        P = (P - np.array([0.5 * (P[:, 0].min() + P[:, 0].max()), ymin, 0.5 * (P[:, 2].min() + P[:, 2].max())])) * s
        col = np.clip(alb, 0, 1)
        vb = ctx.buffer(np.hstack([P, N, col]).astype(np.float32).tobytes())
        ib = ctx.buffer(M["F"].astype(np.uint32).reshape(-1).tobytes())
        self.vao = ctx.vertex_array(level.gprog, [(vb, "3f 3f 3f", "in_pos", "in_nrm", "in_col")], ib)
        self.prog = level.gprog
        self.level = level

    def draw(self, mvp, eye, ldir, torch, amb, gain, lights, pos, yaw, hop, tilt, pulse):
        c, s_ = math.cos(yaw), math.sin(yaw)
        ct, st = math.cos(tilt), math.sin(tilt)
        R = np.array([[c, 0, s_, 0], [0, 1, 0, 0], [-s_, 0, c, 0], [0, 0, 0, 1]], np.float64)
        T = np.array([[1, 0, 0, 0], [0, ct, -st, 0], [0, st, ct, 0], [0, 0, 0, 1]], np.float64)   # lean forward in the hop
        M = np.eye(4); M[:3, 3] = (pos[0], pos[1] + hop, pos[2])
        M = M @ R @ T
        self.level.begin(self.prog, mvp, eye, ldir, torch, amb, gain, lights)
        self.prog["model"].write(np.ascontiguousarray(M.T, np.float32).tobytes())
        self.prog["pulse"].value = pulse
        self.vao.render()


def persp(fov_y_deg, asp, n, f):
    t = 1 / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([[t / asp, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], np.float64)


def look(eye, yaw, pitch=0.0):
    f = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    V = np.eye(4); V[0, :3] = r; V[1, :3] = u; V[2, :3] = -f
    V[:3, 3] = -V[:3, :3] @ np.asarray(eye, np.float64)
    return V, f
