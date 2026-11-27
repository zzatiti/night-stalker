"""THE SCHOOL - 3D edition.  Plays inside your own horror_school.glb with real doors and hiding places.

    py -3.13 -m pip install moderngl pygame numpy
    py -3.13 play_gl.py            (add --window 1600x900, --brain hmm|patrol|omniscient|hmm_nomem)

Keys: WASD move | mouse look | F flashlight | E open/close doors, hide, search | Q search | Shift sprint |
      Ctrl/C sneak | B AI view | P pause | [ ] brightness | F3 fps | R next round | N new game | Esc quit
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moderngl

import level_gl as LG
import play3d
import school_gl as S
from audio import Audio
from gamegl import Game, SEARCH_TIME, warmup

GLB = os.path.join(HERE, "horror_school.glb")
GHOST_GLB = os.path.join(HERE, "pocong.glb")
_tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
EYE, EYE_SNEAK = 155.0, 108.0
TURN_SPEED, MOUSE_SENS = 2.2, 0.0021
BRAINS = play3d.BRAINS

# (x, y, z, room box [xmin, xmax, zmin, zmax], kind, strength)   - the model is in centimetres
_RZ = [-2280, -1075, -185, 690, 1450]
_LZ = [-2280, -1450, -480, 420, 1450]
LAMPS = []
for i in range(4):
    LAMPS.append(((-1590, 320, (_RZ[i] + _RZ[i + 1]) / 2), (-1995, -1170, _RZ[i], _RZ[i + 1]), ["flicker", "dim", "dead", "flicker"][i]))
    LAMPS.append(((-2900, 320, (_LZ[i] + _LZ[i + 1]) / 2), (-3330, -2510, _LZ[i], _LZ[i + 1]), ["dim", "flicker", "flicker", "dead"][i]))
for i, z in enumerate((-1950, -1300, -650, 0, 650, 1250)):
    LAMPS.append(((-2250, 330, z), (-2510, -1995, z - 330, z + 330), ["dim", "flicker", "dead", "flicker", "dim", "dead"][i]))
LAMPS.append(((-2250, 280, -2200), (-2510, -1995, -2290, -1900), "red"))
LAMPS.append(((-2250, 280, 1400), (-2510, -1995, 1000, 1460), "red"))


class Hud(play3d.View):
    """Reuses the HUD / AI-panel / caught-screen drawing of the 2.5D version on a plain surface."""
    def __init__(self):
        self.font = pygame.font.SysFont("georgia,timesnewroman,serif", 26)
        self.small = pygame.font.SysFont("consolas,couriernew,monospace", 17)
        self.tiny = pygame.font.SysFont("consolas,couriernew,monospace", 14)
        self.big = pygame.font.SysFont("georgia,timesnewroman,serif", 84, bold=True)
        self.mid = pygame.font.SysFont("georgia,timesnewroman,serif", 36)
        self.toast, self.caught_at = ("", 0.0), None
        self.scare_front, self.scare3d, self.scare_surf = None, [], None

        def load(name, alpha=False):
            try:
                img = pygame.image.load(os.path.join(HERE, name))
                return img.convert_alpha() if alpha else img.convert()
            except Exception:
                return None
        self.scare_front = load("pocong_front.png")
        sp = load("pocong_scare3d.png", True)
        if sp is not None:
            fw = sp.get_width() // 5
            self.scare3d = [sp.subsurface((i * fw, 0, fw, sp.get_height())).copy() for i in range(5)]
        self.scare_surf = load("pocong_scare.png") or pygame.Surface((10, 10))


class Renderer:
    def __init__(self, ctx, size, gain=1.0):
        self.ctx, self.W, self.H = ctx, size[0], size[1]
        self.gain = gain
        self.sdoors = None
        t0 = time.time()
        d = S.build()
        self.doors_meta = d["doors"]
        self.level = LG.Level(ctx, GLB, door_nodes=[x["node"] for x in d["doors"]])
        self.ghost = LG.GhostModel(ctx, self.level, GHOST_GLB) if os.path.exists(GHOST_GLB) else None
        self.exit_sign = self.level.add_box((-2290, 245, -2262), (-2210, 275, -2256), (0.8, 0.1, 0.1, 1), (0.6, 0.0, 0.0))
        self.hud_tex = ctx.texture((self.W, self.H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self.hud = Hud()
        self.lights = [{"pos": p, "box": (b[0], b[1], b[2], b[3]), "kind": k, "col": (1.0, 0.1, 0.05) if k == "red" else (1.0, 0.78, 0.55),
                        "ph": 1.7 * i, "now": 0.0} for i, (p, b, k) in enumerate(LAMPS)]
        self.lag_yaw = self.lag_pitch = None
        self.prev_ghost = None
        self.gface = 0.0
        self.bob = 0.0
        self.crouch = 0.0
        self.print_t = time.time() - t0

    # --------------------------------------------------------------- lights
    @staticmethod
    def _flicker(kind, t, ph):
        if kind == "dim":
            return 0.38 + 0.03 * math.sin(t * 2.1 + ph)
        if kind == "flicker":
            x = math.sin(t * 13.0 + ph) * math.sin(t * 5.3 + ph * 2) + 0.5 * math.sin(t * 31 + ph)
            on = 0.9 if (x > -0.55 and math.sin(t * 0.7 + ph) > -0.8) else 0.05
            return on * (0.8 + 0.2 * math.sin(t * 50 + ph))
        if kind == "dead":
            return 0.55 if math.sin(t * 0.37 + ph) > 0.992 else 0.0
        if kind == "red":
            return 0.32 + 0.10 * math.sin(t * 2.0 + ph)
        return 0.3

    def _update_lights(self, t):
        for L in self.lights:
            L["now"] = self._flicker(L["kind"], t, L["ph"]) * 0.75

    # ------------------------------------------------------------- the frame
    def frame(self, game, dt, yaw_extra=0.0, pitch=0.0, show_ai=False, paused=False, show_fps=False, fps=0.0):
        ctx = self.ctx
        W, H = self.W, self.H
        t = game.t
        # --- camera
        px, pz = S.X0 + game.px * S.CELL, S.Z0 + game.py * S.CELL
        yaw = game.angle + math.pi / 2
        want = 1.0 if (game.mode == "sneak") else 0.0
        self.crouch += (want - self.crouch) * min(1.0, dt * 8)
        eye_y = EYE + (EYE_SNEAK - EYE) * self.crouch
        if game.moving:
            self.bob += dt * {"sprint": 11.0, "walk": 7.5, "sneak": 5.0}.get(game.mode, 7)
        bob = math.sin(self.bob) * (1.6 if game.mode != "sprint" else 3.0) * (0 if not game.moving else 1)
        hb = game.hide_blend
        if game.hidden and game.hide_obj is not None:
            ox, oz = S.to_world(*game.hide_obj["centre"])
            if game.hide_kind == "desk":
                px, pz, eye_y = ox, oz, 52.0 * hb + eye_y * (1 - hb)
            else:                                           # pressed into the cupboard: just behind its doors
                ex, ez = S.to_world(*game.hide_exit)
                dx, dz = ox - ex, oz - ez
                n = math.hypot(dx, dz) or 1.0
                px, pz = ex + dx / n * 28 * hb, ez + dz / n * 28 * hb
                eye_y = 150.0
        eye = np.array([px, eye_y + bob, pz])
        V, f = LG.look(eye, yaw, pitch)
        P = LG.persp(72, W / H, 4.0, 4500.0)
        mvp = P @ V
        # --- flashlight lags behind the view
        if self.lag_yaw is None:
            self.lag_yaw, self.lag_pitch = yaw, pitch
        k = min(1.0, dt * 9.0)
        dy = (yaw - self.lag_yaw + math.pi) % (2 * math.pi) - math.pi
        self.lag_yaw += dy * k
        self.lag_pitch += (pitch - self.lag_pitch) * k
        ldir = np.array([math.sin(self.lag_yaw) * math.cos(self.lag_pitch), math.sin(self.lag_pitch), -math.cos(self.lag_yaw) * math.cos(self.lag_pitch)])
        torch = 1.0 if (game.flashlight and not game.hidden) else 0.0
        if torch and game.threat > 0.5 and math.sin(t * 40) * math.sin(t * 7.3) > 0.93:
            torch = 0.25                                   # it flickers when the ghost is close
        # --- doors
        dm = {}
        for dd, meta in zip(game.doors, self.doors_meta):
            hx, hz = dd["hinge"]
            a = dd["angle"] * dd["dir"]
            c, s_ = math.cos(a), math.sin(a)
            R = np.array([[c, 0, s_, 0], [0, 1, 0, 0], [-s_, 0, c, 0], [0, 0, 0, 1]], np.float64)
            T1 = np.eye(4); T1[0, 3], T1[2, 3] = hx, hz
            T0 = np.eye(4); T0[0, 3], T0[2, 3] = -hx, -hz
            dm[dd["node"]] = T1 @ R @ T0
        self._update_lights(t)
        unlocked = game.keys_got >= game.keys_total
        self.exit_sign["color"] = (0.1, 0.9, 0.25, 1) if unlocked else (0.8, 0.1, 0.1, 1)
        self.exit_sign["emis"] = (0.0, 0.8, 0.2) if unlocked else (0.5 + 0.2 * math.sin(t * 3), 0.0, 0.0)
        amb = 0.085
        gain = self.gain * (1.0 + 1.3 * hb)                # eyes adjust to the dark while you hide
        # --- draw
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(0.003, 0.004, 0.007, 1.0)
        self.level.draw_scene(mvp, eye, ldir, torch, amb, gain, self.lights, dm)
        if self.ghost is not None:
            gx, gz = S.X0 + game.gx * S.CELL, S.Z0 + game.gy * S.CELL
            if self.prev_ghost is not None:
                mx, mz = gx - self.prev_ghost[0], gz - self.prev_ghost[1]
                if math.hypot(mx, mz) > 0.4:
                    target = math.atan2(mx, mz)
                    d = (target - self.gface + math.pi) % (2 * math.pi) - math.pi
                    self.gface += d * min(1.0, dt * 9)
            if game.ghost_sees or game.ghost_state == "CHASE":
                target = math.atan2(px - gx, pz - gz)
                d = (target - self.gface + math.pi) % (2 * math.pi) - math.pi
                self.gface += d * min(1.0, dt * 6)
            if game.inspect:
                ox, oz = S.to_world(*game.inspect["obj"]["centre"])
                target = math.atan2(ox - gx, oz - gz)
                d = (target - self.gface + math.pi) % (2 * math.pi) - math.pi
                self.gface += d * min(1.0, dt * 6)
            self.prev_ghost = (gx, gz)
            hop = 0.0
            tilt = 0.0
            if game.ghost_moving and game.ghost_phase > 0.3:
                ph = (game.ghost_phase - 0.3) / 0.7
                hop = 34.0 * math.sin(math.pi * ph)
                tilt = 0.14 * math.sin(math.pi * ph)
            pulse = 0.8 + 0.2 * math.sin(t * 6.0)
            self.ghost.draw(mvp, eye, ldir, torch, amb, gain, self.lights, (gx, 0.0, gz), self.gface, hop, tilt, pulse)
        # --- HUD
        self._hud(game, dt, show_ai, paused, show_fps, fps)
        vig = 1.0 + 0.7 * game.threat + (0.6 if game.hidden else 0.0)
        self.level.composite_hud(self.hud_tex, vig, (float(W), float(H)), t)

    def _hud(self, game, dt, show_ai, paused, show_fps, fps):
        surf = self.hud_surf
        surf.fill((0, 0, 0, 0))
        W, H = self.W, self.H
        hud = self.hud
        if game.hidden:
            self._hide_overlay(surf, game)
        if game.status == "playing":
            hud._hud(surf, game, W, H)
        if show_ai:
            hud._ai_panel(surf, game, W, H)
        if game.status == "caught":
            surf.fill((0, 0, 0, 255))
            hud._caught(surf, game, W, H)
        elif game.status == "escaped":
            hud._escaped(surf, game, W, H)
        if paused:
            shade = pygame.Surface((W, H), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            surf.blit(shade, (0, 0))
            hud.text(surf, "PAUSED", (W // 2, H // 2 - 30), (230, 230, 230), hud.mid, True)
        if show_fps:
            hud.text(surf, "%d fps  %dx%d  %dk tris" % (fps, W, H, self.level.tris // 1000), (W - 260, H - 26), (170, 170, 170), hud.tiny)
        if hud.toast[1] > time.time():
            hud.text(surf, hud.toast[0], (W // 2, H - 150), (200, 200, 200), hud.small, True)
        self.hud_tex.write(_tobytes(surf, "RGBA", False))

    def _hide_overlay(self, surf, game):
        W, H = self.W, self.H
        b = game.hide_blend
        if game.hide_kind == "cupboard":                     # you peer through the slit between the doors
            gap = int(W * 0.035)
            a = int(255 * b)
            pygame.draw.rect(surf, (4, 3, 2, a), (0, 0, W // 2 - gap, H))
            pygame.draw.rect(surf, (4, 3, 2, a), (W // 2 + gap, 0, W // 2 - gap, H))
            pygame.draw.rect(surf, (2, 1, 1, a), (W // 2 - gap - 4, 0, 4, H))
            pygame.draw.rect(surf, (2, 1, 1, a), (W // 2 + gap, 0, 4, H))
        else:                                                # the underside of the desk
            pygame.draw.rect(surf, (6, 4, 3, int(235 * b)), (0, 0, W, int(H * 0.30)))
            pygame.draw.rect(surf, (6, 4, 3, int(150 * b)), (0, int(H * 0.30), W, int(H * 0.06)))


def title_screen(surf, hud, W, H):
    surf.fill((4, 4, 6, 255))
    hud.text(surf, "THE SCHOOL", (W // 2, H // 2 - 170), (190, 175, 150), hud.big, True)
    hud.text(surf, "Something is still teaching here.", (W // 2, H // 2 - 70), (130, 120, 110), hud.mid, True)
    lines = ["Find the 3 keys hidden in the school, then reach the exit at the north end of the main corridor.",
             "The ghost hunts by sound and by what it remembers of you. Hide in cupboards or under desks.",
             "It learns where you like to hide.  Doors creak - and it can hear them.", "",
             "WASD move | mouse look | F flashlight | E open door / hide | Q search | Shift sprint | Ctrl sneak",
             "", "press any key"]
    for i, ln in enumerate(lines):
        hud.text(surf, ln, (W // 2, H // 2 + 10 + 28 * i), (170, 165, 155), hud.small, True)


def main(brain="hmm", windowed_size=(1280, 720), gain=1.0, max_frames=None, fullscreen=False):
    if not os.path.exists(GLB):
        raise SystemExit("Put your horror_school.glb in the same folder as play_gl.py (%s) and run again." % HERE)
    pygame.init()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else 0)
    screen = None
    for depth, msaa in ((24, 4), (24, 0), (16, 0), (0, 0)):          # use the best the graphics driver offers
        try:
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, depth)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1 if msaa else 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, msaa)
            screen = pygame.display.set_mode(windowed_size, flags)
            break
        except pygame.error:
            continue
    if screen is None:
        raise SystemExit("Could not open an OpenGL window. Update your graphics driver.")
    pygame.display.set_caption("THE SCHOOL")
    ctx = moderngl.create_context()
    W, H = screen.get_size()
    print("OpenGL:", ctx.info.get("GL_RENDERER"))
    r = Renderer(ctx, (W, H), gain)
    warmup()
    audio = Audio()
    game = Game(brain)
    title = pygame.Surface((W, H), pygame.SRCALPHA)
    title_screen(title, r.hud, W, H)
    r.hud_tex.write(_tobytes(title, "RGBA", False))
    ctx.clear(0, 0, 0, 1)
    r.level.composite_hud(r.hud_tex, 0.0, (float(W), float(H)), 0.0)
    pygame.display.flip()
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
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()
    clock = pygame.time.Clock()
    pitch, base_yaw, frames = 0.0, None, 0
    while max_frames is None or frames < max_frames:
        dt = clock.tick(120) / 1000.0
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
            elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                r.gain = max(0.5, min(3.0, r.gain + (0.15 if k == pygame.K_RIGHTBRACKET else -0.15)))
                r.hud.toast = ("brightness %d%%" % (r.gain * 100), time.time() + 1.2)
            elif k == pygame.K_n or k in BRAINS:
                brain = BRAINS.get(k, game.brain)
                game = Game(brain)
                r.hud.caught_at = None
            elif k == pygame.K_r:
                game = Game(game.brain, memory=game.memory, round_no=game.round) if game.status == "playing" else game.next_round()
                r.hud.caught_at = None
        keys = pygame.key.get_pressed()
        if not paused:
            forward = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
            strafe = keys[pygame.K_d] - keys[pygame.K_a]
            mx, my = pygame.mouse.get_rel()
            turn = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * TURN_SPEED * dt + mx * MOUSE_SENS
            pitch = max(-1.2, min(1.2, pitch - my * MOUSE_SENS))
            if game.hidden:                                   # you can only glance around a little while hiding
                if base_yaw is None:
                    base_yaw = game.angle
                d = (game.angle + turn - base_yaw + math.pi) % (2 * math.pi) - math.pi
                turn = max(-0.6, min(0.6, d)) - ((game.angle - base_yaw + math.pi) % (2 * math.pi) - math.pi)
                pitch = max(-0.4, min(0.4, pitch))
            else:
                base_yaw = None
            game.update(dt, forward, strafe, turn, sprint=keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
                        crouch=keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_c])
            for e in game.events:
                audio.handle(e, game)
            game.events.clear()
            audio.update(game, dt)
        r.frame(game, dt if not paused else 0.0, 0.0, pitch, show_ai, paused, show_fps, clock.get_fps())
        pygame.display.flip()
        frames += 1
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["hmm", "patrol", "omniscient", "hmm_nomem"], default="hmm")
    ap.add_argument("--window", default="1280x720")
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--fullscreen", action="store_true")
    a = ap.parse_args()
    ws = tuple(int(v) for v in a.window.lower().split("x"))
    main(a.brain, ws, a.brightness, fullscreen=a.fullscreen)
