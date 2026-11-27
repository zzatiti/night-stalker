"""Sound for the school: everything is synthesised with numpy (no audio files needed).

The game engine just appends event names to game.events ("gstep", "key", "hide_desk", ...);
Audio.handle() turns each one into a sound, panned left/right by where it happens and
quieter with distance. If no audio device exists the game still runs, silently.

Design rule from the brief: the ghost's sounds tell you it is NEAR (louder, heartbeat, drone)
but never give away exactly where it is (its direction is blurred with random jitter).
"""
import math
import random

import numpy as np
import pygame

from world import line_of_sight

SR = 22050
_rng = np.random.default_rng(7)


# ------------------------------------------------------------------ synthesis
def _t(sec):
    return np.arange(int(SR * sec)) / SR


def _noise(sec):
    return _rng.uniform(-1, 1, int(SR * sec)).astype(np.float32)


def _lowpass(x, cutoff):
    k = max(2, int(SR / cutoff))
    kernel = np.exp(-np.arange(k * 4) / k).astype(np.float32)
    return np.convolve(x, kernel / kernel.sum(), mode="same")


def _bandpass(x, lo, hi):
    return _lowpass(x, hi) - _lowpass(x, lo)


def _env(n, attack=0.005, release=0.1):
    e = np.ones(n, np.float32)
    a, r = max(1, int(SR * attack)), max(1, int(SR * release))
    e[:a] *= np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, r)
    return e


def _snd(mono, gain=0.8):
    m = np.clip(mono * gain, -1, 1)
    st = np.repeat((m * 32000).astype(np.int16)[:, None], 2, axis=1)
    return pygame.mixer.Sound(buffer=np.ascontiguousarray(st).tobytes())


def _stereo_snd(left, right, gain=0.8):
    st = np.stack([np.clip(left * gain, -1, 1), np.clip(right * gain, -1, 1)], 1)
    return pygame.mixer.Sound(buffer=np.ascontiguousarray((st * 32000).astype(np.int16)).tobytes())


def make_all():
    S = {}
    t = _t(0.16)
    for i, tone in enumerate((95, 80)):                          # two footstep variants
        thud = np.sin(2 * np.pi * tone * t) * np.exp(-t * 38)
        scuff = _lowpass(_noise(0.16), 900) * np.exp(-t * 30) * 2.2
        S["step%d" % i] = _snd(thud + scuff, 0.9)
    t = _t(0.6)                                                    # ghost landing: heavy wet cloth thump
    thump = np.sin(2 * np.pi * 52 * t) * np.exp(-t * 9) + 0.5 * np.sin(2 * np.pi * 104 * t) * np.exp(-t * 16)
    slap = _lowpass(_noise(0.6), 500) * np.exp(-t * 14) * 2.5
    drag = _bandpass(_noise(0.6), 200, 1400) * np.exp(-((t - 0.18) / 0.12) ** 2) * 1.2
    S["ghost_land"] = _snd((thump + slap + drag) * _env(len(t), 0.002, 0.2), 0.9)
    t = _t(0.05)
    S["click"] = _snd(np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 160) + _noise(0.05) * np.exp(-t * 120) * 0.4, 0.8)
    t = _t(0.9)                                                    # door creak: swept band-pass noise + wobble
    sweep = np.sin(2 * np.pi * (170 + 130 * t + 25 * np.sin(2 * np.pi * 9 * t)) * t)
    S["creak"] = _snd((sweep * 0.4 + _bandpass(_noise(0.9), 150, 900) * 1.6) * _env(len(t), 0.08, 0.3) *
                      (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t) ** 2), 0.8)
    t = _t(0.25)
    S["close"] = _snd((np.sin(2 * np.pi * 70 * t) * np.exp(-t * 22) + _lowpass(_noise(0.25), 700) * np.exp(-t * 25)) * 1.2, 0.9)
    t = _t(0.6)                                                    # desk scrape
    S["scrape"] = _snd(_bandpass(_noise(0.6), 120, 1100) * _env(len(t), 0.05, 0.25) * (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t) ** 2) * 3, 0.7)
    t = _t(0.45)                                                   # rummaging / rustling paper
    pops = (np.sin(2 * np.pi * 3 * t * 7) > 0.2).astype(np.float32)
    S["rustle"] = _snd(_bandpass(_noise(0.45), 1500, 6000) * _env(len(t), 0.01, 0.15) * (0.4 + 0.6 * _lowpass(pops, 60)) * 2.4, 0.6)
    S["drawer"] = _snd(_bandpass(_noise(0.5), 100, 800) * _env(len(_t(0.5)), 0.04, 0.2) * 3, 0.7)
    t = _t(1.2)                                                    # key pickup: bright metallic chime
    ring = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) * a
               for f, d, a in ((2093, 5, 0.5), (3136, 7, 0.35), (4186, 9, 0.25), (1568, 4, 0.4)))
    S["key"] = _snd(ring * _env(len(t), 0.001, 0.4), 0.7)
    t = _t(0.3)
    S["nothing"] = _snd((np.sin(2 * np.pi * 120 * t) * np.exp(-t * 20) + _lowpass(_noise(0.3), 800) * np.exp(-t * 18)) * 0.8, 0.6)
    t = _t(0.7)                                                    # door handle rattled
    rat = np.zeros(len(t), np.float32)
    for k in range(7):
        i = int(SR * (0.02 + 0.08 * k + 0.01 * (k % 2)))
        seg = np.exp(-_t(0.05) * 90) * np.sin(2 * np.pi * (700 + 60 * k) * _t(0.05))
        rat[i:i + len(seg)] += seg
    S["rattle"] = _snd(rat + _lowpass(_noise(0.7), 300) * 0.3 * _env(len(t), 0.1, 0.4), 0.8)
    t = _t(1.8)                                                    # ghost whisper / moan
    form = _bandpass(_noise(1.8), 700, 1300) + 0.7 * _bandpass(_noise(1.8), 1800, 2800)
    S["whisper"] = _snd(form * (0.5 + 0.5 * np.sin(2 * np.pi * 4.3 * t) ** 2) * _env(len(t), 0.5, 0.7) * 1.8 +
                        np.sin(2 * np.pi * (110 - 25 * t) * t) * 0.15 * _env(len(t), 0.4, 0.8), 0.7)
    t = _t(0.8)                                                    # knocks
    kn = np.zeros(len(t), np.float32)
    for off in (0.0, 0.22, 0.5):
        i = int(SR * off)
        seg = np.sin(2 * np.pi * 150 * _t(0.12)) * np.exp(-_t(0.12) * 45) + _lowpass(_noise(0.12), 600) * np.exp(-_t(0.12) * 40)
        kn[i:i + len(seg)] += seg
    S["knock"] = _snd(kn, 0.9)
    t = _t(1.3)                                                    # distant slam
    S["bang"] = _snd((np.sin(2 * np.pi * 45 * t) * np.exp(-t * 6) + _lowpass(_noise(1.3), 400) * np.exp(-t * 5) * 2) * _env(len(t), 0.002, 0.3), 0.9)
    t = _t(0.18)
    S["drip"] = _snd(np.sin(2 * np.pi * (1400 - 3500 * t) * t) * np.exp(-t * 28), 0.5)
    t = _t(1.1)                                                    # jump-scare shriek (only when caught)
    shr = np.sin(2 * np.pi * (650 + 900 * t + 120 * np.sin(2 * np.pi * 14 * t)) * t) * 0.5 + _noise(1.1) * 0.7
    S["scare"] = _snd(np.tanh(shr * 2.2) * _env(len(t), 0.01, 0.5), 0.9)
    t = _t(1.6)
    S["win"] = _snd(sum(np.sin(2 * np.pi * f * t) * a for f, a in ((392, 1.0), (494, 0.8), (587, 0.7))) * _env(len(t), 0.2, 0.9) * 0.3, 0.6)
    # loops
    t = _t(0.85)
    beat = sum(np.sin(2 * np.pi * 52 * np.clip(t - s, 0, None)) * np.exp(-np.clip(t - s, 0, None) * 18) * (t >= s) * a
               for s, a in ((0.0, 1.0), (0.26, 0.7)))
    S["heart"] = _snd(beat, 0.9)
    t = _t(4.0)
    S["drone"] = _snd((np.sin(2 * np.pi * 48 * t) + np.sin(2 * np.pi * 49.5 * t) + 0.4 * np.sin(2 * np.pi * 96 * t)) * 0.35, 0.8)
    n = _lowpass(_noise(7.0), 260) * 5.0                           # hallway air: low rumble + slow gusts, seam crossfaded
    n = n * (0.65 + 0.35 * np.sin(2 * np.pi * 0.31 * _t(7.0)) ** 2)
    fade = int(SR * 1.0)
    loop = n[:-fade].copy()
    loop[:fade] = loop[:fade] * np.linspace(0, 1, fade) + n[-fade:] * np.linspace(1, 0, fade)
    S["air"] = _snd(loop, 0.5)
    return S


# --------------------------------------------------------------------- player
class Audio:
    def __init__(self):
        self.ok = False
        self.S = {}
        self._step = 0
        self._next_ambient = 0.0
        try:
            pygame.mixer.init(SR, -16, 2, 512)
            pygame.mixer.set_num_channels(24)
            self.S = make_all()
            self.loops = {}
            for name in ("air", "heart", "drone"):
                ch = self.S[name].play(loops=-1)
                ch.set_volume(0.0, 0.0)
                self.loops[name] = ch
            self.ok = True
        except Exception as exc:               # no sound card, headless box, ...
            print("(audio disabled: %s)" % exc)

    # ---- one-shots
    def play(self, name, vol=1.0, pan=0.0):
        if not self.ok or vol <= 0.01:
            return
        ch = self.S[name].play()
        if ch is not None:
            a = (pan + 1) * math.pi / 4
            ch.set_volume(min(1.0, vol) * math.cos(a), min(1.0, vol) * math.sin(a))

    def _pan_to(self, game, x, y, jitter=0.0):
        dx, dy = x - game.px, y - game.py
        rel = math.atan2(dy, dx) - game.angle
        return max(-1.0, min(1.0, math.sin(rel) + random.uniform(-jitter, jitter)))

    def _ghost_vol(self, game, reach=18.0):
        d = game.ghost_dist()
        v = max(0.0, 1.0 - d / reach) ** 1.4
        if not game.ghost_sees:                       # walls muffle it
            if not line_of_sight(game.tall, game.pcell, game.gcell):
                v *= 0.55
        if game.hidden:
            v *= 0.8
        return v

    def handle(self, ev, game):
        if not self.ok:
            return
        if ev.startswith("pstep_"):
            kind = ev[6:]
            self._step ^= 1
            self.play("step%d" % self._step, {"walk": 0.30, "sprint": 0.55, "sneak": 0.10}[kind],
                      pan=(-0.15 if self._step else 0.15))
        elif ev == "gstep":
            self.play("ghost_land", 1.15 * self._ghost_vol(game), self._pan_to(game, game.gx, game.gy, 0.45))
        elif ev == "whisper":
            self.play("whisper", 0.25 + 0.6 * game.threat, random.uniform(-0.9, 0.9))      # direction deliberately vague
        elif ev == "ambient":
            name = random.choice(("knock", "bang", "drip", "creak", "drip"))
            self.play(name, random.uniform(0.18, 0.4), random.uniform(-1, 1))
        elif ev == "search":
            self.play("drawer", 0.5)
        elif ev == "rummage":
            self.play("rustle", 0.5, random.uniform(-0.2, 0.2))
        elif ev == "key":
            self.play("key", 0.8)
        elif ev == "nothing":
            self.play("nothing", 0.5)
        elif ev == "hide_cupboard":
            self.play("creak", 0.8)
            self.play("close", 0.7)
        elif ev == "hide_desk":
            self.play("scrape", 0.7)
        elif ev == "unhide":
            self.play("creak" if game.hide_kind == "cupboard" else "scrape", 0.6)
        elif ev == "inspect":
            o = game.inspect["obj"] if game.inspect else None
            pan = self._pan_to(game, game.gx, game.gy, 0.3)
            self.play("rattle" if o and o["hide_type"] == "cupboard" else "scrape", 0.9 * max(self._ghost_vol(game, 14), 0.25), pan)
        elif ev == "scare":
            self.play("scare", 1.0)
        elif ev == "win":
            self.play("win", 0.8)

    def update(self, game, dt):
        """Continuous layers: hallway air, heartbeat and a low drone that grow as the ghost nears."""
        if not self.ok:
            return
        playing = game.status == "playing"
        th = game.threat if playing else 0.0
        air = 0.20 * (0.55 if game.hidden else 1.0)
        self.loops["air"].set_volume(air, air)
        hv = min(1.0, th ** 1.3 * 1.2) * 0.9
        self.loops["heart"].set_volume(hv, hv)
        dv = min(1.0, th ** 1.2) * 0.45
        self.loops["drone"].set_volume(dv, dv)

    def click(self):
        self.play("click", 0.6)
