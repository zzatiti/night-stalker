# THE SCHOOL - 3D edition (plays inside your own `horror_school.glb`)

## Run
1. Put your `horror_school.glb` in this folder (next to `play_gl.py`).
2. `py -3.13 -m pip install pygame numpy moderngl`
3. `py -3.13 play_gl.py`   (options: `--window 1600x900`, `--fullscreen`, `--brain hmm|patrol|omniscient|hmm_nomem`, `--brightness 1.3`)

The first start converts the model into a walkable map (a couple of seconds) and saves `level_cache.pkl`.
The older 2.5D version still runs with `py -3.13 play3d.py`.

## Controls
WASD move | mouse look | **E** open/close doors, hide in cupboards / under desks | **Q** search a desk drawer / cupboard / bed |
F flashlight | Shift sprint | Ctrl or C sneak | B AI view | P pause | [ ] brightness | F3 fps | R next round | N new game | Esc quit

## What is new
* The level is your model: real walls, rooms, corridor, classrooms (about 380k triangles).
* 9 doors swing open and shut with a hinge animation (double doors move together). A closed door blocks the player AND the ghost's line of sight; the pocong shoves doors open when it hunts you, and doors creak loudly enough for it to hear.
* Cupboards (18) and desks (67) can be searched (keys are hidden in them) or hidden in/under; the ghost checks them, and its memory learns which kind you prefer.
* The pocong is the real 3D mesh (`pocong.glb`), hopping through the school.

## Files
play_gl.py (game), level_gl.py (renderer), school_gl.py (model -> map), gamegl.py (game rules + ghost AI),
glbscene.py (GLB loader), belief.py / world.py / ghostmemory.py (HMM, A*, memory), audio.py, bake_pocong.py, pocong.glb, pocong_*.png.
