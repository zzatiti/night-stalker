"""The abandoned school: map, furniture / hiding spots / key containers, light fixtures.

Legend
    #  wall            L  lockers (decor)       W  window (moonlight)     K  chalkboard
    C  cupboard (hide) B  bookshelf (search)    S  storage shelf (search)
    D  student desk (hide under it / search the drawer)
    T  table        (hide under it / search it)
    P  player start    G  ghost start    E  exit door
"""
import numpy as np

from world import visible_mask

FLOOR, WALL, CUPBOARD, EXIT, LOCKER, BOOKSHELF, WINDOW, STORAGE, BOARD = range(9)
CHAR_TILE = {"#": WALL, "C": CUPBOARD, "E": EXIT, "L": LOCKER, "B": BOOKSHELF,
             "W": WINDOW, "S": STORAGE, "K": BOARD}
CHAR_KIND = {"C": "cupboard", "B": "bookshelf", "S": "storage", "D": "desk", "T": "table"}
HIDE_TYPE = {"cupboard": "cupboard", "desk": "desk", "table": "desk"}    # -> ghost memory type
CAN_SEARCH = {"bookshelf", "storage", "desk", "table", "cupboard"}

MAP = [
    "##C#KKKK#C###SSSS###CC##W#KKKK#WC##C#KKKK##C##",
    "#..........#..........#..........#...........#",
    "#....T.....#..........#....T.....#....T......#",
    "W..........#...T...T..#..........#...........W",
    "#.D.D..D.D.#..........#.D.D..D.D.#.D.D..D.D..#",
    "W..........#..........#..........#...........W",
    "#.D.D..D.D.#..T...T.D.#.D.D..D.D.#.D.D..D.D..#",
    "W..........#..........#..........#...........W",
    "#.D.D..D.D.#....T...D.#.D.D..D.D.#.D.D..D.D..#",
    "#..........#..........#..........#...........#",
    "#............................................#",
    "##LL#.#LLLLLLLL#.#LLLLLLLL#.#LLLLLLLL#.#LLLL##",
    "W............................................#",
    "#.P..........................................E",
    "W............................................#",
    "##LL#.#LLLLLLLL#.#LLLLLLLL#.#LLLLLLLL#.#LLLL##",
    "#..........#..........#..........#...........#",
    "B..........#.SSSS.SSS.#..........#...........#",
    "#..BBBBBB..#..........#.D.D..D.D.#.D.D..D.D..W",
    "B..........#..........#..........#...........#",
    "#...T..T...#..........#.D.D..D.D.#.D.D..D.D..W",
    "B..........#.SSSS.....#..........#...........#",
    "#..BBBBBB..#..........#.D.D..D.D.#.D.D..D.D..W",
    "B..........#......SSS.#..........#...........#",
    "#..........#..........#..........#.......G...W",
    "B..........................T..........T......#",
    "#..........#..........#..........#...........#",
    "##C######C###C######C###W#KKKK#WC##C#KKKK##C##",
]

H, W = len(MAP), len(MAP[0])
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

LIGHT_KINDS = {            # colour, strength, radius (cells)
    "flicker": ((255, 235, 190), 1.15, 7.0),
    "dim": ((235, 215, 160), 0.55, 5.5),
    "red": ((255, 40, 25), 0.8, 6.0),
    "moon": ((95, 125, 215), 0.50, 4.5),
    "dead": ((0, 0, 0), 0.0, 0.0),
}


def _free_neighbours(solid, r, c):
    return [(r + dr, c + dc) for dr, dc in DIRS
            if 0 <= r + dr < H and 0 <= c + dc < W and not solid[r + dr, c + dc]]


def _nearest_free(solid, r, c):
    for rad in range(0, 4):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not solid[rr, cc]:
                    return rr, cc
    return r, c


def build():
    grid = np.zeros((H, W), np.int8)             # tall things the rays hit
    solid = np.zeros((H, W), bool)               # things you can't walk through
    cells, start, ghost, door = [], None, None, None
    for r, row in enumerate(MAP):
        for c, ch in enumerate(row):
            if ch in CHAR_TILE:
                grid[r, c] = CHAR_TILE[ch]
                solid[r, c] = True
            elif ch in "DT":
                solid[r, c] = True               # low furniture: blocks walking, not sight
            if ch in CHAR_KIND:
                cells.append((ch, r, c))
            elif ch == "P":
                start = (c + 0.5, r + 0.5)
            elif ch == "G":
                ghost = (c + 0.5, r + 0.5)
            elif ch == "E":
                door = (c + 0.5, r + 0.5)
    tall = grid != FLOOR

    objects = []
    for ch, r, c in cells:
        kind = CHAR_KIND[ch]
        fronts = _free_neighbours(solid, r, c)
        objects.append({
            "id": len(objects), "kind": kind, "cell": (r, c), "centre": (c + 0.5, r + 0.5),
            "fronts": fronts,
            "hide_type": HIDE_TYPE.get(kind),
            "can_search": kind in CAN_SEARCH,
            "has_key": False, "searched": False,
        })

    lights = []

    def add(x, y, kind, phase=0.0):
        cell = _nearest_free(solid, int(y), int(x))
        _, strength, radius = LIGHT_KINDS[kind]
        vis = (visible_mask(tall, cell, int(radius)).reshape(H, W) if radius else np.zeros((H, W), bool))
        lights.append({"x": x, "y": y, "kind": kind, "phase": phase, "vis": vis,
                       "colour": np.array(LIGHT_KINDS[kind][0], np.float32) / 255.0,
                       "strength": strength, "radius": radius})

    hall_kinds = ["dim", "flicker", "dead", "flicker", "dim", "dead", "flicker"]
    for i, k in enumerate(hall_kinds):
        add(4.5 + 6 * i, 13.5, k, phase=1.7 * i)
    add(1.5, 13.5, "red", 0.0)
    add(43.5, 13.5, "red", 2.0)
    room_kinds = [["flicker", "dead", "dim", "flicker"], ["dim", "dead", "flicker", "dim"]]
    for row, y in enumerate((5.5, 21.5)):
        for i, (c0, _) in enumerate([(1, 10), (12, 21), (23, 32), (34, 44)]):
            add(c0 + 4.5, y, room_kinds[row][i], phase=0.9 * (i + 4 * row))
    for r, row in enumerate(MAP):                 # moonlight through the windows
        for c, ch in enumerate(row):
            if ch == "W":
                fr = _free_neighbours(solid, r, c)
                if fr:
                    add(fr[0][1] + 0.5, fr[0][0] + 0.5, "moon")
    return {"grid": grid, "solid": solid, "tall": tall, "objects": objects, "lights": lights,
            "player": start, "ghost": ghost, "door": door}
