"""HMM belief tracker for the stalker enemy.

The stalker never reads the player's true position. It keeps a probability
distribution (the "belief") over grid cells and updates it every tick with the
forward algorithm:

    belief_t(cell)  ∝  P(obs_t | cell) * sum_prev P(cell | prev) * belief_{t-1}(prev)
                       (observation update)   (prediction step)

Hidden state : the player's cell (row, col)
Transition   : stay with prob p_stay, otherwise move to a uniformly chosen free
               4-neighbour
Observations : (a) sighting       -> the exact cell
               (b) no sighting    -> every cell in the stalker's view is (almost)
                                     ruled out ("negative information")
               (c) heard noise    -> a noisy reported location; likelihood falls
                                     off as a Gaussian in grid distance
"""
import numpy as np

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class BeliefTracker:
    def __init__(self, walls, p_stay=0.4):
        self.walls = np.asarray(walls, dtype=bool)
        self.h, self.w = self.walls.shape
        self.n = self.h * self.w
        self.free = ~self.walls.reshape(-1)
        self.rows, self.cols = np.divmod(np.arange(self.n), self.w)
        self.T = self._build_transition(p_stay)  # T[to, from]
        self.belief = np.zeros(self.n)
        self.reset()

    # ------------------------------------------------------------------ model
    def _build_transition(self, p_stay):
        T = np.zeros((self.n, self.n))
        for r in range(self.h):
            for c in range(self.w):
                if self.walls[r, c]:
                    continue
                i = r * self.w + c
                nbrs = [
                    (r + dr, c + dc)
                    for dr, dc in DIRS
                    if 0 <= r + dr < self.h
                    and 0 <= c + dc < self.w
                    and not self.walls[r + dr, c + dc]
                ]
                if not nbrs:
                    T[i, i] = 1.0
                    continue
                T[i, i] += p_stay
                for rr, cc in nbrs:
                    T[rr * self.w + cc, i] += (1.0 - p_stay) / len(nbrs)
        return T

    def reset(self):
        """Total uncertainty: uniform over every free cell."""
        b = self.free.astype(float)
        self.belief = b / b.sum()

    def _normalize(self):
        self.belief[~self.free] = 0.0
        s = self.belief.sum()
        if s < 1e-12:  # observation contradicted everything -> start over
            self.reset()
        else:
            self.belief /= s

    # ------------------------------------------------------------ filter steps
    def predict(self):
        """Prediction step: the player may have moved since last tick."""
        self.belief = self.T @ self.belief

    def observe_seen(self, cell):
        """Stalker has line of sight to the player: belief collapses."""
        self.belief[:] = 0.0
        self.belief[cell[0] * self.w + cell[1]] = 1.0

    def observe_not_seen(self, visible_mask, detect_prob=0.95):
        """Stalker looked at these cells and saw nothing."""
        self.belief[visible_mask] *= 1.0 - detect_prob
        self._normalize()

    def observe_noise(self, reported_cell, sigma=2.0):
        """Stalker heard a noise that seemed to come from reported_cell."""
        r, c = reported_cell
        d2 = (self.rows - r) ** 2 + (self.cols - c) ** 2
        lik = np.exp(-d2 / (2.0 * sigma ** 2)) + 1e-4  # floor avoids collapse
        self.belief *= lik
        self._normalize()

    # ----------------------------------------------------------------- queries
    def most_likely_cell(self, near=None, tol=0.85):
        """Cell with the highest belief. Near-ties go to the cell closest to
        `near`, which makes the stalker sweep nearby unseen rooms first."""
        idx = np.flatnonzero(self.belief >= tol * self.belief.max())
        if near is None:
            i = idx[0]
        else:
            d = np.abs(self.rows[idx] - near[0]) + np.abs(self.cols[idx] - near[1])
            i = idx[np.argmin(d)]
        return int(self.rows[i]), int(self.cols[i])

    def prob_at(self, cell):
        return float(self.belief[cell[0] * self.w + cell[1]])

    def entropy(self):
        b = self.belief[self.belief > 0]
        return float(-(b * np.log(b)).sum())

    def grid(self):
        return self.belief.reshape(self.h, self.w)
