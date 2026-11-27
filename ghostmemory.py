"""The ghost's memory of how the player likes to hide (persists across rounds).

What it stores
    - recency-weighted counts of hides per TYPE ("cupboard" / "desk") and per SPOT
    - raw totals: times hidden, times found there, times escaped from there
    - the sequence of recent hides (to spot streaks and alternation)

What it produces
    type_prior()  -> {"cupboard": p, "desk": 1 - p}   how likely the ghost thinks each type is
    spot_bonus(i) -> multiplier >= 1 for a specific spot the player keeps reusing

The prior is a smoothed frequency estimate (Laplace / Dirichlet smoothing), so it is
never 0 or 1: the ghost keeps checking other places and is never omniscient.
Every finished round multiplies all counts by `decay`, so old habits fade gradually
and a new habit takes over after a few rounds.
"""

TYPES = ("cupboard", "desk")


class HidingMemory:
    def __init__(self, decay=0.75, alpha=0.7):
        self.decay = decay          # per-round forgetting (1.0 = never forget)
        self.alpha = alpha          # smoothing: higher = slower to trust the evidence
        self.type_w = {t: 0.0 for t in TYPES}
        self.spot_w = {}
        self.hides = {t: 0 for t in TYPES}
        self.found = {t: 0 for t in TYPES}
        self.escaped = {t: 0 for t in TYPES}
        self.history = []           # [(round, kind, spot_id)]
        self.rounds_finished = 0

    # ----------------------------------------------------------- learning
    def record_hide(self, kind, spot_id, round_no=1):
        self.type_w[kind] += 1.0
        self.spot_w[spot_id] = self.spot_w.get(spot_id, 0.0) + 1.0
        self.hides[kind] += 1
        self.history.append((round_no, kind, spot_id))

    def record_outcome(self, kind, spot_id, found):
        """The ghost found the player in this spot (found=True) or the player got away."""
        if found:
            self.found[kind] += 1
            # a spot that worked is reinforced a little: the ghost "remembers" it paid off
            self.type_w[kind] += 0.5
            self.spot_w[spot_id] = self.spot_w.get(spot_id, 0.0) + 1.0
        else:
            self.escaped[kind] += 1

    def end_round(self):
        """Forget a little. Called once when a round finishes."""
        for k in self.type_w:
            self.type_w[k] *= self.decay
        for k in list(self.spot_w):
            self.spot_w[k] *= self.decay
            if self.spot_w[k] < 0.05:
                del self.spot_w[k]
        self.rounds_finished += 1

    # ---------------------------------------------------------- predicting
    def type_prior(self):
        total = sum(self.type_w.values()) + self.alpha * len(TYPES)
        p = {t: (self.type_w[t] + self.alpha) / total for t in TYPES}
        last = [h[1] for h in self.history[-4:]]
        if len(last) >= 3 and len(set(last[-3:])) == 1:
            p[last[-1]] *= 1.35                    # streak: the player is repeating themselves
        if len(last) >= 4 and all(last[i] != last[i + 1] for i in range(3)):
            p = {t: 0.5 * p[t] + 0.5 / len(TYPES) for t in TYPES}  # alternating: don't over-commit
        z = sum(p.values())
        return {t: v / z for t, v in p.items()}

    def spot_bonus(self, spot_id):
        """>= 1. Large only if this exact spot makes up most of the player's recent hides."""
        total = sum(self.spot_w.values())
        if total <= 0:
            return 1.0
        return 1.0 + 2.0 * self.spot_w.get(spot_id, 0.0) / (total + 1.0)

    # -------------------------------------------------------------- display
    def summary(self):
        prior = self.type_prior()
        return {
            "prior": prior,
            "hides": dict(self.hides),
            "found": dict(self.found),
            "escaped": dict(self.escaped),
            "rounds": self.rounds_finished,
            "recent": [h[1] for h in self.history[-5:]],
        }
