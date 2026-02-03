from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        if info is not None and "action_mask" in info:
            mask = np.array(info["action_mask"], dtype=bool)
        else:
            hits = obs[0] > 0.5
            misses = obs[1] > 0.5
            mask = np.logical_not(np.logical_or(hits, misses)).reshape(-1)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        return int(self.rng.choice(valid))
