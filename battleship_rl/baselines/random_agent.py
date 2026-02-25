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
            # obs is (4, H, W): 0=ActiveHit, 1=Miss, 2=Sunk, 3=Unknown.
            # Exclude all cells that have been fired at: active hits, misses,
            # and cells of already-sunk ships (obs[2]).
            fired = (obs[0] > 0.5) | (obs[1] > 0.5) | (obs[2] > 0.5)
            mask = np.logical_not(fired).reshape(-1)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        return int(self.rng.choice(valid))
