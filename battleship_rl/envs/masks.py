"""
Action mask computation for the Battleship attacker.

compute_action_mask(env) returns a flat bool array of length board_size**2
where True = cell has not yet been fired at (valid action).

This is the canonical entry-point used by sb3-contrib's ActionMasker wrapper.
"""
from __future__ import annotations

import numpy as np


def compute_action_mask(env) -> np.ndarray:
    """Return a boolean mask of valid (un-fired) cells.

    Works with any BattleshipEnv instance.  Delegates to the env's own
    get_action_mask() so the logic lives in one place.
    """
    return env.get_action_mask()
