from __future__ import annotations

import numpy as np


def compute_action_mask(hits_grid: np.ndarray, miss_grid: np.ndarray) -> np.ndarray:
    """Return flattened boolean mask of valid actions."""
    mask = np.logical_not(np.logical_or(hits_grid, miss_grid))
    return mask.reshape(-1).astype(bool)
