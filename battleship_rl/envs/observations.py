from __future__ import annotations

import numpy as np


def build_observation(hits_grid: np.ndarray, miss_grid: np.ndarray) -> np.ndarray:
    """Build channel-first observation: 0=Hit, 1=Miss, 2=Unknown."""
    hits = hits_grid.astype(np.float32)
    misses = miss_grid.astype(np.float32)
    unknown = 1.0 - (hits + misses)
    unknown = np.clip(unknown, 0.0, 1.0)
    obs = np.stack([hits, misses, unknown], axis=0)
    return obs.astype(np.float32)
