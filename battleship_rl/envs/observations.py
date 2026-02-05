from __future__ import annotations

import numpy as np


def build_observation(
    hits_grid: np.ndarray,
    miss_grid: np.ndarray,
    ship_id_grid: np.ndarray,
    ship_sunk: np.ndarray
) -> np.ndarray:
    """Build channel-first observation: 0=ActiveHit, 1=Miss, 2=Sunk, 3=Unknown."""
    hits = hits_grid.astype(bool)
    misses = miss_grid.astype(bool)
    
    # Compute sunk mask: cells where ship_id >= 0 and that ship is sunk
    sunk_mask = np.zeros_like(hits, dtype=bool)
    for ship_id in range(len(ship_sunk)):
        if ship_sunk[ship_id]:
            sunk_mask |= (ship_id_grid == ship_id)
    
    # Mutually exclusive channels
    active_hit = hits & ~sunk_mask
    sunk = sunk_mask
    unknown = ~hits & ~misses
    
    obs = np.stack([
        active_hit.astype(np.float32),
        misses.astype(np.float32),
        sunk.astype(np.float32),
        unknown.astype(np.float32)
    ], axis=0)
    
    return obs.astype(np.float32)
