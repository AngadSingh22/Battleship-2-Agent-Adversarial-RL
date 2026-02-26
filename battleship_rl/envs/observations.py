from __future__ import annotations

import numpy as np


def build_observation(
    hits_grid: np.ndarray,
    miss_grid: np.ndarray,
) -> np.ndarray:
    """Build channel-first observation strictly defined by the LaTeX mathematical specification.
    
    Channel 0: Hit (0/1)
    Channel 1: Miss (0/1)
    Channel 2: Unknown (0/1)
    
    Invariants:
    - Hit and Miss are mutually disjoint.
    - Unknown = 1 - (Hit + Miss)
    - Shape is (3, H, W)
    - Dtype is float32
    """
    # Enforce boolean semantics and cast to float32
    hits = hits_grid.astype(bool).astype(np.float32)
    misses = miss_grid.astype(bool).astype(np.float32)
    
    # Enforce exactly: Unknown = 1.0 - (Hit + Miss)
    # This also acts as an implicit check: if Hit and Miss overlap, Unknown goes negative 
    # (though the environment logic prevents this intrinsically).
    unknown = 1.0 - (hits + misses)
    
    # Exact 3-channel representation
    obs = np.stack([
        hits,
        misses,
        unknown
    ], axis=0)
    
    return obs.astype(np.float32)
