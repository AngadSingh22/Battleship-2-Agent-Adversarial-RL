import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_mask_shape_dtype():
    """Verify mask length exactly H*W and dtype=bool under all conditions."""
    env = BattleshipEnv(board_size=(10, 10))
    obs, info = env.reset()
    
    mask = info["action_mask"]
    assert mask.shape == (100,), "Mask shape must exactly be 1D of length H*W"
    assert mask.dtype == bool, "Mask dtype MUST be strictly boolean"
    
    # Fire at cell 0
    obs, reward, terminated, truncated, info = env.step(0)
    
    mask2 = info["action_mask"]
    assert mask2.shape == (100,)
    assert mask2.dtype == bool
    assert mask2[0] == False, "Cell 0 must be invalid after firing"
